# -*- coding: utf-8 -*-
"""PeakQEngine: near-lossless PEAK-Q (10.6 bpw, 54dB SNR) deploy with the
same interface as Int8XEngine (tokenizer / generate / stream), so the
CLI chat/generate and the OpenAI-compatible server work unchanged:

    python -m ixrun.cli generate "..." --codec peakq --mode streaming

Modes:
- 'cached'    PeakQLinear(cache='full'): bf16 decode once, resident
- 'streaming' PeakQLinear(cache='none'): fused GEMV per step, packed
              GPU-resident (~half the VRAM)
"""
from __future__ import annotations

import torch

from .generate import generate_text, stream_generate

__all__ = ["PeakQEngine"]


class PeakQEngine:
    def __init__(self, model, tokenizer, stats=None):
        self.model = model
        self.tokenizer = tokenizer
        self.stats = stats or {}
        self.model.eval()

    @classmethod
    def from_pretrained(cls, model_path, mode="streaming", layout="rows",
                        verbose=True):
        from transformers import AutoTokenizer

        from .engine import Int8XEngine
        from .peakq import deploy_peakq

        tokenizer = AutoTokenizer.from_pretrained(model_path,
                                                  trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        streaming = mode == "streaming"
        model = Int8XEngine._load_any(model_path, torch.bfloat16,
                                      low_cpu=streaming)
        if not streaming:
            model = model.cuda()
        stats = deploy_peakq(model, layout=layout,
                             cache="none" if streaming else "full",
                             verbose=verbose)
        eng = cls(model, tokenizer, stats)
        if streaming:
            eng._finalize_device()
        return eng

    def _finalize_device(self):
        for _, mod in self.model.named_modules():
            for p in mod.parameters(recurse=False):
                if p.data.device.type != "cuda":
                    p.data = p.data.cuda()
            for bname, b in mod.named_buffers(recurse=False):
                if b.device.type != "cuda":
                    setattr(mod, bname, b.cuda())
        import gc

        gc.collect()
        torch.cuda.empty_cache()
        self.model.eval()

    def generate(self, prompt, **kw):
        return generate_text(self.model, self.tokenizer, prompt, **kw)

    def stream(self, prompt, **kw):
        yield from stream_generate(self.model, self.tokenizer, prompt, **kw)
