# -*- coding: utf-8 -*-
"""StepGraphEngine: whole-decode-step CUDA-Graph engine (Llama-arch).

Benchmark finding (bench_formats_minicpm5): a 1B eager decode step is
~77% Python/launch overhead — full-step CUDA graph replay is 13.2ms vs
57.4ms eager. This engine captures the ENTIRE one-token forward in one
graph (StaticCache-backed) and replays per token.

Formats:
  codec='bf16'        resident bf16 (fastest possible step)
  codec='int8x'       INT8-X cache='full' (bf16 resident after decode)
  codec='udcq'        UDCQ cache='full' (bf16 resident) — default trade
  codec='udcq-stream' UDCQ streaming: packed GPU-resident, fused-GEMV
                      decode kernels INSIDE the captured step (~1GB VRAM)

Interface identical to Int8XEngine: tokenizer / generate / stream.
"""
from __future__ import annotations

import gc
import time

import torch
import torch.nn.functional as F

from .config import MODEL_PATH

__all__ = ["StepGraphEngine"]

DEFAULT_MAX_CTX = 2048


class StepGraphEngine:
    def __init__(self, model, tokenizer, stats=None, max_ctx=DEFAULT_MAX_CTX,
                 verbose=True):
        self.model = model
        self.tokenizer = tokenizer
        self.stats = stats or {}
        self.max_ctx = max_ctx
        model.eval()
        self._cfg = model.config
        self._deploy_static(max_ctx, verbose)

    # ------------------------------------------------------------------ #
    @classmethod
    def from_pretrained(cls, model_path=MODEL_PATH, codec='udcq',
                        max_ctx=DEFAULT_MAX_CTX, verbose=True):
        from transformers import AutoTokenizer

        from .engine import Int8XEngine

        tokenizer = AutoTokenizer.from_pretrained(model_path,
                                                  trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = Int8XEngine._load_any(model_path, torch.bfloat16,
                                      low_cpu=True)
        stats = {'codec': codec}
        if codec == 'int8x':
            from .linear import deploy_model

            deploy_model(model, level_bits=(3, 5, 8), cache='full',
                         verbose=verbose)
        elif codec == 'udcq':
            from .udcq import deploy_udcq

            stats.update(deploy_udcq(model, cache='full', verbose=verbose))
        elif codec == 'udcq-stream':
            from .udcq import deploy_udcq

            stats.update(deploy_udcq(model, cache='stream', verbose=verbose))
        elif codec != 'bf16':
            raise ValueError(f'unknown codec: {codec}')
        model = model.cuda()
        gc.collect()
        torch.cuda.empty_cache()
        eng = cls(model, tokenizer, stats=stats, max_ctx=max_ctx,
                  verbose=verbose)
        return eng

    # ------------------------------------------------------------------ #
    def _deploy_static(self, max_ctx, verbose):
        from transformers.cache_utils import StaticCache

        self.cache = StaticCache(
            config=self._cfg, max_batch_size=1, max_cache_len=max_ctx)
        dev = 'cuda'
        self.in_ids = torch.zeros(1, 1, dtype=torch.long, device=dev)
        self.pos = torch.zeros(1, 1, dtype=torch.long, device=dev)
        self.logits = torch.zeros(
            1, 1, self.model.lm_head.out_features
            if hasattr(self.model, 'lm_head')
            else self._cfg.vocab_size,
            dtype=torch.bfloat16, device=dev)

        # warmup on a side stream then capture the full decode step
        if verbose:
            print('[step-graph] capturing full decode step...', flush=True)
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                self._step()
        torch.cuda.current_stream().wait_stream(s)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            lg = self._step()
            self.logits.copy_(lg)          # static output (pool-safe)
        self.graph = g
        self.hard_reset()
        if verbose:
            print('[step-graph] captured; cache reset', flush=True)

    @torch.no_grad()
    def _step(self):
        return self.model(self.in_ids, position_ids=self.pos,
                          past_key_values=self.cache, use_cache=True,
                          attention_mask=None).logits

    # ------------------------------------------------------------------ #
    def hard_reset(self):
        for layer in self.cache.layers:
            for attr in ('keys', 'values', 'key_cache', 'value_cache'):
                t = getattr(layer, attr, None)
                if isinstance(t, torch.Tensor):
                    t.zero_()
            for attr in ('cumulative_length', 'cache_position'):
                t = getattr(layer, attr, None)
                if isinstance(t, torch.Tensor):
                    t.zero_()

    @torch.no_grad()
    def prefill(self, ids):
        """Eager prefill of the full prompt; returns next-token logits."""
        t = torch.tensor(ids, dtype=torch.long, device='cuda').view(1, -1)
        pos = torch.arange(len(ids), device='cuda').view(1, -1)
        out = self.model(t, position_ids=pos, past_key_values=self.cache,
                         use_cache=True, attention_mask=None)
        return out.logits

    def _first_token(self, ids):
        logits = self.prefill(ids)
        return int(logits[:, -1].argmax(-1).item())

    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def _gen_tokens(self, ids, max_new_tokens, temperature=0.0):
        self.hard_reset()
        nxt = self._first_token(ids)
        out = [nxt]
        eos = self.tokenizer.eos_token_id
        t = len(ids)
        while len(out) < max_new_tokens and t < self.max_ctx - 1:
            self.in_ids.fill_(nxt)
            self.pos.fill_(t)
            self.graph.replay()
            if temperature and temperature > 0:
                probs = F.softmax(self.logits[:, -1].float() / temperature,
                                  dim=-1)
                nxt = int(torch.multinomial(probs, 1).item())
            else:
                nxt = int(self.logits[:, -1].argmax(-1).item())
            if nxt == eos:
                break
            out.append(nxt)
            t += 1
        return out

    def generate(self, prompt, max_new_tokens=64, temperature=0.0,
                 do_sample=False, **kw):
        if not do_sample:
            temperature = 0.0
        ids = self.tokenizer(prompt, return_tensors='pt')['input_ids'][0] \
            .tolist()
        out = self._gen_tokens(ids, max_new_tokens, temperature)
        return self.tokenizer.decode(out)

    def stream(self, prompt, max_new_tokens=64, temperature=0.0,
               do_sample=False, **kw):
        if not do_sample:
            temperature = 0.0
        ids = self.tokenizer(prompt, return_tensors='pt')['input_ids'][0] \
            .tolist()
        self.hard_reset()
        nxt = self._first_token(ids)
        eos = self.tokenizer.eos_token_id
        t = len(ids)
        n = 0
        while n < max_new_tokens and t < self.max_ctx - 1:
            self.in_ids.fill_(nxt)
            self.pos.fill_(t)
            self.graph.replay()
            if temperature and temperature > 0:
                probs = F.softmax(self.logits[:, -1].float() / temperature,
                                  dim=-1)
                nxt = int(torch.multinomial(probs, 1).item())
            else:
                nxt = int(self.logits[:, -1].argmax(-1).item())
            if nxt == eos:
                break
            yield self.tokenizer.decode([nxt])
            n += 1
            t += 1


if __name__ == '__main__':
    # quick self-check: bf16 vs udcq-stream token agreement + speed
    import sys
    sys.path.insert(0, r'E:\IXRUN')
    import pandas
    import time as _t

    for codec in ('bf16', 'udcq', 'udcq-stream'):
        e = StepGraphEngine.from_pretrained(codec=codec)
        ids = e.tokenizer('The theory of relativity states that',
                          return_tensors='pt')['input_ids'][0].tolist()
        t0 = _t.time()
        toks = e._gen_tokens(ids, 40, 0.0)
        torch.cuda.synchronize()
        dt = (_t.time() - t0) / len(toks) * 1e3
        print(f'[{codec}] {1000/dt:.1f} tok/s ({dt:.1f}ms/tok) -> '
              f'{e.tokenizer.decode(toks)[:80]!r} | gpu '
              f'{torch.cuda.memory_allocated()/1e9:.2f}GB', flush=True)
        del e
        gc.collect()
        torch.cuda.empty_cache()
