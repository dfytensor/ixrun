# -*- coding: utf-8 -*-
"""UDCQ 4-bit on Qwen3.8-27B (hybrid linear/full-attention, 64 layers).

VRAM plan (24GB card):
  bf16 full model ~55GB — impossible resident. UDCQ packed:
    linear weights ~26B params? No — 27B total incl vision+embed. Linear-
    only packed at 6.0bpw ~= (see print). Strategy = lazy per-layer CPU
    quantize -> GPU-resident packed streams (stream mode), same as the U1.5
    playbook. Generation uses the fused GEMV (single token) + stream decode
    + cublas for prefill (multi-token).
"""
import sys, os, time, gc, json
sys.setrecursionlimit(20000)
sys.path.insert(0, r'E:\IXRUN')
import pandas
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from ixrun.udcq import (udcq_fit_codebook, udcq_quantize, UdcqLinear,
                        UDCQ_G, _get_shared_w_buf)
from ixrun.linear import _set_parent_child
from ixrun.config import QWEN38_PATH
import ixrun.linear as _lin_mod

MD = QWEN38_PATH
OUT = r'E:\IXRUN\experiments\qwen38_udcq'
os.makedirs(OUT, exist_ok=True)

tok = AutoTokenizer.from_pretrained(MD)


def _iter_quantizable_incl_lmhead(m):
    """iter_quantizable_linears minus the size/skip guards, PLUS lm_head
    (kept out of the stock iterator). embed_tokens is NOT a Linear."""
    for name, mod in m.named_modules():
        if isinstance(mod, nn.Linear) and mod.weight.numel() >= 1000:
            if 'embed' in name:
                continue
            yield name, mod

PROMPTS = [
    "The theory of relativity states that",
    "def quick_sort(arr):",
    "北京最值得游览的三个景点是",
]


@torch.no_grad()
def gen(m, prompt, n=40):
    ids = tok(prompt, return_tensors="pt")["input_ids"].cuda()
    out = m(ids, use_cache=True)
    past = out.past_key_values
    nxt = out.logits[:, -1].argmax(-1, keepdim=True)
    toks = [nxt[0, 0].item()]
    torch.cuda.synchronize(); t0 = time.time()
    for _ in range(n - 1):
        out = m(nxt, past_key_values=past, use_cache=True)
        past = out.past_key_values
        nxt = out.logits[:, -1].argmax(-1, keepdim=True)
        toks.append(nxt[0, 0].item())
    torch.cuda.synchronize()
    return tok.decode(toks, skip_special_tokens=True), (n - 1) / (time.time() - t0)


@torch.no_grad()
def deploy_stream_lazy(m, verbose=True):
    """CPU lazy quantize each Linear -> GPU-resident packed (stream mode).
    Host RAM peak ~= one bf16 layer; GPU gets packed streams incrementally."""
    targets = list(_iter_quantizable_incl_lmhead(m))
    cb = None
    tot_b = 0
    tot_n = 0
    t0 = time.time()
    for i, (name, mod) in enumerate(targets):
        w = mod.weight.data
        if w.is_cuda:
            w = w.cpu()
        if cb is None:
            cb = udcq_fit_codebook(w, g=UDCQ_G)
        packed = udcq_quantize(w.to(torch.bfloat16) if w.dtype != torch.bfloat16 else w,
                               cb, g=UDCQ_G)
        bias = mod.bias.data if mod.bias is not None else None
        new = UdcqLinear(packed, bias=bias, cache="stream")
        _set_parent_child(m, name, new)
        mod.weight.data = torch.empty(0)
        tot_b += packed["total_bytes"]
        tot_n += packed["N"]
        del w, packed
        if (i + 1) % 60 == 0:
            gc.collect()
            if verbose:
                print(f'  [{i+1}/{len(targets)}] {tot_b/1e9:.1f}GB packed, '
                      f'{time.time()-t0:.0f}s', flush=True)
    gc.collect(); torch.cuda.empty_cache()
    if verbose:
        print(f'[udcq-27B] {len(targets)} layers | {tot_b/1e9:.2f}GB packed '
              f'({tot_b*8/tot_n:.2f} bpw) in {time.time()-t0:.0f}s', flush=True)
    return tot_b, tot_n


def to_gpu_selective(m, verbose=True):
    """Move ONLY text-backbone non-linear params to GPU. Vision tower,
    embed_tokens and lm_head stay on CPU (never touched in text-only
    generation; lm_head only produces logits for argmax -> its final rows
    are what matters, and HF text models run lm_head on GPU only if
    resident). Saves ~6GB of the 35.7GB resident."""
    moved = kept = 0
    for name, p in m.named_parameters():
        if not p.numel():
            continue
        if p.device.type != 'cpu':
            continue
        if any(s in name for s in ('visual', 'vision')):
            kept += p.numel() * 2
            continue
        p.data = p.data.cuda()
        moved += p.numel() * 2
    torch.cuda.empty_cache()
    if verbose:
        print(f'[gpu] moved {moved/1e9:.2f}GB backbone params; '
              f'kept CPU {kept/1e9:.2f}GB (vision/embed/lm_head)', flush=True)


def main():
    print('loading Qwen3.8-27B (lazy cpu) ...', flush=True)
    t0 = time.time()
    m = AutoModelForCausalLM.from_pretrained(
        MD, dtype=torch.bfloat16, low_cpu_mem_usage=True, device_map='cpu')
    print(f'  meta-loaded {time.time()-t0:.0f}s; quantizing layer-by-layer ...', flush=True)
    deploy_stream_lazy(m)
    m.eval()
    to_gpu_selective(m)
    print(f'  resident GPU = {torch.cuda.memory_allocated()/1e9:.2f}GB', flush=True)

    for p in PROMPTS:
        txt, tps = gen(m, p)
        print(f'\n[{tps:.2f} tok/s] {p!r}\n  -> {txt!r}', flush=True)
    print(f'\npeak GPU = {torch.cuda.max_memory_allocated()/1e9:.2f}GB', flush=True)


if __name__ == '__main__':
    main()
