# -*- coding: utf-8 -*-
"""Qwen3.8-27B slim-resident on 24GB: UDCQ packed streams fully GPU-resident.

Budget fix vs run5 (34.8GB -> ~22GB):
  - embed_tokens stays on CPU; a tiny module intercepts the embedding lookup
    and DMAs only the needed rows (10KB/token) to GPU
  - mtp / vision modules are never materialized (not needed for text gen)
  - 497 UDCQ-packed linears (19.2GB) GPU-resident -> fused GEMV per token
"""
import sys, os, time, gc, json
sys.setrecursionlimit(20000)
sys.path.insert(0, r'E:\IXRUN')
import pandas
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors import safe_open
from transformers import AutoModelForCausalLM, AutoTokenizer

from ixrun.udcq import (udcq_fit_codebook, udcq_quantize, UdcqLinear,
                        UDCQ_G)
from ixrun.linear import _set_parent_child
from ixrun.config import QWEN38_PATH

MD = QWEN38_PATH
tok = AutoTokenizer.from_pretrained(MD)

PROMPTS = [
    "The theory of relativity states that",
    "def quick_sort(arr):",
    "北京最值得游览的三个景点是",
]


class CpuEmbed(nn.Module):
    """Embedding with weights on CPU; rows DMA'd on demand (10KB/token)."""

    def __init__(self, weight_cpu):
        super().__init__()
        self.register_buffer("weight_cpu", weight_cpu, persistent=False)
        self.out_dim = weight_cpu.shape[1]

    def forward(self, ids):
        rows = self.weight_cpu[ids.reshape(-1).cpu()].cuda(non_blocking=True)
        return rows.view(*ids.shape, self.out_dim)


@torch.no_grad()
def deploy_slim(m, verbose=True):
    """Pack every linear (incl lm_head) -> GPU-resident stream layers;
    replace embed with CpuEmbed; leave mtp/vision as empty (never called)."""
    targets = []
    for name, mod in m.named_modules():
        if isinstance(mod, nn.Linear) and mod.weight.numel() >= 1000 \
                and 'embed' not in name:
            if name.startswith('mtp') or '.mtp.' in name:
                continue                          # speculative module: skip
            targets.append((name, mod))
    cb = None
    tot_b = tot_n = 0
    t0 = time.time()
    for i, (name, mod) in enumerate(targets):
        w = mod.weight.data.cpu()
        if cb is None:
            cb = udcq_fit_codebook(w, g=UDCQ_G)
        p = udcq_quantize(w, cb, g=UDCQ_G)
        bias = mod.bias.data if mod.bias is not None else None
        lay = UdcqLinear(p, bias=bias, cache="stream")
        _set_parent_child(m, name, lay)
        mod.weight.data = torch.empty(0)
        tot_b += p["total_bytes"]
        tot_n += p["N"]
        del w, p
        if (i + 1) % 100 == 0:
            gc.collect()
            print(f'  [{i+1}/{len(targets)}] {time.time()-t0:.0f}s', flush=True)
    gc.collect()

    # embed -> CpuEmbed (rows on demand)
    emb = m.get_input_embeddings()
    emb_w = emb.weight.data.cpu()                 # stays CPU
    emb_new = CpuEmbed(emb_w.to(torch.bfloat16))
    # find parent of embedding
    for name, mod in m.named_modules():
        if mod is emb and name:
            parts = name.split('.')
            parent = m
            for q in parts[:-1]:
                parent = getattr(parent, q)
            setattr(parent, parts[-1], emb_new)
            break
    else:
        m.model.embed_tokens = emb_new

    # remaining non-linear params (norms, A_log, dt_bias...) -> GPU
    moved = 0
    params = dict(m.named_parameters())
    for name, p in params.items():
        if p.numel() == 0 or p.device.type != 'cpu':
            continue
        if 'mtp' in name or 'visual' in name or 'vision' in name:
            continue                              # never called in text gen
        if name.endswith('weight_cpu'):           # CpuEmbed buffer
            continue
        p.data = p.data.cuda()
        moved += p.numel() * p.element_size()
    torch.cuda.empty_cache()
    if verbose:
        print(f'[slim-27B] {len(targets)} packed linears {tot_b/1e9:.2f}GB '
              f'({tot_b*8/tot_n:.2f}bpw) | norms {moved/1e9:.2f}GB | '
              f'{time.time()-t0:.0f}s', flush=True)
    return tot_b


def load_model():
    """meta-build then selectively materialize: linears from the UDCQ blob
    path is done by deploy_slim on a low_cpu_mem load; mtp/vision tensors
    are never read (lazy loading skips them)."""
    m = AutoModelForCausalLM.from_pretrained(
        MD, dtype=torch.bfloat16, low_cpu_mem_usage=True, device_map='cpu')
    return m


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


def main():
    print('loading (lazy cpu) + slim deploy ...', flush=True)
    m = load_model()
    deploy_slim(m)
    m.eval()
    print(f'  resident GPU = {torch.cuda.memory_allocated()/1e9:.2f}GB',
          flush=True)
    for p in PROMPTS:
        txt, tps = gen(m, p)
        print(f'\n[{tps:.2f} tok/s] {p!r}\n  -> {txt!r}', flush=True)
    print(f'\npeak GPU = {torch.cuda.max_memory_allocated()/1e9:.2f}GB', flush=True)


if __name__ == '__main__':
    main()
