# -*- coding: utf-8 -*-
"""GEMM-prefill gate + timing: segments of 256 via step27_prefill_gemm,
A/B vs the legacy per-token path (text comparison)."""
import os
os.environ['IXRUN_PREBUILT'] = '1'
import sys, time
sys.path.insert(0, r'E:\IXRUN')
import pandas  # noqa
import torch
from ixrun.cpp_engine_27b import CppQwen27bEngine

eng = CppQwen27bEngine.from_blob(
    r'F:\models\qwen38_uls_blob.pt', r'E:\models\Qwen3.8-27B',
    ctx=4096, verbose=True)
eng.generate("warm", max_new_tokens=2, graph=True)   # capture + warmup

P = ("The history of computing spans several decades of rapid change. "
     "From mechanical calculators to vacuum tubes, transistors, and "
     "integrated circuits, each generation reshaped what machines could do. ")
text = P * 12
ids = eng.tok(text)['input_ids']
n = len(ids)
print(f'prompt tokens: {n}', flush=True)

hT = torch.empty(256, eng.hidden, dtype=torch.float32, device='cuda')
dposT = torch.empty(256, dtype=torch.int32, device='cuda')


def run_gemm_prefill():
    eng.ext.s27_reset()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    pos = 0
    while pos < n:
        m = min(256, n - pos)
        hT[:m].copy_(eng.emb[ids[pos:pos + m]].float())
        dposT[:m].copy_(torch.arange(pos, pos + m, dtype=torch.int32))
        eng.ext.step27_prefill_gemm(hT[:m], dposT[:m], eng.theta)
        pos += m
    torch.cuda.synchronize()
    return time.perf_counter() - t0


t_gemm = run_gemm_prefill()
tok = int(eng.ext.s27_get_tok().item())
print(f'gemm prefill: {t_gemm:.2f}s ({n/t_gemm:.0f} tok/s) first tok {tok}',
      flush=True)

# continue decode from pos n using the captured decode graph
def decode_from(pos, first_tok, n_new=16):
    toks = [first_tok]
    for p in range(pos, pos + n_new):
        t = toks[-1]
        eng._he_buf.copy_(eng.emb[t].float())
        eng._dpos.fill_(p)
        eng._graph.replay()
        toks.append(int(eng.ext.s27_get_tok().item()))
    return toks


toks_g = decode_from(n, tok)
txt_g = eng.tok.decode(toks_g[:-1])
print(f'gemm out: {txt_g!r}', flush=True)

# legacy reference: recompute the same prompt via the per-token path
os.environ['IXRUN_NO_PREFILL'] = '1'
eng.ext.s27_reset()
torch.cuda.synchronize()
pos = 0
for pos in range(n):
    eng._he_buf.copy_(eng.emb[ids[pos]].float())
    eng._dpos.fill_(pos)
    eng._graph.replay()
tok_l = int(eng.ext.s27_get_tok().item())
toks_l = decode_from(n, tok_l)
txt_l = eng.tok.decode(toks_l[:-1])
print(f'legacy  : {txt_l!r}', flush=True)
print('TEXT MATCH' if txt_l == txt_g else '*** TEXT DIFFERS ***', flush=True)
os.environ['IXRUN_NO_PREFILL'] = '0'
os.environ['IXRUN_GEMM_PREFILL'] = '1'
t0 = time.perf_counter()
out = eng.generate(text, max_new_tokens=16, graph=True)
torch.cuda.synchronize()
dt = time.perf_counter() - t0
nw = len(eng.tok(out).input_ids)
print(f'wrapper gemm: {dt:.2f}s, {nw} tok -> ~{nw/dt:.0f} tok/s total', flush=True)
print(f'wrapper out: {out[-70:]!r}', flush=True)
