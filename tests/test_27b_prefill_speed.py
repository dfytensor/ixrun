# -*- coding: utf-8 -*-
"""Blocked-prefill throughput: per-block ms and effective prefill tok/s."""
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
eng.generate("Hello", max_new_tokens=2, graph=True)   # capture + warmup
eng.ext.s27_reset()
torch.cuda.synchronize()

chunk = ("The history of computing spans several decades of rapid change. "
         "From mechanical calculators to vacuum tubes, transistors, and "
         "integrated circuits, each generation reshaped what machines could do. ")
ids = eng.tok(chunk * 25)['input_ids']
n = (len(ids) // 8) * 8
print(f'prompt tokens: {len(ids)} (blocks cover {n})', flush=True)

h8 = torch.empty(8, eng.hidden, dtype=torch.float32, device='cuda')
d8 = torch.zeros(8, dtype=torch.int32, device='cuda')
pos8 = torch.zeros(8, dtype=torch.int32)
t0 = time.perf_counter()
for p0 in range(0, n, 8):
    h8.copy_(eng.emb[ids[p0:p0 + 8]].float())
    pos8.copy_(torch.tensor([p0 + i for i in range(8)], dtype=torch.int32))
    d8.copy_(pos8)
    eng.ext.step27_prefill(h8, d8, eng.theta, 0)
torch.cuda.synchronize()
dt = time.perf_counter() - t0
nb = n // 8
print(f'blocked prefill: {dt:.2f}s for {n} tok = {n/dt:.0f} tok/s '
      f'({dt/nb*1000:.0f} ms/block)', flush=True)

# compare: legacy per-token
eng.ext.s27_reset()
torch.cuda.synchronize()
t0 = time.perf_counter()
for pos in range(n):
    eng._he_buf.copy_(eng.emb[ids[pos]].float())
    eng._dpos.fill_(pos)
    eng._graph.replay()
torch.cuda.synchronize()
dt2 = time.perf_counter() - t0
print(f'legacy prefill : {dt2:.2f}s for {n} tok = {n/dt2:.0f} tok/s',
      flush=True)
print(f'prefill speedup: {dt2/dt:.1f}x', flush=True)
