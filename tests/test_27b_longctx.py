# -*- coding: utf-8 -*-
"""Long-context decode characterization: measure ms/step at pos ~0 vs
~1024 vs ~2048 vs ~3072 (ctx=4096 graph). Reveals the attn/KV scaling."""
import sys, time
sys.path.insert(0, r'E:\IXRUN')
import pandas  # noqa
import torch
from ixrun.cpp_engine_27b import CppQwen27bEngine

CTX = 4096
eng = CppQwen27bEngine.from_blob(
    r'F:\models\qwen38_uls_blob.pt', r'E:\models\Qwen3.8-27B',
    ctx=CTX, verbose=True)

# build a ~3000-token prompt from repeated natural text
chunk = ("The history of computing spans several decades of rapid change. "
         "From mechanical calculators to vacuum tubes, transistors, and "
         "integrated circuits, each generation reshaped what machines could do. ")
ids = eng.tok(chunk * 25)['input_ids']
print(f'prompt tokens: {len(ids)}', flush=True)

# init graph via a short warmup generation at pos 0..3 (writes KV slots 0..)
eng.generate("The capital of France is", max_new_tokens=4, graph=True)
eng.ext.s27_reset()
torch.cuda.synchronize()

he = eng._he_buf
dpos = eng._dpos
marks = {}
N = len(ids)

# prefill through the graph (per-token replays), sampling step time
t0 = time.perf_counter()
for pos, t in enumerate(ids):
    he.copy_(eng.emb[t].float())
    dpos.fill_(pos)
    eng._graph.replay()
    if pos in (128, 512, 1024, 2048, N - 1):
        torch.cuda.synchronize()
        marks[pos] = time.perf_counter() - t0
# measure steady decode at ~1024, ~2048, ~near-max with 16-token windows
def measure_at(start, n=16):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for pos in range(start, start + n):
        he.copy_(eng.emb[123].float())
        dpos.fill_(pos)
        eng._graph.replay()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n * 1e3

for pos in (160, 600, 1100, 2100, 2900, 3800):
    ms = measure_at(pos)
    print(f'pos {pos:5d}: {ms:.1f} ms/step = {1e3/ms:.1f} tok/s', flush=True)
print(f'prefill total {N} tokens: {time.perf_counter()-t0:.0f}s'
      f' (last-window incl)', flush=True)
