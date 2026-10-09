# -*- coding: utf-8 -*-
"""Profile the CAPTURED graph replays (true in-graph kernel times)."""
import sys, time
sys.path.insert(0, r'E:\IXRUN')
from ixrun.cpp_engine_27b import CppQwen27bEngine
import torch

eng = CppQwen27bEngine.from_blob(
    r'E:\IXRUN\experiments\qwen38_udcq\q38_blob.pt',
    r'E:\models\Qwen3.8-27B', ctx=256, verbose=True)

ids = eng.tok("The capital of France is").input_ids[0].tolist()
eng._graph_generate(ids, 2)      # captures the graph
eng.ext.s27_reset()
torch.cuda.synchronize()

REPS = 16
t0 = time.perf_counter()
for pos in range(REPS):
    t = ids[pos] if pos < len(ids) else 2614
    eng._he_buf.copy_(eng.emb[t].float())
    eng._dpos.fill_(pos)
    eng._graph.replay()
torch.cuda.synchronize()
print(f'wall { (time.perf_counter()-t0)/REPS*1000:.1f} ms/replay', flush=True)

from torch.profiler import profile, ProfilerActivity
eng.ext.s27_reset()
torch.cuda.synchronize()
with profile(activities=[ProfilerActivity.CUDA]) as p:
    for pos in range(REPS):
        t = ids[pos] if pos < len(ids) else 2614
        eng._he_buf.copy_(eng.emb[t].float())
        eng._dpos.fill_(pos)
        eng._graph.replay()
    torch.cuda.synchronize()
print(p.key_averages().table(sort_by='cuda_time_total', row_limit=25))
