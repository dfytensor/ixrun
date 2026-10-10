# -*- coding: utf-8 -*-
"""Per-kernel profile of the C++ 27B ULS engine decode graph (20 replays)."""
import sys
sys.path.insert(0, r'E:\IXRUN')
import pandas  # noqa
import torch
from ixrun.cpp_engine_27b import CppQwen27bEngine

eng = CppQwen27bEngine.from_blob(
    r'F:\models\qwen38_uls_blob.pt', r'E:\models\Qwen3.8-27B',
    ctx=256, verbose=True)
eng.generate("The capital of France is", max_new_tokens=4, graph=True)
torch.cuda.synchronize()

eng._dpos.fill_(60)
eng._he_buf.copy_(eng.emb[123].float())
for _ in range(3):
    eng._graph.replay()
torch.cuda.synchronize()

from torch.profiler import profile, ProfilerActivity
with profile(activities=[ProfilerActivity.CUDA]) as prof:
    for _ in range(20):
        eng._graph.replay()
    torch.cuda.synchronize()
tab = prof.key_averages().table(sort_by='cuda_time_total', row_limit=25)
print(tab, flush=True)
tot = sum(e.self_device_time_total for e in prof.key_averages())
print(f'TOTAL GPU per 20 replays: {tot/1000:.1f}ms -> {tot/20/1000:.2f}ms/tok',
      flush=True)
