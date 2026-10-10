# -*- coding: utf-8 -*-
"""Per-kernel profile of the C++ 27B ULS engine decode graph (20 replays)."""
import sys
sys.path.insert(0, r'E:\IXRUN')
import pandas  # noqa
import torch
from ixrun.cpp_engine_27b import CppQwen27bEngine

eng = CppQwen27bEngine.from_blob(
    r'F:\models\qwen38_uls_blob.pt', r'E:\models\Qwen3.8-27B',
    ctx=4096, verbose=True)
eng.generate("The capital of France is", max_new_tokens=4, graph=True)
torch.cuda.synchronize()

eng._dpos.fill_(60)
eng._he_buf.copy_(eng.emb[123].float())
for _ in range(3):
    eng._graph.replay()
torch.cuda.synchronize()

from torch.profiler import profile, ProfilerActivity


def prof_at(pos, n=20):
    eng._dpos.fill_(pos)
    eng._he_buf.copy_(eng.emb[123].float())
    for _ in range(3):
        eng._graph.replay()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(n):
            eng._graph.replay()
        torch.cuda.synchronize()
    tot = sum(e.self_device_time_total for e in prof.key_averages())
    rows = {e.key[:40]: e for e in prof.key_averages()}
    import re
    for k, e in rows.items():
        if 'attn_b' in k:
            print(f'pos {pos}: attn_b {e.self_device_time_total/n/1000:.3f}ms/call '
                  f'x{e.count//n}', flush=True)
    print(f'pos {pos}: TOTAL {tot/n/1000:.2f} ms/tok', flush=True)
    return prof


prof_at(60)
prof_at(3800)

