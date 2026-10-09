# -*- coding: utf-8 -*-
"""Profile 27B C++ step27: per-kernel CUDA time for one decode token."""
import sys, time
sys.path.insert(0, r'E:\IXRUN')
from ixrun.cpp_engine_27b import CppQwen27bEngine
import torch

eng = CppQwen27bEngine.from_blob(
    r'E:\IXRUN\experiments\qwen38_udcq\q38_blob.pt',
    r'E:\models\Qwen3.8-27B', ctx=256, verbose=True)
ext = eng.ext
ids = eng.tok("The capital of France is").input_ids
he_list = [eng.emb[t].cuda().float() for t in ids]


def run(prof=False):
    if prof:
        from torch.profiler import profile, ProfilerActivity
        with profile(activities=[ProfilerActivity.CUDA]) as p:
            for i, _ in enumerate(ids):
                ext.step27(he_list[i], i, eng.theta)
            torch.cuda.synchronize()
        return p
    t0 = time.perf_counter()
    for i, _ in enumerate(ids):
        ext.step27(he_list[i], i, eng.theta)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / len(ids) * 1000


run()  # warmup
print(f'wall {run():.1f} ms/step', flush=True)
p = run(prof=True)
print(p.key_averages().table(sort_by='cuda_time_total', row_limit=25))
