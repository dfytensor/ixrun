# -*- coding: utf-8 -*-
"""Profile a few blocked-prefill blocks: where does the 170ms go?"""
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
eng.generate("Hello", max_new_tokens=2, graph=True)
eng.ext.s27_reset()
torch.cuda.synchronize()

ids = eng.tok("The history of computing spans several decades. ")['input_ids']
h8 = torch.empty(8, eng.hidden, dtype=torch.float32, device='cuda')
h8.copy_(eng.emb[ids[:8]].float())
d8 = torch.zeros(8, dtype=torch.int32, device='cuda')
d8.copy_(torch.arange(100, 108, dtype=torch.int32))

for _ in range(2):
    eng.ext.step27_prefill(h8, d8, eng.theta, 0)
torch.cuda.synchronize()

from torch.profiler import profile, ProfilerActivity
with profile(activities=[ProfilerActivity.CUDA]) as prof:
    for i in range(3):
        eng.ext.step27_prefill(h8, d8, eng.theta, 0)
    torch.cuda.synchronize()
tab = prof.key_averages().table(sort_by='cuda_time_total', row_limit=18)
print(tab, flush=True)
tot = sum(e.self_device_time_total for e in prof.key_averages())
print(f'TOTAL per block: {tot/3/1000:.1f}ms', flush=True)
