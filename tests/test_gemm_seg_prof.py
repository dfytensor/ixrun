# -*- coding: utf-8 -*-
"""Profile the current GEMM prefill segment (T=256) with batch cores."""
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
eng.generate("warm", max_new_tokens=2, graph=True)

ids = eng.tok("The history of computing spans several decades. ")['input_ids']
hT = torch.empty(256, eng.hidden, dtype=torch.float32, device='cuda')
dT = torch.empty(256, dtype=torch.int32, device='cuda')
hT.copy_(eng.emb[(ids * 40)[:256]].float())
dT.copy_(torch.arange(256, dtype=torch.int32))
for _ in range(2):
    eng.ext.step27_prefill_gemm(hT, dT, eng.theta)
torch.cuda.synchronize()

from torch.profiler import profile, ProfilerActivity
with profile(activities=[ProfilerActivity.CUDA]) as prof:
    for i in range(3):
        dT.copy_(torch.arange(256, dtype=torch.int32))
        eng.ext.step27_prefill_gemm(hT, dT, eng.theta)
    torch.cuda.synchronize()
tab = prof.key_averages().table(sort_by='cuda_time_total', row_limit=14)
print(tab, flush=True)
tot = sum(e.self_device_time_total for e in prof.key_averages())
print(f'TOTAL per segment: {tot/3/1000:.1f}ms', flush=True)
