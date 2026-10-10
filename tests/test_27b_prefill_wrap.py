# -*- coding: utf-8 -*-
"""Wrapper-path prefill timing: graph-captured blocked prefill, real path."""
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

chunk = ("The history of computing spans several decades of rapid change. "
         "From mechanical calculators to vacuum tubes, transistors, and "
         "integrated circuits, each generation reshaped what machines could do. ")
text = chunk * 25
ids = eng.tok(text)['input_ids']
n = (len(ids) // 8) * 8
print(f'prompt tokens: {len(ids)}', flush=True)

# first call captures graphs
t0 = time.perf_counter()
out = eng.generate(text, max_new_tokens=4, graph=True)
dt0 = time.perf_counter() - t0
print(f'run1 (incl capture): {dt0:.2f}s', flush=True)
print(f'out: {out[-80:]!r}', flush=True)

# second call: pure prefill+4 decode through the graphs
t0 = time.perf_counter()
out2 = eng.generate(text, max_new_tokens=4, graph=True)
torch.cuda.synchronize()
dt = time.perf_counter() - t0
print(f'run2: {dt:.2f}s total -> prefill ~{n/dt:.0f} tok/s', flush=True)
print(f'out2: {out2[-80:]!r}', flush=True)
