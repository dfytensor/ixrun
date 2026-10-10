# -*- coding: utf-8 -*-
"""PY 27B graph engine pure-decode rate (differential timing)."""
import os
os.environ['UDCQ_CUDA_GEMV'] = '1'
os.environ['Q38_GREEDY_ONLY'] = '1'
import sys, time
sys.path.insert(0, r'E:\IXRUN')
import pandas  # noqa: F401
import torch
from ixrun.q38_graph import Q38GraphEngine

BLOB = sys.argv[1] if len(sys.argv) > 1 \
    else r'E:\IXRUN\experiments\qwen38_udcq\q38_blob.pt'
MODEL = r'E:\models\Qwen3.8-27B'
eng = Q38GraphEngine.from_blob(BLOB, MODEL, max_ctx=256)
prompt = "The capital of France is"


def run(n):
    t0 = time.perf_counter()
    eng.generate(prompt, max_new_tokens=n)
    torch.cuda.synchronize()
    return time.perf_counter() - t0


run(8)
t8 = run(8)
t136 = run(136)
dec = (t136 - t8) / 128
print(f'PY-graph(udcq-cuda-v2) PURE decode: {1/dec:.1f} tok/s '
      f'({dec*1000:.1f} ms/tok)', flush=True)
out = eng.generate(prompt, max_new_tokens=32)
print('text:', repr(out), flush=True)
