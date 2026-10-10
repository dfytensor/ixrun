# -*- coding: utf-8 -*-
"""Locate the PY q38_graph ULS slowdown: prefill vs graph replay vs eager."""
import os
os.environ['UDCQ_CUDA_GEMV'] = '1'
os.environ['Q38_GREEDY_ONLY'] = '1'
import sys, time
sys.path.insert(0, r'E:\IXRUN')
import pandas  # noqa
import torch
from ixrun.q38_graph import Q38GraphEngine

BLOB = sys.argv[1]
MODEL = r'E:\models\Qwen3.8-27B'
eng = Q38GraphEngine.from_blob(BLOB, MODEL, max_ctx=256)
ids = eng.tokenizer("The capital of France is",
                    return_tensors='pt')['input_ids'][0].tolist()
print(f'prompt tokens: {len(ids)}', flush=True)

eng.hard_reset()
t0 = time.perf_counter()
eng.prefill(ids)
torch.cuda.synchronize()
print(f'prefill S={len(ids)}: {(time.perf_counter()-t0)*1e3:.0f} ms', flush=True)

eng.hard_reset()
eng.prefill(ids)
torch.cuda.synchronize()
N = 50
t0 = time.perf_counter()
for i in range(N):
    eng._set_token(123, 40 + i)
    eng.graph.replay()
torch.cuda.synchronize()
print(f'graph replay only: {(time.perf_counter()-t0)/N*1e3:.1f} ms/tok',
      flush=True)

t0 = time.perf_counter()
for i in range(5):
    eng._forward(eng.emb1, eng.cos1, eng.sin1, eng.pos1)
torch.cuda.synchronize()
print(f'eager _forward  : {(time.perf_counter()-t0)/5*1e3:.1f} ms/tok',
      flush=True)
