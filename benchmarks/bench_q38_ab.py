# -*- coding: utf-8 -*-
"""In-model A/B: Triton mt-GEMV vs hand-written CUDA GEMV on the 27B blob.
Run under vcvars2022 env (Windows CUDA build):
    UDCQ_CUDA_GEMV=1 python -X utf8 bench_q38_ab.py   (cuda path)
    python -X utf8 bench_q38_ab.py                    (triton baseline)
"""
import sys, time, gc, os
sys.path.insert(0, r'E:\IXRUN')
import pandas                       # before torch (heap rule)
import torch
from transformers import AutoTokenizer

BLOB = r'E:\IXRUN\experiments\qwen38_udcq\q38_blob.pt'
MODEL = r'E:\models\Qwen3.8-27B'
N = 50
CUDA = os.environ.get('UDCQ_CUDA_GEMV') == '1'

if CUDA:
    print('=== hand-written CUDA GEMV (UDCQ_CUDA_GEMV=1) ===', flush=True)
else:
    print('=== Triton mt-GEMV baseline ===', flush=True)

from ixrun.q38_graph import Q38GraphEngine
eng = Q38GraphEngine.from_blob(BLOB, MODEL, verbose=False)
tok = AutoTokenizer.from_pretrained(MODEL)
ids = tok('The theory of relativity states that',
          return_tensors='pt')['input_ids'][0].tolist()
eng.hard_reset()
logits = eng.prefill(ids)
nxt = int(logits[:, -1].argmax(-1).item())
out = [nxt]
t = len(ids)
t0 = time.time()
for _ in range(N - 1):
    eng._set_token(nxt, t)
    eng.graph.replay()
    nxt = int(eng.log1[:, -1].argmax(-1).item())
    out.append(nxt)
    t += 1
torch.cuda.synchronize()
dt = (time.time() - t0) / N
print(f'  [{"cuda" if CUDA else "triton"}] {1/dt:.2f} tok/s '
      f'({dt*1000:.1f}ms/tok) -> {tok.decode(out[:20])[:60]!r}', flush=True)
