# -*- coding: utf-8 -*-
"""27B greedy-decode kernel-config sweep (in-context, AGENTS.md protocol).

For each (R, BK, warps): deploy blob, capture, time N-token generation.
Run: python -X utf8 benchmarks/bench_q38_cfg.py
"""
import sys, time, gc
sys.path.insert(0, r'E:\IXRUN')
import pandas                       # before torch (heap rule)
import torch
from transformers import AutoTokenizer

BLOB = r'E:\IXRUN\experiments\qwen38_udcq\q38_blob.pt'
MODEL = r'E:\models\Qwen3.8-27B'
N = 50

CONFIGS = [('S1-default', dict(r=4, stages=1)),
           ('R2S1', dict(r=2, stages=1)),
           ('R2B128S1', dict(r=2, bk=128, stages=1)),
           ('R4B128S1', dict(r=4, bk=128, stages=1))]

for tag, cfg in CONFIGS:
    from ixrun import udcq as U
    U.UDCQ_GEMV_R = cfg.get('r', 4)
    U.UDCQ_GEMV_BK = cfg.get('bk', 256)
    U.UDCQ_GEMV_WARPS = cfg.get('num_warps', 2)
    U.UDCQ_GEMV_STAGES = cfg.get('stages', 3)
    gc.collect(); torch.cuda.empty_cache()
    print(f'=== {tag}: R={U.UDCQ_GEMV_R} BK={U.UDCQ_GEMV_BK} '
          f'W={U.UDCQ_GEMV_WARPS} S={U.UDCQ_GEMV_STAGES} ===', flush=True)
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
    print(f'  [{tag}] {1/dt:.2f} tok/s ({dt*1000:.1f}ms/tok) -> '
          f'{tok.decode(out[:20])[:60]!r}', flush=True)
    del eng; gc.collect(); torch.cuda.empty_cache()
