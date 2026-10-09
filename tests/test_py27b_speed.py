# -*- coding: utf-8 -*-
"""PY 27B engine speed: udcq-graph vs udcq-spec, same box, greedy."""
import sys, time, os
sys.path.insert(0, r'E:\IXRUN')
os.environ['Q38_GREEDY_ONLY'] = '1'
os.environ.setdefault('HF_HUB_OFFLINE', '1')
import torch

BLOB = r'E:\IXRUN\experiments\qwen38_udcq\q38_blob.pt'
MODEL = r'E:\models\Qwen3.8-27B'
which = sys.argv[1] if len(sys.argv) > 1 else 'graph'
if which == 'graph':
    from ixrun.q38_graph import Q38GraphEngine as E
else:
    from ixrun.q38_spec import Q38SpecEngine as E

t0 = time.perf_counter()
eng = E.from_blob(BLOB, MODEL, max_ctx=256)
print(f'[load {time.perf_counter()-t0:.0f}s]', flush=True)
try:
    r = eng.generate('The capital of France is', max_new_tokens=4)
    print('warm:', repr(r), flush=True)
except Exception as e:
    print('warm failed:', e, flush=True)
t0 = time.perf_counter()
r = eng.generate('The capital of France is', max_new_tokens=32)
dt = time.perf_counter() - t0
print('out:', repr(r), flush=True)
print(f'PY-{which}: {dt:.2f}s / 32 tok = {32/dt:.1f} tok/s '
      f'({dt/32*1000:.0f} ms/tok)', flush=True)
