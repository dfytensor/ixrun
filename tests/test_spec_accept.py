# -*- coding: utf-8 -*-
"""Acceptance-pattern forensics for Q38SpecEngine: per-iteration
(pend, committed) distribution via graph-replay proxies."""
import os, sys, time
os.environ['UDCQ_CUDA_GEMV'] = '1'
os.environ['Q38_GREEDY_ONLY'] = '1'
sys.path.insert(0, r'E:\IXRUN')
import pandas  # noqa
import torch

from ixrun.q38_spec import Q38SpecEngine

BLOB = r'E:\IXRUN\experiments\qwen38_udcq\q38_blob.pt'
MODEL = r'E:\models\Qwen3.8-27B'
CTX = 128

eng = Q38SpecEngine.from_blob(BLOB, MODEL, max_ctx=CTX)
torch.cuda.synchronize()

# proxies record the pend (chain-graph index) per replay
seq = []
class Proxy:
    def __init__(self, g, k):
        self.g, self.k = g, k
    def replay(self):
        seq.append(self.k)
        self.g.replay()
eng.g_cp = {p: Proxy(g, p) for p, g in eng.g_cp.items()}
eng.g4dec = Proxy(eng.g4dec, -1)

orig = eng._spec_iter
iters = []
def wrapper(ids, mn, **kw):
    for batch in orig(ids, mn, **kw):
        iters.append(len(batch))
        yield batch
eng._spec_iter = wrapper

prompt = "The capital of France is"
eng.generate(prompt, max_new_tokens=4)
seq.clear(); iters.clear()
t0 = time.perf_counter()
r = eng.generate(prompt, max_new_tokens=200)
torch.cuda.synchronize()
dt = time.perf_counter() - t0

# per iteration: pend = the g_cp index (1..4), committed = iters[i]
pends = [k for k in seq if k > 0]
print(f'iters {len(pends)} tok {sum(iters)} E {sum(iters)/len(pends):.2f} '
      f'time {dt:.2f}s')
from collections import Counter
pc = Counter(pends)
cc = Counter(iters)
print('pend histogram (graph used per iter):', dict(sorted(pc.items())))
print('committed-per-iter histogram:', dict(sorted(cc.items())))
# joint: pend -> committed
joint = {}
for p, c in zip(pends, iters):
    joint.setdefault(p, Counter())[c] += 1
for p in sorted(joint):
    print(f'  pend={p}: committed {dict(sorted(joint[p].items()))}')
print('text:', repr(r[:80]), flush=True)
