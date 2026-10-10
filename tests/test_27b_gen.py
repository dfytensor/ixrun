# -*- coding: utf-8 -*-
"""27B C++ e2e: CUDA-graph decode vs eager, token-exact + speed."""
import sys, time
sys.path.insert(0, r'E:\IXRUN')
from ixrun.cpp_engine_27b import CppQwen27bEngine

eng = CppQwen27bEngine.from_blob(
    sys.argv[1] if len(sys.argv) > 1
    else r'E:\IXRUN\experiments\qwen38_udcq\q38_blob.pt',
    r'E:\models\Qwen3.8-27B', ctx=256, verbose=True)

P = "The capital of France is"
n = len(eng.tok(P).input_ids)

t0 = time.perf_counter()
te = eng.generate(P, max_new_tokens=8, graph=False)
dt_e = time.perf_counter() - t0
print(f'eager: {te!r}  [{dt_e:.2f}s]', flush=True)

t0 = time.perf_counter()
tg = eng.generate(P, max_new_tokens=8, graph=True)
dt_g = time.perf_counter() - t0
print(f'graph: {tg!r}  [{dt_g:.2f}s]', flush=True)
print('TOKEN-EXACT MATCH' if te == tg else '*** MISMATCH ***', flush=True)

t0 = time.perf_counter()
tg2 = eng.generate("The capital of France is", max_new_tokens=16, graph=True)
dt_g2 = time.perf_counter() - t0
ns = n + 16
print(f'graph steady: {dt_g2/ns*1000:.0f} ms/step = {ns/dt_g2:.1f} tok/s '
      f'| {tg2!r}', flush=True)

t0 = time.perf_counter()
tc = eng.generate("北京最值得游览的三个景点是", max_new_tokens=8, graph=True)
print(f'zh graph: {tc!r}  [{time.perf_counter()-t0:.2f}s]', flush=True)
