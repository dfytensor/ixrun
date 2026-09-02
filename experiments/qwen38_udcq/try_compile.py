# -*- coding: utf-8 -*-
"""Try torch.compile(mode='reduce-overhead') on the slim-resident 27B —
the GEMV kernels are ~53ms/token while e2e is ~770ms; the gap is Python
dispatch. CUDA-graph capture via compile is the cheap shot."""
import sys, os, time, gc
sys.setrecursionlimit(20000)
sys.path.insert(0, r'E:\IXRUN')
import pandas
import torch
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained(r'E:\models\Qwen3.8-27B')

# reuse the slim deploy by importing its functions
sys.argv = ['x']
import importlib.util
spec = importlib.util.spec_from_file_location(
    'slim', r'E:\IXRUN\experiments\qwen38_udcq\qwen38_slim_resident.py')
slim = importlib.util.module_from_spec(spec)
spec.loader.exec_module(slim)

print('deploying slim...', flush=True)
m = slim.load_model()
slim.deploy_slim(m, verbose=True)
m.eval()
print(f'resident GPU = {torch.cuda.memory_allocated()/1e9:.2f}GB', flush=True)

# ---- baseline speed (eager) ----
txt0, tps0 = slim.gen(m, 'The theory of relativity states that', 30)
print(f'\n[eager] {tps0:.2f} tok/s\n  -> {txt0[:80]!r}', flush=True)

# ---- compile attempt ----
print('\ncompiling (reduce-overhead)...', flush=True)
t0 = time.time()
try:
    mc = torch.compile(m, mode='reduce-overhead', fullgraph=False)
    txt1, tps1 = slim.gen(mc, 'The theory of relativity states that', 30)
    print(f'[compiled] {tps1:.2f} tok/s (compile {time.time()-t0:.0f}s)\n'
          f'  -> {txt1[:80]!r}', flush=True)
    # second prompt (recompiled paths warm)
    txt2, tps2 = slim.gen(mc, 'def quick_sort(arr):', 30)
    print(f'[compiled 2nd] {tps2:.2f} tok/s\n  -> {txt2[:60]!r}', flush=True)
except Exception as e:
    print(f'compile FAILED after {time.time()-t0:.0f}s: '
          f'{type(e).__name__}: {str(e)[:300]}', flush=True)
