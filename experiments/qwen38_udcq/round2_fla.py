# -*- coding: utf-8 -*-
"""27B round 2: apply fla_patch (the missing fused linear-attention kernels),
then measure. This alone may unlock most of the 'Python dispatch' time —
the 48 linear layers were running HF's eager python fallback."""
import sys, time, gc
sys.setrecursionlimit(20000)
sys.path.insert(0, r'E:\IXRUN')
import pandas
import torch
from transformers import AutoTokenizer

sys.argv = ['x']
import importlib.util
spec = importlib.util.spec_from_file_location(
    'slim', r'E:\IXRUN\experiments\qwen38_udcq\qwen38_slim_resident.py')
slim = importlib.util.module_from_spec(spec)
spec.loader.exec_module(slim)

# THE fix: bind fla Triton kernels before model use
from ixrun.fla_patch import apply_fla_kernels
ok = apply_fla_kernels(verbose=True)
print(f'fla fused kernels active: {ok}', flush=True)

print('deploying slim...', flush=True)
m = slim.load_model()
slim.deploy_slim(m, verbose=True)
m.eval()
print(f'resident GPU = {torch.cuda.memory_allocated()/1e9:.2f}GB', flush=True)

for p in ['The theory of relativity states that',
          'def quick_sort(arr):',
          '北京最值得游览的三个景点是']:
    txt, tps = slim.gen(m, p, 40)
    print(f'\n[{tps:.2f} tok/s] {p!r}\n  -> {txt[:100]!r}', flush=True)
