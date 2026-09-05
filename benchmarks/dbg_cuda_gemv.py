# -*- coding: utf-8 -*-
"""Debug: cuda_gemv vs triton on REAL blob layer tensors."""
import sys, torch
sys.path.insert(0, r'E:\IXRUN')
import pandas

BLOB = r'E:\IXRUN\experiments\qwen38_udcq\q38_blob.pt'
MODEL = r'E:\models\Qwen3.8-27B'

from ixrun.q38_graph import Q38GraphEngine
import importlib.util
spec = importlib.util.spec_from_file_location(
    'cg', r'E:\IXRUN\experiments\udcq_gemv_cuda\udcq_gemv_cuda.py')
cg = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cg)

from ixrun.udcq import udcq_fused_gemv
eng = Q38GraphEngine.from_blob(BLOB, MODEL, verbose=False)
m = eng.model
n = 0
for name, mod in m.named_modules():
    if type(mod).__name__ == 'UdcqLinear' and n < 2:
        n += 1
        x = (torch.randn(eng.H, device='cuda') * 0.5).to(torch.bfloat16)
        y_t = udcq_fused_gemv(x, mod._idx, mod._sign, mod._scale, mod._cb,
                              mod.out_features, mod.in_features,
                              g=mod.packed['g'])
        y_c = cg.cuda_gemv(x, mod._idx, mod._sign, mod._scale,
                           mod._cb.float(), mod.out_features,
                           mod.in_features)
        d = (y_c.float() - y_t.float()).abs().max().item()
        print(f'{name}: in={mod.in_features} out={mod.out_features} '
              f'|y| {y_t.float().abs().mean().item():.4f} '
              f'cuda-vs-triton dmax {d:.6f}', flush=True)
        if d > 1e-3:
            idx = (y_c != y_t).nonzero()[:3].flatten().tolist()
            print(f'   diff rows {idx}', flush=True)
            for r in idx:
                print(f'   row {r}: cuda {y_c[r].item():.5f} '
                      f'triton {y_t[r].item():.5f}', flush=True)
