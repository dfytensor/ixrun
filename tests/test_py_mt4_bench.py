# -*- coding: utf-8 -*-
"""PY ext (v4) mt4 kernel efficiency check: mt4 vs 4x single on real packs."""
import sys
sys.path.insert(0, r'E:\IXRUN')
import pandas  # noqa
import torch

from experiments.udcq_gemv_cuda.udcq_gemv_cuda import (cuda_gemv,
                                                       cuda_gemv_mt,
                                                       cuda_gemv_mt8,
                                                       install_codebook)

ub = torch.load(r'F:\models\qwen38_uls_blob.pt', map_location='cpu',
                mmap=True, weights_only=True)
cb = ub['codebook'].float().cuda()
install_codebook(cb)

SHAPES = [
    ('gate/up', 17408, 5120, 'model.layers.0.mlp.gate_proj'),
    ('down   ', 5120, 17408, 'model.layers.0.mlp.down_proj'),
    ('o_proj ', 5120, 6144, 'model.layers.0.linear_attn.out_proj'),
]


def bench(fn, n=30):
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    t0 = torch.__dict__.get('_x', None)
    import time
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n * 1e3


for name, of, inf, key in SHAPES:
    p = ub['layers'][key]
    idx, sg, sc = p['idx'].cuda(), p['sign'].cuda(), p['scale'].cuda()
    x1 = (torch.randn(inf, device='cuda') * 0.5).to(torch.bfloat16)
    x4 = (torch.randn(4, inf, device='cuda') * 0.5).to(torch.bfloat16)
    t1 = bench(lambda: cuda_gemv(x1, idx, sg, sc, cb, of, inf, uls=1))
    t4 = bench(lambda: cuda_gemv_mt(x4, idx, sg, sc, of, inf, uls=1))
    x8 = (torch.randn(8, inf, device='cuda') * 0.5).to(torch.bfloat16)
    t8 = bench(lambda: cuda_gemv_mt8(x8, idx, sg, sc, of, inf, uls=1))
    print(f'{name} {of}x{inf}: single {t1*1e3:.0f}us | mt4 {t4*1e3:.0f}us ({t4/t1:.2f}x1) | mt8 {t8*1e3:.0f}us ({t8/t1:.2f}x1, vs8 {8*t1/t8:.2f}x)', flush=True)
