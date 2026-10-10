# -*- coding: utf-8 -*-
"""Isolated mt8q vs v2-single vs mt4 timings on real 27B shapes."""
import os
os.environ['IXRUN_PREBUILT'] = '1'
import sys, time
sys.path.insert(0, r'E:\IXRUN')
import pandas  # noqa
import torch
from ixrun.cpp_engine_27b import _build_ext

ext = _build_ext()
ub = torch.load(r'F:\models\qwen38_uls_blob.pt', map_location='cpu',
                mmap=True, weights_only=True)
cb = ub['codebook'].float().cuda()
ext.udcq_set_uls(1)

SHAPES = [
    ('gate/up', 17408, 5120, 'model.layers.0.mlp.gate_proj'),
    ('z      ', 6144, 5120, 'model.layers.0.linear_attn.in_proj_z'),
    ('down   ', 5120, 17408, 'model.layers.0.mlp.down_proj'),
    ('o_proj ', 5120, 6144, 'model.layers.0.linear_attn.out_proj'),
    ('b (48) ', 48, 5120, 'model.layers.0.linear_attn.in_proj_b'),
]


def bench(fn, n=30):
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n * 1e3


for name, of, inf, key in SHAPES:
    p = ub['layers'][key]
    idx, sg, sc = p['idx'].cuda(), p['sign'].cuda(), p['scale'].cuda()
    x1 = torch.randn(inf, device='cuda')
    x8 = torch.randn(8, inf, device='cuda')
    t1 = bench(lambda: ext.udcq_gemv_out(x1, idx, sg, sc, cb, of, inf, 16))
    t8 = bench(lambda: ext.udcq_gemv_mt8_out(x8, idx, sg, sc, cb, of, inf, 16))
    print(f'{name} {of}x{inf}: v2 {t1*1e3:.1f}us | mt8q {t8*1e3:.1f}us | '
          f'mt8q/v2 {t8/t1:.2f} | 8x-v2 {8*t1*1e3:.0f}us | vs-8x {8*t1/t8:.2f}x',
          flush=True)
