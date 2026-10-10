# -*- coding: utf-8 -*-
"""Per-shape decode GEMV rates on the ULS blob (find the drag)."""
import os
os.environ['IXRUN_PREBUILT'] = '1'
import sys, time
sys.path.insert(0, r'E:\IXRUN')
import pandas  # noqa
import torch
from ixrun.cpp_engine_27b import _build_ext

ext = _build_ext()
ext.udcq_set_uls(1)
ub = torch.load(r'F:\models\qwen38_uls_blob.pt', map_location='cpu',
                mmap=True, weights_only=True)
cb = ub['codebook'].float().cuda()

SHAPES = [
    ('qkv(z-b-a merged)  ', 10240, 5120, 'model.layers.0.linear_attn.in_proj_qkv'),
    ('z                  ', 6144, 5120, 'model.layers.0.linear_attn.in_proj_z'),
    ('q                  ', 12288, 5120, 'model.layers.3.self_attn.q_proj'),
    ('gate               ', 17408, 5120, 'model.layers.0.mlp.gate_proj'),
    ('down               ', 5120, 17408, 'model.layers.0.mlp.down_proj'),
    ('o_proj (in=6144)   ', 5120, 6144, 'model.layers.0.linear_attn.out_proj'),
    ('lm_head            ', 248320, 5120, None),
]


def bench(fn, n=40):
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n * 1e3


tot_ideal = 0.0
for name, of, inf, key in SHAPES:
    if key:
        p = ub['layers'][key]
    else:
        p = ub['layers']['lm_head']
    idx, sg, sc = p['idx'].cuda(), p['sign'].cuda(), p['scale'].cuda()
    x = (torch.randn(inf, device='cuda') * 0.5).to(torch.bfloat16).float()
    t = bench(lambda: ext.udcq_gemv_out(x, idx, sg, sc, cb, of, inf, 16))
    gb = of * inf * 0.6875 / 1e9
    print(f'{name} [{of}x{inf}] {t*1e3:7.1f}us {gb/t*1e3:6.0f}GB/s '
          f'(ideal@900 {gb/0.9*1e3:6.1f}us)', flush=True)
