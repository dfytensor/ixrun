# -*- coding: utf-8 -*-
"""mt-GEMV T=4 vs T=8: correctness + timing on real 27B packs."""
import sys, time
sys.path.insert(0, r'E:\IXRUN')
import pandas  # noqa
import torch

from ixrun.udcq import udcq_fused_gemv, udcq_fused_gemv_mt

blob = torch.load(r'E:\IXRUN\experiments\qwen38_udcq\q38_blob.pt',
                  map_location='cpu', mmap=True, weights_only=True)
cb = blob['codebook'].float().cuda()
p = blob['layers']['model.layers.0.mlp.gate_proj']
of, inf = 17408, 5120
idx = p['idx'].cuda()
sign = p['sign'].cuda()
scale = p['scale'].cuda()
x = torch.randn(8, inf, dtype=torch.bfloat16, device='cuda')

for T in (4, 8):
    y = udcq_fused_gemv_mt(x[:T], idx, sign, scale, cb, of, inf)
    ref = torch.stack([udcq_fused_gemv(
        x[t], idx, sign, scale, cb, of, inf, g=16)
        for t in range(T)]).to(torch.bfloat16)
    eq = torch.equal(y, ref)
    d = (y.float() - ref.float()).abs().max().item()
    for _ in range(3):
        udcq_fused_gemv_mt(x[:T], idx, sign, scale, cb, of, inf)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(20):
        udcq_fused_gemv_mt(x[:T], idx, sign, scale, cb, of, inf)
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / 20 * 1e3
    print(f'T={T}: bit-exact {eq} maxdiff {d:.5f} | {dt:.3f}ms/call '
          f'({of*inf*0.5625/dt/1e6:.0f}GB/s eff)', flush=True)
