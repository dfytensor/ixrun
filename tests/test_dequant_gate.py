# -*- coding: utf-8 -*-
"""Dequant kernel gate (vs torch i8-scale ref) + cuBLAS bf16 matmul proof."""
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
p = ub['layers']['model.layers.0.mlp.gate_proj']
OF, INF = 17408, 5120
idx, sg, sc = p['idx'].cuda(), p['sign'].cuda(), p['scale'].cuda()

y = ext.udcq_dequant_test(idx, sg, sc, cb, OF, INF, 16)   # bf16
buf = sc
hdr = buf[:, :8].contiguous().view(torch.float32).view(OF, 2)
ib = buf[:, 8:].float()
sd = torch.exp2(hdr[:, 0:1] + hdr[:, 1:2] * ib)
fl = idx.view(OF, INF // 2).long()
nib = torch.empty(OF, INF, dtype=torch.long, device='cuda')
nib[:, 0::2] = fl & 0xF
nib[:, 1::2] = fl >> 4
sgn = (((sg.view(OF, INF // 32, 1).int()
         >> torch.arange(32, device='cuda')) & 1)
       .reshape(OF, INF).float() * 2 - 1)
W = cb[nib] * sd.repeat_interleave(16, 1) * sgn
rel = ((y.float() - W).abs().max() / W.abs().max()).item()
print(f'dequant vs fp32 ref: max rel {rel:.4f} (bf16-out tier)', flush=True)
assert rel < 1e-2, 'dequant gate failed'

def bench(fn, n=30):
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n * 1e3

t_deq = bench(lambda: ext.udcq_dequant_test(idx, sg, sc, cb, OF, INF, 16))
gbytes = (OF * INF * 0.6875 + OF * INF * 2) / 1e9
print(f'dequant gate_proj: {t_deq:.2f}ms ({gbytes/t_deq*1000:.0f}GB/s, '
      f'ideal ~{gbytes/1000*1e3:.2f}ms)', flush=True)

xt = (torch.randn(256, INF, device='cuda') * 0.5).to(torch.bfloat16)
wt = y.t().contiguous()
torch.backends.cuda.matmul.allow_tf32 = False
t_mm = bench(lambda: torch.matmul(xt, wt), n=20)
flops = 2 * 256 * OF * INF / 1e12
print(f'bf16 matmul [256x5120]x[5120x17408]: {t_mm:.2f}ms '
      f'({flops/(t_mm/1e3):.0f} TFLOPS)', flush=True)
print('PASS', flush=True)
