# -*- coding: utf-8 -*-
"""ULS (log2-scale UDCQ) kernel gate: ext gemv vs torch dequant with i8
scales (fp32 reorder tier), legacy path unchanged, and timing pair."""
import sys, time
sys.path.insert(0, r'E:\IXRUN')
import pandas  # noqa
import torch

OLD = r'E:\IXRUN\experiments\qwen38_udcq\q38_blob.pt'
ULS = r'F:\models\qwen38_uls_blob.pt'
KEY = 'model.layers.0.mlp.gate_proj'
OF, INF = 17408, 5120

wb = torch.load(OLD, map_location='cpu', mmap=True, weights_only=True)
ub = torch.load(ULS, map_location='cpu', mmap=True, weights_only=True)
cb = wb['codebook'].float().cuda()
p = wb['layers'][KEY]
q = ub['layers'][KEY]
assert bool(ub['uls'])
assert q['scale'].shape == (OF, 8 + INF // 16)

from ixrun.cpp_engine_27b import _build_ext
ext = _build_ext()
x = torch.randn(INF, dtype=torch.float32, device='cuda')

# torch reference with i8 scales
buf = q['scale'].cuda()
hdr = buf[:, :8].contiguous().view(torch.float32).view(OF, 2)
ib = buf[:, 8:].float()
n_gr = INF // 16
sd = torch.exp2(hdr[:, 0:1] + hdr[:, 1:2] * ib)        # [OF, n_gr]
i8flat = q['idx'].cuda().view(OF, INF // 2).long()
nib = torch.empty(OF, INF, dtype=torch.long, device='cuda')
nib[:, 0::2] = i8flat & 0xF
nib[:, 1::2] = i8flat >> 4
sg = q['sign'].cuda().view(OF, INF // 32, 1).int()
bit = (sg >> torch.arange(32, device='cuda')) & 1
sgn = bit.reshape(OF, INF).float() * 2 - 1
W = cb[nib] * sd.repeat_interleave(16, 1) * sgn
yref = W @ x
del W, nib
torch.cuda.empty_cache()

ext.udcq_set_uls(0)
y0 = ext.udcq_gemv_out(x, p['idx'].cuda(), p['sign'].cuda(),
                       p['scale'].cuda(), cb, OF, INF, 16)
ext.udcq_set_uls(1)
y1 = ext.udcq_gemv_out(x, q['idx'].cuda(), q['sign'].cuda(),
                       q['scale'].cuda(), cb, OF, INF, 16)
r1 = ((y1 - yref).abs().max() / yref.abs().max()).item()
r0 = ((y0 - yref).abs().max() / yref.abs().max()).item()
d01 = ((y1 - y0).abs().max() / y0.abs().max()).item()
print(f'ULS kernel vs i8-scale torch ref : {r1:.3e}', flush=True)
print(f'legacy kernel vs i8-scale ref   : {r0:.3e} (scale-round delta)', flush=True)
print(f'ULS vs legacy outputs           : {d01:.3e}', flush=True)
assert r1 < 1e-5, 'ULS kernel gate FAILED'
assert d01 < 3e-2, 'scale-round delta implausibly large'

def bench(fn, n=50):
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n * 1e3

ext.udcq_set_uls(0)
t0 = bench(lambda: ext.udcq_gemv_out(
    x, p['idx'].cuda(), p['sign'].cuda(), p['scale'].cuda(), cb, OF, INF, 16))
ext.udcq_set_uls(1)
t1 = bench(lambda: ext.udcq_gemv_out(
    x, q['idx'].cuda(), q['sign'].cuda(), q['scale'].cuda(), cb, OF, INF, 16))
print(f'legacy 6.0bpw {t0:.3f}ms | ULS 5.5bpw {t1:.3f}ms | ratio '
      f'{t1/t0:.3f}', flush=True)
print('PASS', flush=True)
