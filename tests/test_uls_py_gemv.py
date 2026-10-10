# -*- coding: utf-8 -*-
"""PY ext (udcq_gemv_cuda v4) ULS gate: single-T + mt4 bit-exact vs
sequential single-token ULS calls + torch i8-scale reference."""
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
p, q = wb['layers'][KEY], ub['layers'][KEY]

from experiments.udcq_gemv_cuda.udcq_gemv_cuda import (cuda_gemv,
                                                       cuda_gemv_mt,
                                                       install_codebook)
t0 = time.perf_counter()
install_codebook(cb)
print(f'ext built {time.perf_counter()-t0:.0f}s', flush=True)

x = (torch.randn(INF, device='cuda') * 0.5).to(torch.bfloat16)
y0 = cuda_gemv(x, p['idx'].cuda(), p['sign'].cuda(), p['scale'].cuda(),
               cb, OF, INF, uls=0).float()
y1 = cuda_gemv(x, q['idx'].cuda(), q['sign'].cuda(), q['scale'].cuda(),
               cb, OF, INF, uls=1).float()

# torch reference with i8 scales
buf = q['scale'].cuda()
hdr = buf[:, :8].contiguous().view(torch.float32).view(OF, 2)
ib = buf[:, 8:].float()
sd = torch.exp2(hdr[:, 0:1] + hdr[:, 1:2] * ib)
fl = q['idx'].cuda().view(OF, INF // 2).long()
nib = torch.empty(OF, INF, dtype=torch.long, device='cuda')
nib[:, 0::2] = fl & 0xF
nib[:, 1::2] = fl >> 4
sg = q['sign'].cuda().view(OF, INF // 32, 1).int()
sgn = ((sg >> torch.arange(32, device='cuda')) & 1).reshape(OF, INF).float() * 2 - 1
W = cb[nib] * sd.repeat_interleave(16, 1) * sgn
yref = (W @ x.float().unsqueeze(-1)).squeeze(-1).to(torch.bfloat16).float()
del W, nib
torch.cuda.empty_cache()
r1 = ((y1 - yref).abs().max() / yref.abs().max()).item()
d01 = ((y1 - y0).abs().max() / y0.abs().max()).item()
print(f'PY ULS single vs torch i8 ref : {r1:.3e}', flush=True)
print(f'PY ULS vs legacy single       : {d01:.3e}', flush=True)
assert r1 < 5e-3, 'ULS single-T gate FAILED'

# mt4: ULS vs 4 sequential single-token ULS calls
# NOTE: kernel outputs are bf16; boundary-straddling elements can differ by
# 1 bf16 ulp (~4e-3 rel). Compare legacy-mode the same way as the control.
X4 = (torch.randn(4, INF, device='cuda') * 0.5).to(torch.bfloat16)
ym = cuda_gemv_mt(X4, q['idx'].cuda(), q['sign'].cuda(), q['scale'].cuda(),
                  OF, INF, uls=1).float()
ymr = torch.stack([cuda_gemv(X4[i], q['idx'].cuda(), q['sign'].cuda(),
                             q['scale'].cuda(), cb, OF, INF, uls=1).float()
                   for i in range(4)])
dmt = ((ym - ymr).abs().max() / ymr.abs().max()).item()
n_bad = ((ym - ymr).abs() > 1e-4 * ymr.abs().max()).sum().item()
print(f'PY ULS mt4 vs 4x single       : {dmt:.3e} '
      f'(elems > 1e-4 rel: {n_bad}/{ym.numel()})', flush=True)

ymL = cuda_gemv_mt(X4, p['idx'].cuda(), p['sign'].cuda(), p['scale'].cuda(),
                   OF, INF, uls=0).float()
ymrL = torch.stack([cuda_gemv(X4[i], p['idx'].cuda(), p['sign'].cuda(),
                              p['scale'].cuda(), cb, OF, INF, uls=0).float()
                    for i in range(4)])
dmtL = ((ymL - ymrL).abs().max() / ymrL.abs().max()).item()
n_badL = ((ymL - ymrL).abs() > 1e-4 * ymrL.abs().max()).sum().item()
print(f'legacy mt4 vs 4x single       : {dmtL:.3e} '
      f'(elems > 1e-4 rel: {n_badL}/{ymL.numel()})', flush=True)
assert dmt < 8e-3 and n_bad < 32, 'mt4 ULS deviates beyond bf16-ulp tier'
assert dmtL < 8e-3 and n_badL < 32, 'legacy mt4 baseline deviates (control)'
print('PASS', flush=True)
