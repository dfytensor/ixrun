# -*- coding: utf-8 -*-
"""Tiny-case kernel-vs-ref debug for ig32."""
import sys
sys.path.insert(0, r'E:\IXRUN')
import torch
from benchmarks.ig32_runtime import ig32_pack, ig32_decode_ref
from experiments.ig32_gemv_cuda import ig32_gemv, ig32p_gemv, install_tables

torch.manual_seed(1)
of, inf = 4, 64
W = (torch.randn(of, inf) * 0.05).cuda()
pk = ig32_pack(W)
install_tables()
dref = ig32_decode_ref(pk).float()
x = torch.randn(inf, dtype=torch.bfloat16, device='cuda')
y = ig32_gemv(x, pk).float()
y_ref = (dref @ x.float())

print('codes[0][:8] =', pk['codes'][0, :8].tolist())
print('sign[0] =', hex(pk['sign'][0, 0].item() & 0xffffffff))
print('levels[0,0] =', [round(v, 4) for v in pk['levels'][0, 0].tolist()])
for r in range(of):
    print(f'row{r}: y={y[r].item():+.4f} ref={y_ref[r].item():+.4f}')
# manual per-element decode of row 0, first group
lv0 = pk['levels'][0, 0].float()
c0 = pk['codes'][0, :16].long()
sw = pk['sign'][0, 0].int()
man = torch.zeros(32, device='cuda')
for i in range(32):
    w = c0[i // 2]
    nib = (w >> (4 * (i % 2))) & 0xF
    s = -1.0 if ((sw >> i) & 1) else 1.0
    man[i] = s * lv0[nib]
W0 = W[0].float()
print('manual g0[:8] :', [round(v, 4) for v in man[:8].tolist()])
print('W g0[:8]      :', [round(v, 4) for v in W0[:8].tolist()])
part = (man @ x[:32].float()).item()
part_ref = (dref[0, :32].float() @ x[:32].float()).item()
print(f'g0 partial: kernel-manual {part:+.4f} vs decode-ref {part_ref:+.4f}')
print('--- one-hot element probe (row 0, first 8 elements) ---')
for j in range(8):
    xj = torch.zeros(inf, dtype=torch.bfloat16, device='cuda')
    xj[j] = 1.0
    yk = ig32_gemv(xj, pk).float()[0].item()
    yk2 = ig32p_gemv(xj, pk).float()[0].item()
    wr = dref[0, j].float().item()
    print(f'elem {j}: tbl {yk:+.4f} prm {yk2:+.4f} ref {wr:+.4f}')
x0 = torch.zeros(inf, dtype=torch.bfloat16, device='cuda'); x0[:32] = x[:32]
x1 = torch.zeros(inf, dtype=torch.bfloat16, device='cuda'); x1[32:] = x[32:]
ya = ig32_gemv(x0, pk).float(); yb = ig32_gemv(x1, pk).float()
ra = (dref[:, :32].float() @ x[:32].float())
rb = (dref[:, 32:].float() @ x[32:].float())
for r in range(of):
    print(f'row{r}: g0 {ya[r].item():+.4f} vs {ra[r].item():+.4f} | g1 {yb[r].item():+.4f} vs {rb[r].item():+.4f}')
