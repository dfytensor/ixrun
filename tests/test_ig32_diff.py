# -*- coding: utf-8 -*-
"""Locate ig32p vs ig32 divergence at scale."""
import sys
sys.path.insert(0, r'E:\IXRUN')
import torch
from benchmarks.ig32_runtime import ig32_pack, ig32_decode_ref
from experiments.ig32_gemv_cuda import (ig32_gemv, ig32p_gemv,
                                        install_tables, _load)

torch.manual_seed(3)
_load()
install_tables()
of, inf = 64, 256
W = (torch.randn(of, inf) * 0.05).cuda()
pk = ig32_pack(W)
dref = ig32_decode_ref(pk).float()
x = torch.randn(inf, dtype=torch.bfloat16, device='cuda')
y_t = ig32_gemv(x, pk).float().view(-1)
y_p = ig32p_gemv(x, pk).float().view(-1)
y_r = dref @ x.float()
dt = (y_t - y_r).abs().max().item()
dp = (y_p - y_r).abs().max().item()
print(f'tbl max {dt:.4f} | prm max {dp:.4f}', flush=True)
bad_rows = (y_p - y_r).abs() > 0.05
print(f'bad rows: {bad_rows.sum().item()} / {of}', flush=True)
print('--- full one-hot sweep, row 0, group 0 ---', flush=True)
for j in range(32):
    xj = torch.zeros(inf, dtype=torch.bfloat16, device='cuda')
    xj[j] = 1.0
    a = ig32_gemv(xj, pk).float()[0].item()
    b = ig32p_gemv(xj, pk).float()[0].item()
    cc = dref[0, j].float().item()
    mk = '' if abs(b - cc) < 0.02 else ' <-- MISMATCH'
    if mk or j < 4:
        print(f'  elem {j}: tbl {a:+.4f} prm {b:+.4f} ref {cc:+.4f}{mk}', flush=True)
n_gr = inf // 32
for g in range(n_gr):
    xg = torch.zeros(inf, dtype=torch.bfloat16, device='cuda')
    xg[g * 32:(g + 1) * 32] = x[g * 32:(g + 1) * 32]
    a = ig32_gemv(xg, pk).float().view(-1)
    b = ig32p_gemv(xg, pk).float().view(-1)
    rr = (dref[:, g * 32:(g + 1) * 32].float()
          @ x[g * 32:(g + 1) * 32].float())
    dm = (b - rr).abs().max().item()
    dtg = (a - rr).abs().max().item()
    prm0 = pk['prm'][0, g].item()
    per = (b - rr).abs()
    print('  per-row err:', ' '.join(f'{v:.3f}' for v in per.tolist()), flush=True)
    print(f'group {g}: tbl {dtg:.4f} prm {dm:.4f} (prm byte {prm0:#x})',
          flush=True)

if bad_rows.any():
    r0 = bad_rows.nonzero()[0].item()
    print(f'first bad row {r0}: prm {y_p[r0].item():+.4f} '
          f'tbl {y_t[r0].item():+.4f} ref {y_r[r0].item():+.4f}', flush=True)
    # one-hot sweep on that row
    for j in [0, 1, 31, 32, 33, 63, 100, 255]:
        xj = torch.zeros(inf, dtype=torch.bfloat16, device='cuda')
        xj[j] = 1.0
        a = ig32_gemv(xj, pk).float()[r0].item()
        b = ig32p_gemv(xj, pk).float()[r0].item()
        c = dref[r0, j].float().item()
        mark = '' if abs(b - c) < 0.02 else '  <-- MISMATCH'
        print(f'  elem {j}: tbl {a:+.4f} prm {b:+.4f} ref {c:+.4f}{mark}',
              flush=True)
