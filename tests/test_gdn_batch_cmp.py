# -*- coding: utf-8 -*-
"""GDN batch core A/B: per-token vs batched, token-by-token."""
import os
os.environ['IXRUN_PREBUILT'] = '1'
import sys
sys.path.insert(0, r'E:\IXRUN')
import pandas  # noqa
import torch
from ixrun.cpp_engine_27b import _build_ext

ext = _build_ext()
ext.udcq_set_uls(1)
ub = torch.load(r'F:\models\qwen38_uls_blob.pt', map_location='cpu',
                mmap=True, weights_only=True)

# use real layer-0 GDN weights (conv/A_log/dt_bias/norm) via the engine path?
# simpler: random inputs + random small weights of the correct shapes
T, NV, NK, DK, DV = 8, 48, 16, 128, 128
conv_dim = 2 * NK * DK + NV * DV
value_dim = NV * DV
torch.manual_seed(0)
qkvT = (torch.randn(T, conv_dim, device='cuda') * 0.5)
zT = (torch.randn(T, value_dim, device='cuda') * 0.5)
boT = (torch.randn(T, NV, device='cuda') * 0.5)
aoT = (torch.randn(T, NV, device='cuda') * 0.5)
conv_w = (torch.randn(conv_dim * 4, device='cuda') * 0.1)
conv_b = (torch.randn(conv_dim, device='cuda') * 0.1)
A_log = torch.randn(NV, device='cuda') * 0.1
dt_bias = torch.randn(NV, device='cuda') * 0.1
norm_w = torch.ones(DV, device='cuda')

on_pt, on_b = ext.gdn_cmp_test(qkvT, zT, boT, aoT, conv_w, conv_b,
                               A_log, dt_bias, norm_w, T, NV, NK, DK, DV)
d = (on_pt - on_b).abs()
mx = on_pt.abs().max().item()
per_tok = d.amax(dim=1) / mx
print('per-token max abs diff / scale:', per_tok.tolist(), flush=True)
print('overall rel:', (d.max() / mx).item(), flush=True)
if d.max() > 1e-5:
    t0 = int((d.amax(dim=1) > 1e-5).nonzero()[0])
    j = int(d[t0].argmax())
    print(f'first bad token {t0} elem {j}: pt {on_pt[t0][j].item():.6f} '
          f'batch {on_b[t0][j].item():.6f}', flush=True)
