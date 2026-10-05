# -*- coding: utf-8 -*-
"""Minimal repro: GSQ gemv at lm_head shape (151936x1536)."""
import sys

sys.path.insert(0, r'E:\IXRUN')
import pandas  # noqa: F401
import torch
from torch.utils.cpp_extension import load_inline
from benchmarks.gsq_runtime import gs_pack, gs_decode_ref
from experiments.gsq_gemv_cuda.gsq_gemv_cuda import _load as _l0

torch.manual_seed(0)
W = (torch.randn(151936, 1536) * 0.02).to(torch.bfloat16).cuda()
print('packing...', flush=True)
pk = gs_pack(W)
print('packed: codes', pk['codes5'].shape, 'cb', pk['cb'].shape,
      's', pk['s_i8'].shape, flush=True)
for k in ('codes5', 'cb', 's_i8'):
    pk[k] = pk[k].cuda()
d = gs_decode_ref(pk)
print('ref ok, rel =', ((d.float() - W.float()).norm()
                        / W.float().norm()).item(), flush=True)
x = torch.randn(1536, dtype=torch.bfloat16, device='cuda')
ext = _l0()
y = ext.gemv(x.contiguous(), pk['codes5'], pk['cb'], pk['s_i8'],
             pk['s_base'], pk['s_step'], 151936, 1536)
torch.cuda.synchronize()
y_ref = (d.float() @ x.float()).to(torch.bfloat16)
g = (y.float() - y_ref.float()).abs().max().item()
print(f'gemv gmax = {g:.4f}', flush=True)
print('PASS' if g < 0.05 else 'FAIL', flush=True)
