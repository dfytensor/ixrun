# -*- coding: utf-8 -*-
"""int8 dequant chain gate: q/rowscale vs the exact reference formula."""
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
cb = ub['codebook'].float().cuda()
p = ub['layers']['model.layers.0.mlp.gate_proj']
OF, INF = 17408, 5120
idx, sg, sc = p['idx'].cuda(), p['sign'].cuda(), p['scale'].cuda()

q, rs = ext.i8_dequant_test(idx, sg, sc, cb, OF, INF)

# reference weight decode
buf = sc
hdr = buf[:, :8].contiguous().view(torch.float32).view(OF, 2)
ib = buf[:, 8:].float()
sd = torch.exp2(hdr[:, 0:1] + hdr[:, 1:2] * ib)          # [OF, n_gr]
fl = idx.view(OF, INF // 2).long()
nib = torch.empty(OF, INF, dtype=torch.long, device='cuda')
nib[:, 0::2] = fl & 0xF
nib[:, 1::2] = fl >> 4
sgn = (((sg.view(OF, INF // 32, 1).int()
         >> torch.arange(32, device='cuda')) & 1)
       .reshape(OF, INF).float() * 2 - 1)
W = cb[nib] * sd.repeat_interleave(16, 1) * sgn             # [OF, INF]

ref_rs = W.abs().amax(dim=1) / 127.0
rs_err = ((rs - ref_rs).abs() / ref_rs.clamp_min(1e-9)).max().item()
print(f'rowscale rel err vs true row max/127: {rs_err:.4f}', flush=True)

ref_q = torch.round(W / ref_rs[:, None]).clamp(-127, 127)
q_err = (q.float() - ref_q).abs()
print(f'q max abs diff: {q_err.max().item():.0f} '
      f'(frac elems off by >1: {(q_err > 1).float().mean().item():.5f})',
      flush=True)
# quant reconstruction error (the real quality metric)
Wq = q.float() * rs[:, None]
rel = ((Wq - W).abs().max() / W.abs().max()).item()
print(f'quant reconstruction max rel: {rel:.5f}', flush=True)
