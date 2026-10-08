# -*- coding: utf-8 -*-
"""rmsnorm_fw_kernel gate: C++ kernel vs torch formula at real scale."""
import sys
sys.path.insert(0, r'E:\IXRUN')
import pandas  # noqa: F401
import torch
from torch.utils.cpp_extension import load_inline
from ixrun.config import QWEN38_PATH

src = open(r'E:\IXRUN\ixrun\cpp\engine_v5.cu',
           encoding='utf-8').read()
src27 = open(r'E:\IXRUN\ixrun\cpp\engine_27b.cu',
             encoding='utf-8').read()
proto = '''
torch::Tensor rmsnorm_fw_out(torch::Tensor x2d, torch::Tensor w,
    double eps);
torch::Tensor gdn_recurrent_out(torch::Tensor q, torch::Tensor k,
    torch::Tensor v, torch::Tensor g, torch::Tensor beta,
    torch::Tensor S);
torch::Tensor gated_rmsnorm_out(torch::Tensor o, torch::Tensor z,
    torch::Tensor w, double eps);
torch::Tensor conv1d_update_out(torch::Tensor x,
    torch::Tensor conv_state, torch::Tensor w, torch::Tensor bias,
    int64_t use_act);
torch::Tensor l2norm_out(torch::Tensor x2d);
'''
ext = load_inline(name='ixrun_rnfw2', cpp_sources=[proto],
                  cuda_sources=[src, src27],
                  functions=['rmsnorm_fw_out', 'gdn_recurrent_out',
                             'gated_rmsnorm_out',
                             'conv1d_update_out', 'l2norm_out'],
                  extra_cuda_cflags=['-O3', '--use_fast_math',
                                     '-allow-unsupported-compiler'],
                  verbose=False)

g = torch.Generator(device='cuda').manual_seed(59)
d = 5120
h = torch.randn(d, generator=g, device='cuda').float() * 0.015
w = torch.randn(d, generator=g, device='cuda').float()
y = ext.rmsnorm_fw_out(h.view(1, -1), w, 1e-6).view(-1)
y_ref = w * h * torch.rsqrt(h.pow(2).mean() + 1e-6)
e = ((y - y_ref).norm() / y_ref.norm()).item()
print(f'[randn-scale] rmsnorm_fw rel-err: {e:.2e}', flush=True)

# real embed row
blob = torch.load(r'E:\IXRUN\experiments\qwen38_udcq\q38_blob.pt',
                  map_location='cpu', mmap=True, weights_only=True)
he = blob['embed'][760].cuda().float()
y2 = ext.rmsnorm_fw_out(he.view(1, -1), w, 1e-6).view(-1)
y2r = w * he * torch.rsqrt(he.pow(2).mean() + 1e-6)
e2 = ((y2 - y2r).norm() / y2r.norm()).item()
print(f'[real-embed ] rmsnorm_fw rel-err: {e2:.2e} '
      f'(in norm {he.norm():.2f})', flush=True)

# gdn recurrent at real-scale inputs (zero S, tiny q/k/v)
nv, dk, dv = 48, 128, 128
q = torch.randn(nv, dk, generator=g, device='cuda').float() * 0.01
k = torch.randn(nv, dk, generator=g, device='cuda').float() * 0.01
v = torch.randn(nv, dv, generator=g, device='cuda').float() * 0.01
gv = -torch.exp(torch.randn(nv, generator=g,
                            device='cuda')).float()
beta = torch.sigmoid(torch.randn(nv, generator=g,
                                 device='cuda')).float()
S0 = torch.zeros(nv, dk, dv, device='cuda').float()
S1 = S0.clone()
o1 = ext.gdn_recurrent_out(q * 0.0883883, k, v, gv, beta, S1)
S2 = S0.clone()
S2 = S2 * gv.exp().view(-1, 1, 1)
kv_mem = (S2 * k.unsqueeze(-1)).sum(1)
delta = (v - kv_mem) * beta.unsqueeze(-1)
S2 = S2 + k.unsqueeze(-1) * delta.unsqueeze(1)
o2 = (S2 * (q * 0.0883883).unsqueeze(-1)).sum(1)
e3 = ((o1.double() - o2.double()).norm()
      / o2.double().norm().clamp_min(1e-30)).item()
print(f'[real-scale ] gdn_recurrent rel-err: {e3:.2e} '
      f'(o norm {o2.norm():.4f})', flush=True)
