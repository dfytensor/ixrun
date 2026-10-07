# -*- coding: utf-8 -*-
"""GDN primitives gate: l2norm + causal_conv1d_update vs HF math."""
import sys
sys.path.insert(0, r'E:\IXRUN')
import pandas  # noqa: F401
import torch
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

src = open(r'E:\IXRUN\ixrun\cpp\engine_v5.cu',
           encoding='utf-8').read()
proto = '''
torch::Tensor l2norm_out(torch::Tensor x2d);
torch::Tensor conv1d_update_out(torch::Tensor x,
    torch::Tensor conv_state, torch::Tensor w, torch::Tensor bias,
    int64_t use_act);
'''
ext = load_inline(name='ixrun_cpp_v5gdn2', cpp_sources=[proto],
                  cuda_sources=[src],
                  functions=['l2norm_out', 'conv1d_update_out'],
                  extra_cuda_cflags=['-O3', '--use_fast_math',
                                     '-allow-unsupported-compiler'],
                  verbose=False)

g = torch.Generator(device='cuda').manual_seed(17)

# A. l2norm vs HF l2norm()
x = torch.randn(8, 128, generator=g, device='cuda').float()
y1 = ext.l2norm_out(x)
inv = torch.rsqrt((x * x).sum(-1, keepdim=True) + 1e-6)
y_ref = x * inv
e = ((y1 - y_ref).norm() / y_ref.norm()).item()
print(f'l2norm rel-err: {e:.2e}', flush=True)
assert e < 1e-5

# B. causal_conv1d_update vs HF body (3 sequential steps)
C, K = 512, 4
state_len = K - 1
w = torch.randn(C, K, generator=g, device='cuda').float()
bias = torch.randn(C, generator=g, device='cuda').float()
xs = [torch.randn(C, generator=g, device='cuda').float()
      for _ in range(3)]

st_cpp = torch.randn(C, state_len, generator=g,
                     device='cuda').float()
st_ref = st_cpp.clone()
for t, xt in enumerate(xs):
    oc = ext.conv1d_update_out(xt, st_cpp, w, bias, 1)
    # HF body
    new = torch.cat([st_ref, xt.unsqueeze(1)], dim=1)  # [C, K]
    st_ref.copy_(new[:, 1:])
    o_ref = F.conv1d(new.unsqueeze(0), w.unsqueeze(1),
                     bias, groups=C)[:, :, -1].squeeze(0)
    o_ref = F.silu(o_ref)
    er = ((oc - o_ref).norm() / o_ref.norm()).item()
    es = ((st_cpp - st_ref).norm()
          / (st_ref.norm() + 1e-9)).item()
    print(f'step {t}: out rel-err {er:.2e} state {es:.2e}',
          flush=True)
    assert er < 1e-5 and es < 1e-6, 'conv gate FAIL'
print('GDN PRIMITIVES GATE PASSED', flush=True)
