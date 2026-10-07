# -*- coding: utf-8 -*-
"""GDN recurrent kernel gate vs torch_recurrent_gated_delta_rule."""
import sys, time
sys.path.insert(0, r'E:\IXRUN')
import pandas  # noqa: F401
import torch
from torch.utils.cpp_extension import load_inline

src = open(r'E:\IXRUN\ixrun\cpp\engine_v5.cu',
           encoding='utf-8').read()
proto = '''
torch::Tensor gdn_recurrent_out(torch::Tensor q, torch::Tensor k,
    torch::Tensor v, torch::Tensor g, torch::Tensor beta,
    torch::Tensor S);
'''
ext = load_inline(name='ixrun_cpp_v5gdn1', cpp_sources=[proto],
                  cuda_sources=[src],
                  functions=['gdn_recurrent_out'],
                  extra_cuda_cflags=['-O3', '--use_fast_math',
                                     '-allow-unsupported-compiler'],
                  verbose=False)

g = torch.Generator(device='cuda').manual_seed(13)
nh, dk, dv = 8, 128, 128
q = torch.randn(nh, dk, generator=g, device='cuda').float()
k = torch.randn(nh, dk, generator=g, device='cuda').float()
v = torch.randn(nh, dv, generator=g, device='cuda').float()
gv = (torch.randn(nh, generator=g, device='cuda') * -1).float()
beta = torch.sigmoid(torch.randn(nh, generator=g,
                                 device='cuda')).float()
S0 = torch.randn(nh, dk, dv, generator=g,
                 device='cuda').float() * 0.1

# reference: the HF torch_recurrent_gated_delta_rule body (S=1)
# all-3D, head = dim0 (avoids the 4D broadcast misalignment)
def ref(q, k, v, g, beta, S_init):
    scale = 1 / (q.shape[-1] ** 0.5)
    qs = q * scale
    S = S_init.clone()
    S = S * g.exp().view(-1, 1, 1)
    kv_mem = (S * k.unsqueeze(-1)).sum(1)        # [nh, dv]
    delta = (v - kv_mem) * beta.unsqueeze(-1)    # [nh, dv]
    S = S + k.unsqueeze(-1) * delta.unsqueeze(1) # rank-1
    o = (S * qs.unsqueeze(-1)).sum(1)            # [nh, dv]
    return o, S

o_ref, S_ref = ref(q, k, v, gv, beta, S0)

S2 = S0.clone()
o2 = ext.gdn_recurrent_out(q / (dk ** 0.5), k, v, gv, beta, S2)
torch.cuda.synchronize()

eo = ((o2.double() - o_ref.double()).norm()
      / o_ref.double().norm()).item()
eS = ((S2.double() - S_ref.double()).norm()
      / S_ref.double().norm()).item()
print(f'out  rel-err vs torch ref: {eo:.2e}', flush=True)
print(f'state rel-err vs torch ref: {eS:.2e}', flush=True)

# multi-step drift check (5 tokens)
S3 = S0.clone()
outs = []
for t in range(5):
    ot = ext.gdn_recurrent_out(q, k, v, gv, beta, S3)
    outs.append(ot)
S5_ref = None
S_ = S0.clone()
for t in range(5):
    o_r, S_r = ref(q, k, v, gv, beta, S_)
    S_ = S_r
e5 = ((S3.double() - S_.double()).norm()
      / S_.double().norm()).item()
print(f'5-step state rel-err: {e5:.2e}', flush=True)
assert eo < 1e-4 and eS < 1e-4 and e5 < 1e-4, 'GDN gate FAIL'
print('GDN GATE PASSED', flush=True)
