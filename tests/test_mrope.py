# -*- coding: utf-8 -*-
"""27B mRoPE kernel gate vs HF formula (partial rotary 64/256)."""
import sys
sys.path.insert(0, r'E:\IXRUN')
import pandas  # noqa: F401
import torch
from torch.utils.cpp_extension import load_inline

src = open(r'E:\IXRUN\ixrun\cpp\engine_v5.cu',
           encoding='utf-8').read()
proto = '''
torch::Tensor rope27_out(torch::Tensor q, torch::Tensor k,
    int64_t n_heads, int64_t n_kv_heads, int64_t head_dim,
    double theta, int64_t pos);
'''
ext = load_inline(name='ixrun_cpp_v5mr1', cpp_sources=[proto],
                  cuda_sources=[src],
                  functions=['rope27_out'],
                  extra_cuda_cflags=['-O3', '--use_fast_math',
                                     '-allow-unsupported-compiler'],
                  verbose=False)

g = torch.Generator(device='cuda').manual_seed(41)
nh, nkv, hd = 24, 4, 256
theta, pos = 1e7, 123

q = torch.randn(nh, hd, generator=g, device='cuda').float()
k = torch.randn(nkv, hd, generator=g, device='cuda').float()
q_ref, k_ref = q.clone(), k.clone()

# HF formula: inv_freq[32], emb=cat(f,f), rotate first 64 dims
inv = 1.0 / (theta ** (torch.arange(0, 64, 2,
                                    device='cuda').float() / 64))
fr = pos * inv                                       # [32]
cos = torch.cat([fr, fr]).cos()
sin = torch.cat([fr, fr]).sin()

def rot_half64(x):
    x1, x2 = x[..., :32], x[..., 32:]
    return torch.cat((-x2, x1), dim=-1)

for t, r in ((q_ref, False), (k_ref, True)):
    pass
qr, qp = q_ref[..., :64], q_ref[..., 64:]
q_ref = torch.cat([qr * cos + rot_half64(qr) * sin, qp], dim=-1)
kr, kp = k_ref[..., :64], k_ref[..., 64:]
k_ref = torch.cat([kr * cos + rot_half64(kr) * sin, kp], dim=-1)

ext.rope27_out(q, k, nh, nkv, hd, theta, pos)
torch.cuda.synchronize()

eq = ((q.double() - q_ref.double()).norm()
      / q_ref.double().norm()).item()
ek = ((k.double() - k_ref.double()).norm()
      / k_ref.double().norm()).item()
passthrough = torch.equal(q[:, 64:].view(torch.int32),
                          torch.cat([qr * cos
                                     + rot_half64(qr) * sin,
                                     qp], -1)[:, 64:].view(
                              torch.int32))
print(f'q rel-err: {eq:.2e} | k rel-err: {ek:.2e}', flush=True)
# 1e-6-ish = fast_math powf/sinf/cosf ulp tier (same as 1B rope)
assert eq < 1e-5 and ek < 1e-5, 'mROPE GATE FAIL'
print('mROPE GATE PASSED', flush=True)
