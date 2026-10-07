# -*- coding: utf-8 -*-
"""Stage 3b: full GatedDeltaNet decode-step assembly gate.
C++ kernel chain vs torch ref per docs/gdn_layer_spec.md.
Projections shared (torch matmul both sides) to isolate math."""
import sys
sys.path.insert(0, r'E:\IXRUN')
import pandas  # noqa: F401
import torch
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

src = open(r'E:\IXRUN\ixrun\cpp\engine_v5.cu',
           encoding='utf-8').read()
proto = '''
torch::Tensor conv1d_update_out(torch::Tensor x,
    torch::Tensor conv_state, torch::Tensor w, torch::Tensor bias,
    int64_t use_act);
torch::Tensor l2norm_out(torch::Tensor x2d);
torch::Tensor gdn_recurrent_out(torch::Tensor q, torch::Tensor k,
    torch::Tensor v, torch::Tensor g, torch::Tensor beta,
    torch::Tensor S);
torch::Tensor gated_rmsnorm_out(torch::Tensor o, torch::Tensor z,
    torch::Tensor w, double eps);
'''
ext = load_inline(name='ixrun_cpp_v5gdn3', cpp_sources=[proto],
                  cuda_sources=[src],
                  functions=['conv1d_update_out', 'l2norm_out',
                             'gdn_recurrent_out',
                             'gated_rmsnorm_out'],
                  extra_cuda_cflags=['-O3', '--use_fast_math',
                                     '-allow-unsupported-compiler'],
                  verbose=False)

g = torch.Generator(device='cuda').manual_seed(23)
# structure-valid dims (27B uses config values; math is dim-agnostic)
hidden, nk, nv, dk, dv, K = 256, 4, 8, 64, 64, 4
key_dim, value_dim = nk * dk, nv * dv
conv_dim = key_dim * 2 + value_dim
eps = 1e-6

h = torch.randn(hidden, generator=g, device='cuda').float()
Wqkv = torch.randn(conv_dim, hidden, generator=g,
                   device='cuda').float() * 0.05
Wz = torch.randn(value_dim, hidden, generator=g,
                 device='cuda').float() * 0.05
Wout = torch.randn(hidden, value_dim, generator=g,
                   device='cuda').float() * 0.05
conv_w = torch.randn(conv_dim, K, generator=g,
                     device='cuda').float() * 0.1
conv_b = torch.randn(conv_dim, generator=g, device='cuda').float()
conv_state0 = torch.randn(conv_dim, K - 1, generator=g,
                          device='cuda').float()
norm_w = torch.randn(dv, generator=g, device='cuda').float()
A_log = torch.randn(nv, generator=g, device='cuda').float()
dt_bias = torch.randn(nv, generator=g, device='cuda').float()
S0 = torch.randn(nv, dk, dv, generator=g,
                 device='cuda').float() * 0.1

def gates(hv):
    b = hv[:nv]                    # in_proj_b out (shared weight)
    a = hv[nv:2 * nv]              # in_proj_a out
    return b, a

# shared projections (same on both sides)
hv_small = torch.randn(2 * nv, generator=g, device='cuda').float()
b_in, a_in = hv_small[:nv], hv_small[nv:]
qkv = (Wqkv @ h)
z = (Wz @ h)

# ---- torch reference (spec order) ----
def ref_step(qkv, z, b_in, a_in, conv_state, S):
    # conv1d_update + silu
    new = torch.cat([conv_state, qkv.unsqueeze(1)], dim=1)
    st = new[:, 1:]
    mq = F.conv1d(new.unsqueeze(0), conv_w.unsqueeze(1),
                  conv_b, groups=conv_dim)[:, :, -1].squeeze(0)
    mq = F.silu(mq)
    q, k, v = mq.split([key_dim, key_dim, value_dim])
    q = q.reshape(nk, dk)
    k = k.reshape(nk, dk)
    v = v.reshape(nv, dv)
    beta = torch.sigmoid(b_in)
    gvec = -torch.exp(A_log) * F.softplus(a_in + dt_bias)
    q = q.repeat_interleave(nv // nk, dim=0)
    k = k.repeat_interleave(nv // nk, dim=0)
    q = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    k = k / k.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    q = q / (dk ** 0.5)
    S2 = S * gvec.exp().view(-1, 1, 1)
    kv_mem = (S2 * k.unsqueeze(-1)).sum(1)
    delta = (v - kv_mem) * beta.unsqueeze(-1)
    S2 = S2 + k.unsqueeze(-1) * delta.unsqueeze(1)
    o = (S2 * q.unsqueeze(-1)).sum(1)
    o32 = o
    o32 = o32 * torch.rsqrt(o32.pow(2).mean(-1, keepdim=True) + eps)
    o32 = norm_w * o32
    o32 = o32 * F.silu(z.reshape(nv, dv).float())
    return o32.reshape(-1) @ Wout.T, st, S2

o_ref, st_ref, S_ref = ref_step(qkv, z, b_in, a_in,
                                conv_state0.clone(), S0.clone())

# ---- C++ chain ----
st_cpp = conv_state0.clone()
S_cpp = S0.clone()
mq = ext.conv1d_update_out(qkv, st_cpp, conv_w, conv_b, 1)
q, k, v = mq.split([key_dim, key_dim, value_dim])
q = q.reshape(nk, dk)
k = k.reshape(nk, dk)
v = v.reshape(nv, dv)
q = q.repeat_interleave(nv // nk, dim=0).contiguous()
k = k.repeat_interleave(nv // nk, dim=0).contiguous()
q = ext.l2norm_out(q)
k = ext.l2norm_out(k)
q = q / (dk ** 0.5)
beta = torch.sigmoid(b_in)
gvec = -torch.exp(A_log) * F.softplus(a_in + dt_bias)
o = ext.gdn_recurrent_out(q.contiguous(), k.contiguous(),
                          v.contiguous(), gvec.contiguous(),
                          beta.contiguous(), S_cpp)
zr = z.reshape(nv, dv).contiguous()
o = ext.gated_rmsnorm_out(o, zr, norm_w, eps)
o_cpp = o.reshape(-1) @ Wout.T
torch.cuda.synchronize()

eo = ((o_cpp.double() - o_ref.double()).norm()
      / o_ref.double().norm()).item()
eS = ((S_cpp.double() - S_ref.double()).norm()
      / S_ref.double().norm()).item()
est = ((st_cpp - st_ref).norm()
       / (st_ref.norm() + 1e-9)).item()
print(f'layer out rel-err: {eo:.2e}', flush=True)
print(f'State S rel-err: {eS:.2e}', flush=True)
print(f'conv_state rel-err: {est:.2e}', flush=True)
assert eo < 1e-4 and eS < 1e-4 and est < 1e-6, 'LAYER GATE FAIL'
print('GDN LAYER GATE PASSED', flush=True)
