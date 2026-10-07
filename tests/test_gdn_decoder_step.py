# -*- coding: utf-8 -*-
"""Step 4 gate: gdn_decoder_step (full decoder semantics) vs explicit
sequenced C++ calls. Synthetic packs — kernels already real-gated."""
import sys
sys.path.insert(0, r'E:\IXRUN')
import pandas  # noqa: F401
import torch
from torch.utils.cpp_extension import load_inline

src = open(r'E:\IXRUN\ixrun\cpp\engine_v5.cu',
           encoding='utf-8').read()
proto = '''
torch::Tensor gdn_decoder_step(torch::Tensor h, torch::Tensor cb,
    torch::Tensor qkv_i, torch::Tensor qkv_s, torch::Tensor qkv_sc,
    torch::Tensor z_i, torch::Tensor z_s, torch::Tensor z_sc,
    torch::Tensor b_i, torch::Tensor b_s, torch::Tensor b_sc,
    torch::Tensor a_i, torch::Tensor a_s, torch::Tensor a_sc,
    torch::Tensor o_i, torch::Tensor o_s, torch::Tensor o_sc,
    torch::Tensor gg_i, torch::Tensor gg_s, torch::Tensor gg_sc,
    torch::Tensor uu_i, torch::Tensor uu_s, torch::Tensor uu_sc,
    torch::Tensor dd_i, torch::Tensor dd_s, torch::Tensor dd_sc,
    torch::Tensor in_w, torch::Tensor post_w,
    torch::Tensor conv_w, torch::Tensor conv_b,
    torch::Tensor A_log, torch::Tensor dt_bias,
    torch::Tensor gnorm_w, torch::Tensor conv_state,
    torch::Tensor S, int64_t nv, int64_t nk, int64_t dk,
    int64_t dv, int64_t inter);
torch::Tensor rmsnorm_fw_out(torch::Tensor x2d, torch::Tensor w,
    double eps);
torch::Tensor gdn_layer_step(torch::Tensor h, torch::Tensor cb,
    torch::Tensor qkv_i, torch::Tensor qkv_s, torch::Tensor qkv_sc,
    torch::Tensor z_i, torch::Tensor z_s, torch::Tensor z_sc,
    torch::Tensor b_i, torch::Tensor b_s, torch::Tensor b_sc,
    torch::Tensor a_i, torch::Tensor a_s, torch::Tensor a_sc,
    torch::Tensor o_i, torch::Tensor o_s, torch::Tensor o_sc,
    torch::Tensor conv_w, torch::Tensor conv_b,
    torch::Tensor A_log, torch::Tensor dt_bias,
    torch::Tensor norm_w, torch::Tensor conv_state,
    torch::Tensor S, int64_t nv, int64_t nk, int64_t dk,
    int64_t dv);
torch::Tensor udcq_gemv_out(torch::Tensor x, torch::Tensor idx,
    torch::Tensor sign, torch::Tensor scale, torch::Tensor cb,
    int64_t out_f, int64_t in_f, int64_t group);
'''
ext = load_inline(name='ixrun_cpp_v5s4e', cpp_sources=[proto],
                  cuda_sources=[src],
                  functions=['gdn_decoder_step', 'rmsnorm_fw_out',
                             'gdn_layer_step', 'udcq_gemv_out'],
                  extra_cuda_cflags=['-O3', '--use_fast_math',
                                     '-allow-unsupported-compiler'],
                  verbose=False)

g = torch.Generator(device='cuda').manual_seed(47)
hidden, inter = 5120, 17408
nv, nk, dk, dv = 48, 16, 128, 128
key_dim, value_dim = nk * dk, nv * dv
conv_dim = key_dim * 2 + value_dim

def mkpack(of, inf):
    return (torch.randint(0, 255, (of * inf // 2,),
                          dtype=torch.uint8, device='cuda'),
            torch.randint(-2**31, 2**31 - 1,
                          (of * inf // 32,),
                          dtype=torch.int32, device='cuda'),
            (torch.randn(of * inf // 16, generator=g,
                         device='cuda') * 0.01).float())

h = torch.randn(hidden, generator=g, device='cuda').float()
cb = torch.randn(16, generator=g, device='cuda').float()
P = {'qkv': mkpack(conv_dim, hidden), 'z': mkpack(value_dim, hidden),
     'b': mkpack(nv, hidden), 'a': mkpack(nv, hidden),
     'o': mkpack(hidden, value_dim), 'g': mkpack(inter, hidden),
     'u': mkpack(inter, hidden), 'd': mkpack(hidden, inter)}
in_w = torch.randn(hidden, generator=g, device='cuda').float()
post_w = torch.randn(hidden, generator=g, device='cuda').float()
conv_w = (torch.randn(conv_dim, 4, generator=g,
                      device='cuda') * 0.1).float()
conv_b = torch.zeros(conv_dim, device='cuda')
A_log = torch.randn(nv, generator=g, device='cuda').float()
dt_bias = torch.randn(nv, generator=g, device='cuda').float()
gnorm_w = torch.randn(dv, generator=g, device='cuda').float()
st0 = torch.randn(conv_dim, 3, generator=g,
                  device='cuda').float() * 0.1
S0 = torch.randn(nv, dk, dv, generator=g,
                 device='cuda').float() * 0.1

# ---- explicit sequence ----
st1, S1 = st0.clone(), S0.clone()
xn = ext.rmsnorm_fw_out(h.view(1, -1), in_w, 1e-6).view(-1)
core = ext.gdn_layer_step(xn, cb, *P['qkv'], *P['z'], *P['b'],
                          *P['a'], *P['o'], conv_w, conv_b,
                          A_log, dt_bias, gnorm_w, st1, S1,
                          nv, nk, dk, dv)
h1 = h + core
xn2 = ext.rmsnorm_fw_out(h1.view(1, -1), post_w, 1e-6).view(-1)
mg = ext.udcq_gemv_out(xn2, *P['g'], cb, inter, hidden, 16)
mu = ext.udcq_gemv_out(xn2, *P['u'], cb, inter, hidden, 16)
act = mg / (1 + torch.exp(-mg)) * mu
md = ext.udcq_gemv_out(act, *P['d'], cb, hidden, inter, 16)
o_ref = h1 + md

# ---- fn path ----
st2, S2 = st0.clone(), S0.clone()
o_fn = ext.gdn_decoder_step(
    h, cb, *P['qkv'], *P['z'], *P['b'], *P['a'], *P['o'],
    *P['g'], *P['u'], *P['d'],
    in_w, post_w, conv_w, conv_b, A_log, dt_bias, gnorm_w,
    st2, S2, nv, nk, dk, dv, inter)
torch.cuda.synchronize()

d = (o_fn - o_ref).abs().max().item()
eS = ((S2 - S1).abs().max().item())
est = (st2 - st1).abs().max().item()
print(f'out maxdiff {d:.3e} | S {eS:.3e} | conv {est:.3e}',
      flush=True)
assert d < 1e-5 and eS < 1e-5 and est < 1e-6, 'DECODER GATE FAIL'
print('GDN DECODER STEP GATE PASSED', flush=True)
