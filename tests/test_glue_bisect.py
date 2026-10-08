# -*- coding: utf-8 -*-
"""gdn_layer_step (scheduler path) vs direct sequence on IDENTICAL
scheduler tensors (xn from probe, real packs, zero states)."""
import sys
sys.path.insert(0, r'E:\IXRUN')
import pandas  # noqa: F401
import torch
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

src = open(r'E:\IXRUN\ixrun\cpp\engine_v5.cu',
           encoding='utf-8').read()
proto = '''
torch::Tensor gdn_layer_step(torch::Tensor h, torch::Tensor cb,
    std::vector<torch::Tensor> PK,
    torch::Tensor in_w, torch::Tensor post_w,
    torch::Tensor conv_w, torch::Tensor conv_b,
    torch::Tensor A_log, torch::Tensor dt_bias,
    torch::Tensor gnorm_w,
    torch::Tensor conv_state, torch::Tensor S,
    int64_t nv, int64_t nk, int64_t dk, int64_t dv,
    int64_t inter, int64_t l);
torch::Tensor udcq_gemv_out(torch::Tensor x, torch::Tensor idx,
    torch::Tensor sign, torch::Tensor scale, torch::Tensor cb,
    int64_t out_f, int64_t in_f, int64_t group);
torch::Tensor conv1d_update_out(torch::Tensor x,
    torch::Tensor conv_state, torch::Tensor w, torch::Tensor bias,
    int64_t use_act);
torch::Tensor l2norm_out(torch::Tensor x2d);
torch::Tensor gdn_recurrent_out(torch::Tensor q, torch::Tensor k,
    torch::Tensor v, torch::Tensor g, torch::Tensor beta,
    torch::Tensor S);
torch::Tensor gated_rmsnorm_out(torch::Tensor o, torch::Tensor z,
    torch::Tensor w, double eps);
torch::Tensor bf16_round_out(torch::Tensor x);
'''
ext = load_inline(name='ixrun_cpp_v5gl1', cpp_sources=[proto],
                  cuda_sources=[src],
                  functions=['gdn_layer_step', 'udcq_gemv_out',
                             'conv1d_update_out', 'l2norm_out',
                             'gdn_recurrent_out',
                             'gated_rmsnorm_out', 'bf16_round_out'],
                  extra_cuda_cflags=['-O3', '--use_fast_math',
                                     '-allow-unsupported-compiler'],
                  verbose=False)

BLOB = r'E:\IXRUN\experiments\qwen38_udcq\q38_blob.pt'
blob = torch.load(BLOB, map_location='cpu', mmap=True,
                  weights_only=True)
cb_g = blob['codebook'].float().cuda()
L = 'model.layers.0.linear_attn.'
GROUP = 16

def pack0(nm):
    p = blob['layers'][L + nm]
    return (p['idx'].cuda(), p['sign'].cuda(), p['scale'].cuda())

def decode0(nm):
    idx, sign, scale = pack0(nm)
    n = idx.numel() * 2
    shapes = {'in_proj_qkv': (10240, 5120), 'in_proj_z': (6144, 5120),
              'in_proj_b': (48, 5120), 'in_proj_a': (48, 5120),
              'out_proj': (5120, 6144)}
    of, inf = shapes[nm]
    bb = idx.cpu().long()
    nib = torch.stack([bb & 0x0F, (bb >> 4) & 0x0F], 1).reshape(-1)
    bit = ((sign.cpu().long().unsqueeze(1) >> torch.arange(32))
           & 1).reshape(-1)[:n]
    W = (cb_g.cpu().double()[nib]
         * scale.cpu().double().repeat_interleave(GROUP)
         * (bit * 2.0 - 1.0))
    return W.reshape(of, inf).float().cuda()

from transformers import AutoModelForCausalLM
m = AutoModelForCausalLM.from_pretrained(
    r'E:\models\Qwen3.8-27B', dtype=torch.bfloat16,
    low_cpu_mem_usage=True, device_map='cpu')
at = m.model.layers[0].linear_attn
conv_w = at.conv1d.weight.data.squeeze(1).float().cuda()
conv_b = torch.zeros(10240, device='cuda')
A_log = at.A_log.data.float().cuda()
dt_bias = at.dt_bias.data.float().cuda()
gnorm_w = at.norm.weight.data.float().cuda()

nv, nk, dk, dv = 48, 16, 128, 128
key_dim, value_dim = nk * dk, nv * dv
conv_dim = key_dim * 2 + value_dim

# scheduler's exact xn: LN(embed[760], in_w)
emb = blob['embed']
in_w = m.model.layers[0].input_layernorm.weight.data.float().cuda()
he = emb[760].cuda().float()
xn = in_w * he * torch.rsqrt(he.pow(2).mean() + 1e-6)

P = []
for nm in ('in_proj_qkv', 'in_proj_z', 'in_proj_b', 'in_proj_a',
           'out_proj'):
    P += list(pack0(nm))

st_a = torch.zeros(conv_dim, 3, device='cuda')
S_a = torch.zeros(nv, dk, dv, device='cuda')
core_a = ext.gdn_layer_step(
    xn, cb_g, P[0], P[1], P[2], P[3], P[4], P[5], P[6], P[7], P[8],
    P[9], P[10], P[11], P[12], P[13], P[14],
    conv_w, conv_b, A_log, dt_bias, gnorm_w, st_a, S_a,
    nv, nk, dk, dv, 17408, 0)

# direct sequence on same tensors
st_b = torch.zeros(conv_dim, 3, device='cuda')
S_b = torch.zeros(nv, dk, dv, device='cuda')
qkv = ext.udcq_gemv_out(xn, *P[0:3], cb_g, conv_dim, 5120, GROUP)
z = ext.udcq_gemv_out(xn, *P[3:6], cb_g, value_dim, 5120, GROUP)
b = ext.udcq_gemv_out(xn, *P[6:9], cb_g, nv, 5120, GROUP)
a = ext.udcq_gemv_out(xn, *P[9:12], cb_g, nv, 5120, GROUP)
mq = ext.conv1d_update_out(qkv, st_b, conv_w, conv_b, 1)
q, k, v = mq.split([key_dim, key_dim, value_dim])
q = q.reshape(nk, dk).repeat_interleave(3, 0).contiguous()
k = k.reshape(nk, dk).repeat_interleave(3, 0).contiguous()
v = v.reshape(nv, dv).contiguous()
q = ext.l2norm_out(q)
k = ext.l2norm_out(k)
q = q / (dk ** 0.5)
beta = torch.sigmoid(b)
gv = -torch.exp(A_log) * F.softplus(a + dt_bias)
o = ext.gdn_recurrent_out(q, k, v, gv.contiguous(),
                          beta.contiguous(), S_b)
zg = z.reshape(nv, dv).contiguous()
gated = ext.gated_rmsnorm_out(o, zg, gnorm_w, 1e-6)
core_b = ext.udcq_gemv_out(gated.reshape(-1).contiguous(),
                           *P[12:15], cb_g, 5120, 6144, GROUP)
torch.cuda.synchronize()

d = (core_a - core_b).abs().max().item()
na, nb = core_a.norm().item(), core_b.norm().item()
ca = torch.nn.functional.cosine_similarity(
    core_a.float(), core_b.float(), dim=0).item()
print(f'core norms: gdn_layer_step {na:.3f} vs direct {nb:.3f} '
      f'| maxdiff {d:.3e} | cos {ca:.4f}', flush=True)
if ca > 0.999:
    print('GLUE CLEAN — divergence is elsewhere (decoder tail/other)',
          flush=True)
else:
    print('GLUE DIVERGES — bisect inside gdn_layer_step', flush=True)
