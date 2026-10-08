# -*- coding: utf-8 -*-
"""Stage 4 step 2: REAL layer-0 GDN assembly gate.
C++ chain (UDCQ blob packs + gated kernels) vs torch ref (REAL bf16
weights). Expected tier: UDCQ quantization ~2e-3..5e-3 (NOT math
tier — projections are quantized on C++ side only)."""
import sys, time
sys.path.insert(0, r'E:\IXRUN')
import pandas  # noqa: F401
import torch
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline
from ixrun.config import QWEN38_PATH

BLOB = r'E:\IXRUN\experiments\qwen38_udcq\q38_blob.pt'
src = open(r'E:\IXRUN\ixrun\cpp\engine_v5.cu',
           encoding='utf-8').read()
proto = '''
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
'''
ext = load_inline(name='ixrun_cpp_v5s4a', cpp_sources=[proto],
                  cuda_sources=[src],
                  functions=['udcq_gemv_out', 'conv1d_update_out',
                             'l2norm_out', 'gdn_recurrent_out',
                             'gated_rmsnorm_out'],
                  extra_cuda_cflags=['-O3', '--use_fast_math',
                                     '-allow-unsupported-compiler'],
                  verbose=False)

t0 = time.perf_counter()
blob = torch.load(BLOB, map_location='cpu', mmap=True,
                  weights_only=True)
print(f'blob {time.perf_counter()-t0:.1f}s', flush=True)
cb_g = blob['codebook'].float().cuda()
L = 'model.layers.0.linear_attn.'
def pack(nm):
    p = blob['layers'][L + nm]
    return (p['idx'].cuda(), p['sign'].cuda(),
            p['scale'].cuda())

from transformers import AutoModelForCausalLM
t0 = time.perf_counter()
m = AutoModelForCausalLM.from_pretrained(
    QWEN38_PATH, dtype=torch.bfloat16, low_cpu_mem_usage=True,
    device_map='cpu')
print(f'hf cpu load {time.perf_counter()-t0:.0f}s', flush=True)
attn = m.model.layers[0].linear_attn
nv, nk, dk, dv = 48, 16, 128, 128
key_dim, value_dim = nk * dk, nv * dv
conv_dim = key_dim * 2 + value_dim
GROUP = 16
conv_w = attn.conv1d.weight.data.squeeze(1).float().cuda()  # [C,K]
conv_b = torch.zeros(conv_dim, device='cuda')  # HF conv1d bias=False
dt_bias = attn.dt_bias.data.float().cuda()
A_log = attn.A_log.data.float().cuda()
norm_w = attn.norm.weight.data.float().cuda()
Wout_bf = attn.out_proj.weight.data  # [5120, 6144] bf16 CPU
Wqkv_bf = attn.in_proj_qkv.weight.data
Wz_bf = attn.in_proj_z.weight.data
Wb_bf = attn.in_proj_b.weight.data
Wa_bf = attn.in_proj_a.weight.data

g = torch.Generator(device='cuda').manual_seed(37)
blob = blob if 'blob' in dir() else blob
# REAL-SCALE input: actual embedding row (norm ~1.1), not randn
h = blob['embed'][760].cuda().float()
from torch.utils.cpp_extension import load_inline as _li
# replicate scheduler entry: rmsnorm with the layer's in_w FIRST
in_w = m.model.layers[0].input_layernorm.weight.data.float().cuda()
h = in_w * h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + 1e-6)
conv_state = torch.randn(conv_dim, 3, generator=g,
                         device='cuda').float() * 0.1
S0 = torch.randn(nv, dk, dv, generator=g,
                 device='cuda').float() * 0.1

# ---- torch ref (REAL bf16 weights, fp32 math) ----
def ref_step():
    qkv = (Wqkv_bf.cuda().float() @ h)
    z = (Wz_bf.cuda().float() @ h)
    b = (Wb_bf.cuda().float() @ h)
    a = (Wa_bf.cuda().float() @ h)
    new = torch.cat([conv_state, qkv.unsqueeze(1)], dim=1)
    st = new[:, 1:]
    mq = F.conv1d(new.unsqueeze(0), conv_w.unsqueeze(1),
                  conv_b, groups=conv_dim)[:, :, -1].squeeze(0)
    mq = F.silu(mq)
    q, k, v = mq.split([key_dim, key_dim, value_dim])
    q = q.reshape(nk, dk).repeat_interleave(3, dim=0)
    k = k.reshape(nk, dk).repeat_interleave(3, dim=0)
    v = v.reshape(nv, dv)
    beta = torch.sigmoid(b)
    gvec = -torch.exp(A_log) * F.softplus(a + dt_bias)
    q = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    k = k / k.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    q = q / (dk ** 0.5)
    S2 = S0 * gvec.exp().view(-1, 1, 1)
    kv_mem = (S2 * k.unsqueeze(-1)).sum(1)
    delta = (v - kv_mem) * beta.unsqueeze(-1)
    S2 = S2 + k.unsqueeze(-1) * delta.unsqueeze(1)
    o = (S2 * q.unsqueeze(-1)).sum(1)
    o = o * torch.rsqrt(o.pow(2).mean(-1, keepdim=True) + 1e-6)
    o = norm_w * o
    o = o * F.silu(z.reshape(nv, dv))
    return o.reshape(-1) @ Wout_bf.cuda().float().T, st, S2


L2 = 'model.layers.0.'
def pack2(nm):
    p = blob['layers'][L2 + nm]
    return (p['idx'].cuda(), p['sign'].cuda(),
            p['scale'].cuda())
def decode_pack2(nm):
    idx, sign, scale = pack2(nm)
    n = idx.numel() * 2
    bb = idx.cpu().long()
    nib = torch.stack([bb & 0x0F, (bb >> 4) & 0x0F], 1).reshape(-1)
    bit = ((sign.cpu().long().unsqueeze(1) >> torch.arange(32))
           & 1).reshape(-1)[:n]
    shapes = {'mlp.gate_proj': (17408, 5120),
              'mlp.up_proj': (17408, 5120),
              'mlp.down_proj': (5120, 17408),
              'linear_attn.in_proj_qkv': (10240, 5120),
              'linear_attn.in_proj_z': (6144, 5120),
              'linear_attn.out_proj': (5120, 6144)}
    of, inf = shapes[nm]
    W = (cb_g.cpu().double()[nib]
         * scale.cpu().double().repeat_interleave(GROUP)
         * (bit * 2.0 - 1.0))
    return W.reshape(of, inf).float().cuda()

o_ref, st_ref, S_ref = ref_step()
print('ref done', flush=True)

# ---- C++ chain (UDCQ packs) ----
def gemv(x, nm):
    idx, sign, scale = pack(nm)
    return ext.udcq_gemv_out(x, idx, sign, scale, cb_g,
                             idx.numel() * 2 // in_f_of(nm), 5120,
                             GROUP)
def in_f_of(nm):
    return {'in_proj_qkv': 5120, 'in_proj_z': 5120,
            'in_proj_b': 5120, 'in_proj_a': 5120,
            'out_proj': 6144}[nm]

st_cpp = conv_state.clone()
S_cpp = S0.clone()
qkv = gemv(h, 'in_proj_qkv')
z = gemv(h, 'in_proj_z')
b_o = gemv(h, 'in_proj_b')
a_o = gemv(h, 'in_proj_a')
mq = ext.conv1d_update_out(qkv, st_cpp, conv_w, conv_b, 1)
q, k, v = mq.split([key_dim, key_dim, value_dim])
q = q.reshape(nk, dk).repeat_interleave(3, dim=0).contiguous()
k = k.reshape(nk, dk).repeat_interleave(3, dim=0).contiguous()
v = v.reshape(nv, dv).contiguous()
q = ext.l2norm_out(q)
k = ext.l2norm_out(k)
q = q / (dk ** 0.5)
beta = torch.sigmoid(b_o)
gvec = -torch.exp(A_log) * F.softplus(a_o + dt_bias)
o = ext.gdn_recurrent_out(q, k, v, gvec.contiguous(),
                          beta.contiguous(), S_cpp)
zr = z.reshape(nv, dv).contiguous()
o = ext.gated_rmsnorm_out(o, zr, norm_w, 1e-6)
o_cpp = ext.udcq_gemv_out(o.reshape(-1).contiguous(),
                          *pack('out_proj')[:2],
                          pack('out_proj')[2], cb_g,
                          5120, 6144, GROUP)
torch.cuda.synchronize()

e = ((o_cpp.double() - o_ref.double()).norm()
     / o_ref.double().norm()).item()
eS = ((S_cpp.double() - S_ref.double()).norm()
      / S_ref.double().norm()).item()
print(f'REAL layer out rel-err (UDCQ vs bf16 tier): {e:.2e}',
      flush=True)
print(f'S rel-err: {eS:.2e}', flush=True)

# ---- decisive crosscheck: same QUANTIZED weights through the
# torch ref chain -> isolates C++ math from format error ----
def decode_pack(nm):
    idx, sign, scale = pack(nm)
    n = idx.numel() * 2
    bb = idx.cpu().long()
    nib = torch.stack([bb & 0x0F, (bb >> 4) & 0x0F], 1).reshape(-1)
    bit = ((sign.cpu().long().unsqueeze(1) >> torch.arange(32))
           & 1).reshape(-1)[:n]
    W = (cb_g.cpu().double()[nib]
         * scale.cpu().double().repeat_interleave(GROUP)
         * (bit * 2.0 - 1.0))
    return W.reshape(-1, 5120 if nm != 'out_proj' else 6144).float()

Wq_q = decode_pack('in_proj_qkv').cuda()
Wz_q = decode_pack('in_proj_z').cuda()
Wb_q = decode_pack('in_proj_b').cuda()
Wa_q = decode_pack('in_proj_a').cuda()
Wo_q = decode_pack('out_proj').cuda()
o_q, _, _ = (lambda: ref_step.__wrapped__)() if False else (None, None, None)
def ref_quant():
    qkv = (Wq_q @ h)
    z = (Wz_q @ h)
    b = (Wb_q @ h)
    a = (Wa_q @ h)
    new = torch.cat([conv_state, qkv.unsqueeze(1)], dim=1)
    mq = F.conv1d(new.unsqueeze(0), conv_w.unsqueeze(1),
                  conv_b, groups=conv_dim)[:, :, -1].squeeze(0)
    mq = F.silu(mq)
    q, k, v = mq.split([key_dim, key_dim, value_dim])
    q = q.reshape(nk, dk).repeat_interleave(3, dim=0)
    k = k.reshape(nk, dk).repeat_interleave(3, dim=0)
    v = v.reshape(nv, dv)
    beta = torch.sigmoid(b)
    gvec = -torch.exp(A_log) * F.softplus(a + dt_bias)
    q = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    k = k / k.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    q = q / (dk ** 0.5)
    S2 = S0 * gvec.exp().view(-1, 1, 1)
    kv_mem = (S2 * k.unsqueeze(-1)).sum(1)
    delta = (v - kv_mem) * beta.unsqueeze(-1)
    S2 = S2 + k.unsqueeze(-1) * delta.unsqueeze(1)
    o = (S2 * q.unsqueeze(-1)).sum(1)
    o = o * torch.rsqrt(o.pow(2).mean(-1, keepdim=True) + 1e-6)
    o = norm_w * o
    o = o * F.silu(z.reshape(nv, dv))
    return o.reshape(-1) @ Wo_q.T

o_qref = ref_quant()
e_q = ((o_cpp.double() - o_qref.double()).norm()
       / o_qref.double().norm()).item()
print(f'C++ vs torch-same-quant-weights: {e_q:.2e} '
      f'(math isolation)', flush=True)
assert e_q < 1e-4, 'MATH ISOLATION FAIL'
assert e < 0.1, 'FORMAT TIER UNEXPECTED'
print('REAL LAYER GATE PASSED (format tier + math clean)',
      flush=True)

# ---- MLP trio real-scale comparison (torch-same-quant vs C++) ----
Wg_q = decode_pack2('mlp.gate_proj').cuda()
Wu_q = decode_pack2('mlp.up_proj').cuda()
Wd_q = decode_pack2('mlp.down_proj').cuda()
xn2 = (Wout_q_o if False else None) or None
h1 = h + o_qref            # ref attn-layer out as operating point
xn2 = (m.model.layers[0].post_attention_layernorm.weight.data
       .float().cuda()) * h1 * torch.rsqrt(
    h1.pow(2).mean(-1, keepdim=True) + 1e-6)
mg_t = Wg_q @ xn2
mu_t = Wu_q @ xn2
act_t = F.silu(mg_t) * mu_t
md_t = Wd_q @ act_t
mg_c = ext.udcq_gemv_out(xn2, *pack2('mlp.gate_proj')[:2],
                         pack2('mlp.gate_proj')[2], cb_g,
                         17408, 5120, GROUP)
mu_c = ext.udcq_gemv_out(xn2, *pack2('mlp.up_proj')[:2],
                         pack2('mlp.up_proj')[2], cb_g,
                         17408, 5120, GROUP)
act_c = F.silu(mg_c) * mu_c
md_c = ext.udcq_gemv_out(act_c, *pack2('mlp.down_proj')[:2],
                         pack2('mlp.down_proj')[2], cb_g,
                         5120, 17408, GROUP)
for tag, a, bt in [('gate', mg_c, mg_t), ('up', mu_c, mu_t),
                   ('down', md_c, md_t)]:
    ee = ((a.double() - bt.double()).norm()
          / bt.double().norm().clamp_min(1e-30)).item()
    print(f'MLP {tag}: rel-err {ee:.2e} (norm {bt.norm():.3f})',
          flush=True)
# ---- AUTHORITATIVE decode cross-check (udcq.py ref) ----
from ixrun.udcq import _decode_udcq_ref
p = blob['layers'][L + 'in_proj_qkv']
p_full = dict(p)
p_full.update({'g': GROUP, 'sign_packed': p['sign'], 'N': p['idx'].numel() * 2,
               'out_f': 10240, 'in_f': 5120,
               'codebook': blob['codebook']})
W_auth = _decode_udcq_ref(p_full).float()
W_mine = decode_pack2('linear_attn.in_proj_qkv')
d = (W_auth.float().cuda() - W_mine).abs()
nz = (d > 1e-3).sum().item()
print(f'auth-vs-mine: maxdiff {d.max().item():.4f} '
      f'mismatch>{1e-3}: {nz}', flush=True)
if nz > 0:
    k = d.argmax().item()
    r, cc = k // 5120, k % 5120
    print(f'  worst elem ({r},{cc}): auth {W_auth.float().cuda().flatten()[k].item():.4f} '
          f'mine {W_mine.flatten()[k].item():.4f}', flush=True)