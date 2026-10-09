# -*- coding: utf-8 -*-
"""Multi-token state-drift bisect: C++ (real packs) vs torch ref loop
over the 5 real prompt tokens, zero init states. Prints S/conv diff
per step; sub-stage diffs at the first jump."""
import sys
sys.path.insert(0, r'E:\IXRUN')
import pandas  # noqa: F401
import torch
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

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
torch::Tensor gdn_decoder_step(torch::Tensor h, torch::Tensor cb,
    std::vector<torch::Tensor> PK,
    torch::Tensor in_w, torch::Tensor post_w,
    torch::Tensor conv_w, torch::Tensor conv_b,
    torch::Tensor A_log, torch::Tensor dt_bias,
    torch::Tensor gnorm_w,
    torch::Tensor conv_state, torch::Tensor S,
    int64_t nv, int64_t nk, int64_t dk, int64_t dv,
    int64_t inter, int64_t l);
void init27(torch::Tensor cb,
    std::vector<torch::Tensor> packs,
    std::vector<torch::Tensor> nw1,
    std::vector<torch::Tensor> nw2,
    std::vector<torch::Tensor> gex,
    std::vector<torch::Tensor> gnorm,
    std::vector<torch::Tensor> aex,
    torch::Tensor fnw,
    torch::Tensor lh_i, torch::Tensor lh_s, torch::Tensor lh_sc,
    std::vector<int64_t> attn_layers,
    int64_t hidden, int64_t inter, int64_t ctx);
int64_t step27(torch::Tensor h, int64_t pos, double theta);
void s27_set_probe(int64_t l);
torch::Tensor s27d_get_xn2();
'''
src27 = open(r'E:\IXRUN\ixrun\cpp\engine_27b.cu',
             encoding='utf-8').read()
ext = load_inline(name='ixrun_cpp_v5ms2', cpp_sources=[proto],
                  cuda_sources=[src, src27],
                  functions=['udcq_gemv_out', 'conv1d_update_out',
                             'l2norm_out', 'gdn_recurrent_out',
                             'gdn_decoder_step', 'init27', 'step27',
                             's27_set_probe', 's27d_get_xn2'],
                  extra_cuda_cflags=['-O3', '--use_fast_math',
                                     '-allow-unsupported-compiler'],
                  verbose=False)

BLOB = r'E:\IXRUN\experiments\qwen38_udcq\q38_blob.pt'
blob = torch.load(BLOB, map_location='cpu', mmap=True,
                  weights_only=True)
cb_g = blob['codebook'].float().cuda()
GROUP = 16
L = 'model.layers.0.linear_attn.'

from transformers import AutoModelForCausalLM
m = AutoModelForCausalLM.from_pretrained(
    r'E:\models\Qwen3.8-27B', dtype=torch.bfloat16,
    low_cpu_mem_usage=True, device_map='cpu')
at = m.model.layers[0].linear_attn
conv_w = at.conv1d.weight.data.squeeze(1).float().cuda()
conv_b = torch.zeros(10240, device='cuda')
A_log = at.A_log.data.float().cuda()
dt_bias = at.dt_bias.data.float().cuda()

nv, nk, dk, dv = 48, 16, 128, 128
key_dim, value_dim = nk * dk, nv * dv
conv_dim = key_dim * 2 + value_dim

def pack(nm):
    p = blob['layers'][('model.layers.0.' if 'mlp' in nm else L) + nm]
    return (p['idx'].cuda(), p['sign'].cuda(), p['scale'].cuda())

def decode(nm):
    idx, sign, scale = pack(nm)
    n = idx.numel() * 2
    of, inf = 10240, 5120
    if nm == 'in_proj_z':
        of, inf = 6144, 5120
    if nm in ('in_proj_b', 'in_proj_a'):
        of, inf = 48, 5120
    if nm == 'mlp.gate_proj' or nm == 'mlp.up_proj':
        of, inf = 17408, 5120
    if nm == 'mlp.down_proj':
        of, inf = 5120, 17408
    if nm == 'out_proj':
        of, inf = 5120, 6144
    bb = idx.cpu().long()
    nib = torch.stack([bb & 0x0F, (bb >> 4) & 0x0F], 1).reshape(-1)
    bit = ((sign.cpu().long().unsqueeze(1) >> torch.arange(32))
           & 1).reshape(-1)[:n]
    W = (cb_g.cpu().double()[nib]
         * scale.cpu().double().repeat_interleave(GROUP)
         * (bit * 2.0 - 1.0))
    return W.reshape(of, inf).float().cuda()

Wq, Wk, Wv = decode('in_proj_qkv').split(
    [key_dim, key_dim, value_dim], dim=0)
Wz = decode('in_proj_z')
Wb = decode('in_proj_b')
Wa = decode('in_proj_a')
Wg = decode('mlp.gate_proj')
Wu = decode('mlp.up_proj')
Wd = decode('mlp.down_proj')
out_w = m.model.layers[0].linear_attn.out_proj.weight.data.float().cuda()
post_w = m.model.layers[0].post_attention_layernorm.weight.data.float().cuda()
gnorm_w = at.norm.weight.data.float().cuda()

emb = blob['embed']
ids = [760, 6511, 314, 9338, 369]
in_w = m.model.layers[0].input_layernorm.weight.data.float().cuda()

st_c = torch.zeros(conv_dim, 3, device='cuda')
S_c = torch.zeros(nv, dk, dv, device='cuda')
st_r = st_c.clone()
S_r = torch.zeros(nv, dk, dv, device='cuda')
S_r_snap = []
st_r_snap = []
h_c = None
h_r = None

print(f'{"step":>4} {"dS":>10} {"dst":>10} {"dmq":>10} {"dh1":>10} {"dmlp":>10}',
      flush=True)
for pos, t in enumerate(ids):
    he = emb[t].cuda().float()
    if h_c is None:
        h_c = he.clone()
        h_r = he.clone()
    h_c_n = in_w * h_c * torch.rsqrt(h_c.pow(2).mean() + 1e-6)
    h_r_n = in_w * h_r * torch.rsqrt(h_r.pow(2).mean() + 1e-6)
    # torch ref step
    qkv_r = torch.cat([Wq @ h_r_n, Wk @ h_r_n, Wv @ h_r_n])
    z_r = Wz @ h_r_n
    b_r = Wb @ h_r_n
    a_r = Wa @ h_r_n
    new_c = torch.cat([st_r, qkv_r.unsqueeze(1)], dim=1)
    mq_r = F.conv1d(new_c.unsqueeze(0), conv_w.unsqueeze(1),
                    conv_b, groups=conv_dim)[:, :, -1].squeeze(0)
    mq_r = F.silu(mq_r)
    st_r.copy_(new_c[:, 1:])
    q_r, k_r, v_r = mq_r.split([key_dim, key_dim, value_dim])
    q_r = q_r.reshape(nk, dk).repeat_interleave(3, dim=0)
    k_r = k_r.reshape(nk, dk).repeat_interleave(3, dim=0)
    v_r = v_r.reshape(nv, dv)
    beta = torch.sigmoid(b_r)
    gv = -torch.exp(A_log) * F.softplus(a_r + dt_bias)
    q_r = q_r * torch.rsqrt((q_r * q_r).sum(-1, keepdim=True)
                            + 1e-6)
    k_r = k_r * torch.rsqrt((k_r * k_r).sum(-1, keepdim=True)
                            + 1e-6)
    q_r = q_r / (dk ** 0.5)
    S_r = S_r * gv.exp().view(-1, 1, 1)
    kv_mem = (S_r * k_r.unsqueeze(-1)).sum(1)
    delta = (v_r - kv_mem) * beta.unsqueeze(-1)
    S_r = S_r + k_r.unsqueeze(-1) * delta.unsqueeze(1)
    S_r_snap.append(S_r.clone())
    st_r_snap.append(st_r.clone())
    o_r = (S_r * q_r.unsqueeze(-1)).sum(1).reshape(-1)
    core_r = out_w @ o_r
    h1_r = h_r + core_r
    if pos == 0:
        print(f'torch-same-decode h1(pos0) norm: {h1_r.norm().item():.3f}', flush=True)
    xn2_r = post_w * h1_r * torch.rsqrt(
        h1_r.pow(2).mean() + 1e-6)
    mg_r = Wg @ xn2_r
    mu_r = Wu @ xn2_r
    md_r = Wd @ (F.silu(mg_r) * mu_r)
    h_r = h1_r + md_r
    # C++ step
    qkv_c = ext.udcq_gemv_out(h_c_n, *pack('in_proj_qkv'), cb_g,
                              conv_dim, 5120, GROUP)
    b_c = ext.udcq_gemv_out(h_c_n, *pack('in_proj_b'), cb_g, 48,
                            5120, GROUP)
    a_c = ext.udcq_gemv_out(h_c_n, *pack('in_proj_a'), cb_g, 48,
                            5120, GROUP)
    mq_c = ext.conv1d_update_out(qkv_c, st_c, conv_w, conv_b, 1)
    q_c, k_c, v_c = mq_c.split([key_dim, key_dim, value_dim])
    q_c = q_c.reshape(nk, dk).repeat_interleave(3, dim=0).contiguous()
    k_c = k_c.reshape(nk, dk).repeat_interleave(3, dim=0).contiguous()
    v_c = v_c.reshape(nv, dv).contiguous()
    q_c = ext.l2norm_out(q_c)
    k_c = ext.l2norm_out(k_c)
    q_c = q_c / (dk ** 0.5)
    beta_c = torch.sigmoid(b_c)
    gv_c = -torch.exp(A_log) * F.softplus(a_c + dt_bias)
    S_new = S_c.clone()
    o_c = ext.gdn_recurrent_out(q_c.contiguous(), k_c.contiguous(),
                                v_c, gv_c.contiguous(),
                                beta_c.contiguous(), S_new)
    S_c.copy_(S_new)
    # core out_proj + residual + MLP + residual (full decoder layer)
    core_c = ext.udcq_gemv_out(o_c, *pack('out_proj')[:2],
                               pack('out_proj')[2], cb_g, 5120,
                               6144, GROUP)
    h1_c = h_c + core_c
    h1_c_n = post_w * h1_c * torch.rsqrt(
        h1_c.pow(2).mean() + 1e-6)
    mg_c = ext.udcq_gemv_out(h1_c_n, *pack('mlp.gate_proj')[:2],
                             pack('mlp.gate_proj')[2], cb_g, 17408,
                             5120, GROUP)
    mu_c = ext.udcq_gemv_out(h1_c_n, *pack('mlp.up_proj')[:2],
                             pack('mlp.up_proj')[2], cb_g, 17408,
                             5120, GROUP)
    act_c = F.silu(mg_c) * mu_c
    md_c = ext.udcq_gemv_out(act_c, *pack('mlp.down_proj')[:2],
                             pack('mlp.down_proj')[2], cb_g, 5120,
                             17408, GROUP)
    dmlp = (md_c.double() - md_r.double()).norm().item()
    h_c = h1_c + md_c
    dS = (S_c.double() - S_r.double()).norm().item()
    dst = (st_c - st_r).abs().max().item()
    dmq = (mq_c - mq_r).abs().max().item()
    dh1 = (h1_c.double() - h1_r.double()).norm().item()
    print(f'{pos:>4} {dS:>10.3e} {dst:>10.3e} {dmq:>10.3e} '
          f'{dh1:>10.3e} {dmlp:>10.3e}', flush=True)

# ---- DECISIVE: same 5 tokens through gdn_decoder_step (scheduler
# entry) vs the torch loop above — compare conv_state/S per step ----
L0P = 'model.layers.0.'
def pack0(nm):
    pre = '' if 'mlp' in nm else 'linear_attn.'
    p = blob['layers'][L0P + pre + nm]
    return (p['idx'].cuda(), p['sign'].cuda(), p['scale'].cuda())
P = []
for nm in ('in_proj_qkv', 'in_proj_z', 'in_proj_b', 'in_proj_a',
           'out_proj', 'mlp.gate_proj', 'mlp.up_proj',
           'mlp.down_proj'):
    P += list(pack0(nm))
nw1_0 = in_w.clone()
nw2_0 = post_w.clone()
gnorm_0 = gnorm_w.clone()
fnw_d = torch.randn(5120, device='cuda')
lh_d = (torch.zeros(8, dtype=torch.uint8, device='cuda'),
        torch.zeros(1, dtype=torch.int32, device='cuda'),
        torch.zeros(1, dtype=torch.float16, device='cuda'))
ext.init27(cb_g, packs if 'packs' in dir() else [], [nw1_0],
           [nw2_0], gex if 'gex' in dir() else
           [conv_w.flatten(), conv_b, A_log, dt_bias],
           [gnorm_0], [], fnw_d, *lh_d, [], 5120, 17408, 512)
st_g = torch.zeros(conv_dim, 3, device='cuda')
S_g = torch.zeros(nv, dk, dv, device='cuda')
h_g = None
print('gdn_decoder_step vs torch:', flush=True)
for pos, t in enumerate(ids):
    he = emb[t].cuda().float()
    if h_g is None:
        h_g = he.clone()
    torch.manual_seed(1234 + pos)  # no rand use; placeholder
    ext.s27_set_probe(0)
    h_new = ext.gdn_decoder_step(
        h_g, cb_g, P, in_w, post_w, conv_w, conv_b, A_log,
        dt_bias, gnorm_w, st_g, S_g, 48, 16, 128, 128, 17408, pos)
    if pos == 0:
        xn2_c = ext.s27d_get_xn2().cuda().float()
        he0 = emb[t].cuda().float()
        xnv = in_w * he0 * torch.rsqrt(
            he0.pow(2).mean() + 1e-6) + core_c
        xn2_t = post_w * xnv * torch.rsqrt(
            xnv.pow(2).mean() + 1e-6)
        cosx = torch.nn.functional.cosine_similarity(
            xn2_c.float(), xn2_t, dim=0).item()
        print(f'xn2(pos0): cpp {xn2_c.float().norm().item():.3f} '
              f'vs torch-sq {xn2_t.float().norm().item():.3f} '
              f'cos {cosx:.4f}', flush=True)
    torch.cuda.synchronize()
    dS_g = (S_g.double() - S_r_snap[pos].double()).norm().item()
    dst_g = (st_g - st_r_snap[pos]).abs().max().item()
    dh_g = (h_new.double() - h_r.double()).norm().item()
    print(f'  step {pos}: dS {dS_g:.3e} dst {dst_g:.3e} '
          f'dh {dh_g:.3e}', flush=True)
    # advance torch ref already done above per iteration? NO — the
    # torch loop above consumed its own tokens; here we just compare
    # the C++ step against the torch states from the SAME step index.
    h_g = h_new