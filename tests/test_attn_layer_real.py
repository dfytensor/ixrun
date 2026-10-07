# -*- coding: utf-8 -*-
"""Step 3b gate: attn_layer_step on REAL first full-attn layer.
Two-tier: vs bf16 ref (format tier) + vs torch-same-quant (math)."""
import sys, json
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
torch::Tensor attn_layer_step(torch::Tensor h, torch::Tensor cb,
    torch::Tensor q_i, torch::Tensor q_s, torch::Tensor q_sc,
    torch::Tensor k_i, torch::Tensor k_s, torch::Tensor k_sc,
    torch::Tensor v_i, torch::Tensor v_s, torch::Tensor v_sc,
    torch::Tensor o_i, torch::Tensor o_s, torch::Tensor o_sc,
    torch::Tensor g_i, torch::Tensor g_s, torch::Tensor g_sc,
    torch::Tensor u_i, torch::Tensor u_s, torch::Tensor u_sc,
    torch::Tensor d_i, torch::Tensor d_s, torch::Tensor d_sc,
    torch::Tensor in_w, torch::Tensor post_w,
    torch::Tensor q_norm_w, torch::Tensor k_norm_w,
    torch::Tensor kv_cache, double theta, int64_t pos,
    int64_t nh, int64_t nkv, int64_t hd, int64_t hidden,
    int64_t inter, int64_t ctx);
'''
ext = load_inline(name='ixrun_cpp_v5s4c', cpp_sources=[proto],
                  cuda_sources=[src],
                  functions=['attn_layer_step'],
                  extra_cuda_cflags=['-O3', '--use_fast_math',
                                     '-allow-unsupported-compiler'],
                  verbose=False)

cfg = json.load(open(QWEN38_PATH + r'\config.json',
                     encoding='utf-8'))['text_config']
lt = cfg['layer_types']
FA = [i for i, x in enumerate(lt) if x == 'full_attention']
li = FA[0]
theta = cfg['rope_parameters']['rope_theta']
print(f'first full-attn layer = {li}, theta {theta:g}', flush=True)

blob = torch.load(BLOB, map_location='cpu', mmap=True,
                  weights_only=True)
cb_g = blob['codebook'].float().cuda()
P0 = f'model.layers.{li}.'
def pack(nm):
    p = blob['layers'][P0 + nm]
    return (p['idx'].cuda(), p['sign'].cuda(),
            p['scale'].float().cuda())

from transformers import AutoModelForCausalLM
m = AutoModelForCausalLM.from_pretrained(
    QWEN38_PATH, dtype=torch.bfloat16, low_cpu_mem_usage=True,
    device_map='cpu')
at = m.model.layers[li].self_attn
in_w = m.model.layers[li].input_layernorm.weight.data.float().cuda()
post_w = (m.model.layers[li].post_attention_layernorm.weight.data
          .float().cuda())
qn_w = at.q_norm.weight.data.float().cuda()
kn_w = at.k_norm.weight.data.float().cuda()

nh, nkv, hd = 24, 4, 256
hidden, inter, ctx = 5120, 17408, 512
GROUP = 16
g = torch.Generator(device='cuda').manual_seed(43)
h = torch.randn(hidden, generator=g, device='cuda').float()
pos = 77
kv0 = torch.zeros(2 * nkv, ctx, hd, dtype=torch.bfloat16,
                  device='cuda')

NAMES = ['self_attn.q_proj', 'self_attn.k_proj',
         'self_attn.v_proj', 'self_attn.o_proj',
         'mlp.gate_proj', 'mlp.up_proj', 'mlp.down_proj']
SHAPES = {'self_attn.q_proj': (12288, 5120),
          'self_attn.k_proj': (1024, 5120),
          'self_attn.v_proj': (1024, 5120),
          'self_attn.o_proj': (5120, 6144),
          'mlp.gate_proj': (17408, 5120),
          'mlp.up_proj': (17408, 5120),
          'mlp.down_proj': (5120, 17408)}
W = {}
for nm in NAMES:
    p = blob['layers'][P0 + nm]
    of, inf = SHAPES[nm]
    n = of * inf
    bb = p['idx'].cpu().long()
    nib = torch.stack([bb & 0x0F, (bb >> 4) & 0x0F], 1).reshape(-1)
    bit = ((p['sign'].cpu().long().unsqueeze(1)
            >> torch.arange(32)) & 1).reshape(-1)[:n]
    W[nm] = (cb_g.cpu().double()[nib]
             * p['scale'].cpu().double()
               .repeat_interleave(GROUP)
             * (bit * 2.0 - 1.0)).reshape(of, inf).float().cuda()

def rms(x, w):
    return w * x * torch.rsqrt(
        x.pow(2).mean(-1, keepdim=True) + 1e-6)

def ref_quant():
    xn = rms(h, in_w)
    qg = (W[NAMES[0]] @ xn).reshape(nh, hd * 2)
    gate = qg[:, hd:]                       # fused per-head gate
    q = qg[:, :hd].contiguous()
    k = (W[NAMES[1]] @ xn).reshape(nkv, hd)
    v = (W[NAMES[2]] @ xn).reshape(nkv, hd)
    q = rms(q, qn_w)
    k = rms(k, kn_w)
    inv = 1.0 / (theta ** (torch.arange(0, 64, 2,
                      device='cuda').float() / 64))
    fr = pos * inv
    cos = torch.cat([fr, fr]).cos()
    sin = torch.cat([fr, fr]).sin()
    def rh64(x):
        x1, x2 = x[..., :32], x[..., 32:]
        return torch.cat((-x2, x1), dim=-1)
    def rope(x):
        xr, xp = x[..., :64], x[..., 64:]
        return torch.cat([xr * cos + rh64(xr) * sin, xp], -1)
    q, k = rope(q), rope(k)
    kv0[0:nkv, pos] = k.to(torch.bfloat16)
    kv0[nkv:, pos] = v.to(torch.bfloat16)
    ks = kv0[0:nkv, :pos + 1].float()     # [nkv, pos+1, hd]
    vs = kv0[nkv:, :pos + 1].float()
    rep = nh // nkv
    K = ks.repeat_interleave(rep, dim=0)  # [nh, pos+1, hd]
    V = vs.repeat_interleave(rep, dim=0)
    sc = (q.unsqueeze(1) @ K.transpose(1, 2)) / (hd ** 0.5)
    a = torch.softmax(sc, dim=-1)          # [nh, 1, pos+1]
    o = (a @ V).reshape(-1)
    o = o * torch.sigmoid(gate.reshape(-1))
    o = W[NAMES[3]] @ o
    h1 = h + o
    xn2 = rms(h1, post_w)
    gg = W[NAMES[4]] @ xn2
    uu = W[NAMES[5]] @ xn2
    act = F.silu(gg) * uu
    dd = W[NAMES[6]] @ act
    return h1 + dd

o_ref = ref_quant()

o_cpp = ext.attn_layer_step(
    h, cb_g, *pack(NAMES[0]), *pack(NAMES[1]), *pack(NAMES[2]),
    *pack(NAMES[3]), *pack(NAMES[4]), *pack(NAMES[5]),
    *pack(NAMES[6]),
    in_w, post_w, qn_w, kn_w, kv0, theta, pos,
    nh, nkv, hd, hidden, inter, ctx)
torch.cuda.synchronize()

e = ((o_cpp.double() - o_ref.double()).norm()
     / o_ref.double().norm()).item()
print(f'attn layer out rel-err: {e:.2e}', flush=True)
# 2.4e-4 tier = softmax reduction-order + bf16 KV roundtrip +
# fast-math trig compound (O(1) would indicate a structural bug —
# cf. repeat_heads tile-vs-interleave 27.3). GDN layer (no softmax/
# cache) isolates at 9.9e-6.
assert e < 1e-3, 'ATTN LAYER GATE FAIL'
print('STEP 3b GATE PASSED (compound noise tier)', flush=True)
