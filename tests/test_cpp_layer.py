# -*- coding: utf-8 -*-
"""C2 gate: C++ layer_forward vs manual torch reference, layer 0,
positions 0..2 (cache growth)."""
import sys

sys.path.insert(0, r'E:\IXRUN')
import pandas  # noqa: F401
import torch
from torch.utils.cpp_extension import load_inline
from transformers import AutoModelForCausalLM

from ixrun.config import MODEL_PATH
from benchmarks.gsq_runtime import gs_pack, gs_decode_ref

m = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, dtype=torch.bfloat16,
    trust_remote_code=True).eval().cuda()
cfg = m.config
H = cfg.hidden_size
nh = cfg.num_attention_heads
nkv = cfg.num_key_value_heads
hd = cfg.head_dim if hasattr(cfg, 'head_dim') else H // nh
theta = float(getattr(cfg, 'rope_theta', 10000.0))
CTX = 32
print(f'H={H} nh={nh} nkv={nkv} hd={hd} theta={theta}', flush=True)

sd = dict(m.named_modules())
p = 'model.layers.0.'
names = {'q': p + 'self_attn.q_proj', 'k': p + 'self_attn.k_proj',
         'v': p + 'self_attn.v_proj', 'o': p + 'self_attn.o_proj',
         'g': p + 'mlp.gate_proj', 'u': p + 'mlp.up_proj',
         'd': p + 'mlp.down_proj'}
pks = {k: gs_pack(sd[n].weight.data.cuda()) for k, n in names.items()}
inw = sd[p + 'input_layernorm'].weight.data.cuda()
pnw = sd[p + 'post_attention_layernorm'].weight.data.cuda()
W = {k: gs_decode_ref(pks[k]).cuda() for k in names}

src = open(r'E:\IXRUN\ixrun\cpp\engine.cu', encoding='utf-8').read()
proto = open(r'E:\IXRUN\tests\cpp_proto.h', encoding='utf-8').read()
ext = load_inline(name='ixrun_cpp_v4', cpp_sources=[proto],
                  cuda_sources=[src], functions=['layer_forward'],
                  extra_cuda_cflags=['-O3', '--use_fast_math',
                                     '-allow-unsupported-compiler'],
                  verbose=False)


def rms(x, w):
    xf = x.float()
    return (xf / torch.sqrt((xf * xf).mean() + 1e-5) * w.float())


def rope_ref(t, pos):
    t = t.reshape(-1, hd)
    d = torch.arange(0, hd, 2, device=t.device).float()
    th = pos / (theta ** (d / hd))
    c, s = torch.cos(th), torch.sin(th)
    out = t.clone()
    out[:, 0::2] = t[:, 0::2] * c - t[:, 1::2] * s
    out[:, 1::2] = t[:, 0::2] * s + t[:, 1::2] * c
    return out


def ref_layer(h, pos, kc, vc):
    xn = rms(h, inw)
    q = (xn @ W['q'].float().t()).reshape(nh, hd)
    k = (xn @ W['k'].float().t()).reshape(nkv, hd)
    v = (xn @ W['v'].float().t()).reshape(nkv, hd)
    q = rope_ref(q, pos)
    k = rope_ref(k, pos)
    kc[:, pos, :] = k
    vc[:, pos, :] = v
    kvh = nkv if nkv == nh else None
    qg = q.reshape(nkv, nh // nkv, hd)
    kt = kc[:, :pos + 1, :].float()
    vt = vc[:, :pos + 1, :].float()
    sc = torch.einsum('ghd,ktd->ghtk', qg, kt) / (hd ** 0.5)
    att = torch.einsum('ghtk,ktd->ghd',
                       torch.softmax(sc, dim=-1), vt)
    att = att.reshape(nh * hd)
    o = att @ W['o'].float().t()
    h1 = h.float() + o
    xn2 = rms(h1.to(torch.bfloat16), pnw)
    g = xn2 @ W['g'].float().t()
    u = xn2 @ W['u'].float().t()
    act = g / (1 + torch.exp(-g)) * u
    d = act @ W['d'].float().t()
    return (h1 + d).to(torch.bfloat16)


def cpp_layer(h, pos, kc):
    pk = pks
    return ext.layer_forward(
        h, inw, pnw,
        pk['q']['codes5'].cuda(), pk['q']['cb'].cuda(),
        pk['q']['s_i8'].cuda(), pk['q']['s_base'], pk['q']['s_step'],
        pk['q']['out_f'], pk['q']['in_f'],
        pk['k']['codes5'].cuda(), pk['k']['cb'].cuda(),
        pk['k']['s_i8'].cuda(), pk['k']['s_base'], pk['k']['s_step'],
        pk['k']['out_f'], pk['k']['in_f'],
        pk['v']['codes5'].cuda(), pk['v']['cb'].cuda(),
        pk['v']['s_i8'].cuda(), pk['v']['s_base'], pk['v']['s_step'],
        pk['v']['out_f'], pk['v']['in_f'],
        pk['o']['codes5'].cuda(), pk['o']['cb'].cuda(),
        pk['o']['s_i8'].cuda(), pk['o']['s_base'], pk['o']['s_step'],
        pk['o']['out_f'], pk['o']['in_f'],
        pk['g']['codes5'].cuda(), pk['g']['cb'].cuda(),
        pk['g']['s_i8'].cuda(), pk['g']['s_base'], pk['g']['s_step'],
        pk['g']['out_f'], pk['g']['in_f'],
        pk['u']['codes5'].cuda(), pk['u']['cb'].cuda(),
        pk['u']['s_i8'].cuda(), pk['u']['s_base'], pk['u']['s_step'],
        pk['u']['out_f'], pk['u']['in_f'],
        pk['d']['codes5'].cuda(), pk['d']['cb'].cuda(),
        pk['d']['s_i8'].cuda(), pk['d']['s_base'], pk['d']['s_step'],
        pk['d']['out_f'], pk['d']['in_f'],
        kc, pos, nh, nkv, hd, CTX, theta)


torch.manual_seed(0)
kc_ref = torch.zeros(2 * nkv, CTX, hd, dtype=torch.bfloat16,
                     device='cuda')
kc_cpp = torch.zeros_like(kc_ref)
ok = True
for pos in range(3):
    h = (torch.randn(H, device='cuda') * 0.3).to(torch.bfloat16)
    y_ref = ref_layer(h, pos, kc_ref[:nkv], kc_ref[nkv:])
    y_cpp = cpp_layer(h, pos, kc_cpp)
    rel = ((y_cpp.float() - y_ref.float()).norm()
           / y_ref.float().norm()).item()
    print(f'pos{pos} rel={rel:.4f}', flush=True)
    ok &= rel < 0.05
print('PASS' if ok else 'FAIL', flush=True)
