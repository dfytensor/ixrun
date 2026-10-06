# -*- coding: utf-8 -*-
"""Per-layer hidden norm dump: C++ chain vs Python reference, single token."""
import sys

sys.path.insert(0, r'E:\IXRUN')
import pandas  # noqa: F401
import torch
from torch.utils.cpp_extension import load_inline
from transformers import AutoTokenizer, AutoModelForCausalLM

from ixrun.config import MODEL_PATH, DATASET_CACHE
from ixrun.eval_utils import load_wikitext
from ixrun.linear import iter_quantizable_linears
from benchmarks.gsq_runtime import gs_pack, gs_decode_ref

m = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, dtype=torch.bfloat16,
    trust_remote_code=True).eval().cuda()
cfg = m.config
H = cfg.hidden_size
nh, nkv = cfg.num_attention_heads, cfg.num_key_value_heads
hd = getattr(cfg, 'head_dim', H // nh)
theta = float(getattr(cfg, 'rope_theta', 10000.0))
CTX = 32

sd = dict(m.named_modules())
targets = list(iter_quantizable_linears(m))
names = [n for n, _ in targets]
lay0 = names[0].rsplit('.', 3)[0]

pks = []
for name, mod in targets:
    pk = gs_pack(mod.weight.data.cuda())
    for k in ('codes5', 'cb', 's_i8'):
        pk[k] = pk[k].cuda()
    pks.append(pk)
Ws = [gs_decode_ref(pk).float().cuda() for pk in pks]

src = open(r'E:\IXRUN\ixrun\cpp\engine.cu', encoding='utf-8').read()
proto = open(r'E:\IXRUN\tests\cpp_proto.h', encoding='utf-8').read()
ext = load_inline(name='ixrun_cpp_v4', cpp_sources=[proto],
                  cuda_sources=[src], functions=['layer_forward'],
                  extra_cuda_cflags=['-O3', '--use_fast_math',
                                     '-allow-unsupported-compiler'],
                  verbose=False)

in_ws = [sd[lay0 + f'.{l}.input_layernorm'].weight.data.cuda()
         for l in range(24)]
post_ws = [sd[lay0 + f'.{l}.post_attention_layernorm'].weight
           .data.cuda() for l in range(24)]
embed_w = m.model.embed_tokens.weight.data.cuda()

torch.manual_seed(42)
h0 = (torch.randn(H, device='cuda') * 0.3).to(torch.bfloat16)
pos = 0

# Python per-layer reference
h_ref = h0.clone().float()
kc_ref_k = torch.zeros(2, 1, hd, device='cuda')
kc_ref_v = torch.zeros(2, 1, hd, device='cuda')
ref_norms = []
for l in range(24):
    b = l * 7
    xn = h_ref / torch.sqrt((h_ref * h_ref).mean() + 1e-5) \
        * in_ws[l].float()
    q = (xn @ Ws[b + 0].t()).reshape(nh, hd)
    k = (xn @ Ws[b + 1].t()).reshape(nkv, hd)
    v = (xn @ Ws[b + 2].t()).reshape(nkv, hd)
    # rope
    d = torch.arange(0, hd, 2, device='cuda').float()
    th = pos / (theta ** (d / hd))
    c, s = torch.cos(th), torch.sin(th)
    for t in (q, k):
        t0 = t[:, 0::2].clone()
        t1 = t[:, 1::2].clone()
        t[:, 0::2] = t0 * c.unsqueeze(0) - t1 * s.unsqueeze(0)
        t[:, 1::2] = t0 * s.unsqueeze(0) + t1 * c.unsqueeze(0)
    kc_ref_k[:, 0, :] = k
    kc_ref_v[:, 0, :] = v
    qg = q.reshape(nkv, nh // nkv, hd)
    sc = torch.einsum('ghd,gtd->ght', qg,
                      kc_ref_k[:, 0:1, :].squeeze(1).unsqueeze(0)
                      .expand(nkv, -1, -1).reshape(nkv, 1, hd)
                      ) if False else torch.einsum(
        'ghd,gtd->ght',
        qg.reshape(nkv, nh // nkv, hd),
        kc_ref_k[:, 0:1, :])
    sc = sc / (hd ** 0.5)
    a = torch.softmax(sc, dim=-1)
    att = torch.einsum('ght,gtd->ghd', a,
                       kc_ref_v[:, 0:1, :]).reshape(nh * hd)
    o = att @ Ws[b + 3].t()
    h_ref = h_ref + o
    xn2 = h_ref / torch.sqrt((h_ref * h_ref).mean() + 1e-5) \
        * post_ws[l].float()
    g = xn2 @ Ws[b + 4].t()
    u = xn2 @ Ws[b + 5].t()
    act = g / (1 + torch.exp(-g)) * u
    d_out = act @ Ws[b + 6].t()
    h_ref = h_ref + d_out
    ref_norms.append(h_ref.norm().item())

# C++ chain
kc_cpp = torch.zeros(2 * nkv, 1, hd, dtype=torch.bfloat16, device='cuda')
hh = h0.clone()
cpp_norms = []
for l in range(24):
    b = l * 7
    hh = ext.layer_forward(
        hh, in_ws[l], post_ws[l],
        pks[b + 0]['codes5'], pks[b + 0]['cb'], pks[b + 0]['s_i8'],
        pks[b + 0]['s_base'], pks[b + 0]['s_step'],
        pks[b + 0]['out_f'], pks[b + 0]['in_f'],
        pks[b + 1]['codes5'], pks[b + 1]['cb'], pks[b + 1]['s_i8'],
        pks[b + 1]['s_base'], pks[b + 1]['s_step'],
        pks[b + 1]['out_f'], pks[b + 1]['in_f'],
        pks[b + 2]['codes5'], pks[b + 2]['cb'], pks[b + 2]['s_i8'],
        pks[b + 2]['s_base'], pks[b + 2]['s_step'],
        pks[b + 2]['out_f'], pks[b + 2]['in_f'],
        pks[b + 3]['codes5'], pks[b + 3]['cb'], pks[b + 3]['s_i8'],
        pks[b + 3]['s_base'], pks[b + 3]['s_step'],
        pks[b + 3]['out_f'], pks[b + 3]['in_f'],
        pks[b + 4]['codes5'], pks[b + 4]['cb'], pks[b + 4]['s_i8'],
        pks[b + 4]['s_base'], pks[b + 4]['s_step'],
        pks[b + 4]['out_f'], pks[b + 4]['in_f'],
        pks[b + 5]['codes5'], pks[b + 5]['cb'], pks[b + 5]['s_i8'],
        pks[b + 5]['s_base'], pks[b + 5]['s_step'],
        pks[b + 5]['out_f'], pks[b + 5]['in_f'],
        pks[b + 6]['codes5'], pks[b + 6]['cb'], pks[b + 6]['s_i8'],
        pks[b + 6]['s_base'], pks[b + 6]['s_step'],
        pks[b + 6]['out_f'], pks[b + 6]['in_f'],
        kc_cpp, pos, nh, nkv, hd, CTX, theta)
    cpp_norms.append(hh.float().norm().item())

print('\nlayer | py_norm | cpp_norm | ratio')
print('-' * 45)
for l in range(24):
    r = cpp_norms[l] / max(ref_norms[l], 1e-8)
    flag = ' <<<' if abs(r - 1) > 0.1 else ''
    print(f'  {l:2d}  | {ref_norms[l]:8.4f} | {cpp_norms[l]:8.4f} '
          f'| {r:.4f}{flag}', flush=True)
