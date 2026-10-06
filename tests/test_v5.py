# -*- coding: utf-8 -*-
"""v5 clean engine per-layer norm dump: C++ vs Python."""
import sys
sys.path.insert(0, r'E:\IXRUN')
import pandas  # noqa: F401
import torch
from torch.utils.cpp_extension import load_inline
from transformers import AutoModelForCausalLM
from ixrun.config import MODEL_PATH
from ixrun.linear import iter_quantizable_linears
from benchmarks.gsq_runtime import gs_pack, gs_decode_ref

m = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, dtype=torch.bfloat16, trust_remote_code=True
).eval().cuda()
cfg = m.config
H, nh = cfg.hidden_size, cfg.num_attention_heads
nkv = cfg.num_key_value_heads
hd = getattr(cfg, 'head_dim', H // nh)
theta = float(getattr(cfg, 'rope_theta', 10000.0))
CTX = 8

sd = dict(m.named_modules())
targets = list(iter_quantizable_linears(m))
names = [n for n, _ in targets]
lay0 = names[0].rsplit('.', 3)[0]
print(f'H={H} nh={nh} nkv={nkv} hd={hd}', flush=True)

pks = []
for name, mod in targets:
    pk = gs_pack(mod.weight.data.cuda())
    for k in ('codes5', 'cb', 's_i8'):
        pk[k] = pk[k].cuda()
    pks.append(pk)
Ws = [gs_decode_ref(pk).float().cuda() for pk in pks]
print('packed', flush=True)

src = open(r'E:\IXRUN\ixrun\cpp\engine_v5.cu',
           encoding='utf-8').read()
proto = '''
torch::Tensor layer_forward(
    torch::Tensor h, torch::Tensor in_nw, torch::Tensor post_nw,
    torch::Tensor qc, torch::Tensor qcb, torch::Tensor qs,
    double qb, double qst, int64_t qo, int64_t qi,
    torch::Tensor kc, torch::Tensor kcb, torch::Tensor ks,
    double kb, double kst, int64_t ko, int64_t ki,
    torch::Tensor vc, torch::Tensor vcb, torch::Tensor vs,
    double vb, double vst, int64_t vo, int64_t vi,
    torch::Tensor oc, torch::Tensor ocb, torch::Tensor os8,
    double ob, double ost, int64_t oo, int64_t oi,
    torch::Tensor gc, torch::Tensor gcb, torch::Tensor gs8,
    double gb, double gst, int64_t go, int64_t gi,
    torch::Tensor uc, torch::Tensor ucb, torch::Tensor us8,
    double ub, double ust, int64_t uo, int64_t ui,
    torch::Tensor dc, torch::Tensor dcb, torch::Tensor ds8,
    double db, double dst, int64_t dof, int64_t dif,
    torch::Tensor kv_cache, int64_t pos,
    int64_t n_heads, int64_t n_kv_heads,
    int64_t head_dim, int64_t ctx, double theta);
'''
ext = load_inline(name='ixrun_cpp_v5', cpp_sources=[proto],
                  cuda_sources=[src], functions=['layer_forward'],
                  extra_cuda_cflags=['-O3', '--use_fast_math',
                                     '-allow-unsupported-compiler'],
                  verbose=False)

in_ws = [sd[lay0 + f'.{l}.input_layernorm'].weight.data.cuda()
         for l in range(24)]
post_ws = [sd[lay0 + f'.{l}.post_attention_layernorm'].weight
           .data.cuda() for l in range(24)]

torch.manual_seed(42)
h0 = (torch.randn(H, device='cuda') * 0.3).to(torch.bfloat16)

# Python reference (single layer 0, pos=0, full-precision weights)
b = 0
h_ref = h0.clone().float()
xn = h_ref / torch.sqrt((h_ref*h_ref).mean() + 1e-5) * in_ws[0].float()
q = (xn @ Ws[b].t()).reshape(nh, hd)
k = (xn @ Ws[b+1].t()).reshape(nkv, hd)
v = (xn @ Ws[b+2].t()).reshape(nkv, hd)
d = torch.arange(0, hd, 2, device='cuda').float()
th = 0.0 / (theta ** (d / hd))  # pos=0 -> no rotation
c_, s_ = torch.cos(th), torch.sin(th)
q[:, 0::2] = q[:, 0::2] * c_.unsqueeze(0) - q[:, 1::2] * s_.unsqueeze(0)
q[:, 1::2] = q[:, 0::2] * s_.unsqueeze(0) + q[:, 1::2] * c_.unsqueeze(0)
# note: this is wrong for pos=0 (identity), but rope at pos=0 is identity
# so let's just skip rope for pos=0
q = (xn @ Ws[b].t()).reshape(nh, hd)
k = (xn @ Ws[b+1].t()).reshape(nkv, hd)
v = (xn @ Ws[b+2].t()).reshape(nkv, hd)
qg = q.reshape(nkv, nh // nkv, hd)
sc = torch.zeros(nkv, nh // nkv, 1, device='cuda')
for kvh in range(nkv):
    for g in range(nh // nkv):
        sc[kvh, g, 0] = (qg[kvh, g] @ k[kvh]) / (hd ** 0.5)
a = torch.softmax(sc, dim=-1)
att = torch.zeros(nh, hd, device='cuda')
for kvh in range(nkv):
    for g in range(nh // nkv):
        att[kvh * (nh//nkv) + g] = a[kvh, g, 0] * v[kvh]
o = att.reshape(-1) @ Ws[b+3].t()
h1 = h_ref + o
xn2 = h1 / torch.sqrt((h1*h1).mean() + 1e-5) * post_ws[0].float()
g = xn2 @ Ws[b+4].t()
u = xn2 @ Ws[b+5].t()
act = g / (1 + torch.exp(-g)) * u
d_out = act @ Ws[b+6].t()
y_ref = (h1 + d_out).to(torch.bfloat16)
print(f'ref layer0 norm = {y_ref.float().norm().item():.4f}', flush=True)

# C++ layer 0
kc = torch.zeros(2*nkv, CTX, hd, dtype=torch.bfloat16,
                 device='cuda')
pk = lambda i: (pks[b+i]['codes5'], pks[b+i]['cb'],
                pks[b+i]['s_i8'], pks[b+i]['s_base'],
                pks[b+i]['s_step'], pks[b+i]['out_f'],
                pks[b+i]['in_f'])
y_cpp = ext.layer_forward(
    h0, in_ws[0], post_ws[0],
    *pk(0), *pk(1), *pk(2), *pk(3), *pk(4), *pk(5), *pk(6),
    kc, 0, nh, nkv, hd, CTX, theta)
print(f'cpp layer0 norm = {y_cpp.float().norm().item():.4f}',
      flush=True)
print(f'cpp nan = {bool(y_cpp.float().isnan().any())}', flush=True)
rel = ((y_cpp.float() - y_ref.float()).norm()
       / y_ref.float().norm()).item()
print(f'layer0 rel_err = {rel:.4f}', flush=True)
print('PASS' if rel < 0.1 and not bool(
    y_cpp.float().isnan().any()) else 'FAIL', flush=True)
