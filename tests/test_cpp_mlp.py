# -*- coding: utf-8 -*-
"""C1 gate: C++ mlp_forward vs Python GSQ reference, layer-0 MLP."""
import sys

sys.path.insert(0, r'E:\IXRUN')
import pandas  # noqa: F401
import torch
from torch.utils.cpp_extension import load_inline
from transformers import AutoModelForCausalLM

from ixrun.config import MODEL_PATH
from benchmarks.gsq_runtime import gs_pack

m = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, dtype=torch.bfloat16,
    trust_remote_code=True).eval().cuda()
sd = dict(m.named_modules())
gate = sd['model.layers.0.mlp.gate_proj']
up = sd['model.layers.0.mlp.up_proj']
down = sd['model.layers.0.mlp.down_proj']
norm = sd['model.layers.0.post_attention_layernorm']

pk_g = gs_pack(gate.weight.data.cuda())
pk_u = gs_pack(up.weight.data.cuda())
pk_d = gs_pack(down.weight.data.cuda())

src = open(r'E:\IXRUN\ixrun\cpp\engine.cu', encoding='utf-8').read()
proto = '''
torch::Tensor mlp_forward(torch::Tensor x, torch::Tensor norm_w,
                 torch::Tensor gc, torch::Tensor gcb, torch::Tensor gs,
                 double gb, double gst, int64_t go, int64_t gi,
                 torch::Tensor uc, torch::Tensor ucb, torch::Tensor us,
                 double ub, double ust, int64_t uo, int64_t ui,
                 torch::Tensor dc, torch::Tensor dcb, torch::Tensor ds,
                 double dbase, double dstep, int64_t dof, int64_t dif);
torch::Tensor rmsnorm_out(torch::Tensor x, torch::Tensor w);
torch::Tensor gsq_gemv_out(torch::Tensor x,
                           torch::Tensor codes, torch::Tensor cb,
                           torch::Tensor s8, double s_base, double s_step,
                           int64_t out_f, int64_t in_f);
'''
ext = load_inline(name='ixrun_cpp_v2', cpp_sources=[proto],
                  cuda_sources=[src], functions=['mlp_forward', 'rmsnorm_out', 'gsq_gemv_out'],
                  extra_cuda_cflags=['-O3', '--use_fast_math',
                                     '-allow-unsupported-compiler'],
                  verbose=False)

torch.manual_seed(0)
x = (torch.randn(1536, device='cuda') * 0.5).to(torch.bfloat16)


from benchmarks.gsq_runtime import gs_decode_ref
Wg, Wu = gs_decode_ref(pk_g).cuda(), gs_decode_ref(pk_u).cuda()
Wd = gs_decode_ref(pk_d).cuda()
nw = norm.weight.data.cuda()
xr = x.float()
h = xr / torch.sqrt((xr * xr).mean() + 1e-5) * nw.float()
g = (h @ Wg.float().t())
u = (h @ Wu.float().t())
act = g / (1 + torch.exp(-g)) * u
y_ref = act @ Wd.float().t()

hn0 = ext.rmsnorm_out(x, nw)
h_ref = xr / torch.sqrt((xr * xr).mean() + 1e-5) * nw.float()
r0 = ((hn0.float() - h_ref.float()).norm()
      / h_ref.float().norm()).item()
print(f'PRE  rmsnorm rel={r0:.4f}', flush=True)
yg0 = ext.gsq_gemv_out(hn0, pk_g['codes5'].cuda(), pk_g['cb'].cuda(),
                       pk_g['s_i8'].cuda(), pk_g['s_base'],
                       pk_g['s_step'], pk_g['out_f'], pk_g['in_f'])
g_ref = (h_ref @ Wg.float().t())
r0b = ((yg0.float() - g_ref.float()).norm()
       / g_ref.float().norm()).item()
print(f'PRE  gate_gemv rel={r0b:.4f}', flush=True)

y = ext.mlp_forward(x, nw,
                    pk_g['codes5'].cuda(), pk_g['cb'].cuda(),
                    pk_g['s_i8'].cuda(), pk_g['s_base'], pk_g['s_step'],
                    pk_g['out_f'], pk_g['in_f'],
                    pk_u['codes5'].cuda(), pk_u['cb'].cuda(),
                    pk_u['s_i8'].cuda(), pk_u['s_base'], pk_u['s_step'],
                    pk_u['out_f'], pk_u['in_f'],
                    pk_d['codes5'].cuda(), pk_d['cb'].cuda(),
                    pk_d['s_i8'].cuda(), pk_d['s_base'], pk_d['s_step'],
                    pk_d['out_f'], pk_d['in_f'])
rel = ((y.float() - y_ref.float()).norm()
       / y_ref.float().norm()).item()
print(f'C1 mlp rel_err = {rel:.4f}', flush=True)
print('PASS' if rel < 0.05 else 'FAIL', flush=True)
