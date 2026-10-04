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
'''
ext = load_inline(name='ixrun_cpp_v1', cpp_sources=[proto],
                  cuda_sources=[src], functions=['mlp_forward'],
                  extra_cuda_cflags=['-O3', '--use_fast_math',
                                     '-allow-unsupported-compiler'],
                  verbose=False)

torch.manual_seed(0)
x = (torch.randn(1536, device='cuda') * 0.5).to(torch.bfloat16)


def dref(pk):
    cb = pk['cb'].cuda()
    nG = pk['s_i8'].numel()
    lo = pk['codes5'][:, 0:8].long().cuda()
    lo64 = (lo * (1 << (8 * torch.arange(8, device=lo.device)))).sum(1)
    hi = pk['codes5'][:, 8:10].long().cuda()
    hi16 = (hi * (1 << (8 * torch.arange(2, device=hi.device)))).sum(1)
    codes = torch.empty(nG, 16, dtype=torch.long, device='cuda')
    a12 = torch.arange(12, device='cuda')
    codes[:, :12] = ((lo64[:, None] >> (5 * a12)) & 0x1F)
    codes[:, 12] = ((lo64 >> 60) | (hi16 << 4)) & 0x1F
    codes[:, 13] = (hi16 >> 1) & 0x1F
    codes[:, 14] = (hi16 >> 6) & 0x1F
    codes[:, 15] = (hi16 >> 11) & 0x1F
    s = torch.pow(2.0, pk['s_base'] + pk['s_i8'].float().cuda()
                  * pk['s_step'])
    rec = (cb[codes.reshape(-1)]
           * s.reshape(-1, 1).expand(-1, 16).reshape(-1))
    return rec.reshape(pk['out_f'], pk['in_f'])


Wg, Wu = dref(pk_g), dref(pk_u)
Wd = dref(pk_d)
nw = norm.weight.data.cuda()
xr = x.float()
h = xr / torch.sqrt((xr * xr).mean() + 1e-5) * nw.float()
g = (h @ Wg.float().t())
u = (h @ Wu.float().t())
act = g / (1 + torch.exp(-g)) * u
y_ref = act @ Wd.float().t()

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
