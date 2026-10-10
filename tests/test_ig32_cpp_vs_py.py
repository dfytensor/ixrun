# -*- coding: utf-8 -*-
"""C++ ig32p vs PY ig32p (verified) on the same pack."""
import sys
sys.path.insert(0, r'E:\IXRUN')
import pandas  # noqa: F401
import torch
from torch.utils.cpp_extension import load_inline

BLOB = r'F:\models\qwen38_ig32_blob.pt'
src = open(r'E:\IXRUN\ixrun\cpp\engine_v5.cu', encoding='utf-8').read()
src27 = open(r'E:\IXRUN\ixrun\cpp\engine_27b.cu', encoding='utf-8').read()
proto = '''
torch::Tensor ig32p_out(torch::Tensor x, torch::Tensor codes,
    torch::Tensor sign, torch::Tensor aux,
    int64_t out_f, int64_t in_f);
'''
ext = load_inline(name='ixrun_cpp_i32g', cpp_sources=[proto],
                  cuda_sources=[src, src27],
                  functions=['ig32p_out'],
                  extra_cuda_cflags=['-O3', '--use_fast_math',
                                     '-allow-unsupported-compiler'],
                  verbose=False)
from experiments.ig32_gemv_cuda import ig32p_gemv, install_tables, _load
_load()
install_tables()

blob = torch.load(BLOB, map_location='cpu', mmap=True, weights_only=True)
p = blob['layers']['model.layers.0.mlp.gate_proj']
of, inf = p['out_f'], p['in_f']
nc = inf // 32
aux = p['aux'].view(of, nc, 3)
pk = {'codes': p['codes'], 'sign': p['sign'],
      'gmax': aux[:, :, 1:3].contiguous().view(torch.float16)
                        .view(of, nc),
      'prm': aux[:, :, 0].contiguous(),
      'out_f': of, 'in_f': inf}
x_bf = torch.randn(inf, dtype=torch.bfloat16, device='cuda')
x_f = x_bf.float()

y_cpp = ext.ig32p_out(x_f.contiguous(), p['codes'].cuda(),
                      p['sign'].cuda(), p['aux'].cuda(),
                      of, inf).float()
y_py = ig32p_gemv(x_bf, pk).float().view(-1)
d = (y_cpp - y_py).abs()
print(f'cpp vs py: max {d.max().item():.4f} mean {d.mean().item():.5f}')
print('py first5:', [round(v, 4) for v in y_py[:5].tolist()])
print('cc first5:', [round(v, 4) for v in y_cpp[:5].tolist()])
bad = d > 0.05
print(f'bad rows: {bad.sum().item()} / {of}')
if bad.any():
    print('bad idx:', bad.nonzero().flatten()[:10].tolist())
