# -*- coding: utf-8 -*-
"""Side-by-side: engine.cu gsq_gemv vs experiments/gsq_gemv_cuda (bit-verified)."""
import sys

sys.path.insert(0, r'E:\IXRUN')
import pandas  # noqa: F401
import torch
from torch.utils.cpp_extension import load_inline
from experiments.gsq_gemv_cuda.gsq_gemv_cuda import _load as _l0
from benchmarks.gsq_runtime import gs_pack

src = open(r'E:\IXRUN\ixrun\cpp\engine.cu', encoding='utf-8').read()
proto = '''
torch::Tensor gsq_gemv_out(torch::Tensor x,
                           torch::Tensor codes, torch::Tensor cb,
                           torch::Tensor s8, double s_base, double s_step,
                           int64_t out_f, int64_t in_f);
'''
ext = load_inline(name='ixrun_cpp_v3', cpp_sources=[proto],
                  cuda_sources=[src], functions=['gsq_gemv_out'],
                  extra_cuda_cflags=['-O3', '--use_fast_math',
                                     '-allow-unsupported-compiler'],
                  verbose=False)
orig = _l0()

torch.manual_seed(0)
for of, inf in [(512, 512), (1536, 4608)]:
    W = (torch.randn(of, inf) * 0.02).to(torch.bfloat16).cuda()
    pk = gs_pack(W)
    for k in ('codes5', 'cb', 's_i8'):
        pk[k] = pk[k].cuda()
    x = torch.randn(inf, dtype=torch.bfloat16, device='cuda')
    yo = orig.gemv(x.contiguous(), pk['codes5'], pk['cb'], pk['s_i8'],
                   pk['s_base'], pk['s_step'], of, inf)
    ye = ext.gsq_gemv_out(x, pk['codes5'], pk['cb'], pk['s_i8'],
                          pk['s_base'], pk['s_step'], of, inf)
    d = (yo.float() - ye.float()).abs()
    bad = d > 1e-2
    print(f'[{of}x{inf}] bad rows: {int(bad.sum())}/{of} '
          f'gmax={d.max().item():.4f}', flush=True)
    if bad.any():
        r = bad.nonzero().flatten()[:5].tolist()
        print('  bad row idx:', r)
        print('  y_orig[:6]:', [round(v, 3) for v in yo[:6].tolist()])
        print('  y_eng [:6]:', [round(v, 3) for v in ye[:6].tolist()])
