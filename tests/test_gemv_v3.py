# -*- coding: utf-8 -*-
"""GEMV v3 (pipelined) gate: bit-exact vs v2 + timing."""
import sys, time
sys.path.insert(0, r'E:\IXRUN')
import pandas  # noqa: F401
import torch
from torch.utils.cpp_extension import load_inline

src = open(r'E:\IXRUN\ixrun\cpp\engine_v5.cu',
           encoding='utf-8').read()
proto = '''
torch::Tensor gemv_v2_out(torch::Tensor x, torch::Tensor codes,
    torch::Tensor cb, torch::Tensor s_i8,
    double s_base, double s_step,
    int64_t out_f, int64_t in_f);
torch::Tensor gemv_v3_out(torch::Tensor x, torch::Tensor codes,
    torch::Tensor cb, torch::Tensor s_i8,
    double s_base, double s_step,
    int64_t out_f, int64_t in_f);
'''
ext = load_inline(name='ixrun_cpp_v5gemv3', cpp_sources=[proto],
                  cuda_sources=[src],
                  functions=['gemv_v2_out', 'gemv_v3_out'],
                  extra_cuda_cflags=['-O3', '--use_fast_math',
                                     '-allow-unsupported-compiler'],
                  verbose=False)

g = torch.Generator(device='cuda').manual_seed(3)
for out_f, in_f in [(1024, 4096), (4096, 1024), (1536, 1536),
                    (151936, 1536)]:
    codes = torch.randint(0, 255, (out_f * in_f // 16 * 10,),
                          dtype=torch.uint8, device='cuda')
    cb = torch.randn(32, generator=g, device='cuda').float()
    s = torch.randint(0, 255, (out_f * in_f // 16,),
                      dtype=torch.uint8, device='cuda')
    x = torch.randn(in_f, generator=g, device='cuda').float()
    y2 = ext.gemv_v2_out(x, codes, cb, s, -3.0, 0.05, out_f, in_f)
    y3 = ext.gemv_v3_out(x, codes, cb, s, -3.0, 0.05, out_f, in_f)
    be = torch.equal(y2.view(torch.int32), y3.view(torch.int32))
    for f in (ext.gemv_v2_out, ext.gemv_v3_out):
        for _ in range(10):
            f(x, codes, cb, s, -3.0, 0.05, out_f, in_f)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(50):
        ext.gemv_v2_out(x, codes, cb, s, -3.0, 0.05, out_f, in_f)
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    for _ in range(50):
        ext.gemv_v3_out(x, codes, cb, s, -3.0, 0.05, out_f, in_f)
    torch.cuda.synchronize()
    t2 = time.perf_counter()
    v2 = (t1 - t0) / 50 * 1e6
    v3 = (t2 - t1) / 50 * 1e6
    print(f'{out_f}x{in_f}: bit-exact={be} | '
          f'v2={v2:.0f}us v3={v3:.0f}us ({v2/v3:.2f}x)', flush=True)
