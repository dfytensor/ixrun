# -*- coding: utf-8 -*-
"""GEMM v1 gate: per-token bit-exact vs GEMV v2 + T-scaling timing."""
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
torch::Tensor gemm_out(torch::Tensor x2d, torch::Tensor codes,
    torch::Tensor cb, torch::Tensor s_i8,
    double s_base, double s_step,
    int64_t out_f, int64_t in_f);
'''
ext = load_inline(name='ixrun_cpp_v5gemm1', cpp_sources=[proto],
                  cuda_sources=[src],
                  functions=['gemv_v2_out', 'gemm_out'],
                  extra_cuda_cflags=['-O3', '--use_fast_math',
                                     '-allow-unsupported-compiler'],
                  verbose=False)

g = torch.Generator(device='cuda').manual_seed(7)
out_f, in_f = 4096, 1024
codes = torch.randint(0, 255, (out_f * in_f // 16 * 10,),
                      dtype=torch.uint8, device='cuda')
cb = torch.randn(32, generator=g, device='cuda').float()
s = torch.randint(0, 255, (out_f * in_f // 16,),
                  dtype=torch.uint8, device='cuda')

T = 4
X = torch.randn(T, in_f, generator=g, device='cuda').float()
Yg = ext.gemm_out(X, codes, cb, s, -3.0, 0.05, out_f, in_f)
ok = True
for t in range(T):
    yv = ext.gemv_v2_out(X[t].contiguous(), codes, cb, s,
                         -3.0, 0.05, out_f, in_f)
    be = torch.equal(Yg[t].view(torch.int32),
                     yv.view(torch.int32))
    ok &= be
print(f'per-token bit-exact (T={T}): {ok}', flush=True)

# T-scaling timing (weights re-read amortized?)
for T in (1, 16, 64, 256):
    X = torch.randn(T, in_f, generator=g, device='cuda').float()
    for _ in range(3):
        ext.gemm_out(X, codes, cb, s, -3.0, 0.05, out_f, in_f)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(20):
        ext.gemm_out(X, codes, cb, s, -3.0, 0.05, out_f, in_f)
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / 20
    print(f'T={T:4d}: {dt*1e3:.2f}ms '
          f'({T/dt:.0f} tok/s, {dt*1e6/T:.1f}us/tok)', flush=True)
