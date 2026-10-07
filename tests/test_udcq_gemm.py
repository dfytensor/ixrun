# -*- coding: utf-8 -*-
"""UDCQ batched GEMM gate: fp64 rel-err + T-token timing."""
import sys, time
sys.path.insert(0, r'E:\IXRUN')
import pandas  # noqa: F401
import torch
from torch.utils.cpp_extension import load_inline

src = open(r'E:\IXRUN\ixrun\cpp\engine_v5.cu',
           encoding='utf-8').read()
proto = '''
torch::Tensor udcq_gemm_out(torch::Tensor x2d, torch::Tensor idx,
    torch::Tensor sign, torch::Tensor scale, torch::Tensor cb,
    int64_t out_f, int64_t in_f, int64_t group);
'''
ext = load_inline(name='ixrun_cpp_v5udcq2', cpp_sources=[proto],
                  cuda_sources=[src],
                  functions=['udcq_gemm_out'],
                  extra_cuda_cflags=['-O3', '--use_fast_math',
                                     '-allow-unsupported-compiler'],
                  verbose=False)

g = torch.Generator(device='cuda').manual_seed(11)
GROUP = 16
out_f, in_f = 1024, 4096
N = out_f * in_f
ng = N // GROUP
cb16 = torch.randn(16, generator=g, device='cuda').half()
idx = torch.randint(0, 255, (N // 2,),
                    dtype=torch.uint8, device='cuda')
sign = torch.randint(0, 2**31 - 1, (N // 32,),
                     dtype=torch.int32, device='cuda')
scale = (torch.randn(ng, generator=g, device='cuda')
         * 0.01).half()

T = 8
X = torch.randn(T, in_f, generator=g, device='cuda').float()
Y = ext.udcq_gemm_out(X, idx, sign, scale.float(),
                      cb16.float(), out_f, in_f, GROUP)

b = idx.long()
lo_n = b & 0x0F
hi_n = (b >> 4) & 0x0F
nib = torch.stack([lo_n, hi_n], 1).reshape(-1)
bit = ((sign.long().unsqueeze(1) >> torch.arange(
    32, device='cuda')) & 1).reshape(-1)[:N]
W = (cb16.float().double()[nib]
     * scale.float().double().repeat_interleave(GROUP)
     * (bit * 2.0 - 1.0)).reshape(out_f, in_f)
ref = (W @ X.double().T).T
e = ((Y.double() - ref).norm() / ref.norm()).item()
print(f'T={T} rel-err vs fp64: {e:.2e}', flush=True)

for T in (1, 16, 64):
    X = torch.randn(T, in_f, generator=g, device='cuda').float()
    for _ in range(5):
        ext.udcq_gemm_out(X, idx, sign, scale.float(),
                          cb16.float(), out_f, in_f, GROUP)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(30):
        ext.udcq_gemm_out(X, idx, sign, scale.float(),
                          cb16.float(), out_f, in_f, GROUP)
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / 30
    print(f'T={T:3d}: {dt*1e6:.0f}us ({dt*1e6/T:.1f}us/tok)',
          flush=True)
