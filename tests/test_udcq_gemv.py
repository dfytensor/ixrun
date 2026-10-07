# -*- coding: utf-8 -*-
"""UDCQ C++ GEMV gate: rel-err vs fp64 decode reference + timing."""
import sys, time
sys.path.insert(0, r'E:\IXRUN')
import pandas  # noqa: F401
import torch
from torch.utils.cpp_extension import load_inline

src = open(r'E:\IXRUN\ixrun\cpp\engine_v5.cu',
           encoding='utf-8').read()
proto = '''
torch::Tensor udcq_gemv_out(torch::Tensor x, torch::Tensor idx,
    torch::Tensor sign, torch::Tensor scale, torch::Tensor cb,
    int64_t out_f, int64_t in_f, int64_t group);
'''
ext = load_inline(name='ixrun_cpp_v5udcq1', cpp_sources=[proto],
                  cuda_sources=[src],
                  functions=['udcq_gemv_out'],
                  extra_cuda_cflags=['-O3', '--use_fast_math',
                                     '-allow-unsupported-compiler'],
                  verbose=False)

g = torch.Generator(device='cuda').manual_seed(11)
GROUP = 16
for out_f, in_f in [(1024, 4096), (4096, 1024), (6144, 5120)]:
    N = out_f * in_f
    ng = N // GROUP
    cb16 = torch.randn(16, generator=g, device='cuda').half()
    idx = torch.randint(0, 255, (N // 2,),
                        dtype=torch.uint8, device='cuda')
    sign = torch.randint(0, 2**31 - 1, (N // 32,),
                         dtype=torch.int32, device='cuda')
    scale = (torch.randn(ng, generator=g, device='cuda')
             * 0.01).half()
    x = torch.randn(in_f, generator=g, device='cuda').float()

    y = ext.udcq_gemv_out(x, idx, sign, scale.float(),
                          cb16.float(), out_f, in_f, GROUP)

    # fp64 reference: decode W from nibbles
    b = idx.long()
    lo_n = b & 0x0F                 # even-offset nibbles
    hi_n = (b >> 4) & 0x0F          # odd-offset nibbles
    nib = torch.stack([lo_n, hi_n], 1).reshape(-1)   # [N]
    bit = ((sign.long().unsqueeze(1) >> torch.arange(
        32, device='cuda')) & 1).reshape(-1)[:N]     # [N]
    W = (cb16.float().double()[nib]
         * scale.float().double().repeat_interleave(GROUP)
         * (bit * 2.0 - 1.0)).reshape(out_f, in_f)
    ref = (W @ x.double())

    e = ((y.double() - ref).norm() / ref.norm()).item()
    print(f'{out_f}x{in_f}: rel-err vs fp64 = {e:.2e}', flush=True)

# timing
out_f, in_f = 6144, 5120   # ~27B o_proj/mlp-ish shape
N = out_f * in_f
cb16 = torch.randn(16, generator=g, device='cuda').half()
idx = torch.randint(0, 255, (N // 2,),
                    dtype=torch.uint8, device='cuda')
sign = torch.randint(0, 2**31 - 1, (N // 32,),
                     dtype=torch.int32, device='cuda')
scale = (torch.randn(N // GROUP, generator=g,
                     device='cuda') * 0.01).half()
x = torch.randn(in_f, generator=g, device='cuda').float()
for _ in range(5):
    ext.udcq_gemv_out(x, idx, sign, scale.float(), cb16.float(),
                      out_f, in_f, GROUP)
torch.cuda.synchronize()
t0 = time.perf_counter()
for _ in range(30):
    ext.udcq_gemv_out(x, idx, sign, scale.float(), cb16.float(),
                      out_f, in_f, GROUP)
torch.cuda.synchronize()
dt = (time.perf_counter() - t0) / 30
mb = N * 0.5 + N / 8 + N / 16 / 1e6 * 2
print(f'{out_f}x{in_f}: {dt*1e6:.0f}us '
      f'({N*0.5/1e6/dt/1e3:.0f} GB/s codes traffic)', flush=True)
