# -*- coding: utf-8 -*-
"""Stage 4 step 1: blob layer-0 loader + real-pack UDCQ GEMV gate.
Gate: C++ UDCQ GEMV (REAL q38 blob pack) vs python decode+matmul
at fp64 -> expect 1e-7 tier (kernel math), on real 27B weights."""
import sys, time
sys.path.insert(0, r'E:\IXRUN')
import pandas  # noqa: F401
import torch
from torch.utils.cpp_extension import load_inline

BLOB = r'E:\IXRUN\experiments\qwen38_udcq\q38_blob.pt'

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

t0 = time.perf_counter()
blob = torch.load(BLOB, map_location='cpu', mmap=True,
                  weights_only=True)
print(f'blob mmap-loaded in {time.perf_counter()-t0:.1f}s',
      flush=True)
cb = blob['codebook']          # f16 [16]
l0_keys = sorted(k for k in blob['layers'].keys()
                 if '.layers.0.' in k)
print(f'codebook: {tuple(cb.shape)} {cb.dtype}', flush=True)
print(f'layer-0 tensors ({len(l0_keys)}):', flush=True)
for k in l0_keys:
    p = blob['layers'][k]
    print(f'  {k.split(".", 3)[-1]:24s} idx{tuple(p["idx"].shape)} '
          f'{p["idx"].dtype}', flush=True)

# real-pack gate on in_proj_qkv
name = [k for k in l0_keys if 'in_proj_qkv' in k][0]
pk = blob['layers'][name]
idx = pk['idx'].cuda()
scale = pk['scale'].float().cuda()      # f16 -> f32 at init
sign = pk['sign'].cuda()
GROUP = 16
N = idx.numel() * 2
in_f, out_f = 5120, N // 5120
print(f'in_proj_qkv: {out_f}x{in_f} N={N}', flush=True)

g = torch.Generator(device='cuda').manual_seed(31)
x = torch.randn(in_f, generator=g, device='cuda').float()
t0 = time.perf_counter()
y = ext.udcq_gemv_out(x, idx, sign, scale, cb.float().cuda(),
                      out_f, in_f, GROUP)
torch.cuda.synchronize()
t_k = time.perf_counter() - t0

# fp64 ref: decode REAL pack
idx_c = idx.cpu().long()
b = idx_c
lo_n, hi_n = b & 0x0F, (b >> 4) & 0x0F
nib = torch.stack([lo_n, hi_n], 1).reshape(-1)
bit = ((sign.cpu().long().unsqueeze(1) >> torch.arange(
        32)) & 1).reshape(-1)[:N]
W = (cb.float().double()[nib]
     * scale.cpu().double().repeat_interleave(GROUP)
     * (bit * 2.0 - 1.0)).reshape(out_f, in_f)
ref = (W @ x.cpu().double())
e = ((y.cpu().double() - ref).norm() / ref.norm()).item()
print(f'REAL pack rel-err vs fp64: {e:.2e} '
      f'(kernel {t_k*1e3:.1f}ms)', flush=True)
assert e < 1e-4, 'REAL PACK GATE FAIL'
print('REAL-PACK GATE PASSED', flush=True)
