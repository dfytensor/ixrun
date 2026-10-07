# -*- coding: utf-8 -*-
"""Split-K GEMV gate: fp64-reference rel-err + S sweep timing."""
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
torch::Tensor gemv_sk_out(torch::Tensor x, torch::Tensor codes,
    torch::Tensor cb, torch::Tensor s_i8,
    double s_base, double s_step,
    int64_t out_f, int64_t in_f, int64_t S);
'''
ext = load_inline(name='ixrun_cpp_v5sk1', cpp_sources=[proto],
                  cuda_sources=[src],
                  functions=['gemv_v2_out', 'gemv_sk_out'],
                  extra_cuda_cflags=['-O3', '--use_fast_math',
                                     '-allow-unsupported-compiler'],
                  verbose=False)

g = torch.Generator(device='cuda').manual_seed(5)
out_f, in_f = 1024, 4096
codes = torch.randint(0, 255, (out_f * in_f // 16 * 10,),
                      dtype=torch.uint8, device='cuda')
cb = torch.randn(32, generator=g, device='cuda').float()
s = torch.randint(0, 255, (out_f * in_f // 16,),
                  dtype=torch.uint8, device='cuda')
x = torch.randn(in_f, generator=g, device='cuda').float()

# fp64 reference: decode weights then double-precision GEMV (order-free)
nG = out_f * in_f // 16
c5 = codes.view(torch.int64)  # not real decode — use cb gather path:
# decode via the documented path (10B/16elem, 5-bit codes)
b = codes.view(nG, 10).long()
d0 = b[:, 0] | (b[:, 1] << 8) | (b[:, 2] << 16) | (b[:, 3] << 24)
d1 = b[:, 4] | (b[:, 5] << 8) | (b[:, 6] << 16) | (b[:, 7] << 24)
d2 = b[:, 8] | (b[:, 9] << 8)
lo = d0 | (d1 << 32)
codes5 = []
for i in range(12):
    codes5.append(((lo >> (5 * i)) & 0x1F))
codes5.append((((lo >> 60) | (d2 << 4)) & 0x1F))
for i in range(3):
    codes5.append(((d2 >> (5 * i + 1)) & 0x1F))
C = torch.stack(codes5, 1)                       # [nG, 16]
sc = torch.pow(2.0, -3.0 + s.double() * 0.05)    # [nG]
W = (cb.double()[C.long()] * sc.unsqueeze(1)).reshape(out_f, in_f)
ref = (W @ x.double())                           # fp64, order-free

y2 = ext.gemv_v2_out(x, codes, cb, s, -3.0, 0.05, out_f, in_f)
e2 = ((y2.double() - ref).norm() / ref.norm()).item()
print(f'v2  rel-err vs fp64: {e2:.2e}', flush=True)
for S in (2, 4, 8):
    ys = ext.gemv_sk_out(x, codes, cb, s, -3.0, 0.05,
                         out_f, in_f, S)
    esk = ((ys.double() - ref).norm() / ref.norm()).item()
    d2v = (ys - y2).abs().max().item()
    print(f'sk S={S}: rel-err {esk:.2e} | maxdiff vs v2 {d2v:.2e}',
          flush=True)

# timing sweep on layer + lm_head shapes
for of, inf in [(1024, 4096), (4096, 1024), (151936, 1536)]:
    cd = torch.randint(0, 255, (of * inf // 16 * 10,),
                       dtype=torch.uint8, device='cuda')
    ss = torch.randint(0, 255, (of * inf // 16,),
                       dtype=torch.uint8, device='cuda')
    xx = torch.randn(inf, generator=g, device='cuda').float()
    res = {}
    for name, fn in [('v2', lambda: ext.gemv_v2_out(
                        xx, cd, cb, ss, -3.0, 0.05, of, inf))] + [
                    (f'S{S}', lambda S=S: ext.gemv_sk_out(
                        xx, cd, cb, ss, -3.0, 0.05, of, inf, S))
                    for S in (2, 4, 8)]:
        for _ in range(10):
            fn()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(50):
            fn()
        torch.cuda.synchronize()
        res[name] = (time.perf_counter() - t0) / 50 * 1e6
    best = min(res, key=res.get)
    print(f'{of}x{inf}: ' +
          ' '.join(f'{k}={v:.0f}us' for k, v in res.items()) +
          f' -> best {best}', flush=True)
