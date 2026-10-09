# -*- coding: utf-8 -*-
"""rmsnorm_fw on the REAL h1_c dumped from the scheduler — standalone
build with fresh ext name to eliminate any binary/cache confusion."""
import sys
sys.path.insert(0, r'E:\IXRUN')
import pandas
import torch
from torch.utils.cpp_extension import load_inline

src = open(r'E:\IXRUN\ixrun\cpp\engine_v5.cu', encoding='utf-8').read()
proto = 'torch::Tensor rmsnorm_fw_out(torch::Tensor x2d, torch::Tensor w, double eps);'
ext = load_inline(name='ixrun_rnfw_real2', cpp_sources=[proto],
                  cuda_sources=[src], functions=['rmsnorm_fw_out'],
                  extra_cuda_cflags=['-O3', '--use_fast_math',
                                     '-allow-unsupported-compiler'],
                  verbose=False)

d = torch.load(r'C:\Users\Administrator\AppData\Local\Temp\opencode\h1_dump.pt')
h1 = d['h1'].cuda().float()
w = d['w'].cuda().float()

y = ext.rmsnorm_fw_out(h1.view(1, -1), w, 1e-6).view(-1)
y_ref = w * h1 * torch.rsqrt(h1.pow(2).mean() + 1e-6)
e = ((y - y_ref).norm() / y_ref.norm().clamp_min(1e-9)).item()

print(f'h1 norm: {h1.norm().item():.4f} | w norm: {w.norm().item():.4f}', flush=True)
print(f'kernel out norm: {y.norm().item():.4f} | torch out norm: {y_ref.norm().item():.4f}', flush=True)
print(f'rel-err: {e:.2e}', flush=True)

topv, topi = torch.topk(y.abs(), 3)
topv2, topi2 = torch.topk(y_ref.abs(), 3)
print(f'kernel top3: {[(round(v.item(), 2), int(i)) for v, i in zip(topv, topi)]}', flush=True)
print(f'torch  top3: {[(round(v.item(), 2), int(i)) for v, i in zip(topv2, topi2)]}', flush=True)

if e > 0.01:
    # element-wise diff to find pattern
    diff = (y - y_ref).abs()
    topd, topdi = torch.topk(diff, 5)
    print(f'top5 diffs: {[(round(v.item(), 2), int(i)) for v, i in zip(topd, topdi)]}', flush=True)
    # check: is the kernel output = input × w (no normalization)?
    y_no_norm = w * h1
    e_nn = ((y - y_no_norm).norm() / y_no_norm.norm().clamp_min(1e-9)).item()
    print(f'  kernel vs NO-NORM (w*x): rel {e_nn:.2e}', flush=True)
    # check: is the kernel output = input only (no w, no norm)?
    e_x = ((y - h1).norm() / h1.norm().clamp_min(1e-9)).item()
    print(f'  kernel vs x only: rel {e_x:.2e}', flush=True)
