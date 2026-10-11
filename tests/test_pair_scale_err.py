# -*- coding: utf-8 -*-
"""Measure the per-32 shared-scale error cost on real 27B ULS scales."""
import sys
sys.path.insert(0, r'E:\IXRUN')
import pandas  # noqa
import torch
import math

blob = torch.load(r'F:\models\qwen38_uls_blob.pt', map_location='cpu',
                  mmap=True, weights_only=True)
keys = list(blob['layers'].keys())
worst = 0.0
worst_k = None
mean_err = 0.0
n = 0
ratios = []
for k in keys:
    s = blob['layers'][k]['scale']
    if s.dtype != torch.uint8:
        continue
    of = None
    # shape [out, 8+n_gr]
    buf = s.float() if False else s
    hdr = buf[:, :8].contiguous().view(torch.float32).view(buf.shape[0], 2)
    ib = buf[:, 8:].float()
    sd = torch.exp2(hdr[:, 0:1] + hdr[:, 1:2] * ib)     # [out, n_gr]
    if sd.shape[1] < 2:
        continue
    if sd.shape[1] % 2:
        sd = sd[:, :-1]
    a = sd[:, 0::2]
    b = sd[:, 1::2]
    shared = torch.sqrt(a * b)
    err = torch.maximum((shared - a).abs() / a, (shared - b).abs() / b)
    ratios.append(torch.maximum(a / b, b / a).flatten())
    m = err.max().item()
    mean_err += err.mean().item()
    n += 1
    if m > worst:
        worst, worst_k = m, k
r = torch.cat(ratios)
print(f'matrices: {n}, mean per-group err {mean_err/n:.5f}', flush=True)
print(f'worst group err {worst:.5f} ({worst_k})', flush=True)
print(f'adjacent ratio: mean {r.mean():.4f} p99 {r.quantile(0.99):.4f} '
      f'max {r.max():.4f}', flush=True)
