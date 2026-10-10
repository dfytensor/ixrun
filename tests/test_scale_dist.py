# -*- coding: utf-8 -*-
"""Measure UDCQ per-16 scale log2 distribution across the 27B blob:
can an 8-bit log scale (global base/step) replace fp16 with <2% error?"""
import sys
sys.path.insert(0, r'E:\IXRUN')
import pandas  # noqa
import torch

blob = torch.load(r'E:\IXRUN\experiments\qwen38_udcq\q38_blob.pt',
                  map_location='cpu', mmap=True, weights_only=True)
keys = list(blob['layers'].keys())
lo_g, hi_g = 1e9, -1e9
for key in keys:
    s = blob['layers'][key]['scale'].float().flatten()
    lg = torch.log2(s.clamp_min(1e-30))
    lo_g = min(lo_g, lg.min().item())
    hi_g = max(hi_g, lg.max().item())
print(f'global log2(scale) range: [{lo_g:.2f}, {hi_g:.2f}] '
      f'span {hi_g-lo_g:.2f} octaves', flush=True)

# per-matrix spans
spans = []
for key in keys[:80]:
    s = blob['layers'][key]['scale'].float().flatten()
    lg = torch.log2(s.clamp_min(1e-30))
    spans.append((lg.max() - lg.min()).item())
spans = torch.tensor(spans)
print(f'per-matrix span: mean {spans.mean():.2f} p90 '
      f'{spans.quantile(0.9):.2f} max {spans.max():.2f} octaves', flush=True)

B = 10 ** ((hi_g - lo_g) / 255)   # error factor per step
print(f'global 8-bit log step: {B:.4f}x -> max scale quant err '
      f'{(B-1)/2*100:.2f}% (global)', flush=True)
Bm = 10 ** (spans.mean().item() / 255)
print(f'per-matrix 8-bit step: {Bm:.4f}x -> max err '
      f'{(Bm-1)/2*100:.2f}% (per-matrix)', flush=True)
