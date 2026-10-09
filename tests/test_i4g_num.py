# -*- coding: utf-8 -*-
"""Numeric debug of quant_g32_warm internals."""
import sys
sys.path.insert(0, r'E:\IXRUN')
import torch
from benchmarks.int4g32_bare import quant_g32_warm

torch.manual_seed(0)
W = (torch.randn(256, 512) * 0.02).cuda()
Wq = quant_g32_warm(W)
rel = ((W - Wq).norm() / W.norm()).item()
print(f'rel-L2 {rel:.4f}')

# vanilla per-group-32 uniform 8-level RTN reference
G = W.float().view(-1, 32)
A = G.abs()
gmax = A.amax(1, keepdim=True)
q = torch.round(A / gmax * 7)
rtn = (q / 7 * gmax) * torch.sign(G)
rel_rtn = ((G - rtn).norm() / G.norm()).item()
print(f'vanilla int4-g32 RTN rel-L2 {rel_rtn:.4f}')

# internals of the search (recompute best-only, no where-tracking)
B_REL = torch.logspace(-3, 2, 15).tolist()
BETAS = [0.50, 0.55, 0.65, 0.75, 0.85, 0.95, 1.00]
best = torch.full((A.shape[0],), 1e9, device=A.device)
gmax = A.amax(1, keepdim=True).clamp_min(1e-12)
for br in B_REL:
    b = br * gmax
    for beta in BETAS:
        f = A * (b + gmax ** beta) / (gmax * (b + A ** beta))
        q = torch.round(f * 7).clamp_(0, 7)
        fd = (q / 7).clamp_(max=1 - 1e-6)
        wr = (fd * b / (1 - fd)).clamp_min(0)
        if beta != 1.0:
            wr = torch.minimum(wr, gmax) ** (1.0 / beta)
            wr = torch.minimum(wr, gmax)
        else:
            wr = torch.minimum(wr, gmax)
        mse = ((A - wr) ** 2).mean(1)
        best = torch.minimum(best, mse)
print(f'search-internal per-group MSE: mean {best.mean():.6f} '
      f'max {best.max():.6f}')
print(f'implied rel-L2 from search MSE: '
      f'{(best.mean() ** 0.5) / A.norm().item() * (A.numel() ** 0.5):.4f}')
