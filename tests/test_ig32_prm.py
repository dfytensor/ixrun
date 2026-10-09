# -*- coding: utf-8 -*-
"""Compare pack levels vs prm-decoded levels (torch reimpl of the kernel)."""
import sys
sys.path.insert(0, r'E:\IXRUN')
import torch
from benchmarks.ig32_runtime import ig32_pack

torch.manual_seed(1)
W = (torch.randn(64, 256) * 0.05).cuda()
pk = ig32_pack(W)
B_REL = torch.logspace(-3, 2, 15).cuda()
BETAS = torch.tensor([0.50, 0.55, 0.65, 0.75, 0.85, 0.95, 1.00]).cuda()

gmx = pk['gmax'].cuda().float()
prm = pk['prm'].cuda().int()
lv_pack = pk['levels'].cuda().float()                   # [r, g, 16]
tanh_f = ((prm & 0x80) != 0).unsqueeze(-1)
b = (B_REL[(prm >> 4) & 0xF] * gmx).unsqueeze(-1)
beta = BETAS[prm & 7].unsqueeze(-1)
gmx = gmx.unsqueeze(-1)
gb = torch.exp2(beta * torch.log2(gmx))
fmax = torch.where(tanh_f, torch.tanh(gmx / b), gb / (b + gb))
fd = (torch.arange(16, device='cuda').view(1, 1, 16) / 15.0) * fmax
fd = fd.clamp(max=1 - 1e-6)
w = torch.where(tanh_f, b * torch.atanh(fd), fd * b / (1 - fd))
w = torch.where(beta != 1.0, w ** (1.0 / beta), w)
w = w.clamp_min(0); w = torch.minimum(w, gmx)
d = (w - lv_pack).abs()
print(f'prm-levels vs pack-levels: max {d.max().item():.6f} '
      f'mean {d.mean().item():.6f}')
bad = (d.amax(-1) > 0.01)
print(f'groups with >0.01 diff: {bad.sum().item()} / {bad.numel()}')
if bad.any():
    gi = bad.nonzero()[0].tolist()
    r, g = gi
    print(f'example r{r} g{g}: prm {prm[r,g].item():02x} '
          f'tanh {tanh_f[r,g].item()} beta {beta[r,g].item()} '
          f'gmax {gmx[r,g].item():.5f} b {b[r,g].item():.5f}')
    print('  pack :', [round(v, 5) for v in lv_pack[r, g].tolist()])
    print('  prmlv:', [round(v, 5) for v in w[r, g].tolist()])
