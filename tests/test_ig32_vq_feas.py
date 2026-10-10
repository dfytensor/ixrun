# -*- coding: utf-8 -*-
"""Feasibility: can a small shared codebook (VQ) represent the level
vectors well enough? Sample real 27B groups -> shapes lv/gmax -> kmeans."""
import sys
sys.path.insert(0, r'E:\IXRUN')
import pandas  # noqa: F401
import torch

blob = torch.load(r'F:\models\qwen38_ig32_blob.pt', map_location='cpu',
                  mmap=True, weights_only=True)
B_REL = torch.logspace(-3, 2, 15).cuda()
BETAS = torch.tensor([0.50, 0.55, 0.65, 0.75, 0.85, 0.95, 1.00]).cuda()
TANH_B = torch.tensor([0.25, 0.4, 0.6, 0.85, 1.2, 1.8, 3.0]).cuda()

KEYS = ['model.layers.0.mlp.gate_proj',
        'model.layers.0.linear_attn.in_proj_qkv',
        'model.layers.12.mlp.down_proj',
        'model.layers.3.self_attn.q_proj']

shapes = []
for key in KEYS:
    p = blob['layers'][key]
    of, inf = p['out_f'], p['in_f']
    nc = inf // 32
    aux = p['aux'].cuda().view(of, nc, 3)
    prm = aux[:, :, 0].int()
    gmx = (aux[:, :, 1:3].contiguous().view(torch.float16)
           .float().view(of, nc).unsqueeze(-1))
    tanh_f = ((prm & 0x80) != 0).unsqueeze(-1)
    TANH_Bc = TANH_B.cuda()
    fidx = (prm >> 3) & 0xF
    brev = torch.where(tanh_f, TANH_Bc[fidx.clamp(max=6)].unsqueeze(-1),
                       B_REL[fidx].unsqueeze(-1))
    b = brev * gmx
    beta = BETAS[prm & 7].unsqueeze(-1)
    gb = torch.exp2(beta * torch.log2(gmx))
    fmax = torch.where(tanh_f, torch.tanh(gmx / b), gb / (b + gb))
    fd = (torch.arange(16, device='cuda').view(1, 1, 16) / 15.0) * fmax
    fd = fd.clamp(max=1 - 1e-6)
    w_rat = fd * b / (1 - fd)
    w_rat = torch.where(beta != 1.0, w_rat ** (1.0 / beta), w_rat)
    Wl = torch.where(tanh_f, b * torch.atanh(fd), w_rat).clamp_min(0)
    Wl = torch.minimum(Wl, gmx)
    sh = (Wl / gmx).reshape(-1, 16)          # scale-free shape [nG, 16]
    idx = torch.randperm(sh.shape[0])[:200000]
    shapes.append(sh[idx])
    print(f'{key}: {sh.shape[0]/1e6:.1f}M groups', flush=True)

S = torch.cat(shapes)                        # [N, 16]
print(f'sample {S.shape[0]} shapes', flush=True)

for K in (256, 1024, 4096, 16384):
    torch.manual_seed(0)
    C = S[torch.randperm(S.shape[0])[:K]].clone()
    for _ in range(15):
        a = torch.cdist(S[::10], C).argmin(1)
        for j in range(K):
            m = a == j
            if m.any():
                C[j] = S[::10][m].mean(0)
    d = torch.cdist(S, C).min(1).values
    # error in weight units: |lv - lv'| / typical lv (lv ~ up to gmax)
    rel = d.mean().item()
    print(f'K={K}: mean shape residual {rel:.4f} '
          f'(weight-units ~ {rel:.3f} x gmax)', flush=True)
print('DONE', flush=True)
