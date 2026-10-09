# -*- coding: utf-8 -*-
"""Stage-by-stage debug: search-only rec vs post-Lloyd rec."""
import sys
sys.path.insert(0, r'E:\IXRUN')
import torch

torch.manual_seed(0)
W = (torch.randn(256, 512) * 0.02).cuda()
of, inf = W.shape
G = W.float().view(-1, 32)
sign = (G < 0)
A = G.abs()
gmax = A.amax(1, keepdim=True).clamp_min(1e-12)
nG = A.shape[0]

B_REL = torch.logspace(-3, 2, 15).tolist()
BETAS = [0.50, 0.55, 0.65, 0.75, 0.85, 0.95, 1.00]
TANH_B = [0.25, 0.4, 0.6, 0.85, 1.2, 1.8, 3.0]

best_mse = torch.full((nG,), float('inf'), device=A.device)
best_lv = torch.zeros(nG, 8, device=A.device)

def rel(x):
    return ((x - W).norm() / W.norm()).item()

for br in B_REL:
    b = br * gmax
    for beta in BETAS:
        f = A ** beta / (b + A ** beta)
        fmax = gmax ** beta / (b + gmax ** beta)
        q = torch.round(f / fmax * 7).clamp_(0, 7)
        fd = (q / 7 * fmax).clamp_(max=1 - 1e-6)
        wr = (fd * b / (1 - fd)).clamp_min(0)
        if beta != 1.0:
            wr = wr ** (1.0 / beta)
        wr = torch.minimum(wr, gmax)
        mse = ((A - wr) ** 2).mean(dim=1)
        imp = mse < best_mse
        if imp.any():
            best_mse = torch.where(imp, mse, best_mse)
            fdi = ((torch.arange(8, device=A.device, dtype=torch.float32)
                    / 7).view(1, 8) * fmax).clamp(max=1 - 1e-6)
            lv = (fdi * b / (1 - fdi)).clamp_min(0)
            if beta != 1.0:
                lv = lv ** (1.0 / beta)
            lv = torch.minimum(lv, gmax)
            best_lv = torch.where(imp.view(nG, 1), lv, best_lv)

for br in TANH_B:
    b = br * gmax
    f = torch.tanh(A / b)
    fmax = torch.tanh(gmax / b)
    q = torch.round(f / fmax * 7).clamp_(0, 7)
    fd = (q / 7 * fmax).clamp_(max=1 - 1e-6)
    wr = (b * torch.atanh(fd)).clamp_min(0)
    wr = torch.minimum(wr, gmax)
    mse = ((A - wr) ** 2).mean(dim=1)
    imp = mse < best_mse
    if imp.any():
        best_mse = torch.where(imp, mse, best_mse)
        fdi = ((torch.arange(8, device=A.device, dtype=torch.float32)
                / 7).view(1, 8) * fmax).clamp(max=1 - 1e-6)
        lv = (b * torch.atanh(fdi)).clamp_min(0)
        lv = torch.minimum(lv, gmax)
        best_lv = torch.where(imp.view(nG, 1), lv, best_lv)

print(f'search mse mean {best_mse.mean():.6f}')
am0 = (A.unsqueeze(-1) - best_lv.unsqueeze(1)).abs().min(-1).values.pow(2).mean(-1)
print(f'assign-mse: mean {am0.mean():.6f} p50 {am0.median():.6f} p99 {am0.quantile(0.99):.6f} max {am0.max():.6f}')
worst = am0.topk(3).indices
for wgi in worst.tolist():
    print(f'  worst g{wgi}: mse {am0[wgi].item():.5f} gmax {gmax[wgi,0].item():.4f} lv {[round(v,4) for v in best_lv[wgi].tolist()]}')
print('g0: gmax', gmax[0,0].item(), 'A0[:8]', [round(v,4) for v in A[0,:8].tolist()])
print('g0: best_lv0', [round(v,5) for v in best_lv[0].tolist()])
print('g0: search-mse', best_mse[0].item(), 'assign-mse', (A[0].unsqueeze(-1) - best_lv[0].unsqueeze(0)).abs().min(-1).values.pow(2).mean().item())
d = (A.unsqueeze(-1) - best_lv.unsqueeze(1)).abs()
idx = d.argmin(-1)
rec = sign.to(A.dtype) * best_lv.gather(1, idx)
W1 = rec.view(of, inf)
print(f'post-search rel-L2 {rel(W1):.4f}')

lv = best_lv.clone()
for it in range(25):
    d = (A.unsqueeze(-1) - lv.unsqueeze(1)).abs()
    idx = d.argmin(-1)
    new = lv.clone()
    for k in range(8):
        mm = (idx == k)
        cnt = mm.sum(1, keepdim=True)
        s = (A * mm).sum(1, keepdim=True)
        new[:, k:k + 1] = torch.where(cnt > 0, s / cnt.clamp_min(1),
                                      lv[:, k:k + 1])
    d_new = (A.unsqueeze(-1) - new.unsqueeze(1)).abs()
    mse_new = d_new.min(-1).values.mean(1)
    mse_old = d.min(-1).values.mean(1)
    better = (mse_new <= mse_old).view(nG, 1)
    lv = torch.where(better, new, lv)
    if it % 5 == 0 or it == 24:
        print(f'  lloyd {it}: mse {mse_new.mean():.6f} '
              f'lvl-max {lv.max().item():.4f}', flush=True)
d = (A.unsqueeze(-1) - lv.unsqueeze(1)).abs()
idx = d.argmin(-1)
sgn = torch.where(sign, -1.0, 1.0)
rec = sgn * lv.gather(1, idx)
W2 = rec.view(of, inf)
print(f'post-lloyd rel-L2 {rel(W2):.4f}')
print(f'level stats: min {lv.min():.5f} max {lv.max():.4f} '
      f'nan {torch.isnan(lv).any().item()}')
