# -*- coding: utf-8 -*-
"""Repack ig32(prm) blob -> ig32v (shared 256-entry shape codebook).

aux[g*3] becomes the fp16-shape-codebook index instead of the prm byte;
gmax bytes unchanged. codes/sign are copied (codes were assigned against
the exact levels; VQ error << level spacing). Trains the codebook by VQ
on sampled shapes from the source blob.
"""
import sys, time
sys.path.insert(0, r'E:\IXRUN')
import pandas  # noqa: F401
import torch

SRC = r'F:\models\qwen38_ig32_blob.pt'
DST = r'F:\models\qwen38_ig32v_blob.pt'
K = 256
SAMPLE_CAP = 60000          # groups sampled per matrix for training

B_REL = torch.logspace(-3, 2, 15).cuda()
BETAS = torch.tensor([0.50, 0.55, 0.65, 0.75, 0.85, 0.95, 1.00]).cuda()
TANH_B = torch.tensor([0.25, 0.4, 0.6, 0.85, 1.2, 1.8, 3.0]).cuda()


def shapes_chunk(prm, gmx):
    """prm [m] int, gmx [m,1] float -> [m,16] shape = lv/gmx."""
    m = prm.shape[0]
    tanh_f = ((prm & 0x80) != 0).unsqueeze(-1)
    fidx = (prm >> 3) & 0xF
    brev = torch.where(tanh_f, TANH_B[fidx.clamp(max=6)].unsqueeze(-1),
                       B_REL[fidx].unsqueeze(-1))
    b = brev * gmx
    beta = BETAS[prm & 7].unsqueeze(-1)
    gb = torch.exp2(beta * torch.log2(gmx))
    fmax = torch.where(tanh_f, torch.tanh(gmx / b), gb / (b + gb))
    fd = (torch.arange(16, device='cuda').view(1, 16) / 15.0) * fmax
    fd = fd.clamp(max=1 - 1e-6)
    w_rat = fd * b / (1 - fd)
    w_rat = torch.where(beta != 1.0, w_rat ** (1.0 / beta), w_rat)
    Wl = torch.where(tanh_f, b * torch.atanh(fd), w_rat).clamp_min(0)
    Wl = torch.minimum(Wl, gmx)
    return Wl / gmx


def groups_of(p, of, inf):
    nc = inf // 32
    aux = p['aux'].cuda().view(of, nc, 3)
    prm = aux[:, :, 0].reshape(-1).int()
    gmx = (aux[:, :, 1:3].contiguous().view(torch.float16)
           .float().view(of, nc).reshape(-1, 1))
    return prm, gmx, nc


t0 = time.time()
blob = torch.load(SRC, map_location='cpu', mmap=True, weights_only=True)
keys = list(blob['layers'].keys())

# ---- phase 1: sample shapes, train codebook ----
samples = []
for key in keys:
    p = blob['layers'][key]
    prm, gmx, nc = groups_of(p, p['out_f'], p['in_f'])
    n = prm.shape[0]
    sel = torch.randperm(n, device='cuda')[:SAMPLE_CAP]
    sh = shapes_chunk(prm[sel], gmx[sel])
    samples.append(sh)
    del prm, gmx
S = torch.cat(samples)
print(f'sample {S.shape[0]/1e6:.1f}M shapes [{time.time()-t0:.0f}s]',
      flush=True)
torch.manual_seed(42)
C = S[torch.randperm(S.shape[0])[:K]].clone()
for it in range(25):
    a = torch.cdist(S, C).argmin(1)
    for j in range(K):
        m = a == j
        if m.any():
            C[j] = S[m].mean(0)
resid = torch.cdist(S, C).min(1).values.mean().item()
print(f'codebook K={K} residual {resid:.5f} [{time.time()-t0:.0f}s]',
      flush=True)
del S, samples
torch.cuda.empty_cache()

# ---- phase 2: index all groups ----
out = {'format': 'ig32v-k256', 'cbv': C.half().cpu().contiguous(),
       'embed': blob['embed'], 'codebook': blob['codebook'], 'layers': {}}
for ki, key in enumerate(keys):
    p = blob['layers'][key]
    of, inf = p['out_f'], p['in_f']
    prm, gmx, nc = groups_of(p, of, inf)
    n = prm.shape[0]
    idxb = torch.empty(n, dtype=torch.uint8, device='cuda')
    CH = 2_000_000
    for i0 in range(0, n, CH):
        i1 = min(n, i0 + CH)
        sh = shapes_chunk(prm[i0:i1], gmx[i0:i1])
        idxb[i0:i1] = torch.cdist(sh, C).argmin(1).to(torch.uint8)
    gb = gmx.view(-1).half().view(torch.uint8).view(n, 2)
    aux = torch.empty(n, 3, dtype=torch.uint8, device='cuda')
    aux[:, 0] = idxb
    aux[:, 1:3] = gb
    out['layers'][key] = {
        'codes': p['codes'], 'sign': p['sign'],
        'aux': aux.view(of, nc, 3).cpu().contiguous(),
        'out_f': of, 'in_f': inf}
    if (ki + 1) % 40 == 0 or ki + 1 == len(keys):
        print(f'[{ki+1}/{len(keys)}] {key} [{time.time()-t0:.0f}s]',
              flush=True)

torch.save(out, DST)
print(f'SAVED {DST} in {time.time()-t0:.0f}s', flush=True)
