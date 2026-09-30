# -*- coding: utf-8 -*-
"""GSQ runtime: K32 learned scalar codebook + per-16 int8 linear scale.
Layout: codes5 uint8 [nG, 10] (16 x 5-bit = 80 bits, LE d0|d1<<32 lo64
+ d2 hi), s_i8 uint8 [nG] (scale = s * smax/255), cb float [32]."""
import sys

import torch


def gs_pack(W, K=32, seed=42):
    of, inf = W.shape
    assert inf % 16 == 0
    g = W.float().reshape(-1, 16)
    sc = g.abs().amax(dim=1, keepdim=True).clamp_min(1e-8)
    smax = sc.max()
    u = torch.log2(sc)
    umin, umax_u = u.min(), u.max()
    v8 = ((u - umin) / (umax_u - umin + 1e-30) * 255) \
        .round().clamp(0, 255)
    s_i8 = v8.to(torch.uint8).reshape(-1)
    sc_q = torch.pow(2.0, umin + v8 / 255 * (umax_u - umin))
    g = g / sc_q
    sys.path.insert(0, r'E:\IXRUN')
    from benchmarks.hpq_minicpm5 import kmeans_gpu
    X = g.reshape(-1, 1)
    samp = X[torch.randperm(X.numel(), device=X.device)[:2_000_000]]
    gen = torch.Generator(device=X.device)
    gen.manual_seed(seed)
    idx = samp[torch.randperm(samp.numel(), device=X.device,
                              generator=gen)[:K]]
    C = idx.clone()
    for _ in range(20):
        a = torch.cdist(samp, C).argmin(1)
        for j in range(K):
            mk = a == j
            if mk.any():
                C[j] = samp[mk].mean(0)
    a = torch.empty(X.numel(), dtype=torch.long, device=X.device)
    blk = 4_000_000
    for i0 in range(0, X.numel(), blk):
        a[i0:i0 + blk] = torch.cdist(X[i0:i0 + blk], C).argmin(1)
    rec = C[a].reshape(g.shape) * sc_q
    out = rec.reshape(of, inf).to(torch.bfloat16)
    # pack 16 x 5-bit codes -> 10 bytes (d0|d1<<32 lo64, d2 hi16)
    a = a.reshape(-1, 16)
    ar5 = torch.arange(5, device=a.device)
    ar12 = torch.arange(12, device=a.device)
    ar4 = torch.arange(4, device=a.device)
    v = (a[:, :12].long() * (1 << (5 * ar12))).sum(1)      # 60 bits
    w = (a[:, 12:].long() * (1 << (5 * ar4))).sum(1)       # 20 bits
    lb = v.view(torch.uint8).reshape(-1, 8)
    lb[:, 7] = (lb[:, 7] & 0x0F) | ((w & 0xF).to(torch.uint8) << 4)
    c5 = torch.empty(a.shape[0], 10, dtype=torch.uint8)
    c5[:, 0:8] = lb
    c5[:, 8:10] = ((w >> 4) & 0xFFFF).to(torch.uint16) \
        .view(torch.uint8).reshape(-1, 2)
    return {'codes5': c5.contiguous().cpu(),
            's_i8': s_i8.cpu().contiguous(),
            's_base': float(umin),
            's_step': float((umax_u - umin) / 255),
            'cb': C.reshape(-1).float().cpu(),
            'out_f': of, 'in_f': inf, 'recon_ref': out}


def gs_decode_ref(pk):
    C = pk['cb'].cuda()
    nG = pk['s_i8'].numel()
    lo = pk['codes5'][:, 0:8].long().cuda()
    lo64 = (lo * (1 << (8 * torch.arange(8, device=lo.device)
                        ))).sum(1)
    hi = pk['codes5'][:, 8:10].long().cuda()
    hi16 = (hi * (1 << (8 * torch.arange(2, device=hi.device)
                        ))).sum(1)
    codes = torch.empty(nG, 16, dtype=torch.long, device='cuda')
    ar12 = torch.arange(12, device='cuda')
    codes[:, :12] = ((lo64[:, None] >> (5 * ar12)) & 0x1F)
    codes[:, 12] = ((lo64 >> 60) | (hi16 << 4)) & 0x1F
    codes[:, 13] = (hi16 >> 1) & 0x1F
    codes[:, 14] = (hi16 >> 6) & 0x1F
    codes[:, 15] = (hi16 >> 11) & 0x1F
    s = torch.pow(2.0, pk['s_base'] + pk['s_i8'].float().cuda()
                  * pk['s_step'])
    rec = (C[codes.reshape(-1)] * s.reshape(-1, 1).expand(-1, 16)
           .reshape(-1))
    of, inf = pk['out_f'], pk['in_f']
    return rec.reshape(of, inf).to(torch.bfloat16)


if __name__ == '__main__':
    torch.manual_seed(0)
    W = (torch.randn(512, 512) * 0.02).cuda()
    pk = gs_pack(W)
    d = gs_decode_ref(pk)
    w = W.float()
    r = d.float()
    print('pack/decode rel_err =',
          ((w - r).norm() / w.norm()).item(), flush=True)
