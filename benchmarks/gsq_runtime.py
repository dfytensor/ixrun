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
    nG = a.shape[0]
    ar5 = torch.arange(5, device=a.device)
    ar8 = torch.arange(8, device=a.device)
    c5 = torch.empty(nG, 10, dtype=torch.uint8)
    blk = 4_000_000
    for i0 in range(0, nG, blk):
        b = a[i0:i0 + blk]
        bits = torch.zeros(b.shape[0], 80, dtype=torch.int64,
                           device=a.device)
        for i in range(16):
            bits[:, i * 5:(i + 1) * 5] = (b[:, i:i + 1] >> ar5) & 1
        c5[i0:i0 + blk] = (bits.reshape(-1, 10, 8)
                           * (1 << ar8)).sum(-1).to(torch.uint8)
    return {'codes5': c5.contiguous().cpu(),
            's_i8': s_i8.cpu().contiguous(),
            's_base': float(umin),
            's_step': float((umax_u - umin) / 255),
            'cb': C.reshape(-1).float().cpu(),
            'out_f': of, 'in_f': inf, 'recon_ref': out}


def gs_decode_ref(pk):
    C = pk['cb'].cuda()
    c5 = pk['codes5'].long().cuda()
    nG = c5.shape[0]
    ar8 = torch.arange(8, device=c5.device)
    ar5 = torch.arange(5, device=c5.device)
    bits = ((c5.unsqueeze(-1) >> ar8) & 1).reshape(nG, 80)
    codes = (bits.reshape(nG, 16, 5) * (1 << ar5)).sum(-1)
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
