# -*- coding: utf-8 -*-
"""int5-g32+warm runtime: pack, decode-ref, deploy wrapper."""
import sys

import torch


def ig32_pack(W, group=32, nlev=16, lloyd=25):
    """Quantize W [out, in] -> dict(codes, sign, levels, ...).
    Mirrors benchmarks/int4g32_sweep.py::quant_gw but returns packed
    pieces + per-group level table."""
    of, inf = W.shape
    assert inf % group == 0 and nlev == 16
    G = W.float().view(-1, group)
    signb = (G < 0)
    A = G.abs()
    gmax = A.amax(1, keepdim=True).clamp_min(1e-12)
    nm1 = nlev - 1

    best_mse = torch.full((A.shape[0],), float('inf'), device=A.device)
    best_lv = torch.zeros(A.shape[0], nlev, device=A.device)

    B_REL = torch.logspace(-3, 2, 15).tolist()
    BETAS = [0.50, 0.55, 0.65, 0.75, 0.85, 0.95, 1.00]
    TANH_B = [0.25, 0.4, 0.6, 0.85, 1.2, 1.8, 3.0]

    for br in B_REL:
        b = br * gmax
        for beta in BETAS:
            f = A ** beta / (b + A ** beta)
            fmax = gmax ** beta / (b + gmax ** beta)
            q = torch.round(f / fmax * nm1).clamp_(0, nm1)
            fd = (q / nm1 * fmax).clamp_(max=1 - 1e-6)
            wr = (fd * b / (1 - fd)).clamp_min(0)
            if beta != 1.0:
                wr = wr ** (1.0 / beta)
            wr = torch.minimum(wr, gmax)
            mse = ((A - wr) ** 2).mean(dim=1)
            imp = mse < best_mse
            if imp.any():
                best_mse = torch.where(imp, mse, best_mse)
                fdi = ((torch.arange(nlev, device=A.device,
                                     dtype=torch.float32)
                        / nm1).view(1, -1) * fmax).clamp(max=1 - 1e-6)
                lv = (fdi * b / (1 - fdi)).clamp_min(0)
                if beta != 1.0:
                    lv = lv ** (1.0 / beta)
                lv = torch.minimum(lv, gmax)
                best_lv = torch.where(imp.view(-1, 1), lv, best_lv)

    for br in TANH_B:
        b = br * gmax
        f = torch.tanh(A / b)
        fmax = torch.tanh(gmax / b)
        q = torch.round(f / fmax * nm1).clamp_(0, nm1)
        fd = (q / nm1 * fmax).clamp_(max=1 - 1e-6)
        wr = (b * torch.atanh(fd)).clamp_min(0)
        wr = torch.minimum(wr, gmax)
        mse = ((A - wr) ** 2).mean(dim=1)
        imp = mse < best_mse
        if imp.any():
            best_mse = torch.where(imp, mse, best_mse)
            fdi = ((torch.arange(nlev, device=A.device,
                                 dtype=torch.float32)
                    / nm1).view(1, -1) * fmax).clamp(max=1 - 1e-6)
            lv = (b * torch.atanh(fdi)).clamp_min(0)
            lv = torch.minimum(lv, gmax)
            best_lv = torch.where(imp.view(-1, 1), lv, best_lv)

    lv = best_lv.clone()
    for it in range(lloyd):
        d = (A.unsqueeze(-1) - lv.unsqueeze(1)).abs()
        idx = d.argmin(-1)
        new = lv.clone()
        for k in range(nlev):
            mm = (idx == k)
            cnt = mm.sum(1, keepdim=True)
            s = (A * mm).sum(1, keepdim=True)
            new[:, k:k + 1] = torch.where(cnt > 0, s / cnt.clamp_min(1),
                                          lv[:, k:k + 1])
        d_new = (A.unsqueeze(-1) - new.unsqueeze(1)).abs()
        mse_new = d_new.min(-1).values.mean(1)
        mse_old = d.min(-1).values.mean(1)
        better = (mse_new <= mse_old).view(-1, 1)
        lv = torch.where(better, new, lv)
    d = (A.unsqueeze(-1) - lv.unsqueeze(1)).abs()
    idx = d.argmin(-1)                                     # [nG, 32]

    # pack 32 x 4-bit -> 16 B per group (low nibble = even element)
    nG = idx.shape[0]
    bits = ((idx.unsqueeze(-1) >> torch.arange(
        4, device=A.device, dtype=torch.int64)) & 1)        # [nG,32,4]
    c16 = bits.reshape(nG, 16, 8)
    ar8 = torch.arange(8, device=A.device)
    codes = (c16 * (1 << ar8)).sum(-1).to(torch.uint8)      # [nG, 16]
    # sign: 1 bit per weight, bit j of group word j-th element
    sb = signb.to(torch.int64)
    signw = (sb * (1 << torch.arange(32, device=A.device))).sum(-1)
    signw = signw.to(torch.int32)                           # [nG]

    n_rows = of
    return {'codes': codes.view(n_rows, inf // 2).cpu().contiguous(),
            'sign': signw.view(n_rows, inf // 32).cpu().contiguous(),
            'levels': lv.reshape(n_rows, inf // group, nlev).half()
                        .cpu().contiguous(),
            'out_f': of, 'in_f': inf}


def ig32_decode_ref(pk):
    """bf16 reconstruction (ground truth for gates + dense prefill)."""
    lv = pk['levels'].float().cuda()                        # [r, g, 16]
    r, gc, nlev = lv.shape
    in_f = pk['in_f']
    b = pk['codes'].cuda().long().view(r, in_f // 2)
    nib = torch.empty(r, in_f, dtype=torch.long, device=lv.device)
    nib[:, 0::2] = b & 0xF
    nib[:, 1::2] = b >> 4
    idx3 = nib.view(r, gc, 32)                              # level idx
    sgn = pk['sign'].cuda().int().view(r, gc, 1)
    bits = (sgn >> torch.arange(32, device=lv.device)) & 1
    sgnf = 1 - bits.reshape(r, gc * 32).float() * 2
    rec = lv.gather(2, idx3.clamp(0, nlev - 1)).squeeze(-1)  # [r, gc, 32]
    rec = rec.reshape(r, -1) * sgnf
    return rec.reshape(pk['out_f'], pk['in_f']).to(torch.bfloat16)


class Ig32Linear(torch.nn.Module):
    """M=1 -> hand-CUDA GEMV; M>1 -> dense decode + cublas."""

    def __init__(self, pk):
        super().__init__()
        self.pk = pk
        self.pk['codes'] = pk['codes'].cuda()
        self.pk['sign'] = pk['sign'].cuda()
        self.pk['levels'] = pk['levels'].cuda()
        self.out_features = pk['out_f']
        self.in_features = pk['in_f']
        self._dense = None

    def _decode(self):
        if self._dense is None:
            self._dense = ig32_decode_ref(self.pk)
        return self._dense

    def forward(self, x):
        if x.numel() == self.in_features:
            from experiments.ig32_gemv_cuda import ig32_gemv
            return ig32_gemv(x.reshape(-1), self.pk)
        W = self._decode()
        return torch.nn.functional.linear(x.to(W.dtype), W)
