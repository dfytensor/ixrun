# -*- coding: utf-8 -*-
"""ABQ (Adaptive Book Quantization) — synthesis:
  ig32's per-group adaptive level placement (via a shared book VQ)
+ UDCQ's decode-cheap structure (global books in smem, sign-fold,
  per-group scale factored out of the element loop)
+ Lloyd warm-start training.

Format: 4b code + 1b sign + 4b book-sel/32 + fp16 scale/32 = 5.625 bpw.
All training is weight-only (zero calibration), offline, per model.
"""
import sys
sys.path.insert(0, r'E:\IXRUN')
import torch

CH = 1_000_000


def _assign(w, s, books):
    """w [m,32]; s [m,1]; books [K,16]. -> sel, codes (both [m,...]),
    normalized MSE per group."""
    v = w / s
    m = v.shape[0]
    K = books.shape[0]
    errs = torch.empty(m, K, device=w.device)
    for k in range(K):
        d2 = (v.unsqueeze(-1) - books[k]).pow(2).amin(-1)   # [m,32]
        errs[:, k] = d2.sum(1)
        del d2
    sel = errs.argmin(1)
    best = books[sel]
    dd = (v.unsqueeze(-1) - best.unsqueeze(1)).abs()
    codes = dd.argmin(-1)
    return sel, codes


def _fit_scale(w, s, books, sel, codes):
    lv = books[sel].gather(1, codes)
    num = (w * lv).sum(1, keepdim=True)
    den = lv.pow(2).sum(1, keepdim=True).clamp_min(1e-12)
    return (num / den).clamp_min(1e-12), lv


def abq_train_books(Gs, K=16, group=32, iters=6, seed=42, verbose=True,
                    n_books=None):
    """Gs: list of [nG, 32] fp32 GPU group tensors."""
    torch.manual_seed(seed)
    n_books = n_books or K
    Ss = [g.abs().amax(1, keepdim=True) for g in Gs]
    n_tot = sum(g.shape[0] for g in Gs)
    if verbose:
        print(f'[abq] {n_tot/1e6:.1f}M groups, books={n_books} x {K} levels',
              flush=True)

    # book init: jittered uniform quantiles over a large value sample
    samp = []
    for g, s in zip(Gs, Ss):
        v = (g / s).reshape(-1)
        n = min(v.numel(), 3_000_000)
        samp.append(v[torch.randint(0, v.numel(), (n,), device=v.device)])
    samp = torch.cat(samp)
    qs = (torch.arange(K, device=samp.device, dtype=torch.float32) + 0.5) / K
    base = torch.quantile(samp[torch.randperm(samp.shape[0])[:5_000_000]],
                          qs).sort().values
    jit = torch.linspace(-0.18, 0.18, n_books, device=samp.device)
    books = torch.empty(n_books, K, device=samp.device)
    for k in range(n_books):
        books[k] = (base + jit[k]).sort().values
    del samp
    torch.cuda.empty_cache()

    for it in range(iters):
        sums = torch.zeros(n_books, K, device=Gs[0].device)
        cnts = torch.zeros(n_books, K, device=Gs[0].device)
        tot = 0.0
        for g, s in zip(Gs, Ss):
            for i0 in range(0, g.shape[0], CH):
                w = g[i0:i0 + CH]
                sc = s[i0:i0 + CH]
                sel, codes = _assign(w, sc, books)
                s_new, lv = _fit_scale(w, sc, books, sel, codes)
                v = w / sc
                tot += (v - lv).pow(2).sum().item()
                # pooled scatter over the assigned book
                for k in range(n_books):
                    mk = sel == k
                    if mk.any():
                        sums[k].index_add_(0, codes[mk].reshape(-1),
                                           v[mk].reshape(-1))
                        cnts[k].index_add_(0, codes[mk].reshape(-1),
                                           torch.ones_like(
                                               codes[mk].reshape(-1),
                                               dtype=torch.float32))
        old = books.clone()
        for k in range(n_books):
            m = cnts[k] > 0
            books[k, m] = sums[k, m] / cnts[k, m]
        books, _ = books.sort(dim=1)
        delta = (books - old).abs().max().item()
        if verbose:
            print(f'[abq] iter {it}: norm-MSE {tot/n_tot/32:.5f} '
                  f'book-delta {delta:.5f}', flush=True)
        if delta < 1e-5:
            break
    return books


def abq_quantize(W, books, group=32):
    """W [out, in] -> (Wq bf16, sel, codes, scale, bpw)."""
    of, inf = W.shape
    g = W.reshape(-1, group)
    s = g.abs().amax(1, keepdim=True)
    sel, codes = _assign(g, s, books)
    s, _ = _fit_scale(g, s, books, sel, codes)
    sel, codes = _assign(g, s, books)          # reassign at LS scale
    s, lv = _fit_scale(g, s, books, sel, codes)
    Wq = (lv * s).reshape(of, inf).to(torch.bfloat16)
    K = books.shape[0]
    bpw = 4 + 1 + (K.bit_length() - 1) / group + 16 / group
    return Wq, sel, codes, s, bpw
