# -*- coding: utf-8 -*-
"""Bayesian-GMM quantization + residual compensation on MiniCPM5-1B.

Scheme under test (user proposal):
  1. Bayesian GMM fits the weight distribution -> codebook = component
     means (SIGNED, so no separate sign bit unlike UDCQ) + Dirichlet
     weight prior with automatic component pruning.
  2. Per-group scale (group=64) adapts the global codebook locally.
  3. Residual r = w - w_hat re-quantized (uniform, per-group) for
     precision compensation ("second layer").
  4. Storage = K-bit component index + fp16 group scales + optional
     residual stream -> compare bpw and ppl vs plain int8.

Configs:
  int8        : per-tensor int8 baseline (8 bpw)
  gmm         : GMM codebook, 4-bit index, fp16 scale/64 (~4.25 bpw)
  gmm+r4      : + 4-bit residual (~8.5 bpw, vs int8 at same 8)
  gmm+r2      : + 2-bit residual (~6.3 bpw)
  int8g       : per-64-group int8 (~8.25 bpw, fair "group int8" control)

Run: python -X utf8 -m benchmarks.bench_gmm_minicpm5
"""
import sys, time, gc
sys.path.insert(0, r'E:\IXRUN')
import pandas                       # before torch (heap rule)
import torch
import torch.nn as nn

from transformers import AutoTokenizer
from ixrun.config import MODEL_PATH, DATASET_CACHE
from ixrun.eval_utils import eval_ppl, load_wikitext

torch.manual_seed(0)
GROUP = 64


# --------------------------------------------------------------------- #
def fit_bayesian_gmm(x, K=16, iters=40, prune_frac=0.25):
    """1-D isotropic GMM via EM with Dirichlet prior + component pruning
    (poor-man's variational Bayes: components with pi < prune_frac/K are
    dropped, effective K is data-driven)."""
    x = x.float()
    qs = torch.linspace(0.02, 0.98, K)
    mu = torch.quantile(x, qs)
    var = torch.full((K,), float(x.var()) * 0.1)
    pi = torch.full((K,), 1.0 / K)
    prior = 1.0 / K          # Dirichlet pseudo-count
    for _ in range(iters):
        lp = (-0.5 * ((x[:, None] - mu[None, :]) ** 2 / var[None, :]
                      + torch.log(var)[None, :])
              + torch.log(pi)[None, :])
        lp = lp - lp.logsumexp(1, keepdim=True)
        r = lp.exp()
        Nk = r.sum(0) + prior
        mu = (r * x[:, None]).sum(0) / Nk
        var = (r * (x[:, None] - mu[None, :]) ** 2).sum(0) / Nk + 1e-8
        pi = Nk / Nk.sum()
    keep = pi > prune_frac / K
    return mu[keep], var[keep], pi[keep]


def _nearest(y, mu):
    """y [nb, group] -> idx [nb, group] into mu (loop over K, low mem)."""
    idx = torch.zeros_like(y, dtype=torch.long)
    best = (y - mu[0]).abs()
    for k in range(1, mu.numel()):
        d = (y - mu[k]).abs()
        take = d < best
        best = torch.where(take, d, best)
        idx = torch.where(take, torch.full_like(idx, k), idx)
    return idx


def quantize_layer_gmm(w, mu, group=GROUP, residual_bits=0, block=200000,
                       idx_bits=4):
    """Returns (w_deq [same shape], bits_per_weight)."""
    flat = w.reshape(-1).float().cpu()
    N = flat.numel()
    pad = (-N) % group
    if pad:
        flat = torch.cat([flat, flat.new_zeros(pad)])
    g = flat.view(-1, group)
    ng = g.shape[0]
    idx_all = torch.empty_like(g, dtype=torch.long)
    deq = torch.empty_like(g)
    res_all = torch.empty_like(g) if residual_bits else None
    mmax = float(mu.abs().max())
    for b0 in range(0, ng, block):
        gb = g[b0:b0 + block]
        sc = gb.abs().amax(1) / mmax            # group scale (fp16 stored)
        y = gb / sc[:, None]
        idx = _nearest(y, mu)
        wq = mu[idx] * sc[:, None]
        idx_all[b0:b0 + block] = idx
        deq[b0:b0 + block] = wq
        if residual_bits:
            r = gb - wq
            rs = r.abs().amax(1) / ((1 << (residual_bits - 1)) - 1)
            rs = rs.clamp_min(1e-12)
            rq = (r / rs[:, None]).round().clamp(
                -(1 << (residual_bits - 1)), (1 << (residual_bits - 1)) - 1)
            deq[b0:b0 + block] = wq + rq * rs[:, None]
    wd = deq.reshape(-1)[:N].reshape(w.shape)
    bits = idx_bits + 16 / group                # idx + fp16 scale/group
    if residual_bits:
        bits += residual_bits + 16 / group
    return wd, bits


def quantize_layer_int8(w, group=None):
    flat = w.reshape(-1).float().cpu()
    if group is None:                            # per-tensor
        sc = flat.abs().max() / 127.0
        q = (flat / sc).round().clamp(-127, 127)
        return (q * sc).reshape(w.shape), 8.0
    pad = (-flat.numel()) % group
    if pad:
        flat = torch.cat([flat, flat.new_zeros(pad)])
    g = flat.view(-1, group)
    sc = g.abs().amax(1) / 127.0
    sc = sc.clamp_min(1e-12)
    q = (g / sc[:, None]).round().clamp(-127, 127)
    return (q * sc[:, None]).reshape(-1)[:w.numel()].reshape(w.shape), \
        8.0 + 16 / group


# --------------------------------------------------------------------- #
def main():
    from transformers import AutoModelForCausalLM
    from ixrun.linear import iter_quantizable_linears

    tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    texts = load_wikitext(cache_dir=DATASET_CACHE)
    print(f'[gmm] wikitext samples {len(texts)}', flush=True)

    m = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16, trust_remote_code=True)
    m.eval()
    targets = list(iter_quantizable_linears(m))
    print(f'[gmm] {len(targets)} quantizable linears', flush=True)

    # ---- fit the global Bayesian GMM on a weight sample ----
    t0 = time.time()
    samp = []
    per = max(1, 3_000_000 // len(targets))
    for _, mod in targets:
        w = mod.weight.data.reshape(-1).float()
        if w.numel() > per:
            sel = torch.randint(0, w.numel(), (per,))
            w = w[sel]
        samp.append(w)
    xs = torch.cat(samp)
    print(f'[gmm] fitting Bayesian GMM on {xs.numel()} samples...',
          flush=True)
    mu, var, pi = fit_bayesian_gmm(xs, K=16, iters=40)
    print(f'[gmm] effective components {len(mu)}/16 (pruned by prior), '
          f'{time.time()-t0:.0f}s', flush=True)
    print(f'[gmm] mu = {[round(float(v), 4) for v in mu]}', flush=True)
    mu32, _, _ = fit_bayesian_gmm(xs, K=32, iters=40)
    print(f'[gmm] K=32 fit: {len(mu32)} effective components', flush=True)

    # keep a pristine copy of the weights (CPU bf16) for repeated runs
    master = [(name, mod.weight.data.clone()) for name, mod in targets]

    def restore():
        for (name, w0), (_, mod) in zip(master, targets):
            mod.weight.data = w0.clone()

    results = []
    for kind in ('int8', 'int8g', 'gmm', 'gmm+r4', 'gmm+r2',
                 'g16', 'g16+r2', 'g32', 'g32+r2'):
        restore()
        t1 = time.time()
        bpw_acc = []
        for _, mod in targets:
            w = mod.weight.data
            if kind == 'int8':
                wd, bits = quantize_layer_int8(w)
            elif kind == 'int8g':
                wd, bits = quantize_layer_int8(w, group=GROUP)
            elif kind == 'gmm':
                wd, bits = quantize_layer_gmm(w, mu)
            elif kind == 'gmm+r4':
                wd, bits = quantize_layer_gmm(w, mu, residual_bits=4)
            elif kind == 'gmm+r2':
                wd, bits = quantize_layer_gmm(w, mu, residual_bits=2)
            elif kind == 'g16':
                wd, bits = quantize_layer_gmm(w, mu, group=16)
            elif kind == 'g16+r2':
                wd, bits = quantize_layer_gmm(w, mu, group=16,
                                              residual_bits=2)
            elif kind == 'g32':
                wd, bits = quantize_layer_gmm(w, mu32, group=16,
                                              idx_bits=5)
            else:
                wd, bits = quantize_layer_gmm(w, mu32, group=16,
                                              idx_bits=5, residual_bits=2)
            mod.weight.data = wd.to(torch.bfloat16)
            bpw_acc.append(bits)
        m = m.cuda()
        ppl = eval_ppl(m, tok, texts)
        m = m.cpu()
        gc.collect(); torch.cuda.empty_cache()
        bpw = sum(bpw_acc) / len(bpw_acc)
        results.append((kind, bpw, ppl))
        print(f'[gmm] {kind}: bpw={bpw:.2f} ppl={ppl:.2f} '
              f'({time.time()-t1:.0f}s)', flush=True)

    print('\n=== MiniCPM5-1B: Bayesian-GMM vs int8 ===')
    print(f'{"scheme":<9}{"bpw":>7}{"ppl":>9}')
    base = next(r for r in results if r[0] == 'int8')
    for kind, bpw, ppl in results:
        print(f'{kind:<9}{bpw:>7.2f}{ppl:>9.2f}   '
              f'({ppl - base[2]:+.2f} vs int8, '
              f'{(1 - bpw / 8) * 100:+.0f}% size)')
    print(f'\nbf16 reference ppl = 56.02')


if __name__ == '__main__':
    main()
