# -*- coding: utf-8 -*-
"""NF4-style analytic-quantile codebooks vs GSQ learned codebook.
All configs: per-16 absmax normalize (log-i8 scale storage like GSQ).
Levels: (a) K32 gaussian-quantile analytic, (b) NF4 exact 16 levels,
(c) K64 gaussian-quantile. Weight swap -> ppl."""
import gc
import sys
import time

sys.path.insert(0, r'E:\IXRUN')
import pandas  # noqa: F401
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from ixrun.config import MODEL_PATH, DATASET_CACHE
from ixrun.eval_utils import eval_ppl, load_wikitext
from ixrun.linear import iter_quantizable_linears

G = 16
NF4 = [-1.0, -0.6961928009986877, -0.5250730514526367,
       -0.39491748809814453, -0.28444138169288635, -0.18477312624454498,
       -0.09105003625154495, 0.0, 0.07958029955625534, 0.16093020141124725,
       0.24611230194568634, 0.33791524171829224, 0.44072481393814087,
       0.5626170039176941, 0.7229568362236023, 1.0]
NF4 = torch.tensor(NF4)


def gauss_levels(K):
    from torch.distributions import Normal
    n = Normal(0.0, 1.0)
    qs = torch.linspace(0.5 / K, 1 - 0.5 / K, K)
    return n.icdf(qs)


def quant(W, levels, gsize, scale_bits):
    of, inf = W.shape
    g = W.float().reshape(-1, gsize)
    sc = g.abs().amax(dim=1, keepdim=True).clamp_min(1e-8)
    u = torch.log2(sc)
    umin, umax = u.min(), u.max()
    v8 = ((u - umin) / (umax - umin + 1e-30) * 255).round() \
        .clamp(0, 255)
    sc_q = torch.pow(2.0, umin + v8 / 255 * (umax - umin)) \
        if scale_bits == 'i8log' else sc.clone()
    g = g / sc_q
    L = levels.to(W.device).float()
    a = torch.empty(g.numel(), dtype=torch.long, device=W.device)
    X = g.reshape(-1, 1)
    blk = 4_000_000
    for i0 in range(0, X.numel(), blk):
        d = (X[i0:i0 + blk] - L.unsqueeze(0)).abs()
        a[i0:i0 + blk] = d.argmin(1)
    rec = L[a].reshape(g.shape) * sc_q
    bpw = (L.numel().bit_length() - 1) + \
        (8.0 / gsize if scale_bits == 'i8log' else 16.0 / gsize)
    return rec.reshape(of, inf).to(torch.bfloat16), bpw


def main():
    tok = AutoTokenizer.from_pretrained(MODEL_PATH,
                                        trust_remote_code=True)
    texts = load_wikitext(cache_dir=DATASET_CACHE)
    m = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, dtype=torch.bfloat16,
        trust_remote_code=True).eval().to('cuda')
    targets = list(iter_quantizable_linears(m))
    orig = {n: mod.weight.data.clone().cpu() for n, mod in targets}
    print(f'{len(targets)} linears', flush=True)

    CONFIGS = [
        ('Q32-gauss +16', gauss_levels(32), 16, 'i8log'),
        ('Q64-gauss +16', gauss_levels(64), 16, 'i8log'),
        ('NF4 +64fp16', NF4, 64, 'fp16'),
    ]
    for cname, levels, gsize, sb in CONFIGS:
        t0 = time.time()
        errs = []
        snrs = []
        bpws = []
        for name, mod in targets:
            W = mod.weight.data
            W2, bpw = quant(W, levels, gsize, sb)
            w = orig[name].float().cuda()
            r = W2.float().cuda()
            errs.append(((w - r).norm() / w.norm()).item())
            snrs.append(10 * torch.log10(
                w.var() / ((w - r) ** 2).mean()).item())
            bpws.append(bpw)
            mod.weight.data = W2
        ppl = eval_ppl(m, tok, texts)
        print(f'[{cname}] bpw={sum(bpws)/len(bpws):.2f} '
              f'rel_err={sum(errs)/len(errs):.4f} '
              f'SNR={sum(snrs)/len(snrs):.1f}dB '
              f'ppl={ppl:.2f} ({time.time()-t0:.0f}s)', flush=True)
        for name, mod in targets:
            mod.weight.data = orig[name].cuda()
        gc.collect()
        torch.cuda.empty_cache()
    print('refs: bf16 56.02 | GSQ-kmeans 5.5bpw ~57.9 | UDCQ 6bpw '
          '58.06 | NF4-paper: near-fp16 at 4.25bpw')


if __name__ == '__main__':
    main()
