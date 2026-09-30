# -*- coding: utf-8 -*-
"""Synthesis format study: GMM-style learned codebook (signed, no sign
bit) x per-16 group scale (the HPQ-x-scale lesson) x compact scale
storage. Targets UDCQ 6bpw quality (58.06) below 6bpw.
Configs: K{16,32,64} x scale{none, fp16, e4m3-as-8b}."""
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
from benchmarks.hpq_minicpm5 import kmeans_gpu

G = 16
CONFIGS = [(32, 8), (32, 16)]


def gs_quant(W, K, sbits):
    """Per-16 group scale + learned signed codebook (kmeans surrogate).
    Returns recon bf16 + bpw."""
    of, inf = W.shape
    g = W.float().reshape(-1, G)
    sc = g.abs().amax(dim=1, keepdim=True).clamp_min(1e-8)
    sc_q = torch.ones_like(sc)
    if sbits == 16:
        sc_q = sc.half().float()
    elif sbits == 8:
        sc_q = sc.to(torch.bfloat16).float()
    g = g / sc_q
    X = g.reshape(-1)
    samp = X[torch.randperm(X.numel(), device=X.device)[:2_000_000]]
    C = kmeans_gpu(samp.unsqueeze(1), K, iters=20)
    C = C.sort(dim=0).values
    a = torch.empty(X.numel(), dtype=torch.long, device=X.device)
    blk = 4_000_000
    for i0 in range(0, X.numel(), blk):
        a[i0:i0 + blk] = torch.cdist(X[i0:i0 + blk].unsqueeze(1),
                                     C).argmin(1)
    rec = (C[a.squeeze()]).reshape(g.shape) * sc_q
    out = rec.reshape(of, inf).to(torch.bfloat16)
    bpw = (K.bit_length() - 1) + (sbits / G if sbits else 0)
    return out, bpw


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
    for K, sbits in CONFIGS:
        t0 = time.time()
        errs = []
        snrs = []
        bpws = []
        for name, mod in targets:
            W = mod.weight.data
            W2, bpw = gs_quant(W, K, sbits)
            w = orig[name].float().cuda()
            r = W2.float().cuda()
            errs.append(((w - r).norm() / w.norm()).item())
            snrs.append(10 * torch.log10(
                w.var() / ((w - r) ** 2).mean()).item())
            bpws.append(bpw)
            mod.weight.data = W2
        ppl = eval_ppl(m, tok, texts)
        print(f'[GS K={K} scale={sbits}] bpw={sum(bpws)/len(bpws):.2f} '
              f'rel_err={sum(errs)/len(errs):.4f} '
              f'SNR={sum(snrs)/len(snrs):.1f}dB '
              f'ppl={ppl:.2f} ({time.time()-t0:.0f}s)', flush=True)
        for name, mod in targets:
            mod.weight.data = orig[name].cuda()
        gc.collect()
        torch.cuda.empty_cache()
    print('refs: bf16 56.02 | UDCQ 6bpw 58.06 | GMM-noscale 6bpw 57.92 '
          '| HPQxscale-mixed 6.36bpw 57.22')


if __name__ == '__main__':
    main()
