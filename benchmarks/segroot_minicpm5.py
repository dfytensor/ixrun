# -*- coding: utf-8 -*-
"""Segmented root-power non-uniform 6bit quantization on MiniCPM5-1B.
|w| bucketed into segments; per-segment root p flattens the span;
3-bit linear levels inside; sign 1b. Sweeps: decade vs quantile
boundaries, 4/8 segments, root on/off."""
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


def root_p(lo, hi):
    """doc rule: span decades -> root 1/8, 1/4, 1/2, 1."""
    d = max(torch.log10(hi / lo.clamp_min(1e-30)).item(), 1.0)
    return float(2 ** min(3, max(0, int(torch.log2(torch.tensor(d))
                                            .ceil().item()))))


def segroot_quant(W, nseg, use_q, use_root):
    """W bf16 -> recon bf16, bpw (always 6.0)."""
    w = W.float()
    aw = w.abs()
    hi = aw.max()
    if use_q:
        qs = torch.quantile(aw.reshape(-1).float(),
                            torch.linspace(0, 1, nseg + 1,
                                           device=w.device)[1:-1])
        bounds = torch.cat([torch.zeros(1, device=w.device), qs,
                            torch.ones(1, device=w.device)]) * hi
        bounds = torch.unique(bounds)
    else:
        bounds = hi * torch.logspace(0, -3 * (nseg - 1), nseg,
                                     device=w.device).flip(0)
        bounds = torch.cat([torch.zeros(1, device=w.device), bounds])
    if bounds.numel() < nseg + 1:
        bounds = torch.linspace(0, hi, nseg + 1, device=w.device)
    bits_seg = max(1, (nseg - 1).bit_length())
    bits_lvl = 6 - 1 - bits_seg
    L = (1 << bits_lvl) - 1
    nb = bounds.numel() - 1
    recon = torch.zeros_like(w)
    bits_used = 1 + bits_seg + bits_lvl
    for s in range(nb):
        lo, hh = bounds[s], bounds[s + 1]
        mk = (aw >= lo) & (aw < hh) if s < nb - 1 else \
            (aw >= lo) & (aw <= hh)
        if not mk.any():
            continue
        p = root_p(lo, hh) if use_root else 1.0
        ylo = float(lo.item()) ** (1.0 / p) if lo > 0 else 0.0
        yhi = float(hh.item()) ** (1.0 / p)
        y = aw[mk].clamp(lo, hh) ** (1.0 / p)
        q = torch.round((y - ylo) / (yhi - ylo + 1e-30) * L) \
            .clamp(0, L)
        yq = ylo + q / L * (yhi - ylo)
        rec = yq ** p
        recon[mk] = rec * torch.sign(w[mk])
    return recon.to(torch.bfloat16), bits_used


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
        ('4seg-decade-root', 4, False, True),
        ('4seg-quant-root', 4, True, True),
        ('4seg-quant-noroot', 4, True, False),
        ('8seg-quant-root', 8, True, True),
        ('8seg-quant-noroot', 8, True, False),
    ]
    for cname, nseg, use_q, use_root in CONFIGS:
        t0 = time.time()
        errs = []
        snrs = []
        for name, mod in targets:
            W = mod.weight.data
            W2, bpw = segroot_quant(W, nseg, use_q, use_root)
            w = orig[name].float().cuda()
            r = W2.float().cuda()
            errs.append(((w - r).norm() / w.norm()).item())
            snrs.append(10 * torch.log10(
                w.var() / ((w - r) ** 2).mean()).item())
            mod.weight.data = W2
        ppl = eval_ppl(m, tok, texts)
        print(f'[{cname}] bpw=6.00 rel_err={sum(errs)/len(errs):.4f} '
              f'SNR={sum(snrs)/len(snrs):.1f}dB '
              f'ppl={ppl:.2f} ({time.time()-t0:.0f}s)', flush=True)
        for name, mod in targets:
            mod.weight.data = orig[name].cuda()
        gc.collect()
        torch.cuda.empty_cache()
    print('refs: bf16 56.02 | GSQ 5.5bpw ~57.9 | UDCQ 6bpw 58.06 | '
          'int8 8bpw 56.33')


if __name__ == '__main__':
    main()
