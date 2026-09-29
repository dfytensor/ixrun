# -*- coding: utf-8 -*-
"""Full-model int4+GMM-correction ppl on MiniCPM5-1B.

Places the int4+GMM recipe on the model-ppl curve: every quantized
linear gets int4 RTN row-wise weights + a fitted K=64 r=32 correction
head; eval vs pure-int4 and published references.
"""
import gc
import sys
import time

sys.path.insert(0, r'E:\IXRUN')
import pandas  # noqa: F401
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM

from ixrun.config import MODEL_PATH, DATASET_CACHE
from ixrun.eval_utils import eval_ppl, load_wikitext
from ixrun.linear import iter_quantizable_linears
from benchmarks.gmm_correct_minicpm5 import GMMCorrect, quant4_rows

DEV = 'cuda'
K = 64
R_IN = 32
ROWS = 8192          # calib rows per layer (bf16 CPU store)
STEPS = 800
BS = 1024


class Q4GMMLinear(nn.Module):
    def __init__(self, w4, gmm):
        super().__init__()
        self.w4 = w4            # [D, d] bf16 cuda (dequantized)
        self.gmm = gmm

    def forward(self, x):
        return (torch.nn.functional.linear(
            x.to(self.w4.dtype), self.w4)
            + self.gmm(x)).to(torch.bfloat16)


def main():
    tok = AutoTokenizer.from_pretrained(MODEL_PATH,
                                        trust_remote_code=True)
    texts = load_wikitext(cache_dir=DATASET_CACHE)
    m = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, dtype=torch.bfloat16,
        trust_remote_code=True).eval().to(DEV)
    targets = list(iter_quantizable_linears(m))
    print(f'targets: {len(targets)}', flush=True)

    # ---- one calibration pass, hooks on every target ----
    caps = {n: [] for n, _ in targets}
    hooks = [mod.register_forward_pre_hook(
        (lambda nm: (lambda md, inp: caps[nm].append(
            inp[0].detach().reshape(-1, inp[0].shape[-1])
            [:ROWS].cpu())))(name)) for name, mod in targets]
    big = '\n'.join(texts)
    chunks = [big[i:i + 8000] for i in range(0, len(big), 8000)][:8]
    with torch.no_grad():
        for t in chunks:
            m(tok(t, return_tensors='pt').input_ids.cuda())
    for hk in hooks:
        hk.remove()
    Xs = {}
    for name, _ in targets:
        X = torch.cat(caps[name]).to(torch.bfloat16)
        Xs[name] = X[:ROWS]
        del caps[name]
    gc.collect()
    print(f'calib: {len(Xs)} layers x {ROWS} rows', flush=True)

    # ---- fit per-layer int4 + GMM correction ----
    t00 = time.time()
    for li, (name, mod) in enumerate(targets):
        t0 = time.time()
        W = mod.weight.data.float().cuda()
        D, d = W.shape
        W4 = quant4_rows(W)
        E = W - W4
        Xl = Xs[name].float().cuda()                # [ROWS, d]
        EX = Xl @ E.t()                             # [ROWS, D]
        g = torch.Generator(device=DEV)
        g.manual_seed(42)
        C0 = Xl[torch.randperm(ROWS, device=DEV, generator=g)[:K]]
        for _ in range(10):
            a = torch.cdist(Xl[::4], C0).argmin(1)
            for j in range(K):
                mk = a == j
                if mk.any():
                    C0[j] = Xl[::4][mk].mean(0)
        gmm = GMMCorrect(C0, Xl.std(0).clamp_min(1e-3) + 1e-6,
                         R_IN, K, D).to(DEV)
        with torch.no_grad():
            af = torch.cdist(Xl, C0).argmin(1)
            for k in range(K):
                mkk = af == k
                if mkk.any():
                    gmm.C.data[:, k] = E @ Xl[mkk].mean(0)
        opt = torch.optim.AdamW(gmm.parameters(), lr=3e-3)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=STEPS, eta_min=1e-5)
        for step in range(STEPS):
            i = torch.randint(0, ROWS, (BS,), device=DEV)
            loss = ((gmm(Xl[i]) - EX[i]) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(gmm.parameters(), 0.5)
            opt.step()
            sch.step()
        new = Q4GMMLinear(W4.to(torch.bfloat16), gmm)
        parent = m.get_submodule(name.rsplit('.', 1)[0])
        parent._modules[name.rsplit('.', 1)[1]] = new
        del W, W4, E, Xl, EX
        torch.cuda.empty_cache()
        if li % 20 == 0:
            print(f'  [{li+1}/{len(targets)}] {name} '
                  f'({time.time()-t0:.0f}s)', flush=True)
    print(f'all layers fitted ({time.time()-t00:.0f}s)', flush=True)
    del Xs
    gc.collect()

    ppl = eval_ppl(m, tok, texts)
    print(f'\n[int4+GMM full-model] K={K} r={R_IN} steps={STEPS} '
          f'ppl = {ppl:.2f}', flush=True)
    print('refs: bf16 56.02 | GMM 6bpw 57.92 | UDCQ 6bpw 58.06 | '
          'GMM per-16 5.03bpw 60.93')


if __name__ == '__main__':
    main()
