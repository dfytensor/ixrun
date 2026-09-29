# -*- coding: utf-8 -*-
"""int4 + GMM responsibility correction layer on MiniCPM5-1B.

Reproduces the external recipe: int4 RTN row-wise quantize a linear,
then fit a low-rank GMM-gated correction head  corr(x) = softmax(...)·C^T
approximating the quantization error response  E @ x  (E = W - W_int4).
Reports relative output error: int8 / int4 / int4+GMM(K=64,r=32).
"""
import sys
import time

sys.path.insert(0, r'E:\IXRUN')
import numpy as np
import pandas  # noqa: F401
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM

from ixrun.config import MODEL_PATH, DATASET_CACHE
from ixrun.eval_utils import load_wikitext

DEV = 'cuda'
K = 64
R_IN = 32
TARGET = 'model.layers.12.self_attn.q_proj'
STEPS = 2000
BS = 1024


def quant4_rows(w):
    qmax = 7
    s = w.abs().max(dim=1, keepdim=True).values.clamp_min(1e-12) / qmax
    return (w / s).round().clamp(-qmax, qmax) * s


def rel_err(y, y_ref):
    return ((y - y_ref).norm() / y_ref.norm()).item()


class GMMCorrect(nn.Module):
    def __init__(self, M0, s0, r_in, K, D_out):
        super().__init__()
        Ur, Sr, Vh = torch.linalg.svd(M0, full_matrices=False)
        rr = min(r_in, Sr.shape[0])
        self.U = nn.Parameter(
            (Ur[:, :rr] * Sr[:rr].unsqueeze(0)).contiguous())
        self.V = nn.Parameter(
            (Vh[:rr] * Sr[:rr].unsqueeze(1).sqrt()).contiguous())
        self.log_tau = nn.Parameter(torch.log(torch.full(
            (1, K), max(float(Sr[rr - 1]), 0.5))))
        self.b = nn.Parameter(torch.zeros(K))
        self.s = nn.Parameter(s0.clone())
        self.C = nn.Parameter(torch.zeros(D_out, K))

    def forward(self, x):
        xf = x.float()
        h = (xf / self.s.float().clamp_min(1e-6)) @ self.V.float().t()
        logits = h @ self.U.float().t() / \
            self.log_tau.float().exp().clamp(1e-4, 3000) + self.b
        r = torch.softmax(logits, -1)
        return r @ self.C.float().t()


def main():
    tok = AutoTokenizer.from_pretrained(MODEL_PATH,
                                        trust_remote_code=True)
    texts = load_wikitext(cache_dir=DATASET_CACHE)
    m = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, dtype=torch.bfloat16,
        trust_remote_code=True).eval().to('cuda')

    # ---- collect activations at the target linear ----
    cap = []
    mod = dict(m.named_modules())[TARGET]
    W = mod.weight.data.float().cuda()          # [D, d]
    D, d = W.shape

    def hook(md, inp):
        t = inp[0].detach()
        cap.append(t.reshape(-1, t.shape[-1]).cpu())
    h = mod.register_forward_pre_hook(hook)
    big = '\n'.join(texts)
    chunks = [big[i:i + 8000] for i in range(0, len(big), 8000)][:64]
    n_txt = 0
    with torch.no_grad():
        for t in chunks:
            ids = tok(t, return_tensors='pt').input_ids[:, :1024]
            if ids.shape[1] < 64:
                continue
            m(ids.cuda())
            n_txt += 1
            if n_txt >= 64:
                break
    h.remove()
    X = torch.cat(cap).float().cuda()           # [N, d]
    print(f'activations: {tuple(X.shape)} from {n_txt} texts', flush=True)

    # ---- quantize + error responses ----
    W4 = quant4_rows(W)
    E = W - W4                                  # [D, d]
    s8 = W.abs().max(dim=1, keepdim=True).values.clamp_min(1e-12) / 127
    W8 = (W / s8).round().clamp(-127, 127) * s8
    torch.cuda.empty_cache()
    NS = min(65536, X.shape[0])
    sel = torch.randperm(X.shape[0], device=DEV)[:NS]
    Xs = X[sel]
    Y_ref = Xs @ W.t()
    t0 = time.time()
    EXs = Xs @ E.t()                            # [NS, D] targets
    print(f'targets ready ({time.time()-t0:.0f}s)', flush=True)

    e8 = rel_err(Xs @ W8.t(), Y_ref)
    e4 = rel_err(Xs @ W4.t(), Y_ref)
    print(f'int8 RTN rel_err = {e8:.4f}')
    print(f'int4 RTN rel_err = {e4:.4f}', flush=True)

    # ---- kmeans on activations (GPU, seeded) ----
    g = torch.Generator(device=DEV)
    g.manual_seed(42)
    C0 = Xs[torch.randperm(NS, device=DEV, generator=g)[:K]]
    for _ in range(15):
        a = torch.cdist(Xs[::4], C0).argmin(1)
        for j in range(K):
            mk = a == j
            if mk.any():
                C0[j] = Xs[::4][mk].mean(0)
    afull = torch.cdist(Xs, C0).argmin(1)
    x_mean = torch.stack([
        Xs[afull == j].mean(0) if (afull == j).any() else C0[j]
        for j in range(K)])

    gmm = GMMCorrect(C0, X.std(0).clamp_min(1e-3) + 1e-6, R_IN, K, D).to(DEV)
    with torch.no_grad():
        for k in range(K):
            gmm.C.data[:, k] = E @ x_mean[k]
    opt = torch.optim.AdamW(gmm.parameters(), lr=3e-3)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=STEPS,
                                                     eta_min=1e-5)
    t0 = time.time()
    for step in range(STEPS):
        i = torch.randint(0, NS, (BS,), device=DEV)
        xb = Xs[i]
        tgt = EXs[i]                            # [BS, D]
        loss = ((gmm(xb) - tgt) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(gmm.parameters(), 0.5)
        opt.step()
        sch.step()
    print(f'trained {STEPS} steps in {time.time()-t0:.0f}s '
          f'(final loss {loss.item():.5f})', flush=True)

    with torch.no_grad():
        parts = []
        for i0 in range(0, NS, 16384):
            parts.append(gmm(Xs[i0:i0 + 16384]))
        corr = torch.cat(parts)
    e4g = rel_err(Xs @ W4.t() + corr, Y_ref)
    st = (K * d // 2 + K * R_IN // 2 + D * K * 2 + d * 2 + K * 2 * 3)
    print(f'int4+GMM (K={K}, r={R_IN}) rel_err = {e4g:.4f} '
          f'({(1 - e4g / e4) * 100:.1f}% better than int4)')
    print(f'correction storage: {st/1024:.0f} KB '
          f'({st / (W.numel() * 2) * 100:.1f}% of fp16 W)')


if __name__ == '__main__':
    main()
