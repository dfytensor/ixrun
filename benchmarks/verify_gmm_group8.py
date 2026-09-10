# -*- coding: utf-8 -*-
"""Bit-exact check: GMM pack (group=8 vs 16) decode vs reference.

Rule (AGENTS.md): any new kernel constexpr variant must pass a
decode-vs-reference bit-exact check before touching a model.
"""
import sys
sys.path.insert(0, r'E:\IXRUN')
import pandas
import torch

from benchmarks.gmm_stream_minicpm5 import gmm_pack, fit_bayesian_gmm
from ixrun.udcq import decode_udcq_triton

torch.manual_seed(0)
for out_f, in_f in [(2048, 4096), (5120, 5120), (17408, 5120)]:
    W = torch.randn(out_f, in_f) * 0.02
    wf = W.reshape(-1)
    if wf.numel() > 3_000_000:
        wf = wf[torch.randint(0, wf.numel(), (3_000_000,))]
    mu, _, _ = fit_bayesian_gmm(wf, K=16, iters=20)
    for g in (16, 8):
        packed = gmm_pack(W, mu, group=g)
        # kernel decode path
        w_k = decode_udcq_triton(packed, 'cuda').float().cpu()
        # reference: nibble unpack + mu[idx] * scale[group]
        b = packed['idx']
        N = packed['N']
        idx = torch.empty(N, dtype=torch.long)
        idx[0::2] = (b & 0x0F).long()[:len(idx[0::2])]
        idx[1::2] = (b >> 4).long()[:len(idx[1::2])]
        sc = packed['scale'].float()
        ref = mu[idx] * sc[torch.arange(N) // g]
        ref = ref[:w_k.numel()].reshape(w_k.shape)
        err = (w_k - ref).abs().max().item()
        print(f'{out_f}x{in_f} g={g}: kernel-vs-ref dmax={err:.8f} '
              f'| scale len={len(sc)} expect={-(-N//g)} '
              f'| fp16-subnormal={((sc.abs() < 6.1e-5) & (sc != 0)).sum().item()}',
              flush=True)
