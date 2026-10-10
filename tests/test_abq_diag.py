# -*- coding: utf-8 -*-
"""Diagnose ABQ error vs ig32 on the same 1B matrices."""
import sys
sys.path.insert(0, r'E:\IXRUN')
import pandas  # noqa: F401
import torch
from transformers import AutoModelForCausalLM
from ixrun.config import MODEL_PATH
from ixrun.linear import iter_quantizable_linears
from benchmarks.abq_quantize import abq_train_books, abq_quantize

m = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, torch_dtype=torch.bfloat16, trust_remote_code=True)
m.eval().cuda()
targets = list(iter_quantizable_linears(m))
Gs = [mod.weight.data.float().reshape(-1, 32) for _, mod in targets]
books = abq_train_books(Gs, K=16, iters=6, n_books=16, verbose=False)
del Gs
torch.cuda.empty_cache()

import gc
tot_e = 0.0
for name, mod in targets:
    W = mod.weight.data.float()
    Wq, sel, codes, s, bpw = abq_quantize(W, books)
    e = ((Wq.float() - W).norm() / W.norm()).item()
    tot_e += e * W.numel()
    if name in ('model.layers.0.self_attn.q_proj',
                'model.layers.0.mlp.down_proj'):
        d = (Wq.float() - W)
        print(f'{name}: rel {e:.4f} | err rms {d.pow(2).mean().sqrt().item():.5f} '
              f'| w rms {W.pow(2).mean().sqrt().item():.5f} '
              f'| sel hist {torch.bincount(sel, minlength=16).tolist()}',
              flush=True)
nw = sum(mod.weight.numel() for _, mod in targets)
print(f'overall rel {tot_e/nw:.4f}', flush=True)
# ig32 ref (from saved numbers): same-model rel ~0.026
print('ref: ig32(32,16) params rel ~0.026', flush=True)
