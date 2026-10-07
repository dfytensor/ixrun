# -*- coding: utf-8 -*-
"""Spot-check: decode blob in_proj_b/a packs vs HF weights.
Tiny matrices (48x5120) — a packing/scale-assumption error shows
as ~0 cosine; healthy UDCQ tier shows rel-err ~5e-3..5e-2."""
import sys
sys.path.insert(0, r'E:\IXRUN')
import pandas  # noqa: F401
import torch
from ixrun.config import QWEN38_PATH

BLOB = r'E:\IXRUN\experiments\qwen38_udcq\q38_blob.pt'
blob = torch.load(BLOB, map_location='cpu', mmap=True,
                  weights_only=True)
cb = blob['codebook'].float()
GROUP = 16

from transformers import AutoModelForCausalLM
m = AutoModelForCausalLM.from_pretrained(
    QWEN38_PATH, dtype=torch.bfloat16, low_cpu_mem_usage=True,
    device_map='cpu')
la = m.model.layers[0].linear_attn

for nm, w_hf in [('in_proj_b', la.in_proj_b.weight.data.float()),
                 ('in_proj_a', la.in_proj_a.weight.data.float())]:
    p = blob['layers'][f'model.layers.0.linear_attn.{nm}']
    idx, sign, scale = p['idx'], p['sign'], p['scale']
    N = w_hf.numel()
    b = idx.long()
    nib = torch.stack([b & 0x0F, (b >> 4) & 0x0F], 1).reshape(-1)
    bit = ((sign.long().unsqueeze(1) >> torch.arange(32))
           & 1).reshape(-1)[:N]
    W = (cb.double()[nib] * scale.double()
         .repeat_interleave(GROUP)
         * (bit * 2.0 - 1.0)).reshape(48, 5120)
    e = ((W - w_hf.double()).norm() / w_hf.double().norm()).item()
    cos = torch.nn.functional.cosine_similarity(
        W.reshape(-1), w_hf.double().reshape(-1), dim=0).item()
    print(f'{nm}: rel-err {e:.3e} cosine {cos:.4f} '
          f'(N={N}, scale {tuple(scale.shape)})', flush=True)
