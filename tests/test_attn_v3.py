# -*- coding: utf-8 -*-
"""attn_b_v3 (split-K) gate: vs torch fp32 reference at short and long pos."""
import sys
sys.path.insert(0, r'E:\IXRUN')
import pandas  # noqa
import torch

from ixrun.cpp_engine_27b import _build_ext
ext = _build_ext()

torch.manual_seed(0)
NH, NKV, HD, CTX = 24, 4, 256, 4096
for pos in (50, 512, 3800, 4095):
    q = torch.randn(NH * HD, device='cuda') * 0.5
    kv = (torch.randn(8, CTX, HD, device='cuda') * 0.5).to(torch.bfloat16)
    out = ext.attn_v3_test(q, kv, NH, NKV, HD, CTX, pos).view(NH, HD)
    ref = torch.empty(NH, HD, device='cuda')
    scale = HD ** -0.5
    for h in range(NH):
        kvh = h // (NH // NKV)
        k = kv[kvh, :pos + 1].float()
        v = kv[NKV + kvh, :pos + 1].float()
        sc = (q.view(NH, HD)[h] @ k.T) * scale
        w = torch.softmax(sc, dim=-1)
        ref[h] = w @ v
    rel = ((out - ref).abs().max() / ref.abs().max()).item()
    print(f'pos {pos:5d}: max rel err {rel:.3e}', flush=True)
    assert rel < 1e-4, f'gate failed at pos {pos}'
print('PASS', flush=True)
