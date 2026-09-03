# -*- coding: utf-8 -*-
"""Bit-exactness test for multi-token GEMV (udcq_fused_gemv_mt).

For T in {2,4,8}: y_mt[t] must equal udcq_fused_gemv(x[t]) BIT-EXACTLY
(same decode walk + same accumulation expression per token).
Run: $env:HF_HUB_OFFLINE='1'; python -X utf8 experiments/qwen38_udcq/test_mt_gemv.py
"""
import torch

from ixrun.udcq import (udcq_fit_codebook, udcq_quantize, udcq_fused_gemv,
                        udcq_fused_gemv_mt, UDCQ_G)

torch.manual_seed(0)
dev = 'cuda'

# representative shapes (Qwen3.8-27B linears) + odd-ish sizes
SHAPES = [(3072, 4096), (4096, 4096), (12288, 4096), (4096, 12288),
          (1536, 2048), (2048, 2048)]

for out_f, in_f in SHAPES:
    W = (torch.randn(out_f, in_f, device=dev) * 0.05).to(torch.bfloat16)
    cb = udcq_fit_codebook(W.float().cpu(), nlev=16, g=UDCQ_G)
    packed = udcq_quantize(W.float().cpu(), cb, g=UDCQ_G)
    # stage packed on GPU
    for k in ('idx', 'scale', 'sign_packed', 'codebook'):
        if isinstance(packed[k], torch.Tensor):
            packed[k] = packed[k].cuda()
    for T in (2, 4, 8):
        x = (torch.randn(T, in_f, device=dev) * 0.5).to(torch.bfloat16)
        y_ref = torch.stack([udcq_fused_gemv(
            x[t], packed['idx'], packed['sign_packed'], packed['scale'],
            packed['codebook'], out_f, in_f, g=UDCQ_G) for t in range(T)], dim=0)
        y_mt = udcq_fused_gemv_mt(
            x, packed['idx'], packed['sign_packed'], packed['scale'],
            packed['codebook'], out_f, in_f, g=UDCQ_G)
        same = torch.equal(y_ref, y_mt)
        ndiff = (y_ref != y_mt).sum().item()
        dmax = (y_ref.float() - y_mt.float()).abs().max().item()
        tag = 'OK' if same else 'FAIL'
        print(f'[{tag}] {out_f}x{in_f} T={T}: bit-exact={same} '
              f'ndiff={ndiff}/{y_ref.numel()} dmax={dmax:.6f}')
        assert same, f'NOT bit-exact at {out_f}x{in_f} T={T}'

# timing sanity: mt cost ~= single-token cost
import time
out_f, in_f = 4096, 4096
W = (torch.randn(out_f, in_f, device=dev) * 0.05).to(torch.bfloat16)
cb = udcq_fit_codebook(W.float().cpu(), nlev=16, g=UDCQ_G)
packed = udcq_quantize(W.float().cpu(), cb, g=UDCQ_G)
for k in ('idx', 'scale', 'sign_packed', 'codebook'):
    if isinstance(packed[k], torch.Tensor):
        packed[k] = packed[k].cuda()
for T in (1, 2, 4, 8):
    x = (torch.randn(max(T, 1), in_f, device=dev) * 0.5).to(torch.bfloat16)
    def go():
        if T == 1:
            udcq_fused_gemv(x[0], packed['idx'], packed['sign_packed'],
                            packed['scale'], packed['codebook'], out_f, in_f)
        else:
            udcq_fused_gemv_mt(x, packed['idx'], packed['sign_packed'],
                               packed['scale'], packed['codebook'], out_f, in_f)
    for _ in range(10):
        go()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(50):
        go()
    torch.cuda.synchronize()
    dt = (time.time() - t0) / 50 * 1e3
    print(f'timing T={T}: {dt:.3f} ms')

print('ALL PASS')
