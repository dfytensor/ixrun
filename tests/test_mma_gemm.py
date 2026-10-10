# -*- coding: utf-8 -*-
"""Fused mma decode-GEMM numeric gate vs the bf16 reference."""
import os
os.environ['IXRUN_PREBUILT'] = '1'
import sys, time
sys.path.insert(0, r'E:\IXRUN')
import pandas  # noqa
import torch
from ixrun.cpp_engine_27b import _build_ext

ext = _build_ext()
ext.udcq_set_uls(1)
ub = torch.load(r'F:\models\qwen38_uls_blob.pt', map_location='cpu',
                mmap=True, weights_only=True)
cb = ub['codebook'].float().cuda()


def ref_weights(key, OF, INF):
    p = ub['layers'][key]
    idx, sg, sc = p['idx'].cuda(), p['sign'].cuda(), p['scale'].cuda()
    buf = sc
    hdr = buf[:, :8].contiguous().view(torch.float32).view(OF, 2)
    ib = buf[:, 8:].float()
    sd = torch.exp2(hdr[:, 0:1] + hdr[:, 1:2] * ib)
    fl = idx.view(OF, INF // 2).long()
    nib = torch.empty(OF, INF, dtype=torch.long, device='cuda')
    nib[:, 0::2] = fl & 0xF
    nib[:, 1::2] = fl >> 4
    sgn = (((sg.view(OF, INF // 32, 1).int()
             >> torch.arange(32, device='cuda')) & 1)
           .reshape(OF, INF).float() * 2 - 1)
    W = cb[nib] * sd.repeat_interleave(16, 1) * sgn
    return idx, sg, sc, W


torch.manual_seed(0)
for name, OF, INF, T in [
        ('gate', 17408, 5120, 256),
        ('o_proj', 5120, 6144, 256),
        ('q edge', 12288, 5120, 128),
        ('b tiny', 48, 5120, 64),
        ('tail T=8', 6144, 5120, 8)]:
    key = ('model.layers.0.mlp.gate_proj' if name == 'gate' else
           'model.layers.0.linear_attn.out_proj' if name == 'o_proj' else
           'model.layers.3.self_attn.q_proj' if name.startswith('q') else
           'model.layers.0.linear_attn.in_proj_b' if name.startswith('b') else
           'model.layers.0.linear_attn.in_proj_z')
    idx, sg, sc, W = ref_weights(key, OF, INF)
    x = (torch.randn(T, INF, device='cuda') * 0.5)
    y = ext.mma_gemm_test(x, idx, sg, sc, cb, T, OF, INF)
    ref = (x.bfloat16().float() @ W.bfloat16().float().t())
    rel = ((y - ref).abs().max() / ref.abs().max()).item()
    print(f'{name:8s} [{OF}x{INF}] T={T}: rel err {rel:.5f}', flush=True)
    assert rel < 2e-2, f'mma gate failed on {name}'
    del W, ref, y
    torch.cuda.empty_cache()

# timing vs the dequant+cublas pair
key = 'model.layers.0.mlp.gate_proj'
idx, sg, sc, W = ref_weights(key, 17408, 5120)
x = torch.randn(256, 5120, device='cuda') * 0.5
wb = W.bfloat16()


def bench(fn, n=20):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n * 1e3


t_mma = bench(lambda: ext.mma_gemm_test(x, idx, sg, sc, cb, 256, 17408, 5120))
t_ref = bench(lambda: torch.matmul(x.bfloat16(), wb.t()))
print(f'mma fused: {t_mma:.2f}ms | bf16 matmul (weights ready): {t_ref:.2f}ms',
      flush=True)
print('PASS', flush=True)
