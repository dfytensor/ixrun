# -*- coding: utf-8 -*-
"""Gate: udcq_gemv_mt4 (T=4, one weight walk) bit-exact vs 4x udcq_gemv_out."""
import sys, time
sys.path.insert(0, r'E:\IXRUN')
import pandas  # noqa: F401
import torch
from torch.utils.cpp_extension import load_inline

src = open(r'E:\IXRUN\ixrun\cpp\engine_v5.cu', encoding='utf-8').read()
src27 = open(r'E:\IXRUN\ixrun\cpp\engine_27b.cu', encoding='utf-8').read()
proto = '''
torch::Tensor udcq_gemv_out(torch::Tensor x, torch::Tensor idx,
    torch::Tensor sign, torch::Tensor scale, torch::Tensor cb,
    int64_t out_f, int64_t in_f, int64_t group);
torch::Tensor udcq_gemv_mt4_out(torch::Tensor x, torch::Tensor idx,
    torch::Tensor sign, torch::Tensor scale, torch::Tensor cb,
    int64_t out_f, int64_t in_f, int64_t group);
'''
ext = load_inline(name='ixrun_cpp_gemvmt', cpp_sources=[proto],
                  cuda_sources=[src, src27],
                  functions=['udcq_gemv_out', 'udcq_gemv_mt4_out'],
                  extra_cuda_cflags=['-O3', '--use_fast_math',
                                     '-allow-unsupported-compiler'],
                  verbose=False)

blob = torch.load(r'E:\IXRUN\experiments\qwen38_udcq\q38_blob.pt',
                  map_location='cpu', mmap=True, weights_only=True)
cb = blob['codebook'].float().cuda()

SHAPES = [
    ('gate [17408,5120]', 'model.layers.0.mlp.gate_proj', 17408, 5120),
    ('down [5120,17408]', 'model.layers.0.mlp.down_proj', 5120, 17408),
    ('out [5120,6144]', 'model.layers.0.linear_attn.out_proj', 5120, 6144),
    ('qkv [10240,5120]', 'model.layers.0.linear_attn.in_proj_qkv', 10240, 5120),
    ('attn.k [1024,5120]', 'model.layers.3.self_attn.k_proj', 1024, 5120),
    ('lm_head [248320,5120]', 'lm_head', 248320, 5120),
]

ok_all = True
for name, key, of, inf in SHAPES:
    p = blob['layers'][key]
    idx, sign, scale = p['idx'].cuda(), p['sign'].cuda(), p['scale'].cuda()
    x4 = torch.randn(4, inf, device='cuda')
    y4 = ext.udcq_gemv_mt4_out(x4, idx, sign, scale, cb, of, inf, 16)
    ys = [ext.udcq_gemv_out(x4[t], idx, sign, scale, cb, of, inf, 16)
          for t in range(4)]
    yref = torch.stack(ys)
    eq = torch.equal(y4, yref)
    maxd = (y4 - yref).abs().max().item()
    ok_all &= eq
    print(f'{name:24} bitexact={eq} maxdiff={maxd:.2e}', flush=True)

print('MT4 GATE:', 'PASS' if ok_all else 'FAIL', flush=True)

# timing on the big shape
p = blob['layers']['model.layers.0.mlp.gate_proj']
idx, sign, scale = p['idx'].cuda(), p['sign'].cuda(), p['scale'].cuda()
x4 = torch.randn(4, 5120, device='cuda')
for _ in range(3):
    ext.udcq_gemv_mt4_out(x4, idx, sign, scale, cb, 17408, 5120, 16)
    for t in range(4):
        ext.udcq_gemv_out(x4[t], idx, sign, scale, cb, 17408, 5120, 16)
torch.cuda.synchronize()
t0 = time.perf_counter()
for _ in range(10):
    ext.udcq_gemv_mt4_out(x4, idx, sign, scale, cb, 17408, 5120, 16)
torch.cuda.synchronize()
dtm = (time.perf_counter() - t0) / 10
t0 = time.perf_counter()
for _ in range(10):
    for t in range(4):
        ext.udcq_gemv_out(x4[t], idx, sign, scale, cb, 17408, 5120, 16)
torch.cuda.synchronize()
dts = (time.perf_counter() - t0) / 10
print(f'gate mt4 {dtm*1e6:.1f}us vs 4x single {dts*1e6:.1f}us '
      f'({dts/dtm:.2f}x)', flush=True)
