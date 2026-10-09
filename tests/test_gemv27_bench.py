# -*- coding: utf-8 -*-
"""Per-shape UDCQ GEMV microbench (no init27, no model load)."""
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
'''
ext = load_inline(name='ixrun_cpp_gemvbench', cpp_sources=[proto],
                  cuda_sources=[src, src27],
                  functions=['udcq_gemv_out'],
                  extra_cuda_cflags=['-O3', '--use_fast_math',
                                     '-allow-unsupported-compiler'],
                  verbose=False)
BLOB = r'E:\IXRUN\experiments\qwen38_udcq\q38_blob.pt'
blob = torch.load(BLOB, map_location='cpu', mmap=True, weights_only=True)
cb = blob['codebook'].float().cuda()

SHAPES = [
    ('attn.q [12288,5120]', 'model.layers.3.self_attn.q_proj', 12288, 5120),
    ('attn.k [1024,5120]', 'model.layers.3.self_attn.k_proj', 1024, 5120),
    ('attn.o [5120,6144]', 'model.layers.3.self_attn.o_proj', 5120, 6144),
    ('gdn.qkv [10240,5120]', 'model.layers.0.linear_attn.in_proj_qkv', 10240, 5120),
    ('gdn.z [6144,5120]', 'model.layers.0.linear_attn.in_proj_z', 6144, 5120),
    ('gdn.b [48,5120]', 'model.layers.0.linear_attn.in_proj_b', 48, 5120),
    ('gdn.out [5120,6144]', 'model.layers.0.linear_attn.out_proj', 5120, 6144),
    ('mlp.gate [17408,5120]', 'model.layers.0.mlp.gate_proj', 17408, 5120),
    ('mlp.down [5120,17408]', 'model.layers.0.mlp.down_proj', 5120, 17408),
    ('lm_head [248320,5120]', 'lm_head', 248320, 5120),
]

tot_b = 0.0
tot_t = 0.0

# ---- numeric verification vs torch dequant reference (gate rows 0-63) ----
p = blob['layers']['model.layers.0.mlp.gate_proj']
idx = p['idx'].cuda(); sign = p['sign'].cuda(); scale = p['scale'].cuda()
in_f = 5120
x = torch.randn(in_f, device='cuda')
y_full = ext.udcq_gemv_out(x, idx, sign, scale, cb, 17408, in_f, 16)
NR = 64
i64 = idx[:NR * (in_f // 2)].cuda().long().view(NR, in_f // 2)
nib = torch.empty(NR, in_f, dtype=torch.long, device='cuda')
nib[:, 0::2] = i64 & 0xF
nib[:, 1::2] = i64 >> 4
sc = scale[:NR * (in_f // 16)].float().cuda().view(NR, in_f // 16)
scf = sc.repeat_interleave(16, dim=1)
sg = sign[:NR * (in_f // 32)].cuda().view(NR, in_f // 32)
bits = (sg.unsqueeze(-1) >> torch.arange(
    32, device='cuda', dtype=torch.int64)) & 1
sgn = bits.reshape(NR, -1).float() * 2 - 1
Wref = cb[nib] * scf * sgn
yref = Wref @ x
relerr = ((y_full[:NR] - yref).abs() / (yref.abs() + 1e-3)).max().item()
print(f'GEMV v2 numeric check vs torch dequant: max rel err {relerr:.2e}',
      flush=True)
for name, key, of, inf in SHAPES:
    p = blob['layers'][key]
    idx = p['idx'].cuda()
    sign = p['sign'].cuda()
    scale = p['scale'].cuda()
    x = torch.randn(inf, device='cuda')
    for _ in range(3):
        y = ext.udcq_gemv_out(x, idx, sign, scale, cb, of, inf, 16)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(10):
        y = ext.udcq_gemv_out(x, idx, sign, scale, cb, of, inf, 16)
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / 10
    by = idx.numel() + sign.numel() * 4 + scale.numel() * 2
    tot_b += by
    tot_t += dt
    print(f'{name:24} {dt*1e6:9.1f}us  {by/dt/1e9:7.0f} GB/s', flush=True)
# 27B per-token GEMV budget
n_gdn, n_attn = 48, 16
per_tok = (n_gdn * (39.3 + 23.6 + 0.18 + 0.18 + 23.6 + 66.9 + 66.9 + 66.9)
           + n_attn * (47.2 + 3.9 + 3.9 + 23.6 + 66.9 + 66.9 + 66.9)
           + 953) * 1e6
print(f'\nper-token weight bytes ~{per_tok/1e9:.1f}GB', flush=True)
r = 12288 + 1024 + 5120 + 10240 + 6144 + 48 + 5120 + 17408 + 5120 + 248320
print(f'rough avg GB/s {tot_b/tot_t/1e9:.0f}', flush=True)
print(f'extrapolated GEMV ms/token {per_tok/(tot_b/tot_t)/1e3:.1f}', flush=True)
