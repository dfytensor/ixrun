# -*- coding: utf-8 -*-
"""27B GSQ kernel speed probe: gs_gemv_cuda (smem-x) vs udcq v3 single,
same real matrix, plus the GSQ pack error on this shape."""
import sys, time, json
sys.path.insert(0, r'E:\IXRUN')
import pandas  # noqa
import torch

SHAPE = (17408, 5120)   # gate_proj
KEY = 'model.language_model.layers.0.mlp.gate_proj.weight'

# real bf16 weights
from safetensors import safe_open
MODEL = r'E:\models\Qwen3.8-27B'
index = json.load(open(MODEL + r'\model.safetensors.index.json',
                       encoding='utf-8'))['weight_map']
with safe_open(MODEL + '\\' + index[KEY], framework='pt') as f:
    W = f.get_tensor(KEY).float()
of, inf = SHAPE
assert tuple(W.shape) == SHAPE

# ---- GSQ pack (1B gs_pack; per-matrix kmeans) ----
from benchmarks.gsq_runtime import gs_pack, gs_decode_ref
t0 = time.time()
pk = gs_pack(W)
pk['codes5'] = pk['codes5'].cuda()
pk['cb'] = pk['cb'].cuda()
pk['s_i8'] = pk['s_i8'].cuda()
print(f'gs_pack {time.time()-t0:.0f}s', flush=True)
Wg = gs_decode_ref(pk).float()
e_gs = ((Wg.cuda() - W.cuda()).norm() / W.cuda().norm()).item()
print(f'GSQ weight rel {e_gs:.4f} | bpw 5.50', flush=True)

# ---- UDCQ pack from the existing blob (dequant) ----
blob = torch.load(r'E:\IXRUN\experiments\qwen38_udcq\q38_blob.pt',
                  map_location='cpu', mmap=True, weights_only=True)
p = blob['layers']['model.layers.0.mlp.gate_proj']
cb = blob['codebook'].float().cuda()
idx = p['idx'].cuda().long().view(of, inf // 2)
nib = torch.empty(of, inf, dtype=torch.long, device='cuda')
nib[:, 0::2] = idx & 0xF
nib[:, 1::2] = idx >> 4
usg = p['sign'].int().view(of, inf // 32, 1).cuda()
ubit = (usg >> torch.arange(32, device='cuda')) & 1
usgn = ubit.reshape(of, inf).float() * 2 - 1
usc = p['scale'].float().view(of, inf // 16).cuda()
usc = usc.repeat_interleave(16, dim=1)
Wu = cb[nib] * usc * usgn
e_u = ((Wu - W.cuda()).norm() / W.cuda().norm()).item()
print(f'UDCQ weight rel {e_u:.4f} | bpw 6.00', flush=True)
del Wu, nib
torch.cuda.empty_cache()

# ---- kernel benches ----
from experiments.gsq_gemv_cuda.gsq_gemv_cuda import gs_gemv_cuda
from experiments.udcq_gemv_cuda.udcq_gemv_cuda import (cuda_gemv,
                                                       install_codebook)
install_codebook(cb)
x = torch.randn(inf, dtype=torch.bfloat16, device='cuda')

def bench(name, fn, n=30):
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / n * 1e3
    print(f'{name}: {dt:.3f}ms/call', flush=True)
    return dt

idx_g = p['idx'].cuda(); sign_g = p['sign'].cuda(); scale_g = p['scale'].cuda()
t_gs = bench('gs_gemv T=1 ', lambda: gs_gemv_cuda(x, pk))
t_ud = bench('udcq v3 T=1 ', lambda: cuda_gemv(
    x, idx_g, sign_g, scale_g, cb, of, inf))
print(f'GSQ/UDCQ time ratio {t_gs/t_ud:.2f} '
      f'(bytes ratio 5.50/6.00 = 0.917)', flush=True)
