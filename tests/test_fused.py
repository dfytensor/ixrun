"""Validate + benchmark the fused decode+GEMV kernel vs decode+F.linear."""
import sys, time
sys.setrecursionlimit(10000)
import torch
import torch.nn.functional as F
from ixrun.quantize import int8x_quantize
from ixrun.triton_kernels import decode_weight_triton
from ixrun.fused import fused_gemv, compute_row_prefixes

dev = "cuda"
torch.manual_seed(0)

# realistic heavy-tail weights + the exact layer shapes we deploy on
shapes = [(1536, 1536), (4608, 1536), (1536, 4608), (5120, 5120), (17408, 5120), (5120, 17408)]

for out_f, in_f in shapes:
    base = torch.randn(out_f, in_f) * 0.005
    n_out = int(out_f * in_f * 0.04)
    flat = base.view(-1)
    idx = torch.randperm(flat.numel())[:n_out]
    flat[idx] = torch.randn(n_out) * 0.05
    w = base.bfloat16().to(dev)

    p = int8x_quantize(w, (3, 5, 8))
    w_dec = decode_weight_triton(p, device=dev)  # bf16 weight, reference path
    x = torch.randn(in_f, dtype=torch.bfloat16, device=dev)

    y_ref = F.linear(x, w_dec)
    b1, b2 = p["bitmaps"]; l1, l2, l3 = p["streams"]
    q1, q2 = compute_row_prefixes(p)
    y_fused = fused_gemv(x, b1.to(dev), b2.to(dev), l1.to(dev), l2.to(dev), l3.to(dev),
                         q1.to(dev), q2.to(dev), p["scale"].to(dev), out_f, in_f)

    diff = (y_fused.float() - y_ref.float()).abs()
    rel = diff.max() / y_ref.float().abs().max()
    print(f"  [{out_f}x{in_f}] max_abs={diff.max():.3e} rel={rel:.2e} "
          f"{'OK' if rel < 5e-3 else 'FAIL'}")

# benchmark: fused vs (decode + linear) on a Qwen-sized layer
out_f, in_f = 17408, 5120
base = torch.randn(out_f, in_f) * 0.005
flat = base.view(-1); idx = torch.randperm(flat.numel())[:int(out_f*in_f*0.04)]
flat[idx] = torch.randn(len(idx)) * 0.05
w = base.bfloat16().to(dev)
p = int8x_quantize(w, (3, 5, 8))
b1, b2 = p["bitmaps"]; l1, l2, l3 = p["streams"]
q1, q2 = compute_row_prefixes(p)
gb = [t.to(dev) for t in (b1, b2, l1, l2, l3, q1, q2, p["scale"])]
x = torch.randn(in_f, dtype=torch.bfloat16, device=dev)
w_dec = decode_weight_triton(p, device=dev)

for _ in range(10): fused_gemv(x, *gb, out_f, in_f)
torch.cuda.synchronize(); t0 = time.time()
for _ in range(100): fused_gemv(x, *gb, out_f, in_f)
torch.cuda.synchronize(); t_f = (time.time()-t0)/100*1000

for _ in range(5): decode_weight_triton(p, device=dev)
torch.cuda.synchronize(); t0 = time.time()
for _ in range(100):
    wd = decode_weight_triton(p, device=dev); F.linear(x, wd)
torch.cuda.synchronize(); t_c = (time.time()-t0)/100*1000

print(f"\n  layer {out_f}x{in_f}: fused_gemv={t_f:.3f}ms  decode+linear={t_c:.3f}ms  speedup={t_c/t_f:.2f}x")
