"""Validate split-K GEMV + benchmark wide-layer improvement."""
import sys, time
sys.setrecursionlimit(10000)
import torch
import torch.nn.functional as F
from ixrun.quantize import int8x_quantize
from ixrun.triton_kernels import decode_weight_triton
from ixrun.fused import fused_gemv, compute_row_prefixes, _pick_split

dev = "cuda"
torch.manual_seed(0)

for out_f, in_f in [(5120, 17408), (1536, 4608), (5120, 6144)]:
    split = _pick_split(in_f) if out_f < in_f else 1
    base = torch.randn(out_f, in_f) * 0.005
    n_out = int(out_f * in_f * 0.04)
    flat = base.view(-1); idx = torch.randperm(flat.numel())[:n_out]
    flat[idx] = torch.randn(n_out) * 0.05
    w = base.bfloat16().to(dev)
    p = int8x_quantize(w, (3, 5, 8))
    w_dec = decode_weight_triton(p, device=dev)
    x = torch.randn(in_f, dtype=torch.bfloat16, device=dev)
    y_ref = F.linear(x, w_dec)

    if split > 1:
        chunk = in_f // split
        q1, q2 = compute_row_prefixes(p, chunk)
        y32 = torch.zeros(out_f, dtype=torch.float32, device=dev)
        fused_gemv(x, p["bitmaps"][0].to(dev), p["bitmaps"][1].to(dev),
                   p["streams"][0].to(dev), p["streams"][1].to(dev),
                   p["streams"][2].to(dev), q1.to(dev), q2.to(dev),
                   p["scale"].to(dev), out_f, in_f, chunk=chunk, y32=y32)
        y = y32.to(torch.bfloat16)
        print(f"[{out_f}x{in_f}] split={split} chunk={chunk} rel_err="
              f"{(y.float()-y_ref.float()).abs().max()/y_ref.float().abs().max():.2e}")
    else:
        print(f"[{out_f}x{in_f}] no split applicable")

# bench: down_proj shape with vs without split
out_f, in_f = 5120, 17408
base = torch.randn(out_f, in_f) * 0.005
flat = base.view(-1); idx = torch.randperm(flat.numel())[:int(out_f*in_f*0.04)]
flat[idx] = torch.randn(len(idx)) * 0.05
w = base.bfloat16().to(dev)
p = int8x_quantize(w, (3, 5, 8))
b1, b2 = p["bitmaps"]; l1, l2, l3 = p["streams"]
x = torch.randn(in_f, dtype=torch.bfloat16, device=dev)

# split path
chunk = in_f // 2
q1c, q2c = compute_row_prefixes(p, chunk)
gbs = [t.to(dev) for t in (b1, b2, l1, l2, l3, q1c, q2c, p["scale"])]
y32 = torch.zeros(out_f, dtype=torch.float32, device=dev)
def call_split():
    y32.zero_()
    fused_gemv(x, *gbs, out_f, in_f, chunk=chunk, y32=y32)
for _ in range(20): call_split()
torch.cuda.synchronize(); t0 = time.time()
for _ in range(300): call_split()
torch.cuda.synchronize(); t_split = (time.time()-t0)/300*1e3

# old single-kernel path
q1, q2 = compute_row_prefixes(p)
gb = [t.to(dev) for t in (b1, b2, l1, l2, l3, q1, q2, p["scale"])]
for _ in range(20): fused_gemv(x, *gb, out_f, in_f)
torch.cuda.synchronize(); t0 = time.time()
for _ in range(300): fused_gemv(x, *gb, out_f, in_f)
torch.cuda.synchronize(); t_one = (time.time()-t0)/300*1e3

rate_s = out_f*in_f/t_split/1e6
rate_o = out_f*in_f/t_one/1e6
print(f"\n5120x17408: single={t_one:.3f}ms ({rate_o:.0f}G/s)  split2={t_split:.3f}ms ({rate_s:.0f}G/s)  speedup={t_one/t_split:.2f}x")
