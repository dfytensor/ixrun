"""Try different chunk sizes for 5120x17408 split-K."""
import sys, time
sys.setrecursionlimit(10000)
import torch
import torch.nn.functional as F
from ixrun.quantize import int8x_quantize
from ixrun.triton_kernels import decode_weight_triton
from ixrun.fused import fused_gemv, compute_row_prefixes

dev = "cuda"
torch.manual_seed(0)
out_f, in_f = 5120, 17408
base = torch.randn(out_f, in_f) * 0.005
flat = base.view(-1); idx = torch.randperm(flat.numel())[:int(out_f*in_f*0.04)]
flat[idx] = torch.randn(len(idx)) * 0.05
w = base.bfloat16().to(dev)
p = int8x_quantize(w, (3, 5, 8))
b1, b2 = p["bitmaps"]; l1, l2, l3 = p["streams"]
x = torch.randn(in_f, dtype=torch.bfloat16, device=dev)
w_dec = decode_weight_triton(p, device=dev)
y_ref = F.linear(x, w_dec)

for chunk in (8704, 4352*0+2176*0+17408//2, 2176, 1024):
    if in_f % chunk or chunk % 512:
        continue
    q1c, q2c = compute_row_prefixes(p, chunk)
    gbs = [t.to(dev) for t in (b1, b2, l1, l2, l3, q1c, q2c, p["scale"])]
    y32 = torch.zeros(out_f, dtype=torch.float32, device=dev)
    def call():
        y32.zero_()
        fused_gemv(x, *gbs, out_f, in_f, chunk=chunk, y32=y32)
    call(); torch.cuda.synchronize()
    err = (y32.to(torch.bfloat16).float()-y_ref.float()).abs().max()/y_ref.float().abs().max()
    for _ in range(20): call()
    torch.cuda.synchronize(); t0 = time.time()
    for _ in range(300): call()
    torch.cuda.synchronize()
    ms = (time.time()-t0)/300*1e3
    print(f"chunk={chunk:5d} S={in_f//chunk:2d}: {ms:.3f}ms ({out_f*in_f/ms/1e6:.0f}G/s) err={err:.1e}")
