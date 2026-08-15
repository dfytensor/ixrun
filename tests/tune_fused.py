"""Autotune fused GEMV: num_warps x BK variants on Qwen layer shapes."""
import sys, time, itertools
sys.setrecursionlimit(10000)
import torch
import torch.nn.functional as F
from ixrun.quantize import int8x_quantize
from ixrun.triton_kernels import decode_weight_triton
from ixrun.fused import _ix_gemv_kernel, compute_row_prefixes, FUSED_TILE

dev = "cuda"
torch.manual_seed(0)

def bench_one(out_f, in_f, num_warps, bk):
    base = torch.randn(out_f, in_f) * 0.005
    n_out = int(out_f * in_f * 0.04)
    flat = base.view(-1); idx = torch.randperm(flat.numel())[:n_out]
    flat[idx] = torch.randn(n_out) * 0.05
    w = base.bfloat16().to(dev)
    p = int8x_quantize(w, (3, 5, 8))
    b1, b2 = p["bitmaps"]; l1, l2, l3 = p["streams"]
    q1, q2 = compute_row_prefixes(p)
    b1, b2, l1, l2, l3, q1, q2, sc = [t.to(dev) for t in (b1, b2, l1, l2, l3, q1, q2, p["scale"])]
    x = torch.randn(in_f, dtype=torch.bfloat16, device=dev)
    y = torch.empty(out_f, dtype=torch.bfloat16, device=dev)
    def call():
        _ix_gemv_kernel[(out_f,)](x, y, b1, b2, l1, l2, l3, q1, q2, sc,
                                  IN_F=in_f, OUT_F=out_f, BK=bk, num_warps=num_warps)
    for _ in range(15): call()
    torch.cuda.synchronize(); t0 = time.time()
    for _ in range(200): call()
    torch.cuda.synchronize()
    ms = (time.time()-t0)/200*1e3
    # correctness spot check
    y_ref = F.linear(x, decode_weight_triton(p, device=dev))
    err = (y.float()-y_ref.float()).abs().max().item()/max(y_ref.float().abs().max().item(),1e-9)
    return ms, err

shapes = [(17408, 5120), (5120, 17408), (5120, 5120)]
for out_f, in_f in shapes:
    print(f"[{out_f}x{in_f}]")
    for nw, bk in itertools.product([2, 4, 8], [512, 1024]):
        if in_f % bk: continue
        ms, err = bench_one(out_f, in_f, nw, bk)
        rate = out_f*in_f/ms/1e6
        print(f"  warps={nw} BK={bk}: {ms:.3f}ms  {rate:.0f}G elem/s  err={err:.1e} {'OK' if err<5e-3 else 'FAIL'}")
