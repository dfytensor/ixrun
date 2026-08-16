"""Batch decode economics: batched decode-to-bf16 + one GEMM vs N x fused GEMV.

At batch B, one decode_tpab_triton (bandwidth-bound stream of packed) +
one cuBLAS GEMM [B, I] x [I, O] replaces B separate GEMV steps. GEMM has
tensor cores; decode cost is amortized over B. Find the B where the
GEMM path beats B x single-token GEMV path.
"""
import sys, time
sys.setrecursionlimit(10000)
import torch
import torch.nn.functional as F
from ixrun.tpab import encode_tpab, decode_tpab_triton, stage_gpu
from ixrun.tpab_gemv import prepare_gemv_stage
from ixrun.tpab_gemv_mr import fused_gemv_tpab_mr

LOG = open(r"E:\IXRUN\tests\becon_out.txt", "w", encoding="utf-8", errors="replace")
def P(*a):
    s = " ".join(str(x) for x in a)
    LOG.write(s + "\n"); LOG.flush()

torch.manual_seed(0)
O, I = 17408, 5120
base = torch.randn(O, I) * 0.005
flat = base.view(-1)
idx = torch.randperm(flat.numel())[: int(O*I*0.02)]
flat[idx] = torch.randn(len(idx)) * 0.05
w = base.bfloat16().cuda()
p = encode_tpab(w, snr_target_db=26.0)
st = stage_gpu(p, "cuda")
ws = torch.zeros(p["T"]*p["n_per"], dtype=torch.float32, device="cuda")

P(f"layer {O}x{I} ({O*I/1e6:.0f}M elems), tpab bpw={p['bpw']:.2f}")
for B in (1, 2, 4, 8, 16):
    xs = torch.randn(B, I, dtype=torch.bfloat16, device="cuda")

    # path A: B sequential fused GEMV (current server behavior)
    gs = prepare_gemv_stage(p, "cuda", staged=st)
    def seq_gemv():
        outs = []
        for b in range(B):
            outs.append(fused_gemv_tpab_mr(xs[b], gs, O, I, tile_r=64, r=8))
        return outs

    # path B: one decode + one GEMM
    def dec_gemm():
        wd = decode_tpab_triton(p, device="cuda", out_f32=ws, staged=st)
        return F.linear(xs, wd)

    for f, name in ((dec_gemm, "decode+GEMM"),):
        for _ in range(5): f()
        torch.cuda.synchronize(); t0 = time.time()
        for _ in range(50): f()
        torch.cuda.synchronize()
        t = (time.time()-t0)/50*1e3
        P(f"B={B:2d} {name}: {t:.3f}ms  ({t/B*1000:.0f} us/effective-token)")

    # single-token GEMV cost for comparison
    x1 = xs[0]
    for _ in range(5): fused_gemv_tpab_mr(x1, gs, O, I, tile_r=64, r=8)
    torch.cuda.synchronize(); t0 = time.time()
    for _ in range(50): fused_gemv_tpab_mr(x1, gs, O, I, tile_r=64, r=8)
    torch.cuda.synchronize()
    t1 = (time.time()-t0)/50*1e3
    P(f"        1xGEMV={t1:.3f}ms -> B x GEMV would cost {B*t1:.3f}ms; "
      f"GEMM-path {'WINS' if t < B*t1 else 'loses'} ({B*t1/t:.2f}x)")
P("DONE")
LOG.close()
