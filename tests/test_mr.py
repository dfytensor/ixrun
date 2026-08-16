"""Multi-row TPAB GEMV: correctness + R sweep vs 1-row kernel."""
import sys, time
sys.setrecursionlimit(10000)
import torch
import torch.nn.functional as F
from ixrun.tpab import encode_tpab, decode_tpab_ref, stage_gpu
from ixrun.tpab_gemv import fused_gemv_tpab, prepare_gemv_stage
from ixrun.tpab_gemv_mr import fused_gemv_tpab_mr

LOG = open(r"E:\IXRUN\tests\mr_out.txt", "w", encoding="utf-8", errors="replace")
def P(*a):
    s = " ".join(str(x) for x in a)
    LOG.write(s + "\n"); LOG.flush()

torch.manual_seed(0)
N = 200
for O, I in [(17408, 5120), (5120, 17408), (5120, 5120), (10240, 5120)]:
    base = torch.randn(O, I) * 0.005
    flat = base.view(-1)
    idx = torch.randperm(flat.numel())[: int(O*I*0.02)]
    flat[idx] = torch.randn(len(idx)) * 0.05
    w = base.bfloat16().cuda()
    p = encode_tpab(w, snr_target_db=26.0)
    st = stage_gpu(p, "cuda")
    gs = prepare_gemv_stage(p, "cuda", staged=st)
    x = torch.randn(I, dtype=torch.bfloat16, device="cuda")
    y_ref = F.linear(x, decode_tpab_ref(p, device="cuda")).float()

    # baseline 1-row
    for _ in range(10): fused_gemv_tpab(x, gs, O, I, tile_r=p["tile_r"])
    torch.cuda.synchronize(); t0 = time.time()
    for _ in range(N): fused_gemv_tpab(x, gs, O, I, tile_r=p["tile_r"])
    torch.cuda.synchronize()
    t_base = (time.time()-t0)/N*1e3

    line = f"{O}x{I}: base={t_base:.3f}ms ({O*I/t_base/1e6:.0f}G/s)"
    best = (t_base, 1)
    for R in (2, 4, 8, 16):
        if p["tile_r"] % R or O % R:
            continue
        try:
            y = fused_gemv_tpab_mr(x, gs, O, I, tile_r=p["tile_r"], r=R)
            rel = (y.float()-y_ref).abs().max().item()/y_ref.abs().max().item()
            for _ in range(10): fused_gemv_tpab_mr(x, gs, O, I, tile_r=p["tile_r"], r=R)
            torch.cuda.synchronize(); t0 = time.time()
            for _ in range(N): fused_gemv_tpab_mr(x, gs, O, I, tile_r=p["tile_r"], r=R)
            torch.cuda.synchronize()
            t_r = (time.time()-t0)/N*1e3
            line += f"  R{R}={t_r:.3f}ms({O*I/t_r/1e6:.0f}G/s,ok={rel<5e-3})"
            if t_r < best[0]:
                best = (t_r, R)
        except Exception as e:
            line += f"  R{R}=FAIL({type(e).__name__})"
    line += f"  -> best R={best[1]} ({t_base/best[0]:.2f}x)"
    P(line)
    del w, p, st, gs
    torch.cuda.empty_cache()
P("DONE")
LOG.close()
