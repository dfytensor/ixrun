"""Split-K sweep for TPAB GEMV on tall/square shapes (+ outlier overlay done
via the shared in-kernel contract comparison)."""
import sys, time
sys.setrecursionlimit(10000)
import torch
from ixrun.tpab import encode_tpab, decode_tpab_ref, stage_gpu
from ixrun.tpab_gemv import fused_gemv_tpab, prepare_gemv_stage
from ixrun.tpab_gemv_splitk import fused_gemv_tpab_splitk

LOG = open(r"E:\IXRUN\tests\sk_out.txt", "w", encoding="utf-8", errors="replace")
def P(*a):
    s = " ".join(str(x) for x in a)
    LOG.write(s + "\n"); LOG.flush()

def mk(O, I, seed=0):
    g = torch.Generator().manual_seed(seed)
    base = torch.randn(O, I, generator=g) * 0.005
    flat = base.view(-1)
    idx = torch.randperm(flat.numel(), generator=g)[: int(O*I*0.02)]
    flat[idx] = torch.randn(len(idx), generator=g) * 0.05
    return base.bfloat16().cuda()

N = 200
for O, I in [(5120, 5120), (17408, 5120), (10240, 5120), (5120, 17408)]:
    w = mk(O, I)
    x = torch.randn(I, dtype=torch.bfloat16, device="cuda")
    p = encode_tpab(w, snr_target_db=26.0)
    st = stage_gpu(p, "cuda")
    gs = prepare_gemv_stage(p, "cuda", staged=st)

    # baseline (no split)
    for _ in range(10): fused_gemv_tpab(x, gs, O, I, tile_r=p["tile_r"])
    torch.cuda.synchronize(); t0 = time.time()
    for _ in range(N): fused_gemv_tpab(x, gs, O, I, tile_r=p["tile_r"])
    torch.cuda.synchronize()
    t_base = (time.time()-t0)/N*1e3

    line = f"{O}x{I}: base={t_base:.3f}ms ({O*I/t_base/1e6:.0f}G/s)"
    best = (t_base, 1)
    for S in (2, 4, 8):
        if (I // 64) % S:
            continue
        y32 = torch.zeros(O, dtype=torch.float32, device="cuda")
        def call():
            fused_gemv_tpab_splitk(x, gs, O, I, tile_r=p["tile_r"],
                                   split=S, y32=y32)
        call(); torch.cuda.synchronize()
        for _ in range(10): call()
        torch.cuda.synchronize(); t0 = time.time()
        for _ in range(N): call()
        torch.cuda.synchronize()
        t_s = (time.time()-t0)/N*1e3
        line += f"  S{S}={t_s:.3f}ms ({O*I/t_s/1e6:.0f}G/s)"
        if t_s < best[0]:
            best = (t_s, S)
    line += f"  -> best S={best[1]} ({t_base/best[0]:.2f}x)"
    P(line)

    # correctness: split result (with overlay) == baseline kernel
    S = best[1] if (I // 64) % best[1] == 0 else 1
    y_ref = fused_gemv_tpab(x, gs, O, I, tile_r=p["tile_r"]).float()
    y32 = fused_gemv_tpab_splitk(x, gs, O, I, tile_r=p["tile_r"], split=S)
    # overlay outliers manually
    olk, olv = gs["ol_row_k"], gs["ol_row_v"]
    contrib = olv * x[olk.long()].float()
    rows = torch.bucketize(torch.arange(O, device=x.device),
                           gs["ol_offs"].float(), right=True)  # not needed; use index_add
    # derive rows properly
    ol_t = p["ol_t"].cuda().long(); ol_l = p["ol_l"].cuda().long()
    rows2 = ol_t // (I // 64) * p["tile_r"] + ol_l // 64
    y32.index_add_(0, rows2, contrib)
    d = (y32.to(torch.bfloat16).float() - y_ref).abs().max().item()
    P(f"   split S={S} + overlay == base: maxdiff={d:.2e}")
    del w, p, st, gs
    torch.cuda.empty_cache()
P("DONE")
LOG.close()
