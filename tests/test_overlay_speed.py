"""Speed experiment: TPAB GEMV outlier-overlay cost — in-kernel dynamic loop
vs host-side batched correction.

Hypothesis: the per-row `for j in tl.range(lo, hi)` overlay causes warp
divergence and is the reason TPAB loses on small/tall shapes (214 vs
INT8-X 654 G/s on 5120x5120). Moving the overlay out (host: y += corr
where corr = olv * x[olk] batched for all rows at once) makes the kernel
loop-free/static.
"""
import sys, time
sys.setrecursionlimit(10000)
import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from ixrun.tpab import encode_tpab, decode_tpab_ref, stage_gpu
from ixrun.tpab_gemv import prepare_gemv_stage, fused_gemv_tpab, _tpab_gemv_kernel

LOG = open(r"E:\IXRUN\tests\olsp_out.txt", "w", encoding="utf-8", errors="replace")
def P(*a):
    s = " ".join(str(x) for x in a)
    LOG.write(s + "\n"); LOG.flush()

# kernel variant WITHOUT the overlay loop
@triton.jit
def _tpab_gemv_nooverlay(
    x_ptr, y_ptr,
    body_ptr, bits_ptr, scales_ptr, goff_ptr, gbase_ptr,
    T_C: tl.constexpr, IN_F: tl.constexpr,
    TILE_R: tl.constexpr, TILE_C: tl.constexpr,
):
    n = tl.program_id(0)
    row_tile = n // TILE_R
    in_row = n % TILE_R
    acc = 0.0
    for kc in tl.range(0, T_C):
        t = row_tile * T_C + kc
        b = tl.load(bits_ptr + t).to(tl.int32)
        s = tl.load(scales_ptr + t).to(tl.float32)
        gbase = tl.load(gbase_ptr + b)
        goff = tl.load(goff_ptr + t).to(tl.int64)
        L = in_row * TILE_C + tl.arange(0, TILE_C)
        bitpos = gbase + (goff + L.to(tl.int64)) * b
        word = (bitpos // 32).to(tl.int32)
        shift = (bitpos % 32).to(tl.int32)
        w1 = tl.load(body_ptr + word).to(tl.uint32)
        cross = (shift + b) > 32
        w2 = tl.where(cross, tl.load(body_ptr + word + 1).to(tl.uint32),
                      tl.zeros((TILE_C,), tl.uint32))
        raw = tl.where(cross, (w1 >> shift) | (w2 << (32 - shift)), w1 >> shift)
        mask = tl.exp2(b.to(tl.float32)).to(tl.int32) - 1
        v = (raw & mask.to(tl.uint32)).to(tl.int32) - ((mask + 1) // 2 - 1)
        k0 = kc * TILE_C + tl.arange(0, TILE_C)
        x = tl.load(x_ptr + k0).to(tl.float32)
        wq = (v.to(tl.float32) * s).to(tl.bfloat16).to(tl.float32)
        acc += tl.sum(x * wq, axis=0)
    tl.store(y_ptr + n, acc.to(tl.bfloat16))


def mk(O, I, seed=0):
    g = torch.Generator().manual_seed(seed)
    base = torch.randn(O, I, generator=g) * 0.005
    flat = base.view(-1)
    idx = torch.randperm(flat.numel(), generator=g)[: int(O*I*0.02)]
    flat[idx] = torch.randn(len(idx), generator=g) * 0.05
    return base.bfloat16().cuda()

N = 200
for O, I in [(5120, 5120), (17408, 5120), (10240, 5120)]:
    w = mk(O, I)
    x = torch.randn(I, dtype=torch.bfloat16, device="cuda")
    p = encode_tpab(w, snr_target_db=26.0)
    st = stage_gpu(p, "cuda")
    gs = prepare_gemv_stage(p, "cuda", staged=st)

    # A: current (in-kernel overlay)
    for _ in range(10): fused_gemv_tpab(x, gs, O, I, tile_r=p["tile_r"])
    torch.cuda.synchronize(); t0 = time.time()
    for _ in range(N): fused_gemv_tpab(x, gs, O, I, tile_r=p["tile_r"])
    torch.cuda.synchronize()
    t_a = (time.time()-t0)/N*1e3

    # B: no-overlay kernel + host batched correction
    y = torch.empty(O, dtype=torch.bfloat16, device="cuda")
    olk, olv, oloff = gs["ol_row_k"], gs["ol_row_v"], gs["ol_offs"]
    def call_b():
        _tpab_gemv_nooverlay[(O,)](
            x, y, st["bodies_g"], st["bits_g"], st["scales_g"],
            st["goff_g"], st["gbase_g"],
            T_C=I//64, IN_F=I, TILE_R=p["tile_r"], TILE_C=64, num_warps=2)
        # batched overlay: scatter-add val*x[k] per row via index_add
        contrib = olv * x[olk.long()].float()
        y32 = y.float()
        y32.index_add_(0, (olk.long()*0 + torch.arange(0, device='cuda')*0), contrib)  # placeholder
    # simpler correct host overlay: use row indices precomputed
    ol_row = gs.get("ol_row")
    # compute row index per outlier once (not timed fairly but stable):
    from ixrun.tpab_gemv import prepare_gemv_stage as _p
    # derive rows: ol_t/ol_l are in packed (CPU) — recompute quickly
    ol_t = p["ol_t"].cuda().long(); ol_l = p["ol_l"].cuda().long()
    rows = ol_t // (I // 64) * p["tile_r"] + ol_l // 64
    def call_b2():
        _tpab_gemv_nooverlay[(O,)](
            x, y, st["bodies_g"], st["bits_g"], st["scales_g"],
            st["goff_g"], st["gbase_g"],
            T_C=I//64, IN_F=I, TILE_R=p["tile_r"], TILE_C=64, num_warps=2)
        contrib = olv * x[( (ol_l % 64) + (ol_t % (I//64)) * 64 ).long()].float()
        y32 = y.float().index_add(0, rows, contrib)
        y.copy_(y32.to(torch.bfloat16))
    call_b2()
    for _ in range(10): call_b2()
    torch.cuda.synchronize(); t0 = time.time()
    for _ in range(N): call_b2()
    torch.cuda.synchronize()
    t_b = (time.time()-t0)/N*1e3

    # correctness
    y_a = fused_gemv_tpab(x, gs, O, I, tile_r=p["tile_r"])
    call_b2()
    d = (y_a.float() - y.float()).abs().max().item()
    P(f"{O}x{I}: in-kernel={t_a:.3f}ms ({O*I/t_a/1e6:.0f}G/s)  "
      f"no-overlay+host={t_b:.3f}ms ({O*I/t_b/1e6:.0f}G/s)  "
      f"ratio={t_a/t_b:.2f}x  maxdiff={d:.2e}")
    del w, p, st, gs
    torch.cuda.empty_cache()
P("DONE")
LOG.close()
