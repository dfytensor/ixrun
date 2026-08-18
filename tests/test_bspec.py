"""Benchmark b-specialized kernels vs generic, on single-B layers."""
import sys, time
sys.setrecursionlimit(10000)
import torch
from ixrun.tpab import encode_tpab, stage_gpu
from ixrun.tpab_gemv import prepare_gemv_stage, fused_gemv_tpab
from ixrun.tpab_gemv_b import fused_gemv_tpab_b
from ixrun.bitpack import unpack_bits_stream

torch.manual_seed(0)

def make_layer(O, I, tail=0.02):
    base = torch.randn(O, I) * 0.005
    flat = base.view(-1)
    idx = torch.randperm(flat.numel())[: int(O*I*tail)]
    flat[idx] = torch.randn(len(idx)) * 0.05
    return base.bfloat16().cuda()

def aligned_stage(p, b):
    """Rebuild single-B layout: since encode lays tiles contiguously within
    each bit-group, for a single-B layer the WHOLE body is one group and
    every tile starts at gbase(=0) + goff*b with goff already element-
    counts — same as generic. The specialized kernel only needs gbase."""
    st = stage_gpu(p, "cuda")
    return st

for O, I in [(17408, 5120), (5120, 17408), (5120, 5120)]:
    w = make_layer(O, I)
    p = encode_tpab(w, snr_target_db=26.0)
    bs = sorted(set(p["bits"].tolist()))
    if len(bs) != 1:
        print(f"{O}x{I}: mixed bits {bs} — skipping (single-B only)")
        continue
    b = bs[0]
    st = aligned_stage(p, b)
    gs = prepare_gemv_stage(p, "cuda", staged=st)
    x = torch.randn(I, dtype=torch.bfloat16, device="cuda")

    y0 = fused_gemv_tpab(x, gs, O, I, tile_r=64)
    y1 = fused_gemv_tpab_b(x, gs, O, I, tile_r=64, b=b)
    d = (y0.float()-y1.float()).abs().max().item()

    for _ in range(10): fused_gemv_tpab(x, gs, O, I, tile_r=64)
    torch.cuda.synchronize(); t0 = time.time()
    for _ in range(200): fused_gemv_tpab(x, gs, O, I, tile_r=64)
    torch.cuda.synchronize(); ms0 = (time.time()-t0)/200*1e3

    for _ in range(10): fused_gemv_tpab_b(x, gs, O, I, tile_r=64, b=b)
    torch.cuda.synchronize(); t0 = time.time()
    for _ in range(200): fused_gemv_tpab_b(x, gs, O, I, tile_r=64, b=b)
    torch.cuda.synchronize(); ms1 = (time.time()-t0)/200*1e3

    # multi-row specialized (R=8)
    y2 = fused_gemv_tpab_b(x, gs, O, I, tile_r=64, b=b, r=8)
    d2 = (y0.float()-y2.float()).abs().max().item()
    for _ in range(10): fused_gemv_tpab_b(x, gs, O, I, tile_r=64, b=b, r=8)
    torch.cuda.synchronize(); t0 = time.time()
    for _ in range(200): fused_gemv_tpab_b(x, gs, O, I, tile_r=64, b=b, r=8)
    torch.cuda.synchronize(); ms2 = (time.time()-t0)/200*1e3

    print(f"{O}x{I} b={b}: generic={ms0:.3f}ms ({O*I/ms0/1e6:.0f}G/s)  "
          f"spec={ms1:.3f}ms ({O*I/ms1/1e6:.0f}G/s, {ms0/ms1:.2f}x)  "
          f"spec+R8={ms2:.3f}ms ({O*I/ms2/1e6:.0f}G/s, {ms0/ms2:.2f}x)  "
          f"maxdiff={d:.1e}/{d2:.1e}")
