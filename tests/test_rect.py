"""Rect-tile TPAB validation: 48x5120 gates + square regression."""
import sys, math
sys.setrecursionlimit(10000)
import torch
import torch.nn.functional as F
from ixrun.tpab import encode_tpab, decode_tpab_ref, decode_tpab_triton, stage_gpu
from ixrun.tpab_gemv import fused_gemv_tpab, prepare_gemv_stage

LOG = open(r"E:\IXRUN\tests\rect_out.txt", "w", encoding="utf-8", errors="replace")
def P(*a):
    s = " ".join(str(x) for x in a)
    LOG.write(s + "\n"); LOG.flush()

def mk(O, I, seed=0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    base = torch.randn(O, I, generator=g) * 0.005
    flat = base.view(-1)
    idx = torch.randperm(flat.numel(), generator=g)[: int(O*I*0.02)]
    flat[idx] = torch.randn(len(idx), generator=g) * 0.05
    return base.bfloat16().cuda()

def snr(w, wd):
    p = w.float().pow(2).mean().item()
    e = (w.float() - wd.float()).pow(2).mean().item()
    return 10 * math.log10(p / max(e, 1e-30))

for O, I in [(48, 5120), (96, 5120), (1152, 4304), (2048, 1536), (5120, 5120)]:
    if I % 64:
        P(f"[{O}x{I}] in_f%64!=0 -> skipped by design")
        continue
    w = mk(O, I)
    p = encode_tpab(w, snr_target_db=24.0)
    P(f"[{O}x{I}] tile_r={p['tile_r']} T={p['T']} n_per={p['n_per']} "
      f"bpw={p['bpw']:.2f} snr={snr(w, decode_tpab_ref(p, device='cuda')):.2f}dB")

    # triton full decode == ref
    st = stage_gpu(p, "cuda")
    a = decode_tpab_ref(p, device="cuda")
    b = decode_tpab_triton(p, device="cuda", staged=st)
    P(f"   triton==ref: {torch.equal(a, b)}")

    # fused GEMV == decode+linear
    gs = prepare_gemv_stage(p, "cuda", staged=st)
    x = torch.randn(I, dtype=torch.bfloat16, device="cuda")
    y1 = fused_gemv_tpab(x, gs, O, I, tile_r=p["tile_r"])
    y2 = F.linear(x, a)
    rel = (y1.float() - y2.float()).abs().max().item() / max(y2.float().abs().max().item(), 1e-9)
    P(f"   gemv rel_err={rel:.2e} {'OK' if rel < 5e-3 else 'FAIL'}")
    del w, p, st, gs
    torch.cuda.empty_cache()
P("DONE")
LOG.close()
