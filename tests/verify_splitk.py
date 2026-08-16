"""Authoritative correctness: split-K TPAB GEMV vs decode_ref + F.linear."""
import sys
sys.setrecursionlimit(10000)
import torch
import torch.nn.functional as F
from ixrun.tpab import encode_tpab, decode_tpab_ref, stage_gpu
from ixrun.tpab_gemv import prepare_gemv_stage
from ixrun.tpab_gemv_splitk import fused_gemv_tpab_splitk

torch.manual_seed(0)
for O, I, S in [(17408, 5120, 4), (5120, 17408, 2), (5120, 5120, 8), (10240, 5120, 2)]:
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
    y32 = fused_gemv_tpab_splitk(x, gs, O, I, tile_r=p["tile_r"], split=S)
    # overlay outliers (same bf16 rounding as reference path)
    ol_t = p["ol_t"].cuda().long(); ol_l = p["ol_l"].cuda().long()
    rows = ol_t // (I // 64) * p["tile_r"] + ol_l // 64
    ks = (ol_t % (I // 64)) * 64 + ol_l % 64
    contrib = (p["ol_val"].cuda().float().to(torch.bfloat16).float()
               * x[ks].float())
    y32.index_add_(0, rows, contrib)
    y = y32.to(torch.bfloat16).float()
    rel = (y - y_ref).abs().max().item() / y_ref.abs().max().item()
    print(f"{O}x{I} S={S}: rel_err={rel:.2e} {'OK' if rel < 5e-3 else 'FAIL'}")
