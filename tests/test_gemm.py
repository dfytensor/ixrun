"""Validate + benchmark fused decode+GEMM vs decode+cuBLAS on real shapes."""
import sys, time
sys.setrecursionlimit(10000)
import torch
import torch.nn.functional as F
from ixrun.tpab import encode_tpab, decode_tpab_triton, stage_gpu, decode_tpab_ref
from ixrun.tpab_gemm import fused_gemm_tpab

LOG = open(r"E:\IXRUN\tests\fg_out.txt", "w", encoding="utf-8", errors="replace")
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
from ixrun.tpab_gemv import prepare_gemv_stage
gs = prepare_gemv_stage(p, "cuda", staged=st)
w_ref = decode_tpab_ref(p, device="cuda")

for M in (8, 32, 128, 512):
    x = torch.randn(M, I, dtype=torch.bfloat16, device="cuda")
    y_ref = F.linear(x, w_ref)

    try:
        y = fused_gemm_tpab(x, gs, O, I, tile_r=64)
        rel = (y.float() - y_ref.float()).abs().max().item() / y_ref.float().abs().max().item()
        ok = rel < 5e-3
    except Exception as e:
        P(f"M={M}: FUSED FAILED {type(e).__name__}: {str(e)[:200]}")
        continue

    # current path: decode to workspace + cuBLAS
    ws = torch.zeros(p["T"]*p["n_per"], dtype=torch.float32, device="cuda")
    def cur():
        wd = decode_tpab_triton(p, device="cuda", out_f32=ws, staged=st)
        return F.linear(x, wd)
    for _ in range(5): fused_gemm_tpab(x, gs, O, I, tile_r=64)
    torch.cuda.synchronize(); t0 = time.time()
    for _ in range(50): fused_gemm_tpab(x, gs, O, I, tile_r=64)
    torch.cuda.synchronize()
    t_f = (time.time()-t0)/50*1e3
    for _ in range(5): cur()
    torch.cuda.synchronize(); t0 = time.time()
    for _ in range(50): cur()
    torch.cuda.synchronize()
    t_c = (time.time()-t0)/50*1e3

    P(f"M={M:4d}: fused={t_f:.3f}ms  current={t_c:.3f}ms  "
      f"ratio={t_c/t_f:.2f}x  rel_err={rel:.1e} {'OK' if ok else 'FAIL'}")
P("DONE")
LOG.close()
