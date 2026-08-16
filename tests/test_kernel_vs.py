"""Per-kernel-sum comparison on the 27B: TPAB GEMV layers vs INT8-X GEMV."""
import sys, time
sys.setrecursionlimit(10000)
import pandas
import torch
from transformers import AutoTokenizer
from ixrun.config import QWEN38_PATH, DEFAULT_LEVELS
from ixrun.engine import Int8XEngine, StreamingLinear
from ixrun.tpab_linear import TpabLinear
from ixrun.linear import iter_quantizable_linears
from ixrun.fused import fused_gemv as ix_gemv
from ixrun.tpab_gemv import fused_gemv_tpab
from ixrun.quantize import int8x_quantize
from ixrun.fused import compute_row_prefixes, _pick_split

LOG = open(r"E:\IXRUN\tests\kv_out.txt", "w", encoding="utf-8", errors="replace")
def P(*a):
    s = " ".join(str(x) for x in a)
    LOG.write(s + "\n"); LOG.flush()

tok = AutoTokenizer.from_pretrained(QWEN38_PATH, trust_remote_code=True)
m = Int8XEngine._load_any(QWEN38_PATH, torch.bfloat16, low_cpu=True)

# build fresh TPAB layers on GPU for a few representative text shapes,
# and int8x equivalents — time GEMV calls back to back
SHAPES = [(17408, 5120), (5120, 17408), (5120, 5120), (5120, 6144), (10240, 5120)]
torch.manual_seed(0)
N = 100
for O, I in SHAPES:
    w = (torch.randn(O, I, device="cuda") * 0.005).bfloat16()
    # heavy-tailify
    flat = w.view(-1)
    idx = torch.randperm(flat.numel(), device="cuda")[: int(O*I*0.01)]
    flat[idx] = (torch.randn(len(idx), device="cuda") * 0.05).bfloat16()

    x = torch.randn(I, dtype=torch.bfloat16, device="cuda")

    tl = TpabLinear(w, snr_target_db=24.0)
    for _ in range(10): fused_gemv_tpab(x, tl.gemv_stage, O, I)
    torch.cuda.synchronize(); t0 = time.time()
    for _ in range(N): fused_gemv_tpab(x, tl.gemv_stage, O, I)
    torch.cuda.synchronize()
    t_tpab = (time.time()-t0)/N*1e3
    del tl
    torch.cuda.empty_cache()

    p = int8x_quantize(w, DEFAULT_LEVELS)
    b1, b2 = p["bitmaps"]; l1, l2, l3 = p["streams"]
    from ixrun.fused import _pick_split
    if O < I and _pick_split(I) > 1:
        chunk = I // _pick_split(I)
        q1, q2 = compute_row_prefixes(p, chunk)
        y32 = torch.zeros(O, dtype=torch.float32, device="cuda")
        gx = [t.cuda() for t in (b1, b2, l1, l2, l3, q1, q2, p["scale"])]
        s = _pick_split(I)
        def call(): y32.zero_(); return ix_gemv(x, *gx, O, I, chunk=chunk, y32=y32)
    else:
        q1, q2 = compute_row_prefixes(p)
        gx = [t.cuda() for t in (b1, b2, l1, l2, l3, q1, q2, p["scale"])]
        def call(): return ix_gemv(x, *gx, O, I)
    for _ in range(10): call()
    torch.cuda.synchronize(); t0 = time.time()
    for _ in range(N): call()
    torch.cuda.synchronize()
    t_ix = (time.time()-t0)/N*1e3
    del p, gx
    torch.cuda.empty_cache()
    P(f"{O}x{I}: TPAB={t_tpab:.3f}ms ({O*I/t_tpab/1e6:.0f}G/s)  "
      f"INT8-X={t_ix:.3f}ms ({O*I/t_ix/1e6:.0f}G/s)  ratio={t_ix/t_tpab:.2f}x")
P("DONE")
LOG.close()
