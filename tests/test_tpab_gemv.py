"""Validate + benchmark TPAB fused GEMV vs INT8-X fused GEMV (hot loop path)."""
import sys, os, time, json, math
sys.setrecursionlimit(10000)
import torch
import torch.nn.functional as F
from safetensors import safe_open

from ixrun.tpab import encode_tpab, decode_tpab_ref
from ixrun.tpab_gemv import fused_gemv_tpab, prepare_gemv_stage
from ixrun.quantize import int8x_quantize
from ixrun.triton_kernels import decode_weight_triton
from ixrun.fused import fused_gemv, compute_row_prefixes, _pick_config

LOG = open(r"E:\IXRUN\tests\tpab_gemv_out.txt", "w", encoding="utf-8", errors="replace")
def P(*a):
    s = " ".join(str(x) for x in a)
    LOG.write(s + "\n"); LOG.flush(); print(s, flush=True)

def load_weight(base, key):
    with open(os.path.join(base, "model.safetensors.index.json")) as f:
        idx = json.load(f)["weight_map"]
    with safe_open(os.path.join(base, idx[key]), "pt") as sf:
        return sf.get_tensor(key)

MINICPM = r"F:\dg_minicpm5\hf_cache\models--openbmb--MiniCPM5-1B\snapshots\4e9de7a0778dc1c362e983e6858f0e77542cbdca"
QWEN = r"E:\models\Qwen3.8-27B"
LAYERS = [
    ("MiniCPM5 down_proj 1536x4608", "model.layers.0.mlp.down_proj.weight", MINICPM),
    ("Qwen3.8  gate_proj 17408x5120", "model.language_model.layers.10.mlp.gate_proj.weight", QWEN),
    ("Qwen3.8  down_proj 5120x17408", "model.language_model.layers.10.mlp.down_proj.weight", QWEN),
]

for label, key, base in LAYERS:
    P(f"\n=== {label} ===")
    w = load_weight(base, key).cuda()
    O, I = w.shape
    x = torch.randn(I, dtype=torch.bfloat16, device="cuda")

    # --- TPAB GEMV ---
    p_t = encode_tpab(w, snr_target_db=26.0, outlier_frac=0.004)
    st = prepare_gemv_stage(p_t, "cuda")
    y_t = fused_gemv_tpab(x, st, O, I)
    w_ref = decode_tpab_ref(p_t, device="cuda")
    y_ref = F.linear(x, w_ref)
    err = (y_t.float() - y_ref.float()).abs().max().item() / max(y_ref.float().abs().max().item(), 1e-9)
    P(f"  TPAB gemv vs decode+linear ref: rel_err={err:.2e}  {'OK' if err < 5e-3 else 'FAIL'}")

    N = 100
    for _ in range(10): fused_gemv_tpab(x, st, O, I)
    torch.cuda.synchronize(); t0 = time.time()
    for _ in range(N): fused_gemv_tpab(x, st, O, I)
    torch.cuda.synchronize()
    t_tpab = (time.time()-t0)/N*1000

    # --- INT8-X fused GEMV (current hot-loop path) ---
    p_x = int8x_quantize(w, (3, 5, 8))
    b1, b2 = p_x["bitmaps"]; l1, l2, l3 = p_x["streams"]
    from ixrun.fused import _pick_split
    split = _pick_split(I) if O < I else 1
    if split > 1:
        chunk = I // split
        q1, q2 = compute_row_prefixes(p_x, chunk)
        y32 = torch.zeros(O, dtype=torch.float32, device="cuda")
        gx = [t.cuda() for t in (b1, b2, l1, l2, l3, q1, q2, p_x["scale"])]
        def call_ix():
            y32.zero_()
            fused_gemv(x, *gx, O, I, chunk=chunk, y32=y32)
    else:
        q1, q2 = compute_row_prefixes(p_x)
        gx = [t.cuda() for t in (b1, b2, l1, l2, l3, q1, q2, p_x["scale"])]
        def call_ix(): return fused_gemv(x, *gx, O, I)
    call_ix()
    torch.cuda.synchronize(); t0 = time.time()
    for _ in range(N): call_ix()
    torch.cuda.synchronize()
    t_ix = (time.time()-t0)/N*1000

    P(f"  speed: TPAB={t_tpab:.3f}ms ({O*I/t_tpab/1e6:.0f}G/s)  "
      f"INT8-X={t_ix:.3f}ms ({O*I/t_ix/1e6:.0f}G/s)  ratio={t_ix/t_tpab:.2f}x")
    P(f"  bpw : TPAB={p_t['bpw']:.2f}  INT8-X={p_x['bits_per_weight']:.2f}")
    del w, w_ref, st
    torch.cuda.empty_cache()

P("\nDONE")
LOG.close()
