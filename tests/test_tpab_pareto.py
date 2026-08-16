"""Pareto: TPAB joint (k,b) search vs previous fixed-outlier encoding."""
import sys, os, time, json, math
sys.setrecursionlimit(10000)
import torch
from safetensors import safe_open
from ixrun.tpab import encode_tpab, decode_tpab_ref
from ixrun.quantize import int8x_quantize
from ixrun.triton_kernels import decode_weight_triton

LOG = open(r"E:\IXRUN\tests\tpab_pareto_out.txt", "w", encoding="utf-8", errors="replace")
def P(*a):
    s = " ".join(str(x) for x in a)
    LOG.write(s + "\n"); LOG.flush(); print(s, flush=True)

def load_weight(base, key):
    with open(os.path.join(base, "model.safetensors.index.json")) as f:
        idx = json.load(f)["weight_map"]
    with safe_open(os.path.join(base, idx[key]), "pt") as sf:
        return sf.get_tensor(key)

def snr_db(w, wd):
    p = w.float().pow(2).mean().item()
    e = (w.float() - wd.float()).pow(2).mean().item()
    return 10 * math.log10(p / max(e, 1e-30))

MINICPM = r"F:\dg_minicpm5\hf_cache\models--openbmb--MiniCPM5-1B\snapshots\4e9de7a0778dc1c362e983e6858f0e77542cbdca"
QWEN = r"E:\models\Qwen3.8-27B"
LAYERS = [
    ("MiniCPM down", "model.layers.0.mlp.down_proj.weight", MINICPM),
    ("Qwen gate", "model.language_model.layers.10.mlp.gate_proj.weight", QWEN),
]

for label, key, base in LAYERS:
    P(f"\n=== {label} ===")
    w = load_weight(base, key).cuda()
    p_x = int8x_quantize(w, (3, 5, 8))
    w_x = decode_weight_triton(p_x, device="cuda")
    P(f"  INT8-X: bpw={p_x['bits_per_weight']:.2f} SNR={snr_db(w, w_x):.2f}dB")

    for target in (22.0, 24.0, 26.0, 28.0):
        t0 = time.time()
        p_t = encode_tpab(w, snr_target_db=target)
        t_enc = time.time() - t0
        w_t = decode_tpab_ref(p_t, device="cuda")
        s = snr_db(w, w_t)
        hist = torch.bincount(p_t["bits"].int(), minlength=8).tolist()
        P(f"  TPAB@{target:.0f}dB: bpw={p_t['bpw']:.2f} SNR={s:.2f}dB "
          f"maxerr={(w.float()-w_t.float()).abs().max():.2e} "
          f"ol={p_t['n_outliers']/p_t['T']/p_t['n_per']*100:.2f}% "
          f"fb={p_t['fallback_tiles']} enc={t_enc:.1f}s bits_hist={hist[2:7]}")
    del w, w_t, w_x
    torch.cuda.empty_cache()

P("DONE")
LOG.close()
