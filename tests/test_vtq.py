"""VT-Queue prototype validation on MiniCPM5 real layers (quality first)."""
import sys, os, json, math
sys.setrecursionlimit(10000)
import torch
from safetensors import safe_open

from ixrun.vtq import encode_vtq, decode_vtq_ref
from ixrun.quantize import int8x_quantize
from ixrun.triton_kernels import decode_weight_triton
from ixrun.tpab import encode_tpab, decode_tpab_ref

LOG = open(r"E:\IXRUN\tests\vtq_out.txt", "w", encoding="utf-8", errors="replace")
def P(*a):
    s = " ".join(str(x) for x in a)
    LOG.write(s + "\n"); LOG.flush()

BASE = r"F:\dg_minicpm5\hf_cache\models--openbmb--MiniCPM5-1B\snapshots\4e9de7a0778dc1c362e983e6858f0e77542cbdca"
KEYS = [
    "model.layers.0.self_attn.q_proj.weight",
    "model.layers.3.mlp.gate_proj.weight",
    "model.layers.11.mlp.down_proj.weight",
]

def load(key):
    with open(os.path.join(BASE, "model.safetensors.index.json")) as f:
        idx = json.load(f)["weight_map"]
    with safe_open(os.path.join(BASE, idx[key]), "pt") as sf:
        return sf.get_tensor(key)

def snr(w, wd):
    p = w.float().pow(2).mean().item()
    e = (w.float() - wd.float()).pow(2).mean().item()
    return 10 * math.log10(p / max(e, 1e-30))

for key in KEYS:
    w = load(key).cuda()
    P(f"\n=== {key} {tuple(w.shape)} ===")

    p_x = int8x_quantize(w, (3, 5, 8))
    s_x = snr(w, decode_weight_triton(p_x, device="cuda"))
    P(f"  INT8-X : bpw={p_x['bits_per_weight']:.2f} SNR={s_x:.2f}dB")

    p_t = encode_tpab(w, snr_target_db=26.0)
    s_t = snr(w, decode_tpab_ref(p_t, device="cuda"))
    P(f"  TPAB@26: bpw={p_t['bpw']:.2f} SNR={s_t:.2f}dB")

    try:
        p_v = encode_vtq(w)
        w_v = decode_vtq_ref(p_v, device="cuda")
        s_v = snr(w, w_v)
        P(f"  VT-Queue: bpw={p_v['bpw']:.2f} SNR={s_v:.2f}dB "
          f"esc={p_v['n_esc']/p_v['S']/p_v['n_per']*100:.2f}% "
          f"tiers={p_v['cnts']}")
        P(f"     maxerr={(w.float()-w_v.float()).abs().max():.2e}")
    except Exception as e:
        P(f"  VT-Queue FAILED: {type(e).__name__} {e}")
    del w
    torch.cuda.empty_cache()
P("DONE")
LOG.close()
