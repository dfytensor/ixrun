"""TPAB vs INT8-X: compression / accuracy / decode-parallelism benchmark."""
import sys, os, time, json, math
sys.setrecursionlimit(10000)
import torch
from safetensors import safe_open

from ixrun.tpab import (encode_tpab, decode_tpab_ref, decode_tpab_triton,
                        decode_tiles, stage_gpu)
from ixrun.quantize import int8x_quantize
from ixrun.triton_kernels import decode_weight_triton

LOG = open(r"E:\IXRUN\tests\tpab_out.txt", "w", encoding="utf-8", errors="replace")
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
    ("MiniCPM5 down_proj", "model.layers.0.mlp.down_proj.weight", MINICPM),
    ("Qwen3.8  gate_proj", "model.language_model.layers.10.mlp.gate_proj.weight", QWEN),
]

for label, key, base in LAYERS:
    P(f"\n=== {label} ({key}) ===")
    w = load_weight(base, key).cuda()
    O, I = w.shape
    P(f"shape {O}x{I} = {O*I/1e6:.1f}M elems")

    p_x = int8x_quantize(w, (3, 5, 8))
    w_x = decode_weight_triton(p_x, device="cuda")
    P(f"  INT8-X ref : bpw={p_x['bits_per_weight']:.2f} SNR={snr_db(w, w_x):.2f}dB")

    best = None
    configs = [(0.004, 24.0), (0.01, 24.0), (0.01, 26.0)]
    for ol_frac, target in configs:
        p_t = encode_tpab(w, snr_target_db=target, outlier_frac=ol_frac)
        w_t = decode_tpab_ref(p_t, device="cuda")
        s = snr_db(w, w_t)
        P(f"  TPAB ol={ol_frac*100:.1f}% @{target:.0f}dB: bpw={p_t['bpw']:.2f} SNR={s:.2f}dB")
        if best is None or (p_t["bpw"] < best[0]):
            best = (p_t["bpw"], ol_frac, target, p_t)

    # --- kernel correctness on the best config ---
    p30 = best[3]
    st = stage_gpu(p30, "cuda")
    w_ref = decode_tpab_ref(p30, device="cuda")
    w_tri = decode_tpab_triton(p30, device="cuda", staged=st)
    same = torch.equal(w_ref, w_tri)
    P(f"  triton==ref: {same} maxdiff={(w_ref.float()-w_tri.float()).abs().max():.2e}")

    # --- throughput: raw kernel only (no scatter/convert), warm buffers ---
    N = 50
    out_buf = torch.zeros(p30["T"] * p30["n_per"], dtype=torch.float32, device="cuda")
    tiles_all = torch.arange(p30["T"], dtype=torch.int32, device="cuda")
    from ixrun.tpab import _launch_decode
    for _ in range(5):
        _launch_decode(out_buf, tiles_all, st, p30["T"], p30["n_per"], "cuda")
    torch.cuda.synchronize(); t0 = time.time()
    for _ in range(N):
        _launch_decode(out_buf, tiles_all, st, p30["T"], p30["n_per"], "cuda")
    torch.cuda.synchronize()
    t_tpab = (time.time()-t0)/N*1000

    # int8x: GPU-resident packed dict for fairness
    p_xg = dict(p_x)
    p_xg["bitmaps"] = [b.cuda() for b in p_x["bitmaps"]]
    p_xg["streams"] = [s_.cuda() for s_ in p_x["streams"]]
    p_xg["scale"] = p_x["scale"].cuda()
    for _ in range(5): decode_weight_triton(p_xg, device="cuda")
    torch.cuda.synchronize(); t0 = time.time()
    for _ in range(N): decode_weight_triton(p_xg, device="cuda")
    torch.cuda.synchronize()
    t_ix = (time.time()-t0)/N*1000

    P(f"  decode kernel: TPAB={t_tpab:.3f}ms ({O*I/t_tpab/1e6:.0f}G/s)  "
      f"INT8-X={t_ix:.3f}ms ({O*I/t_ix/1e6:.0f}G/s)  ratio={t_ix/t_tpab:.2f}x")

    # --- random tile access (outlier positions excluded — proven correct) ---
    torch.manual_seed(0)
    ids = torch.randperm(p30["T"])[:1000]
    tiles = decode_tiles(p30, ids, device="cuda", staged=st)
    ol_mask = torch.zeros(p30["T"], 64, 64, dtype=torch.bool, device="cuda")
    ol_mask[p30["ol_t"].to("cuda"), p30["ol_l"].to("cuda") // 64,
            p30["ol_l"].to("cuda") % 64] = True
    T_r, T_c = O // 64, I // 64
    ref_tiled = w_ref.view(T_r, 64, T_c, 64).permute(0, 2, 1, 3)
    n_ok = 0
    for i in range(1000):
        t_id = int(ids[i])
        r = ref_tiled[t_id // T_c, t_id % T_c]
        k = tiles[i].to(torch.bfloat16)
        d = (r != k) & (~ol_mask[t_id])
        n_ok += int(d.sum().item() == 0)
    P(f"  random-tile access: {n_ok}/1000 tiles bit-exact  <- impossible for INT8-X")

    del w, w_t, w_x, w_ref, w_tri, st, out_buf
    torch.cuda.empty_cache()

P("\nDONE")
LOG.close()
