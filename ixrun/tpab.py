"""TPAB: Tile-parallel Adaptive Bit-width weight compression.

Prototype successor to INT8-X. The (3,5,8) nested bitmap needs global rank
prefixes (sequential walks + cumsum) to locate each value's stream position,
capping decode throughput and blocking random access. TPAB cuts the weight
into 64x64 tiles, each independently coded:

    tile header : fp16 scale + bit width b in {2..6} (smallest b meeting a
                  per-tile SNR target; ~1% outliers escape first, shrinking
                  the tile's dynamic range)
    tile body   : fixed-width b-bit codes. Element L of tile t sits at bit
                  group_base[b] + tile_off[t] * b + L * b — pure bit
                  extraction, no ranks, no cumsum, ANY tile in ANY order.

Bodies are grouped by bit width (each group fixed-width => one batched pack);
`goff` gives each tile's element offset inside its group, `gbase` each
group's bit base in the concatenated stream.

decode_tpab_ref   — vectorized torch reference
decode_tpab_triton— tile-parallel Triton kernel (grid = tiles x chunks)
decode_tiles      — random-access decode of an arbitrary subset of tiles
                    (impossible for INT8-X without a full sequential walk)
"""
from __future__ import annotations
import torch

from .bitpack import pack_bits_stream, unpack_bits_stream

__all__ = ["encode_tpab", "decode_tpab_ref", "decode_tpab_triton", "decode_tiles",
           "TILE", "CAND_BITS"]

TILE = 64
CAND_BITS = (2, 3, 4, 5, 6)


@torch.no_grad()
def encode_tpab(w: torch.Tensor, snr_target_db: float = 30.0,
                tile: int = TILE, outlier_frac: float = 0.01,
                max_tile_chunk: int = 4096) -> dict:
    """bf16 weight [O, I] (O, I multiples of `tile`) -> TPAB packed dict."""
    O, I = w.shape
    if O % tile or I % tile:
        raise ValueError(f"shape {O}x{I} not divisible by tile {tile}")
    T_r, T_c = O // tile, I // tile
    T = T_r * T_c
    n_per = tile * tile
    dev = w.device

    v = w.float().view(T_r, tile, T_c, tile).permute(0, 2, 1, 3).reshape(T, n_per)

    k = max(1, int(outlier_frac * n_per))
    top = v.abs().topk(k + 1, dim=1).values          # [T, k+1] descending
    thr = top[:, k]                                   # (k+1)-th largest
    is_ol = v.abs() > thr.unsqueeze(1)
    v2 = v * (~is_ol)

    ol_t, ol_l = is_ol.nonzero(as_tuple=True)
    ol_val = v[ol_t, ol_l].to(torch.float16).cpu()

    power = v2.pow(2).sum(dim=1).clamp(min=1e-30)
    M = v2.abs().amax(dim=1).clamp(min=1e-12)

    bits = torch.zeros(T, dtype=torch.uint8, device=dev)
    scales = torch.ones(T, dtype=torch.float16, device=dev)
    q_sel = torch.zeros(T, n_per, dtype=torch.int32, device=dev)
    assigned = torch.zeros(T, dtype=torch.bool, device=dev)

    for b in CAND_BITS:
        qmax = 2 ** (b - 1) - 1
        s = (M / qmax).half().float()                 # fp16-round BEFORE quantize
        q = (v2 / s.unsqueeze(1)).round().clamp(-qmax, qmax).to(torch.int32)
        err = (v2 - q.float() * s.unsqueeze(1)).pow(2).sum(dim=1)
        ok = (10 * torch.log10(power / err.clamp(min=1e-30)) >= snr_target_db) & (~assigned)
        if ok.any():
            bits[ok] = b
            scales[ok] = s[ok].half()
            q_sel[ok] = q[ok]
            assigned |= ok
    if (~assigned).any():                             # fallback: widest
        b = CAND_BITS[-1]
        qmax = 2 ** (b - 1) - 1
        s = (M / qmax).half().float()
        q = (v2 / s.unsqueeze(1)).round().clamp(-qmax, qmax).to(torch.int32)
        bits[~assigned] = b
        scales[~assigned] = s[~assigned].half()
        q_sel[~assigned] = q[~assigned]
    n_fallback = int((~assigned).sum().item())

    # group bodies by bit width; tiles keep order inside their group
    goff = torch.zeros(T, dtype=torch.int64)          # element offset in group
    gbase_bit = [0] * (max(CAND_BITS) + 1)            # bit base per b
    bodies = []
    cursor_bit = 0
    for b in CAND_BITS:
        sel = bits == b
        n_t = int(sel.sum())
        if n_t == 0:
            bodies.append(torch.zeros(0, dtype=torch.int32))
            gbase_bit[b] = cursor_bit                 # empty group base
            continue
        qmax = 2 ** (b - 1) - 1
        codes = (q_sel[sel] + qmax).to(torch.int32).reshape(-1).cpu()
        body = pack_bits_stream(codes, b)
        bodies.append(body)
        idx = sel.nonzero(as_tuple=True)[0]
        goff[idx] = torch.arange(n_t, dtype=torch.int64) * n_per
        gbase_bit[b] = cursor_bit
        cursor_bit += n_t * n_per * b

    total_bytes = (
        sum(bd.numel() * 4 for bd in bodies)
        + T * 2 + T + T * 4                            # scales, bits, goff(int32)
        + len(ol_t) * 6                                # outlier idx(2x int16-packable) + fp16
    )
    return {
        "shape": (O, I), "tile": tile, "T": T, "n_per": n_per,
        "bits": bits.cpu(), "scales": scales.cpu(),
        "goff": goff.to(torch.int32).cpu(),
        "gbase_bit": torch.tensor(gbase_bit, dtype=torch.int64),
        "bodies": bodies,
        "ol_t": ol_t.to(torch.int32).cpu(), "ol_l": ol_l.to(torch.int32).cpu(),
        "ol_val": ol_val,
        "total_bytes": total_bytes,
        "bpw": total_bytes * 8 / (O * I),
        "n_outliers": int(len(ol_t)),
        "fallback_tiles": n_fallback,
    }


@torch.no_grad()
def decode_tpab_ref(packed: dict, device=None) -> torch.Tensor:
    """Vectorized torch reference decode (tile-layout -> [O, I])."""
    O, I = packed["shape"]
    T, n_per = packed["T"], packed["n_per"]
    dev = device or torch.device("cpu")
    out = torch.zeros(T, n_per, dtype=torch.float32, device=dev)
    for gi, b in enumerate(CAND_BITS):
        sel = packed["bits"] == b
        n_t = int(sel.sum())
        if n_t == 0:
            continue
        qmax = 2 ** (b - 1) - 1
        codes = unpack_bits_stream(packed["bodies"][gi].to(dev), n_t * n_per, b, device=dev)
        q = codes.view(n_t, n_per).float() - qmax
        idx = sel.nonzero(as_tuple=True)[0].to(dev)
        s = packed["scales"][sel].to(dev).float().unsqueeze(1)
        out[idx] = q * s
    ol_t = packed["ol_t"].to(dev)
    out[ol_t, packed["ol_l"].to(dev)] = packed["ol_val"].to(dev).float()
    T_r, T_c = O // packed["tile"], I // packed["tile"]
    t = packed["tile"]
    return (out.view(T_r, T_c, t, t).permute(0, 2, 1, 3).reshape(O, I)).to(torch.bfloat16)


# --------------------------------------------------------------------------- #
#  Triton tile-parallel kernels
# --------------------------------------------------------------------------- #
try:
    import triton
    import triton.language as tl

    _HAS = True
except Exception:
    _HAS = False

if _HAS:

    @triton.jit
    def _tpab_decode_kernel(
        out_ptr,             # [T * n_per] f32 (tile layout)
        tiles_ptr,           # [n_launch] tile ids to decode (or arange(T))
        body_ptr,            # uint32 stream
        bits_ptr, scales_ptr, goff_ptr,
        gbase_ptr,           # [8] int64 bit base per bit-width group
        N_LAUNCH: tl.constexpr,
        N_PER: tl.constexpr,
        BLK: tl.constexpr,
    ):
        pid_t = tl.program_id(0)
        pid_c = tl.program_id(1)
        t = tl.load(tiles_ptr + pid_t).to(tl.int32)   # which tile's DATA to read
        # output slot follows the LAUNCH order (== t for full decode; lets
        # decode_tiles gather a subset into a compact buffer)

        b = tl.load(bits_ptr + t).to(tl.int32)
        s = tl.load(scales_ptr + t).to(tl.float32)
        gbase = tl.load(gbase_ptr + b)
        goff = tl.load(goff_ptr + t).to(tl.int64)

        L = pid_c * BLK + tl.arange(0, BLK)
        bitpos = gbase + (goff + L.to(tl.int64)) * b
        word = (bitpos // 32).to(tl.int32)
        shift = (bitpos % 32).to(tl.int32)

        w1 = tl.load(body_ptr + word).to(tl.uint32)
        cross = (shift + b) > 32
        w2 = tl.where(cross, tl.load(body_ptr + word + 1).to(tl.uint32),
                      tl.zeros((BLK,), tl.uint32))
        raw = tl.where(cross,
                       (w1 >> shift) | (w2 << (32 - shift)),
                       w1 >> shift)
        mask = tl.exp2(b.to(tl.float32)).to(tl.int32) - 1
        raw = (raw & mask.to(tl.uint32)).to(tl.int32)
        v = raw - ((mask + 1) // 2 - 1)
        w = v.to(tl.float32) * s
        tl.store(out_ptr + pid_t.to(tl.int64) * N_PER + L, w)


def stage_gpu(packed: dict, device="cuda") -> dict:
    """Upload packed arrays to GPU once (kernels take pre-staged dicts)."""
    dev = torch.device(device)
    out = dict(packed)
    out["bodies_g"] = torch.cat(packed["bodies"]).to(dev)
    out["bits_g"] = packed["bits"].to(dev)
    out["scales_g"] = packed["scales"].to(dev)
    out["goff_g"] = packed["goff"].to(dev)
    out["gbase_g"] = packed["gbase_bit"].to(dev)
    out["ol_t_g"] = packed["ol_t"].to(dev)
    out["ol_l_g"] = packed["ol_l"].to(dev)
    out["ol_val_g"] = packed["ol_val"].to(dev)
    return out


def _launch_decode(out, tiles, st, n, n_per, device):
    BLK = 256
    _tpab_decode_kernel[(n, n_per // BLK)](
        out, tiles,
        st["bodies_g"], st["bits_g"], st["scales_g"], st["goff_g"], st["gbase_g"],
        N_LAUNCH=n, N_PER=n_per, BLK=BLK, num_warps=2,
    )


def decode_tpab_triton(packed: dict, device="cuda", out_f32: torch.Tensor | None = None,
                       staged: dict | None = None):
    """Full decode [O, I] bf16 via the tile-parallel kernel + outlier scatter.

    `staged`: result of stage_gpu(packed) to avoid re-uploads in hot loops.
    """
    assert _HAS
    dev = torch.device(device)
    st = staged or stage_gpu(packed, device)
    T, n_per = packed["T"], packed["n_per"]
    out = out_f32 if out_f32 is not None else torch.zeros(T * n_per, dtype=torch.float32, device=dev)
    tiles = torch.arange(T, dtype=torch.int32, device=dev)
    _launch_decode(out, tiles, st, T, n_per, dev)
    flat = st["ol_t_g"].to(torch.int64) * n_per + st["ol_l_g"].to(torch.int64)
    out[flat] = st["ol_val_g"].float()
    O, I = packed["shape"]
    T_r, T_c = O // packed["tile"], I // packed["tile"]
    t = packed["tile"]
    return (out.view(T_r, T_c, t, t).permute(0, 2, 1, 3).reshape(O, I)).to(torch.bfloat16)


@torch.no_grad()
def decode_tiles(packed: dict, tile_ids: torch.Tensor, device="cuda",
                 staged: dict | None = None) -> torch.Tensor:
    """Random-access decode of an arbitrary subset of tiles: [n, tile, tile] f32.

    Outliers are NOT applied here (they live outside the tile streams) —
    callers needing exact values should overlay them from the packed table.
    """
    assert _HAS
    dev = torch.device(device)
    st = staged or stage_gpu(packed, device)
    n = tile_ids.numel()
    n_per = packed["n_per"]
    out = torch.zeros(n * n_per, dtype=torch.float32, device=dev)
    tiles = tile_ids.to(torch.int32).to(dev)
    _launch_decode(out, tiles, st, n, n_per, dev)
    return out.view(n, packed["tile"], packed["tile"])
