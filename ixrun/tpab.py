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
                tile: int | None = None, outlier_frac: float = 0.01,
                k_cands: tuple = (0, 1, 2, 4, 8, 16, 32, 64),
                tile_r: int | None = None) -> dict:
    """bf16 weight [O, I] -> TPAB packed dict with RECTANGULAR tiles.

    Tiles are tile_r x TILE_C (TILE_C=64). tile_r auto-picks a divisor of O
    that is <= 64 (largest power-of-two divisor capped at 64; e.g. O=48 ->
    tile_r=48, O=1152 -> 48, O=5120 -> 64) so row-starved shapes like the
    Qwen delta-gates (48x5120) tesselate without padding. I must be a
    multiple of 64.

    Per-tile joint search over (outlier count k, bit width b): pick the
    cheapest combination meeting the SNR target.
    """
    O, I = w.shape
    if I % TILE:
        raise ValueError(f"in_f {I} not a multiple of {TILE}")
    if tile is not None:                    # legacy square-tile API
        tile_r = tile
    if tile_r is None:
        tile_r = TILE
        while O % tile_r and tile_r > 1:
            tile_r //= 2
        if O % tile_r:                      # odd primes (e.g. 43) fallback
            raise ValueError(f"out_f {O} has no power-of-two divisor <= 64")
    if O % tile_r:
        raise ValueError(f"out_f {O} not divisible by tile_r {tile_r}")
    T_r, T_c = O // tile_r, I // TILE
    T = T_r * T_c
    n_per = tile_r * TILE
    dev = w.device

    v = (w.float().view(T_r, tile_r, T_c, TILE).permute(0, 2, 1, 3)
          .reshape(T, n_per))

    # sorted magnitudes per tile (descending) for cheap k-threshold lookup
    abs_sorted = v.abs().sort(dim=1, descending=True).values     # [T, n_per]
    power = v.pow(2).sum(dim=1).clamp(min=1e-30)

    best_cost = torch.full((T,), float("inf"), device=dev)
    best_b = torch.zeros(T, dtype=torch.uint8, device=dev)
    best_s = torch.ones(T, dtype=torch.float16, device=dev)
    best_q = torch.zeros(T, n_per, dtype=torch.int32, device=dev)
    best_is_ol = torch.zeros(T, n_per, dtype=torch.bool, device=dev)

    OL_BYTE = 6.0        # per-outlier storage: idx(2B packable) + fp16(2B) ~6B
    for k in k_cands:
        if k == 0:
            thr = torch.full((T,), float("inf"), device=dev)
        else:
            thr = abs_sorted[:, k - 1]        # k-th largest magnitude
        is_ol = v.abs() > thr.unsqueeze(1)
        v2 = v * (~is_ol)
        M = v2.abs().amax(dim=1).clamp(min=1e-12)
        n_ol = is_ol.sum(dim=1).to(torch.int32)
        power_k = v2.pow(2).sum(dim=1).clamp(min=1e-30)

        for b in CAND_BITS:
            qmax = 2 ** (b - 1) - 1
            s = (M / qmax).half().float()
            q = (v2 / s.unsqueeze(1)).round().clamp(-qmax, qmax).to(torch.int32)
            # end-to-end error: include the decoder's bf16 rounding of the
            # reconstructed weight — at low bpw this term is a real share of
            # the error budget (search targets must match measured SNR)
            rec = (q.float() * s.unsqueeze(1)).to(torch.bfloat16).float()
            err = (v2 - rec).pow(2).sum(dim=1)
            # grid fidelity vs the in-grid power (outliers decode exactly,
            # so the layer-level error is dominated by this term)
            snr = 10 * torch.log10(power_k / err.clamp(min=1e-30))
            ok = snr >= snr_target_db
            cost = (n_per * b / 8.0) + n_ol * OL_BYTE + 7.0     # + header
            better = ok & (cost < best_cost)
            if better.any():
                best_cost[better] = cost[better]
                best_b[better] = b
                best_s[better] = s[better].half()
                best_q[better] = q[better]
                best_is_ol[better] = is_ol[better]

    # tiles that never met the target: force widest b, zero outliers
    fail = best_cost.isinf()
    if fail.any():
        b = CAND_BITS[-1]
        qmax = 2 ** (b - 1) - 1
        M = v.abs().amax(dim=1).clamp(min=1e-12)
        s = (M / qmax).half().float()
        q = (v / s.unsqueeze(1)).round().clamp(-qmax, qmax).to(torch.int32)
        best_b[fail] = b
        best_s[fail] = s[fail].half()
        best_q[fail] = q[fail]
        best_is_ol[fail] = False
    n_fallback = int(fail.sum().item())

    # materialize outliers from the EXACT sets chosen during the search
    # (recomputing thresholds by index is off-by-one under ties)
    is_ol = best_is_ol
    ol_t, ol_l = is_ol.nonzero(as_tuple=True)
    ol_val = v[ol_t, ol_l].to(torch.float16).cpu()

    bits, scales, q_sel = best_b, best_s, best_q

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
        "shape": (O, I), "tile": tile_r, "tile_r": tile_r, "tile_c": TILE,
        "T": T, "n_per": n_per,
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
    tr, tc = packed["tile_r"], packed["tile_c"]
    T_r, T_c = O // tr, I // tc
    return (out.view(T_r, T_c, tr, tc).permute(0, 2, 1, 3).reshape(O, I)).to(torch.bfloat16)


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
    tr, tc = packed["tile_r"], packed["tile_c"]
    T_r, T_c = O // tr, I // tc
    used = out[: T * n_per]  # shared workspace may be larger than this layer
    return (used.view(T_r, T_c, tr, tc).permute(0, 2, 1, 3).reshape(O, I)).to(torch.bfloat16)


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
    return out.view(n, packed["tile_r"], packed["tile_c"])
