"""PEAK-Q: Peak-Exact Adaptive K-bit quantization (bf16 re-encoding).

Design (measured on MiniCPM5-1B weights, group=16 consecutive elements):

  Per group of 16: emax = max bf16 exponent byte (0.5 bpw overhead).
  Per element: delta = emax - exponent. Real distribution:
    delta<=1: 46.3%   delta<=3: 85.0%   delta>=8: 0.93%

  Tiered payload, tier chosen by TRUE delta (saturated value stored):

    T1  delta<=1  : mant7 + d1 = 8 bits  -> BIT-EXACT bf16 (all group peaks)
    T2  delta<=3  : mant6 + d1 = 7 bits  -> err <= 2^-10 of group peak (RTN)
    T3  delta<=7+ : mant5 + d2 = 7 bits  -> err <= 2^-10 of group peak;
                     delta>7 saturates to 7 (0.93% of weights, err ~2^-7 peak)

  Overheads: sign 1b + emax 0.5b + B1 bitmap 1b + B2 bitmap ~0.54b.
  Total 10.50 bpw (1.52x vs bf16), ~14% smaller than flat BF16X (12.26 bpw)
  while keeping every delta<=1 weight bit-exact (69% of elements measured:
  all T1 exact + even-mantissa T2 + aligned T3).

  Error profile vs INT8-X (3,5,8): int8 rounds EVERY element to 1/127 of the
  TENSOR max (peaks included). PEAK-Q keeps peaks exact and bounds per-element
  error relative to the GROUP peak (~4-8x below tensor max) — measured +21 to
  +35 dB SNR on MiniCPM5 weights (54 dB vs 20-33 dB) at 10.50 bpw.

  Acceleration by construction (no BF16X-style sparse fixups):
    * no overflow table in the hot path — saturation into T3 instead, so the
      decode is ONE kernel, no CPU sync, no per-layer DMA, CUDA-Graph safe;
    * all streams fixed-width & GPU-resident; nested-bitmap + tl.cumsum rank
      derivation copied from the proven `_ix_decode_kernel` algebra;
    * T1 stream is byte-aligned (8b payloads never cross int32 words);
      sign stream is word-aligned (1b); emax lookup is a single shift (//16).
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

from .bitpack import pack_bits_stream, unpack_bits_stream, pack_bitmap
from .config import SKIP_PATTERNS, MIN_LINEAR_ELEMS, TRITON_BLOCK
from .linear import iter_quantizable_linears, _set_parent_child

__all__ = [
    "PEAKQ_TIERS",
    "peakq_quantize",
    "decode_peakq_scatter",
    "decode_peakq_triton",
    "PeakQLinear",
    "deploy_peakq",
    "deploy_peakq_lazy",
    "peakq_snr",
    "peakq_exact_pct",
]

# (delta_max, mantissa_bits) per tier, ascending; last tier saturates delta.
# Payload bits = mantissa_bits + bits(delta_max - prev_delta_max).
# Triton kernel supports exactly this 3-tier layout ((1,7),(3,6),(7,5)).
PEAKQ_TIERS = ((1, 7), (3, 6), (7, 5))
PEAKQ_GROUP = 16

_HAS_TRITON = False
try:
    import triton
    import triton.language as tl

    _HAS_TRITON = True
except Exception:
    pass


# --------------------------------------------------------------------------- #
#  Quantize (pure torch, CPU-friendly)
# --------------------------------------------------------------------------- #
@torch.no_grad()
def peakq_quantize(
    weight: torch.Tensor,
    group: int = PEAKQ_GROUP,
    tiers: tuple = PEAKQ_TIERS,
    layout: str = "global",
) -> dict:
    """Pack a bf16 weight matrix into PEAK-Q tiered streams.

    layout='global' (v1): tier streams concatenated globally; decode needs
        block/row prefix tables (cumsum rank algebra).
    layout='rows' (v2, TPAB-inspired): every row restarts each tier stream
        at a word boundary; B2 becomes a per-row bitmap. Row offsets replace
        ALL prefix computation -> rows decode independently (random access,
        no compute_row_prefixes at deploy, multi-row GEMV for free).
        Costs < ~31 padding bits/row/stream (<0.1% bpw).

    Returns dict: out_f, in_f, N, group, tiers, layout, emax (uint8/16),
    sign_packed (1-bit stream), bitmaps (nested B1/B2 for v1; B1 only for
    v2), streams (3 payloads), counts, total_bytes, bits_per_weight,
    compression_vs_bf16. v2 additionally: t1_off / t2_bit_off / t3_bit_off /
    b2_bit_off (int32 [out_f+1]) and b2_rows (per-row B2 words).
    """
    tiers = tuple(tuple(t) for t in tiers)
    if len(tiers) != 3:
        raise ValueError("PEAK-Q packer expects exactly 3 tiers (kernel contract)")
    of, inf_ = weight.shape
    # packing is one-time; do all bit bookkeeping on CPU (mirrors int8x_quantize)
    w16 = weight.detach().to(torch.bfloat16).reshape(-1).cpu().contiguous()
    N = w16.numel()

    bits = w16.view(torch.int16).to(torch.int64)
    sign = (bits >> 15) & 1
    expo = (bits >> 7) & 0xFF
    mant = bits & 0x7F

    # group emax over exponents (pad with 0s: never raises the max)
    pad = (-N) % group
    e_pad = expo if pad == 0 else torch.cat(
        [expo, torch.zeros(pad, dtype=torch.int64)]
    )
    emax = e_pad.view(-1, group).amax(-1)                     # (Ng,)

    gidx = torch.arange(N) // group
    delta_true = (emax[gidx] - expo).clamp(min=0)             # true distance
    dmax_last = tiers[-1][0]
    delta_sat = delta_true.clamp(max=dmax_last)               # stored value

    # tier masks from TRUE delta (saturation only affects storage)
    m1 = delta_true <= tiers[0][0]
    rest = ~m1
    m2 = rest & (delta_true <= tiers[1][0])
    m3 = rest & ~m2
    masks = [m1, m2, m3]

    # mantissa round-to-nearest truncation per tier
    mants, lows = [], []
    bounds = [-1] + [t[0] for t in tiers]
    for i, (_, mbits) in enumerate(tiers):
        shift = 7 - mbits
        if shift > 0:
            half = 1 << (shift - 1)
            m_t = ((mant + half) >> shift).clamp(max=(1 << mbits) - 1)
        else:
            m_t = mant
        lo = bounds[i] + 1
        low = (delta_sat - lo).clamp(min=0)                   # delta - tier base
        mants.append(m_t)
        lows.append(low)

    # tier payload widths: mant bits + delta-field bits.
    # delta in tier i spans [lo, hi] with lo = bounds[i]+1, hi = tiers[i][0];
    # stored low = delta - lo in [0, hi-lo] -> needs ceil(log2(hi-lo+1)) bits.
    payload_bits = []
    for i, (_, mbits) in enumerate(tiers):
        dspan = tiers[i][0] - (bounds[i] + 1)          # hi - lo
        dbits = max(1, dspan.bit_length())
        payload_bits.append(mbits + dbits)

    if layout == "rows":
        return _pack_rows_layout(
            w16, of, inf_, N, group, tiers, payload_bits,
            sign, emax, masks, mants, lows,
        )

    streams = []
    for i in range(3):
        vals = (mants[i] | (lows[i] << tiers[i][1])).to(torch.int64)[masks[i]]
        if payload_bits[i] == 8:
            streams.append(vals.to(torch.uint8).cpu().contiguous())
        else:
            streams.append(pack_bits_stream(vals, payload_bits[i]))

    bitmaps = []
    remaining = torch.ones(N, dtype=torch.bool)
    for i in range(2):
        bitmaps.append(pack_bitmap(masks[i][remaining]))
        remaining = remaining & ~masks[i]

    counts = [int(m.sum().item()) for m in masks]

    total = 0
    total += emax.numel()                                      # 1 byte/group
    total += pack_bits_stream(sign, 1).numel() * 4
    for s, pb in zip(streams, payload_bits):
        total += s.numel() if pb == 8 else s.numel() * 4
    for bm in bitmaps:
        total += bm.numel() * 4

    return {
        "group": group,
        "tiers": tiers,
        "layout": "global",
        "payload_bits": payload_bits,
        "out_f": of,
        "in_f": inf_,
        "N": N,
        "emax": emax.to(torch.uint8).cpu().contiguous(),
        "sign_packed": pack_bits_stream(sign, 1).cpu().contiguous(),
        "bitmaps": bitmaps,
        "streams": streams,
        "counts": counts,
        "total_bytes": total,
        "bits_per_weight": (total * 8) / N,
        "compression_vs_bf16": (N * 2) / total,
    }


# --------------------------------------------------------------------------- #
#  v2 "rows" layout: per-row stream restart (TPAB-style local addressing)
# --------------------------------------------------------------------------- #
def _row_ranks(pos: torch.Tensor, in_f: int, out_f: int):
    """For sorted flat positions, return (rows, local_rank_within_row)."""
    rows = pos // in_f
    starts = torch.searchsorted(
        pos, torch.arange(out_f, dtype=torch.int64) * in_f
    )
    lr = torch.arange(pos.numel(), dtype=torch.int64) - starts[rows]
    return rows, lr


def _scatter_bits(vals: torch.Tensor, bit_pos: torch.Tensor, total_bits: int,
                  bits: int) -> torch.Tensor:
    """Scatter fixed-width values at absolute (non-overlapping) bit positions.

    Bits never collide so sum == OR; int64 accumulation avoids sign issues.
    Chunked to cap temporaries on very large streams.
    """
    nw = (total_bits + 31) // 32
    out = torch.zeros(nw, dtype=torch.int64)
    n = vals.numel()
    step = 1 << 22
    for s in range(0, n, step):
        v = vals[s : s + step]
        bp = bit_pos[s : s + step]
        for b in range(bits):
            p = bp + b
            out.scatter_add_(0, p >> 5, ((v >> b) & 1) << (p & 31))
    return out.to(torch.int32)


def _gather_bits(words: torch.Tensor, bit_pos: torch.Tensor, bits: int) -> torch.Tensor:
    """Inverse of _scatter_bits (gather fixed-width values at bit positions)."""
    out = torch.zeros(bit_pos.numel(), dtype=torch.int64, device=bit_pos.device)
    for b in range(bits):
        p = bit_pos + b
        out |= ((words[(p >> 5)].to(torch.int64) >> (p & 31)) & 1) << b
    return out


def _pack_rows_layout(w16, of, inf_, N, group, tiers, payload_bits,
                      sign, emax, masks, mants, lows) -> dict:
    """v2 packing: each row's tier streams + B2 bitmap restart at word bounds."""
    out_f, in_f = of, inf_
    t1_off = torch.zeros(out_f + 1, dtype=torch.int64)
    t2_bit_off = torch.zeros(out_f + 1, dtype=torch.int64)
    t3_bit_off = torch.zeros(out_f + 1, dtype=torch.int64)
    b2_bit_off = torch.zeros(out_f + 1, dtype=torch.int64)

    # ---- per-tier row streams ----
    streams = []
    for i in range(3):
        vals = (mants[i] | (lows[i] << tiers[i][1])).to(torch.int64)[masks[i]]
        pos = masks[i].nonzero(as_tuple=True)[0]
        rows, lr = _row_ranks(pos, in_f, out_f)
        cnt = torch.bincount(rows, minlength=out_f)
        if payload_bits[i] == 8:
            # raw uint8: per-row segments, element offsets
            t1_off[1:] = cnt.cumsum(0)
            buf = torch.zeros(int(t1_off[-1]), dtype=torch.uint8)
            buf[(t1_off[rows] + lr).to(torch.int64)] = vals.to(torch.uint8)
            streams.append(buf.contiguous())
        else:
            words = (cnt * payload_bits[i] + 31) // 32
            offs = torch.zeros(out_f + 1, dtype=torch.int64)
            offs[1:] = words.cumsum(0) * 32      # exclusive bit offset per row
            if i == 1:
                t2_bit_off = offs
            else:
                t3_bit_off = offs
            bit_pos = offs[rows] + lr * payload_bits[i]
            streams.append(_scatter_bits(vals, bit_pos, int(offs[-1]),
                                         payload_bits[i]).contiguous())

    # ---- per-row B2 bitmap (indexed by within-row non-T1 rank) ----
    m1 = masks[0]
    nt1 = (~m1).nonzero(as_tuple=True)[0]
    nrows, nlr = _row_ranks(nt1, in_f, out_f)
    nt1_cnt = torch.bincount(nrows, minlength=out_f)
    b2_words = (nt1_cnt + 31) // 32
    b2_offs = torch.zeros(out_f + 1, dtype=torch.int64)
    b2_offs[1:] = b2_words.cumsum(0) * 32
    b2_bit_off = b2_offs
    b2_flags = masks[1][nt1].to(torch.int64)
    b2_rows_buf = _scatter_bits(
        b2_flags, b2_offs[nrows] + nlr, int(b2_offs[-1]), 1
    ).contiguous()

    sign_packed = pack_bits_stream(sign, 1)
    b1 = pack_bitmap(m1)
    counts = [int(m.sum().item()) for m in masks]

    total = 0
    total += emax.numel()
    total += sign_packed.numel() * 4
    total += b1.numel() * 4
    total += streams[0].numel()                      # uint8 T1
    total += streams[1].numel() * 4
    total += streams[2].numel() * 4
    total += b2_rows_buf.numel() * 4
    total += (out_f + 1) * 4 * 4                     # 4 offset arrays

    for arr in (t1_off, t2_bit_off, t3_bit_off, b2_bit_off):
        assert int(arr[-1]) < 2**31, "stream offsets exceed int32"

    return {
        "group": group,
        "tiers": tiers,
        "layout": "rows",
        "payload_bits": payload_bits,
        "out_f": out_f,
        "in_f": in_f,
        "N": N,
        "emax": emax.to(torch.uint8).cpu().contiguous(),
        "sign_packed": sign_packed.cpu().contiguous(),
        "bitmaps": [b1],                             # B1 only (flat-indexed)
        "b2_rows": b2_rows_buf,
        "t1_off": t1_off.to(torch.int32).contiguous(),
        "t2_bit_off": t2_bit_off.to(torch.int32).contiguous(),
        "t3_bit_off": t3_bit_off.to(torch.int32).contiguous(),
        "b2_bit_off": b2_bit_off.to(torch.int32).contiguous(),
        "streams": streams,
        "counts": counts,
        "total_bytes": total,
        "bits_per_weight": (total * 8) / N,
        "compression_vs_bf16": (N * 2) / total,
    }


def _rebuild_masks(packed: dict) -> list:
    """Rebuild the 3 tier masks (one-time, CPU). Handles both layouts."""
    N = packed["N"]
    from .bitpack import unpack_bits_stream as _ubs

    if packed.get("layout", "global") == "rows":
        m1 = _ubs(packed["bitmaps"][0], N, 1).bool()
        in_f, out_f = packed["in_f"], packed["out_f"]
        nt1 = (~m1).nonzero(as_tuple=True)[0]
        nrows, nlr = _row_ranks(nt1, in_f, out_f)
        b2_pos = packed["b2_bit_off"].to(torch.int64)[nrows] + nlr
        is2 = _gather_bits(packed["b2_rows"], b2_pos, 1)
        m2 = torch.zeros(N, dtype=torch.bool)
        m2[nt1] = is2.bool()
        return [m1, m2, (~m1) & ~m2]

    remaining = torch.ones(N, dtype=torch.bool)
    masks = []
    for i in range(2):
        bm_vals = _ubs(packed["bitmaps"][i], int(remaining.sum()), 1)
        cur = torch.zeros(N, dtype=torch.bool)
        idx = remaining.nonzero(as_tuple=True)[0]
        cur[idx[: len(bm_vals)]] = bm_vals.bool()
        cur = cur & remaining
        masks.append(cur)
        remaining = remaining & ~cur
    masks.append(remaining.clone())
    return masks


def _tier_positions(packed: dict) -> list:
    masks = _rebuild_masks(packed)
    return [m.nonzero(as_tuple=True)[0].to(torch.int64) for m in masks]


# --------------------------------------------------------------------------- #
#  Reference decode (pure torch scatter, no Triton)
# --------------------------------------------------------------------------- #
@torch.no_grad()
def decode_peakq_scatter(packed: dict, device=None, dtype=torch.bfloat16) -> torch.Tensor:
    """Decode PEAK-Q packed dict -> bf16 weight (reference path)."""
    device = device or torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    N, group = packed["N"], packed["group"]
    tiers = packed["tiers"]
    bounds = [-1] + [t[0] for t in tiers]

    sign = unpack_bits_stream(packed["sign_packed"], N, 1, device=device)
    emax = packed["emax"].to(device).to(torch.int64)
    gidx = torch.arange(N, device=device) // group
    rows_layout = packed.get("layout", "global") == "rows"
    in_f, out_f = packed["in_f"], packed["out_f"]
    tier_bit_off = {1: "t2_bit_off", 2: "t3_bit_off"}

    mant_flat = torch.zeros(N, dtype=torch.int64, device=device)
    delta_flat = torch.zeros(N, dtype=torch.int64, device=device)
    positions = _tier_positions(packed)
    for i in range(3):
        cnt = packed["counts"][i]
        if cnt == 0:
            continue
        pos = positions[i].to(device)
        pb = packed["payload_bits"][i]
        if rows_layout:
            rows, lr = _row_ranks(pos.cpu(), in_f, out_f)
            rows, lr = rows.to(device), lr.to(device)
            if pb == 8:
                off = packed["t1_off"].to(device).to(torch.int64)
                vals = packed["streams"][i].to(device)[off[rows] + lr].to(torch.int64)
            else:
                off = packed[tier_bit_off[i]].to(device).to(torch.int64)
                bit_pos = off[rows] + lr * pb
                vals = _gather_bits(packed["streams"][i].to(device), bit_pos, pb)
        else:
            if pb == 8:
                vals = packed["streams"][i][:cnt].to(torch.int64).to(device)
            else:
                vals = unpack_bits_stream(packed["streams"][i], cnt, pb, device=device)
        mbits = tiers[i][1]
        m = ((vals & ((1 << mbits) - 1)) << (7 - mbits))
        low = vals >> mbits
        mant_flat[pos] = m.to(torch.int64)
        delta_flat[pos] = (bounds[i] + 1 + low).to(torch.int64)

    expo = (emax[gidx] - delta_flat).clamp(0, 255)
    vbits = (sign << 15) | (expo << 7) | mant_flat
    w = vbits.to(torch.int16).view(torch.bfloat16).to(dtype)
    return w.view(packed["out_f"], packed["in_f"])


# --------------------------------------------------------------------------- #
#  Fused Triton decode (single kernel, no fixups, CUDA-Graph safe)
# --------------------------------------------------------------------------- #
if _HAS_TRITON:

    @triton.jit
    def _peakq_decode_kernel(
        out_ptr,
        sign_ptr,
        emax_ptr,
        b1_ptr,
        b2_ptr,
        t1_ptr,
        t2_ptr,
        t3_ptr,
        b1_blk_ptr,
        b2_blk_ptr,
        N: tl.constexpr,
        GROUP: tl.constexpr,
        BLK: tl.constexpr,
    ):
        """Fused 3-tier PEAK-Q decode. Rank algebra mirrors `_ix_decode_kernel`."""
        pid = tl.program_id(0)
        offs = pid * BLK + tl.arange(0, BLK)
        mask = offs < N

        # sign: word-aligned 1-bit stream (never crosses words)
        sw = tl.load(sign_ptr + (offs // 32), mask=mask, other=0).to(tl.uint32)
        sign = (sw >> (offs % 32)) & 1

        # --- T1: bitmap B1 + cumsum rank -> byte-aligned 8-bit payload ---
        b1w = tl.load(b1_ptr + (offs // 32), mask=mask, other=0).to(tl.uint32)
        is_t1 = ((b1w >> (offs % 32)) & 1).to(tl.int32)

        t1_before = tl.load(b1_blk_ptr + pid)
        t1_local = tl.cumsum(is_t1, axis=0) - 1
        t1_rank = tl.where(is_t1 == 1, t1_before + t1_local, 0)

        p1 = t1_rank                                # uint8 stream: 1 byte per elem
        v1 = tl.load(t1_ptr + p1, mask=mask, other=0).to(tl.int32)
        mant1 = v1 & 0x7F                           # 7-bit mantissa, exact
        delta1 = v1 >> 7                            # delta in {0,1}

        # --- non-T1 rank derived algebraically (no 2nd cumsum for ranks) ---
        nt1_bit = 1 - is_t1
        nt1_before = pid * BLK - t1_before
        nt1_local = tl.arange(0, BLK) - t1_local - 1
        nt1_rank = tl.where(nt1_bit == 1, nt1_before + nt1_local, 0)

        # --- T2: bitmap B2 (indexed over non-T1) + cumsum rank ---
        b2v = tl.load(b2_ptr + (nt1_rank // 32), mask=mask, other=0).to(tl.uint32)
        is_t2 = ((b2v >> (nt1_rank % 32)) & 1).to(tl.int32)
        is_t2 = tl.where(nt1_bit == 1, is_t2, 0)

        t2_before = tl.load(b2_blk_ptr + pid)
        t2_local = tl.cumsum(is_t2, axis=0) - 1
        t2_rank = tl.where(is_t2 == 1, t2_before + t2_local, 0)

        p2 = t2_rank * 7                        # 7-bit payload, may cross words
        wi2 = p2 // 32
        s2 = p2 % 32
        w2a = tl.load(t2_ptr + wi2, mask=mask, other=0).to(tl.uint32)
        c2 = (s2 + 7) > 32
        w2b = tl.where(
            c2,
            tl.load(t2_ptr + wi2 + 1, mask=mask, other=0).to(tl.uint32),
            tl.zeros((BLK,), tl.uint32),
        )
        v2 = tl.where(
            c2, ((w2a >> s2) | (w2b << (32 - s2))) & 0x7F, (w2a >> s2) & 0x7F
        )
        mant2 = (v2 & 0x3F) << 1                # 6-bit mantissa
        delta2 = 2 + (v2 >> 6)                  # delta in {2,3}

        # --- T3: rank among (non-T1 AND non-T2), derived (no 3rd cumsum) ---
        is_t3 = tl.where(nt1_bit == 1, 1 - is_t2, 0)
        t3_total_before = nt1_before - t2_before
        t3r = t3_total_before + (nt1_local - t2_local - 1)
        t3r = tl.where(is_t3 == 1, t3r, 0)
        t3r_safe = tl.where(t3r < 0, 0, t3r)

        p3 = t3r_safe * 7
        wi3 = p3 // 32
        s3 = p3 % 32
        w3a = tl.load(t3_ptr + wi3, mask=mask, other=0).to(tl.uint32)
        c3 = (s3 + 7) > 32
        w3b = tl.where(
            c3,
            tl.load(t3_ptr + wi3 + 1, mask=mask, other=0).to(tl.uint32),
            tl.zeros((BLK,), tl.uint32),
        )
        v3 = tl.where(
            c3, ((w3a >> s3) | (w3b << (32 - s3))) & 0x7F, (w3a >> s3) & 0x7F
        )
        mant3 = (v3 & 0x1F) << 2                # 5-bit mantissa
        delta3 = 4 + (v3 >> 5)                  # delta in {4..7}, saturated

        # --- assemble bf16 bits ---
        mant = tl.where(is_t1 == 1, mant1.to(tl.int32),
                        tl.where(is_t2 == 1, mant2.to(tl.int32),
                                 mant3.to(tl.int32)))
        delta = tl.where(is_t1 == 1, delta1.to(tl.int32),
                         tl.where(is_t2 == 1, delta2.to(tl.int32),
                                  delta3.to(tl.int32)))

        e8 = tl.load(emax_ptr + (offs // GROUP), mask=mask, other=0).to(tl.int32)
        expo = e8 - delta
        expo = tl.minimum(tl.maximum(expo, 0), 255)

        w16 = ((sign.to(tl.int32) << 15) | (expo << 7) | mant).to(tl.uint16)
        tl.store(out_ptr + offs, w16.to(tl.bfloat16, bitcast=True), mask=mask)


if _HAS_TRITON:

    @triton.jit
    def _peakq_decode_v2_kernel(
        out_ptr,
        sign_ptr, emax_ptr,
        b1_ptr, b2r_ptr,
        t1_ptr, t2_ptr, t3_ptr,
        t1o_ptr, t2o_ptr, t3o_ptr, b2o_ptr,   # int32 [out_f+1] row offsets
        IN_F: tl.constexpr,
        GROUP: tl.constexpr,
        BK: tl.constexpr,
    ):
        """v2 rows-layout decode: one program per row, rank counters start at
        zero (streams restart each row) — no prefix tables at all."""
        r = tl.program_id(0)
        base = r * IN_F
        t1b = tl.load(t1o_ptr + r)
        t2b = tl.load(t2o_ptr + r)
        t3b = tl.load(t3o_ptr + r)
        b2b = tl.load(b2o_ptr + r)

        t1c = tl.zeros((), tl.int32)
        t2c = tl.zeros((), tl.int32)
        nt1c = tl.zeros((), tl.int32)

        for k0 in tl.range(0, IN_F, BK):
            offs = base + k0 + tl.arange(0, BK)

            sw = tl.load(sign_ptr + offs // 32).to(tl.uint32)
            sign = ((sw >> (offs % 32)) & 1).to(tl.int32)
            e8 = tl.load(emax_ptr + offs // GROUP).to(tl.int32)

            b1v = tl.load(b1_ptr + offs // 32).to(tl.uint32)
            is_t1 = ((b1v >> (offs % 32)) & 1).to(tl.int32)
            t1_local = tl.cumsum(is_t1, axis=0) - 1
            t1_rank = tl.where(is_t1 == 1, t1c + t1_local, 0)
            v1 = tl.load(t1_ptr + t1b + t1_rank).to(tl.int32)
            expo1 = tl.minimum(tl.maximum(e8 - (v1 >> 7), 0), 255)
            w1 = ((sign << 15) | (expo1 << 7) | (v1 & 0x7F)).to(tl.uint16)

            nt1_bit = 1 - is_t1
            nt1_local = tl.arange(0, BK) - t1_local - 1
            nt1_rank = nt1c + nt1_local

            # B2: per-row bitmap indexed by within-row non-T1 rank
            bpm = b2b + nt1_rank
            b2v = tl.load(b2r_ptr + (bpm >> 5)).to(tl.uint32)
            is_t2 = ((b2v >> (bpm & 31)) & 1).to(tl.int32)
            is_t2 = tl.where(nt1_bit == 1, is_t2, 0)
            b2_local = tl.cumsum(is_t2, axis=0) - 1
            t2_rank = tl.where(is_t2 == 1, t2c + b2_local, 0)

            bp2 = t2b + t2_rank * 7
            w2i = bp2 // 32
            s2 = bp2 % 32
            w2a = tl.load(t2_ptr + w2i).to(tl.uint32)
            c2 = (s2 + 7) > 32
            w2b = tl.where(c2, tl.load(t2_ptr + w2i + 1).to(tl.uint32),
                           tl.zeros((BK,), tl.uint32))
            v2 = tl.where(c2, ((w2a >> s2) | (w2b << (32 - s2))) & 0x7F,
                          (w2a >> s2) & 0x7F).to(tl.int32)
            expo2 = tl.minimum(tl.maximum(e8 - (2 + (v2 >> 6)), 0), 255)
            w2 = ((sign << 15) | (expo2 << 7) | ((v2 & 0x3F) << 1)).to(tl.uint16)

            is_t3 = nt1_bit - is_t2
            t3r = nt1_rank - (t2c + b2_local + 1)
            t3r = tl.where(is_t3 == 1, t3r, 0)
            bp3 = t3b + t3r * 7
            w3i = bp3 // 32
            s3 = bp3 % 32
            w3a = tl.load(t3_ptr + w3i).to(tl.uint32)
            c3 = (s3 + 7) > 32
            w3b = tl.where(c3, tl.load(t3_ptr + w3i + 1).to(tl.uint32),
                           tl.zeros((BK,), tl.uint32))
            v3 = tl.where(c3, ((w3a >> s3) | (w3b << (32 - s3))) & 0x7F,
                          (w3a >> s3) & 0x7F).to(tl.int32)
            expo3 = tl.minimum(tl.maximum(e8 - (4 + (v3 >> 5)), 0), 255)
            w3 = ((sign << 15) | (expo3 << 7) | ((v3 & 0x1F) << 2)).to(tl.uint16)

            w16 = tl.where(is_t1 == 1, w1, tl.where(is_t2 == 1, w2, w3))
            tl.store(out_ptr + offs, w16.to(tl.bfloat16, bitcast=True))

            t1c += tl.sum(is_t1, axis=0)
            t2c += tl.sum(is_t2, axis=0)
            nt1c += BK - tl.sum(is_t1, axis=0)


def _pick_v2_blk(in_f: int) -> int:
    for bk in (512, 256, 128, 64):
        if in_f % bk == 0:
            return bk
    return 0


def precompute_peakq_offsets(packed: dict, blk: int = TRITON_BLOCK, device=None):
    """Per-block exclusive prefix counts of T1 / T2(over non-T1) for the kernel.

    Returns (t1_blk, t2_blk) int32 tensors of length n_blocks.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    N = packed["N"]
    n_blk = (N + blk - 1) // blk
    pos = torch.arange(N, device=device, dtype=torch.long)

    b1_vals = (packed["bitmaps"][0].to(device)[pos // 32] >> (pos % 32)) & 1
    is_t1 = b1_vals.to(torch.int32)
    if is_t1.numel() < n_blk * blk:                     # pad tail block
        is_t1 = torch.cat(
            [is_t1, torch.zeros(n_blk * blk - N, dtype=torch.int32, device=device)]
        )
    t1_per_blk = is_t1.view(n_blk, blk).sum(dim=1).to(torch.int32)
    t1_blk = torch.zeros(n_blk, dtype=torch.int32, device=device)
    t1_blk[1:] = t1_per_blk[:-1].cumsum(0)

    n_non = packed["counts"][1] + packed["counts"][2]
    if n_non > 0:
        n_t1_global = packed["counts"][0]
        p2 = torch.arange(n_non, device=device, dtype=torch.long)
        b2_vals = (packed["bitmaps"][1].to(device)[p2 // 32] >> (p2 % 32)) & 1
        is_t2_non = b2_vals.to(torch.int32)
        non_elem = (~is_t1.bool()).nonzero(as_tuple=True)[0]
        non_blk = non_elem[:n_non] // blk
        t2_per_blk = torch.zeros(n_blk, dtype=torch.int32, device=device)
        t2_per_blk.scatter_add_(0, non_blk, is_t2_non)
        t2_blk = torch.zeros(n_blk, dtype=torch.int32, device=device)
        t2_blk[1:] = t2_per_blk[:-1].cumsum(0)
    else:
        t2_blk = torch.zeros(n_blk, dtype=torch.int32, device=device)
    return t1_blk, t2_blk


@torch.no_grad()
def decode_peakq_triton(packed: dict, device=None, dtype=torch.bfloat16) -> torch.Tensor:
    """Single-kernel fused decode. Falls back to scatter when unavailable."""
    supported = (
        _HAS_TRITON
        and torch.cuda.is_available()
        and tuple(packed["tiers"]) == PEAKQ_TIERS
        and packed["payload_bits"] == [8, 7, 7]
    )
    if not supported:
        return decode_peakq_scatter(packed, device, dtype)

    device = device or torch.device("cuda")

    if packed.get("layout", "global") == "rows":
        in_f = packed["in_f"]
        bk = _pick_v2_blk(in_f)
        if bk == 0:
            return decode_peakq_scatter(packed, device, dtype)
        out = torch.empty(packed["N"], dtype=torch.bfloat16, device=device)
        _peakq_decode_v2_kernel[(packed["out_f"],)](
            out,
            packed["sign_packed"].to(device),
            packed["emax"].to(device),
            packed["bitmaps"][0].to(device),
            packed["b2_rows"].to(device),
            packed["streams"][0].to(device),
            packed["streams"][1].to(device),
            packed["streams"][2].to(device),
            packed["t1_off"].to(device),
            packed["t2_bit_off"].to(device),
            packed["t3_bit_off"].to(device),
            packed["b2_bit_off"].to(device),
            IN_F=in_f,
            GROUP=packed["group"],
            BK=bk,
        )
        return out.view(packed["out_f"], packed["in_f"]).to(dtype)

    N = packed["N"]
    out = torch.empty(N, dtype=torch.bfloat16, device=device)
    blk = TRITON_BLOCK
    n_blk = (N + blk - 1) // blk
    t1_blk, t2_blk = precompute_peakq_offsets(packed, blk, device)

    _peakq_decode_kernel[(n_blk,)](
        out,
        packed["sign_packed"].to(device),
        packed["emax"].to(device),
        packed["bitmaps"][0].to(device),
        packed["bitmaps"][1].to(device),
        packed["streams"][0].to(device),
        packed["streams"][1].to(device),
        packed["streams"][2].to(device),
        t1_blk,
        t2_blk,
        N,
        GROUP=packed["group"],
        BLK=blk,
    )
    return out.view(packed["out_f"], packed["in_f"]).to(dtype)


# --------------------------------------------------------------------------- #
#  Fused decode+GEMV for single-token steps (generation hot loop)
# --------------------------------------------------------------------------- #
if _HAS_TRITON:

    @triton.jit
    def _peakq_gemv_kernel(
        x_ptr,                # [in_f] bf16 activations (single token)
        y_ptr,                # [out_f] bf16 output
        sign_ptr, emax_ptr,   # flat streams: 1-bit sign, uint8 emax/16
        b1_ptr, b2_ptr,       # nested bitmaps (int32 words)
        t1_ptr, t2_ptr, t3_ptr,
        q1_ptr, q2_ptr,       # [out_f] per-row global T1/T2 prefix counts
        IN_F: tl.constexpr,
        OUT_F: tl.constexpr,
        BK: tl.constexpr,
        GROUP: tl.constexpr,
    ):
        """y = x @ W.T with W decoded on the fly; bf16 W never materialized.

        Register-carried rank counters seeded from per-row prefixes
        (compute_row_prefixes in fused.py — same dict layout works).
        """
        n = tl.program_id(0)
        row0 = n * IN_F

        t1_cnt = tl.load(q1_ptr + n)
        t2_cnt = tl.load(q2_ptr + n)
        nt1_cnt = row0 - t1_cnt

        acc = 0.0
        for k0 in tl.range(0, IN_F, BK):
            offs = row0 + k0 + tl.arange(0, BK)
            kidx = k0 + tl.arange(0, BK)

            # flat streams: sign + emax
            sw = tl.load(sign_ptr + offs // 32).to(tl.uint32)
            sign = ((sw >> (offs % 32)) & 1).to(tl.int32)
            e8 = tl.load(emax_ptr + offs // GROUP).to(tl.int32)

            # --- B1 / T1 (raw uint8 stream, byte-indexed by rank) ---
            b1v = tl.load(b1_ptr + offs // 32).to(tl.uint32)
            is_t1 = ((b1v >> (offs % 32)) & 1).to(tl.int32)
            t1_local = tl.cumsum(is_t1, axis=0) - 1
            t1_rank = tl.where(is_t1 == 1, t1_cnt + t1_local, 0)
            v1 = tl.load(t1_ptr + t1_rank).to(tl.int32)
            expo1 = tl.minimum(tl.maximum(e8 - (v1 >> 7), 0), 255)
            w1 = ((sign << 15) | (expo1 << 7) | (v1 & 0x7F)).to(tl.uint16)

            # --- non-T1 rank (derived) + B2 / T2 (7-bit stream) ---
            nt1_bit = 1 - is_t1
            nt1_local = tl.arange(0, BK) - t1_local - 1
            nt1_rank = tl.where(nt1_bit == 1, nt1_cnt + nt1_local, 0)

            b2v = tl.load(b2_ptr + nt1_rank // 32).to(tl.uint32)
            is_t2 = ((b2v >> (nt1_rank % 32)) & 1).to(tl.int32)
            is_t2 = tl.where(nt1_bit == 1, is_t2, 0)
            b2_local = tl.cumsum(is_t2, axis=0) - 1
            t2_rank = tl.where(is_t2 == 1, t2_cnt + b2_local, 0)

            bp2 = t2_rank * 7
            w2i = bp2 // 32
            s2 = bp2 % 32
            w2a = tl.load(t2_ptr + w2i).to(tl.uint32)
            c2 = (s2 + 7) > 32
            w2b = tl.where(c2, tl.load(t2_ptr + w2i + 1).to(tl.uint32),
                           tl.zeros((BK,), tl.uint32))
            v2 = tl.where(c2, ((w2a >> s2) | (w2b << (32 - s2))) & 0x7F,
                          (w2a >> s2) & 0x7F).to(tl.int32)
            expo2 = tl.minimum(tl.maximum(e8 - (2 + (v2 >> 6)), 0), 255)
            w2 = ((sign << 15) | (expo2 << 7) | ((v2 & 0x3F) << 1)).to(tl.uint16)

            # --- T3 (rank derived, no third cumsum) ---
            is_t3 = nt1_bit - is_t2
            t3r = nt1_rank - (t2_cnt + b2_local + 1)
            t3r = tl.where(is_t3 == 1, t3r, 0)
            bp3 = t3r * 7
            w3i = bp3 // 32
            s3 = bp3 % 32
            w3a = tl.load(t3_ptr + w3i).to(tl.uint32)
            c3 = (s3 + 7) > 32
            w3b = tl.where(c3, tl.load(t3_ptr + w3i + 1).to(tl.uint32),
                           tl.zeros((BK,), tl.uint32))
            v3 = tl.where(c3, ((w3a >> s3) | (w3b << (32 - s3))) & 0x7F,
                          (w3a >> s3) & 0x7F).to(tl.int32)
            expo3 = tl.minimum(tl.maximum(e8 - (4 + (v3 >> 5)), 0), 255)
            w3 = ((sign << 15) | (expo3 << 7) | ((v3 & 0x1F) << 2)).to(tl.uint16)

            wbits = tl.where(is_t1 == 1, w1, tl.where(is_t2 == 1, w2, w3))
            w = wbits.to(tl.bfloat16, bitcast=True).to(tl.float32)
            x = tl.load(x_ptr + kidx).to(tl.float32)
            acc += tl.sum(x * w, axis=0)

            t1_cnt += tl.sum(is_t1, axis=0)
            t2_cnt += tl.sum(is_t2, axis=0)
            nt1_cnt += BK - tl.sum(is_t1, axis=0)

        tl.store(y_ptr + n, acc.to(tl.bfloat16))

    @triton.jit
    def _peakq_gemv_split_kernel(
        x_ptr, y_ptr,          # y fp32 [out_f], atomic accumulate
        sign_ptr, emax_ptr, b1_ptr, b2_ptr, t1_ptr, t2_ptr, t3_ptr,
        q1_ptr, q2_ptr,        # [out_f * NSPLIT] chunk-boundary prefixes
        IN_F: tl.constexpr,
        OUT_F: tl.constexpr,
        BK: tl.constexpr,
        GROUP: tl.constexpr,
        CHUNK: tl.constexpr,   # in_f // NSPLIT
    ):
        """Split-K variant for wide layers (down_proj): grid (out_f, NSPLIT)."""
        n = tl.program_id(0)
        c = tl.program_id(1)
        start = n * IN_F + c * CHUNK
        NSPLIT: tl.constexpr = IN_F // CHUNK
        t1_cnt = tl.load(q1_ptr + n * NSPLIT + c)
        t2_cnt = tl.load(q2_ptr + n * NSPLIT + c)
        nt1_cnt = start - t1_cnt

        acc = 0.0
        for k0 in tl.range(0, CHUNK, BK):
            offs = start + k0 + tl.arange(0, BK)
            kidx = c * CHUNK + k0 + tl.arange(0, BK)

            sw = tl.load(sign_ptr + offs // 32).to(tl.uint32)
            sign = ((sw >> (offs % 32)) & 1).to(tl.int32)
            e8 = tl.load(emax_ptr + offs // GROUP).to(tl.int32)

            b1v = tl.load(b1_ptr + offs // 32).to(tl.uint32)
            is_t1 = ((b1v >> (offs % 32)) & 1).to(tl.int32)
            t1_local = tl.cumsum(is_t1, axis=0) - 1
            t1_rank = tl.where(is_t1 == 1, t1_cnt + t1_local, 0)
            v1 = tl.load(t1_ptr + t1_rank).to(tl.int32)
            expo1 = tl.minimum(tl.maximum(e8 - (v1 >> 7), 0), 255)
            w1 = ((sign << 15) | (expo1 << 7) | (v1 & 0x7F)).to(tl.uint16)

            nt1_bit = 1 - is_t1
            nt1_local = tl.arange(0, BK) - t1_local - 1
            nt1_rank = tl.where(nt1_bit == 1, nt1_cnt + nt1_local, 0)

            b2v = tl.load(b2_ptr + nt1_rank // 32).to(tl.uint32)
            is_t2 = ((b2v >> (nt1_rank % 32)) & 1).to(tl.int32)
            is_t2 = tl.where(nt1_bit == 1, is_t2, 0)
            b2_local = tl.cumsum(is_t2, axis=0) - 1
            t2_rank = tl.where(is_t2 == 1, t2_cnt + b2_local, 0)

            bp2 = t2_rank * 7
            w2i = bp2 // 32
            s2 = bp2 % 32
            w2a = tl.load(t2_ptr + w2i).to(tl.uint32)
            c2 = (s2 + 7) > 32
            w2b = tl.where(c2, tl.load(t2_ptr + w2i + 1).to(tl.uint32),
                           tl.zeros((BK,), tl.uint32))
            v2 = tl.where(c2, ((w2a >> s2) | (w2b << (32 - s2))) & 0x7F,
                          (w2a >> s2) & 0x7F).to(tl.int32)
            expo2 = tl.minimum(tl.maximum(e8 - (2 + (v2 >> 6)), 0), 255)
            w2 = ((sign << 15) | (expo2 << 7) | ((v2 & 0x3F) << 1)).to(tl.uint16)

            is_t3 = nt1_bit - is_t2
            t3r = nt1_rank - (t2_cnt + b2_local + 1)
            t3r = tl.where(is_t3 == 1, t3r, 0)
            bp3 = t3r * 7
            w3i = bp3 // 32
            s3 = bp3 % 32
            w3a = tl.load(t3_ptr + w3i).to(tl.uint32)
            c3 = (s3 + 7) > 32
            w3b = tl.where(c3, tl.load(t3_ptr + w3i + 1).to(tl.uint32),
                           tl.zeros((BK,), tl.uint32))
            v3 = tl.where(c3, ((w3a >> s3) | (w3b << (32 - s3))) & 0x7F,
                          (w3a >> s3) & 0x7F).to(tl.int32)
            expo3 = tl.minimum(tl.maximum(e8 - (4 + (v3 >> 5)), 0), 255)
            w3 = ((sign << 15) | (expo3 << 7) | ((v3 & 0x1F) << 2)).to(tl.uint16)

            wbits = tl.where(is_t1 == 1, w1, tl.where(is_t2 == 1, w2, w3))
            w = wbits.to(tl.bfloat16, bitcast=True).to(tl.float32)
            x = tl.load(x_ptr + kidx).to(tl.float32)
            acc += tl.sum(x * w, axis=0)

            t1_cnt += tl.sum(is_t1, axis=0)
            t2_cnt += tl.sum(is_t2, axis=0)
            nt1_cnt += BK - tl.sum(is_t1, axis=0)

        tl.atomic_add(y_ptr + n, acc)


def peakq_fused_gemv(x, sign, emax, b1, b2, t1, t2, t3, q1, q2,
                     out_f: int, in_f: int, group: int = PEAKQ_GROUP,
                     chunk: int = 0, y32: torch.Tensor | None = None):
    """y = x @ W.T for a single token; W decoded on the fly (never stored).

    chunk > 0 selects the split-K path accumulating into y32 (fp32); caller
    converts to bf16 and adds bias. Returns [out_f] bf16 or the y32 buffer.
    """
    if chunk > 0:
        nsplit = in_f // chunk
        _peakq_gemv_split_kernel[(out_f, nsplit)](
            x.reshape(-1), y32,
            sign, emax, b1, b2, t1, t2, t3, q1, q2,
            IN_F=in_f, OUT_F=out_f, BK=512, GROUP=group, CHUNK=chunk,
            num_warps=2,
        )
        return y32
    y = torch.empty(out_f, dtype=torch.bfloat16, device=x.device)
    # in-context (deployed model, KV-cache generation) measured on 4090 /
    # WDDM-shared GPU: (2,256) 34.6 ms/tok, (4,512) 36.3, (2,512) 38.4,
    # (1,256) 39.6. Isolated min-timing favors (1,256) but does not
    # reproduce in-model; trust the in-context numbers.
    num_warps, bk = 2, 256
    assert in_f % bk == 0, "in_f must be a multiple of BK (row tiles stay in-bounds)"
    _peakq_gemv_kernel[(out_f,)](
        x.reshape(-1), y,
        sign, emax, b1, b2, t1, t2, t3, q1, q2,
        IN_F=in_f, OUT_F=out_f, BK=bk, GROUP=group,
        num_warps=num_warps,
    )
    return y


if _HAS_TRITON:

    @triton.jit
    def _peakq_gemv_v2_kernel(
        x_ptr, y_ptr,
        sign_ptr, emax_ptr,
        b1_ptr, b2r_ptr,
        t1_ptr, t2_ptr, t3_ptr,
        t1o_ptr, t2o_ptr, t3o_ptr, b2o_ptr,
        IN_F: tl.constexpr,
        OUT_F: tl.constexpr,
        GROUP: tl.constexpr,
        BK: tl.constexpr,
        R: tl.constexpr,       # rows per program (rows independent in v2)
    ):
        """v2 rows-layout fused GEMV, R rows per program.

        Row-restart streams make every row's rank state local: no prefix
        seeds, no cross-row coupling — R is only a work-aggregation knob.
        The x[k] tile is loaded once and shared by all R rows.
        """
        pid = tl.program_id(0)
        rows = pid * R + tl.arange(0, R)          # [R]
        base = rows.to(tl.int64) * IN_F           # [R]

        t1b = tl.load(t1o_ptr + rows)             # [R]
        t2b = tl.load(t2o_ptr + rows)
        t3b = tl.load(t3o_ptr + rows)
        b2b = tl.load(b2o_ptr + rows)

        t1c = tl.zeros((R,), tl.int32)
        t2c = tl.zeros((R,), tl.int32)
        nt1c = tl.zeros((R,), tl.int32)
        acc = tl.zeros((R,), tl.float32)

        for k0 in tl.range(0, IN_F, BK):
            kidx = k0 + tl.arange(0, BK)                     # [BK]
            offs = base[:, None] + kidx[None, :]             # [R, BK]

            sw = tl.load(sign_ptr + offs // 32).to(tl.uint32)
            sign = ((sw >> (offs % 32)) & 1).to(tl.int32)
            e8 = tl.load(emax_ptr + offs // GROUP).to(tl.int32)

            b1v = tl.load(b1_ptr + offs // 32).to(tl.uint32)
            is_t1 = ((b1v >> (offs % 32)) & 1).to(tl.int32)
            t1_local = tl.cumsum(is_t1, axis=1) - 1          # [R, BK]
            t1_rank = tl.where(is_t1 == 1, t1c[:, None] + t1_local, 0)
            v1 = tl.load(t1_ptr + t1b[:, None] + t1_rank).to(tl.int32)
            expo1 = tl.minimum(tl.maximum(e8 - (v1 >> 7), 0), 255)
            w1 = ((sign << 15) | (expo1 << 7) | (v1 & 0x7F)).to(tl.uint16)

            nt1_bit = 1 - is_t1
            nt1_local = tl.arange(0, BK)[None, :] - t1_local - 1
            nt1_rank = nt1c[:, None] + nt1_local

            bpm = b2b[:, None] + nt1_rank
            b2v = tl.load(b2r_ptr + (bpm >> 5)).to(tl.uint32)
            is_t2 = ((b2v >> (bpm & 31)) & 1).to(tl.int32)
            is_t2 = tl.where(nt1_bit == 1, is_t2, 0)
            b2_local = tl.cumsum(is_t2, axis=1) - 1
            t2_rank = tl.where(is_t2 == 1, t2c[:, None] + b2_local, 0)

            bp2 = t2b[:, None] + t2_rank * 7
            w2i = bp2 // 32
            s2 = bp2 % 32
            w2a = tl.load(t2_ptr + w2i).to(tl.uint32)
            c2 = (s2 + 7) > 32
            w2b = tl.where(c2, tl.load(t2_ptr + w2i + 1).to(tl.uint32),
                           tl.zeros((R, BK), tl.uint32))
            v2 = tl.where(c2, ((w2a >> s2) | (w2b << (32 - s2))) & 0x7F,
                          (w2a >> s2) & 0x7F).to(tl.int32)
            expo2 = tl.minimum(tl.maximum(e8 - (2 + (v2 >> 6)), 0), 255)
            w2 = ((sign << 15) | (expo2 << 7) | ((v2 & 0x3F) << 1)).to(tl.uint16)

            is_t3 = nt1_bit - is_t2
            t3r = nt1_rank - (t2c[:, None] + b2_local + 1)
            t3r = tl.where(is_t3 == 1, t3r, 0)
            bp3 = t3b[:, None] + t3r * 7
            w3i = bp3 // 32
            s3 = bp3 % 32
            w3a = tl.load(t3_ptr + w3i).to(tl.uint32)
            c3 = (s3 + 7) > 32
            w3b = tl.where(c3, tl.load(t3_ptr + w3i + 1).to(tl.uint32),
                           tl.zeros((R, BK), tl.uint32))
            v3 = tl.where(c3, ((w3a >> s3) | (w3b << (32 - s3))) & 0x7F,
                          (w3a >> s3) & 0x7F).to(tl.int32)
            expo3 = tl.minimum(tl.maximum(e8 - (4 + (v3 >> 5)), 0), 255)
            w3 = ((sign << 15) | (expo3 << 7) | ((v3 & 0x1F) << 2)).to(tl.uint16)

            wbits = tl.where(is_t1 == 1, w1, tl.where(is_t2 == 1, w2, w3))
            w = wbits.to(tl.bfloat16, bitcast=True).to(tl.float32)
            x = tl.load(x_ptr + kidx).to(tl.float32)          # shared by R rows
            acc += tl.sum(w * x[None, :], axis=1)

            t1c += tl.sum(is_t1, axis=1)
            t2c += tl.sum(is_t2, axis=1)
            nt1c += BK - tl.sum(is_t1, axis=1)

        tl.store(y_ptr + rows, acc.to(tl.bfloat16))


# v2 GEMV defaults; measured in-context on MiniCPM5/4090 (WDDM):
#   R=4 BK=256 w=2 -> 35.2 ms/tok; R=2 BK=512 w=2 -> 35.2; R=1 BK=256 -> 37.1;
#   v1 global (2,256) reference 35.5. R>1 needs out_f % R == 0 (auto-fallback).
PEAKQ_V2_R = 4
PEAKQ_V2_BK = 256
PEAKQ_V2_WARPS = 2


def peakq_fused_gemv_v2(x, sign, emax, b1, b2r, t1, t2, t3,
                        t1o, t2o, t3o, b2o,
                        out_f: int, in_f: int, group: int = PEAKQ_GROUP,
                        r: int = PEAKQ_V2_R, bk: int = PEAKQ_V2_BK,
                        num_warps: int = PEAKQ_V2_WARPS):
    """y = x @ W.T, single token, v2 rows layout, R rows per program."""
    y = torch.empty(out_f, dtype=torch.bfloat16, device=x.device)
    while r > 1 and out_f % r != 0:                 # odd out_f fallback
        r //= 2
    assert in_f % bk == 0
    _peakq_gemv_v2_kernel[(out_f // r,)](
        x.reshape(-1), y,
        sign, emax, b1, b2r, t1, t2, t3,
        t1o, t2o, t3o, b2o,
        IN_F=in_f, OUT_F=out_f, GROUP=group, BK=bk, R=r,
        num_warps=num_warps,
    )
    return y


# --------------------------------------------------------------------------- #
#  Host-memory hygiene (learned from tpab_linear.py)
# --------------------------------------------------------------------------- #
# shared decode workspace singleton: one buffer sized to the largest layer
# serves ALL PeakQLinear instances when no engine-injected buffer is set
# (streaming layers decode sequentially anyway).
_SHARED_W_BUF = None
_SHARED_W_SIZE = 0


def _get_shared_w_buf(n_elems: int, device) -> torch.Tensor:
    global _SHARED_W_BUF, _SHARED_W_SIZE
    if _SHARED_W_BUF is None or _SHARED_W_SIZE < n_elems:
        _SHARED_W_BUF = torch.empty(n_elems, dtype=torch.bfloat16, device=device)
        _SHARED_W_SIZE = n_elems
    return _SHARED_W_BUF


_HEAVY_KEYS = ("bitmaps", "streams", "sign_packed", "emax", "b2_rows")


def _strip_packed_bodies(packed: dict) -> dict:
    """Drop the CPU tensor bodies from a packed dict, keep metadata.

    Once GPU copies are registered (or the bf16 decode is done), keeping the
    CPU originals costs ~1.3x the packed size per layer in host RAM — on a
    27B model that OOMs the box (tpab hit the same wall).
    """
    return {k: v for k, v in packed.items() if k not in _HEAVY_KEYS}


# --------------------------------------------------------------------------- #
#  Drop-in layer + deploy
# --------------------------------------------------------------------------- #
class PeakQLinear(nn.Module):
    """nn.Linear backed by a PEAK-Q packed weight.

    cache='full'  : decode once at deploy, plain F.linear afterwards.
    cache='none'  : per-forward Triton decode. Packed streams are moved to GPU
        once (registered buffers, no per-forward DMA); the decode writes into
        the shared buffer installed via ``_set_shared_buf`` (reused across
        layers by the streaming engine) or its own allocation otherwise.
        Single kernel, no fixups, no CPU sync -> CUDA-Graph capturable.
    """

    def __init__(self, packed: dict, bias=None, cache: str = "full",
                 use_triton: bool = True):
        super().__init__()
        self.out_features = packed["out_f"]
        self.in_features = packed["in_f"]
        self.N = packed["N"]
        self.packed = packed
        self._cache = cache
        self._use_triton = use_triton and _HAS_TRITON
        if bias is not None:
            self.register_buffer("_bias_buf", bias.detach().clone())
        else:
            self._bias_buf = None
        self._w = None
        self._w_buf = None
        self._layout = packed.get("layout", "global")

        if cache == "full":
            self._w = self._decode(torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"))
            # bf16 weight decoded & resident — the CPU packed bodies are dead
            self.packed = _strip_packed_bodies(packed)
        else:
            # streaming: packed data GPU-resident once (no per-forward DMA)
            cuda = torch.cuda.is_available()
            self._gpu = (
                self._use_triton and cuda
                and tuple(packed["tiers"]) == PEAKQ_TIERS
                and packed["payload_bits"] == [8, 7, 7]
            )
            if self._gpu and self._layout == "rows":
                for key, val in [
                    ("_sign", packed["sign_packed"]),
                    ("_emax", packed["emax"]),
                    ("_b1", packed["bitmaps"][0]),
                    ("_b2r", packed["b2_rows"]),
                    ("_t1", packed["streams"][0]),
                    ("_t2", packed["streams"][1]),
                    ("_t3", packed["streams"][2]),
                    ("_t1o", packed["t1_off"]),
                    ("_t2o", packed["t2_bit_off"]),
                    ("_t3o", packed["t3_bit_off"]),
                    ("_b2o", packed["b2_bit_off"]),
                ]:
                    self.register_buffer(key, val.cuda())
                # fused v2 GEMV: no prefix tables needed at all (the whole
                # point of the rows layout)
                self._use_fused = self.in_features % PEAKQ_V2_BK == 0
            elif self._gpu:
                for key, val in [
                    ("_sign", packed["sign_packed"]),
                    ("_emax", packed["emax"]),
                    ("_b1", packed["bitmaps"][0]),
                    ("_b2", packed["bitmaps"][1]),
                    ("_t1", packed["streams"][0]),
                    ("_t2", packed["streams"][1]),
                    ("_t3", packed["streams"][2]),
                ]:
                    self.register_buffer(key, val.cuda())
                t1_blk, t2_blk = precompute_peakq_offsets(packed)
                self.register_buffer("_t1_blk", t1_blk)
                self.register_buffer("_t2_blk", t2_blk)

                # fused decode+GEMV for single-token steps (generation hot
                # loop): reads only packed streams, bf16 W never materialized.
                # Row prefixes reuse fused.compute_row_prefixes (same dict
                # layout: bitmaps + counts of 3 levels). Single-kernel path
                # (measured faster than split-K on 4090 for these shapes).
                self._use_fused = False
                if self.in_features % 512 == 0:
                    from .fused import compute_row_prefixes

                    q1, q2 = compute_row_prefixes(packed)
                    self.register_buffer("_q1", q1.cuda())
                    self.register_buffer("_q2", q2.cuda())
                    self._use_fused = True
            if self._gpu:
                # GPU copies registered — free the CPU bodies (host RAM)
                self.packed = _strip_packed_bodies(packed)

    def _stream_decode(self):
        """Single Triton kernel decode into the shared buffer (GPU-resident)."""
        if self._layout == "rows":
            bk = _pick_v2_blk(self.in_features)
            w_flat = self._w_buf[: self.N]
            _peakq_decode_v2_kernel[(self.out_features,)](
                w_flat, self._sign, self._emax, self._b1, self._b2r,
                self._t1, self._t2, self._t3,
                self._t1o, self._t2o, self._t3o, self._b2o,
                IN_F=self.in_features,
                GROUP=self.packed["group"], BK=bk,
            )
            return w_flat.view(self.out_features, self.in_features)
        import triton

        w_flat = self._w_buf[: self.N]
        blk = TRITON_BLOCK
        n_blk = (self.N + blk - 1) // blk
        _peakq_decode_kernel[(n_blk,)](
            w_flat, self._sign, self._emax, self._b1, self._b2,
            self._t1, self._t2, self._t3, self._t1_blk, self._t2_blk,
            self.N, GROUP=self.packed["group"], BLK=blk,
        )
        return w_flat.view(self.out_features, self.in_features)

    def _decode(self, device):
        if self._use_triton:
            return decode_peakq_triton(self.packed, device)
        return decode_peakq_scatter(self.packed, device)

    def _set_shared_buf(self, w_buf):
        self._w_buf = w_buf

    def forward(self, x):
        if self._cache == "full":
            w = self._w
        elif getattr(self, "_use_fused", False) and x.numel() == self.in_features:
            # single-token decode step: fused decode+GEMV, no W materialization
            if self._layout == "rows":
                y = peakq_fused_gemv_v2(
                    x, self._sign, self._emax, self._b1, self._b2r,
                    self._t1, self._t2, self._t3,
                    self._t1o, self._t2o, self._t3o, self._b2o,
                    self.out_features, self.in_features,
                    group=self.packed["group"],
                ).to(x.dtype)
            else:
                y = peakq_fused_gemv(
                    x, self._sign, self._emax, self._b1, self._b2,
                    self._t1, self._t2, self._t3, self._q1, self._q2,
                    self.out_features, self.in_features,
                    group=self.packed["group"],
                ).to(x.dtype)
            if self._bias_buf is not None:
                y = y + self._bias_buf.to(x.dtype)
            return y.view(*x.shape[:-1], self.out_features)
        elif self._gpu:
            if self._w_buf is None:
                self._w_buf = _get_shared_w_buf(self.N, x.device)
            w = self._stream_decode()
        else:
            w = self._decode(x.device)
        bias = self._bias_buf.to(x.dtype) if self._bias_buf is not None else None
        return F.linear(x, w, bias)

    def extra_repr(self) -> str:
        return (
            f"in={self.in_features}, out={self.out_features}, "
            f"tiers={self.packed['tiers']}, bpw={self.packed['bits_per_weight']:.2f}"
        )


@torch.no_grad()
def deploy_peakq(
    model: nn.Module,
    tiers: tuple = PEAKQ_TIERS,
    group: int = PEAKQ_GROUP,
    cache: str = "full",
    layout: str = "global",
    verbose: bool = True,
) -> dict:
    """Replace every quantizable Linear in *model* with PeakQLinear (in-place)."""
    targets = list(iter_quantizable_linears(model))
    total_bytes = 0
    n_elems = 0
    for name, mod in targets:
        packed = peakq_quantize(mod.weight.data, group=group, tiers=tiers,
                                layout=layout)
        bias = mod.bias.data if mod.bias is not None else None
        new_layer = PeakQLinear(packed, bias=bias, cache=cache)
        _set_parent_child(model, name, new_layer)
        total_bytes += packed["total_bytes"]
        n_elems += packed["N"]
    stats = {
        "n_layers": len(targets),
        "total_bytes": total_bytes,
        "n_elems": n_elems,
        "bits_per_weight": (total_bytes * 8) / max(n_elems, 1),
        "compression_vs_bf16": (n_elems * 2) / max(total_bytes, 1),
    }
    if verbose:
        print(
            f"[peakq-deploy] {stats['n_layers']} layers | "
            f"{total_bytes / 1e6:.0f}MB packed ({stats['compression_vs_bf16']:.2f}x vs bf16) | "
            f"bpw={stats['bits_per_weight']:.2f} | tiers={tiers}",
            flush=True,
        )
    return stats


@torch.no_grad()
def deploy_peakq_lazy(
    model: nn.Module,
    tiers: tuple = PEAKQ_TIERS,
    group: int = PEAKQ_GROUP,
    layout: str = "rows",
    verbose: bool = True,
) -> dict:
    """Big-model deploy: encode layer-by-layer, never holding two layers.

    Mirrors tpab's lazy path: works with low_cpu_mem_usage models (weights
    stay meta/disk until touched) — each layer is materialized to CPU, packed,
    staged to GPU, then BOTH its bf16 tensor and CPU packed bodies are
    dropped. Deploy peak host RAM ~= one layer's packed size; deploy peak GPU
    ~= final resident + one layer's decode.

    Always streaming (cache='none'): packed GPU-resident, shared decode buf.
    """
    targets = list(iter_quantizable_linears(model))
    total_bytes = 0
    n_elems = 0
    for name, mod in targets:
        w = mod.weight.data
        if w.is_cuda or w.dtype != torch.bfloat16:
            w = w.to("cpu", torch.bfloat16)     # materialize just this layer
        packed = peakq_quantize(w, group=group, tiers=tiers, layout=layout)
        bias = mod.bias.data if mod.bias is not None else None
        ql = PeakQLinear(packed, bias=bias, cache="none")
        total_bytes += packed["total_bytes"]
        n_elems += packed["N"]
        mod.weight.data = torch.empty(0)        # release the bf16 reference
        _set_parent_child(model, name, ql)
        del w, packed
    stats = {
        "n_layers": len(targets),
        "total_bytes": total_bytes,
        "n_elems": n_elems,
        "bits_per_weight": (total_bytes * 8) / max(n_elems, 1),
        "compression_vs_bf16": (n_elems * 2) / max(total_bytes, 1),
    }
    if verbose:
        print(
            f"[peakq-lazy] {stats['n_layers']} layers | "
            f"{total_bytes / 1e6:.0f}MB packed ({stats['compression_vs_bf16']:.2f}x) | "
            f"bpw={stats['bits_per_weight']:.2f} | tiers={tiers}",
            flush=True,
        )
    return stats


# --------------------------------------------------------------------------- #
#  Metrics
# --------------------------------------------------------------------------- #
def peakq_snr(w: torch.Tensor, w_dec: torch.Tensor) -> float:
    """SNR (dB) of a decoded weight vs its bf16 original."""
    a = w.reshape(-1).float()
    b = w_dec.reshape(-1).float()
    mse = ((a - b) ** 2).mean().item()
    var = a.var(unbiased=False).item()
    return 10 * float(torch.log10(torch.tensor(var / max(mse, 1e-30))))


def peakq_exact_pct(w: torch.Tensor, w_dec: torch.Tensor) -> float:
    """Percentage of bit-identical elements."""
    a = w.reshape(-1).to(torch.bfloat16).view(torch.int16)
    b = w_dec.reshape(-1).to(torch.bfloat16).view(torch.int16)
    return (a == b).float().mean().item() * 100
