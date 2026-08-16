"""VT-Queue: value-tiered buckets + fixed-width queues per strip.

Design synthesis of everything measured in IXRUN:
  - ixgs  : fine-grained scales are the #1 accuracy lever
  - (3,5,8): value-magnitude tiering is the main compression source
  - TPAB  : fixed-width layouts (no ranks) give parallel decode + random
            access; per-tile headers must be AMORTIZED (fewer, bigger)

Structure: weight [O, I] -> strips of 16 x 256 (I % 256 == 0).
Each strip:
    scale    : fp16, |v|-mean-free symmetric grid
    map      : 2 bits/element bucket id (fixed position L*2 -> random access)
    buckets  : 3 streams (b0, b1, b2 in {1, 2, 4} bits), fixed width
               within each bucket; element codes at group base + off[b]*b
Tier assignment: per-strip magnitude quantiles (top 6.25%, next 12.5%,
rest) -> tier 0 stores sign only (1 bit), tier 1 a 2-bit signed code
(zero-centred), tier 2 a 4-bit signed code. scale chosen so tier-2 covers
the strip max after the top outliers escape to a global sparse table
(1 per 2 strips).

Target budget: 2 (map) + 6.25%*1 + 12.5%*2 + 81.25%*4 = 5.69 bpw before
outlier savings; realistic 4.6-5.0 with strip-scale accuracy gains.
"""
from __future__ import annotations
import torch

from .bitpack import pack_bits_stream, unpack_bits_stream

__all__ = ["encode_vtq", "decode_vtq_ref", "STRIP_R", "STRIP_C"]

STRIP_R = 16
STRIP_C = 256
BUCKET_BITS = (1, 2, 4)      # tier widths


@torch.no_grad()
def encode_vtq(w: torch.Tensor, snr_target_db: float = 26.0) -> dict:
    O, I = w.shape
    if O % STRIP_R or I % STRIP_C:
        raise ValueError(f"need O%{STRIP_R}==0 and I%{STRIP_C}==0, got {O}x{I}")
    S_r, S_c = O // STRIP_R, I // STRIP_C
    S = S_r * S_c
    n_per = STRIP_R * STRIP_C
    dev = w.device

    v = (w.float().view(S_r, STRIP_R, S_c, STRIP_C).permute(0, 2, 1, 3)
          .reshape(S, n_per))

    # ---- per-strip: one outlier escape + tier thresholds by quantile ----
    abs_sorted = v.abs().sort(dim=1, descending=True).values
    n_esc = max(1, n_per // 2048)              # ~2 per strip
    esc_thr = abs_sorted[:, n_esc - 1]          # n_esc-th largest
    is_esc = v.abs() > esc_thr.unsqueeze(1)
    v2 = v * (~is_esc)

    power = v.pow(2).sum(dim=1).clamp(min=1e-30)     # FULL power (outliers exact)
    M = v2.abs().amax(dim=1).clamp(min=1e-12)

    # tier thresholds among remaining values: top 6.25% -> t0, next -> t1
    n0 = n_per // 16          # 6.25% of elements
    n1 = n_per // 8           # 12.5%
    t0 = abs_sorted[:, n_esc + n0 - 1]   # boundary magnitudes
    t1 = abs_sorted[:, n_esc + n0 + n1 - 1]

    mag = v2.abs()
    tier = torch.zeros_like(v2, dtype=torch.uint8)
    tier[mag > t1.unsqueeze(1)] = 1
    tier[mag > t0.unsqueeze(1)] = 2
    # escaped positions marked tier 0 code 0 (overlaid later)
    tier[is_esc] = 0

    # ---- scale: tier-2 grid covers M with 4-bit signed (qmax 7) ----
    qmax2 = 7
    s = (M / qmax2).half().float()
    # per-tier quantization: value -> signed code on the SAME scale
    qf = v2 / s.unsqueeze(1)
    codes = [None, None, None]
    masks = [tier == i for i in range(3)]
    # tier0: sign only (value = ±s)
    c0 = torch.where(v2 > 0, 1, 0).to(torch.int32)
    # tier1: 2-bit signed: -1..1 => 0..2? use -1,0,+1 codes 0,1,2
    c1 = qf.round().clamp(-1, 1).to(torch.int32) + 1
    # tier2: 4-bit signed: clamp to -7..7
    c2 = qf.round().clamp(-qmax2, qmax2).to(torch.int32) + qmax2

    codes = torch.zeros_like(v2, dtype=torch.int32)
    codes[masks[0]] = c0[masks[0]]
    codes[masks[1]] = c1[masks[1]]
    codes[masks[2]] = c2[masks[2]]

    # ---- pack map (2b/elem) + per-tier streams (fixed width, group-packed)
    map_bits = pack_bits_stream(
        tier.reshape(-1).to(torch.int32).cpu(), 2)

    offs = []                 # element offset of each strip inside its tier
    cnts = [int(m.sum()) for m in
            (tier.reshape(S, n_per) == 0,
             tier.reshape(S, n_per) == 1,
             tier.reshape(S, n_per) == 2)]
    streams = []
    cursor_bit = 0
    gbase = [0, 0, 0]
    tier_flat = tier.reshape(S, n_per)
    for ti, b in enumerate(BUCKET_BITS):
        sel = tier_flat == ti
        n_t = int(sel.sum())
        idx = sel.nonzero(as_tuple=True)[0]
        if n_t:
            vals = codes.reshape(S, n_per)[sel].cpu()
            body = pack_bits_stream(vals, b)
        else:
            body = torch.zeros(0, dtype=torch.int32)
        streams.append(body)
        gbase[ti] = cursor_bit
        cursor_bit += n_t * b

    # per-strip per-tier element offsets: precompute for decode as
    # off[strip, tier] (int32) — the element's queue position is
    # (cumulative tier count before it within strip) + off[strip, tier]
    # global base handled by gbase. For the kernel we need per-element
    # queue ranks... which reintroduces prefix state. INSTEAD: store
    # per-element ranks explicitly? too big.
    # => decode uses scatter (reference path only for the prototype);
    #    the fused kernel is the next milestone once quality is proven.
    esc_t, esc_l = is_esc.reshape(S, n_per).nonzero(as_tuple=True)
    esc_val = v.reshape(S, n_per)[esc_t, esc_l].to(torch.float16).cpu()

    total_bytes = (
        sum(bd.numel() * 4 for bd in streams) + map_bits.numel() * 4
        + S * 2                                     # scales
        + len(esc_t) * 6
        + S * 8                                     # (residual metadata)
    )
    return {
        "shape": (O, I), "S": S, "n_per": n_per,
        "scales": s.half().cpu(), "tier": tier_flat.cpu(),
        "codes": codes.reshape(S, n_per).cpu(),
        "streams": streams, "map_bits": map_bits, "gbase": gbase,
        "cnts": cnts,
        "esc_t": esc_t.to(torch.int32).cpu(), "esc_l": esc_l.to(torch.int32).cpu(),
        "esc_val": esc_val,
        "total_bytes": total_bytes, "bpw": total_bytes * 8 / (O * I),
        "n_esc": int(len(esc_t)),
    }


@torch.no_grad()
def decode_vtq_ref(packed: dict, device=None) -> torch.Tensor:
    """Scatter reference decode (correctness first; fused kernel later)."""
    O, I = packed["shape"]
    S, n_per = packed["S"], packed["n_per"]
    dev = device or "cpu"
    tier = packed["tier"].to(dev)
    codes = packed["codes"].to(dev)
    s = packed["scales"].to(dev).float().unsqueeze(1)

    qf = torch.zeros_like(codes, dtype=torch.float32)
    m0 = tier == 0
    m1 = tier == 1
    m2 = tier == 2
    qf[m0] = torch.where(codes[m0] > 0, 1.0, -1.0)
    qf[m1] = (codes[m1] - 1).float()
    qf[m2] = (codes[m2] - 7).float()
    w = (qf * s).reshape(S, n_per)
    w[packed["esc_t"].to(dev), packed["esc_l"].to(dev)] = (
        packed["esc_val"].to(dev).float())
    S_r, S_c = O // STRIP_R, I // STRIP_C
    out = (w.view(S_r, S_c, STRIP_R, STRIP_C).permute(0, 2, 1, 3)
            .reshape(O, I))
    return out.to(torch.bfloat16)
