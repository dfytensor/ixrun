"""Triton decode kernels for INT8-X nested-bitmap packed weights.

Two decode paths are provided:

1. ``decode_weight_triton`` — fused single-kernel decode that reads the nested
   bitmaps + per-level bit streams with in-block ``tl.cumsum`` prefix scans and
   writes the reconstructed bf16 weight directly. Fast, no scatter needed.
   Requires precomputed per-block level-count prefix sums.

2. ``decode_weight_scatter`` — pure-PyTorch scatter fallback that unpacks each
   level stream and scatters to reconstructed positions. Always works (no Triton
   autotune surprises) and is used when CUDA/Triton is unavailable.
"""
from __future__ import annotations
import torch

from .bitpack import unpack_bits_stream
from .config import BIT_TO_THRESHOLD, TRITON_BLOCK

__all__ = ["decode_weight_triton", "decode_weight_scatter", "precompute_block_offsets"]

_HAS_TRITON = False
try:
    import triton
    import triton.language as tl

    _HAS_TRITON = True
except Exception:
    pass


# --------------------------------------------------------------------------- #
#  Pure-PyTorch scatter fallback
# --------------------------------------------------------------------------- #
@torch.no_grad()
def decode_weight_scatter(packed: dict, device=None, dtype=torch.bfloat16) -> torch.Tensor:
    """Robust decode via per-level unpack + scatter (no Triton)."""
    from .quantize import compute_positions

    level_bits = packed["level_bits"]
    of, inf, N = packed["out_f"], packed["in_f"], packed["N"]
    if device is None:
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    scale = packed["scale"].to(device)
    streams = [s.to(device) for s in packed["streams"]]

    positions = [p.to(device) for p in compute_positions(packed)]
    i8_flat = torch.zeros(N, dtype=torch.int32, device=device)
    for i, (b, cnt) in enumerate(zip(level_bits, packed["counts"])):
        if cnt == 0:
            continue
        offset = BIT_TO_THRESHOLD[b]
        pos = positions[i]
        if i == len(level_bits) - 1 and b == 8:
            vals = streams[i][:cnt].to(torch.int32) - offset
        else:
            vals = unpack_bits_stream(streams[i], cnt, b, device=device) - offset
        i8_flat[pos] = vals
    return (i8_flat.float() * scale.float()).to(dtype).view(of, inf)


# --------------------------------------------------------------------------- #
#  Triton fused kernel
# --------------------------------------------------------------------------- #
if _HAS_TRITON:

    @triton.jit
    def _ix_decode_kernel(
        out_ptr,
        b1_ptr,
        b2_ptr,
        l1_ptr,
        l2_ptr,
        l3_ptr,
        b1_blk_ptr,
        b2_blk_ptr,
        scale_ptr,
        N: tl.constexpr,
        BLK: tl.constexpr,
    ):
        """Fused (3,5,8) decode. One program per BLK-element chunk.

        Uses nested bitmap B1 (is level-1?) and B2 (among non-L1, is level-2?)
        with in-block ``tl.cumsum`` to compute per-level stream ranks, then
        extracts 3/5-bit values or loads the raw 8-bit value.
        """
        pid = tl.program_id(0)
        offs = pid * BLK + tl.arange(0, BLK)
        mask = offs < N

        # --- B1 bitmap: is element a level-1 (|v|<=3) value? ---
        b1_word = offs // 32
        b1_bit = offs % 32
        b1_val = tl.load(b1_ptr + b1_word, mask=mask, other=0).to(tl.uint32)
        is_l1 = ((b1_val >> b1_bit) & 1).to(tl.int32)

        l1_before = tl.load(b1_blk_ptr + pid)
        l1_local = tl.cumsum(is_l1, axis=0) - 1
        l1_rank = l1_before + l1_local
        l1_rank = tl.where(is_l1 == 1, l1_rank, 0)

        # L1: extract 3-bit from l1 stream
        bp1 = l1_rank * 3
        w1i = bp1 // 32
        s1 = bp1 % 32
        w1a = tl.load(l1_ptr + w1i, mask=mask, other=0).to(tl.uint32)
        c1 = (s1 + 3) > 32
        w1b = tl.where(
            c1,
            tl.load(l1_ptr + w1i + 1, mask=mask, other=0).to(tl.uint32),
            tl.zeros((BLK,), tl.uint32),
        )
        l1v = (
            tl.where(c1, ((w1a >> s1) | (w1b << (32 - s1))) & 0x7, (w1a >> s1) & 0x7)
            .to(tl.int32)
            - 3
        )

        # --- non-L1 rank: derived from arange & l1_local (no second cumsum) ---
        # cumsum(nl1)[i] = (i+1) - cumsum(is_l1)[i]  =>  local rank = i - l1_local - 1
        nl1_bit = 1 - is_l1
        nl1_before = pid * BLK - l1_before
        nl1_local = tl.arange(0, BLK) - l1_local - 1
        nl1_rank = nl1_before + nl1_local
        nl1_rank = tl.where(nl1_bit == 1, nl1_rank, 0)

        # B2 bitmap: among non-L1 elements, is it level-2?
        b2w = nl1_rank // 32
        b2b = nl1_rank % 32
        b2v = tl.load(b2_ptr + b2w, mask=mask, other=0).to(tl.uint32)
        is_l2 = ((b2v >> b2b) & 1).to(tl.int32)
        is_l2 = tl.where(nl1_bit == 1, is_l2, 0)

        b2_before = tl.load(b2_blk_ptr + pid)
        b2_local = tl.cumsum(is_l2, axis=0) - 1
        b2_rank = b2_before + b2_local
        b2_rank = tl.where(is_l2 == 1, b2_rank, 0)

        # L2: extract 5-bit from l2 stream
        bp2 = b2_rank * 5
        w2i = bp2 // 32
        s2 = bp2 % 32
        w2a = tl.load(l2_ptr + w2i, mask=mask, other=0).to(tl.uint32)
        c2 = (s2 + 5) > 32
        w2b = tl.where(
            c2,
            tl.load(l2_ptr + w2i + 1, mask=mask, other=0).to(tl.uint32),
            tl.zeros((BLK,), tl.uint32),
        )
        l2v = (
            tl.where(c2, ((w2a >> s2) | (w2b << (32 - s2))) & 0x1F, (w2a >> s2) & 0x1F)
            .to(tl.int32)
            - 15
        )

        # L3: rank among (non-L1 AND non-L2), derived (no third cumsum):
        # for an l3 element, count-before = nl1_count_before - l2_count_before;
        # l2_count_before = b2_local + 1 at elements where is_l2==0
        # (cumsum counts up to AND including i, subtract is_l2[i]=0 -> +1 back)
        is_l3 = tl.where(nl1_bit == 1, 1 - is_l2, 0)
        nl1_total_before = pid * BLK - l1_before
        l3_total_before = nl1_total_before - b2_before
        l3r = l3_total_before + (nl1_local - b2_local - 1)
        l3r = tl.where(is_l3 == 1, l3r, 0)
        l3r_safe = tl.where(l3r < 0, 0, l3r)
        l3v = tl.load(l3_ptr + l3r_safe, mask=mask, other=0).to(tl.int32) - 127

        # select
        val = tl.where(
            is_l1 == 1,
            l1v,
            tl.where(is_l2 == 1, l2v, l3v),
        )

        sc = tl.load(scale_ptr).to(tl.float32)
        w = (val.to(tl.float32) * sc).to(tl.bfloat16)
        tl.store(out_ptr + offs, w, mask=mask)


def precompute_block_offsets(packed: dict, blk: int = TRITON_BLOCK, device=None) -> tuple:
    """Precompute per-block prefix sums of level-1 / level-2 counts.

    Returns (b1_blk, b2_blk) int32 tensors of length n_blocks, where
    b1_blk[k] = #level-1 elements in blocks [0, k);  similar for b2_blk.
    """
    if device is None:
        device = torch.device("cuda")
    N = packed["N"]
    n_blk = (N + blk - 1) // blk
    bitmaps = packed["bitmaps"]
    # rebuild per-element is_l1 from B1
    pos = torch.arange(N, device=device, dtype=torch.long)
    b1_vals = (bitmaps[0].to(device)[pos // 32] >> (pos % 32)) & 1
    is_l1 = b1_vals.to(torch.int32)
    l1_per_blk = is_l1.view(n_blk, blk).sum(dim=1).to(torch.int32)
    b1_blk = torch.zeros(n_blk, dtype=torch.int32, device=device)
    b1_blk[1:] = l1_per_blk[:-1].cumsum(0)

    # level-2 prefix among non-L1: rebuild B2 layout
    nl1_per_blk = blk - l1_per_blk
    nl1_offset = torch.zeros(n_blk, dtype=torch.int32, device=device)
    nl1_offset[1:] = nl1_per_blk[:-1].cumsum(0)
    # is_l2 per element among non-L1
    n_non = packed["counts"][1] + packed["counts"][2]
    if n_non > 0 and len(bitmaps) > 1:
        p2 = torch.arange(n_non, device=device, dtype=torch.long)
        b2_vals = (bitmaps[1].to(device)[p2 // 32] >> (p2 % 32)) & 1
        is_l2_nonl1 = b2_vals.to(torch.int32)
        l2_per_blk = torch.zeros(n_blk, dtype=torch.int32, device=device)
        # map non-L1 elements back to their block id
        nonl1_elem = (~is_l1.bool()).nonzero(as_tuple=True)[0]
        nonl1_blk = nonl1_elem[:n_non] // blk
        # accumulate l2 counts per block
        ones = torch.ones(n_non, dtype=torch.int32, device=device)
        l2_per_blk.scatter_add_(0, nonl1_blk.to(torch.int64), is_l2_nonl1.to(torch.int32))
        b2_blk = torch.zeros(n_blk, dtype=torch.int32, device=device)
        b2_blk[1:] = l2_per_blk[:-1].cumsum(0)
    else:
        b2_blk = torch.zeros(n_blk, dtype=torch.int32, device=device)
    return b1_blk, b2_blk


@torch.no_grad()
def decode_weight_triton(packed: dict, device=None, dtype=torch.bfloat16) -> torch.Tensor:
    """Fused Triton decode of a (3,5,8) packed dict -> bf16 weight tensor.

    Falls back to the scatter path if Triton/CUDA is unavailable or the scheme
    is not (3,5,8).
    """
    if not _HAS_TRITON or not torch.cuda.is_available():
        return decode_weight_scatter(packed, device, dtype)
    if tuple(packed["level_bits"]) != (3, 5, 8):
        return decode_weight_scatter(packed, device, dtype)

    device = device if device is not None else torch.device("cuda")
    of, inf, N = packed["out_f"], packed["in_f"], packed["N"]
    b1, b2 = packed["bitmaps"]
    l1, l2, l3 = packed["streams"]
    scale = packed["scale"].to(device)

    out = torch.empty(N, dtype=torch.bfloat16, device=device)
    blk = TRITON_BLOCK
    n_blk = (N + blk - 1) // blk
    b1_blk, b2_blk = precompute_block_offsets(packed, blk, device)

    _ix_decode_kernel[(n_blk,)](
        out,
        b1.to(device),
        b2.to(device),
        l1.to(device),
        l2.to(device),
        l3.to(device),
        b1_blk,
        b2_blk,
        scale,
        N,
        BLK=blk,
    )
    return out.view(of, inf).to(dtype)
