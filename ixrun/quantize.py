"""bf16 -> int8 -> nested-bitmap (INT8-X) quantization.

Pipeline:
  1. bf16 weight -> per-tensor int8  (scale = max_abs / 127)
  2. int8 values split into N magnitude levels, each losslessly packed
     into fewer bits via a nested-bitmap structure.

Default scheme (3, 5, 8):
  L1: |v| <= 3   (~55% of weights)  -> 3 bits
  L2: 3 < |v| <= 15 (~40%)          -> 5 bits
  L3: |v| > 15   (~4%)              -> 8 bits
  Nested bitmaps add ~1.49 bits/w flag overhead.
  Total ~5.5 bits/w  ->  ~2.9x compression vs bf16, lossless on the int8.
"""
from __future__ import annotations
import math
import torch

from .bitpack import pack_bits_stream, pack_bitmap
from .config import BIT_TO_THRESHOLD, DEFAULT_LEVELS

__all__ = ["int8x_quantize", "compute_positions", "decode_to_weight", "DEFAULT_LEVELS"]


def _level_masks(abs_v: torch.Tensor, level_bits: tuple) -> list:
    """Return list of boolean masks, one per level (mutually exclusive, covering all)."""
    thresholds = [BIT_TO_THRESHOLD[b] for b in level_bits]
    masks = []
    prev_t = -1
    for i, t in enumerate(thresholds):
        if i == len(thresholds) - 1:
            mask = abs_v > prev_t
        else:
            mask = (abs_v > prev_t) & (abs_v <= t)
        masks.append(mask)
        prev_t = t
    return masks


@torch.no_grad()
def int8x_quantize(weight: torch.Tensor, level_bits: tuple = DEFAULT_LEVELS) -> dict:
    """Quantize a bf16 weight matrix into an INT8-X packed dict.

    Parameters
    ----------
    weight : 2-D tensor (out_features, in_features), typically bf16.
    level_bits : tuple of bit widths, ascending, last must be 8.

    Returns
    -------
    dict with keys:
        level_bits, out_f, in_f, N, scale (bf16 scalar),
        bitmaps (list[int32]), streams (list[Tensor]),
        counts (list[int]), total_bytes (int), bits_per_weight (float).
    """
    level_bits = tuple(sorted(level_bits))
    if level_bits[-1] != 8:
        raise ValueError("last level must be 8-bit to cover the full int8 range")

    of, inf = weight.shape
    N = of * inf

    # 1. bf16 -> per-tensor int8 (bookkeeping on CPU; quantize is one-time)
    scale = weight.abs().max().clamp(min=1e-8) / 127.0
    i8 = (
        (weight.float() / scale)
        .round()
        .clamp(-127, 127)
        .to(torch.int8)
        .reshape(-1)
        .cpu()
    )
    scale = scale.detach().cpu().bfloat16().reshape(())

    abs_v = i8.abs()
    masks = _level_masks(abs_v, level_bits)
    counts = [int(m.sum().item()) for m in masks]

    # 2. per-level packed streams (unsigned offset makes values >= 0)
    streams = []
    for i, (b, m) in enumerate(zip(level_bits, masks)):
        offset = BIT_TO_THRESHOLD[b]
        vals = (i8[m] + offset).to(torch.int32)
        if i == len(level_bits) - 1 and b == 8:
            # last 8-bit level: raw uint8 (1 byte/elem), no bit-packing needed
            streams.append(vals.to(torch.uint8))
        else:
            streams.append(pack_bits_stream(vals, b))

    # 3. nested bitmaps: B_k indexed over the set of elements not yet assigned
    bitmaps = []
    remaining = torch.ones(N, dtype=torch.bool)
    for i in range(len(level_bits) - 1):
        bm = masks[i][remaining]
        bitmaps.append(pack_bitmap(bm))
        remaining = remaining & ~masks[i]

    # 4. storage accounting
    total = 2  # scale (fp16)
    for i, (b, s) in enumerate(zip(level_bits, streams)):
        if i == len(level_bits) - 1 and b == 8:
            total += s.numel()  # uint8: 1 byte/elem
        else:
            total += s.numel() * 4  # int32 words
    for bm in bitmaps:
        total += bm.numel() * 4

    bpw = (total * 8) / N

    return {
        "level_bits": level_bits,
        "out_f": of,
        "in_f": inf,
        "N": N,
        "scale": scale,
        "bitmaps": bitmaps,
        "streams": streams,
        "counts": counts,
        "total_bytes": total,
        "bits_per_weight": bpw,
        "compression_vs_bf16": (N * 2) / total,
    }


@torch.no_grad()
def compute_positions(packed: dict) -> list:
    """Derive per-level global scatter positions from the nested bitmaps.

    One-time cost; used by the scatter-based decode path. The positions are
    returned as int64 tensors on CPU; callers move to the desired device.
    """
    level_bits = packed["level_bits"]
    N = packed["N"]
    i8_placeholder = torch.zeros(N, dtype=torch.int32)
    abs_fake = torch.zeros(N, dtype=torch.int32)
    # rebuild masks from bitmaps to avoid storing them
    bitmaps = packed["bitmaps"]
    from .bitpack import unpack_bits_stream

    remaining = torch.ones(N, dtype=torch.bool)
    masks = []
    for i in range(len(level_bits) - 1):
        bm_vals = unpack_bits_stream(bitmaps[i], int(remaining.sum()), 1)
        cur_mask = torch.zeros(N, dtype=torch.bool)
        idx = remaining.nonzero(as_tuple=True)[0]
        cur_mask[idx[: len(bm_vals)]] = bm_vals.bool()
        masks.append(cur_mask & remaining)
        remaining = remaining & ~masks[i]
    masks.append(remaining.clone())  # last level
    return [m.nonzero(as_tuple=True)[0].to(torch.int64) for m in masks]


@torch.no_grad()
def decode_to_weight(packed: dict, device=None, dtype=torch.bfloat16) -> torch.Tensor:
    """Decode an INT8-X packed dict back to a full weight tensor.

    Robust scatter-based path: unpack each level stream and scatter to the
    reconstructed positions, then apply the per-tensor scale.
    """
    from .bitpack import unpack_bits_stream

    level_bits = packed["level_bits"]
    of, inf, N = packed["out_f"], packed["in_f"], packed["N"]
    scale = packed["scale"].to(device)
    streams = [s.to(device) for s in packed["streams"]]

    positions = compute_positions(packed)
    positions = [p.to(device) for p in positions]

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

    w = (i8_flat.float() * scale.float()).to(dtype)
    return w.view(of, inf)
