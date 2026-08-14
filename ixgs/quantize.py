"""bf16 -> per-group int8 -> (3,5,8) nested-bitmap quantization (Group-Scale INT8-X).

Design (validated on MiniMax-H3 video DiT, see README.md):
  1. weights split into groups of `group_size` (default 64)
  2. per-group scale = group_max_abs / 15  -> max |int| = 15 (fits L1+L2)
  3. value-range tiers on the integers (lossless, no percentile clipping):
       L1: v in [-3, 4]   -> 3 bits
       L2: v in [-15, 16] -> 5 bits
       L3: |v| > 15       -> 8 bits raw
  4. nested bitmaps b1/b2 index membership; group scales stored fp16.

Encoding layer is 100% lossless on the integer representation.
"""
from __future__ import annotations
import torch

from .bitpack import pack_bits_stream, unpack_bits_stream

__all__ = [
    "GROUP_SIZE",
    "SCALE_DIVISOR",
    "int8gs_quantize",
    "decode_weight_scatter",
    "per_tensor_int8_reference",
]

GROUP_SIZE = 64
SCALE_DIVISOR = 15.0  # group_max / 15 -> max |int| = 15, L3 empty by construction

_L1_MIN, _L1_MAX = -3, 4
_L2_MIN, _L2_MAX = -15, 16


@torch.no_grad()
def int8gs_quantize(weight: torch.Tensor, group_size: int = GROUP_SIZE) -> dict:
    """Quantize a 2-D bf16 weight with per-group scale + (3,5,8) packing.

    Returns dict with keys:
        out_f, in_f, N, group_size, group_scales (fp16 [n_groups]),
        b1, b2 (nested bitmaps, int32), l1, l2 (bit streams, int32),
        l3 (raw uint8, >=1 element), counts [n1, n2, n3],
        total_bytes, bits_per_weight, compression_vs_bf16.
    """
    if weight.dim() != 2:
        raise ValueError(f"expected 2-D weight, got {list(weight.shape)}")
    of, inf_ = weight.shape
    N = of * inf_
    dev = weight.device

    w = weight.float().reshape(-1).to(dev if dev.type == "cuda" else "cpu")
    n_groups = (N + group_size - 1) // group_size
    pad = n_groups * group_size - N
    if pad:
        w = torch.cat([w, torch.zeros(pad, device=w.device)])
    group_max = w.view(n_groups, group_size).abs().max(dim=1).values
    group_scales = (group_max / SCALE_DIVISOR).clamp(min=1e-10)  # [n_groups]
    # round-trip through fp16 NOW so quantize and decode see the identical scale
    group_scales = group_scales.to(torch.float16).float()
    scale_per_elem = group_scales.repeat_interleave(group_size)[:N]

    int_vals = (w[:N] / scale_per_elem).round().clamp(-127, 128).to(torch.int32)

    # value-range tiers (lossless)
    is_l1 = (int_vals >= _L1_MIN) & (int_vals <= _L1_MAX)
    non_l1 = ~is_l1
    non_l1_vals = int_vals[non_l1]
    is_l2_non = (non_l1_vals >= _L2_MIN) & (non_l1_vals <= _L2_MAX)

    n1 = int(is_l1.sum().item())
    n2 = int(is_l2_non.sum().item())
    n3 = N - n1 - n2

    l1_v = (int_vals[is_l1] - _L1_MIN).to(torch.int64)
    l2_v = (non_l1_vals[is_l2_non] - _L2_MIN).to(torch.int64)
    l3_v = (non_l1_vals[~is_l2_non] + 127).to(torch.uint8)

    l1 = pack_bits_stream(l1_v, 3)
    l2 = pack_bits_stream(l2_v, 5)
    # L3 stream must have >=1 element: empty stream crashes the Triton kernel
    l3 = l3_v if n3 > 0 else torch.zeros(1, dtype=torch.uint8)
    b1 = pack_bits_stream(is_l1.to(torch.int64), 1)
    b2 = (
        pack_bits_stream(is_l2_non.to(torch.int64), 1)
        if (n2 + n3) > 0
        else torch.zeros(1, dtype=torch.int32)
    )

    total = (
        l1.numel() * 4
        + l2.numel() * 4
        + max(n3, 1) * 1
        + b1.numel() * 4
        + b2.numel() * 4
        + n_groups * 2  # fp16 group scales
    )

    return {
        "out_f": of,
        "in_f": inf_,
        "N": N,
        "group_size": group_size,
        "group_scales": group_scales.cpu().to(torch.float16),
        "b1": b1.cpu(),
        "b2": b2.cpu(),
        "l1": l1.cpu(),
        "l2": l2.cpu(),
        "l3": l3.cpu(),
        "counts": [n1, n2, n3],
        "total_bytes": total,
        "bits_per_weight": (total * 8) / N,
        "compression_vs_bf16": (N * 2) / total,
    }


@torch.no_grad()
def decode_weight_scatter(packed: dict, device=None, dtype=torch.bfloat16) -> torch.Tensor:
    """Pure-PyTorch decode (no Triton). Inverse of int8gs_quantize."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    N = packed["N"]
    gs = packed["group_size"]
    b1 = packed["b1"].to(device)
    b2 = packed["b2"].to(device)
    scales = packed["group_scales"].to(device).float()

    pos = torch.arange(N, device=device, dtype=torch.long)
    is_l1 = ((b1[pos // 32] >> (pos % 32)) & 1).bool()
    non_l1_idx = (~is_l1).nonzero(as_tuple=True)[0]
    n_non = non_l1_idx.shape[0]

    out = torch.zeros(N, dtype=torch.int32, device=device)
    n1, n2, n3 = packed["counts"]
    if n1 > 0:
        l1_vals = unpack_bits_stream(packed["l1"].to(device), n1, 3, device=device) + _L1_MIN
        ranks = is_l1.long().cumsum(0) - 1
        out[is_l1] = l1_vals[ranks[is_l1]]
    if n_non > 0:
        p2 = torch.arange(n_non, device=device, dtype=torch.long)
        is_l2_non = ((b2[p2 // 32] >> (p2 % 32)) & 1).bool()
        if n2 > 0:
            l2_vals = unpack_bits_stream(packed["l2"].to(device), n2, 5, device=device) + _L2_MIN
            ranks2 = is_l2_non.long().cumsum(0) - 1
            l2_positions = non_l1_idx[is_l2_non]
            out[l2_positions] = l2_vals[ranks2[is_l2_non]]
        if n3 > 0:
            l3_vals = packed["l3"].to(device)[:n3].to(torch.int32) - 127
            l3_positions = non_l1_idx[~is_l2_non]
            out[l3_positions] = l3_vals

    scale_per_elem = scales.repeat_interleave(gs)[:N]
    return (out.float() * scale_per_elem).to(dtype).view(packed["out_f"], packed["in_f"])


@torch.no_grad()
def per_tensor_int8_reference(weight: torch.Tensor) -> torch.Tensor:
    """Per-tensor INT8-X baseline (ixrun v1 quantization layer) for comparison."""
    scale = weight.abs().max().clamp(min=1e-8) / 127.0
    return ((weight.float() / scale).round().clamp(-127, 127) * scale).to(weight.dtype)
