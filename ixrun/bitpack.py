"""Bit-stream packing / unpacking utilities.

All packers operate on int32 word arrays (little-endian, bit 0 = LSB of word 0),
matching the layout consumed by the Triton decode kernels.
"""
import math

import torch

__all__ = ["pack_bits_stream", "unpack_bits_stream", "pack_bitmap"]


def pack_bits_stream(vals: torch.Tensor, bits: int, chunk_vals: int = 1 << 21) -> torch.Tensor:
    """Pack a tensor of unsigned values (each ``bits`` wide) into int32 words.

    The output is a 1-D ``torch.int32`` tensor where consecutive values are
    laid out bit-by-bit across 32-bit words (LSB-first within and across words).

    Memory-lean: processes in word-aligned chunks so the int64 bit-expansion
    temporary is capped (~chunk_vals*bits*8 bytes, default ~50-100MB), which
    matters when packing very large (100M+ element) streams.

    Parameters
    ----------
    vals : 1-D int tensor (any int dtype >= int16).
    bits : number of bits per value (1..31).
    chunk_vals : values per chunk; rounded up to a multiple of 32/gcd(bits,32)
        so every chunk packs to whole int32 words and chunks concatenate.
    """
    if bits <= 0:
        raise ValueError("bits must be positive")
    vals = vals.to(torch.int64).reshape(-1)
    n = vals.numel()
    if n == 0:
        return torch.zeros(0, dtype=torch.int32)

    # word-aligned value count: bits*v must fill whole 32-bit words
    vals_per_word_group = 32 // math.gcd(bits, 32)
    step = max(chunk_vals, vals_per_word_group)
    step = ((step + vals_per_word_group - 1) // vals_per_word_group) * vals_per_word_group

    word_arange = torch.arange(32, dtype=torch.int64)
    bit_arange = torch.arange(bits, dtype=torch.int64)
    total_words = (n * bits + 31) // 32
    out = torch.empty(total_words, dtype=torch.int32)
    for start in range(0, n, step):
        v = vals[start : start + step]
        orig_nbits = v.numel() * bits
        nw = (orig_nbits + 31) // 32  # words produced by this chunk
        bits_flat = ((v.unsqueeze(-1) >> bit_arange) & 1).reshape(-1)
        need = nw * 32
        if bits_flat.numel() < need:
            bits_flat = torch.cat(
                [bits_flat, torch.zeros(need - bits_flat.numel(), dtype=torch.int64)]
            )
        else:
            bits_flat = bits_flat[:need]
        out[start * bits // 32 : start * bits // 32 + nw] = (
            (bits_flat.reshape(nw, 32) << word_arange).sum(-1).to(torch.int32)
        )
    return out


def unpack_bits_stream(
    packed: torch.Tensor, n_elements: int, bits: int, device=None
) -> torch.Tensor:
    """Inverse of :func:`pack_bits_stream`.

    Returns a 1-D int32 tensor of length ``n_elements``.
    Works on CPU or GPU (device follows ``packed`` unless overridden).
    """
    if n_elements == 0:
        dev = device if device is not None else packed.device
        return torch.zeros(0, dtype=torch.int32, device=dev)
    device = device if device is not None else packed.device
    packed = packed.to(device)
    pos = torch.arange(n_elements, device=device, dtype=torch.long)
    out = torch.zeros(n_elements, dtype=torch.int32, device=device)
    for b in range(bits):
        bp = pos * bits + b
        word_idx = bp // 32
        bit_idx = bp % 32
        out += ((packed[word_idx] >> bit_idx) & 1).to(torch.int32) << b
    return out


def pack_bitmap(mask: torch.Tensor) -> torch.Tensor:
    """Pack a boolean mask into a 1-bit bitstream (int32 words)."""
    return pack_bits_stream(mask.to(torch.int32), 1)
