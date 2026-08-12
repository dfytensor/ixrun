"""Bit-stream packing / unpacking utilities.

All packers operate on int32 word arrays (little-endian, bit 0 = LSB of word 0),
matching the layout consumed by the Triton decode kernels.
"""
import torch

__all__ = ["pack_bits_stream", "unpack_bits_stream", "pack_bitmap"]


def pack_bits_stream(vals: torch.Tensor, bits: int) -> torch.Tensor:
    """Pack a tensor of unsigned values (each ``bits`` wide) into int32 words.

    The output is a 1-D ``torch.int32`` tensor where consecutive values are
    laid out bit-by-bit across 32-bit words (LSB-first within and across words).

    Parameters
    ----------
    vals : 1-D int tensor (any int dtype >= int16).
    bits : number of bits per value (1..31).
    """
    if bits <= 0:
        raise ValueError("bits must be positive")
    vals = vals.to(torch.int64).reshape(-1)
    n = vals.numel()
    if n == 0:
        return torch.zeros(0, dtype=torch.int32)
    nbits = n * bits
    pad_bits = (-nbits) % 32
    if pad_bits:
        vals = torch.cat([vals, vals.new_zeros((-(-nbits) % 32 + bits - 1) // bits)])
    # expand each value into `bits` individual bits
    bit_arange = torch.arange(bits, dtype=torch.int64, device=vals.device)
    bits_flat = ((vals.unsqueeze(-1) >> bit_arange) & 1).reshape(-1)
    nw = (bits_flat.numel() + 31) // 32
    if nw * 32 > bits_flat.numel():
        bits_flat = torch.cat(
            [bits_flat, torch.zeros(nw * 32 - bits_flat.numel(), dtype=torch.int64, device=vals.device)]
        )
    word_arange = torch.arange(32, dtype=torch.int64, device=vals.device)
    words = (bits_flat.reshape(nw, 32) << word_arange).sum(-1)
    return words.to(torch.int32)


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
