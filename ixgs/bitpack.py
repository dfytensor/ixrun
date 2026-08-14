"""Bit-stream packing utilities (self-contained copy of ixrun/bitpack + GPU fast path)."""
import torch

__all__ = ["pack_bits_stream", "unpack_bits_stream", "pack_bitmap"]


def pack_bits_stream_cpu(vals: torch.Tensor, bits: int) -> torch.Tensor:
    vals = vals.to(torch.int64).reshape(-1)
    n = vals.numel()
    if n == 0:
        return torch.zeros(0, dtype=torch.int32)
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


def pack_bits_stream_gpu(vals: torch.Tensor, bits: int) -> torch.Tensor:
    """Memory-efficient scatter_add packing. Requires vals on CUDA.

    Correct because each output bit position belongs to exactly one input
    value, so scatter_add == bitwise OR. Values fit in int64 without overflow
    (bits <= 31, shifted stays < 2^63).
    """
    N = vals.shape[0]
    if N == 0:
        return torch.zeros(0, dtype=torch.int32, device=vals.device)
    positions = torch.arange(N, device=vals.device, dtype=torch.long) * bits
    wi = positions >> 5
    si = (positions & 31).to(torch.int64)
    n_words = (N * bits + 31) // 32
    v = vals.to(torch.int64) & ((1 << bits) - 1)
    low = (v << si) & 0xFFFFFFFF
    stream = torch.zeros(n_words, device=vals.device, dtype=torch.int64)
    stream.scatter_add_(0, wi, low)
    cross = (si + bits) > 32
    if cross.any():
        cw = wi[cross] + 1
        cv = v[cross] >> (32 - si[cross])
        valid = cw < n_words
        stream.scatter_add_(0, cw[valid], cv[valid])
    return stream.to(torch.int32)


def pack_bits_stream(vals: torch.Tensor, bits: int) -> torch.Tensor:
    if vals.is_cuda:
        return pack_bits_stream_gpu(vals, bits)
    return pack_bits_stream_cpu(vals, bits)


def unpack_bits_stream(
    packed: torch.Tensor, n_elements: int, bits: int, device=None
) -> torch.Tensor:
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
    return pack_bits_stream(mask.to(torch.int32), 1)
