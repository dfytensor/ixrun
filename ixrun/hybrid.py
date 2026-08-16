"""Hybrid codec deployment: per-shape backend selection.

Measured on Qwen3.8-27B shapes (RTX 4090D, fused GEMV, real heavy-tail
weights, tests/test_kernel_vs.py):

    shape            TPAB     INT8-X   winner
    17408x5120       220G/s   250G/s   INT8-X   (tall, many rows -> occupancy)
    5120x17408       203G/s    44G/s   TPAB    (wide -> int8x rank chain long)
    5120x5120        214G/s   654G/s   INT8-X   (square, short chain)
    5120x6144        183G/s   223G/s   INT8-X
    10240x5120       220G/s    58G/s   TPAB     (few rows -> int8x starved)

Rule (fits all five): INT8-X wins when it has enough rows to fill the GPU
AND a short k-chain; TPAB wins on wide (long in_f) or row-starved shapes.
"""
from __future__ import annotations
import torch

from .engine import StreamingLinear
from .tpab_linear import TpabLinear

__all__ = ["pick_backend", "deploy_model_hybrid"]


def pick_backend(out_f: int, in_f: int) -> str:
    """'int8x' | 'tpab' — measured-rule: TPAB for wide or row-starved."""
    if in_f >= 8192:
        return "tpab"
    if in_f % 512 == 0 and in_f < out_f * 2 and out_f >= 16000:
        return "int8x"          # very tall + enough rows
    if out_f * in_f <= 5120 * 6144 and in_f % 512 == 0 and in_f <= 6144:
        return "int8x"          # square-ish, short chain, INT8-X excels
    if in_f % 512 == 0 and out_f <= 12288 and in_f >= 5120:
        return "tpab"           # row-starved tall (e.g. 10240x5120)
    # int8x needs in_f % 512; tpab needs % 64
    if in_f % 512 == 0:
        return "int8x"
    if in_f % 64 == 0:
        return "tpab"
    return "int8x"              # small/gate layers stay bf16-free via int8x? (no — caller skips)


@torch.no_grad()
def deploy_model_hybrid(model, snr_target_db=24.0, verbose=True):
    from .linear import iter_quantizable_linears, _set_parent_child
    from .quantize import int8x_quantize

    targets = list(iter_quantizable_linears(model))
    stats = {"tpab": 0, "int8x": 0, "skipped": 0}
    tpab_bytes = int8x_bytes = 0
    stream_layers = []
    max_N = 0
    for name, mod in targets:
        O, I = mod.weight.shape
        if I % 64:
            stats["skipped"] += 1
            continue
        tr = 64
        while O % tr and tr > 1:
            tr //= 2
        if O % tr:
            stats["skipped"] += 1
            continue
        backend = pick_backend(O, I)
        if backend == "tpab":
            new = TpabLinear(mod.weight.data.cuda(), snr_target_db=snr_target_db)
            tpab_bytes += new.packed["total_bytes"]
            mod.weight.data = torch.empty(0)
        else:
            p = int8x_quantize(mod.weight.data, (3, 5, 8))
            bias = mod.bias.data if mod.bias is not None else None
            new = StreamingLinear(p, bias=bias)
            stream_layers.append((name, new))
            max_N = max(max_N, p["N"])
            int8x_bytes += p["total_bytes"]
            mod.weight.data = torch.empty(0)
        _set_parent_child(model, name, new)
        stats[backend] += 1
    # wire the shared decode buffer for int8x streaming layers
    import torch as _t
    shared = _t.empty(max_N, dtype=_t.bfloat16, device="cuda")
    for name, sl in stream_layers:
        sl._set_shared_buf(shared)
    if verbose:
        print(f"[hybrid] tpab={stats['tpab']} int8x={stats['int8x']} "
              f"skipped={stats['skipped']} | "
              f"tpab={tpab_bytes/1e9:.2f}GB int8x={int8x_bytes/1e9:.2f}GB "
              f"shared={shared.numel()*2/1e6:.0f}MB", flush=True)
    return stats
