"""TPAB GEMV split-K: free k-direction splitting (tile headers carry goff,
no prefix state) — cuts the per-program serial tile walk on tall/square
shapes where TPAB trailed INT8-X (214 vs 654 G/s on 5120x5120).

grid = (out_f, S); partial sums accumulate via fp32 atomics into y32
(caller zeroes + converts), mirroring the INT8-X split-K contract.
"""
from __future__ import annotations
import torch

__all__ = ["fused_gemv_tpab_splitk"]

try:
    import triton
    import triton.language as tl

    _HAS = True
except Exception:
    _HAS = False

if _HAS:

    @triton.jit
    def _tpab_gemv_splitk_kernel(
        x_ptr, y_ptr,               # y: fp32 [out_f], atomically accumulated
        body_ptr, bits_ptr, scales_ptr, goff_ptr, gbase_ptr,
        T_C: tl.constexpr,          # tiles per row (in_f // 64)
        TILE_R: tl.constexpr, TILE_C: tl.constexpr,
        TPS: tl.constexpr,          # tiles per split (T_C // S)
    ):
        n = tl.program_id(0)
        s = tl.program_id(1)
        row_tile = n // TILE_R
        in_row = n % TILE_R

        acc = 0.0
        for kc in tl.range(s * TPS, (s + 1) * TPS):
            t = row_tile * T_C + kc
            b = tl.load(bits_ptr + t).to(tl.int32)
            sc = tl.load(scales_ptr + t).to(tl.float32)
            gbase = tl.load(gbase_ptr + b).to(tl.int32)
            goff = tl.load(goff_ptr + t)

            L = in_row * TILE_C + tl.arange(0, TILE_C)
            # int32 shifts instead of int64 div/mod (see tpab.py note)
            bitpos = gbase + goff * b + L * b
            word = bitpos >> 5
            shift = bitpos & 31

            w1 = tl.load(body_ptr + word).to(tl.uint32)
            cross = (shift + b) > 32
            w2 = tl.where(cross, tl.load(body_ptr + word + 1).to(tl.uint32),
                          tl.zeros((TILE_C,), tl.uint32))
            raw = tl.where(cross, (w1 >> shift) | (w2 << (32 - shift)),
                           w1 >> shift)
            mask = tl.exp2(b.to(tl.float32)).to(tl.int32) - 1
            v = (raw & mask.to(tl.uint32)).to(tl.int32) - ((mask + 1) // 2 - 1)

            k0 = kc * TILE_C + tl.arange(0, TILE_C)
            x = tl.load(x_ptr + k0).to(tl.float32)
            wq = (v.to(tl.float32) * sc).to(tl.bfloat16).to(tl.float32)
            acc += tl.sum(x * wq, axis=0)

        tl.atomic_add(y_ptr + n, acc)


def fused_gemv_tpab_splitk(x: torch.Tensor, st: dict, out_f: int, in_f: int,
                           tile_r: int = 64, split: int = 1,
                           y32: torch.Tensor | None = None):
    """y = x @ W.T single token, TPAB-decoded, k split across `split`
    programs per row. y32 (fp32 [out_f]) is zeroed here and returned —
    caller converts to bf16 (+ outliers handled outside as usual).

    NOTE: outliers must be overlaid by the caller on the bf16 result
    (same as the non-split kernel's contract with its in-kernel loop).
    """
    assert _HAS
    if y32 is None:
        y32 = torch.zeros(out_f, dtype=torch.float32, device=x.device)
    else:
        y32.zero_()
    T_C = in_f // 64
    tps = T_C // split
    _tpab_gemv_splitk_kernel[(out_f, split)](
        x.reshape(-1), y32,
        st["body_g"], st["bits_g"], st["scales_g"], st["goff_g"], st["gbase_g"],
        T_C=T_C, TILE_R=tile_r, TILE_C=64, TPS=tps,
        num_warps=2,
    )
    return y32
