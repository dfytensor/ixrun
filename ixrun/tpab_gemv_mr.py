"""Multi-row TPAB GEMV: R rows per program sharing tile headers.

Current kernel: 1 row/program, serial walk over T_C tiles. Each tile's
header triple (bits/scales/goff) is loaded per-program => per-row.
Multi-row version: program handles R consecutive rows (same row-tile
when R<=TILE_R), loads each tile header ONCE, extracts codes for all R
rows (their bitpos share the same goff/gbase/b), and dot-accumulates.

Grid = (out_f / R,). R rows x TILE_C cols per tile iteration => bigger
register tiles, better ILP, R-fold fewer header loads.
"""
from __future__ import annotations
import torch

__all__ = ["fused_gemv_tpab_mr"]

try:
    import triton
    import triton.language as tl

    _HAS = True
except Exception:
    _HAS = False

if _HAS:

    @triton.jit
    def _tpab_gemv_mr_kernel(
        x_ptr, y_ptr,
        body_ptr, bits_ptr, scales_ptr, goff_ptr, gbase_ptr,
        olk_ptr, olv_ptr, oloff_ptr,
        T_C: tl.constexpr,
        IN_F: tl.constexpr,
        TILE_R: tl.constexpr, TILE_C: tl.constexpr,
        R: tl.constexpr,            # rows per program (<= TILE_R)
    ):
        pid = tl.program_id(0)
        n0 = pid * R

        acc = tl.zeros((R,), dtype=tl.float32)
        # all R rows share the row-tile iff n0..n0+R-1 within one tile:
        # guaranteed when R divides TILE_R and rows are tile-aligned groups
        row_tile = n0 // TILE_R
        rows_in_tile = n0 % TILE_R + tl.arange(0, R)          # [R] in-tile idx

        for kc in tl.range(0, T_C):
            t = row_tile * T_C + kc
            b = tl.load(bits_ptr + t).to(tl.int32)
            s = tl.load(scales_ptr + t).to(tl.float32)
            gbase = tl.load(gbase_ptr + b)
            goff = tl.load(goff_ptr + t).to(tl.int64)

            L = rows_in_tile[:, None] * TILE_C + tl.arange(0, TILE_C)[None, :]  # [R, C]
            bitpos = gbase + (goff + L.to(tl.int64)) * b        # [R, C]
            word = (bitpos // 32).to(tl.int32)
            shift = (bitpos % 32).to(tl.int32)

            w1 = tl.load(body_ptr + word).to(tl.uint32)         # [R, C]
            cross = (shift + b) > 32
            w2 = tl.where(cross, tl.load(body_ptr + word + 1).to(tl.uint32),
                          tl.zeros((R, TILE_C), tl.uint32))
            raw = tl.where(cross, (w1 >> shift) | (w2 << (32 - shift)),
                           w1 >> shift)
            mask = tl.exp2(b.to(tl.float32)).to(tl.int32) - 1
            v = (raw & mask.to(tl.uint32)).to(tl.int32) - ((mask + 1) // 2 - 1)

            k0 = kc * TILE_C + tl.arange(0, TILE_C)
            x = tl.load(x_ptr + k0).to(tl.float32)              # [C]
            wq = (v.to(tl.float32) * s).to(tl.bfloat16).to(tl.float32)
            acc += tl.sum(wq * x[None, :], axis=1)              # [R]

        # outlier overlay per row
        for r in tl.static_range(R):
            n = n0 + r
            lo = tl.load(oloff_ptr + n)
            hi = tl.load(oloff_ptr + n + 1)
            part = 0.0
            for j in tl.range(lo, hi):
                kk = tl.load(olk_ptr + j)
                vv = tl.load(olv_ptr + j)
                part += vv.to(tl.bfloat16).to(tl.float32) * tl.load(x_ptr + kk).to(tl.float32)
            acc = tl.where(tl.arange(0, R) == r, acc + part, acc)

        tl.store(y_ptr + n0 + tl.arange(0, R), acc.to(tl.bfloat16))


def fused_gemv_tpab_mr(x: torch.Tensor, st: dict, out_f: int, in_f: int,
                       tile_r: int = 64, r: int = 4):
    """y = x @ W.T, R rows per program. Requires out_f % r == 0 and
    r <= tile_r, rows grouped inside tiles (r divides tile_r)."""
    y = torch.empty(out_f, dtype=torch.bfloat16, device=x.device)
    assert tile_r % r == 0, "r must divide tile_r"
    _tpab_gemv_mr_kernel[(out_f // r,)](
        x.reshape(-1), y,
        st["body_g"], st["bits_g"], st["scales_g"], st["goff_g"], st["gbase_g"],
        st["ol_row_k"], st["ol_row_v"], st["ol_offs"],
        T_C=in_f // 64, IN_F=in_f, TILE_R=tile_r, TILE_C=64, R=r,
        num_warps=4,
    )
    return y
