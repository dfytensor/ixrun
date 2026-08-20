"""Fused TPAB decode + GEMM: one kernel, weights never materialize.

For multi-token forwards (prefill, batched decode) the current path is
decode_tpab_triton (whole layer -> f32 workspace) + cuBLAS GEMM, paying
a workspace round-trip (530MB for 17408x5120) plus an f32->bf16 convert.
This kernel fuses: each program computes one [BM, BN] output tile by
walking the k-axis in TILE_C chunks, decoding the [TILE_R, TILE_C]
weight block in-register and contracting with tl.dot (tensor cores).

Random tile access + per-tile headers (TPAB's structural advantage over
INT8-X's global rank streams) are what make this possible.

Grid: (cdiv(M,BM) * cdiv(N,BN)); K walks in 64-steps (= TILE_C).
Weight tile needed: rows [n0, n0+BN) x cols [k0, k0+64) — spans
ceil(BN/TILE_R) row-tiles x 1 col-tile; decode each 64x64 (or TILE_R x
64) block, multiply into the BM x BN accumulator.
"""
from __future__ import annotations
import torch

__all__ = ["fused_gemm_tpab"]

try:
    import triton
    import triton.language as tl

    _HAS = True
except Exception:
    _HAS = False

if _HAS:

    @triton.jit
    def _tpab_gemm_kernel(
        x_ptr, y_ptr,               # x [M, IN_F] bf16, y [M, OUT_F] bf16
        body_ptr, bits_ptr, scales_ptr, goff_ptr, gbase_ptr,
        M, OUT_F,
        T_C: tl.constexpr, IN_F: tl.constexpr,
        TILE_R: tl.constexpr, TILE_C: tl.constexpr,
        BM: tl.constexpr, BN: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        m0 = pid_m * BM
        n0 = pid_n * BN

        offs_m = m0 + tl.arange(0, BM)            # [BM]
        offs_n = n0 + tl.arange(0, BN)            # [BN]
        mask_m = offs_m < M
        mask_n = offs_n < OUT_F

        acc = tl.zeros((BM, BN), dtype=tl.float32)

        for kc in tl.range(0, T_C):
            k0 = kc * TILE_C
            # load activation tile [BM, TILE_C]
            x_tile = tl.load(x_ptr + offs_m[:, None] * IN_F + (k0 + tl.arange(0, TILE_C))[None, :],
                             mask=mask_m[:, None], other=0.0)      # bf16

            # decode weight blocks [TILE_R, TILE_C] and contract each into
            # its slice of acc rows (BN split into BN//TILE_R row-tiles;
            # accumulating block-diagonal via masked writes to acc slices
            # is awkward — instead compute per-row-tile partial dots and
            # add into the matching acc rows using tl.where reshaping.
            # Simplest correct approach: loop row-tiles, each producing a
            # [TILE_R, BN] zero tensor with its rows filled is wasteful —
            # so we accumulate into acc by row ranges via a second dot per
            # row-tile into a [BM, TILE_R] block and store to a register
            # staging of [BM, BN] is impossible in Triton lists.
            # => Restrict BN == TILE_R (64): one weight block per program.
            # Programs with larger BN simply don't exist; grid uses BN=64.
            t = (n0 // TILE_R) * T_C + kc
            b = tl.load(bits_ptr + t).to(tl.int32)
            s = tl.load(scales_ptr + t).to(tl.float32)
            gbase = tl.load(gbase_ptr + b).to(tl.int32)
            goff = tl.load(goff_ptr + t)

            L = tl.arange(0, TILE_R)[:, None] * TILE_C + tl.arange(0, TILE_C)[None, :]
            bitpos = gbase + goff * b + L * b
            word = bitpos >> 5
            shift = bitpos & 31
            w1 = tl.load(body_ptr + word).to(tl.uint32)
            cross = (shift + b) > 32
            w2 = tl.where(cross, tl.load(body_ptr + word + 1).to(tl.uint32),
                          tl.zeros((TILE_R, TILE_C), tl.uint32))
            raw = tl.where(cross, (w1 >> shift) | (w2 << (32 - shift)),
                           w1 >> shift)
            mask = tl.exp2(b.to(tl.float32)).to(tl.int32) - 1
            v = (raw & mask.to(tl.uint32)).to(tl.int32) - ((mask + 1) // 2 - 1)
            wq = (v.to(tl.float32) * s).to(tl.bfloat16)        # [TILE_R, TILE_C]
            acc += tl.dot(x_tile, tl.trans(wq), out_dtype=tl.float32)

        y_ptrs = y_ptr + offs_m[:, None] * OUT_F + offs_n[None, :]
        tl.store(y_ptrs, acc.to(tl.bfloat16), mask=mask_m[:, None] & mask_n[None, :])


def fused_gemm_tpab(x: torch.Tensor, st: dict, out_f: int, in_f: int,
                    tile_r: int = 64, bm: int = 32, bn: int = 64):
    """y = x @ W.T — multi-token, TPAB-decoded in-register, no workspace.

    x: [M, in_f] bf16 (M = tokens in the step).
    """
    M = x.shape[0]
    y = torch.empty(M, out_f, dtype=torch.bfloat16, device=x.device)
    grid = (triton.cdiv(M, bm), triton.cdiv(out_f, bn))
    _tpab_gemm_kernel[grid](
        x, y,
        st["body_g"], st["bits_g"], st["scales_g"], st["goff_g"], st["gbase_g"],
        M, out_f,
        T_C=in_f // 64, IN_F=in_f, TILE_R=tile_r, TILE_C=64,
        BM=bm, BN=bn,
        num_warps=4, num_stages=2,
    )
    # outlier overlay: y[i, row] += val * x[i, k] for each outlier
    olk = st["ol_row_k"].long()
    olv = st["ol_row_v"].to(torch.bfloat16).float()
    rows = st["ol_rows_idx"].long()
    contrib = olv[None, :] * x[:, olk].float()        # [M, n_ol]
    y.index_add_(1, rows, contrib.to(torch.bfloat16))
    return y
