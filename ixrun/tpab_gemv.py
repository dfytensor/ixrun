"""TPAB fused GEMV: single-token decode step with tile-parallel decode.

y[n] = sum_k x[k] * W[n,k], W decoded on the fly from TPAB tiles. One
program per output row walks that row's T_C tiles; each tile needs only its
own 10-byte header (scale/bits/goffset) — no sequential rank state, so rows
map to programs arbitrarily (INT8-X's fused GEMV must resume a rank counter
at row start via prefix tables, and its streams cannot be mid-row split).

Outliers (~1%) are handled in-kernel via a per-row overlay list: the host
precomputes, for each row, the (k, value) corrections from the packed
outlier table (flat arrays ol_row/ol_k/ol_val sorted by row, plus row
offsets) — the kernel adds sum(val * x[k]) for that row's outliers.
"""
from __future__ import annotations
import torch

__all__ = ["fused_gemv_tpab", "prepare_gemv_stage"]

try:
    import triton
    import triton.language as tl

    _HAS = True
except Exception:
    _HAS = False


def prepare_gemv_stage(packed: dict, device="cuda", staged: dict | None = None) -> dict:
    """Stage TPAB packed data + per-row outlier lists for the GEMV kernel.

    Pass `staged` (from tpab.stage_gpu) to SHARE the already-uploaded
    bodies/metadata instead of duplicating them.
    """
    dev = torch.device(device)
    st = dict(packed)
    if staged is not None:
        st["body_g"] = staged["bodies_g"]
        st["bits_g"] = staged["bits_g"]
        st["scales_g"] = staged["scales_g"]
        st["goff_g"] = staged["goff_g"]
        st["gbase_g"] = staged["gbase_g"]
    else:
        st["body_g"] = torch.cat(packed["bodies"]).to(dev)
        st["bits_g"] = packed["bits"].to(dev)
        st["scales_g"] = packed["scales"].to(dev)
        st["goff_g"] = packed["goff"].to(dev)
        st["gbase_g"] = packed["gbase_bit"].to(dev)

    # outliers grouped by row (rect-tile aware)
    O, I = packed["shape"]
    tile_r, tile_c = packed["tile_r"], packed["tile_c"]
    ol_t = packed["ol_t"].to(dev).to(torch.int64)
    ol_l = packed["ol_l"].to(dev).to(torch.int64)
    r = ol_t // (I // tile_c) * tile_r + ol_l // tile_c
    k = (ol_t % (I // tile_c)) * tile_c + ol_l % tile_c
    v = packed["ol_val"].to(dev).float()

    order = torch.argsort(r, stable=True)
    r, k, v = r[order], k[order], v[order]
    counts = torch.bincount(r, minlength=O)
    offs = torch.zeros(O + 1, dtype=torch.int32, device=dev)
    offs[1:] = counts.cumsum(0).to(torch.int32)
    st["ol_row_k"] = k.to(torch.int32)
    st["ol_row_v"] = v
    st["ol_offs"] = offs
    st["ol_rows_idx"] = r                      # per-outlier output row (sorted)
    return st


if _HAS:

    @triton.jit
    def _tpab_gemv_kernel(
        x_ptr, y_ptr,
        body_ptr, bits_ptr, scales_ptr, goff_ptr, gbase_ptr,
        olk_ptr, olv_ptr, oloff_ptr,
        T_C: tl.constexpr,
        IN_F: tl.constexpr,
        TILE_R: tl.constexpr,
        TILE_C: tl.constexpr,
    ):
        n = tl.program_id(0)
        row_tile = n // TILE_R
        in_row = n % TILE_R

        acc = 0.0
        for kc in tl.range(0, T_C):
            t = row_tile * T_C + kc
            b = tl.load(bits_ptr + t).to(tl.int32)
            s = tl.load(scales_ptr + t).to(tl.float32)
            gbase = tl.load(gbase_ptr + b)
            goff = tl.load(goff_ptr + t).to(tl.int64)

            L = in_row * TILE_C + tl.arange(0, TILE_C)
            bitpos = gbase + (goff + L.to(tl.int64)) * b
            word = (bitpos // 32).to(tl.int32)
            shift = (bitpos % 32).to(tl.int32)

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
            # bf16-round the dequantized weight to match the reference path
            # (decode_tpab_ref materializes bf16 weights before F.linear)
            wq = (v.to(tl.float32) * s).to(tl.bfloat16).to(tl.float32)
            acc += tl.sum(x * wq, axis=0)

        # outlier overlay for this row (values already fp16-exact; round to
        # bf16 the same way the reference path does)
        lo = tl.load(oloff_ptr + n)
        hi = tl.load(oloff_ptr + n + 1)
        for j in tl.range(lo, hi):
            kk = tl.load(olk_ptr + j)
            vv = tl.load(olv_ptr + j)
            acc += vv.to(tl.bfloat16).to(tl.float32) * tl.load(x_ptr + kk).to(tl.float32)

        tl.store(y_ptr + n, acc.to(tl.bfloat16))


def fused_gemv_tpab(x: torch.Tensor, st: dict, out_f: int, in_f: int,
                    tile_r: int = 64):
    """y = x @ W.T for a single token; W TPAB-decoded in registers.

    x: [in_f] bf16 (flattened). Returns [out_f] bf16.
    st: prepare_gemv_stage(packed).
    """
    y = torch.empty(out_f, dtype=torch.bfloat16, device=x.device)
    T_C = in_f // 64
    _tpab_gemv_kernel[(out_f,)](
        x.reshape(-1), y,
        st["body_g"], st["bits_g"], st["scales_g"], st["goff_g"], st["gbase_g"],
        st["ol_row_k"], st["ol_row_v"], st["ol_offs"],
        T_C=T_C, IN_F=in_f, TILE_R=tile_r, TILE_C=64,
        num_warps=2,
    )
    return y
