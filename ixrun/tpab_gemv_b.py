"""b-constant specialized TPAB GEMV kernels.

At realistic SNR targets the encoder picks b=6 (26dB+) or b=5/6 mix
(24dB) for heavy-tailed layers — the generic kernel pays a runtime `b`
everywhere: variable shift math, cross-word branches, exp2 mask. A
compile-time-B specialization removes all of it:

  - mask = (1<<B)-1, qmax = 2^(B-1)-1 are constants
  - shift = (L*B) & 31 with B constant -> strength-reduced
  - cross branch stays but with constant B the compiler can specialize

The host dispatches per tile-group: single-B layers (the common case at
26dB+) launch ONE specialized kernel.
"""
from __future__ import annotations
import torch

__all__ = ["fused_gemv_tpab_b"]

try:
    import triton
    import triton.language as tl

    _HAS = True
except Exception:
    _HAS = False

if _HAS:

    @triton.jit
    def _tpab_gemv_b4_kernel(
        x_ptr, y_ptr,
        body_ptr, scales_ptr, goff_ptr,
        olk_ptr, olv_ptr, oloff_ptr,
        T_C: tl.constexpr, IN_F: tl.constexpr,
        TILE_R: tl.constexpr, TILE_C: tl.constexpr,
        GBASE: tl.constexpr,
    ):
        n = tl.program_id(0)
        row_tile = n // TILE_R
        in_row = n % TILE_R
        acc = 0.0
        for kc in tl.range(0, T_C):
            t = row_tile * T_C + kc
            s = tl.load(scales_ptr + t).to(tl.float32)
            goff = tl.load(goff_ptr + t)
            L = in_row * TILE_C + tl.arange(0, TILE_C)
            bitpos = GBASE + goff * 4 + L * 4
            word = bitpos >> 5
            shift = bitpos & 31
            w1 = tl.load(body_ptr + word).to(tl.uint32)
            # B=4: shift in {0,4,...,28}, shift+4 <= 32 -> NEVER crosses
            raw = (w1 >> shift) & 15
            v = raw.to(tl.int32) - 7
            k0 = kc * TILE_C + tl.arange(0, TILE_C)
            x = tl.load(x_ptr + k0).to(tl.float32)
            wq = (v.to(tl.float32) * s).to(tl.bfloat16).to(tl.float32)
            acc += tl.sum(x * wq, axis=0)
        lo = tl.load(oloff_ptr + n)
        hi = tl.load(oloff_ptr + n + 1)
        for j in tl.range(lo, hi):
            kk = tl.load(olk_ptr + j)
            vv = tl.load(olv_ptr + j)
            acc += vv.to(tl.bfloat16).to(tl.float32) * tl.load(x_ptr + kk).to(tl.float32)
        tl.store(y_ptr + n, acc.to(tl.bfloat16))

    @triton.jit
    def _tpab_gemv_b6_kernel(
        x_ptr, y_ptr,
        body_ptr, scales_ptr, goff_ptr,
        olk_ptr, olv_ptr, oloff_ptr,
        T_C: tl.constexpr, IN_F: tl.constexpr,
        TILE_R: tl.constexpr, TILE_C: tl.constexpr,
        GBASE: tl.constexpr,
    ):
        n = tl.program_id(0)
        row_tile = n // TILE_R
        in_row = n % TILE_R
        acc = 0.0
        for kc in tl.range(0, T_C):
            t = row_tile * T_C + kc
            s = tl.load(scales_ptr + t).to(tl.float32)
            goff = tl.load(goff_ptr + t)
            L = in_row * TILE_C + tl.arange(0, TILE_C)
            bitpos = GBASE + goff * 6 + L * 6
            word = bitpos >> 5
            shift = bitpos & 31
            w1 = tl.load(body_ptr + word).to(tl.uint32)
            cross = (shift + 6) > 32            # shift in {30, 31} only
            w2 = tl.where(cross, tl.load(body_ptr + word + 1).to(tl.uint32),
                          tl.zeros((TILE_C,), tl.uint32))
            raw = tl.where(cross, (w1 >> shift) | (w2 << (32 - shift)),
                           w1 >> shift)
            v = (raw & 63).to(tl.int32) - 31
            k0 = kc * TILE_C + tl.arange(0, TILE_C)
            x = tl.load(x_ptr + k0).to(tl.float32)
            wq = (v.to(tl.float32) * s).to(tl.bfloat16).to(tl.float32)
            acc += tl.sum(x * wq, axis=0)
        lo = tl.load(oloff_ptr + n)
        hi = tl.load(oloff_ptr + n + 1)
        for j in tl.range(lo, hi):
            kk = tl.load(olk_ptr + j)
            vv = tl.load(olv_ptr + j)
            acc += vv.to(tl.bfloat16).to(tl.float32) * tl.load(x_ptr + kk).to(tl.float32)
        tl.store(y_ptr + n, acc.to(tl.bfloat16))


if _HAS:

    @triton.jit
    def _tpab_gemv_b4_mr_kernel(
        x_ptr, y_ptr,
        body_ptr, scales_ptr, goff_ptr,
        olk_ptr, olv_ptr, oloff_ptr,
        T_C: tl.constexpr, IN_F: tl.constexpr,
        TILE_R: tl.constexpr, TILE_C: tl.constexpr,
        GBASE: tl.constexpr, R: tl.constexpr,
    ):
        pid = tl.program_id(0)
        n0 = pid * R
        row_tile = n0 // TILE_R
        rows_in_tile = n0 % TILE_R + tl.arange(0, R)
        acc = tl.zeros((R,), dtype=tl.float32)
        for kc in tl.range(0, T_C):
            t = row_tile * T_C + kc
            s = tl.load(scales_ptr + t).to(tl.float32)
            goff = tl.load(goff_ptr + t)
            L = rows_in_tile[:, None] * TILE_C + tl.arange(0, TILE_C)[None, :]
            bitpos = GBASE + goff * 4 + L * 4
            word = bitpos >> 5
            shift = bitpos & 31
            w1 = tl.load(body_ptr + word).to(tl.uint32)
            raw = (w1 >> shift) & 15                 # B=4 never crosses
            v = raw.to(tl.int32) - 7
            k0 = kc * TILE_C + tl.arange(0, TILE_C)
            x = tl.load(x_ptr + k0).to(tl.float32)
            wq = (v.to(tl.float32) * s).to(tl.bfloat16).to(tl.float32)
            acc += tl.sum(wq * x[None, :], axis=1)
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

    @triton.jit
    def _tpab_gemv_b6_mr_kernel(
        x_ptr, y_ptr,
        body_ptr, scales_ptr, goff_ptr,
        olk_ptr, olv_ptr, oloff_ptr,
        T_C: tl.constexpr, IN_F: tl.constexpr,
        TILE_R: tl.constexpr, TILE_C: tl.constexpr,
        GBASE: tl.constexpr, R: tl.constexpr,
    ):
        pid = tl.program_id(0)
        n0 = pid * R
        row_tile = n0 // TILE_R
        rows_in_tile = n0 % TILE_R + tl.arange(0, R)
        acc = tl.zeros((R,), dtype=tl.float32)
        for kc in tl.range(0, T_C):
            t = row_tile * T_C + kc
            s = tl.load(scales_ptr + t).to(tl.float32)
            goff = tl.load(goff_ptr + t)
            L = rows_in_tile[:, None] * TILE_C + tl.arange(0, TILE_C)[None, :]
            bitpos = GBASE + goff * 6 + L * 6
            word = bitpos >> 5
            shift = bitpos & 31
            w1 = tl.load(body_ptr + word).to(tl.uint32)
            cross = (shift + 6) > 32
            w2 = tl.where(cross, tl.load(body_ptr + word + 1).to(tl.uint32),
                          tl.zeros((R, TILE_C), tl.uint32))
            raw = tl.where(cross, (w1 >> shift) | (w2 << (32 - shift)),
                           w1 >> shift)
            v = (raw & 63).to(tl.int32) - 31
            k0 = kc * TILE_C + tl.arange(0, TILE_C)
            x = tl.load(x_ptr + k0).to(tl.float32)
            wq = (v.to(tl.float32) * s).to(tl.bfloat16).to(tl.float32)
            acc += tl.sum(wq * x[None, :], axis=1)
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


def fused_gemv_tpab_b(x, st, out_f, in_f, tile_r=64, b=6, r: int = 1):
    """Specialized single-B GEMV (optional multi-row R). `st` is the
    standard stage; single-B layers share one gbase."""
    assert _HAS
    y = torch.empty(out_f, dtype=torch.bfloat16, device=x.device)
    gbase = int(st["gbase_bit"][b].item()) if torch.is_tensor(st["gbase_bit"]) else int(st["gbase_bit"][b])
    if r > 1:
        assert tile_r % r == 0 and out_f % r == 0
        kern = _tpab_gemv_b4_mr_kernel if b == 4 else _tpab_gemv_b6_mr_kernel
        kern[(out_f // r,)](
            x.reshape(-1), y,
            st["body_g"], st["scales_g"], st["goff_g"],
            st["ol_row_k"], st["ol_row_v"], st["ol_offs"],
            T_C=in_f // 64, IN_F=in_f, TILE_R=tile_r, TILE_C=64,
            GBASE=gbase, R=r, num_warps=4,
        )
        return y
    kern = _tpab_gemv_b4_kernel if b == 4 else _tpab_gemv_b6_kernel
    kern[(out_f,)](
        x.reshape(-1), y,
        st["body_g"], st["scales_g"], st["goff_g"],
        st["ol_row_k"], st["ol_row_v"], st["ol_offs"],
        T_C=in_f // 64, IN_F=in_f, TILE_R=tile_r, TILE_C=64,
        GBASE=gbase, num_warps=2,
    )
    return y
