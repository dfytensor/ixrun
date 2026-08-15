"""Fused decode+GEMV Triton kernel for single-token decode steps.

During generation (KV-cached, 1 new token per forward) every Linear call is a
GEMV: y[n] = sum_k x[k] * W[n,k]. In streaming mode the classic path pays
  read packed (~5.5 bit/elem) + write bf16 W (16 bit) + read bf16 W (16 bit)
per token. The fused kernel reads ONLY the packed streams and accumulates in
registers — the bf16 weight never exists in memory.

Each program walks ONE output row sequentially over k. The value streams are
GLOBAL (indexed across the whole flattened weight), so the rank counters must
start from the GLOBAL prefix counts at the row boundary — supplied by two tiny
per-row prefix arrays (out_f int32 each, ~100KB max) precomputed once at
deploy. Within the row, counters advance in scalar registers; only two
in-tile cumsums per 512-element tile remain.

Layout constraint: in_f % 512 == 0 (holds for MiniCPM5 1536/4608 and
Qwen3.8 5120/17408 shapes).
"""
from __future__ import annotations
import torch

__all__ = ["fused_gemv", "compute_row_prefixes", "FUSED_TILE"]

FUSED_TILE = 512  # k-elements per inner tile; in_f must be a multiple

try:
    import triton
    import triton.language as tl

    _HAS_TRITON = True
except Exception:
    _HAS_TRITON = False


def compute_row_prefixes(packed: dict, chunk: int | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-row (optionally per-row-chunk) global stream prefixes.

    With chunk=None: (#L1, #L2) strictly before each row — arrays [out_f].
    With chunk=C (C a multiple of 512 dividing in_f): prefixes at every
    (row, chunk-start) boundary — arrays [out_f * n_chunks], enabling the
    split-K GEMV (each program resumes the sequential rank walk mid-row).
    """
    from .bitpack import unpack_bits_stream

    N, out_f, in_f = packed["N"], packed["out_f"], packed["in_f"]
    b1 = unpack_bits_stream(packed["bitmaps"][0], N, 1).view(out_f, in_f)
    n_non = packed["counts"][1] + packed["counts"][2]
    b2 = (
        unpack_bits_stream(packed["bitmaps"][1], n_non, 1)
        if n_non > 0
        else torch.zeros(0, dtype=torch.int32)
    )
    l2_global = torch.zeros(N, dtype=torch.int32)
    if n_non > 0:
        flat_nonl1 = (1 - b1).view(-1).bool()
        l2_global[flat_nonl1] = b2
    l1_flag = b1.view(-1).to(torch.int32)
    l2_flag = l2_global.view(-1)

    if chunk is None:
        l1_prefix = torch.zeros(out_f, dtype=torch.int32)
        l2_prefix = torch.zeros(out_f, dtype=torch.int32)
        l1_prefix[1:] = l1_flag.view(out_f, in_f).sum(dim=1)[:-1].cumsum(0)
        l2_prefix[1:] = l2_flag.view(out_f, in_f).sum(dim=1)[:-1].cumsum(0)
        return l1_prefix, l2_prefix

    # per (row, chunk) boundaries: global flat positions p = r*in_f + c*chunk
    n_chunks = in_f // chunk
    # cumulative counts at all positions p (exclusive) via cumsum reshaping
    l1_cum = l1_flag.cumsum(0)  # inclusive
    l2_cum = l2_flag.cumsum(0)
    rows = torch.arange(out_f).unsqueeze(1)          # [out_f, 1]
    chs = (torch.arange(n_chunks) * chunk)           # [n_chunks]
    pos = (rows * in_f + chs).view(-1)               # [out_f*n_chunks]
    # exclusive count at pos: cum[pos-1] (0 for pos==0)
    l1_prefix = torch.where(pos > 0, l1_cum[pos - 1], torch.zeros_like(pos))
    l2_prefix = torch.where(pos > 0, l2_cum[pos - 1], torch.zeros_like(pos))
    return l1_prefix.to(torch.int32), l2_prefix.to(torch.int32)


if _HAS_TRITON:

    @triton.jit
    def _ix_gemv_kernel(
        x_ptr,               # [in_f] bf16 activations (single token)
        y_ptr,               # [out_f] bf16 output
        b1_ptr, b2_ptr,      # nested bitmaps (int32 words)
        l1_ptr, l2_ptr,      # packed 3/5-bit streams (int32 words)
        l3_ptr,              # raw uint8 stream
        q1_ptr, q2_ptr,      # [out_f] per-row global L1/L2 prefix counts
        scale_ptr,
        IN_F: tl.constexpr,
        OUT_F: tl.constexpr,
        BK: tl.constexpr,    # inner tile (=512)
    ):
        n = tl.program_id(0)          # one output row per program
        row0 = n * IN_F               # flat index of W[n, 0]

        # GLOBAL rank counters, pre-loaded to this row's start
        l1_cnt = tl.load(q1_ptr + n)
        l2_cnt = tl.load(q2_ptr + n)
        nl1_cnt = row0 - l1_cnt       # non-L1 before row start

        acc = 0.0
        sc = tl.load(scale_ptr).to(tl.float32)

        for k0 in tl.range(0, IN_F, BK):
            offs = row0 + k0 + tl.arange(0, BK)   # contiguous — coalesced
            kidx = k0 + tl.arange(0, BK)

            # --- B1 bitmap ---
            b1w = offs // 32
            b1b = offs % 32
            b1v = tl.load(b1_ptr + b1w).to(tl.uint32)
            is_l1 = ((b1v >> b1b) & 1).to(tl.int32)

            l1_local = tl.cumsum(is_l1, axis=0) - 1
            l1_rank = l1_cnt + l1_local
            l1_rank = tl.where(is_l1 == 1, l1_rank, 0)

            # L1 value: 3-bit extract (cross-word safe)
            bp1 = l1_rank * 3
            w1i = bp1 // 32
            s1 = bp1 % 32
            w1a = tl.load(l1_ptr + w1i).to(tl.uint32)
            c1 = (s1 + 3) > 32
            w1b = tl.where(c1, tl.load(l1_ptr + w1i + 1).to(tl.uint32),
                           tl.zeros((BK,), tl.uint32))
            l1v = tl.where(c1, ((w1a >> s1) | (w1b << (32 - s1))) & 0x7,
                           (w1a >> s1) & 0x7).to(tl.int32) - 3

            # --- non-L1 rank: derived, no extra cumsum ---
            nl1_bit = 1 - is_l1
            nl1_local = tl.arange(0, BK) - l1_local - 1
            nl1_rank = nl1_cnt + nl1_local
            nl1_rank = tl.where(nl1_bit == 1, nl1_rank, 0)

            # --- B2 bitmap at nl1_rank ---
            b2w = nl1_rank // 32
            b2b = nl1_rank % 32
            b2v = tl.load(b2_ptr + b2w).to(tl.uint32)
            is_l2 = ((b2v >> b2b) & 1).to(tl.int32)
            is_l2 = tl.where(nl1_bit == 1, is_l2, 0)

            b2_local = tl.cumsum(is_l2, axis=0) - 1
            l2_rank = l2_cnt + b2_local
            l2_rank = tl.where(is_l2 == 1, l2_rank, 0)

            # L2 value: 5-bit extract
            bp2 = l2_rank * 5
            w2i = bp2 // 32
            s2 = bp2 % 32
            w2a = tl.load(l2_ptr + w2i).to(tl.uint32)
            c2 = (s2 + 5) > 32
            w2b = tl.where(c2, tl.load(l2_ptr + w2i + 1).to(tl.uint32),
                           tl.zeros((BK,), tl.uint32))
            l2v = tl.where(c2, ((w2a >> s2) | (w2b << (32 - s2))) & 0x1F,
                           (w2a >> s2) & 0x1F).to(tl.int32) - 15

            # --- L3: rank = nl1_rank - l2_count_before (derived; at an l3
            # element b2_local = cumsum-1 = l2_before - 1, hence the +1) ---
            is_l3 = nl1_bit - is_l2
            l3r = nl1_rank - (l2_cnt + b2_local + 1)
            l3r = tl.where(is_l3 == 1, l3r, 0)
            l3v = tl.load(l3_ptr + l3r).to(tl.int32) - 127

            # select int8 value, dequantize to bf16 (same rounding as the
            # standalone decode path), accumulate in fp32
            val = tl.where(is_l1 == 1, l1v, tl.where(is_l2 == 1, l2v, l3v))
            w = (val.to(tl.float32) * sc).to(tl.bfloat16)
            x = tl.load(x_ptr + kidx).to(tl.float32)
            acc += tl.sum(x * w.to(tl.float32), axis=0)

            # advance rank counters (non-L1 in tile = BK - L1 count)
            l1_cnt += tl.sum(is_l1, axis=0)
            l2_cnt += tl.sum(is_l2, axis=0)
            nl1_cnt += BK - tl.sum(is_l1, axis=0)

        tl.store(y_ptr + n, acc.to(tl.bfloat16))


def _pick_config(out_f: int, in_f: int) -> tuple[int, int]:
    """Heuristic (num_warps, BK) from measured autotune on 4090D:

      tall   (out>in, e.g. 17408x5120): (2, 512)  -> 247G elem/s
      wide   (out<in, e.g. 5120x17408): (2, 1024) ->  98G elem/s
      square (5120x5120):               (4, 512)  -> 202G elem/s
    """
    if out_f < in_f and in_f % 1024 == 0:
        return 2, 1024
    if out_f > in_f:
        return 2, 512
    return 4, 512


def _pick_split(in_f: int) -> int:
    """Split factor for wide layers (down_proj). Measured on 4090D: S=2
    (chunk = in_f/2, must be %512==0) reaches ~233G elem/s — same as tall
    layers; deeper splits are slower (prefix/atomic overhead)."""
    for s in (2, 4):
        if in_f % s == 0 and (in_f // s) % 512 == 0:
            return s
    return 1


if _HAS_TRITON:

    @triton.jit
    def _ix_gemv_split_kernel(
        x_ptr, y_ptr,          # y is fp32 [out_f], accumulated via atomic_add
        b1_ptr, b2_ptr, l1_ptr, l2_ptr, l3_ptr,
        q1_ptr, q2_ptr,        # [out_f * NSPLIT] chunk-boundary prefixes
        scale_ptr,
        IN_F: tl.constexpr,
        OUT_F: tl.constexpr,
        BK: tl.constexpr,
        CHUNK: tl.constexpr,   # in_f // NSPLIT
    ):
        """Split-K variant: grid (out_f, NSPLIT); program (n, c) walks the
        c-th k-chunk of row n and atomically accumulates its partial dot."""
        n = tl.program_id(0)
        c = tl.program_id(1)
        start = n * IN_F + c * CHUNK
        NSPLIT: tl.constexpr = IN_F // CHUNK
        l1_cnt = tl.load(q1_ptr + n * NSPLIT + c)
        l2_cnt = tl.load(q2_ptr + n * NSPLIT + c)
        nl1_cnt = start - l1_cnt

        acc = 0.0
        sc = tl.load(scale_ptr).to(tl.float32)

        for k0 in tl.range(0, CHUNK, BK):
            offs = start + k0 + tl.arange(0, BK)
            kidx = c * CHUNK + k0 + tl.arange(0, BK)

            b1w = offs // 32
            b1b = offs % 32
            b1v = tl.load(b1_ptr + b1w).to(tl.uint32)
            is_l1 = ((b1v >> b1b) & 1).to(tl.int32)

            l1_local = tl.cumsum(is_l1, axis=0) - 1
            l1_rank = l1_cnt + l1_local
            l1_rank = tl.where(is_l1 == 1, l1_rank, 0)

            bp1 = l1_rank * 3
            w1i = bp1 // 32
            s1 = bp1 % 32
            w1a = tl.load(l1_ptr + w1i).to(tl.uint32)
            c1 = (s1 + 3) > 32
            w1b = tl.where(c1, tl.load(l1_ptr + w1i + 1).to(tl.uint32),
                           tl.zeros((BK,), tl.uint32))
            l1v = tl.where(c1, ((w1a >> s1) | (w1b << (32 - s1))) & 0x7,
                           (w1a >> s1) & 0x7).to(tl.int32) - 3

            nl1_bit = 1 - is_l1
            nl1_local = tl.arange(0, BK) - l1_local - 1
            nl1_rank = nl1_cnt + nl1_local
            nl1_rank = tl.where(nl1_bit == 1, nl1_rank, 0)

            b2w = nl1_rank // 32
            b2b = nl1_rank % 32
            b2v = tl.load(b2_ptr + b2w).to(tl.uint32)
            is_l2 = ((b2v >> b2b) & 1).to(tl.int32)
            is_l2 = tl.where(nl1_bit == 1, is_l2, 0)

            b2_local = tl.cumsum(is_l2, axis=0) - 1
            l2_rank = l2_cnt + b2_local
            l2_rank = tl.where(is_l2 == 1, l2_rank, 0)

            bp2 = l2_rank * 5
            w2i = bp2 // 32
            s2 = bp2 % 32
            w2a = tl.load(l2_ptr + w2i).to(tl.uint32)
            c2 = (s2 + 5) > 32
            w2b = tl.where(c2, tl.load(l2_ptr + w2i + 1).to(tl.uint32),
                           tl.zeros((BK,), tl.uint32))
            l2v = tl.where(c2, ((w2a >> s2) | (w2b << (32 - s2))) & 0x1F,
                           (w2a >> s2) & 0x1F).to(tl.int32) - 15

            is_l3 = nl1_bit - is_l2
            l3r = nl1_rank - (l2_cnt + b2_local + 1)
            l3r = tl.where(is_l3 == 1, l3r, 0)
            l3v = tl.load(l3_ptr + l3r).to(tl.int32) - 127

            val = tl.where(is_l1 == 1, l1v, tl.where(is_l2 == 1, l2v, l3v))
            w = (val.to(tl.float32) * sc).to(tl.bfloat16)
            x = tl.load(x_ptr + kidx).to(tl.float32)
            acc += tl.sum(x * w.to(tl.float32), axis=0)

            l1_cnt += tl.sum(is_l1, axis=0)
            l2_cnt += tl.sum(is_l2, axis=0)
            nl1_cnt += BK - tl.sum(is_l1, axis=0)

        tl.atomic_add(y_ptr + n, acc)


def fused_gemv(x, b1, b2, l1, l2, l3, q1, q2, scale, out_f: int, in_f: int,
               chunk: int = 0, y32: torch.Tensor | None = None):
    """y = x @ W.T for a single token, W decoded on the fly (never materialized).

    x: [in_f] (any shape with numel == in_f will be viewed flat), bf16.
    q1, q2: per-row global L1/L2 prefix counts (int32 [out_f], GPU) when
        chunk == 0; chunk-boundary prefixes ([out_f*n_split]) when chunk > 0.
    chunk > 0 selects the split-K path accumulating into y32 (fp32) — caller
    converts to bf16 (and adds bias) afterwards.
    Returns [out_f] bf16 (single-kernel path) or the y32 buffer (split path).
    """
    # dummy 1-elem tensors guard the kernel's rank-0 loads when a level is
    # empty (e.g. no L3 outliers) — contents are never selected
    zero_i32 = None
    if l1.numel() == 0 or l2.numel() == 0 or b2.numel() == 0:
        zero_i32 = torch.zeros(1, dtype=torch.int32, device=x.device)
    zero_u8 = torch.zeros(1, dtype=torch.uint8, device=x.device) if l3.numel() == 0 else l3

    if chunk > 0:
        nsplit = in_f // chunk
        _ix_gemv_split_kernel[(out_f, nsplit)](
            x.reshape(-1),
            y32,
            b1, b2 if b2.numel() else zero_i32,
            l1 if l1.numel() else zero_i32,
            l2 if l2.numel() else zero_i32,
            zero_u8,
            q1, q2,
            scale,
            IN_F=in_f, OUT_F=out_f, BK=512, CHUNK=chunk,
            num_warps=2,
        )
        return y32

    y = torch.empty(out_f, dtype=torch.bfloat16, device=x.device)
    num_warps, bk = _pick_config(out_f, in_f)
    _ix_gemv_kernel[(out_f,)](
        x.reshape(-1),
        y,
        b1, b2 if b2.numel() else zero_i32,
        l1 if l1.numel() else zero_i32,
        l2 if l2.numel() else zero_i32,
        zero_u8,
        q1, q2,
        scale,
        IN_F=in_f,
        OUT_F=out_f,
        BK=bk,
        num_warps=num_warps,
    )
    return y
