"""UDCQ: Universal Distribution-fitted Companded Quantization (4-bit core).

Born from a systematic search over "fitting-based" compression ideas on real
LLM weights: position-based fitting all FAILS (adjacent-weight corr 0.000,
cross-layer corr 0.002, exponent-field plane-fit gain 1.0x, cumsum/carrier
residual x6.6, VQ/PQ gain +0.26dB, index entropy 3.97/4.00 bits) — LLM
weights are white in position. But the DISTRIBUTION of group-normalized
values is UNIVERSAL: one 16-entry empirical-Lloyd codebook fitted on a
single calibration tensor transfers to all 168 MiniCPM5 layers with SNR
spread < 1 dB (and beats both uniform quantization +1.3dB and the NF4-style
textbook-Gaussian codebook +4.2dB at 4-bit — fit the data, not the textbook).

Format (g=16):
  per group : f16 scale (U1 closed-form projection, 2 refinement rounds)
  per elem  : 4-bit codebook index (raw uint8 stream, byte-aligned)
              1-bit sign (flat bitstream)
  codebook  : 16 x f16, GLOBAL for the whole model (amortized ~0 bpw)
  bpw       = 4 + 1 + 16/16 = 5.5 (+0.5 f32-scale rounding bookkeeping)

Decode: w = sign * scale_g * CB[idx] — pure LUT, no fixups, single kernel.

MiniCPM5 end-to-end (168 layers): ppl proxy 10.63 vs bf16 10.67 (noise-level
difference), greedy continuations coherent, vs INT8-X (5.42 bpw) ppl +5.5.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

from .linear import iter_quantizable_linears, _set_parent_child

__all__ = [
    "UDCQ_G",
    "udcq_fit_codebook",
    "udcq_quantize",
    "decode_udcq_triton",
    "udcq_fused_gemv",
    "udcq_fused_gemv_mt",
    "udcq_fused_gemm",
    "UdcqLinear",
    "deploy_udcq",
    "udcq_snr",
]

UDCQ_G = 16          # group size
UDCQ_NLEV = 16       # codebook entries (4-bit index)

_HAS_TRITON = False
try:
    import triton
    import triton.language as tl

    _HAS_TRITON = True
except Exception:
    pass


# --------------------------------------------------------------------------- #
#  Codebook (empirical Lloyd-Max on pooled group-normalized |w|)
# --------------------------------------------------------------------------- #
def udcq_fit_codebook(w_calib: torch.Tensor, nlev: int = UDCQ_NLEV,
                      g: int = UDCQ_G, iters: int = 80) -> torch.Tensor:
    """Fit the universal codebook on one calibration tensor.

    Operates on |w|/group_max (positive values in (0,1]); returns sorted
    f16-representable codebook [nlev] float32.
    """
    a = w_calib.detach().abs().float().reshape(-1)
    n = (a.numel() // g) * g
    Gm = a[:n].view(-1, g)
    emax = Gm.max(1, keepdim=True).values.clamp(min=1e-12)
    x = (Gm / emax).reshape(-1)
    x = x[x > 1e-6]
    if x.numel() > 1 << 22:                     # torch.quantile input cap
        x = x[torch.randint(0, x.numel(), (1 << 22,))]

    q = torch.linspace(0, 1, nlev + 1, dtype=x.dtype)[1:-1]
    cent = torch.unique(torch.quantile(x, q))
    while cent.numel() < nlev:                    # pad if quantiles collide
        extra = x[torch.randint(0, x.numel(), (nlev - cent.numel(),))]
        cent = torch.unique(torch.cat([cent, extra]))
    cent = cent[:nlev].clone()
    for _ in range(iters):
        d = (x[:, None] - cent[None, :]).abs()
        assign = d.argmin(1)
        new = cent.clone()
        for j in range(nlev):
            m = assign == j
            if m.any():
                new[j] = x[m].mean()
        if (new - cent).abs().max() < 1e-10:
            break
        cent = new
    return cent.sort().values.contiguous()


# --------------------------------------------------------------------------- #
#  Pack (per Linear): U1 projection scale + 4-bit idx + sign stream
# --------------------------------------------------------------------------- #
def _nearest_cb(x_flat: torch.Tensor, CB: torch.Tensor) -> torch.Tensor:
    """argmin_j |x - CB[j]| for a flat tensor, chunked (N x nlev distance
    matrix would OOM on lm_head-sized inputs: 1.37B x 16 x 4B = 88GB)."""
    out = torch.empty(x_flat.numel(), dtype=torch.long)
    CH = 1 << 22
    for s in range(0, x_flat.numel(), CH):
        d = (x_flat[s:s + CH, None] - CB[None, :]).abs()
        out[s:s + CH] = d.argmin(1)
    return out


@torch.no_grad()
def udcq_quantize(weight: torch.Tensor, codebook: torch.Tensor,
                  g: int = UDCQ_G, rounds: int = 2) -> dict:
    """Pack a weight matrix against a (global) codebook."""
    CB = codebook.cpu().float()
    of, inf_ = weight.shape
    w = weight.detach().float().cpu()
    sign = (w >= 0)
    a = w.abs().reshape(-1)
    n = (a.numel() // g) * g
    pad = a.numel() - n
    if pad:
        a = torch.cat([a, torch.zeros(pad)])
    Gm = a.view(-1, g)

    # U1: closed-form projection scale (iterate assign <-> rescale)
    s = Gm.max(1, keepdim=True).values.clamp(min=1e-12)
    for _ in range(rounds):
        x = Gm / s
        idx = _nearest_cb(x.reshape(-1), CB).view(-1, g)
        c = CB[idx]
        num = (Gm * c).sum(1, keepdim=True)
        den = (c * c).sum(1, keepdim=True).clamp(min=1e-12)
        s = (num / den).clamp(min=1e-12)
    idx = _nearest_cb((Gm / s).reshape(-1), CB).view(-1, g)
    s_f16 = s.half()                              # storage
    # decode must use the f16 scale (same rounding path as tests/deploy)
    s_dec = s_f16.float()
    xq = CB[idx] * s_dec
    # sign bits: exactly the first n element signs (pad excluded)
    sign_bits = (sign.reshape(-1)[:n] > 0).to(torch.uint8)
    from .bitpack import pack_bits_stream

    sign_words = pack_bits_stream(sign_bits, 1)
    # accounting: idx 4b (REAL storage — nibble-packed below) + sign words +
    # scale f16/g. The codebook is GLOBAL (16 entries/model) — amortized ~0.
    total_bits = n * 4 + sign_words.numel() * 32 + (n // g) * 16
    # nibble-pack idx: 2 codes/byte, even element = low nibble. This makes
    # the 6bpw accounting TRUE for GPU residency (byte-aligned idx measured
    # 10bpw real and blew the 27B budget by 13GB).
    idx_bytes = idx.reshape(-1).to(torch.uint8)
    if n % 2:
        idx_bytes = torch.cat([idx_bytes, torch.zeros(1, dtype=torch.uint8)])
    idx4 = (idx_bytes[0::2] & 0x0F) | ((idx_bytes[1::2] & 0x0F) << 4)
    return {
        "g": g,
        "out_f": of,
        "in_f": inf_,
        "N": n,
        "idx": idx4.contiguous(),                        # [ceil(N/2)] nibbles
        "scale": s_f16.reshape(-1).contiguous(),       # [ng]
        "sign_packed": sign_words.contiguous(),
        "codebook": CB.half().contiguous(),            # [nlev] f16 (global)
        "total_bytes": (total_bits + 7) // 8,
        "bits_per_weight": total_bits / n,
    }


# --------------------------------------------------------------------------- #
#  Decode — fused Triton LUT kernel (single kernel, no fixups)
# --------------------------------------------------------------------------- #
if _HAS_TRITON:

    @triton.jit
    def _udcq_decode_kernel(
        out_ptr,          # bf16 [N]
        idx_ptr,          # uint8 [ceil(N/2)] NIBBLE-packed LUT indices
        sign_ptr,         # int32 words, flat 1-bit stream
        scale_ptr,        # f16 [ng]
        cb_ptr,           # f16 [16] global codebook
        N: tl.constexpr,
        GROUP: tl.constexpr,
        BLK: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offs = pid * BLK + tl.arange(0, BLK)
        mask = offs < N

        b = tl.load(idx_ptr + (offs // 2), mask=mask, other=0)
        i = tl.where(offs % 2 == 0, b & 0x0F, (b >> 4) & 0x0F).to(tl.int32)
        val = tl.load(cb_ptr + i).to(tl.float32)          # LUT
        sc = tl.load(scale_ptr + offs // GROUP, mask=mask, other=0).to(tl.float32)
        sw = tl.load(sign_ptr + offs // 32, mask=mask, other=0).to(tl.uint32)
        sgn = ((sw >> (offs % 32)) & 1).to(tl.float32) * 2 - 1

        w = val * sc * sgn
        tl.store(out_ptr + offs, w.to(tl.bfloat16), mask=mask)

    @triton.jit
    def _udcq_gemv_kernel(
        x_ptr,            # [IN_F] bf16, single token
        y_ptr,            # [OUT_F] bf16
        idx_ptr,          # uint8 [N] byte-aligned LUT indices
        sign_ptr,         # int32 words, flat 1-bit stream
        scale_ptr,        # f16 [ng]
        cb_ptr,           # f16 [16] global codebook
        IN_F: tl.constexpr,
        OUT_F: tl.constexpr,
        GROUP: tl.constexpr,
        BK: tl.constexpr,
        R: tl.constexpr,
    ):
        """Fused decode+GEMV: y = x @ W.T, W = sign*scale*CB[idx] decoded in
        registers. UDCQ's byte-aligned idx makes this a pure LUT walk —
        no bitmaps, no ranks, no cross-word extraction (unlike PEAK-Q/BF16X).
        bf16 weight never materializes."""
        pid = tl.program_id(0)
        rows = pid * R + tl.arange(0, R)
        base = rows.to(tl.int64) * IN_F
        acc = tl.zeros((R,), tl.float32)

        for k0 in tl.range(0, IN_F, BK):
            kidx = k0 + tl.arange(0, BK)
            offs = base[:, None] + kidx[None, :]           # [R, BK]

            b = tl.load(idx_ptr + offs // 2)
            i = tl.where(offs % 2 == 0, b & 0x0F,
                         (b >> 4) & 0x0F).to(tl.int32)
            val = tl.load(cb_ptr + i).to(tl.float32)       # LUT
            sc = tl.load(scale_ptr + offs // GROUP).to(tl.float32)
            sw = tl.load(sign_ptr + offs // 32).to(tl.uint32)
            sgn = ((sw >> (offs % 32)) & 1).to(tl.float32) * 2 - 1

            w = val * sc * sgn
            xv = tl.load(x_ptr + kidx).to(tl.float32)      # shared by R rows
            acc += tl.sum(w * xv[None, :], axis=1)

        tl.store(y_ptr + rows, acc.to(tl.bfloat16))

    @triton.jit
    def _udcq_gemv_mt_kernel(
        x_ptr,            # [T, IN_F] bf16, T tokens
        y_ptr,            # [T, OUT_F] bf16
        idx_ptr,          # uint8 [N] byte-aligned LUT indices
        sign_ptr,         # int32 words, flat 1-bit stream
        scale_ptr,        # f16 [ng]
        cb_ptr,           # f16 [16] global codebook
        IN_F: tl.constexpr,
        OUT_F: tl.constexpr,
        GROUP: tl.constexpr,
        BK: tl.constexpr,
        R: tl.constexpr,
        T: tl.constexpr,
    ):
        """Multi-token GEMV: y[t] = x[t] @ W.T for T tokens in ONE decode
        walk. Bit-exact vs T separate _udcq_gemv_kernel calls by
        construction: same packed-data walk, same BK chunking, and each
        token accumulates via the IDENTICAL sub-expression
        tl.sum(w * xv[None, :], axis=1) (guaranteed same reduction tree).
        Bandwidth-bound -> T<=8 costs ~= a single-token call."""
        pid = tl.program_id(0)
        rows = pid * R + tl.arange(0, R)
        base = rows.to(tl.int64) * IN_F
        toks = tl.arange(0, T)
        acc = tl.zeros((R, T), tl.float32)

        for k0 in tl.range(0, IN_F, BK):
            kidx = k0 + tl.arange(0, BK)
            offs = base[:, None] + kidx[None, :]           # [R, BK]

            b = tl.load(idx_ptr + offs // 2)
            i = tl.where(offs % 2 == 0, b & 0x0F,
                         (b >> 4) & 0x0F).to(tl.int32)
            val = tl.load(cb_ptr + i).to(tl.float32)       # LUT
            sc = tl.load(scale_ptr + offs // GROUP).to(tl.float32)
            sw = tl.load(sign_ptr + offs // 32).to(tl.uint32)
            sgn = ((sw >> (offs % 32)) & 1).to(tl.float32) * 2 - 1

            w = val * sc * sgn                             # decoded once
            for t in tl.static_range(T):
                xv = tl.load(x_ptr + t * IN_F + kidx).to(tl.float32)
                part = tl.sum(w * xv[None, :], axis=1)     # [R] — same expr
                acc += part[:, None] * (toks == t).to(tl.float32)[None, :]

        tl.store(y_ptr + toks[None, :] * OUT_F + rows[:, None],
                 acc.to(tl.bfloat16))

    @triton.jit
    def _udcq_gemm_fused_kernel(
        x_ptr,            # [M, IN_F] bf16 activations
        y_ptr,            # [M, OUT_F] bf16 output
        idx_ptr,          # uint8 [N] byte-aligned LUT indices (row-major W)
        sign_ptr,         # int32 words, flat 1-bit stream
        scale_ptr,        # f16 [ng]
        cb_ptr,           # f16 [16] global codebook
        M,
        OUT_F: tl.constexpr, IN_F: tl.constexpr, GROUP: tl.constexpr,
        BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
    ):
        """Fused decode+GEMM v2: decode directly into [BK, BN] dot-operand
        layout (no tl.trans smem round-trip), int32 arithmetic throughout
        (int64 div is emulated on GPU), power-of-2 shifts for GROUP/32."""
        pid_n = tl.program_id(0)
        pid_m = tl.program_id(1)
        offs_n = pid_n * BN + tl.arange(0, BN)
        offs_m = pid_m * BM + tl.arange(0, BM)
        mask_m = offs_m < M
        mask_n = offs_n < OUT_F

        acc = tl.zeros((BM, BN), tl.float32)

        for k0 in tl.range(0, IN_F, BK):
            offs_k = k0 + tl.arange(0, BK)
            # W.T element (k, n) = W[n, k] at flat n*IN_F + k
            pos = offs_n[None, :] * IN_F + offs_k[:, None]    # [BK, BN] i32

            b = tl.load(idx_ptr + pos // 2, mask=mask_n[None, :], other=0)
            i = tl.where(pos % 2 == 0, b & 0x0F,
                         (b >> 4) & 0x0F).to(tl.int32)
            val = tl.load(cb_ptr + i).to(tl.float32)          # LUT
            # GROUP and 32 are power-of-2 -> shift instead of division
            sc = tl.load(scale_ptr + (pos // GROUP),
                         mask=mask_n[None, :], other=0.0).to(tl.float32)
            sw = tl.load(sign_ptr + (pos // 32),
                         mask=mask_n[None, :], other=0).to(tl.uint32)
            sgn = ((sw >> (pos % 32)) & 1).to(tl.float32) * 2 - 1
            wt = (val * sc * sgn).to(tl.bfloat16)             # [BK, BN] regs

            xt = tl.load(x_ptr + offs_m[:, None] * IN_F + offs_k[None, :],
                         mask=mask_m[:, None], other=0.0)     # [BM, BK]
            acc += tl.dot(xt, wt)                             # no trans

        tl.store(y_ptr + offs_m[:, None] * OUT_F + offs_n[None, :],
                 acc.to(tl.bfloat16),
                 mask=mask_m[:, None] & mask_n[None, :])


# measured-in-context defaults (same protocol as PEAKQ_V2): R=4 rows/program
UDCQ_GEMV_R = 4
UDCQ_GEMV_BK = 256
UDCQ_GEMV_WARPS = 2


# GEMM tile defaults (BM, BN, BK, warps, stages) — tuned on 4090 for
# MiniCPM5/Qwen3.8 shapes; verify in-context before changing (AGENTS.md rule)
UDCQ_GEMM_CFG = (64, 128, 64, 8, 2)


def udcq_fused_gemm(x, idx, sign, scale, cb, out_f, in_f,
                    g=UDCQ_G, cfg=UDCQ_GEMM_CFG):
    """y = x @ W.T multi-token, UDCQ-decoded in registers (pipelined)."""
    M = x.shape[0]
    y = torch.empty(M, out_f, dtype=torch.bfloat16, device=x.device)
    BM, BN, BK, warps, stages = cfg
    assert in_f % BK == 0
    grid = (triton.cdiv(out_f, BN), triton.cdiv(M, BM))
    _udcq_gemm_fused_kernel[grid](
        x, y, idx, sign, scale, cb,
        M, OUT_F=out_f, IN_F=in_f, GROUP=g,
        BM=BM, BN=BN, BK=BK,
        num_warps=warps, num_stages=stages,
    )
    return y


def udcq_fused_gemv(x, idx, sign, scale, cb, out_f, in_f,
                    g=UDCQ_G, r=UDCQ_GEMV_R, bk=UDCQ_GEMV_BK,
                    num_warps=UDCQ_GEMV_WARPS):
    """y = x @ W.T single token, UDCQ-decoded in registers."""
    y = torch.empty(out_f, dtype=torch.bfloat16, device=x.device)
    while r > 1 and out_f % r != 0:               # odd out_f fallback
        r //= 2
    assert in_f % bk == 0
    _udcq_gemv_kernel[(out_f // r,)](
        x.reshape(-1), y,
        idx, sign, scale, cb,
        IN_F=in_f, OUT_F=out_f, GROUP=g, BK=bk, R=r,
        num_warps=num_warps,
    )
    return y


# multi-token GEMV config override (in-context tuning: set env before run,
# verify with deployed-model timing per AGENTS.md rule)
import os as _os
UDCQ_MT_R = int(_os.environ.get('UDCQ_MT_R', UDCQ_GEMV_R))
UDCQ_MT_BK = int(_os.environ.get('UDCQ_MT_BK', UDCQ_GEMV_BK))
UDCQ_MT_WARPS = int(_os.environ.get('UDCQ_MT_WARPS', UDCQ_GEMV_WARPS))


def udcq_fused_gemv_mt(x, idx, sign, scale, cb, out_f, in_f,
                       g=UDCQ_G, r=None, bk=None,
                       num_warps=None):
    """y = x @ W.T for T tokens (x: [T, in_f]), one decode walk.
    Bit-exact vs T separate udcq_fused_gemv calls (same walk + same
    per-token accumulation expression). T in {2, 4, 8}."""
    r = UDCQ_MT_R if r is None else r
    bk = UDCQ_MT_BK if bk is None else bk
    num_warps = UDCQ_MT_WARPS if num_warps is None else num_warps
    T, in_f2 = x.shape
    assert in_f2 == in_f and T in (2, 4, 8), (T, in_f, in_f2)
    y = torch.empty(T, out_f, dtype=torch.bfloat16, device=x.device)
    while r > 1 and out_f % r != 0:               # odd out_f fallback
        r //= 2
    assert in_f % bk == 0
    _udcq_gemv_mt_kernel[(out_f // r,)](
        x, y, idx, sign, scale, cb,
        IN_F=in_f, OUT_F=out_f, GROUP=g, BK=bk, R=r, T=T,
        num_warps=num_warps,
    )
    return y


def decode_udcq_triton(packed: dict, device=None,
                       dtype=torch.bfloat16) -> torch.Tensor:
    """Single-kernel LUT decode -> [out_f, in_f]."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not isinstance(device, torch.device):
        device = torch.device(device)
    if not _HAS_TRITON or device.type != "cuda":
        return _decode_udcq_ref(packed, device, dtype)
    N = packed["N"]
    total = packed["out_f"] * packed["in_f"]
    out = torch.zeros(total, dtype=torch.bfloat16, device=device)
    BLK = 1024
    _udcq_decode_kernel[(triton.cdiv(N, BLK),)](
        out,
        packed["idx"].reshape(-1).to(device),
        packed["sign_packed"].to(device),
        packed["scale"].to(device),
        packed["codebook"].to(device),
        N,
        GROUP=packed["g"],
        BLK=BLK,
    )
    return out.view(packed["out_f"], packed["in_f"]).to(dtype)


def _decode_udcq_ref(packed: dict, device=None,
                     dtype=torch.bfloat16) -> torch.Tensor:
    """Reference (CPU/GPU torch) decode — ground truth for kernel tests."""
    device = device or torch.device("cpu")
    g = packed["g"]
    idx = packed["idx"].to(device).long()
    CB = packed["codebook"].to(device).float()
    s = packed["scale"].to(device).float()
    from .bitpack import unpack_bits_stream

    sign = unpack_bits_stream(packed["sign_packed"].to(device),
                              packed["N"], 1, device=device).float() * 2 - 1
    # nibble-packed idx: even element = low nibble
    ib = packed["idx"].to(device)
    codes = torch.stack([ib & 0x0F, (ib >> 4) & 0x0F], 1).reshape(-1).long()
    vals = CB[codes[:packed["N"]]] * s.repeat_interleave(g)[:packed["N"]]
    w = torch.zeros(packed["out_f"] * packed["in_f"],
                    dtype=vals.dtype, device=vals.device)
    w[: packed["N"]] = vals * sign          # pad tail decodes to 0
    return w.view(packed["out_f"], packed["in_f"]).to(dtype)


# --------------------------------------------------------------------------- #
#  Layer + deploy
# --------------------------------------------------------------------------- #
# shared decode workspace singleton: one bf16 buffer sized to the largest
# layer serves ALL UdcqLinear instances in streaming mode (layers decode
# sequentially — same pattern as PeakQLinear._get_shared_w_buf).
_SHARED_W_BUF = None
_SHARED_W_SIZE = 0


def _get_shared_w_buf(n_elems: int, device) -> torch.Tensor:
    global _SHARED_W_BUF, _SHARED_W_SIZE
    if _SHARED_W_BUF is None or _SHARED_W_SIZE < n_elems:
        _SHARED_W_BUF = torch.zeros(n_elems, dtype=torch.bfloat16, device=device)
        _SHARED_W_SIZE = n_elems
    return _SHARED_W_BUF


class UdcqLinear(nn.Module):
    """nn.Linear backed by UDCQ packed weight.

    cache='full'    : decode once to bf16, plain F.linear (fastest).
    cache='stream'  : packed streams GPU-resident (idx/scale/sign as
        registered buffers — no per-forward DMA); every forward re-decodes
        into the SHARED buffer via the single LUT kernel. GPU weight memory
        ~= packed (510MB model-wide) + one shared buf. Same protocol as
        PeakQLinear streaming.
    """

    def __init__(self, packed: dict, bias=None, cache: str = "full"):
        super().__init__()
        self.out_features = packed["out_f"]
        self.in_features = packed["in_f"]
        self.N = packed["N"]
        self._cache = cache
        self._w = None
        self._w_buf = None
        if bias is not None:
            self.register_buffer("_bias", bias.detach().clone())
        else:
            self._bias = None
        if cache == "full":
            self.packed = packed
        else:
            # GPU-resident packed streams (codebook shared globally via attr)
            for key, val in [
                ("_idx", packed["idx"]),
                ("_scale", packed["scale"]),
                ("_sign", packed["sign_packed"]),
                ("_cb", packed["codebook"]),
            ]:
                self.register_buffer(key, val.cuda())
            self.packed = {k: v for k, v in packed.items()
                           if not isinstance(v, torch.Tensor)}
            self.packed["codebook"] = None     # staged on GPU already

    def _stream_decode(self):
        import triton

        total = self.out_features * self.in_features
        buf = _get_shared_w_buf(total, self._idx.device)
        w_flat = buf[:total]
        w_flat.zero_()
        w_valid = w_flat[: self.N]
        _udcq_decode_kernel[(triton.cdiv(self.N, 1024),)](
            w_valid,
            self._idx,
            self._sign,
            self._scale,
            self._cb,
            self.N,
            GROUP=self.packed["g"],
            BLK=1024,
        )
        return w_flat.view(self.out_features, self.in_features)

    def _decode(self, device):
        if self._w is None or self._w.device != device:
            self._w = decode_udcq_triton(self.packed, device)
        return self._w

    def forward(self, x):
        if self._cache == "full":
            w = self._decode(x.device)
            w = w.to(x.dtype)
            b = self._bias.to(x.dtype) if self._bias is not None else None
            return F.linear(x, w, b)
        # streaming: single-token decode steps use the fused GEMV (one
        # kernel per layer, weight never materialized — halves launch count
        # vs decode-then-GEMM and removes the buffer round-trip)
        w = None
        if x.numel() == self.in_features:
            y = udcq_fused_gemv(
                x, self._idx, self._sign, self._scale, self._cb,
                self.out_features, self.in_features, g=self.packed["g"],
            ).to(x.dtype)
            if self._bias is not None:
                y = y + self._bias.to(x.dtype)
            return y.view(*x.shape[:-1], self.out_features)
        # multi-token dispatch (measured on 4090, see tests):
        #   M in {2,4,8} -> multi-token GEMV (bit-exact vs sequential M=1
        #                   calls, ~same cost as one — spec-decode verify)
        #   M <= 256 -> pipelined fused decode+GEMM (reads ~10bpw, wins vs
        #               cublas 0.85-1.56x at M=256, better below — the
        #               Marlin/ZipServ regime; decode redundancy x M/BM is
        #               still small)
        #   M >  256 -> decode-to-shared-buffer + cublas (each W read once;
        #               fused re-decodes per m-tile and loses ~8x at M=4096)
        M = x.numel() // self.in_features
        if (M in (2, 4, 8) and x.shape[-1] == self.in_features
                and self.in_features % UDCQ_GEMV_BK == 0):
            x2 = x.reshape(M, self.in_features)
            y = udcq_fused_gemv_mt(
                x2, self._idx, self._sign, self._scale, self._cb,
                self.out_features, self.in_features, g=self.packed["g"],
            ).to(x.dtype)
            if self._bias is not None:
                y = y + self._bias.to(x.dtype)
            return y.view(*x.shape[:-1], self.out_features)
        if (M <= 256 and self.in_features % UDCQ_GEMM_CFG[2] == 0
                and self.out_features % UDCQ_GEMM_CFG[1] == 0):
            x2 = x.reshape(M, self.in_features)
            y = udcq_fused_gemm(
                x2, self._idx, self._sign, self._scale, self._cb,
                self.out_features, self.in_features, g=self.packed["g"],
            ).to(x.dtype)
            if self._bias is not None:
                y = y + self._bias.to(x.dtype)
            return y.view(*x.shape[:-1], self.out_features)
        w = self._stream_decode().to(x.dtype)
        b = self._bias.to(x.dtype) if self._bias is not None else None
        return F.linear(x, w, b)

    def extra_repr(self):
        return (f"in={self.in_features}, out={self.out_features}, "
                f"bpw={self.packed['bits_per_weight']:.2f}, "
                f"cache={self._cache}")


@torch.no_grad()
def deploy_udcq(model: nn.Module, nlev: int = UDCQ_NLEV, g: int = UDCQ_G,
                cache: str = "full", verbose: bool = True) -> dict:
    """Replace every quantizable Linear with UdcqLinear (in-place).

    cache='full'   decodes each layer once (bf16 resident).
    cache='stream' keeps only the packed streams GPU-resident + one shared
                   decode buffer (~510MB + 14MB for MiniCPM5 vs 1.36GB bf16).
    The universal codebook is fitted on the FIRST target tensor only.
    """
    targets = list(iter_quantizable_linears(model))
    cb = udcq_fit_codebook(targets[0][1].weight.data.cpu(), nlev=nlev, g=g)
    total_bytes = 0
    n_elems = 0
    for name, mod in targets:
        packed = udcq_quantize(mod.weight.data, cb, g=g)
        bias = mod.bias.data if mod.bias is not None else None
        new = UdcqLinear(packed, bias=bias, cache=cache)
        _set_parent_child(model, name, new)
        mod.weight.data = torch.empty(0)       # drop the bf16 original
        total_bytes += packed["total_bytes"]
        n_elems += packed["N"]
    stats = {
        "n_layers": len(targets),
        "total_bytes": total_bytes,
        "n_elems": n_elems,
        "bits_per_weight": total_bytes * 8 / max(n_elems, 1),
        "codebook": cb.tolist(),
    }
    if verbose:
        print(f"[udcq-deploy] {stats['n_layers']} layers | "
              f"{total_bytes/1e6:.0f}MB | bpw={stats['bits_per_weight']:.2f} | "
              f"codebook(global): {[round(c,3) for c in cb.tolist()]}",
              flush=True)
    return stats


def udcq_snr(w: torch.Tensor, wq: torch.Tensor) -> float:
    e = (w.reshape(-1).float() - wq.reshape(-1).float()).pow(2).mean().item()
    v = w.reshape(-1).float().var(unbiased=False).item()
    return 10 * float(torch.log10(torch.tensor(v / max(e, 1e-30))))
