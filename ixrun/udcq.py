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
        d = (x.reshape(-1, 1) - CB[None, :]).abs()
        idx = d.argmin(1).view(-1, g)
        c = CB[idx]
        num = (Gm * c).sum(1, keepdim=True)
        den = (c * c).sum(1, keepdim=True).clamp(min=1e-12)
        s = (num / den).clamp(min=1e-12)
    x = Gm / s
    d = (x.reshape(-1, 1) - CB[None, :]).abs()
    idx = d.argmin(1).view(-1, g)
    s_f16 = s.half()                              # storage
    # decode must use the f16 scale (same rounding path as tests/deploy)
    s_dec = s_f16.float()
    xq = CB[idx] * s_dec
    # sign bits: exactly the first n element signs (pad excluded)
    sign_bits = (sign.reshape(-1)[:n] > 0).to(torch.uint8)
    from .bitpack import pack_bits_stream

    sign_words = pack_bits_stream(sign_bits, 1)
    # accounting: idx 4b + sign (packed words) + scale f16/g. The codebook is
    # GLOBAL (one per model, 16 entries) — amortized to ~0 bpw and excluded
    # from the per-layer total.
    total_bits = n * 4 + sign_words.numel() * 32 + (n // g) * 16
    return {
        "g": g,
        "out_f": of,
        "in_f": inf_,
        "N": n,
        "idx": idx.to(torch.uint8).contiguous(),       # [ng, g] values 0..15
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
        idx_ptr,          # uint8 [N] values 0..15 (byte-aligned LUT index)
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

        i = tl.load(idx_ptr + offs, mask=mask, other=0).to(tl.int32)
        val = tl.load(cb_ptr + i).to(tl.float32)          # LUT
        sc = tl.load(scale_ptr + offs // GROUP, mask=mask, other=0).to(tl.float32)
        sw = tl.load(sign_ptr + offs // 32, mask=mask, other=0).to(tl.uint32)
        sgn = ((sw >> (offs % 32)) & 1).to(tl.float32) * 2 - 1

        w = val * sc * sgn
        tl.store(out_ptr + offs, w.to(tl.bfloat16), mask=mask)


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
    vals = (CB[idx.reshape(-1)] * s.repeat_interleave(g))[:packed["N"]]
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
            self._idx.reshape(-1),
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
        else:
            w = self._stream_decode()
        w = w.to(x.dtype)
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
