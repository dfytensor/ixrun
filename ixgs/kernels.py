"""Triton fused decode kernel for Group-Scale INT8-X (per-group scales).

Same nested-bitmap (3,5,8) layout as ixrun, plus:
  * per-element group scale lookup (`gscale_ptr`, indexed by offs // GS)
  * rank tensors (l1_rank / nl1_rank / b2_rank) precomputed via torch cumsum
  * l1/l2 streams get one trailing zero word (cross-word bit extraction reads
    word i+1 for the last elements; without the pad this reads out of bounds)
"""
from __future__ import annotations
import torch

__all__ = ["decode_weight_triton", "has_triton"]

_HAS_TRITON = False
try:
    import triton
    import triton.language as tl

    _HAS_TRITON = True
except Exception:
    pass


def has_triton() -> bool:
    return _HAS_TRITON and torch.cuda.is_available()


if _HAS_TRITON:

    @triton.jit
    def _ixgs_decode_kernel(
        out_ptr,
        b1_ptr,
        b2_ptr,
        l1_ptr,
        l2_ptr,
        l3_ptr,
        l1_rank_ptr,
        nl1_rank_ptr,
        b2_rank_ptr,
        gscale_ptr,
        N,
        BLK: tl.constexpr,
        GS: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offs = pid * BLK + tl.arange(0, BLK)
        mask = offs < N

        # nested bitmap B1: is this element level-1?
        b1_val = tl.load(b1_ptr + offs // 32, mask=mask, other=0)
        is_l1 = ((b1_val >> (offs % 32)) & 1) == 1

        l1_r = tl.load(l1_rank_ptr + offs, mask=mask, other=0)
        l1_r = tl.where(is_l1, l1_r, 0)
        bp1 = l1_r * 3
        w1i = bp1 // 32
        s1 = bp1 % 32
        w1a = tl.load(l1_ptr + w1i, mask=mask, other=0).to(tl.int64) & 0xFFFFFFFF
        c1 = (s1 + 3) > 32
        w1b = tl.where(
            c1,
            tl.load(l1_ptr + w1i + 1, mask=mask, other=0).to(tl.int64) & 0xFFFFFFFF,
            0,
        )
        l1v = (
            tl.where(c1, ((w1a >> s1) | (w1b << (32 - s1))) & 0x7, (w1a >> s1) & 0x7)
            .to(tl.int32)
            - 3
        )

        # rank among non-L1 elements
        nl1_r = tl.load(nl1_rank_ptr + offs, mask=mask, other=0)
        b2_val = tl.load(b2_ptr + nl1_r // 32, mask=mask, other=0)
        is_l2 = ((b2_val >> (nl1_r % 32)) & 1) == 1
        b2_r = tl.load(b2_rank_ptr + nl1_r, mask=mask, other=0)
        bp2 = b2_r * 5
        w2i = bp2 // 32
        s2 = bp2 % 32
        w2a = tl.load(l2_ptr + w2i, mask=mask, other=0).to(tl.int64) & 0xFFFFFFFF
        c2 = (s2 + 5) > 32
        w2b = tl.where(
            c2,
            tl.load(l2_ptr + w2i + 1, mask=mask, other=0).to(tl.int64) & 0xFFFFFFFF,
            0,
        )
        l2v = (
            tl.where(c2, ((w2a >> s2) | (w2b << (32 - s2))) & 0x1F, (w2a >> s2) & 0x1F)
            .to(tl.int32)
            - 15
        )

        # level-3: rank among (non-L1 and non-L2); stream is raw bytes (padded >=1)
        l3_pos = nl1_r - b2_r - 1
        l3_pos = tl.maximum(l3_pos, 0)
        l3v = tl.load(l3_ptr + l3_pos, mask=mask, other=0).to(tl.int32) - 127

        val = tl.where(is_l1, l1v, tl.where(is_l2, l2v, l3v))

        # per-group scale (fp32 buffer)
        sc = tl.load(gscale_ptr + offs // GS, mask=mask, other=0).to(tl.float32)
        tl.store(out_ptr + offs, (val.to(tl.float32) * sc).to(tl.bfloat16), mask=mask)

    def _compute_ranks(b1: torch.Tensor, b2: torch.Tensor, N: int, n_non: int, device):
        idx = torch.arange(N, device=device, dtype=torch.long)
        b1b = (b1[idx // 32] >> (idx % 32)) & 1
        t1 = b1b.cumsum(0).to(torch.int32) - 1
        t2 = (1 - b1b).cumsum(0).to(torch.int32) - 1
        if n_non > 0:
            idx2 = torch.arange(n_non, device=device, dtype=torch.long)
            b2b = (b2[idx2 // 32] >> (idx2 % 32)) & 1
            t3 = b2b.cumsum(0).to(torch.int32) - 1
        else:
            t3 = torch.zeros(1, dtype=torch.int32, device=device)
        return t1, t2, t3

    @torch.no_grad()
    def decode_weight_triton(packed: dict, device=None, dtype=torch.bfloat16) -> torch.Tensor:
        """Fused Triton decode. Caller must ensure CUDA is available."""
        assert has_triton(), "triton + cuda required"
        device = device if device is not None else torch.device("cuda")
        N = packed["N"]
        n1, n2, n3 = packed["counts"]
        n_non = n2 + n3
        GS = packed["group_size"]

        b1 = packed["b1"].to(device)
        b2 = packed["b2"].to(device)
        # trailing zero word: last value's cross-word extract reads word+1
        l1 = torch.cat([packed["l1"].to(device), torch.zeros(1, dtype=torch.int32, device=device)])
        l2 = torch.cat([packed["l2"].to(device), torch.zeros(1, dtype=torch.int32, device=device)])
        l3 = packed["l3"].to(device)  # >=1 element guaranteed by quantizer
        gscales = packed["group_scales"].to(device).float()  # fp16 -> fp32

        t1, t2, t3 = _compute_ranks(b1, b2, N, n_non, device)

        out = torch.empty(N, dtype=torch.bfloat16, device=device)
        BLK = 4096
        _ixgs_decode_kernel[(triton.cdiv(N, BLK),)](
            out, b1, b2, l1, l2, l3, t1, t2, t3, gscales, N, BLK=BLK, GS=GS
        )
        return out.view(packed["out_f"], packed["in_f"]).to(dtype)
