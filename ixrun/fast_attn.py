"""Fast S=1 decode attention for the StepGraph 1B engine.

transformers 5.15 StaticCache passes FULL-WIDTH k/v (the whole 2048-slot
cache) + a bool mask to SDPA; the cutlass mem-efficient kernel then
burns ~94us/layer regardless of position = 2.25ms/step (38%) on the 1B.
This module intercepts torch.nn.functional.scaled_dot_product_attention
and routes 4-D q_len==1 bf16 calls to a one-program-per-head Triton
kernel that reads the mask tensor directly (fixed loop over the cache
width, masked loads) - no host-side length, no data-dependent control
flow: Dynamo-traceable and CUDA-graph-capture-safe.

Enable: default in StepGraphEngine; IXRUN_FAST_ATTN=0 reverts.
"""
import os
import torch
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl
    _HAS_TRITON = True
except Exception:                                    # pragma: no cover
    _HAS_TRITON = False


if _HAS_TRITON:

    @triton.jit
    def _s1_attn_part_kernel(Q, K, V, M, PM, PL, PA,
                             sq_h,
                             sk_h, sk_l,
                             sv_h, sv_l,
                             sm_h,
                             scale,
                             HD: tl.constexpr, W: tl.constexpr,
                             CW: tl.constexpr, BLK: tl.constexpr):
        """Phase 1: grid (H, W//CW) - chunk-local online softmax partials."""
        h = tl.program_id(0)
        c = tl.program_id(1)
        offs_d = tl.arange(0, HD)
        q = tl.load(Q + h * sq_h + offs_d).to(tl.float32)

        m_i = tl.zeros([1], dtype=tl.float32) - float('inf')
        l_i = tl.zeros([1], dtype=tl.float32)
        acc = tl.zeros([HD], dtype=tl.float32)
        base = c * CW
        for start in range(base, base + CW, BLK):
            offs = start + tl.arange(0, BLK)
            keep = tl.load(M + offs).to(tl.int1)
            kt = tl.load(K + h * sk_h + offs[:, None] * sk_l
                         + offs_d[None, :],
                         mask=keep[:, None], other=0.0).to(tl.float32)
            s = tl.sum(q[None, :] * kt, axis=1) * scale
            s = tl.where(keep, s, -float('inf'))
            m_new = tl.maximum(m_i, tl.max(s, axis=0))
            # a fully-masked block keeps m_new at -inf: s - (-inf) = nan
            m_safe = tl.where(m_new == -float('inf'), 0.0, m_new)
            p = tl.exp(s - m_safe)
            alpha = tl.exp(tl.where(m_i == -float('inf'),
                                    -float('inf'), m_i - m_safe))
            l_i = l_i * alpha + tl.sum(p, axis=0)
            acc = acc * alpha
            vt = tl.load(V + h * sv_h + offs[:, None] * sv_l
                         + offs_d[None, :],
                         mask=keep[:, None], other=0.0).to(tl.float32)
            acc += tl.sum(p[:, None] * vt, axis=0)
            m_i = m_new
        tl.store(PM + h * (W // CW) + c, tl.sum(m_i, axis=0))
        tl.store(PL + h * (W // CW) + c, tl.sum(l_i, axis=0))
        tl.store(PA + (h * (W // CW) + c) * HD + offs_d, acc)

    @triton.jit
    def _s1_attn_merge_kernel(PM, PL, PA, O,
                              so_h, NS: tl.constexpr, HD: tl.constexpr):
        """Phase 2: grid (H,) - merge the chunk partials."""
        h = tl.program_id(0)
        offs_c = tl.arange(0, NS)
        pm = tl.load(PM + h * NS + offs_c)
        pl = tl.load(PL + h * NS + offs_c)
        M = tl.max(pm, axis=0)
        w = tl.exp(pm - M)
        L = tl.sum(pl * w, axis=0)
        offs_d = tl.arange(0, HD)
        acc = tl.zeros([HD], dtype=tl.float32)
        for c in range(0, NS):
            wc = tl.exp(tl.load(PM + h * NS + c) - M)
            acc += tl.load(PA + (h * NS + c) * HD + offs_d) * wc
        acc = acc / L
        tl.store(O + h * so_h + offs_d, acc.to(O.dtype.element_ty))

    def _s1_attention(q, k, v, mask, scale):
        """q [1,H,1,D] bf16, k/v [1,H,W,D] bf16 full-width views,
        mask [1,1,1,W] bool -> [1,H,1,D] like SDPA."""
        H, D = q.shape[1], q.shape[3]
        W = k.shape[2]
        o = torch.empty_like(q)
        NS = 8
        CW = W // NS
        pm = torch.empty(H, NS, device=q.device, dtype=torch.float32)
        pl = torch.empty(H, NS, device=q.device, dtype=torch.float32)
        pa = torch.empty(H, NS, D, device=q.device, dtype=torch.float32)
        _s1_attn_part_kernel[(H, NS)](
            q, k, v, mask, pm, pl, pa,
            q.stride(1),
            k.stride(1), k.stride(2),
            v.stride(1), v.stride(2),
            mask.stride(3),
            scale,
            HD=D, W=W, CW=CW, BLK=128, num_warps=4)
        _s1_attn_merge_kernel[(H,)](pm, pl, pa, o,
                                    o.stride(1), NS=NS, HD=D,
                                    num_warps=1)
        return o


_orig_sdpa = F.scaled_dot_product_attention
_patched = False


@torch.compiler.disable(recursive=True)
def _fast_sdpa(query, key, value, attn_mask=None, dropout_p=0.0,
               is_causal=False, scale=None, **kw):
    if (_HAS_TRITON and query.dim() == 4 and query.shape[0] == 1
            and query.shape[2] == 1
            and query.dtype == torch.bfloat16
            and key.dtype == torch.bfloat16
            and attn_mask is not None and attn_mask.dtype == torch.bool
            and dropout_p == 0.0 and not is_causal
            and query.is_cuda):
        return _s1_attention(query, key, value, attn_mask,
                             scale if scale is not None
                             else 1.0 / (query.shape[-1] ** 0.5))
    return _orig_sdpa(query, key, value, attn_mask=attn_mask,
                      dropout_p=dropout_p, is_causal=is_causal,
                      scale=scale, **kw)


def install_fast_decode_attention():
    """Route q_len==1 bf16 masked SDPA calls to the Triton kernel."""
    global _patched
    if _patched or not _HAS_TRITON or os.environ.get(
            'IXRUN_FAST_ATTN', '1') in ('', '0'):
        return False
    F.scaled_dot_product_attention = _fast_sdpa
    _patched = True
    return True
