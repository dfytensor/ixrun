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
                             scale, REP: tl.constexpr, POS_MODE: tl.constexpr,
                             HD: tl.constexpr, W: tl.constexpr,
                             CW: tl.constexpr, BLK: tl.constexpr):
        """Phase 1: grid (H, W//CW) - chunk-local online softmax partials.
        M is a bool mask [1,1,1,W] (POS_MODE=0) or a device int64 tensor
        holding the current decode position (POS_MODE=1: keep = offs<=pos)."""
        hq = tl.program_id(0)
        h = hq // REP          # GQA: q heads map onto kv heads
        c = tl.program_id(1)
        offs_d = tl.arange(0, HD)
        q = tl.load(Q + hq * sq_h + offs_d).to(tl.float32)

        m_i = tl.zeros([1], dtype=tl.float32) - float('inf')
        l_i = tl.zeros([1], dtype=tl.float32)
        acc = tl.zeros([HD], dtype=tl.float32)
        base = c * CW
        for start in range(base, base + CW, BLK):
            offs = start + tl.arange(0, BLK)
            if POS_MODE:
                pos = tl.load(M)
                keep = offs <= pos
            else:
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
        tl.store(PM + hq * (W // CW) + c, tl.sum(m_i, axis=0))
        tl.store(PL + hq * (W // CW) + c, tl.sum(l_i, axis=0))
        tl.store(PA + (hq * (W // CW) + c) * HD + offs_d, acc)

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
        rep = max(1, H // k.shape[1])
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
            0,
            scale, rep, 0,
            HD=D, W=W, CW=CW, BLK=128, num_warps=4)
        _s1_attn_merge_kernel[(H,)](pm, pl, pa, o,
                                    o.stride(1), NS=NS, HD=D,
                                    num_warps=1)
        return o

    def _s1_attention_pos(q, k, v, pos, scale):
        """pos mode: pos = device int64 tensor with the decode position."""
        H, D = q.shape[1], q.shape[3]
        rep = max(1, H // k.shape[1])
        W = k.shape[2]
        o = torch.empty_like(q)
        NS = 8
        CW = W // NS
        pm = torch.empty(H, NS, device=q.device, dtype=torch.float32)
        pl = torch.empty(H, NS, device=q.device, dtype=torch.float32)
        pa = torch.empty(H, NS, D, device=q.device, dtype=torch.float32)
        _s1_attn_part_kernel[(H, NS)](
            q, k, v, pos, pm, pl, pa,
            q.stride(1),
            k.stride(1), k.stride(2),
            v.stride(1), v.stride(2),
            0,
            scale, rep, 1,
            HD=D, W=W, CW=CW, BLK=128, num_warps=4)
        _s1_attn_merge_kernel[(H,)](pm, pl, pa, o,
                                    o.stride(1), NS=NS, HD=D,
                                    num_warps=1)
        return o


_orig_sdpa = F.scaled_dot_product_attention
_patched = False
_pos_tensor = None      # device int64 tensor with the decode position


@torch.compiler.disable(recursive=True)
def _fast_sdpa(query, key, value, attn_mask=None, dropout_p=0.0,
               is_causal=False, scale=None, **kw):
    if (_HAS_TRITON and query.dim() == 4 and query.shape[0] == 1
            and query.shape[2] == 1
            and query.dtype == torch.bfloat16
            and key.dtype == torch.bfloat16
            and dropout_p == 0.0 and not is_causal
            and query.is_cuda):
        sc = scale if scale is not None else 1.0 / (query.shape[-1] ** 0.5)
        if attn_mask is None and _pos_tensor is not None:
            return _s1_attention_pos(query, key, value, _pos_tensor, sc)
        if attn_mask is not None and attn_mask.dtype == torch.bool:
            return _s1_attention(query, key, value, attn_mask, sc)
    return _orig_sdpa(query, key, value, attn_mask=attn_mask,
                      dropout_p=dropout_p, is_causal=is_causal,
                      scale=scale, **kw)


def attach_pos_tensor(pos):
    """Give the kernel a device int64 tensor holding the decode position;
    mask-less decode SDPA calls then route to the positional kernel and
    the engine can skip HF's per-step causal-mask construction."""
    global _pos_tensor
    _pos_tensor = pos


# ------------------------------------------------------------------ #
# fused rotary application: one kernel per layer handles q AND k rows
# (rotate_half via the partner-element load: rot[j] = x[(j+D/2)%D],
# sign -1 on the low half) - replaces HF's 5-kernel mul/neg/cat/add
# chain per tensor (~0.3ms/step on the 1B).
# ------------------------------------------------------------------ #
if _HAS_TRITON:

    @triton.jit
    def _rope_qk_kernel(Q, K, COS, SIN, NQ, S,
                        sq_h, sq_s, sk_h, sk_s,
                        HD: tl.constexpr):
        pid = tl.program_id(0)
        offs = tl.arange(0, HD)
        half: tl.constexpr = HD // 2
        if pid < NQ:
            h = pid // S
            s = pid % S
            base = Q + h * sq_h + s * sq_s
        else:
            r = pid - NQ
            h = r // S
            s = r % S
            base = K + h * sk_h + s * sk_s
        c = tl.load(COS + s * HD + offs).to(tl.float32)
        sn = tl.load(SIN + s * HD + offs).to(tl.float32)
        row = tl.load(base + offs).to(tl.float32)
        row_p = tl.load(base + ((offs + half) % HD)).to(tl.float32)
        sign = tl.where(offs < half, -1.0, 1.0)
        out = row * c + row_p * sign * sn
        tl.store(base + offs, out.to(Q.dtype.element_ty))


_orig_rope = None
_rope_patched = False


@torch.compiler.disable(recursive=True)
def _fused_apply_rotary(q, k, cos, sin, position_ids=None,
                        unsqueeze_dim=1):
    if (q.shape[0] != 1 or q.dtype != torch.bfloat16
            or not q.is_cuda or not k.is_cuda or not cos.is_cuda):
        return _orig_rope(q, k, cos, sin, position_ids, unsqueeze_dim)
    B, H, S, D = q.shape
    kvh = k.shape[1]
    c2 = cos[0] if cos.dim() == 3 else cos.reshape(
        cos.shape[1], cos.shape[-1])
    s2 = sin[0] if sin.dim() == 3 else sin.reshape(
        sin.shape[1], sin.shape[-1])
    # in-place on the ORIGINAL 4-D tensors (stride-addressed rows;
    # reshape would copy non-contiguous q and orphan the stores)
    _rope_qk_kernel[(B * (H + kvh) * S,)](
        q, k, c2, s2, B * H * S, S,
        q.stride(1), q.stride(2), k.stride(1), k.stride(2),
        HD=D, num_warps=1)
    return q, k


def install_fused_rope():
    """Replace modeling_llama.apply_rotary_pos_emb with the fused
    Triton kernel (in-place).  Returns True if installed."""
    global _orig_rope, _rope_patched
    if _rope_patched or not _HAS_TRITON or os.environ.get(
            'IXRUN_FUSED_ROPE', '1') in ('', '0'):
        return False
    import sys

    mo = sys.modules.get('transformers.models.llama.modeling_llama')
    if mo is None or not hasattr(mo, 'apply_rotary_pos_emb'):
        return False
    _orig_rope = mo.apply_rotary_pos_emb
    mo.apply_rotary_pos_emb = _fused_apply_rotary
    _rope_patched = True
    return True


# ------------------------------------------------------------------ #
# fused residual-add + RMSNorm: one kernel emits BOTH the sum (the new
# residual) and the normed output - replaces HF's separate add kernel
# + layernorm chain at the attention->MLP boundary (~0.1-0.2ms/step).
# ------------------------------------------------------------------ #
if _HAS_TRITON:

    @triton.jit
    def _add_norm_kernel(A, B, W, SUM, OUT, eps,
                         HD: tl.constexpr, BLK: tl.constexpr):
        r = tl.program_id(0)
        offs = tl.arange(0, BLK)
        mask = offs < HD
        a = tl.load(A + r * HD + offs, mask=mask, other=0.0).to(tl.float32)
        b = tl.load(B + r * HD + offs, mask=mask, other=0.0).to(tl.float32)
        s = a + b
        tl.store(SUM + r * HD + offs, s.to(OUT.dtype.element_ty), mask=mask)
        var = tl.sum(s * s, axis=0) / HD
        rr = 1.0 / tl.sqrt(var + eps)
        w = tl.load(W + offs, mask=mask, other=0.0).to(tl.float32)
        tl.store(OUT + r * HD + offs, (s * rr * w).to(OUT.dtype.element_ty),
                 mask=mask)


@torch.compiler.disable(recursive=True)
def _fused_add_norm(residual, x, ln_weight, eps):
    hd = residual.shape[-1]
    rows = residual.numel() // hd
    a = residual.reshape(rows, hd)
    b = x.reshape(rows, hd)
    out = torch.empty_like(a)
    BLK = 2048 if hd <= 2048 else 4096
    _add_norm_kernel[(rows,)](a, b, ln_weight, b, out,
                              float(eps), HD=hd, BLK=BLK, num_warps=8)
    return out.reshape(x.shape), b.reshape(x.shape)


_orig_layer_fwd = None
_layer_patched = False


def _patched_layer_fwd(self, hidden_states, attention_mask=None,
                       position_ids=None, past_key_values=None,
                       use_cache=False, position_embeddings=None,
                       **kwargs):
    residual = hidden_states
    hidden_states = self.input_layernorm(hidden_states)
    hidden_states, _ = self.self_attn(
        hidden_states=hidden_states,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        use_cache=use_cache,
        position_embeddings=position_embeddings,
        **kwargs)
    # fused: sum -> new residual, rmsnorm -> mlp input (in-place on the
    # attention output buffer)
    hidden_states, residual = _fused_add_norm(
        residual, hidden_states, self.post_attention_layernorm.weight,
        self.post_attention_layernorm.variance_epsilon)
    hidden_states = self.mlp(hidden_states)
    hidden_states = residual + hidden_states
    return hidden_states


def install_fused_add_norm():
    """Replace LlamaDecoderLayer.forward with the fused add+norm
    variant.  Returns True if installed."""
    global _orig_layer_fwd, _layer_patched
    if _layer_patched or not _HAS_TRITON or os.environ.get(
            'IXRUN_FUSED_ADDNORM', '1') in ('', '0'):
        return False
    import sys

    mo = sys.modules.get('transformers.models.llama.modeling_llama')
    if mo is None or not hasattr(mo, 'LlamaDecoderLayer'):
        return False
    _orig_layer_fwd = mo.LlamaDecoderLayer.forward
    mo.LlamaDecoderLayer.forward = _patched_layer_fwd
    _layer_patched = True
    return True


# ------------------------------------------------------------------ #
# qwen3_5 (27B PY engines) variant: Qwen3_5RMSNorm is ZERO-CENTERED -
# output = rmsnorm(x) * (1 + weight).  The (1+w) factor is precomputed
# per layer at install time; the kernel math is otherwise the same.
# ------------------------------------------------------------------ #
def install_fused_add_norm_q35():
    """Replace Qwen3_5DecoderLayer.forward with the fused add+norm
    variant (zero-centered RMSNorm).  Returns True if installed."""
    if not _HAS_TRITON or os.environ.get(
            'IXRUN_FUSED_ADDNORM', '1') in ('', '0'):
        return False
    import sys
    import types

    mo = sys.modules.get('transformers.models.qwen3_5.modeling_qwen3_5')
    if mo is None or not hasattr(mo, 'Qwen3_5DecoderLayer'):
        return False
    if getattr(mo.Qwen3_5DecoderLayer, '_fused_an', False):
        return False

    def _fwd(self, hidden_states, position_embeddings,
             attention_mask=None, position_ids=None,
             past_key_values=None, **kwargs):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        if self.block_type == 'linear_attention':
            hidden_states = self.linear_attn(
                hidden_states=hidden_states,
                cache_params=past_key_values,
                attention_mask=attention_mask, **kwargs)
        else:
            hidden_states, _ = self.self_attn(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                position_embeddings=position_embeddings, **kwargs)
        # fused: sum -> new residual (in-place on the attn output),
        # rmsnorm * (1+w) -> mlp input
        ln = self.post_attention_layernorm
        hd = residual.shape[-1]
        rows = residual.numel() // hd
        a = residual.reshape(rows, hd)
        b = hidden_states.reshape(rows, hd)
        out = torch.empty_like(a)
        BLK = 2048 if hd <= 2048 else (4096 if hd <= 4096 else 8192)
        _add_norm_kernel[(rows,)](a, b, self._w1p, b, out,
                                  float(ln.eps), HD=hd, BLK=BLK,
                                  num_warps=8)
        residual = b.reshape(residual.shape)
        hidden_states = out.reshape(residual.shape)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states

    cls = mo.Qwen3_5DecoderLayer
    cls.forward = _fwd
    cls._fused_an = True
    # (1+w) buffers are built post-load by prep_q35_layers - building
    # them in __init__ would capture the META/unloaded weight
    return True


def prep_q35_layers(model):
    """(Re)build the (1+w) buffers from the LOADED weights."""
    n = 0
    for m in model.modules():
        if type(m).__name__ == 'Qwen3_5DecoderLayer':
            ln = m.post_attention_layernorm
            m._w1p = (1.0 + ln.weight.data.float()).contiguous()
            n += 1
    return n


def install_fast_decode_attention():
    """Route q_len==1 bf16 masked SDPA calls to the Triton kernel."""
    global _patched
    if _patched or not _HAS_TRITON or os.environ.get(
            'IXRUN_FAST_ATTN', '1') in ('', '0'):
        return False
    F.scaled_dot_product_attention = _fast_sdpa
    _patched = True
    return True
