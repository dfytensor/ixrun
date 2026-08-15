"""Bind fla (flash-linear-attention) Triton kernels to Qwen3.5 linear attention.

transformers 5.15 expects fla's newer API (`recurrent_gated_delta_rule`), but
the installed fla 0.5.2 exposes the same functionality under different names:
  - prefill (chunked):     fla chunk_gated_delta_rule
  - decode (recurrent):    fla fused_recurrent_gdn
Without this patch the hub-kernel fallback silently resolves to the pure-torch
eager implementations (a python loop over sequence positions in fp32), which
costs ~120ms/token on Qwen3.8-27B — attention+overhead dominates the step.

apply_fla_kernels() rebinds the module-level names inside
transformers.models.qwen3_5.modeling_qwen3_5 so the existing call sites pick
up the fused Triton kernels. Safe no-op when anything is missing.
"""
from __future__ import annotations
import torch

__all__ = ["apply_fla_kernels"]

_APPLIED = False


def apply_fla_kernels(verbose: bool = False) -> bool:
    """Monkeypatch qwen3_5's delta-rule functions to fla Triton kernels.

    Returns True if the fast path is active.
    """
    global _APPLIED
    try:
        import transformers.models.qwen3_5.modeling_qwen3_5 as MQ
        from fla.ops.gated_delta_rule import (
            chunk_gated_delta_rule,
            fused_recurrent_gdn,
        )
    except Exception as e:
        if verbose:
            print(f"[fla] not available ({type(e).__name__}: {e}) — using torch eager", flush=True)
        return False

    # signature adapters (kwargs the call site passes that fla 0.5.2 accepts
    # directly: g, beta, initial_state, output_final_state,
    # use_qk_l2norm_in_kernel, cu_seqlens — all present, no shim needed)
    def _recurrent(query, key, value, *args, **kw):
        with torch.autocast(device_type="cuda", enabled=False):
            out, state = fused_recurrent_gdn(
                query, key, value,
                g=kw.get("g"), beta=kw.get("beta"),
                initial_state=kw.get("initial_state"),
                output_final_state=kw.get("output_final_state", False),
                use_qk_l2norm_in_kernel=kw.get("use_qk_l2norm_in_kernel", False),
                cu_seqlens=kw.get("cu_seqlens"),
            )
        return out.to(query.dtype), state

    def _chunk(query, key, value, *args, **kw):
        with torch.autocast(device_type="cuda", enabled=False):
            out, state = chunk_gated_delta_rule(
                query, key, value,
                g=kw.get("g"), beta=kw.get("beta"),
                initial_state=kw.get("initial_state"),
                output_final_state=kw.get("output_final_state", False),
                use_qk_l2norm_in_kernel=kw.get("use_qk_l2norm_in_kernel", False),
                cu_seqlens=kw.get("cu_seqlens"),
            )
        return out.to(query.dtype), state

    MQ.torch_recurrent_gated_delta_rule = _recurrent
    MQ.torch_chunk_gated_delta_rule = _chunk
    _APPLIED = True
    if verbose:
        print("[fla] delta-rule Triton kernels bound (recurrent+chunk)", flush=True)
    return True
