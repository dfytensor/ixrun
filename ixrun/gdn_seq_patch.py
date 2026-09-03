"""v3: full surgical fix for seeded multi-token (spec-verify) correctness.

TWO bugs in the stock Qwen3_5GatedDeltaNet forward for 1 < S <= 8 with
existing cache state (the speculative-verify configuration):

1. CONV WINDOW MISALIGNMENT (root cause of degenerate text):
   S=1 decode: causal_conv1d_update maintains a (kernel-1)-sized window.
   S>1 block: update_conv_state keeps a kernel-sized window via
   `copy_(full[..., -kernel_size:])`. The NEXT S=1 step then reads a
   misaligned window -> corruption accumulates per verify step.

2. CHUNK-DELTA READOUT (fixed in v2): chunk_gated_delta_rule with
   initial_state gives wrong OUTPUT for small blocks (450x on tok1).

v3 replaces the forward but keeps S=1 decode and prefill paths
BYTE-IDENTICAL to the original (they are proven correct); only the
1 < S <= 8 with-cache path changes: per-token causal_conv1d_update +
per-token recurrent, chained over batched (M=S GEMM) projections.
"""
from __future__ import annotations
import torch
import torch.nn.functional as F

_APPLIED = False
_MAX_S = 8


def apply_gdn_sequential_patch(verbose: bool = False) -> bool:
    global _APPLIED
    if _APPLIED:
        return True
    try:
        from transformers.models.qwen3_5 import modeling_qwen3_5 as MQ
    except Exception as e:
        if verbose:
            print(f"[gdn-patch] unavailable ({e})", flush=True)
        return False

    conv_update = MQ.causal_conv1d_update
    conv_fn = MQ.causal_conv1d_fn
    recurrent = MQ.torch_recurrent_gated_delta_rule
    chunk = MQ.torch_chunk_gated_delta_rule
    globals()['_HITS'] = 0

    def _forward_v3(
        self,
        hidden_states: torch.Tensor,
        cache_params=None,
        attention_mask: torch.Tensor | None = None,
        **kwargs,
    ):
        from transformers.models.qwen3_5.modeling_qwen3_5 import (
            apply_mask_to_padding_states,
        )

        hidden_states = apply_mask_to_padding_states(hidden_states, attention_mask)
        batch_size, seq_len, _ = hidden_states.shape
        use_precomputed_states = cache_params is not None and \
            cache_params.has_previous_state(self.layer_idx)

        # ---- projections: batched at M=seq_len (GEMM, proven clean) ----
        mixed_qkv = self.in_proj_qkv(hidden_states).transpose(1, 2)  # [B,dim,S]
        z = self.in_proj_z(hidden_states)
        z = z.reshape(batch_size, seq_len, -1, self.head_v_dim)
        b = self.in_proj_b(hidden_states)          # [B,S,vh]
        a = self.in_proj_a(hidden_states)

        lay = cache_params.layers[self.layer_idx] if cache_params is not None else None
        is_spec_block = (
            cache_params is not None and use_precomputed_states
            and 1 < seq_len <= _MAX_S and lay is not None
            and not lay.record_past
        )

        if use_precomputed_states and seq_len == 1 and lay is not None \
                and not lay.record_past:
            # ===== ORIGINAL S=1 decode path (byte-identical) =====
            conv_state = lay.conv_states[0]
            mixed_qkv = conv_update(
                mixed_qkv, conv_state,
                self.conv1d.weight.squeeze(1), self.conv1d.bias,
                self.activation,
            )
            mixed_qkv = mixed_qkv.transpose(1, 2)
            query, key, value = torch.split(
                mixed_qkv,
                [self.key_dim, self.key_dim, self.value_dim], dim=-1)
            query = query.reshape(batch_size, 1, -1, self.head_k_dim)
            key = key.reshape(batch_size, 1, -1, self.head_k_dim)
            value = value.reshape(batch_size, 1, -1, self.head_v_dim)
            beta = b.sigmoid()
            g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)
            if self.num_v_heads // self.num_k_heads > 1:
                query = query.repeat_interleave(
                    self.num_v_heads // self.num_k_heads, dim=2)
                key = key.repeat_interleave(
                    self.num_v_heads // self.num_k_heads, dim=2)
            state = lay.recurrent_states[0]
            core_attn_out, last_state = recurrent(
                query, key, value, g=g, beta=beta,
                initial_state=state,
                output_final_state=cache_params is not None,
                use_qk_l2norm_in_kernel=True,
                cu_seqlens=kwargs.pop("cu_seq_lens_q", None),
                **kwargs,
            )
        elif is_spec_block:
            # ===== v3 FIX: per-token conv + delta, batched projections =====
            globals()['_HITS'] = globals().get('_HITS', 0) + 1
            conv_state = lay.conv_states[0]
            state = lay.recurrent_states[0]
            beta_all = b.sigmoid()               # [B,S,vh]
            g_all = (-self.A_log.float().exp()
                     * F.softplus(a.float() + self.dt_bias))
            outs = []
            for t in range(seq_len):
                # NOTE: slices of [B,dim,S] are STRIDED VIEWS — fla kernels
                # index by raw pointers and silently read wrong memory on
                # non-contiguous input. .contiguous() is REQUIRED (this
                # exact bug cost a full session of debugging).
                qkv_t = conv_update(
                    mixed_qkv[:, :, t:t + 1].contiguous(), conv_state,
                    self.conv1d.weight.squeeze(1), self.conv1d.bias,
                    self.activation,
                ).transpose(1, 2)                 # [B,1,dim]
                q_t, k_t, v_t = torch.split(
                    qkv_t, [self.key_dim, self.key_dim, self.value_dim],
                    dim=-1)
                q_t = q_t.reshape(batch_size, 1, -1, self.head_k_dim)
                k_t = k_t.reshape(batch_size, 1, -1, self.head_k_dim)
                v_t = v_t.reshape(batch_size, 1, -1, self.head_v_dim)
                if self.num_v_heads // self.num_k_heads > 1:
                    q_t = q_t.repeat_interleave(
                        self.num_v_heads // self.num_k_heads, dim=2)
                    k_t = k_t.repeat_interleave(
                        self.num_v_heads // self.num_k_heads, dim=2)
                o_t, state = recurrent(
                    q_t.contiguous(), k_t.contiguous(), v_t.contiguous(),
                    g=g_all[:, t:t + 1].contiguous(),
                    beta=beta_all[:, t:t + 1].contiguous(),
                    initial_state=state,
                    output_final_state=True,
                    use_qk_l2norm_in_kernel=True,
                    **kwargs,
                )
                outs.append(o_t)
            core_attn_out = torch.cat(outs, dim=1)
            last_state = state
        else:
            # ===== ORIGINAL prefill / no-state path (byte-identical) =====
            if cache_params is not None:
                mixed_qkv = cache_params.update_conv_state(
                    mixed_qkv, self.layer_idx,
                    conv_kernel_size=self.conv_kernel_size)
            mixed_qkv = conv_fn(
                mixed_qkv,
                self.conv1d.weight.squeeze(1),
                self.conv1d.bias,
                activation=self.activation,
                **kwargs,
            )
            if cache_params is not None:
                mixed_qkv = mixed_qkv[:, :, -seq_len:]
            mixed_qkv = mixed_qkv.transpose(1, 2)
            query, key, value = torch.split(
                mixed_qkv,
                [self.key_dim, self.key_dim, self.value_dim], dim=-1)
            query = query.reshape(batch_size, seq_len, -1, self.head_k_dim)
            key = key.reshape(batch_size, seq_len, -1, self.head_k_dim)
            value = value.reshape(batch_size, seq_len, -1, self.head_v_dim)
            beta = b.sigmoid()
            g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)
            if self.num_v_heads // self.num_k_heads > 1:
                query = query.repeat_interleave(
                    self.num_v_heads // self.num_k_heads, dim=2)
                key = key.repeat_interleave(
                    self.num_v_heads // self.num_k_heads, dim=2)
            recurrent_state = lay.recurrent_states[0] \
                if use_precomputed_states and lay is not None else None
            core_attn_out, last_state = chunk(
                query, key, value, g=g, beta=beta,
                initial_state=recurrent_state,
                output_final_state=cache_params is not None,
                use_qk_l2norm_in_kernel=True,
                cu_seqlens=kwargs.pop("cu_seq_lens_q", None),
                **kwargs,
            )

        if cache_params is not None:
            cache_params.update_recurrent_state(last_state, self.layer_idx)

        core_attn_out = core_attn_out.reshape(-1, self.head_v_dim)
        z = z.reshape(-1, self.head_v_dim)
        core_attn_out = self.norm(core_attn_out, z)
        core_attn_out = core_attn_out.reshape(batch_size, seq_len, -1)
        return self.out_proj(core_attn_out)

    MQ.Qwen3_5GatedDeltaNet.forward = _forward_v3
    _APPLIED = True
    if verbose:
        print("[gdn-patch v3] per-token conv+delta for seeded S<=8 "
              "(S=1 & prefill byte-identical)", flush=True)
    return True


def _dbg_spec_hits(reset=False):
    global _HITS
    if reset:
        globals()['_HITS'] = 0
    return globals().get('_HITS', 0)
