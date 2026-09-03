"""Patch Qwen3_5GatedDeltaNet for seeded-state multi-token (spec verify) correctness.

ROOT CAUSE (found by seeded-state A/B probe): the chunk-gated-delta path
(torch_chunk_gated_delta_rule / fla chunk_gated_delta_rule with
initial_state) produces wrong OUTPUT READOUT for small blocks with existing
state (~450x on token 1), while the state TRANSITION itself stays correct
(states are computed from clean input projections). Single-token recurrent
path (decode) is proven correct.

FIX: when use_precomputed_states and 1 < seq_len <= 8 (spec-verify blocks),
run the conv update + delta core per-token via the S=1 recurrent path,
chaining recurrent_state manually. Projections (in_proj_qkv/z/b/a) still
run batched at M=seq_len (the expensive GEMMs keep their speedup); only the
sequential-core walks token by token.
"""
from __future__ import annotations
import torch
import torch.nn.functional as F

_APPLIED = False


def apply_gdn_sequential_patch(verbose: bool = False) -> bool:
    """Monkeypatch Qwen3_5GatedDeltaNet.forward. Returns True if active."""
    global _APPLIED
    try:
        from transformers.models.qwen3_5 import modeling_qwen3_5 as MQ
    except Exception as e:
        if verbose:
            print(f"[gdn-patch] unavailable ({e})", flush=True)
        return False

    torch_recurrent = MQ.torch_recurrent_gated_delta_rule
    causal_conv1d_update = MQ.causal_conv1d_update

    def _forward_sequential_core(
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

        # ---- projections: batched (M=seq_len GEMM) ----
        mixed_qkv = self.in_proj_qkv(hidden_states).transpose(1, 2)
        z = self.in_proj_z(hidden_states)
        z = z.reshape(batch_size, seq_len, -1, self.head_v_dim)
        b = self.in_proj_b(hidden_states)
        a = self.in_proj_a(hidden_states)

        if use_precomputed_states and seq_len == 1 and not \
                cache_params.layers[self.layer_idx].record_past:
            # original single-token fast path (proven correct) — unchanged
            conv_state = cache_params.layers[self.layer_idx].conv_states[0]
            mixed_qkv = causal_conv1d_update(
                mixed_qkv, conv_state,
                self.conv1d.weight.squeeze(1), self.conv1d.bias,
                self.activation,
            )
            query, key, value = torch.split(
                mixed_qkv.transpose(1, 2),
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
            state = cache_params.layers[self.layer_idx].recurrent_states[0]
            core_attn_out, last_state = torch_recurrent(
                query, key, value, g=g, beta=beta,
                initial_state=state, output_final_state=True,
                use_qk_l2norm_in_kernel=True,
                cu_seqlens=kwargs.pop("cu_seq_lens_q", None), **kwargs)
        elif use_precomputed_states and 1 < seq_len <= 8:
            # ---- THE FIX: per-token sequential core over batched
            # projections (conv update + recurrent chaining per token) ----
            conv_state = cache_params.layers[self.layer_idx].conv_states[0]
            state = cache_params.layers[self.layer_idx].recurrent_states[0]
            beta_all = b.sigmoid().transpose(1, 2)      # [B, v_heads, S]
            g_all = (-self.A_log.float().exp()
                     * F.softplus(a.float() + self.dt_bias)).transpose(1, 2)
            outs = []
            for t in range(seq_len):
                qkv_t = mixed_qkv[:, :, t:t + 1]        # [B, dim, 1]
                qkv_t = causal_conv1d_update(
                    qkv_t, conv_state,
                    self.conv1d.weight.squeeze(1), self.conv1d.bias,
                    self.activation,
                ).transpose(1, 2)                        # [B, 1, dim]
                q_t, k_t, v_t = torch.split(
                    qkv_t, [self.key_dim, self.key_dim, self.value_dim],
                    dim=-1)
                q_t = q_t.reshape(batch_size, 1, -1, self.head_k_dim)
                k_t = k_t.reshape(batch_size, 1, -1, self.head_v_dim) \
                    if False else k_t.reshape(batch_size, 1, -1, self.head_k_dim)
                v_t = v_t.reshape(batch_size, 1, -1, self.head_v_dim)
                if self.num_v_heads // self.num_k_heads > 1:
                    q_t = q_t.repeat_interleave(
                        self.num_v_heads // self.num_k_heads, dim=2)
                    k_t = k_t.repeat_interleave(
                        self.num_v_heads // self.num_k_heads, dim=2)
                o_t, state = torch_recurrent(
                    q_t, k_t, v_t,
                    g=g_all[:, :, t:t + 1].transpose(1, 2),
                    beta=beta_all[:, :, t:t + 1].transpose(1, 2),
                    initial_state=state, output_final_state=True,
                    use_qk_l2norm_in_kernel=True, **kwargs)
                outs.append(o_t)
            core_attn_out = torch.cat(outs, dim=1)
            last_state = state
        else:
            # ---- original paths (prefill / no state): unchanged ----
            if cache_params is not None:
                mixed_qkv = cache_params.update_conv_state(
                    mixed_qkv, self.layer_idx,
                    conv_kernel_size=self.conv_kernel_size)
            mixed_qkv = causal_conv1d_fn_forward(self, mixed_qkv)
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
            recurrent_state = cache_params.layers[self.layer_idx] \
                .recurrent_states[0] if use_precomputed_states else None
            core_attn_out, last_state = MQ.torch_chunk_gated_delta_rule(
                query, key, value, g=g, beta=beta,
                initial_state=recurrent_state,
                output_final_state=cache_params is not None,
                use_qk_l2norm_in_kernel=True,
                cu_seqlens=kwargs.pop("cu_seq_lens_q", None), **kwargs)

        if cache_params is not None:
            cache_params.update_recurrent_state(last_state, self.layer_idx)

        core_attn_out = core_attn_out.reshape(-1, self.head_v_dim)
        z = z.reshape(-1, self.head_v_dim)
        core_attn_out = self.norm(core_attn_out, z)
        core_attn_out = core_attn_out.reshape(batch_size, seq_len, -1)
        return self.out_proj(core_attn_out)

    def causal_conv1d_fn_forward(self, mixed_qkv):
        # original conv for the prefill branch (uses module attrs)
        return MQ.causal_conv1d_fn(
            mixed_qkv,
            self.conv1d.weight.squeeze(1),
            self.conv1d.bias,
            activation=self.activation,
        )

    MQ.Qwen3_5GatedDeltaNet.forward = _forward_sequential_core
    _APPLIED = True
    if verbose:
        print("[gdn-patch] sequential S<=8 core active "
              "(conv+recurrent per token, projections batched)", flush=True)
    return True
