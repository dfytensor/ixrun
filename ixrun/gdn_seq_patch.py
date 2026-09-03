"""Minimal surgical patch: seeded small-block chunk-delta -> per-token recurrent.

Replaces ONLY the module-level `torch_chunk_gated_delta_rule` inside
transformers.models.qwen3_5.modeling_qwen3_5. When called with a small block
(S <= 8) AND an existing initial_state (the speculative-verify case), it
loops the PROVEN single-token recurrent kernel per token, chaining the
state; everything else (S=1 decode, prefill without state) hits the
original functions untouched. The forward code path is 100% original.

Root cause this fixes: chunk_gated_delta_rule with initial_state produces
wrong OUTPUT READOUT for small blocks (~450x on token 1) while its state
transition stays correct — verified by seeded-state A/B probe.
"""
from __future__ import annotations

_APPLIED = False
_MAX_SPLIT_S = 8


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

    prev_chunk = MQ.torch_chunk_gated_delta_rule
    recurrent = MQ.torch_recurrent_gated_delta_rule
    _hits = [0]

    def _chunk_or_seq(query, key, value, *args, **kw):
        S = query.shape[1]
        initial_state = kw.get("initial_state")
        if S <= _MAX_SPLIT_S and initial_state is not None:
            _hits[0] += 1
            if _hits[0] <= 3:
                print(f"[gdn-patch] seq split FIRED (S={S}, "
                      f"hits={_hits[0]})", flush=True)
            # seeded small block (spec verify): per-token recurrent chain
            state = initial_state
            outs = []
            g = kw.get("g")
            beta = kw.get("beta")
            for t in range(S):
                kw_t = dict(kw)
                kw_t["g"] = g[:, t:t + 1] if g is not None else None
                kw_t["beta"] = beta[:, t:t + 1] if beta is not None else None
                kw_t["initial_state"] = state
                o_t, state = recurrent(
                    query[:, t:t + 1], key[:, t:t + 1], value[:, t:t + 1],
                    *args, **kw_t)
                outs.append(o_t)
            import torch
            return torch.cat(outs, dim=1), state
        return prev_chunk(query, key, value, *args, **kw)

    MQ.torch_chunk_gated_delta_rule = _chunk_or_seq
    _APPLIED = True
    if verbose:
        print("[gdn-patch] chunk dispatch hooked (seeded S<=8 -> per-token "
              "recurrent; forward untouched)", flush=True)
    return True
