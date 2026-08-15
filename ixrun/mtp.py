"""Multi-Token Prediction (MTP) head + speculative decoding for Qwen3.8.

Qwen3.8-27B ships an MTP module (config `mtp_num_hidden_layers: 1`) that
transformers loads-and-ignores (`_keys_to_ignore_on_load_unexpected`). The
weights follow the DeepSeek-V3 / Qwen3-Next MTP layout:

    h1 = fc( cat( RMSNorm_e(emb(t)), RMSNorm_h(h) ) )     # 10240 -> 5120
    z  = layer( h1 + h )                                   # 1 x full-attn decoder layer
    logits = lm_head( RMSNorm(z) )                          # shared with main model

`build_mtp_head` reconstructs this module from the safetensors shards and
quantizes its Linears with INT8-X (StreamingLinear, same as the main model) so
it fits alongside the 27B on a 24GB card.

`spec_generate` implements greedy speculative decoding:
  - main model predicts t1; MTP drafts d = next-next from (t1, h)
  - forward [t1, d] through a CLONED cache; position t1's logits verify d
  - accept: emit 2 tokens per forward (cache kept)
  - reject: rerun t1 alone on the clean cache (clone discarded) 鈥?hybrid
    linear-attention recurrent states cannot be cropped retroactively, hence
    the clone instead of cache.crop()
"""
from __future__ import annotations
import copy
import json
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import DEFAULT_LEVELS

__all__ = ["build_mtp_head", "MTPHead", "spec_generate"]


class MTPHead(nn.Module):
    """One-step lookahead head. Linears replaced by INT8-X StreamingLinear."""

    def __init__(self, dim, layer, lm_head):
        super().__init__()
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5RMSNorm

        # qwen3_5 RMSNorm is the (1 + weight) variant — nn.RMSNorm would
        # silently produce wrong values (0.5% spec accept rate)
        self.norm_e = Qwen3_5RMSNorm(dim, eps=1e-6)
        self.norm_h = Qwen3_5RMSNorm(dim, eps=1e-6)
        self.fc = nn.Linear(2 * dim, dim, bias=False)
        self.layer = layer
        self.norm = Qwen3_5RMSNorm(dim, eps=1e-6)
        self.lm_head = lm_head

    def forward(self, tok_emb, h, position_ids, past_key_values=None, use_cache=True,
                position_embeddings=None):
        """tok_emb: [B,1,dim] embedding of the committed token; h: [B,1,dim]
        final-normed last hidden of the main model. Returns logits.

        Structure (validated by teacher-forcing variant search, 62.5% t+2
        accuracy vs 0% for the DeepSeek-style residual variants):
            z = fc( cat( norm_e(emb), norm_h(h) ) )     # NO residual
            logits = lm_head( norm( layer(z) ) )
        """
        x = self.fc(
            torch.cat([self.norm_e(tok_emb), self.norm_h(h)], dim=-1)
        )
        if position_embeddings is None:
            if self._rotary_emb is not None:
                position_embeddings = self._rotary_emb(x, position_ids)
            else:
                position_embeddings = None  # layer may compute rope itself
        z = self.layer(
            x,
            attention_mask=None,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=position_ids,
            position_embeddings=position_embeddings,
        )
        if isinstance(z, tuple):
            z = z[0]
        return self.lm_head(self.norm(z))


def _load_mtp_tensors(model_path: str) -> dict:
    from safetensors import safe_open

    with open(os.path.join(model_path, "model.safetensors.index.json")) as f:
        index = json.load(f)["weight_map"]
    mtp_keys = sorted(k for k in index if k.startswith("mtp."))
    if not mtp_keys:
        return {}
    tensors = {}
    for shard in sorted({index[k] for k in mtp_keys}):
        with safe_open(os.path.join(model_path, shard), "pt") as sf:
            for k in sf.keys():
                if k.startswith("mtp."):
                    tensors[k] = sf.get_tensor(k)
    return tensors


@torch.no_grad()
def build_mtp_head(model, model_path: str, level_bits=DEFAULT_LEVELS, verbose=True):
    """Construct + load the MTP head onto `model`'s device.

    Quantizes fc/gate/up/down/q/k/v/o with INT8-X streaming. Returns the head
    or None if the checkpoint has no MTP weights.
    """
    tensors = _load_mtp_tensors(model_path)
    if not tensors:
        if verbose:
            print("[mtp] no MTP weights in checkpoint", flush=True)
        return None

    from transformers.models.qwen3_5.modeling_qwen3_5 import (
        Qwen3_5DecoderLayer,
    )
    from .quantize import int8x_quantize
    from .engine import StreamingLinear

    cfg = model.config.text_config
    dim = cfg.hidden_size

    # single full-attention layer (weight shapes match the main full-attn
    # layers exactly: q 12288 / k 1024 / v 1024 / o from 6144)
    layer_cfg = copy.deepcopy(cfg)
    layer_cfg.layer_types = ["full_attention"]
    layer = Qwen3_5DecoderLayer(layer_cfg, layer_idx=0).to(torch.bfloat16)

    sd = {}
    for k, t in tensors.items():
        short = k[len("mtp."):]
        t = t.to(torch.bfloat16)
        if short.startswith("layers.0."):
            sd["layer." + short[len("layers.0."):]] = t
        elif short == "fc.weight":
            sd["fc.weight"] = t
        elif short == "norm.weight":
            sd["norm.weight"] = t
        elif short == "pre_fc_norm_embedding.weight":
            sd["norm_e.weight"] = t
        elif short == "pre_fc_norm_hidden.weight":
            sd["norm_h.weight"] = t

    # find the shared lm_head + embed on the main model
    lm_head = model.lm_head if hasattr(model, "lm_head") else model.get_output_embeddings()
    embed = model.get_input_embeddings()

    head = MTPHead(dim, layer, lm_head)
    missing, unexpected = head.load_state_dict(sd, strict=False)
    # only linear weights are missing (we quantize them instead)
    head = head.cuda()

    # swap linears -> StreamingLinear (INT8-X) and free the bf16 originals
    from .linear import _set_parent_child

    def _quant_child(parent, attr):
        lin = getattr(parent, attr)
        p = int8x_quantize(lin.weight.data.cpu(), level_bits)
        sl = StreamingLinear(p, bias=None)
        sl._set_shared_buf(_shared_w_buf)
        _set_parent_child(parent, attr, sl)
        del lin

    # one shared decode buffer for the whole head (largest: fc 10240->5120
    # is out 5120 x in 10240 = 52M; mlp up/gate 17408x5120 = 89M dominates)
    max_n = max(
        tensors["mtp.fc.weight"].numel(),
        tensors["mtp.layers.0.mlp.up_proj.weight"].numel(),
        tensors["mtp.layers.0.self_attn.q_proj.weight"].numel(),
    )
    _shared_w_buf = torch.empty(max_n, dtype=torch.bfloat16, device="cuda")

    _quant_child(head, "fc")
    for attr in ("q_proj", "k_proj", "v_proj", "o_proj"):
        _quant_child(head.layer.self_attn, attr)
    for attr in ("gate_proj", "up_proj", "down_proj"):
        _quant_child(head.layer.mlp, attr)

    head.eval()
    if verbose:
        print("[mtp] head loaded + INT8-X quantized (9 linears)", flush=True)
    head._embed = embed
    # rope lives on the inner text model; find it dynamically
    for _, mod in model.named_modules():
        if type(mod).__name__.endswith("TextModel") and hasattr(mod, "rotary_emb"):
            head._rotary_emb = mod.rotary_emb
            break
    else:
        head._rotary_emb = None
    return head


# --------------------------------------------------------------------------- #
#  Greedy speculative generation
# --------------------------------------------------------------------------- #
@torch.no_grad()
def spec_generate(
    model,
    tokenizer,
    prompt_ids: torch.Tensor,
    mtp: "MTPHead",
    max_new_tokens: int = 200,
    pad_token_id: int | None = None,
    on_tokens=None,
):
    """Greedy generation with 1-token MTP speculation. Yields nothing; calls
    `on_tokens(new_tokens_tensor)` per committed chunk if given. Returns the
    full generated id tensor.
    """
    device = prompt_ids.device
    B = prompt_ids.shape[0]
    assert B == 1, "speculative path is batch-1"

    out = model(
        prompt_ids,
        use_cache=True,
        output_hidden_states=True,
    )
    cache = out.past_key_values
    h_last = out.hidden_states[-1][:, -1:]         # [1,1,dim]
    pos_last = torch.tensor([[prompt_ids.shape[1] - 1]], device=device)

    t1 = out.logits[:, -1].argmax(-1, keepdim=True)  # [1,1]
    d = mtp(mtp._embed(t1), h_last, pos_last + 1)[0].argmax(-1, keepdim=True)

    gen = [t1]
    n = 1
    accepted = rejected = 0

    while n < max_new_tokens:
        # --- try 2-token speculative step on a cloned cache ---
        cache_try = copy.deepcopy(cache)
        out2 = model(
            torch.cat([t1, d], dim=1),
            past_key_values=cache_try,
            use_cache=True,
            output_hidden_states=True,
        )
        t2 = out2.logits[:, 0].argmax(-1, keepdim=True)   # true successor of t1

        if t2.item() == d.item():
            # accept: commit both; next prediction from d's position
            cache = cache_try
            t3 = out2.logits[:, 1].argmax(-1, keepdim=True)
            h_new = out2.hidden_states[-1][:, 1:2]
            gen.extend([d, t3])
            n += 2
            accepted += 1
            if on_tokens:
                on_tokens(torch.cat([d, t3], dim=1))
            if t3.item() == tokenizer.eos_token_id:
                break
            t1, d = t3, mtp(mtp._embed(t3), h_new, pos_last + torch.tensor([[2]], device=device))[0].argmax(-1, keepdim=True)
            pos_last = pos_last + 2
        else:
            # reject: discard speculative cache; redo t1 alone on clean cache
            out3 = model(
                t1,
                past_key_values=cache,
                use_cache=True,
                output_hidden_states=True,
            )
            cache = out3.past_key_values
            h_new = out3.hidden_states[-1][:, -1:]
            gen.append(t2)
            n += 1
            rejected += 1
            if on_tokens:
                on_tokens(t2)
            if t2.item() == tokenizer.eos_token_id:
                break
            t1, d = t2, mtp(mtp._embed(t2), h_new, pos_last + 1)[0].argmax(-1, keepdim=True)
            pos_last = pos_last + 1

    stats = {"accepted": accepted, "rejected": rejected,
             "accept_rate": accepted / max(accepted + rejected, 1)}
    res = torch.cat(gen, dim=1)
    res._mtp_stats = stats
    return res

