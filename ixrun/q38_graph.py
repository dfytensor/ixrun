# -*- coding: utf-8 -*-
"""Q38GraphEngine: Qwen3.8-27B UDCQ 6bpw + StaticCache + CUDA-Graph
greedy/sampling decode, blob fast-deploy.

Same duck-typed interface as Int8XEngine (.tokenizer / .generate / .stream)
so cli chat/generate and the OpenAI-compatible server work unchanged:

    eng = Q38GraphEngine.from_blob(r'...\\q38_blob.pt', r'...\\Qwen3.8-27B')
    eng.generate('The theory of relativity states that', max_new_tokens=64)

Design (proven in experiments/qwen38_udcq/round3_graph.py + round4b_bisect.py):
- UDCQ-packed linears GPU-resident (~20GB), embeddings as CPU rows,
  non-linear params from safetensors shards, loaded from the disk blob
  (18s vs 55min re-quantization)
- Qwen3_5Attention.forward patched for StaticCache (causal mask from
  cumulative_length; transpose before reshape for q_len>1)
- decode: one CUDA graph replay per token (~15 tok/s on RTX 4090 WDDM)
- prefill: blocked S=8 chunks (mt-GEMV M=8, ~8x faster than per-token)

Known hard-won pitfalls baked in (see AGENTS.md):
- cache conv_states/recurrent_states are DICTS 鈥?iterate .values()
- graph outputs must be copied to static buffers inside the capture
- keep VRAM < 24GB or WDDM sysmem paging collapses bandwidth
"""
from __future__ import annotations

import gc
import json
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import QWEN38_PATH
from .fla_patch import apply_fla_kernels
from .gdn_seq_patch import apply_gdn_sequential_patch
from .linear import _set_parent_child
from .udcq import UDCQ_G, UdcqLinear

__all__ = ["Q38GraphEngine"]

MAX_BLOCK = 8          # prefill chunk size (mt-GEMV M=8, bit-exact)


def _apply_static_attention():
    """Patch Qwen3_5Attention.forward for StaticCache decode/prefill."""
    from transformers.models.qwen3_5.modeling_qwen3_5 import (
        Qwen3_5Attention, apply_rotary_pos_emb)

    _orig = Qwen3_5Attention.forward

    def _attn_forward_static(self, hidden_states, position_embeddings,
                             attention_mask=None, past_key_values=None,
                             **kw):
        if past_key_values is None:
            return _orig(self, hidden_states, position_embeddings,
                         attention_mask=attention_mask,
                         past_key_values=None, **kw)
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        q_len = hidden_states.shape[1]

        query_states, gate = torch.chunk(
            self.q_proj(hidden_states).view(
                *input_shape, -1, self.head_dim * 2),
            2, dim=-1)
        gate = gate.reshape(*input_shape, -1)
        query_states = self.q_norm(
            query_states.view(hidden_shape)).transpose(1, 2)
        key_states = self.k_norm(
            self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = self.v_proj(
            hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin)

        k_full, v_full = past_key_values.update(
            key_states, value_states, self.layer_idx)

        cum = past_key_values.layers[self.layer_idx].cumulative_length
        MAX = k_full.shape[-2]
        ar = torch.arange(MAX, device=k_full.device)
        qr = torch.arange(q_len, device=k_full.device)
        keep = ar[None, :] < (cum - q_len + 1 + qr[:, None])
        mask = torch.where(keep, 0.0, float('-inf')).to(query_states.dtype)
        mask = mask.view(1, 1, q_len, MAX)

        n_rep = query_states.shape[1] // k_full.shape[1]
        if n_rep > 1:
            k_full = k_full.repeat_interleave(n_rep, dim=1)
            v_full = v_full.repeat_interleave(n_rep, dim=1)
        if q_len > 1:
            # per-token SDPA: same kernel sequence as sequential q_len=1;
            # transpose REQUIRED before flatten or head/token interleave
            outs = []
            for t in range(q_len):
                outs.append(F.scaled_dot_product_attention(
                    query_states[:, :, t:t + 1], k_full, v_full,
                    attn_mask=mask[:, :, t:t + 1], scale=self.scaling))
            attn_output = torch.cat(outs, dim=2).transpose(1, 2)
        else:
            attn_output = F.scaled_dot_product_attention(
                query_states, k_full, v_full, attn_mask=mask,
                scale=self.scaling).transpose(1, 2)

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = attn_output * torch.sigmoid(gate)
        return self.o_proj(attn_output), None

    Qwen3_5Attention.forward = _attn_forward_static


class _CpuEmbed(nn.Module):
    """Embedding rows kept on CPU (RAM); per-token GPU DMA."""

    def __init__(self, weight_cpu):
        super().__init__()
        self.weight_cpu = weight_cpu

    def forward(self, ids):
        return self.weight_cpu[ids.reshape(-1).cpu()].cuda() \
            .view(*ids.shape, -1)


@torch.no_grad()
def _build_from_blob(blob_path, model_path, verbose=True):
    from accelerate import init_empty_weights
    from safetensors import safe_open
    from transformers import AutoConfig, AutoModelForCausalLM

    if verbose:
        print(f'[q38] loading blob (mmap): {blob_path}', flush=True)
    t0 = time.time()
    blob = torch.load(blob_path, map_location='cpu', mmap=True,
                      weights_only=False)
    if verbose:
        print(f'[q38] blob {time.time() - t0:.0f}s', flush=True)

    cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    with init_empty_weights():
        m = AutoModelForCausalLM.from_config(cfg, trust_remote_code=True)

    cb_gpu = blob['codebook'].cuda()
    n = 0
    for name, mod in list(m.named_modules()):
        if isinstance(mod, nn.Linear) and name in blob['layers']:
            e = blob['layers'][name]
            packed = {'g': UDCQ_G, 'out_f': mod.out_features,
                      'in_f': mod.in_features,
                      'N': mod.out_features * mod.in_features,
                      'idx': e['idx'], 'scale': e['scale'],
                      'sign_packed': e['sign'],
                      'codebook': blob['codebook'],
                      'bits_per_weight': 6.0}
            _set_parent_child(m, name, UdcqLinear(packed, bias=None,
                                                  cache='stream'))
            n += 1

    emb_mod = _CpuEmbed(blob['embed'])
    for name, mod in m.named_modules():
        if type(mod).__name__.endswith('Embedding') and 'embed_tokens' in name:
            parts = name.split('.')
            parent = m
            for q in parts[:-1]:
                parent = getattr(parent, q)
            setattr(parent, parts[-1], emb_mod)
            break

    idx = json.load(open(
        rf'{model_path}\model.safetensors.index.json'))['weight_map']

    def ckpt_key(name):
        if name in idx:
            return name
        alt = name.replace('model.', 'model.language_model.', 1)
        return alt if alt in idx else None

    moved = 0
    for name, p in list(m.named_parameters()):
        if not p.numel() or not p.is_meta:
            continue
        key = ckpt_key(name)
        if key is None:
            continue
        shard = rf'{model_path}\{idx[key]}'
        with safe_open(shard, 'pt') as sf:
            t = sf.get_tensor(key)
        parts = name.split('.')
        parent = m
        for q in parts[:-1]:
            parent = getattr(parent, q)
        parent._parameters[parts[-1]] = torch.nn.Parameter(
            t.cuda(), requires_grad=False)
        moved += t.numel() * t.element_size()
        del t
    gc.collect()
    torch.cuda.empty_cache()
    if verbose:
        print(f'[q38] {n} linears from blob | non-linear {moved / 1e9:.2f}GB'
              f' | {time.time() - t0:.0f}s | resident '
              f'{torch.cuda.memory_allocated() / 1e9:.2f}GB', flush=True)
    return m


class Q38GraphEngine:
    """UDCO 6bpw Qwen3.8-27B, StaticCache + CUDA-Graph token decode."""

    def __init__(self, model, tokenizer, max_ctx=256, verbose=True):
        from transformers import AutoTokenizer
        from transformers.cache_utils import StaticCache

        self.tokenizer = tokenizer or AutoTokenizer.from_pretrained(
            QWEN38_PATH)
        self.max_ctx = max_ctx
        m = model
        m.eval()
        self.model = m
        tm = m.model
        if hasattr(tm, 'language_model'):
            tm = tm.language_model
        self.tm = tm
        self.layers = tm.layers
        self.final_norm = tm.norm
        H = tm.config.hidden_size
        self.H = H
        dev = self.dev = 'cuda'

        pos_all = torch.arange(max_ctx, device=dev).unsqueeze(0)
        with torch.no_grad():
            dummy = torch.zeros(1, max_ctx, H, dtype=torch.bfloat16,
                                device=dev)
            cos_all, sin_all = tm.rotary_emb(dummy, pos_all)
        if cos_all.dim() == 4:
            cos_all = cos_all[:, :, 0]
        if cos_all.dim() == 3:
            cos_all, sin_all = cos_all[0], sin_all[0]
        self._cos_all, self._sin_all = cos_all, sin_all

        self.cache = StaticCache(config=m.config, max_cache_len=max_ctx)

        # decode buffers (S=1) + prefill block buffers (S=8)
        cd = cos_all.shape[-1]
        self.emb1 = torch.zeros(1, 1, H, dtype=torch.bfloat16, device=dev)
        self.cos1 = torch.zeros(1, 1, cd, dtype=cos_all.dtype, device=dev)
        self.sin1 = torch.zeros_like(self.cos1)
        self.pos1 = torch.zeros(1, dtype=torch.long, device=dev)
        self.embB = torch.zeros(1, MAX_BLOCK, H, dtype=torch.bfloat16,
                                device=dev)
        self.cosB = torch.zeros(1, MAX_BLOCK, cd, dtype=cos_all.dtype,
                                device=dev)
        self.sinB = torch.zeros_like(self.cosB)
        self.posB = torch.arange(MAX_BLOCK, device=dev)

        # graph output static buffer (pool-aliasing safe)
        self.log1 = torch.zeros(1, 1, m.lm_head.out_features,
                                dtype=torch.bfloat16, device=dev)

        self._capture(verbose=verbose)

    # ------------------------------------------------------------------ #
    @classmethod
    def from_blob(cls, blob_path, model_path=QWEN38_PATH, tokenizer=None,
                  max_ctx=256, verbose=True):
        apply_fla_kernels()
        apply_gdn_sequential_patch()
        _apply_static_attention()
        m = _build_from_blob(blob_path, model_path, verbose=verbose)
        return cls(m, tokenizer, max_ctx=max_ctx, verbose=verbose)

    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def _forward(self, emb, cos, sin, pos):
        h = emb
        for layer in self.layers:
            h = layer(h, position_embeddings=(cos, sin),
                      attention_mask=None, position_ids=pos.view(1, -1),
                      past_key_values=self.cache)
            if isinstance(h, tuple):
                h = h[0]
        return self.model.lm_head(self.final_norm(h))

    def _capture(self, verbose=True):
        if verbose:
            print('[q38] capturing decode graph...', flush=True)
        self.emb1.zero_(); self.cos1.zero_(); self.sin1.zero_()
        self.pos1.fill_(0)
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                self._forward(self.emb1, self.cos1, self.sin1, self.pos1)
        torch.cuda.current_stream().wait_stream(s)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            logits = self._forward(self.emb1, self.cos1, self.sin1,
                                   self.pos1)
            self.log1.copy_(logits)      # static output (pool-safe)
        self.graph = g
        self.hard_reset()
        if verbose:
            print('[q38] graph captured + cache reset', flush=True)

    # ------------------------------------------------------------------ #
    def hard_reset(self):
        for lay in self.cache.layers:
            cum = getattr(lay, 'cumulative_length', None)
            if cum is not None:
                cum.zero_()
            hps = getattr(lay, 'has_previous_state', None)
            if hps is not None:
                if isinstance(hps, dict):
                    for k in hps:
                        hps[k] = False
                else:
                    lay.has_previous_state = [False] * len(hps)
            # NOTE: conv/rec states are DICTS 鈥?iterate .values()
            for cs in (getattr(lay, 'conv_states', None) or {}).values():
                if isinstance(cs, torch.Tensor):
                    cs.zero_()
            for rs in (getattr(lay, 'recurrent_states', None) or {}).values():
                if isinstance(rs, torch.Tensor):
                    rs.zero_()

    def _set_token(self, token, t):
        emb_w = self._emb_weight()
        self.emb1.copy_(emb_w[token].view(1, 1, self.H).to(
            self.dev, torch.bfloat16))
        self.cos1.copy_(self._cos_all[t].view(1, 1, -1))
        self.sin1.copy_(self._sin_all[t].view(1, 1, -1))
        self.pos1.fill_(t)

    def _emb_weight(self):
        for mod in self.model.modules():
            if type(mod).__name__ == '_CpuEmbed':
                return mod.weight_cpu
        raise RuntimeError('CpuEmbed not found')

    @torch.no_grad()
    def prefill(self, ids):
        """Blocked S=8 prefill; returns last logits [1, 1, V]."""
        emb_w = self._emb_weight()
        last = None
        for i in range(0, len(ids), MAX_BLOCK):
            blk = ids[i:i + MAX_BLOCK]
            S = len(blk)
            if S == MAX_BLOCK:
                emb, cos, sin, pos = self.embB, self.cosB, self.sinB, self.posB
            else:   # tail: carve S-sized views of the block buffers
                emb = self.embB[:, :S]
                cos = self.cosB[:, :S]
                sin = self.sinB[:, :S]
                pos = self.posB[:S]
            emb[0, :S].copy_(emb_w[blk].to(self.dev, torch.bfloat16))
            t0 = i
            idx = torch.arange(t0, t0 + S, device=self.dev)
            cos.copy_(self._cos_all[idx].view(1, S, -1))
            sin.copy_(self._sin_all[idx].view(1, S, -1))
            pos.copy_(idx)
            last = self._forward(emb, cos, sin, pos)
        return last

    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def _stop_ids(self):
        ids = {self.tokenizer.eos_token_id}
        for sym in ('<|im_end|>', '<|endoftext|>'):
            t = self.tokenizer.convert_tokens_to_ids(sym)
            if t is not None and t != self.tokenizer.unk_token_id:
                ids.add(t)
        return ids

    def _gen_tokens(self, ids, max_new_tokens, temperature=0.0):
        self.hard_reset()
        logits = self.prefill(ids)
        nxt = int(logits[:, -1].argmax(-1).item())
        out = [nxt]
        stops = self._stop_ids()
        t = len(ids)
        while len(out) < max_new_tokens and t < self.max_ctx - 1:
            self._set_token(nxt, t)
            self.graph.replay()
            if temperature and temperature > 0:
                probs = F.softmax(
                    self.log1[:, -1].float() / temperature, dim=-1)
                nxt = int(torch.multinomial(probs, 1).item())
            else:
                nxt = int(self.log1[:, -1].argmax(-1).item())
            if nxt in stops:
                break
            out.append(nxt)
            t += 1
        return out

    def generate(self, prompt, max_new_tokens=64, temperature=0.0,
                 do_sample=False, **kw):
        if not do_sample:
            temperature = 0.0
        ids = self.tokenizer(prompt, return_tensors='pt')['input_ids'][0] \
            .tolist()
        out = self._gen_tokens(ids, max_new_tokens, temperature)
        return self.tokenizer.decode(out)

    def stream(self, prompt, max_new_tokens=64, temperature=0.0,
               do_sample=False, **kw):
        if not do_sample:
            temperature = 0.0
        ids = self.tokenizer(prompt, return_tensors='pt')['input_ids'][0] \
            .tolist()
        self.hard_reset()
        logits = self.prefill(ids)
        nxt = int(logits[:, -1].argmax(-1).item())
        stops = self._stop_ids()
        t = len(ids)
        n = 0
        while n < max_new_tokens and t < self.max_ctx - 1:
            self._set_token(nxt, t)
            self.graph.replay()
            if temperature and temperature > 0:
                probs = F.softmax(
                    self.log1[:, -1].float() / temperature, dim=-1)
                nxt = int(torch.multinomial(probs, 1).item())
            else:
                nxt = int(self.log1[:, -1].argmax(-1).item())
            if nxt in stops:
                break
            yield self.tokenizer.decode([nxt])
            n += 1
            t += 1
