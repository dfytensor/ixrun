# -*- coding: utf-8 -*-
"""Q38SpecEngine: Qwen3.8-27B speculative decoding with the engine
interface (tokenizer / generate / stream) so CLI chat/generate and the
OpenAI-compatible server work unchanged:

    python -m ixrun.cli chat --mode udcq-spec \
        --cache experiments/qwen38_udcq/q38_blob.pt

Queue architecture + true-h MTP seeds (see round4b_bisect.py): per
iteration one chain graph (selected by pending length) -> T=4 verify
graph -> decisions; partial accepts roll back and re-queue with drafts
seeded from the REAL main-model hidden states still live in out_h4.
~45-55ms/iter, E~2.2-2.7 -> 53/45/32 tok/s (en/code/zh, UDCQ_CUDA_GEMV=1).

Pitfalls baked in (AGENTS.md): dict conv/rec state iteration, transpose
before reshape, static graph outputs, VRAM < 24GB.
"""
from __future__ import annotations

import gc
import json
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import QWEN38_PATH
from .q38_graph import _apply_static_attention, _build_from_blob

__all__ = ["Q38SpecEngine"]

MAX_CTX = 256

from transformers.models.qwen3_5.modeling_qwen3_5 import (  # noqa: E402
    Qwen3_5Attention, Qwen3_5RMSNorm, apply_rotary_pos_emb)

_ORIG_ATTN_FWD = Qwen3_5Attention.forward    # saved BEFORE static patch


class _MTPHead(nn.Module):
    def __init__(self, dim, layer, rotary, lm_head):
        super().__init__()
        self.norm_e = Qwen3_5RMSNorm(dim, eps=1e-6)
        self.norm_h = Qwen3_5RMSNorm(dim, eps=1e-6)
        self.fc = nn.Linear(2 * dim, dim, bias=False)
        self.layer = layer          # full-attn layer w/ ORIGINAL fwd
        self.rotary = rotary
        self.norm = Qwen3_5RMSNorm(dim, eps=1e-6)
        self.lm_head = lm_head

    @torch.no_grad()
    def forward2(self, tok_emb, h, pos):
        """(logits, normed_z); normed_z chains multi-step drafts."""
        cos, sin = self.rotary(h, pos)
        x = self.fc(torch.cat([self.norm_e(tok_emb), self.norm_h(h)],
                              dim=-1).to(torch.bfloat16))
        z = self.layer(x, attention_mask=None, position_ids=pos,
                       position_embeddings=(cos, sin), use_cache=False)
        if isinstance(z, tuple):
            z = z[0]
        nz = self.norm(z)
        return self.lm_head(nz), nz

    @torch.no_grad()
    def forward(self, tok_emb, h, pos):
        return self.forward2(tok_emb, h, pos)[0]


def _load_mtp(model, model_path):
    """Rebuild the MTP head from checkpoint weights (params excluded by
    the HF loader). lm_head attached without being a child module (bf16
    .to() must not touch the UdcqLinear f16 buffers)."""
    import copy as _copy
    from safetensors import safe_open
    from transformers.models.qwen3_5.modeling_qwen3_5 import (
        Qwen3_5DecoderLayer)

    tensors = {}
    idx = json.load(open(
        rf'{model_path}\model.safetensors.index.json'))['weight_map']
    for shard in sorted({idx[k] for k in idx if k.startswith('mtp.')}):
        with safe_open(rf'{model_path}\{shard}', 'pt') as sf:
            for k in sf.keys():
                if k.startswith('mtp.'):
                    tensors[k] = sf.get_tensor(k).to(torch.bfloat16)
    cfg = model.config.get_text_config()
    layer_cfg = _copy.deepcopy(cfg)
    layer_cfg.layer_types = ['full_attention']
    layer = Qwen3_5DecoderLayer(layer_cfg, layer_idx=0).to(torch.bfloat16)
    layer.self_attn.forward = _ORIG_ATTN_FWD.__get__(
        layer.self_attn, Qwen3_5Attention)
    tm = model.model
    if hasattr(tm, 'language_model'):
        tm = tm.language_model
    head = _MTPHead(cfg.hidden_size, layer, tm.rotary_emb, None) \
        .to(torch.bfloat16)
    head.layer = layer
    object.__setattr__(head, 'lm_head', model.lm_head)
    sd = {}
    for k, t in tensors.items():
        s = k[4:]
        if s.startswith('layers.0.'):
            sd['layer.' + s[9:]] = t
        elif s == 'fc.weight':
            sd['fc.weight'] = t
        elif s == 'norm.weight':
            sd['norm.weight'] = t
        elif s == 'pre_fc_norm_embedding.weight':
            sd['norm_e.weight'] = t
        elif s == 'pre_fc_norm_hidden.weight':
            sd['norm_h.weight'] = t
    missing, _ = head.load_state_dict(sd, strict=False)
    real = [k for k in missing
            if not (k.startswith('layer.') or k == 'fc.weight'
                    or k.startswith('lm_head.'))]
    assert not real, f'{real[:4]}'
    return head.cuda().eval()


class Q38SpecEngine:
    """UDCQ 27B + queue-architecture speculative decode (~50 tok/s)."""

    def __init__(self, model, mtp, tokenizer, max_ctx=MAX_CTX,
                 verbose=True):
        from transformers import AutoTokenizer
        from transformers.cache_utils import StaticCache

        self.tokenizer = tokenizer or AutoTokenizer.from_pretrained(
            QWEN38_PATH)
        self.max_ctx = max_ctx
        self.model = model
        model.eval()
        self.mtp = mtp
        tm = model.model
        if hasattr(tm, 'language_model'):
            tm = tm.language_model
        self.layers = tm.layers
        self.final_norm = tm.norm
        self.H = tm.config.hidden_size
        self.V = model.lm_head.out_features
        dev = self.dev = 'cuda'

        pos_all = torch.arange(max_ctx, device=dev).unsqueeze(0)
        with torch.no_grad():
            dummy = torch.zeros(1, max_ctx, self.H, dtype=torch.bfloat16,
                                device=dev)
            cos_all, sin_all = tm.rotary_emb(dummy, pos_all)
        if cos_all.dim() == 4:
            cos_all = cos_all[:, :, 0]
        if cos_all.dim() == 3:
            cos_all, sin_all = cos_all[0], sin_all[0]
        self._cos_all, self._sin_all = cos_all, sin_all

        self.cache = StaticCache(config=model.config,
                                 max_cache_len=max_ctx)
        cd = cos_all.shape[-1]
        self.emb1 = torch.zeros(1, 1, self.H, dtype=torch.bfloat16,
                                device=dev)
        self.cos1 = torch.zeros(1, 1, cd, dtype=cos_all.dtype, device=dev)
        self.sin1 = torch.zeros_like(self.cos1)
        self.pos1 = torch.zeros(1, dtype=torch.long, device=dev)
        self.emb4 = torch.zeros(1, 4, self.H, dtype=torch.bfloat16,
                                device=dev)
        self.cos4 = torch.zeros(1, 4, cd, dtype=cos_all.dtype, device=dev)
        self.sin4 = torch.zeros_like(self.cos4)
        self.pos4 = torch.zeros(4, dtype=torch.long, device=dev)
        # greedy-graph buffers (sampling fallback when knobs are active)
        self.emb1 = torch.zeros(1, 1, self.H, dtype=torch.bfloat16,
                                device=dev)
        self.cos1 = torch.zeros(1, 1, cd, dtype=cos_all.dtype, device=dev)
        self.sin1 = torch.zeros_like(self.cos1)
        self.pos1 = torch.zeros(1, dtype=torch.long, device=dev)
        self.log1 = torch.zeros(1, 1, self.V, dtype=torch.bfloat16,
                                device=dev)

        # int8 GPU embedding table for in-graph gathers (~1.27GB)
        emb_f = self._emb_weight().float()
        emb_s = (emb_f.abs().amax(dim=1) / 127.0).clamp_min(1e-12)
        self.emb_i8 = (emb_f / emb_s.unsqueeze(1)).round().to(torch.int8)
        self.emb_i8 = self.emb_i8.cuda()
        self.s_g = emb_s.cuda().unsqueeze(1)
        del emb_f
        gc.collect()
        torch.cuda.empty_cache()

        # snap/rollback streams (conv/rec are DICTS — iterate .values()).
        # COLLECTED LAZILY after the cache has been used: the dict values
        # start as None and only become tensors after the first forward —
        # collecting earlier silently yields empty streams and rollback
        # becomes a no-op (state drift -> text degeneration).
        self._dsts, self._srcs = [], []
        self._static_buffers()
        self._capture(verbose=verbose)

    def _collect_snap(self):
        if self._srcs:
            return
        for lay in self.cache.layers:
            for c in (getattr(lay, 'conv_states', None) or {}).values():
                if isinstance(c, torch.Tensor):
                    self._dsts.append(c)
                    self._srcs.append(torch.empty_like(c))
            for r in (getattr(lay, 'recurrent_states', None) or {}).values():
                if isinstance(r, torch.Tensor):
                    self._dsts.append(r)
                    self._srcs.append(torch.empty_like(r))
            cum = getattr(lay, 'cumulative_length', None)
            if cum is not None:
                self._dsts.append(cum)
                self._srcs.append(torch.empty_like(cum))

    # static graph outputs
    def _static_buffers(self):
        dev = self.dev
        self.out_h4 = torch.zeros(1, 4, self.H, dtype=torch.bfloat16,
                                  device=dev)
        self.out_l4 = torch.zeros(1, 4, self.V, dtype=torch.bfloat16,
                                  device=dev)
        self.a_buf = torch.zeros(4, dtype=torch.long, device=dev)
        self.dec_gpu = torch.zeros(6, dtype=torch.long, device=dev)
        self.tok_out = torch.zeros(4, dtype=torch.long, device=dev)
        self.mtp_h_buf = torch.zeros(1, 1, self.H, dtype=torch.bfloat16,
                                     device=dev)
        self.mtp_pos_buf = torch.zeros(1, 1, dtype=torch.long, device=dev)
        self.pin_t = torch.zeros(1, dtype=torch.long, pin_memory=True)
        self.t_gpu = torch.zeros(1, dtype=torch.long, device=dev)
        self.ar4 = torch.arange(4, device=dev)
        self._ev = torch.cuda.Event()

    # ------------------------------------------------------------------ #
    def emb_rows(self, ids_t):
        q = F.embedding(ids_t, self.emb_i8)
        sc = F.embedding(ids_t, self.s_g)
        return (q.float() * sc).to(torch.bfloat16)

    @classmethod
    def from_blob(cls, blob_path, model_path=QWEN38_PATH, tokenizer=None,
                  max_ctx=MAX_CTX, verbose=True):
        from .fla_patch import apply_fla_kernels
        from .gdn_seq_patch import apply_gdn_sequential_patch

        apply_fla_kernels()
        apply_gdn_sequential_patch()
        _apply_static_attention()
        m = _build_from_blob(blob_path, model_path, verbose=verbose)
        mtp = _load_mtp(m, model_path)
        if verbose:
            print('[q38-spec] MTP head loaded', flush=True)
        return cls(m, mtp, tokenizer, max_ctx=max_ctx, verbose=verbose)

    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def _step(self, emb, cos, sin, pos):
        """Single-token step (prefill/decode): returns normed h, logits."""
        h = emb
        for layer in self.layers:
            h = layer(h, position_embeddings=(cos, sin),
                      attention_mask=None, position_ids=pos.view(1, -1),
                      past_key_values=self.cache)
            if isinstance(h, tuple):
                h = h[0]
        h = self.final_norm(h)
        return h, self.model.lm_head(h)

    @torch.no_grad()
    def _forward4(self):
        h = self.emb4
        for layer in self.layers:
            h = layer(h, position_embeddings=(self.cos4, self.sin4),
                      attention_mask=None,
                      position_ids=self.pos4.view(1, -1),
                      past_key_values=self.cache)
            if isinstance(h, tuple):
                h = h[0]
        h = self.final_norm(h)
        return h, self.model.lm_head(h)

    def _emb_weight(self):
        for mod in self.model.modules():
            if type(mod).__name__ == '_CpuEmbed':
                return mod.weight_cpu
        raise RuntimeError('CpuEmbed not found')

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
            for cs in (getattr(lay, 'conv_states', None) or {}).values():
                if isinstance(cs, torch.Tensor):
                    cs.zero_()
            for rs in (getattr(lay, 'recurrent_states', None) or {}).values():
                if isinstance(rs, torch.Tensor):
                    rs.zero_()

    def _fast_snap(self):
        torch._foreach_copy_(self._srcs, self._dsts)

    def _fast_rollback(self):
        torch._foreach_copy_(self._dsts, self._srcs)

    def _capture(self, verbose=True):
        pool = torch.cuda.graph_pool_handle()

        def prep_body():
            self.t_gpu.copy_(self.pin_t, non_blocking=True)
            idx = self.t_gpu + self.ar4
            self.emb4.copy_(self.emb_rows(self.tok_out).view(1, 4, self.H))
            self.cos4.copy_(torch.index_select(
                self._cos_all, 0, idx).view(1, 4, -1))
            self.sin4.copy_(torch.index_select(
                self._sin_all, 0, idx).view(1, 4, -1))
            self.pos4.copy_(idx)
            torch._foreach_copy_(self._srcs, self._dsts)   # fast_snap

        def make_chain_body(p):
            n_drafts = 4 - p
            root_slot = p - 1

            def body():
                if p == 1:
                    self.tok_out[0].copy_(self.a_buf[3])
                    h = self.mtp_h_buf
                else:
                    self.tok_out[root_slot].copy_(self.a_buf[p - 2])
                    self.mtp_h_buf.copy_(self.out_h4[:, p - 2:p - 1])
                    h = self.mtp_h_buf
                tok = self.tok_out[root_slot]
                for s in range(n_drafts):
                    e = self.emb_rows(tok.view(1)).view(1, 1, self.H)
                    lg, h = self.mtp.forward2(e, h, self.mtp_pos_buf)
                    tok = lg[:, -1].argmax()
                    self.tok_out[root_slot + 1 + s].copy_(tok)
                prep_body()
            return body

        def g4dec_body():
            h, lg = self._forward4()
            self.out_h4.copy_(h)
            self.out_l4.copy_(lg)
            a0 = self.out_l4[0, 0].argmax()
            a1 = self.out_l4[0, 1].argmax()
            a2 = self.out_l4[0, 2].argmax()
            a3 = self.out_l4[0, 3].argmax()
            m1 = a0 == self.tok_out[1]
            m2 = (a1 == self.tok_out[2]) & m1
            m3 = (a2 == self.tok_out[3]) & m2
            L = 1 + m1.to(torch.long) + m2.to(torch.long) \
                + m3.to(torch.long)
            self.dec_gpu[0] = L
            self.dec_gpu[1] = self.tok_out[0]
            self.dec_gpu[2] = self.tok_out[1]
            self.dec_gpu[3] = self.tok_out[2]
            self.dec_gpu[4] = self.tok_out[3]
            self.a_buf[0].copy_(a0)
            self.a_buf[1].copy_(a1)
            self.a_buf[2].copy_(a2)
            self.a_buf[3].copy_(a3)

        bodies = {p: make_chain_body(p) for p in (1, 2, 3, 4)}
        if verbose:
            print('[q38-spec] capturing g1/g_cp[1..4]/g4dec...', flush=True)
        # CRITICAL: seed the cache FIRST so the T=4 verify captures the
        # seeded per-token GDN branch. Capturing on a fresh cache fixes
        # the PREFILL branch (chunk kernel) into the graph — replays then
        # corrupt state (text degenerates after ~10-20 tokens).
        emb_w = self._emb_weight()
        for i in range(12):
            self.emb1.copy_(emb_w[5 + i].view(1, 1, self.H).to(
                self.dev, torch.bfloat16))
            self.cos1.copy_(self._cos_all[i].view(1, 1, -1))
            self.sin1.copy_(self._sin_all[i].view(1, 1, -1))
            self.pos1.fill_(i)
            self._step(self.emb1, self.cos1, self.sin1, self.pos1)
        self._collect_snap()        # conv/rec now materialized as tensors

        # greedy single-token graph first (sampling fallback; capture
        # ORDER matters — capturing it after the T=4 graphs hung WDDM)
        def g1_body():
            h, lg = self._step(self.emb1, self.cos1, self.sin1, self.pos1)
            self.log1.copy_(lg)

        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                g1_body()
        torch.cuda.current_stream().wait_stream(s)
        self.g1 = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.g1, pool=pool):
            g1_body()

        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                for p in (1, 2, 3, 4):
                    bodies[p]()
                g4dec_body()
        torch.cuda.current_stream().wait_stream(s)
        self.g_cp = {}
        for p in (1, 2, 3, 4):
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g, pool=pool):
                bodies[p]()
            self.g_cp[p] = g
        self.g4dec = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.g4dec, pool=pool):
            g4dec_body()
        self.hard_reset()
        if verbose:
            print('[q38-spec] graphs captured; cache reset', flush=True)

    # ------------------------------------------------------------------ #
    def _prefill(self, ids):
        """Per-token eager prefill; returns (h_norm, logits) of last."""
        emb_w = self._emb_weight()
        h = logits = None
        for i, tid in enumerate(ids):
            self.emb1.copy_(emb_w[tid].view(1, 1, self.H).to(
                self.dev, torch.bfloat16))
            self.cos1.copy_(self._cos_all[i].view(1, 1, -1))
            self.sin1.copy_(self._sin_all[i].view(1, 1, -1))
            self.pos1.fill_(i)
            h, logits = self._step(self.emb1, self.cos1, self.sin1,
                                   self.pos1)
        return h, logits

    def _stop_ids(self):
        """eos + chat-template end (<|im_end|>) token ids."""
        ids = {self.tokenizer.eos_token_id}
        for sym in ('<|im_end|>', '<|endoftext|>'):
            t = self.tokenizer.convert_tokens_to_ids(sym)
            if t is not None and t != self.tokenizer.unk_token_id:
                ids.add(t)
        return ids

    def _spec_iter(self, ids, max_new_tokens):
        """Generator: yields batches of newly committed tokens."""
        dec_pin = torch.zeros(6, dtype=torch.long, pin_memory=True)
        self.hard_reset()
        h_last, logits_last = self._prefill(ids)
        root = int(logits_last[:, -1].argmax(-1).item())
        self.a_buf[3] = root
        self.mtp_h_buf.copy_(h_last)
        pend_cpu = 1
        stops = self._stop_ids()
        gen = []
        t = len(ids)
        while len(gen) < max_new_tokens and t < self.max_ctx - 6:
            self.pin_t[0] = t - (pend_cpu - 1)
            self.g_cp[pend_cpu].replay()
            self.g4dec.replay()
            dec_pin.copy_(self.dec_gpu, non_blocking=True)
            self._ev.record()
            while not self._ev.query():
                pass
            L = int(dec_pin[0])
            tv = [int(dec_pin[1]), int(dec_pin[2]),
                  int(dec_pin[3]), int(dec_pin[4])]
            committed = tv[pend_cpu - 1: max(L, pend_cpu - 1)]
            gen.extend(committed)
            t += len(committed)
            if L == 4:
                pend_cpu = 1
                self.mtp_h_buf.copy_(self.out_h4[:, 3:4])
            else:
                self._fast_rollback()
                pend_cpu = L + 1
            yield committed
            if stops & set(committed):
                break

    def generate(self, prompt, max_new_tokens=64, temperature=0.0,
                 do_sample=False, top_p=1.0, top_k=0,
                 repetition_penalty=1.0, **kw):
        from .sampling import needs_sampling

        if not do_sample:
            temperature = 0.0
        top_p = float(kw.pop('top_p', top_p))
        top_k = int(kw.pop('top_k', top_k))
        repetition_penalty = float(kw.pop('repetition_penalty',
                                          repetition_penalty))
        ids = self.tokenizer(prompt, return_tensors='pt')['input_ids'][0] \
            .tolist()
        if needs_sampling(do_sample, temperature, top_p, top_k,
                          repetition_penalty):
            out = self._sample_tokens(ids, max_new_tokens, temperature,
                                      top_p, top_k, repetition_penalty)
        else:
            out = []
            for batch in self._spec_iter(ids, max_new_tokens):
                out.extend(batch)
        return self.tokenizer.decode(out)

    def stream(self, prompt, max_new_tokens=64, temperature=0.0,
               do_sample=False, top_p=1.0, top_k=0,
               repetition_penalty=1.0, **kw):
        from .sampling import needs_sampling

        if not do_sample:
            temperature = 0.0
        top_p = float(kw.pop('top_p', top_p))
        top_k = int(kw.pop('top_k', top_k))
        repetition_penalty = float(kw.pop('repetition_penalty',
                                          repetition_penalty))
        ids = self.tokenizer(prompt, return_tensors='pt')['input_ids'][0] \
            .tolist()
        if needs_sampling(do_sample, temperature, top_p, top_k,
                          repetition_penalty):
            toks = self._sample_tokens(ids, max_new_tokens, temperature,
                                       top_p, top_k, repetition_penalty)
            for v in toks:
                yield self.tokenizer.decode([v])
            return
        for batch in self._spec_iter(ids, max_new_tokens):
            yield self.tokenizer.decode(batch)

    @torch.no_grad()
    def _sample_tokens(self, ids, max_new_tokens, temperature, top_p,
                       top_k, repetition_penalty):
        """Greedy-graph decode with CPU-side sampling (no speculation —
        verification requires argmax)."""
        from .sampling import sample_token

        self.hard_reset()
        h_last, logits_last = self._prefill(ids)
        nxt = int(logits_last[:, -1].argmax(-1).item())
        stops = self._stop_ids()
        out = [nxt]
        emb_w = self._emb_weight()
        t = len(ids)
        while len(out) < max_new_tokens and t < self.max_ctx - 1:
            self.emb1.copy_(emb_w[nxt].view(1, 1, self.H).to(
                self.dev, torch.bfloat16))
            self.cos1.copy_(self._cos_all[t].view(1, 1, -1))
            self.sin1.copy_(self._sin_all[t].view(1, 1, -1))
            self.pos1.fill_(t)
            self.g1.replay()
            nxt = sample_token(self.log1[0, 0].float().cpu(),
                               temperature=temperature,
                               top_p=top_p, top_k=top_k,
                               repetition_penalty=repetition_penalty,
                               past_ids=out)
            if nxt in stops:
                break
            out.append(nxt)
            t += 1
        return out
