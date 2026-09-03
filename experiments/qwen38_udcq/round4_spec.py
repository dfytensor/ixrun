# -*- coding: utf-8 -*-
"""round4: speculative decoding on the graph pipeline — MTP + n-gram dual
draft, graph-verified (g2, M=2).

Flow per iteration (state: last committed token c at position t, logits_last
predicts the next token):
  1. t1 = argmax(logits_last)                                  (uncommitted)
  2. draft d (position t+2): n-gram table hit -> free; else MTP head
     eager (~300MB bf16, cacheless 1-layer)
  3. snapshot conv/rec states (48 layers, ~2ms); replay g2 on [t1, d]
     at positions [t+1, t+2] -> L1 (successor of t1 = ground truth), L2, h[.,1]
  4. accept (argmax(L1)==d): commit t1+d, logits_last=L2, h_last=h[:,1]
     reject: rollback states + cum-=2, replay g1 on [t1], commit t1,
             logits_last=L1 (already known), h_last from g1
"""
import sys, time, gc
sys.setrecursionlimit(20000)
sys.path.insert(0, r'E:\IXRUN')
import pandas
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer
from transformers.cache_utils import StaticCache
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5Attention, Qwen3_5RMSNorm, apply_rotary_pos_emb)

sys.argv = ['x']
import importlib.util
spec = importlib.util.spec_from_file_location(
    'slim', r'E:\IXRUN\experiments\qwen38_udcq\qwen38_slim_resident.py')
slim = importlib.util.module_from_spec(spec)
spec.loader.exec_module(slim)

from ixrun.fla_patch import apply_fla_kernels
apply_fla_kernels()

MAX_CTX = 256


# ---------------- generalized static attention (q_len 1..2) ----------------
def _attn_forward_static(self, hidden_states, position_embeddings,
                         attention_mask=None, past_key_values=None, **kw):
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)
    q_len = hidden_states.shape[1]

    query_states, gate = torch.chunk(
        self.q_proj(hidden_states).view(*input_shape, -1, self.head_dim * 2),
        2, dim=-1)
    gate = gate.reshape(*input_shape, -1)
    query_states = self.q_norm(query_states.view(hidden_shape)).transpose(1, 2)
    key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    k_full, v_full = past_key_values.update(key_states, value_states, self.layer_idx)

    cum = past_key_values.layers[self.layer_idx].cumulative_length
    MAX = k_full.shape[-2]
    ar = torch.arange(MAX, device=k_full.device)
    qr = torch.arange(q_len, device=k_full.device)
    # causal-in-block: query i (0-based in this block) attends kv < cum-q_len+i+1
    keep = ar[None, :] < (cum - q_len + 1 + qr[:, None])      # [q_len, MAX]
    mask = torch.where(keep, 0.0, float('-inf')).to(query_states.dtype)
    mask = mask.view(1, 1, q_len, MAX)

    n_rep = query_states.shape[1] // k_full.shape[1]
    if n_rep > 1:
        k_full = k_full.repeat_interleave(n_rep, dim=1)
        v_full = v_full.repeat_interleave(n_rep, dim=1)
    attn_output = F.scaled_dot_product_attention(
        query_states, k_full, v_full, attn_mask=mask, scale=self.scaling)

    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    attn_output = attn_output * torch.sigmoid(gate)
    return self.o_proj(attn_output), None


Qwen3_5Attention.forward = _attn_forward_static


# ---------------- MTP head (bf16, cacheless; recipe from ixrun/mtp.py) ------
class MTPHeadBf16(nn.Module):
    def __init__(self, dim, lm_head):
        super().__init__()
        self.norm_e = Qwen3_5RMSNorm(dim, eps=1e-6)     # (1+w) variant!
        self.norm_h = Qwen3_5RMSNorm(dim, eps=1e-6)
        self.fc = nn.Linear(2 * dim, dim, bias=False)
        self.norm = Qwen3_5RMSNorm(dim, eps=1e-6)
        self.lm_head = lm_head

    def forward(self, tok_emb, h):
        x = self.fc(torch.cat([self.norm_e(tok_emb), self.norm_h(h)],
                              dim=-1).to(torch.bfloat16))
        return self.lm_head(self.norm(x))


def load_mtp(m, mtp_lm_head):
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5DecoderLayer
    import copy as _copy
    tensors = {}
    import json as _json
    idx = _json.load(open(r'E:\models\Qwen3.8-27B\model.safetensors.index.json'))['weight_map']
    from safetensors import safe_open
    for shard in sorted({idx[k] for k in idx if k.startswith('mtp.')}):
        with safe_open(rf'E:\models\Qwen3.8-27B\{shard}', 'pt') as sf:
            for k in sf.keys():
                if k.startswith('mtp.'):
                    tensors[k] = sf.get_tensor(k).to(torch.bfloat16)

    cfg = m.config.get_text_config() if hasattr(m.config, 'get_text_config') \
        else getattr(m.config, 'text_config', m.config)
    layer_cfg = _copy.deepcopy(cfg)
    layer_cfg.layer_types = ['full_attention']
    layer = Qwen3_5DecoderLayer(layer_cfg, layer_idx=0).to(torch.bfloat16)

    head = MTPHeadBf16(cfg.hidden_size, mtp_lm_head).to(torch.bfloat16)
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
    head.layer = layer
    missing, unexpected = head.load_state_dict(sd, strict=False)
    real_missing = [k for k in missing
                    if not (k.startswith('layer.') or k == 'fc.weight'
                            or k.startswith('lm_head.'))]  # lm_head=UDCQ bufs
    assert not real_missing, f'mtp missing {real_missing[:5]}'
    return head.cuda().eval()


# ---------------------------------------------------------------------------- #
@torch.no_grad()
def main():
    tok = AutoTokenizer.from_pretrained(r'E:\models\Qwen3.8-27B')
    print('deploying slim...', flush=True)
    m = slim.load_model()
    slim.deploy_slim(m, verbose=True)
    m.eval()

    tm = m.model
    if hasattr(tm, 'language_model'):
        tm = tm.language_model
    layers = tm.layers
    final_norm = tm.norm
    rotary = tm.rotary_emb
    H = tm.config.hidden_size
    dev = 'cuda'

    pos_all = torch.arange(MAX_CTX, device=dev).unsqueeze(0)
    with torch.no_grad():
        dummy = torch.zeros(1, MAX_CTX, H, dtype=torch.bfloat16, device=dev)
        cos_all, sin_all = rotary(dummy, pos_all)
    if cos_all.dim() == 4:
        cos_all = cos_all[:, :, 0]
    if cos_all.dim() == 3:
        cos_all, sin_all = cos_all[0], sin_all[0]

    cache = StaticCache(config=m.config, max_cache_len=MAX_CTX)

    emb_w = {}
    def embed(tok_id):
        if tok_id not in emb_w:
            for name, mod in m.named_modules():
                if type(mod).__name__ == 'CpuEmbed':
                    emb_w[tok_id] = mod.weight_cpu[tok_id].view(1, 1, H)
                    break
        return emb_w[tok_id]

    # static buffers: M=1 and M=2
    emb1 = torch.zeros(1, 1, H, dtype=torch.bfloat16, device=dev)
    cos1 = torch.zeros(1, 1, cos_all.shape[-1], dtype=cos_all.dtype, device=dev)
    sin1 = torch.zeros_like(cos1)
    pos1 = torch.zeros(1, dtype=torch.long, device=dev)
    emb2 = torch.zeros(1, 2, H, dtype=torch.bfloat16, device=dev)
    cos2 = torch.zeros(1, 2, cos_all.shape[-1], dtype=cos_all.dtype, device=dev)
    sin2 = torch.zeros_like(cos2)
    pos2 = torch.zeros(2, dtype=torch.long, device=dev)

    @torch.no_grad()
    def run(emb, cos, sin, pos):
        h = emb
        for layer in layers:
            h = layer(h, position_embeddings=(cos, sin), attention_mask=None,
                      position_ids=pos.view(1, -1), past_key_values=cache)
            if isinstance(h, tuple):
                h = h[0]
        h = final_norm(h)
        return h, m.lm_head(h)

    def hard_reset():
        for lay in cache.layers:
            cum = getattr(lay, 'cumulative_length', None)
            if cum is not None:
                cum.zero_()
            if hasattr(lay, 'has_previous_state'):
                hps = lay.has_previous_state
                if isinstance(hps, dict):
                    for k in hps:
                        hps[k] = False
                else:
                    lay.has_previous_state = [False] * len(hps)
            for cs in getattr(lay, 'conv_states', []) or []:
                if isinstance(cs, torch.Tensor):
                    cs.zero_()
            for rs in getattr(lay, 'recurrent_states', []) or []:
                if isinstance(rs, torch.Tensor):
                    rs.zero_()

    def snap_light():
        """conv+recurrent states only (full-attn KV rollback = cum rewind)."""
        return [([c.clone() if isinstance(c, torch.Tensor) else c
                  for c in getattr(l, 'conv_states', []) or []],
                 [r.clone() if isinstance(r, torch.Tensor) else r
                  for r in getattr(l, 'recurrent_states', []) or []],
                 l.cumulative_length.clone()
                 if getattr(l, 'cumulative_length', None) is not None else None)
                for l in cache.layers]

    def roll_back(snap):
        for l, (cs, rs, cum) in zip(cache.layers, snap):
            if cum is not None:
                l.cumulative_length.copy_(cum)
            dst_cs = getattr(l, 'conv_states', []) or []
            for a, b in zip(dst_cs, cs):
                if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
                    a.copy_(b)
            dst_rs = getattr(l, 'recurrent_states', []) or []
            for a, b in zip(dst_rs, rs):
                if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
                    a.copy_(b)

    # capture g1 (M=1) and g2 (M=2) on a fresh cache
    print('capturing g1/g2...', flush=True)
    for buf in (emb1, cos1, sin1):
        buf.zero_()
    emb2.zero_(); cos2.zero_(); sin2.zero_()
    pos1.fill_(0); pos2.fill_(0)
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            run(emb1, cos1, sin1, pos1)
            run(emb2, cos2, sin2, pos2)
    torch.cuda.current_stream().wait_stream(s)

    g1 = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g1):
        h1_s, log1_s = run(emb1, cos1, sin1, pos1)
    g2 = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g2):
        h2_s, log2_s = run(emb2, cos2, sin2, pos2)
    hard_reset()
    print('graphs captured + reset.', flush=True)

    # MTP head (shares the UDCQ lm_head)
    mtp = load_mtp(m, m.lm_head)
    print('mtp head loaded (bf16)', flush=True)

    PROMPTS = ['def quick_sort(arr):',
               'The theory of relativity states that',
               '北京最值得游览的三个景点是']

    for prompt in PROMPTS:
        ids = tok(prompt, return_tensors='pt')['input_ids'][0].tolist()
        # prefill eager, token by token; keep the LAST step's outputs
        h_last = logits_last = None
        for i, tid in enumerate(ids):
            emb1.copy_(embed(tid).to(dev, torch.bfloat16))
            cos1.copy_(cos_all[i].view(1, 1, -1))
            sin1.copy_(sin_all[i].view(1, 1, -1))
            pos1.fill_(i)
            h_last, logits_last = run(emb1, cos1, sin1, pos1)
        h_last = h_last.clone()
        logits_last = logits_last.clone()

        # n-gram table from prompt (extend with history as we go)
        from collections import defaultdict
        bigram = defaultdict(list)
        for i in range(2, len(ids)):
            bigram[(ids[i - 2], ids[i - 1])].append(ids[i])

        gen = []
        n_acc = n_rej = n_mtp = n_ng = 0
        t = len(ids)
        t0 = time.time()
        N = 40
        while len(gen) < N and t < MAX_CTX - 4:
            t1 = logits_last[:, -1].argmax(-1).item()
            # --- draft: n-gram first, else MTP ---
            d, src = None, None
            hist = (ids[-2:] + gen)[-2:]
            cand = bigram.get(tuple(hist))
            if cand:
                d, src = cand[-1], 'ng'
            if d is None:
                emb_t = embed(t1).to(dev, torch.bfloat16)
                d = mtp(emb_t, h_last)[:, -1].argmax(-1).item()
                src = 'mtp'
            if src == 'ng':
                n_ng += 1
            else:
                n_mtp += 1

            # --- verify [t1, d] via g2 ---
            snap = snap_light()
            emb2[:, 0].copy_(embed(t1).to(dev, torch.bfloat16).view(1, H))
            emb2[:, 1].copy_(embed(d).to(dev, torch.bfloat16).view(1, H))
            cos2[:, 0].copy_(cos_all[t].view(1, -1))
            cos2[:, 1].copy_(cos_all[t + 1].view(1, -1))
            sin2[:, 0].copy_(sin_all[t].view(1, -1))
            sin2[:, 1].copy_(sin_all[t + 1].view(1, -1))
            pos2[0] = t
            pos2[1] = t + 1
            g2.replay()
            t2 = log2_s[:, 0].argmax(-1).item()

            if t2 == d:
                # accept both
                gen.extend([t1, d])
                bigram[tuple(hist)].append(d)
                h_last = h2_s[:, 1:2].clone()
                logits_last = log2_s[:, 1:2].clone()
                t += 2
                n_acc += 1
                if d == tok.eos_token_id:
                    break
            else:
                # reject: roll back, commit t1 only via g1
                roll_back(snap)
                emb1.copy_(embed(t1).to(dev, torch.bfloat16))
                cos1.copy_(cos_all[t].view(1, 1, -1))
                sin1.copy_(sin_all[t].view(1, 1, -1))
                pos1.fill_(t)
                g1.replay()
                gen.append(t1)
                bigram[tuple(hist)].append(t1)
                h_last = h1_s.clone()
                logits_last = log1_s.clone()      # == log2_s[:,0] mod backend
                t += 1
                n_rej += 1
                if t1 == tok.eos_token_id:
                    break
        torch.cuda.synchronize()
        dt = time.time() - t0
        txt = tok.decode(gen)
        print(f'\n[{len(gen)/dt:.2f} tok/s] {prompt!r} '
              f'(acc {n_acc} rej {n_rej} | drafts: ng {n_ng} mtp {n_mtp})\n'
              f'  -> {txt[:140]!r}', flush=True)
        hard_reset()

    print(f'\npeak GPU = {torch.cuda.max_memory_allocated()/1e9:.2f}GB', flush=True)


if __name__ == '__main__':
    main()
