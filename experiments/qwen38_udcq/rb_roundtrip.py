# -*- coding: utf-8 -*-
"""Rollback round-trip test: prefill -> snap -> restore (no g2) -> g1 decode
vs prefill -> g1 decode. If they differ, snap/restore itself is lossy."""
import pandas  # MUST be before torch
import sys, time, gc, json
sys.setrecursionlimit(20000)
sys.path.insert(0, r'E:\IXRUN')
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoConfig
from transformers.cache_utils import StaticCache
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5Attention, apply_rotary_pos_emb)
from ixrun.fla_patch import apply_fla_kernels
from ixrun.gdn_seq_patch import apply_gdn_sequential_patch
apply_fla_kernels()
apply_gdn_sequential_patch(verbose=False)
_ORIG = Qwen3_5Attention.forward
MAX_CTX = 256
BLOB = r'E:\IXRUN\experiments\qwen38_udcq\q38_blob.pt'


def _static_fwd(self, hidden_states, position_embeddings, attention_mask=None,
                past_key_values=None, **kw):
    if past_key_values is None:
        return _ORIG(self, hidden_states, position_embeddings,
                     attention_mask=attention_mask, past_key_values=None, **kw)
    ish = hidden_states.shape[:-1]
    hsh = (*ish, -1, self.head_dim)
    q_len = hidden_states.shape[1]
    q, gate = torch.chunk(
        self.q_proj(hidden_states).view(*ish, -1, self.head_dim * 2), 2, -1)
    gate = gate.reshape(*ish, -1)
    q = self.q_norm(q.view(hsh)).transpose(1, 2)
    k = self.k_norm(self.k_proj(hidden_states).view(hsh)).transpose(1, 2)
    v = self.v_proj(hidden_states).view(hsh).transpose(1, 2)
    cos, sin = position_embeddings
    q, k = apply_rotary_pos_emb(q, k, cos, sin)
    kf, vf = past_key_values.update(k, v, self.layer_idx)
    cum = past_key_values.layers[self.layer_idx].cumulative_length
    MAXn = kf.shape[-2]
    ar = torch.arange(MAXn, device=kf.device)
    keep = ar < (cum - q_len + 1 +
                 torch.arange(q_len, device=kf.device)[:, None])
    mask = torch.where(keep, 0.0, float('-inf')).to(q.dtype).view(1, 1, q_len, MAXn)
    n_rep = q.shape[1] // kf.shape[1]
    if n_rep > 1:
        kf = kf.repeat_interleave(n_rep, 1)
        vf = vf.repeat_interleave(n_rep, 1)
    o = F.scaled_dot_product_attention(q, kf, vf, attn_mask=mask,
                                       scale=self.scaling)
    o = o.reshape(*ish, -1).contiguous() * torch.sigmoid(gate)
    return self.o_proj(o), None


Qwen3_5Attention.forward = _static_fwd


@torch.no_grad()
def main():
    from transformers import AutoModelForCausalLM
    from accelerate import init_empty_weights
    from ixrun.udcq import UdcqLinear, UDCQ_G
    from ixrun.linear import _set_parent_child
    from ixrun.config import QWEN38_PATH

    blob = torch.load(BLOB, map_location='cpu', mmap=True, weights_only=False)
    cfg = AutoConfig.from_pretrained(QWEN38_PATH, trust_remote_code=True)
    with init_empty_weights():
        m = AutoModelForCausalLM.from_config(cfg, trust_remote_code=True)
    for name, mod in list(m.named_modules()):
        if isinstance(mod, nn.Linear) and name in blob['layers']:
            e = blob['layers'][name]
            packed = {'g': UDCQ_G, 'out_f': mod.out_features,
                      'in_f': mod.in_features,
                      'N': mod.out_features * mod.in_features,
                      'idx': e['idx'], 'scale': e['scale'],
                      'sign_packed': e['sign'],
                      'codebook': blob['codebook'], 'bits_per_weight': 6.0}
            _set_parent_child(m, name, UdcqLinear(packed, bias=None,
                                                  cache='stream'))
    idx = json.load(open(r'E:\models\Qwen3.8-27B\model.safetensors.index.json'))['weight_map']

    def ck(name):
        return name if name in idx else \
            (name.replace('model.', 'model.language_model.', 1)
             if name.replace('model.', 'model.language_model.', 1) in idx else None)

    from safetensors import safe_open as so
    params = dict(m.named_parameters())
    for name, p in params.items():
        if not p.numel() or not p.is_meta:
            continue
        key = ck(name)
        if key is None:
            continue
        with so(rf'E:\models\Qwen3.8-27B\{idx[key]}', 'pt') as sf:
            t = sf.get_tensor(key)
        parts = name.split('.')
        parent = m
        for q_ in parts[:-1]:
            parent = getattr(parent, q_)
        parent._parameters[parts[-1]] = torch.nn.Parameter(
            t.cuda(), requires_grad=False)
        del t
    gc.collect(); torch.cuda.empty_cache()

    tm = m.model
    if hasattr(tm, 'language_model'):
        tm = tm.language_model
    H = tm.config.hidden_size
    dev = 'cuda'
    cache = StaticCache(config=m.config, max_cache_len=MAX_CTX)

    pos_all = torch.arange(MAX_CTX, device=dev).unsqueeze(0)
    dummy = torch.zeros(1, MAX_CTX, H, dtype=torch.bfloat16, device=dev)
    cos_all, sin_all = tm.rotary_emb(dummy, pos_all)
    if cos_all.dim() == 4:
        cos_all = cos_all[:, :, 0]
    if cos_all.dim() == 3:
        cos_all, sin_all = cos_all[0], sin_all[0]

    emb1 = torch.zeros(1, 1, H, dtype=torch.bfloat16, device=dev)
    cos1 = torch.zeros(1, 1, cos_all.shape[-1], dtype=cos_all.dtype, device=dev)
    sin1 = torch.zeros_like(cos1)
    pos1 = torch.zeros(1, dtype=torch.long, device=dev)

    def run(emb, cos, sin, pos):
        h = emb
        for layer in tm.layers:
            h = layer(h, position_embeddings=(cos, sin), attention_mask=None,
                      position_ids=pos.view(1, -1), past_key_values=cache)
            if isinstance(h, tuple):
                h = h[0]
        h = tm.norm(h)
        return h, m.lm_head(h)

    def embed1(tid):
        return blob['embed'][tid].view(1, 1, H).to(dev, torch.bfloat16)

    def hard_reset():
        for lay in cache.layers:
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
            for cs in getattr(lay, 'conv_states', []) or []:
                if isinstance(cs, torch.Tensor):
                    cs.zero_()
            for rs in getattr(lay, 'recurrent_states', []) or []:
                if isinstance(rs, torch.Tensor):
                    rs.zero_()

    def snap():
        return [(list(getattr(l, 'conv_states', []) or []),
                 list(getattr(l, 'recurrent_states', []) or []),
                 int(l.cumulative_length.item())
                 if getattr(l, 'cumulative_length', None) is not None else None,
                 [c.clone() if isinstance(c, torch.Tensor) else c
                  for c in getattr(l, 'conv_states', []) or []],
                 [r.clone() if isinstance(r, torch.Tensor) else r
                  for r in getattr(l, 'recurrent_states', []) or []])
                for l in cache.layers]

    def restore(sn):
        for l, (_, _, cum, cs_c, rs_c) in zip(cache.layers, sn):
            if cum is not None:
                l.cumulative_length.fill_(cum)
            for a, b in zip(getattr(l, 'conv_states', []) or [], cs_c):
                if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
                    a.copy_(b)
            for a, b in zip(getattr(l, 'recurrent_states', []) or [], rs_c):
                if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
                    a.copy_(b)

    # capture g1
    emb1.zero_(); cos1.zero_(); sin1.zero_(); pos1.fill_(0)
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            run(emb1, cos1, sin1, pos1)
    torch.cuda.current_stream().wait_stream(s)
    g1 = torch.cuda.CUDAGraph()
    pool = torch.cuda.graph_pool_handle()
    with torch.cuda.graph(g1, pool=pool):
        h1_s, log1_s = run(emb1, cos1, sin1, pos1)
    hard_reset()

    ids = [552, 867, 279, 4478, 314, 264, 2407]  # fake prompt tokens
    prompt = 'The theory of relativity states that'
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(r'E:\models\Qwen3.8-27B')
    ids = tok(prompt, return_tensors='pt')['input_ids'][0].tolist()

    # path A: pure g1 decode (6 tokens)
    hard_reset()
    for i, tid in enumerate(ids):
        emb1.copy_(embed1(tid))
        cos1.copy_(cos_all[i].view(1, 1, -1))
        sin1.copy_(sin_all[i].view(1, 1, -1))
        pos1.fill_(i)
        run(emb1, cos1, sin1, pos1)
    ref = []
    nxt = None
    logits_last = None
    h, logits_last = run(emb1, cos1, sin1, pos1)  # rerun last
    logits_last = logits_last.clone()
    t = len(ids)
    for _ in range(6):
        nxt = logits_last[:, -1].argmax().item()
        ref.append(nxt)
        emb1.copy_(embed1(nxt))
        cos1.copy_(cos_all[t].view(1, 1, -1))
        sin1.copy_(sin_all[t].view(1, 1, -1))
        pos1.fill_(t)
        g1.replay()
        logits_last = log1_s.clone()
        t += 1

    # path B: prefill -> snap -> restore (NO g2) -> g1 decode
    hard_reset()
    for i, tid in enumerate(ids):
        emb1.copy_(embed1(tid))
        cos1.copy_(cos_all[i].view(1, 1, -1))
        sin1.copy_(sin_all[i].view(1, 1, -1))
        pos1.fill_(i)
        run(emb1, cos1, sin1, pos1)
    sn = snap()
    restore(sn)
    # ALSO do a g1 replay of the LAST prompt token again (harmless overwrite)
    # then decode
    nxt2 = None
    logits_last2 = None
    _, logits_last2 = run(emb1, cos1, sin1, pos1)
    logits_last2 = logits_last2.clone()
    t = len(ids)
    gen2 = []
    for _ in range(6):
        nxt2 = logits_last2[:, -1].argmax().item()
        gen2.append(nxt2)
        emb1.copy_(embed1(nxt2))
        cos1.copy_(cos_all[t].view(1, 1, -1))
        sin1.copy_(sin_all[t].view(1, 1, -1))
        pos1.fill_(t)
        g1.replay()
        logits_last2 = log1_s.clone()
        t += 1

    print(f'ref : {ref}')
    print(f'loop: {gen2}')
    print(f'MATCH: {ref == gen2}')


if __name__ == '__main__':
    main()
