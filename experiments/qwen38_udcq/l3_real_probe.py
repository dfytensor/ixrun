# -*- coding: utf-8 -*-
"""DECISIVE probe: the REAL layer 3 (full-attn, UDCQ weights, blob deploy),
seeded via REAL prompt prefill, then:
  path A: 2x eager S=1    path B: 1x eager S=2 (NO graphs)
compare K at update, hidden in, hidden out. This is the exact configuration
where the stack scan showed 92.25 divergence — with real weights, seeded
state, in-stack layer, eager (graph excluded)."""
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
from ixrun.gdn_seq_patch import apply_gdn_sequential_patch, _dbg_spec_hits
apply_fla_kernels()
apply_gdn_sequential_patch(verbose=True)
_ORIG = Qwen3_5Attention.forward
MAX_CTX = 256
BLOB = r'E:\IXRUN\experiments\qwen38_udcq\q38_blob.pt'
K_DUMP = {}


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
    K_DUMP['q_in'] = hidden_states.detach().clone()
    q = self.q_norm(q.view(hsh)).transpose(1, 2)
    k = self.k_norm(self.k_proj(hidden_states).view(hsh)).transpose(1, 2)
    v = self.v_proj(hidden_states).view(hsh).transpose(1, 2)
    cos, sin = position_embeddings
    q, k = apply_rotary_pos_emb(q, k, cos, sin)
    K_DUMP['k_rot'] = k.detach().clone() if q_len > 1 else None
    kf, vf = past_key_values.update(k, v, self.layer_idx)
    K_DUMP['k_cache'] = kf.detach().clone()
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
    emb2 = torch.zeros(1, 2, H, dtype=torch.bfloat16, device=dev)
    cos2 = torch.zeros(1, 2, cos_all.shape[-1], dtype=cos_all.dtype, device=dev)
    sin2 = torch.zeros_like(cos2)
    pos2 = torch.zeros(2, dtype=torch.long, device=dev)

    L3 = 3        # first full-attn layer

    def run(emb, cos, sin, pos, upto=L3 + 1):
        h = emb
        for layer in list(tm.layers)[:upto]:
            h = layer(h, position_embeddings=(cos, sin), attention_mask=None,
                      position_ids=pos.view(1, -1), past_key_values=cache)
            if isinstance(h, tuple):
                h = h[0]
        return h

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

    def full_snap():
        s = {'cum': [], 'kv': [], 'conv': [], 'rec': []}
        for l in cache.layers:
            s['cum'].append(int(l.cumulative_length.item())
                             if getattr(l, 'cumulative_length', None) is not None
                             else None)
            s['kv'].append((l.keys.clone(), l.values.clone())
                           if getattr(l, 'is_initialized', False) else None)
            s['conv'].append([c.clone() if isinstance(c, torch.Tensor) else c
                              for c in getattr(l, 'conv_states', []) or []])
            s['rec'].append([r.clone() if isinstance(r, torch.Tensor) else r
                             for r in getattr(l, 'recurrent_states', []) or []])
        return s

    def full_restore(s):
        for l, cum, kv, cs, rs in zip(cache.layers, s['cum'], s['kv'],
                                       s['conv'], s['rec']):
            if cum is not None:
                l.cumulative_length.fill_(cum)
            if kv is not None:
                l.keys.copy_(kv[0])
                l.values.copy_(kv[1])
            for a, b in zip(getattr(l, 'conv_states', []) or [], cs):
                if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
                    a.copy_(b)
            for a, b in zip(getattr(l, 'recurrent_states', []) or [], rs):
                if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
                    a.copy_(b)

    # prefill with a REAL prompt
    tok = AutoTokenizer.from_pretrained(r'E:\models\Qwen3.8-27B')
    ids = tok('The theory of relativity states that',
              return_tensors='pt')['input_ids'][0].tolist()
    hard_reset()
    for i, tid in enumerate(ids):
        emb1.copy_(embed1(tid))
        cos1.copy_(cos_all[i].view(1, 1, -1))
        sin1.copy_(sin_all[i].view(1, 1, -1))
        pos1.fill_(i)
        run(emb1, cos1, sin1, pos1)
    base = full_snap()

    torch.manual_seed(0)
    t1 = 279      # ' the'
    t2 = 4478     # ' speed'
    tA = len(ids)

    # ---- path A: 2x S=1 through layers 0..3, capture layer3's K ----
    full_restore(base)
    K_DUMP.clear()
    emb1.copy_(embed1(t1))
    cos1.copy_(cos_all[tA].view(1, 1, -1))
    sin1.copy_(sin_all[tA].view(1, 1, -1))
    pos1.fill_(tA)
    run(emb1, cos1, sin1, pos1)          # writes K_t1
    hA1_in = K_DUMP['q_in'].clone()      # layer3 input for t1
    hA1 = None
    # rerun to get hA1 output (state already includes t1; harmless overwrite)
    K_DUMP.clear()
    hA1 = run(emb1, cos1, sin1, pos1).clone()
    kA_cache = K_DUMP['k_cache'].clone()
    kA_rot = K_DUMP['k_rot'].clone() if K_DUMP['k_rot'] is not None else None
    K_DUMP.clear()
    emb1.copy_(embed1(t2))
    cos1.copy_(cos_all[tA + 1].view(1, 1, -1))
    sin1.copy_(sin_all[tA + 1].view(1, 1, -1))
    pos1.fill_(tA + 1)
    hA = run(emb1, cos1, sin1, pos1).clone()
    kA_cache2 = K_DUMP['k_cache'].clone()
    kA_rot2 = K_DUMP['k_rot'].clone() if K_DUMP['k_rot'] is not None else None
    if kA_rot2 is None:
        # S=1 doesn't dump k_rot; derive from cache slot directly
        kA_rot2 = kA_cache2[:, :, tA + 1:tA + 2]

    # ---- path B: 1x S=2 ----
    full_restore(base)
    K_DUMP.clear()
    emb2[:, 0].copy_(embed1(t1).view(1, H))
    emb2[:, 1].copy_(embed1(t2).view(1, H))
    cos2[:, 0].copy_(cos_all[tA].view(1, -1))
    cos2[:, 1].copy_(cos_all[tA + 1].view(1, -1))
    sin2[:, 0].copy_(sin_all[tA].view(1, -1))
    sin2[:, 1].copy_(sin_all[tA + 1].view(1, -1))
    pos2[0] = tA; pos2[1] = tA + 1
    hB = run(emb2, cos2, sin2, pos2).clone()
    hB_in = K_DUMP['q_in'].clone()
    kB_cache = K_DUMP['k_cache'].clone()
    kB_rot = K_DUMP['k_rot'].clone()

    # ---- bisect: input hidden? K-after-rope? cache? ----
    din0 = (hB_in[0, 0].float() - hA1_in[0, 0].float()).abs().max().item()
    din1 = (hB_in[0, 1].float() - hA[0, 0].float()).abs().max().item()
    print(f'[bisect] layer3 INPUT hidden: tok0 {din0:.5f} tok1 {din1:.5f}',
          flush=True)
    print(f'[v3 hits] spec_block entered {_dbg_spec_hits()} times', flush=True)

    # ---- per-layer input scan (upto=1,2,3) ----
    for upto in (1, 2, 3):
        full_restore(base)
        emb1.copy_(embed1(t1))
        cos1.copy_(cos_all[tA].view(1, 1, -1))
        sin1.copy_(sin_all[tA].view(1, 1, -1))
        pos1.fill_(tA)
        a1 = run(emb1, cos1, sin1, pos1, upto=upto).clone()
        emb1.copy_(embed1(t2))
        cos1.copy_(cos_all[tA + 1].view(1, 1, -1))
        sin1.copy_(sin_all[tA + 1].view(1, 1, -1))
        pos1.fill_(tA + 1)
        a2 = run(emb1, cos1, sin1, pos1, upto=upto).clone()
        full_restore(base)
        emb2[:, 0].copy_(embed1(t1).view(1, H))
        emb2[:, 1].copy_(embed1(t2).view(1, H))
        cos2[:, 0].copy_(cos_all[tA].view(1, -1))
        cos2[:, 1].copy_(cos_all[tA + 1].view(1, -1))
        sin2[:, 0].copy_(sin_all[tA].view(1, -1))
        sin2[:, 1].copy_(sin_all[tA + 1].view(1, -1))
        pos2[0] = tA; pos2[1] = tA + 1
        bb = run(emb2, cos2, sin2, pos2, upto=upto).clone()
        dd0 = (bb[0, 0].float() - a1[0, 0].float()).abs().max().item()
        dd1 = (bb[0, 1].float() - a2[0, 0].float()).abs().max().item()
        print(f'  [scan after layer{upto-1}] tok0 {dd0:.5f} tok1 {dd1:.5f}',
              flush=True)
    # K after rope: path B row1 vs path A t2 call
    dk_rot = (kB_rot[0, :, 1:2].float() - kA_rot2[0].float()).abs().max().item()
    print(f'[bisect] K-after-rope row1: {dk_rot:.5f}', flush=True)

    # layer3's K/V slots [tA, tA+1] = global slots 7,8
    dK_cache = (kA_cache[:, :, 7:9].float()
                 - kB_cache[:, :, 7:9].float()).abs().max().item()
    dK_rot = None
    print(f'[L3 REAL eager] cache slots 7,8 dK = {dK_cache:.5f}', flush=True)
    # hidden out of layer 3
    d0 = (hB[0, 0].float() - hA[0, 0].float()).abs().max().item()
    d1 = (hB[0, 1].float() - (hA[0, 0].float() * 0 + hA[0, 0].float())
          ).abs().max().item()  # hA is only the LAST token's hidden
    # compare row0 with hA-of-token1: we need path A's token1 hidden — rerun
    full_restore(base)
    emb1.copy_(embed1(t1))
    cos1.copy_(cos_all[tA].view(1, 1, -1))
    sin1.copy_(sin_all[tA].view(1, 1, -1))
    pos1.fill_(tA)
    hA1 = run(emb1, cos1, sin1, pos1).clone()
    dh0 = (hB[0, 0].float() - hA1[0, 0].float()).abs().max().item()
    dh1 = (hB[0, 1].float() - hA[0, 0].float()).abs().max().item()
    refm = hA1.float().abs().mean().item()
    print(f'[L3 REAL eager] hidden out: |h| {refm:.4f} | tok0 {dh0:.5f} '
          f'tok1 {dh1:.5f}  (stack scan was 92.25/36.0)', flush=True)


if __name__ == '__main__':
    main()
