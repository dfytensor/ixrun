# -*- coding: utf-8 -*-
"""Validate the gdn-seq patch: seeded-state A/B must go from 450x to clean."""
import sys, time, gc, json
sys.setrecursionlimit(20000)
sys.path.insert(0, r'E:\IXRUN')
import pandas
import torch
import torch.nn as nn
from transformers import AutoConfig
from transformers.cache_utils import StaticCache
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5Attention
from ixrun.fla_patch import apply_fla_kernels
from ixrun.gdn_seq_patch import apply_gdn_sequential_patch
from ixrun.udcq import UdcqLinear
from ixrun.linear import _set_parent_child
from ixrun.config import QWEN38_PATH
from accelerate import init_empty_weights
from transformers import AutoModelForCausalLM

apply_fla_kernels()
ok = apply_gdn_sequential_patch(verbose=True)
assert ok
_ORIG = Qwen3_5Attention.forward
MAX_CTX = 64
BLOB = r'E:\IXRUN\experiments\qwen38_udcq\q38_blob.pt'


def _static_fwd(self, hidden_states, position_embeddings, attention_mask=None,
                past_key_values=None, **kw):
    import torch.nn.functional as F
    from transformers.models.qwen3_5.modeling_qwen3_5 import (
        apply_rotary_pos_emb)
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
    blob = torch.load(BLOB, map_location='cpu', mmap=True, weights_only=False)
    cfg = AutoConfig.from_pretrained(QWEN38_PATH, trust_remote_code=True)
    with init_empty_weights():
        m = AutoModelForCausalLM.from_config(cfg, trust_remote_code=True)
    from ixrun.udcq import UDCQ_G
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
    tm = m.model
    if hasattr(tm, 'language_model'):
        tm = tm.language_model
    H = tm.config.hidden_size
    dev = 'cuda'
    # materialize NON-linear params for layers 0..7 (enough for the probe)
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
    print('params loaded', flush=True)

    pos_all = torch.arange(MAX_CTX, device=dev).unsqueeze(0)
    dummy = torch.zeros(1, MAX_CTX, H, dtype=torch.bfloat16, device=dev)
    cos_all, sin_all = tm.rotary_emb(dummy, pos_all)
    if cos_all.dim() == 4:
        cos_all = cos_all[:, :, 0]
    if cos_all.dim() == 3:
        cos_all, sin_all = cos_all[0], sin_all[0]

    cache = StaticCache(config=m.config, max_cache_len=MAX_CTX)

    def run(emb, cos, sin, pos, upto=8):
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

    torch.manual_seed(0)
    t0 = 10
    h1 = torch.randn(1, 1, H, dtype=torch.bfloat16, device=dev) * 0.05
    h2 = torch.randn(1, 1, H, dtype=torch.bfloat16, device=dev) * 0.05
    seeds = [552, 867, 279, 4478, 314, 264, 2407]

    # path A: 2x S=1
    hard_reset()
    for i, tid in enumerate(seeds):
        run(embed1(tid), cos_all[i].view(1, 1, -1), sin_all[i].view(1, 1, -1),
            torch.tensor([i], device=dev))
    hA1 = run(h1, cos_all[t0].view(1, 1, -1), sin_all[t0].view(1, 1, -1),
              torch.tensor([t0], device=dev)).clone()
    hA2 = run(h2, cos_all[t0 + 1].view(1, 1, -1),
              sin_all[t0 + 1].view(1, 1, -1),
              torch.tensor([t0 + 1], device=dev)).clone()

    # path B: 1x S=2 (patched)
    hard_reset()
    for i, tid in enumerate(seeds):
        run(embed1(tid), cos_all[i].view(1, 1, -1), sin_all[i].view(1, 1, -1),
            torch.tensor([i], device=dev))
    hh = torch.cat([h1, h2], dim=1)
    cos2 = torch.stack([cos_all[t0], cos_all[t0 + 1]]).view(1, 2, -1)
    sin2 = torch.stack([sin_all[t0], sin_all[t0 + 1]]).view(1, 2, -1)
    hB = run(hh, cos2, sin2, torch.tensor([t0, t0 + 1], device=dev)).clone()

    d0 = (hB[0, 0].float() - hA1[0, 0].float()).abs().max().item()
    d1 = (hB[0, 1].float() - hA2[0, 0].float()).abs().max().item()
    refm = hA1.float().abs().mean().item()
    print(f'[PATCHED A/B 8-layer] |h| mean {refm:.4f} | tok0 max {d0:.5f} '
          f'tok1 max {d1:.5f}  (was 0.41 / 3.06)', flush=True)

    # ---- per-layer divergence scan: rebuild paths capturing h after each layer
    def run_capture(emb, cos, sin, pos):
        hs = []
        h = emb
        for layer in list(tm.layers)[:8]:
            h = layer(h, position_embeddings=(cos, sin), attention_mask=None,
                      position_ids=pos.view(1, -1), past_key_values=cache)
            if isinstance(h, tuple):
                h = h[0]
            hs.append(h.clone())
        return hs

    # path A capture
    hard_reset()
    for i, tid in enumerate(seeds):
        run(embed1(tid), cos_all[i].view(1, 1, -1), sin_all[i].view(1, 1, -1),
            torch.tensor([i], device=dev))
    hA1c = run_capture(h1, cos_all[t0].view(1, 1, -1),
                       sin_all[t0].view(1, 1, -1), torch.tensor([t0], device=dev))
    hA2c = run_capture(h2, cos_all[t0 + 1].view(1, 1, -1),
                       sin_all[t0 + 1].view(1, 1, -1),
                       torch.tensor([t0 + 1], device=dev))
    # path B capture
    hard_reset()
    for i, tid in enumerate(seeds):
        run(embed1(tid), cos_all[i].view(1, 1, -1), sin_all[i].view(1, 1, -1),
            torch.tensor([i], device=dev))
    hBc = run_capture(hh, cos2, sin2,
                      torch.tensor([t0, t0 + 1], device=dev))
    for li in range(8):
        ltype = list(tm.layers)[li].block_type[:6]
        da = (hBc[li][0, 0].float() - hA1c[li][0, 0].float()).abs().max().item()
        db = (hBc[li][0, 1].float() - hA2c[li][0, 0].float()).abs().max().item()
        print(f'  layer{li} ({ltype}): tok0 {da:8.4f} tok1 {db:8.4f}', flush=True)


if __name__ == '__main__':
    main()
