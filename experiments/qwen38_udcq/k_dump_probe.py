# -*- coding: utf-8 -*-
"""Surgical probe: dump the K tensor just before cache.update in the patched
attention, for S=1 (x2) vs S=2, on a single full-attn layer with a FROZEN
input hidden. Isolates k_norm/rope vs cache-write."""
import sys, time, gc, json
sys.setrecursionlimit(20000)
sys.path.insert(0, r'E:\IXRUN')
import pandas
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoConfig
from transformers.cache_utils import StaticCache
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5Attention, apply_rotary_pos_emb)
from ixrun.fla_patch import apply_fla_kernels
from ixrun.udcq import UdcqLinear
from ixrun.linear import _set_parent_child
from ixrun.config import QWEN38_PATH
from accelerate import init_empty_weights
from transformers import AutoModelForCausalLM

apply_fla_kernels()
MAX_CTX = 64
BLOB = r'E:\IXRUN\experiments\qwen38_udcq\q38_blob.pt'

# global capture of the K tensor right before update
K_DUMP = {}


def _patched(self, hidden_states, position_embeddings, attention_mask=None,
             past_key_values=None, **kw):
    ish = hidden_states.shape[:-1]
    hsh = (*ish, -1, self.head_dim)
    q_len = hidden_states.shape[1]
    qp = self.q_proj(hidden_states)
    q, gate = torch.chunk(
        qp.view(*ish, -1, self.head_dim * 2), 2, -1)
    gate = gate.reshape(*ish, -1)
    q = self.q_norm(q.view(hsh)).transpose(1, 2)
    k = self.k_norm(self.k_proj(hidden_states).view(hsh)).transpose(1, 2)
    v = self.v_proj(hidden_states).view(hsh).transpose(1, 2)
    cos, sin = position_embeddings
    q, k = apply_rotary_pos_emb(q, k, cos, sin)
    if past_key_values is not None and not isinstance(past_key_values, tuple):
        kf, vf = past_key_values.update(k, v, self.layer_idx)
        K_DUMP['k_in'] = k.detach().clone()
        K_DUMP['k_cache'] = kf[:, :, :].detach().clone()
        raise RuntimeError('DUMP')
    K_DUMP['k'] = k.detach().clone()
    raise RuntimeError('DUMP')


Qwen3_5Attention.forward = _patched


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
    # norms for the attn layer only (avoid full param materialization)
    idx = json.load(open(r'E:\models\Qwen3.8-27B\model.safetensors.index.json'))['weight_map']

    def ck(name):
        return name if name in idx else \
            (name.replace('model.', 'model.language_model.', 1)
             if name.replace('model.', 'model.language_model.', 1) in idx else None)

    from safetensors import safe_open as so
    needed = []
    tm = m.model
    if hasattr(tm, 'language_model'):
        tm = tm.language_model
    lay = next(l for l in tm.layers if l.block_type == 'full_attention')
    lay_no = next(i for i, l in enumerate(tm.layers)
                  if l.block_type == 'full_attention')
    H = tm.config.hidden_size
    dev = 'cuda'
    # materialize q_norm/k_norm + rotary for this layer
    for pname, p in lay.named_parameters():
        key = ck(f'model.layers.{lay_no}.{pname}')
        if key is None:
            continue
        with so(rf'E:\models\Qwen3.8-27B\{idx[key]}', 'pt') as sf:
            t = sf.get_tensor(key)
        parts = pname.split('.')
        parent = lay
        for q_ in parts[:-1]:
            parent = getattr(parent, q_)
        parent._parameters[parts[-1]] = torch.nn.Parameter(
            t.cuda(), requires_grad=False)
    print('layer norms loaded', flush=True)

    pos_all = torch.arange(MAX_CTX, device=dev).unsqueeze(0)
    dummy = torch.zeros(1, MAX_CTX, H, dtype=torch.bfloat16, device=dev)
    cos_all, sin_all = tm.rotary_emb(dummy, pos_all)
    if cos_all.dim() == 4:
        cos_all = cos_all[:, :, 0]
    if cos_all.dim() == 3:
        cos_all, sin_all = cos_all[0], sin_all[0]

    torch.manual_seed(0)
    h1 = (torch.randn(1, 1, H, dtype=torch.bfloat16, device=dev) * 0.05)
    h2 = (torch.randn(1, 1, H, dtype=torch.bfloat16, device=dev) * 0.05)
    t0 = 20

    hh = torch.cat([h1, h2], dim=1)

    # ---- full-update test: does index_copy_ store what we computed? ----
    cache2 = StaticCache(config=m.config, max_cache_len=MAX_CTX)
    # prefill cache via two S=1 dummy updates at positions 0..t0-1 so cum=t0
    class _FakeSelf:
        pass

    def s2_with_cache(hh, t):
        cos2 = torch.stack([cos_all[t], cos_all[t + 1]]).view(1, 2, -1)
        sin2 = torch.stack([sin_all[t], sin_all[t + 1]]).view(1, 2, -1)
        lay.self_attn(hh, position_embeddings=(cos2, sin2),
                      past_key_values=cache2)

    def s1_with_cache(h, t):
        cos1 = cos_all[t].view(1, 1, -1)
        sin1 = sin_all[t].view(1, 1, -1)
        lay.self_attn(h, position_embeddings=(cos1, sin1),
                      past_key_values=cache2)

    # S=1 x2 path (positions t0, t0+1) with real cache writes
    K_DUMP.clear()
    try:
        s1_with_cache(h1, t0)
    except RuntimeError:
        pass
    K_DUMP.clear()
    try:
        s1_with_cache(h2, t0 + 1)
    except RuntimeError:
        pass
    kcache_seq = K_DUMP['k_cache'].clone()      # after two S=1 writes
    k_in_seq = K_DUMP['k_in'].clone()           # K of token B

    # fresh cache, S=2 path at same positions
    cache2 = StaticCache(config=m.config, max_cache_len=MAX_CTX)
    K_DUMP.clear()
    try:
        s2_with_cache(hh, t0)
    except RuntimeError:
        pass
    kcache_blk = K_DUMP['k_cache'].clone()
    k_in_blk = K_DUMP['k_in'].clone()

    cum = 2
    d_slot0 = (kcache_blk[:, :, 0].float() - kcache_seq[:, :, 0].float()).abs().max().item()
    d_slot1 = (kcache_blk[:, :, 1].float() - kcache_seq[:, :, 1].float()).abs().max().item()
    print(f'[indexcopy] cache slot0 dK {d_slot0:.5f} slot1 dK {d_slot1:.5f} '
          f'(positions 0,1; S1x2 vs S2)', flush=True)

    # ---- la layer WITH PRE-SEEDED state: S=2 vs 2xS=1 ----
    # (the model state-diff had 7 prefilled tokens; the earlier la probe ran
    #  from empty states — this is the remaining untested configuration)
    l0 = next(l for l in tm.layers if l.block_type == 'linear_attention')
    H0 = H
    # materialize l0's params (norms/A_log/dt_bias) — they're still meta
    l0_no = next(i for i, l in enumerate(tm.layers)
                 if l.block_type == 'linear_attention')
    for pname, p in l0.named_parameters():
        key = ck(f'model.layers.{l0_no}.{pname}')
        if key is None:
            continue
        with so(rf'E:\models\Qwen3.8-27B\{idx[key]}', 'pt') as sf:
            t = sf.get_tensor(key)
        parts = pname.split('.')
        parent = l0
        for q_ in parts[:-1]:
            parent = getattr(parent, q_)
        parent._parameters[parts[-1]] = torch.nn.Parameter(
            t.cuda(), requires_grad=False)

    def la_out_seeded(h1_, h2_, pos0, seed_tokens):
        # seed states with S=1 calls (eager, honest path)
        hard_cache_reset()
        for i, tid in enumerate(seed_tokens):
            c1 = cos_all[i].view(1, 1, -1)
            s1 = sin_all[i].view(1, 1, -1)
            hh = (blob['embed'][tid].view(1, 1, H0).to(dev, torch.bfloat16))
            l0.linear_attn(hh, cache_params=cache2, attention_mask=None,
                           position_ids=None)
        snap_state = snap_cache_light()
        # sequential path
        c1 = cos_all[pos0].view(1, 1, -1)
        s1 = sin_all[pos0].view(1, 1, -1)
        o1 = l0.linear_attn(h1_, cache_params=cache2, attention_mask=None,
                            position_ids=None)
        c1 = cos_all[pos0 + 1].view(1, 1, -1)
        s1 = sin_all[pos0 + 1].view(1, 1, -1)
        o2 = l0.linear_attn(h2_, cache_params=cache2, attention_mask=None,
                            position_ids=None)
        seq_out = torch.cat([o1, o2], 1)
        seq_state = snap_cache_light()
        # block path from the SAME seeded state
        restore_cache_light(snap_state)
        hh2 = torch.cat([h1_, h2_], dim=1)
        ob = l0.linear_attn(hh2, cache_params=cache2, attention_mask=None,
                            position_ids=None)
        return seq_out, ob

    def snap_cache_light():
        return [([c.clone() if isinstance(c, torch.Tensor) else c
                  for c in getattr(l, 'conv_states', []) or []],
                 [r.clone() if isinstance(r, torch.Tensor) else r
                  for r in getattr(l, 'recurrent_states', []) or []],
                 l.cumulative_length.clone()
                 if getattr(l, 'cumulative_length', None) is not None else None)
                for l in cache2.layers]

    def restore_cache_light(sn):
        for l, (cs, rs, cum) in zip(cache2.layers, sn):
            if cum is not None:
                l.cumulative_length.copy_(cum)
            for a, b in zip(getattr(l, 'conv_states', []) or [], cs):
                if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
                    a.copy_(b)
            for a, b in zip(getattr(l, 'recurrent_states', []) or [], rs):
                if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
                    a.copy_(b)

    def hard_cache_reset():
        for l in cache2.layers:
            cum = getattr(l, 'cumulative_length', None)
            if cum is not None:
                cum.zero_()
            hps = getattr(l, 'has_previous_state', None)
            if hps is not None:
                if isinstance(hps, dict):
                    for k in hps:
                        hps[k] = False
                else:
                    l.has_previous_state = [False] * len(hps)
            for cs in getattr(l, 'conv_states', []) or []:
                if isinstance(cs, torch.Tensor):
                    cs.zero_()
            for rs in getattr(l, 'recurrent_states', []) or []:
                if isinstance(rs, torch.Tensor):
                    rs.zero_()

    # la ignores position_embeddings (uses conv+recurrence) so cos/sin not
    # needed for it; seed with 7 real prompt tokens
    ids_seed = [552, 867, 279, 4478, 314, 264, 2407]
    seq_out, blk_out = la_out_seeded(h1, h2, t0, ids_seed)
    dA = (seq_out[0, 0].float() - blk_out[0, 0].float()).abs().max().item()
    dB = (seq_out[0, 1].float() - blk_out[0, 1].float()).abs().max().item()
    refm = seq_out.float().abs().mean().item()
    print(f'[la SEEDED] |out| mean {refm:.5f} | tok0 max {dA:.5f} '
          f'tok1 max {dB:.5f}', flush=True)
    print(f'[indexcopy] k_in blk row0 vs seq cache slot0: '
          f'{(k_in_blk[0, :, 0].float() - kcache_seq[0, :, 0].float()).abs().max().item():.5f}',
          flush=True)
    print(f'[indexcopy] S=2 k_in row0 vs cache slot0 (self-consistency): '
          f'{(k_in_blk[0, :, 0].float() - kcache_blk[0, :, 0].float()).abs().max().item():.5f}',
          flush=True)
    print(f'[indexcopy] S=2 k_in row1 vs cache slot1 (self-consistency): '
          f'{(k_in_blk[0, :, 1].float() - kcache_blk[0, :, 1].float()).abs().max().item():.5f}',
          flush=True)

    def s1_call(h, t):
        cos1 = cos_all[t].view(1, 1, -1)
        sin1 = sin_all[t].view(1, 1, -1)
        lay.self_attn(h, position_embeddings=(cos1, sin1),
                      past_key_values=object())  # dummy; we raise before use

    print(f'[indexcopy] k_in blk row0 vs seq cache slot0: '
          f'{(k_in_blk[0, :, 0].float() - kcache_seq[0, :, 0].float()).abs().max().item():.5f}',
          flush=True)
    print(f'[indexcopy] S=2 k_in row0 vs cache slot0 (self-consistency): '
          f'{(k_in_blk[0, :, 0].float() - kcache_blk[0, :, 0].float()).abs().max().item():.5f}',
          flush=True)
    print(f'[indexcopy] S=2 k_in row1 vs cache slot1 (self-consistency): '
          f'{(k_in_blk[0, :, 1].float() - kcache_blk[0, :, 1].float()).abs().max().item():.5f}',
          flush=True)


if __name__ == '__main__':
    main()
