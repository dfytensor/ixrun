# -*- coding: utf-8 -*-
"""Real-weight layer0 la A/B with v3 patch: bisect INSIDE the la layer.
Dumps: qkv after conv, delta output, final output — S=2 block vs 2xS=1."""
import pandas
import sys, gc, json
sys.setrecursionlimit(20000)
sys.path.insert(0, r'E:\IXRUN')
import torch
import torch.nn as nn
from transformers import AutoConfig
from accelerate import init_empty_weights
from transformers import AutoModelForCausalLM
from transformers.cache_utils import StaticCache
from ixrun.fla_patch import apply_fla_kernels
from ixrun.gdn_seq_patch import apply_gdn_sequential_patch, _dbg_spec_hits
apply_fla_kernels()
apply_gdn_sequential_patch(verbose=True)
from ixrun.udcq import UdcqLinear, UDCQ_G
from ixrun.linear import _set_parent_child
from ixrun.config import QWEN38_PATH

BLOB = r'E:\IXRUN\experiments\qwen38_udcq\q38_blob.pt'
dev = 'cuda'
MAX_CTX = 64

blob = torch.load(BLOB, map_location='cpu', mmap=True, weights_only=False)
cfg = AutoConfig.from_pretrained(QWEN38_PATH, trust_remote_code=True)
with init_empty_weights():
    m = AutoModelForCausalLM.from_config(cfg, trust_remote_code=True)
for name, mod in list(m.named_modules()):
    if isinstance(mod, nn.Linear) and name in blob['layers']:
        e = blob['layers'][name]
        packed = {'g': UDCQ_G, 'out_f': mod.out_features, 'in_f': mod.in_features,
                  'N': mod.out_features * mod.in_features,
                  'idx': e['idx'], 'scale': e['scale'], 'sign_packed': e['sign'],
                  'codebook': blob['codebook'], 'bits_per_weight': 6.0}
        _set_parent_child(m, name, UdcqLinear(packed, bias=None, cache='stream'))
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
    parent._parameters[parts[-1]] = torch.nn.Parameter(t.cuda(), requires_grad=False)
    del t
gc.collect(); torch.cuda.empty_cache()

tm = m.model
if hasattr(tm, 'language_model'):
    tm = tm.language_model
H = tm.config.hidden_size
l0 = tm.layers[0]
cache = StaticCache(config=m.config, max_cache_len=MAX_CTX)

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
    return ([c.clone() if isinstance(c, torch.Tensor) else c
             for c in cache.layers[0].conv_states],
            [r.clone() if isinstance(r, torch.Tensor) else r
             for r in cache.layers[0].recurrent_states])

def restore(s):
    for a, b in zip(cache.layers[0].conv_states, s[0]):
        if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
            a.copy_(b)
    for a, b in zip(cache.layers[0].recurrent_states, s[1]):
        if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
            a.copy_(b)

# seed with 7 tokens
ids = [552, 867, 279, 4478, 314, 264, 2407]
hard_reset()
for tid in ids:
    l0.linear_attn(embed1(tid), cache_params=cache, attention_mask=None)
base = snap()

torch.manual_seed(0)
h1 = torch.randn(1, 1, H, device=dev, dtype=torch.bfloat16) * 0.05
h2 = torch.randn(1, 1, H, device=dev, dtype=torch.bfloat16) * 0.05

# path A: 2x S=1
restore(base)
oa1 = l0.linear_attn(h1, cache_params=cache, attention_mask=None).clone()
oa2 = l0.linear_attn(h2, cache_params=cache, attention_mask=None).clone()

# path B: 1x S=2
restore(base)
hh = torch.cat([h1, h2], dim=1)
ob = l0.linear_attn(hh, cache_params=cache, attention_mask=None).clone()

d0 = (ob[0, 0].float() - oa1[0, 0].float()).abs().max().item()
d1 = (ob[0, 1].float() - oa2[0, 0].float()).abs().max().item()
refm = oa1.float().abs().mean().item()
print(f'[L0 real seeded A/B] |out| {refm:.5f} | tok0 {d0:.6f} tok1 {d1:.6f} '
      f'| v3 hits {_dbg_spec_hits()}', flush=True)
