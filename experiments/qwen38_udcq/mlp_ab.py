# -*- coding: utf-8 -*-
"""MLP + in_proj A/B on REAL UDCQ weights: M=2 GEMM vs 2x M=1 GEMV."""
import pandas
import sys, gc, json
sys.setrecursionlimit(20000)
sys.path.insert(0, r'E:\IXRUN')
import torch
import torch.nn as nn
from transformers import AutoConfig
from accelerate import init_empty_weights
from transformers import AutoModelForCausalLM
from ixrun.udcq import UdcqLinear, UDCQ_G
from ixrun.linear import _set_parent_child
from ixrun.config import QWEN38_PATH

BLOB = r'E:\IXRUN\experiments\qwen38_udcq\q38_blob.pt'
dev = 'cuda'

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

torch.manual_seed(0)
targets = [
    ('mlp.gate_proj', tm.layers[0].mlp.gate_proj),
    ('mlp.down_proj', tm.layers[0].mlp.down_proj),
    ('mlp.up_proj', tm.layers[0].mlp.up_proj),
    ('la.in_proj_qkv', tm.layers[0].linear_attn.in_proj_qkv),
    ('la.in_proj_z', tm.layers[0].linear_attn.in_proj_z),
]
for tag, mod in targets:
    in_f = mod.in_features
    h1 = torch.randn(1, 1, in_f, dtype=torch.bfloat16, device=dev) * 0.05
    h2 = torch.randn(1, 1, in_f, dtype=torch.bfloat16, device=dev) * 0.05
    y1 = mod(h1).reshape(-1)
    y2 = mod(h2).reshape(-1)
    hh = torch.cat([h1, h2], dim=1)
    y = mod(hh)
    d0 = (y[0, 0].float() - y1.float()).abs().max().item()
    d1 = (y[0, 1].float() - y2.float()).abs().max().item()
    refm = y1.float().abs().mean().item()
    print(f'{tag:16s} ({in_f}->{mod.out_features}): |y| {refm:.4f} '
          f'row0 {d0:.5f} row1 {d1:.5f}', flush=True)
