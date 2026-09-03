# -*- coding: utf-8 -*-
"""Full-attn SEEDED-KV S=2 vs 2xS=1 — single layer, 7 prefilled tokens.
This is the exact configuration the earlier fresh-cache probe missed."""
import pandas  # MUST be before torch (DLL order, env quirk)
import sys
sys.setrecursionlimit(20000)
sys.path.insert(0, r'E:\IXRUN')
import torch
import torch.nn.functional as F
from transformers.cache_utils import StaticCache
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5Attention, apply_rotary_pos_emb)

dev = 'cuda'
torch.manual_seed(0)
MAX_CTX = 64
H = 512
NQ, NKV, D = 24, 4, 256

att = Qwen3_5Attention.__new__(Qwen3_5Attention)
import torch.nn as nn
nn.Module.__init__(att)
att.head_dim = D
att.num_key_value_groups = NQ // NKV
att.scaling = D ** -0.5
att.attention_dropout = 0.0
att.is_causal = True
att.layer_idx = 0
att.q_proj = nn.Linear(H, NQ * D * 2, bias=False)
att.k_proj = nn.Linear(H, NKV * D, bias=False)
att.v_proj = nn.Linear(H, NKV * D, bias=False)
att.o_proj = nn.Linear(NQ * D, H, bias=False)
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5RMSNorm
att.q_norm = Qwen3_5RMSNorm(D, eps=1e-6)
att.k_norm = Qwen3_5RMSNorm(D, eps=1e-6)
att = att.to(dev).to(torch.bfloat16).eval()

# rope tables
inv = 1.0 / (10000.0 ** (torch.arange(0, 128, 2, device=dev).float() / 128))
pos = torch.arange(MAX_CTX, device=dev).float()
freqs = torch.outer(pos, inv)                    # [MAX, 64]
emb = torch.cat([freqs, freqs], -1)             # [MAX, 128]
cos_all = emb.cos().bfloat16()
sin_all = emb.sin().bfloat16()

cache = StaticCache(config=None, max_cache_len=MAX_CTX) if False else None
# build a minimal static cache via the real class (needs config) — use a
# manual stub instead
class StubLayer:
    pass

class StubCache:
    def __init__(self):
        self.layers = [StubLayer()]

stub = StubCache()
L = stub.layers[0]
L.keys = torch.zeros(1, NKV, MAX_CTX, D, dtype=torch.bfloat16, device=dev)
L.values = torch.zeros(1, NKV, MAX_CTX, D, dtype=torch.bfloat16, device=dev)
L.cumulative_length = torch.tensor(0, device=dev)
L.is_initialized = True

def update(k, v):
    qlen = k.shape[2]
    cp = torch.arange(qlen, device=dev) + L.cumulative_length
    L.cumulative_length.add_(qlen)
    L.keys.index_copy_(2, cp, k)
    L.values.index_copy_(2, cp, v)
    return L.keys, L.values

def fwd(h, cos, sin):
    ish = h.shape[:-1]
    hsh = (*ish, -1, D)
    q_len = h.shape[1]
    q, gate = torch.chunk(
        att.q_proj(h).view(*ish, -1, D * 2), 2, -1)
    gate = gate.reshape(*ish, -1)
    q = att.q_norm(q.view(hsh)).transpose(1, 2)
    k = att.k_norm(att.k_proj(h).view(hsh)).transpose(1, 2)
    v = att.v_proj(h).view(hsh).transpose(1, 2)
    q, k = apply_rotary_pos_emb(q, k, cos, sin)
    kf, vf = update(k, v)
    cum = L.cumulative_length
    ar = torch.arange(MAX_CTX, device=dev)
    keep = ar < (cum - q_len + 1 + torch.arange(q_len, device=dev)[:, None])
    mask = torch.where(keep, 0.0, float('-inf')).to(q.dtype).view(1, 1, q_len, MAX_CTX)
    n_rep = q.shape[1] // kf.shape[1]
    if n_rep > 1:
        kf = kf.repeat_interleave(n_rep, 1)
        vf = vf.repeat_interleave(n_rep, 1)
    o = F.scaled_dot_product_attention(q, kf, vf, attn_mask=mask,
                                       scale=att.scaling)
    o = o.reshape(*ish, -1).contiguous() * torch.sigmoid(gate)
    return att.o_proj(o)

# seed 7 tokens
for i in range(7):
    h = torch.randn(1, 1, H, device=dev, dtype=torch.bfloat16) * 0.05
    fwd(h, cos_all[i].view(1, 1, -1), sin_all[i].view(1, 1, -1))

k_snap = L.keys.clone()
v_snap = L.values.clone()
c_snap = L.cumulative_length.clone()

h1 = torch.randn(1, 1, H, device=dev, dtype=torch.bfloat16) * 0.05
h2 = torch.randn(1, 1, H, device=dev, dtype=torch.bfloat16) * 0.05
t0 = 7

# path A: 2x S=1
L.keys.copy_(k_snap); L.values.copy_(v_snap); L.cumulative_length.copy_(c_snap)
oA1 = fwd(h1, cos_all[t0].view(1, 1, -1), sin_all[t0].view(1, 1, -1))
oA2 = fwd(h2, cos_all[t0 + 1].view(1, 1, -1), sin_all[t0 + 1].view(1, 1, -1))

# path B: S=2
L.keys.copy_(k_snap); L.values.copy_(v_snap); L.cumulative_length.copy_(c_snap)
hh = torch.cat([h1, h2], dim=1)
cos2 = torch.stack([cos_all[t0], cos_all[t0 + 1]]).view(1, 2, -1)
sin2 = torch.stack([sin_all[t0], sin_all[t0 + 1]]).view(1, 2, -1)
oB = fwd(hh, cos2, sin2)

d0 = (oB[0, 0].float() - oA1[0, 0].float()).abs().max().item()
d1 = (oB[0, 1].float() - oA2[0, 0].float()).abs().max().item()
refm = oA1.float().abs().mean().item()
print(f'[fullattn SEEDED] |out| mean {refm:.5f} | tok0 {d0:.6f} tok1 {d1:.6f}')
