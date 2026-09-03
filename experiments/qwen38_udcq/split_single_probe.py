# -*- coding: utf-8 -*-
"""Single-layer seeded A/B WITH the split patch — decisive: if the split
reproduces path A's exact recurrent calls, diff must be ~0."""
import sys
sys.setrecursionlimit(20000)
sys.path.insert(0, r'E:\IXRUN')
import torch
from ixrun.fla_patch import apply_fla_kernels
from ixrun.gdn_seq_patch import apply_gdn_sequential_patch
apply_fla_kernels()
apply_gdn_sequential_patch(verbose=True)

from transformers.cache_utils import StaticCache
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5GatedDeltaNet
from transformers.models.qwen3_5 import Qwen3_5Config
import copy

dev = 'cuda'
torch.manual_seed(0)

# minimal GatedDeltaNet on real-ish config shapes
cfg = Qwen3_5Config(text_config_kwargs=None) if False else None
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextConfig
tc = Qwen3_5TextConfig(
    hidden_size=512, linear_num_key_heads=4, linear_num_value_heads=8,
    linear_key_head_dim=64, linear_value_head_dim=64,
    linear_conv_kernel_dim=4, rms_norm_eps=1e-6)
gdn = Qwen3_5GatedDeltaNet(tc, layer_idx=0).to(dev).to(torch.bfloat16).eval()

cache = StaticCache(config=tc, max_cache_len=32, device=dev)

H = 512
# seed with 7 tokens (S=1, proven path)
seeds = [torch.randn(1, 1, H, device=dev, dtype=torch.bfloat16) * 0.05
         for _ in range(7)]
for s in seeds:
    gdn(s, cache_params=cache)

snap = ([c.clone() if isinstance(c, torch.Tensor) else c
         for c in cache.layers[0].conv_states],
        [r.clone() if isinstance(r, torch.Tensor) else r
         for r in cache.layers[0].recurrent_states],
        cache.layers[0].cumulative_length.clone()
        if hasattr(cache.layers[0], 'cumulative_length') else None)

h1 = torch.randn(1, 1, H, device=dev, dtype=torch.bfloat16) * 0.05
h2 = torch.randn(1, 1, H, device=dev, dtype=torch.bfloat16) * 0.05

# path A: 2x S=1
for dst, src in zip(cache.layers[0].conv_states, snap[0]):
    if isinstance(dst, torch.Tensor) and isinstance(src, torch.Tensor):
        dst.copy_(src)
for dst, src in zip(cache.layers[0].recurrent_states, snap[1]):
    if isinstance(dst, torch.Tensor) and isinstance(src, torch.Tensor):
        dst.copy_(src)
o1 = gdn(h1, cache_params=cache)
o2 = gdn(h2, cache_params=cache)

# path B: 1x S=2 (split patch active)
for dst, src in zip(cache.layers[0].conv_states, snap[0]):
    if isinstance(dst, torch.Tensor) and isinstance(src, torch.Tensor):
        dst.copy_(src)
for dst, src in zip(cache.layers[0].recurrent_states, snap[1]):
    if isinstance(dst, torch.Tensor) and isinstance(src, torch.Tensor):
        dst.copy_(src)
hh = torch.cat([h1, h2], dim=1)
ob = gdn(hh, cache_params=cache)

d0 = (ob[0, 0].float() - o1[0, 0].float()).abs().max().item()
d1 = (ob[0, 1].float() - o2[0, 0].float()).abs().max().item()
refm = o1.float().abs().mean().item()
print(f'[SPLIT single-layer] |out| mean {refm:.5f} | tok0 {d0:.6f} '
      f'tok1 {d1:.6f}  (pre-patch: 0.41 / 3.06)')
