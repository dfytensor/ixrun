# -*- coding: utf-8 -*-
"""Conv-path probe: seeded conv state, S=2 block (cat + causal_conv1d_fn)
vs 2x S=1 (causal_conv1d_update). Decides if the conv branch is the culprit."""
import sys
sys.setrecursionlimit(20000)
sys.path.insert(0, r'E:\IXRUN')
import torch
from ixrun.fla_patch import apply_fla_kernels
apply_fla_kernels()

from transformers.models.qwen3_5 import modeling_qwen3_5 as MQ

torch.manual_seed(0)
dev = 'cuda'
B, dim, K = 1, 4096, 4          # conv dim ~ key*2+value, kernel 4
w = (torch.randn(dim, K, device=dev) * 0.05).bfloat16()   # [dim, K] layout?
bias = None
act = 'silu'

# seeded conv state: last K-1 tokens
state = torch.randn(B, dim, K - 1, device=dev, dtype=torch.bfloat16) * 0.1
new2 = torch.randn(B, dim, 2, device=dev, dtype=torch.bfloat16) * 0.1

# path A: two S=1 updates (proven decode path)
stateA = state.clone()
outA = []
for t in range(2):
    x1 = new2[:, :, t:t + 1]
    y = MQ.causal_conv1d_update(x1, stateA, w, bias, act)
    outA.append(y)
outA = torch.cat(outA, dim=-1)

# path B: cat + full conv, keep last 2 (the model's S>1 branch)
stateB = state.clone()
full = torch.cat([stateB, new2], dim=-1)
yB = MQ.causal_conv1d_fn(full, w, bias, activation=act)[:, :, -2:]

d = (outA.float() - yB.float()).abs().max().item()
ref = outA.float().abs().mean().item()
print(f'[conv A/B] |out| mean {ref:.5f} | block-vs-sequential max {d:.6f}')
# also check the state advanced identically
print(f'[conv state] {float((stateA.float()-stateB.float()).abs().max()):.6f}')
