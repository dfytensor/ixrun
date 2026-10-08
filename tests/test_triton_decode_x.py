# -*- coding: utf-8 -*-
"""DECISIVE: real Triton decode of blob in_proj_qkv vs my decode."""
import sys
sys.path.insert(0, r'E:\IXRUN')
import pandas  # noqa: F401
import torch
from ixrun.udcq import decode_udcq_triton

BLOB = r'E:\IXRUN\experiments\qwen38_udcq\q38_blob.pt'
blob = torch.load(BLOB, map_location='cpu', mmap=True,
                  weights_only=True)
p = dict(blob['layers']['model.layers.0.linear_attn.in_proj_qkv'])
N = p['idx'].numel() * 2
p.update({'g': 16, 'sign_packed': p['sign'], 'N': N, 'out_f': 10240, 'in_f': 5120,
          'codebook': blob['codebook']})

# real Triton decode (the PY engine's exact path)
W_triton = decode_udcq_triton(p).float()

# my decode
GROUP = 16
idx, sign, scale = p['idx'], p['sign'], p['scale']
bb = idx.cpu().long()
nib = torch.stack([bb & 0x0F, (bb >> 4) & 0x0F], 1).reshape(-1)
bit = ((sign.cpu().long().unsqueeze(1) >> torch.arange(32))
       & 1).reshape(-1)[:N]
W_mine = (blob['codebook'].float().double()[nib]
          * scale.cpu().double().repeat_interleave(GROUP)
          * (bit * 2.0 - 1.0)).reshape(10240, 5120).float()

W_mine = W_mine.cuda()
d = (W_triton - W_mine).abs()
rel = d.norm() / W_triton.norm()
nz = (d > 1e-2).sum().item()
print(f'triton-vs-mine: rel {rel.item():.2e} '
      f'maxdiff {d.max().item():.4f} mismatches>0.01: {nz}',
      flush=True)
if nz > 0:
    k = d.argmax().item()
    r, cc = k // 5120, k % 5120
    print(f'  worst ({r},{cc}): triton '
          f'{W_triton.flatten()[k].item():.4f} vs mine '
          f'{W_mine.flatten()[k].item():.4f}', flush=True)
    # check neighbors: is it a sign-flip or a scale-group shift?
    print(f'  mine[r, cc-1..cc+1]: '
          f'{W_mine[r, max(0, cc-1):cc+2].cpu().tolist()}', flush=True)
    print(f'  triton same: '
          f'{W_triton[r, max(0, cc-1):cc+2].tolist()}', flush=True)
