# -*- coding: utf-8 -*-
"""Check HPQ code distribution: can codes be quantized to 4 bits (16
values) without loss? If the 64 k-means centers are concentrated (LLM
subspace vectors are), top-16 coverage decides."""
import sys
sys.path.insert(0, r'E:\IXRUN')
import pandas
import torch

from transformers import AutoModelForCausalLM
from ixrun.config import MODEL_PATH

BS, DIM = 4, 16


def kmeans_gpu(X, k, iters=25, blk=200_000):
    n = X.shape[0]
    idx = torch.randperm(n, device=X.device)[:k]
    C = X[idx].clone()
    for _ in range(iters):
        assign = torch.empty(n, dtype=torch.long, device=X.device)
        for b0 in range(0, n, blk):
            assign[b0:b0 + blk] = torch.cdist(X[b0:b0 + blk], C).argmin(1)
        for j in range(k):
            m = assign == j
            if m.any():
                C[j] = X[m].mean(0)
    return C, assign


m = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, torch_dtype=torch.bfloat16, trust_remote_code=True)
m.eval()
for name, mod in m.named_modules():
    if name == 'model.layers.3.self_attn.o_proj':
        W = mod.weight.data.float().cuda()
        break
H, Wd = W.shape
blocks = W.reshape(H // BS, BS, Wd // BS, BS).permute(0, 2, 1, 3) \
    .reshape(-1, DIM)
for mq in (8, 4):
    sub_dim = DIM // mq
    total_active = []
    total_top16 = []
    for i in range(mq):
        sub = blocks[:, i * sub_dim:(i + 1) * sub_dim]
        C, a = kmeans_gpu(sub, 64)
        hist = torch.bincount(a, minlength=64).float()
        active = (hist > 0).sum().item()
        top16 = hist.sort(descending=True).values[:16].sum() / hist.sum()
        total_active.append(active)
        total_top16.append(top16.item())
    print(f'm={mq}: active centers per subspace {total_active} '
          f'(of 64) | top-16 coverage {[round(t, 3) for t in total_top16]}',
          flush=True)
