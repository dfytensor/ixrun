# -*- coding: utf-8 -*-
"""Isolate: pack_chunk logic vs blob serialization on a real 27B slice."""
import sys, json
sys.path.insert(0, r'E:\IXRUN')
import pandas  # noqa: F401
import torch
from safetensors import safe_open
from experiments.repack_ig32_blob import pack_chunk, deq_udcq

OLD = r'E:\IXRUN\experiments\qwen38_udcq\q38_blob.pt'
NEW = r'F:\models\qwen38_ig32_blob.pt'
MODEL = r'E:\models\Qwen3.8-27B'
KEY = 'model.layers.0.mlp.gate_proj'
SKEY = 'model.language_model.layers.0.mlp.gate_proj.weight'
RS = 512

ub = torch.load(OLD, map_location='cpu', mmap=True, weights_only=True)
nb = torch.load(NEW, map_location='cpu', mmap=True, weights_only=True)
import experiments.repack_ig32_blob as RP
RP.CB = ub['codebook'].float().cuda()
up = ub['layers'][KEY]
of = up['idx'].numel() * 2 // 5120
W = deq_udcq(up, of, 5120)[:RS]
c2, s2, a2 = pack_chunk(W)

np_ = nb['layers'][KEY]
c1 = np_['codes'][:RS]
s1 = np_['sign'][:RS]
a1 = np_['aux'][:RS]
print('codes identical:', torch.equal(c1, c2))
print('sign identical:', torch.equal(s1, s2))
print('aux identical:', torch.equal(a1, a2))

index = json.load(open(MODEL + r'\model.safetensors.index.json',
                       encoding='utf-8'))['weight_map']
with safe_open(MODEL + '\\' + index[SKEY], framework='pt') as f:
    Wb = f.get_tensor(SKEY).float()[:RS]

B_REL = torch.logspace(-3, 2, 15).cuda()
BETAS = torch.tensor([0.50, 0.55, 0.65, 0.75, 0.85, 0.95, 1.00]).cuda()


def dec(p_codes, p_sign, p_aux, of_, inf_):
    nc = inf_ // 32
    codes = p_codes.cuda().long().view(of_, inf_ // 2)
    nib = torch.empty(of_, inf_, dtype=torch.long, device='cuda')
    nib[:, 0::2] = codes & 0xF
    nib[:, 1::2] = codes >> 4
    aux = p_aux.cuda().view(of_, nc, 3)
    prm = aux[:, :, 0].int()
    gmx = (aux[:, :, 1:3].contiguous().view(torch.float16)
           .float().view(of_, nc).unsqueeze(-1))
    tanh_f = ((prm & 0x80) != 0).unsqueeze(-1)
    TANH_B = torch.tensor([0.25, 0.4, 0.6, 0.85, 1.2, 1.8, 3.0]).cuda()
    fidx = (prm >> 3) & 0xF
    brev = torch.where(tanh_f,
                       TANH_B[fidx.clamp(max=6)].unsqueeze(-1),
                       B_REL[fidx].unsqueeze(-1))
    b = brev * gmx
    beta = BETAS[prm & 7].unsqueeze(-1)
    gb = torch.exp2(beta * torch.log2(gmx))
    fmax = torch.where(tanh_f, torch.tanh(gmx / b), gb / (b + gb))
    fd = (torch.arange(16, device='cuda').view(1, 1, 16) / 15.0) * fmax
    fd = fd.clamp(max=1 - 1e-6)
    w_rat = fd * b / (1 - fd)
    w_rat = torch.where(beta != 1.0, w_rat ** (1.0 / beta), w_rat)
    Wl = torch.where(tanh_f, b * torch.atanh(fd), w_rat).clamp_min(0)
    Wl = torch.minimum(Wl, gmx)
    idx = nib.view(of_, nc, 32)
    rec = Wl.gather(2, idx.clamp(0, 15)).squeeze(-1)
    sg = p_sign.cuda().int().view(of_, nc, 1)
    sgn = 1 - ((sg >> torch.arange(32, device='cuda')) & 1) * 2
    return (rec * sgn.float()).reshape(of_, inf_)


Wq = dec(np_['codes'][:RS], np_['sign'][:RS], np_['aux'][:RS], RS, 5120)
print('blob decode vs bf16 rel:',
      ((Wq.cpu() - Wb).norm() / Wb.norm()).item())
print('blob decode vs udcq-dequant rel:',
      ((Wq.cpu() - W.cpu()).norm() / W.norm()).item())
