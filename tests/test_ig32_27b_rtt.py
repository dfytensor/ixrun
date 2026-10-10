# -*- coding: utf-8 -*-
"""27B layer-level roundtrip error: ig32-params vs bf16 original."""
import sys, json
sys.path.insert(0, r'E:\IXRUN')
import pandas  # noqa: F401
import torch
from safetensors import safe_open

BLOB = r'F:\models\qwen38_ig32_blob.pt'
MODEL = r'E:\models\Qwen3.8-27B'
blob = torch.load(BLOB, map_location='cpu', mmap=True, weights_only=True)
index = json.load(open(MODEL + r'\model.safetensors.index.json',
                       encoding='utf-8'))['weight_map']
B_REL = torch.logspace(-3, 2, 15).cuda()
BETAS = torch.tensor([0.50, 0.55, 0.65, 0.75, 0.85, 0.95, 1.00]).cuda()


def ig32_decode(p, out_f, in_f):
    nc = in_f // 32
    codes = p['codes'].cuda().long().view(out_f, in_f // 2)
    nib = torch.empty(out_f, in_f, dtype=torch.long, device='cuda')
    nib[:, 0::2] = codes & 0xF
    nib[:, 1::2] = codes >> 4
    aux = p['aux'].cuda().view(out_f, nc, 3)
    prm = aux[:, :, 0].int()
    gmx = (aux[:, :, 1:3].contiguous().view(torch.float16)
           .float().view(out_f, nc).unsqueeze(-1))
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
    W = torch.where(tanh_f, b * torch.atanh(fd), w_rat)
    W = W.clamp_min(0)
    W = torch.minimum(W, gmx)
    idx = nib.view(out_f, nc, 32)
    rec = W.gather(2, idx.clamp(0, 15)).squeeze(-1)
    sg = p['sign'].cuda().int().view(out_f, nc, 1)
    sgn = 1 - ((sg >> torch.arange(32, device='cuda')) & 1) * 2
    return (rec * sgn.float()).reshape(out_f, in_f)


for skey, bkey in [
        ('model.language_model.layers.0.mlp.gate_proj.weight',
         'model.layers.0.mlp.gate_proj'),
        ('model.language_model.layers.0.linear_attn.in_proj_qkv.weight',
         'model.layers.0.linear_attn.in_proj_qkv'),
        ('model.language_model.layers.3.self_attn.q_proj.weight',
         'model.layers.3.self_attn.q_proj')]:
    with safe_open(MODEL + '\\' + index[skey], framework='pt') as f:
        Wb = f.get_tensor(skey).float()
    p = blob['layers'][bkey]
    of, inf = p['out_f'], p['in_f']
    Wq = ig32_decode(p, of, inf)
    # UDCQ decode reference from the ORIGINAL blob
    ub = torch.load(r'E:\IXRUN\experiments\qwen38_udcq\q38_blob.pt', map_location='cpu', mmap=True, weights_only=True)
    up = ub['layers'][bkey]
    cb0 = ub['codebook'].float().cuda()
    uidx = up['idx'].long().view(of, inf // 2).cuda()
    unib = torch.empty(of, inf, dtype=torch.long, device='cuda')
    unib[:, 0::2] = uidx & 0xF
    unib[:, 1::2] = uidx >> 4
    usg = up['sign'].int().view(of, inf // 32, 1).cuda()
    ubit = (usg >> torch.arange(32, device='cuda')) & 1
    usgn = ubit.reshape(of, inf).float() * 2 - 1
    usc = up['scale'].float().view(of, inf // 16).cuda().repeat_interleave(16, dim=1)
    Wu = (cb0[unib] * usc * usgn)
    print(f'  UDCQ rel vs bf16: {(Wu.cpu() - Wb).norm() / Wb.norm():.4f}', flush=True)
    del ub, up, cb0, uidx, unib, usg, ubit, usgn, usc, Wu
    import gc; gc.collect(); torch.cuda.empty_cache()
    xr = torch.randn(inf).cuda()
    yb = Wb.cuda() @ xr
    yq = Wq @ xr
    e_gemv = ((yq - yb).norm() / yb.norm()).item()
    e_w = ((Wq.cpu() - Wb).norm() / Wb.norm()).item()
    # bias check: mean values
    print(f'{bkey}: weight rel {e_w:.4f} gemv rel {e_gemv:.4f} '
          f'| bf16 mean {Wb.mean():+.5f} rec mean {Wq.mean().item():+.5f}',
          flush=True)
