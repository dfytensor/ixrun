# -*- coding: utf-8 -*-
"""Gate: C++ ig32p kernel vs torch parametric-decode reference (real 27B)."""
import sys, time
sys.path.insert(0, r'E:\IXRUN')
import pandas  # noqa: F401
import torch
from torch.utils.cpp_extension import load_inline

BLOB = r'F:\models\qwen38_ig32_blob.pt'
src = open(r'E:\IXRUN\ixrun\cpp\engine_v5.cu', encoding='utf-8').read()
src27 = open(r'E:\IXRUN\ixrun\cpp\engine_27b.cu', encoding='utf-8').read()
proto = '''
torch::Tensor ig32p_out(torch::Tensor x, torch::Tensor codes,
    torch::Tensor sign, torch::Tensor aux,
    int64_t out_f, int64_t in_f);
'''
ext = load_inline(name='ixrun_cpp_i32g', cpp_sources=[proto],
                  cuda_sources=[src, src27],
                  functions=['ig32p_out'],
                  extra_cuda_cflags=['-O3', '--use_fast_math',
                                     '-allow-unsupported-compiler'],
                  verbose=False)
print('ext built', flush=True)

B_REL = torch.logspace(-3, 2, 15).cuda()
BETAS = torch.tensor([0.50, 0.55, 0.65, 0.75, 0.85, 0.95, 1.00]).cuda()


def ig32_decode_torch(p, out_f, in_f):
    """Parametric levels from aux (prm + fp16 gmax) -> bf16 weights."""
    nc = in_f // 32
    codes = p['codes'].cuda().long().view(out_f, in_f // 2)
    nib = torch.empty(out_f, in_f, dtype=torch.long, device='cuda')
    nib[:, 0::2] = codes & 0xF
    nib[:, 1::2] = codes >> 4
    aux = p['aux'].cuda().view(out_f, nc, 3)
    prm = aux[:, :, 0].int()                                  # [r, nc]
    gmx = (aux[:, :, 1:3].contiguous().view(torch.float16)
           .float().view(out_f, nc).unsqueeze(-1))            # [r, nc, 1]
    tanh_f = ((prm & 0x80) != 0).unsqueeze(-1)
    b = B_REL[(prm >> 3) & 0xF].unsqueeze(-1) * gmx           # [r, nc, 1]
    beta = BETAS[prm & 7].unsqueeze(-1)
    gb = torch.exp2(beta * torch.log2(gmx))
    fmax = torch.where(tanh_f, torch.tanh(gmx / b), gb / (b + gb))
    fd = (torch.arange(16, device='cuda').view(1, 1, 16) / 15.0) * fmax
    fd = fd.clamp(max=1 - 1e-6)
    W = torch.where(tanh_f, b * torch.atanh(fd), fd * b / (1 - fd))
    W = torch.where(beta != 1.0, W ** (1.0 / beta), W).clamp_min(0)
    W = torch.minimum(W, gmx)                                 # [r, nc, 16]
    idx = nib.view(out_f, nc, 32)
    rec = W.gather(2, idx.clamp(0, 15)).squeeze(-1)           # [r, nc, 32]
    sg = p['sign'].cuda().int().view(out_f, nc, 1)
    sgn = 1 - ((sg >> torch.arange(32, device='cuda')) & 1) * 2
    rec = rec * sgn.float()
    return rec.reshape(out_f, in_f).to(torch.bfloat16)


blob = torch.load(BLOB, map_location='cpu', mmap=True, weights_only=True)
x = torch.randn(5120, device='cuda')
ok = True
for key in ['model.layers.0.mlp.gate_proj',
            'model.layers.0.linear_attn.in_proj_qkv',
            'lm_head']:
    p = blob['layers'][key]
    of, inf = p['out_f'], p['in_f']
    t0 = time.time()
    W = ig32_decode_torch(p, of, inf).float()
    y_ref = W @ x[:inf]
    y = ext.ig32p_out(x[:inf].contiguous(), p['codes'].cuda(),
                      p['sign'].cuda(), p['aux'].cuda(), of, inf).float()
    d = (y - y_ref).abs().max().item()
    scale = y_ref.abs().mean().item()
    ok &= d < 0.05 + 0.05 * scale
    print(f'{key} [{of}x{inf}]: maxdiff {d:.4f} (mean|y| {scale:.3f}) '
          f'[{time.time()-t0:.0f}s]', flush=True)
print('IG32-C++ GATE:', 'PASS' if ok else 'FAIL', flush=True)
