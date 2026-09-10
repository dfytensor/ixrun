# -*- coding: utf-8 -*-
"""5-bit GMM codec: 4-bit nibble (low bits) + 1-bit high stream (same
bitmap layout as UDCQ's sign stream) + per-16 fp16 scale = 6.03 bpw.

Kernel = UDCQ GEMV variant where the sign bit is reinterpreted as index
bit 4 (codebook carries the sign) and the sgn multiply is dropped.

MUST pass the bit-exact / near-exact unit test (AGENTS rule) before any
model deployment. Self-test: python -X utf8 -m benchmarks.gmm5
"""
import sys
sys.path.insert(0, r'E:\IXRUN')
import pandas
import torch
import triton
import triton.language as tl

from benchmarks.gmm_stream_minicpm5 import fit_bayesian_gmm, _nearest

GROUP = 16
K = 32


@triton.jit
def _gmm5_gemv_kernel(
    x_ptr, y_ptr, idx_ptr, high_ptr, scale_ptr, cb_ptr,
    IN_F: tl.constexpr, OUT_F: tl.constexpr, GROUP: tl.constexpr,
    BK: tl.constexpr, R: tl.constexpr,
):
    pid = tl.program_id(0)
    rows = pid * R + tl.arange(0, R)
    base = rows.to(tl.int64) * IN_F
    acc = tl.zeros((R,), tl.float32)
    for k0 in tl.range(0, IN_F, BK):
        kidx = k0 + tl.arange(0, BK)
        offs = base[:, None] + kidx[None, :]
        b = tl.load(idx_ptr + offs // 2)
        low = tl.where(offs % 2 == 0, b & 0x0F, (b >> 4) & 0x0F)
        hw = tl.load(high_ptr + offs // 32).to(tl.uint32)
        hi = (hw >> (offs % 32)) & 1
        i = (low.to(tl.int32) | (hi.to(tl.int32) << 4))
        val = tl.load(cb_ptr + i).to(tl.float32)
        sc = tl.load(scale_ptr + offs // GROUP).to(tl.float32)
        w = val * sc
        xv = tl.load(x_ptr + kidx).to(tl.float32)
        acc += tl.sum(w * xv[None, :], axis=1)
    tl.store(y_ptr + rows, acc.to(tl.bfloat16))


def gmm5_fused_gemv(x, idx, high, scale, cb, out_f, in_f,
                    g=GROUP, r=4, bk=256, num_warps=2):
    y = torch.empty(out_f, dtype=torch.bfloat16, device=x.device)
    while r > 1 and out_f % r != 0:
        r //= 2
    assert in_f % bk == 0
    _gmm5_gemv_kernel[(out_f // r,)](
        x.reshape(-1), y, idx, high, scale, cb,
        IN_F=in_f, OUT_F=out_f, GROUP=g, BK=bk, R=r,
        num_warps=num_warps)
    return y


def gmm5_pack(w, mu, group=GROUP):
    flat = w.reshape(-1).float().cpu()
    N = flat.numel()
    pad = (-N) % group
    if pad:
        flat = torch.cat([flat, flat.new_zeros(pad)])
    g = flat.view(-1, group)
    sc = (g.abs().amax(1) / float(mu.abs().max())).clamp_min(1e-12)
    y = g / sc[:, None]
    idx = _nearest(y, mu).reshape(-1)[:N]          # 0..31
    low = (idx & 0xF).to(torch.uint8)
    hi = (idx >> 4).to(torch.int32)
    if N % 2 == 0:
        b = low[0::2] | (low[1::2] << 4)
    else:
        b = torch.cat([low[0::2] | (low[1::2] << 4), low[-1:]])
    nw = (N + 31) // 32
    hp = torch.zeros(nw, dtype=torch.int32)
    for bit in range(32):
        sel = hi[bit::32]
        hp[:sel.numel()] |= sel << bit
    return {
        'g': group, 'out_f': w.shape[0], 'in_f': w.shape[1], 'N': N,
        'idx': b.to(torch.uint8), 'sign_packed': hp,
        'scale': sc.to(torch.float16),
        'codebook': mu.to(torch.float16).clone(),
        'bits_per_weight': 4 + 1 + 16 / group,
    }


def gmm5_reference(packed, group=GROUP):
    """Nibble+high unpack -> mu[idx] * scale (fp16 values)."""
    b = packed['idx']
    N = packed['N']
    low = torch.empty(N, dtype=torch.long)
    low[0::2] = (b & 0x0F).long()[:len(low[0::2])]
    low[1::2] = (b >> 4).long()[:len(low[1::2])]
    hp = packed['sign_packed']
    hi = torch.empty(N, dtype=torch.long)
    for bit in range(32):
        sel = (hp >> bit) & 1
        hi[bit::32] = sel[:len(hi[bit::32])].long()
    idx = low | (hi << 4)
    sc = packed['scale'].float()
    mu = packed['codebook'].float()
    return mu[idx] * sc[torch.arange(N) // group]


if __name__ == '__main__':
    torch.manual_seed(0)
    for out_f, in_f in [(2048, 4096), (5120, 5120), (17408, 5120)]:
        W = torch.randn(out_f, in_f) * 0.02
        wf = W.reshape(-1)
        if wf.numel() > 3_000_000:
            wf = wf[torch.randint(0, wf.numel(), (3_000_000,))]
        mu, _, _ = fit_bayesian_gmm(wf, K=K, iters=20)
        packed = gmm5_pack(W, mu)
        ref = gmm5_reference(packed)[:W.numel()].reshape(W.shape)
        x = (torch.randn(in_f) * 0.5).to(torch.bfloat16).cuda()
        for k in ('idx', 'sign_packed', 'scale', 'codebook'):
            packed[k] = packed[k].cuda()
        y_k = gmm5_fused_gemv(x, packed['idx'], packed['sign_packed'],
                              packed['scale'], packed['codebook'],
                              out_f, in_f)
        y_ref = (ref.float().cuda() @ x.float())
        d = (y_k.float() - y_ref).abs().max().item()
        print(f'{out_f}x{in_f}: kernel-vs-ref dmax={d:.6f} '
              f'(|y| mean {y_ref.abs().mean().item():.3f}) '
              f'bpw={packed["bits_per_weight"]:.2f}', flush=True)
