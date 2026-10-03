# -*- coding: utf-8 -*-
"""bf16x-lossless runtime (mb=7, KG=64): TRUE lossless compressed-bf16.

Per 16-elem group: 28B sequential bit stream, LSB-first per element:
  val14 = (delta << 8) | (mant7 << 1) | sign     (delta 6b, 63 = zero)
Per 4-group supergroup (64 elems): one emax byte.
Reconstruction: bf16(sign, emax - delta, m) == original bits exactly.
"""
import sys

import torch


def bf16xl_pack(W, kg=64):
    of, inf = W.shape
    assert inf % 16 == 0 and of % 1 == 0
    b = W.view(torch.uint16).int()
    sign = (b >> 15) & 1
    e = (b >> 7) & 0xFF
    m = b & 0x7F
    e_nz = torch.where(e == 0, torch.ones_like(e), e)
    flat = e_nz.reshape(-1)
    nG = flat.numel() // 16
    sg = kg // 16                      # groups per supergroup
    emax_sg = flat[:nG * 16].reshape(-1, sg, 16).amax(dim=(1, 2))
    emax = emax_sg.repeat_interleave(sg * 16)
    delta = (emax - flat[:nG * 16]).clamp_min(0)
    delta = torch.where(flat[:nG * 16] == 0,
                        torch.full_like(delta, 63), delta)
    maxd = int(delta.max())
    assert maxd <= 62, f'delta {maxd} exceeds 62 (zero-marker 63); ' \
        'this tensor needs the dbits=7 fallback (not implemented)'
    payload = (delta.reshape(-1, 1) << 8) | (m.reshape(-1, 1) << 1) \
        | sign.reshape(-1, 1)
    payload = payload[:nG * 16].reshape(nG, 16)
    ar8 = torch.arange(8, device=W.device)
    bits = torch.zeros(nG, 224, dtype=torch.int64, device=W.device)
    blk = 2_000_000
    for i0 in range(0, nG, blk):
        b_ = payload[i0:i0 + blk]
        bt = torch.zeros(b_.shape[0], 224, dtype=torch.int64,
                         device=W.device)
        for i in range(16):
            bt[:, i * 14:(i + 1) * 14] = \
                (b_[:, i:i + 1] >> torch.arange(14, device=W.device)) & 1
        bits[i0:i0 + blk] = bt
    stream = (bits.reshape(nG, 28, 8) *
              (1 << ar8)).sum(-1).to(torch.uint8)
    return {'stream': stream.contiguous().cpu(),
            'emax': emax_sg.to(torch.uint8).cpu().contiguous(),
            'out_f': of, 'in_f': inf, 'kg': kg}


def bf16xl_decode_ref(pk):
    stream = pk['stream'].long().cuda()
    nG = stream.shape[0]
    ar8 = torch.arange(8, device=stream.device)
    bits = ((stream.unsqueeze(-1) >> ar8) & 1) \
        .reshape(nG, 224).reshape(nG, 16, 14)
    val = (bits * (1 << torch.arange(14, device=stream.device))).sum(-1)
    delta = (val >> 8) & 0x3F
    m = (val >> 1) & 0x7F
    sign = val & 1
    sg = pk['kg'] // 16
    emax = pk['emax'].long().cuda().repeat_interleave(sg)
    e = torch.where(delta == 63, torch.zeros_like(delta),
                    emax[:, None] - delta)
    bits16 = (sign << 15) | (e << 7) | m
    of, inf = pk['out_f'], pk['in_f']
    return bits16.to(torch.uint16).view(torch.bfloat16) \
        .reshape(of, inf)


if __name__ == '__main__':
    torch.manual_seed(0)
    for of, inf in [(512, 512), (1536, 4608)]:
        W = (torch.randn(of, inf) * 0.02).to(torch.bfloat16).cuda()
        pk = bf16xl_pack(W)
        d = bf16xl_decode_ref(pk)
        exact = bool((d.view(torch.uint16) ==
                      W.view(torch.uint16)).all())
        mb = pk['stream'].numel() + pk['emax'].numel()
        bpw = mb * 8 / (of * inf)
        print(f'[{of}x{inf}] bit-exact={exact} bpw={bpw:.2f}',
              flush=True)
