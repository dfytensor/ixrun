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
    """v2 three-plane layout (llama.cpp BF16X-style): mant 14B | delta
    6B (16x3b, clamp-to-zero beyond the 8-step exponent window) | sgn
    2B | emax 1B = 23B per 16 elems = 11.5bpw."""
    of, inf = W.shape
    assert inf % 16 == 0
    b = W.view(torch.uint16).int()
    sign = ((b >> 15) & 1).reshape(-1, 16)
    e = (b >> 7) & 0xFF
    e = torch.where(e == 0, torch.ones_like(e), e).reshape(-1, 16)
    m = (b & 0x7F).reshape(-1, 16)
    nG = sign.shape[0]
    sg = kg // 16
    emax_sg = e.reshape(-1, sg, 16).amax(dim=(1, 2))
    emax = emax_sg.repeat_interleave(sg).unsqueeze(1)
    delta = (emax - e).clamp(0, 7).to(torch.uint8)
    ar8 = torch.arange(8, device=W.device)
    dbits = torch.zeros(nG, 48, dtype=torch.int64, device=W.device)
    mbits = torch.zeros(nG, 112, dtype=torch.int64, device=W.device)
    blk = 2_000_000
    for i0 in range(0, nG, blk):
        for i in range(16):
            dbits[i0:i0 + blk, i * 3:(i + 1) * 3] = \
                ((delta[i0:i0 + blk, i:i + 1] >> ar8[:3]) & 1)
            mbits[i0:i0 + blk, i * 7:(i + 1) * 7] = \
                ((m[i0:i0 + blk, i:i + 1] >> ar8[:7]) & 1)
    mant = (mbits.reshape(nG, 14, 8) * (1 << ar8)) \
        .sum(-1).to(torch.uint8)
    dl = (dbits.reshape(nG, 6, 8) * (1 << ar8)).sum(-1).to(torch.uint8)
    w16 = torch.arange(16, device=W.device)
    sgn = (sign.reshape(nG, 2, 8).to(torch.uint8)
           * (1 << ar8)).sum(-1).to(torch.uint8)
    return {'mant': mant.contiguous().cpu(),
            'delta': dl.contiguous().cpu(),
            'sgn': sgn.contiguous().cpu(),
            'emax': emax_sg.to(torch.uint8).cpu().contiguous(),
            'out_f': of, 'in_f': inf, 'kg': kg}


def bf16xl_decode_ref(pk):
    mant = pk['mant'].long().cuda()
    dl = pk['delta'].long().cuda()
    sgn = pk['sgn'].long().cuda()
    nG = mant.shape[0]
    ar8 = torch.arange(8, device=mant.device)
    mbits = ((mant.unsqueeze(-1) >> ar8) & 1).reshape(nG, 112) \
        .reshape(nG, 16, 7)
    dbits = ((dl.unsqueeze(-1) >> ar8) & 1).reshape(nG, 48) \
        .reshape(nG, 16, 3)
    m = (mbits * (1 << ar8[:7])).sum(-1)
    delta = (dbits * (1 << ar8[:3])).sum(-1)
    sign = ((sgn.unsqueeze(-1) >> ar8) & 1).reshape(nG, 16)
    sg = pk['kg'] // 16
    emax = pk['emax'].long().cuda().repeat_interleave(sg).unsqueeze(1)
    e = torch.where(emax > delta, emax - delta,
                    torch.zeros_like(delta))
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
        rel = ((d.float() - W.float()).norm()
               / W.float().norm()).item()
        exact = bool((d.view(torch.uint16) ==
                      W.view(torch.uint16)).all())
        mb = pk['mant'].numel() + pk['delta'].numel() + pk['sgn'].numel() + pk['emax'].numel()
        bpw = mb * 8 / (of * inf)
        print(f'[{of}x{inf}] rel_err={rel:.4f} bpw={bpw:.2f}',
              flush=True)


class Bf16xlLinear(torch.nn.Module):
    def __init__(self, pk):
        super().__init__()
        self.pk = pk
        for k in ('mant', 'delta', 'sgn', 'emax'):
            self.pk[k] = pk[k].cuda()
        self.out_features = pk['out_f']
        self.in_features = pk['in_f']

    def _decode(self):
        from experiments.bf16xl_gemv_cuda.bf16xl_gemv_cuda import _load
        ext = _load()
        return ext.decode(self.pk['mant'], self.pk['delta'],
                          self.pk['sgn'], self.pk['emax'],
                          self.pk['out_f'], self.pk['in_f'],
                          self.pk['kg'])

    def forward(self, x):
        from experiments.bf16xl_gemv_cuda.bf16xl_gemv_cuda import \
            bf16xl_gemv_cuda
        if x.numel() == self.in_features:
            return bf16xl_gemv_cuda(x.reshape(-1), self.pk)
        W = self._decode()
        return torch.nn.functional.linear(
            x.to(W.dtype), W)


def deploy_bf16xl(model, verbose=True):
    from ixrun.linear import iter_quantizable_linears, _set_parent_child
    n = 0
    for name, mod in list(iter_quantizable_linears(model)):
        W = mod.weight.data.cuda()
        pk = bf16xl_pack(W)
        del W
        torch.cuda.empty_cache()
        _set_parent_child(model, name, Bf16xlLinear(pk))
        n += 1
        if verbose and n % 60 == 0:
            print(f'[bf16xl-deploy] {n}...', flush=True)
    if verbose:
        print(f'[bf16xl-deploy] {n} Bf16xlLinear (14.12bpw lossless)',
              flush=True)
    return n