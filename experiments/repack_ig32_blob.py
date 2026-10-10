# -*- coding: utf-8 -*-
"""Repack the 27B UDCQ blob -> ig32-params (group=32, nlev=16, no Lloyd).

For each matrix: dequant UDCQ (GPU) -> ig32_pack in row chunks (memory) ->
codes [out,in/2] u8 + sign [out,in/32] i32 + aux [out,in/32,3] u8
(aux per group = prm byte + fp16 gmax LE).
Output: F:\models\qwen38_ig32_blob.pt (~17GB).
"""
import sys, time
sys.path.insert(0, r'E:\IXRUN')
import pandas  # noqa: F401
import torch

BLOB_IN = r'E:\IXRUN\experiments\qwen38_udcq\q38_blob.pt'
BLOB_OUT = r'F:\models\qwen38_ig32_blob.pt'
CHUNK_ROWS = 8192          # rows per pack chunk

B_REL = torch.logspace(-3, 2, 15)
BETAS = torch.tensor([0.50, 0.55, 0.65, 0.75, 0.85, 0.95, 1.00])
TANH_B = [0.25, 0.4, 0.6, 0.85, 1.2, 1.8, 3.0]
NLEV = 16


def pack_chunk(W, group=32, nlev=16):
    """ig32 quantize WITHOUT Lloyd -> codes, sign word, aux bytes."""
    of, inf = W.shape
    G = W.float().view(-1, group)
    signb = (G < 0)
    A = G.abs()
    gmax = A.amax(1, keepdim=True).clamp_min(1e-12)
    nm1 = nlev - 1
    best_mse = torch.full((A.shape[0],), float('inf'), device=A.device)
    best_lv = torch.zeros(A.shape[0], nlev, device=A.device)
    best_prm = torch.zeros(A.shape[0], dtype=torch.uint8, device=A.device)
    rng = torch.arange(nlev, device=A.device, dtype=torch.float32)

    for bi, br in enumerate(B_REL):
        b = br.item() * gmax
        for bj, beta in enumerate(BETAS):
            be = beta.item()
            f = A ** be / (b + A ** be)
            fmax = gmax ** be / (b + gmax ** be)
            q = torch.round(f / fmax * nm1).clamp_(0, nm1)
            fd = (q / nm1 * fmax).clamp_(max=1 - 1e-6)
            wr = (fd * b / (1 - fd)).clamp_min(0)
            if be != 1.0:
                wr = wr ** (1.0 / be)
            wr = torch.minimum(wr, gmax)
            mse = ((A - wr) ** 2).mean(dim=1)
            imp = mse < best_mse
            if imp.any():
                best_mse = torch.where(imp, mse, best_mse)
                fdi = (rng / nm1).view(1, -1) * fmax
                fdi = fdi.clamp(max=1 - 1e-6)
                lv = (fdi * b / (1 - fdi)).clamp_min(0)
                if be != 1.0:
                    lv = lv ** (1.0 / be)
                lv = torch.minimum(lv, gmax)
                best_lv = torch.where(imp.view(-1, 1), lv, best_lv)
                pv = ((bi & 0xF) << 3) | (bj & 7)
                best_prm = torch.where(
                    imp, torch.full_like(best_prm, pv), best_prm)

    for ti, br in enumerate(TANH_B):
        b = br * gmax
        f = torch.tanh(A / b)
        fmax = torch.tanh(gmax / b)
        q = torch.round(f / fmax * nm1).clamp_(0, nm1)
        fd = (q / nm1 * fmax).clamp_(max=1 - 1e-6)
        wr = (b * torch.atanh(fd)).clamp_min(0)
        wr = torch.minimum(wr, gmax)
        mse = ((A - wr) ** 2).mean(dim=1)
        imp = mse < best_mse
        if imp.any():
            best_mse = torch.where(imp, mse, best_mse)
            fdi = (rng / nm1).view(1, -1) * fmax
            fdi = fdi.clamp(max=1 - 1e-6)
            lv = (b * torch.atanh(fdi)).clamp_min(0)
            lv = torch.minimum(lv, gmax)
            best_lv = torch.where(imp.view(-1, 1), lv, best_lv)
            pv = 0x80 | ((ti & 0xF) << 3)
            best_prm = torch.where(
                imp, torch.full_like(best_prm, pv), best_prm)

    d = (A.unsqueeze(-1) - best_lv.unsqueeze(1)).abs()
    idx = d.argmin(-1)                                   # [nG, 32]

    # codes: 32 x 4-bit -> 16 B per group (low nibble = even element)
    nG = idx.shape[0]
    bits = ((idx.unsqueeze(-1) >> torch.arange(
        4, device=A.device, dtype=torch.int64)) & 1)
    c16 = bits.reshape(nG, 16, 8)
    ar8 = torch.arange(8, device=A.device)
    codes = (c16 * (1 << ar8)).sum(-1).to(torch.uint8)   # [nG,16]
    sb = signb.to(torch.int64)
    signw = (sb * (1 << torch.arange(32, device=A.device))).sum(-1)
    signw = signw.to(torch.int32)                        # [nG]

    # aux [nG,3]: prm byte + fp16 gmax (LE, 2 bytes)
    aux = torch.empty(nG, 3, dtype=torch.uint8, device=A.device)
    aux[:, 0] = best_prm
    aux[:, 1:3] = gmax.half().view(torch.uint8).view(nG, 2)
    return (codes.view(of, inf // 2).cpu(),
            signw.view(of, inf // 32).cpu(),
            aux.view(of, inf // 32, 3).cpu())


def deq_udcq(p, out_f, in_f, dev='cuda'):
    idx = p['idx'].long().view(out_f, in_f // 2).to(dev)
    nib = torch.empty(out_f, in_f, dtype=torch.long, device=dev)
    nib[:, 0::2] = idx & 0xF
    nib[:, 1::2] = idx >> 4
    sg = p['sign'].int().view(out_f, in_f // 32, 1).to(dev)
    bits = (sg >> torch.arange(32, dtype=torch.int, device=dev)) & 1
    sgn = bits.reshape(out_f, in_f).float() * 2 - 1
    sc = p['scale'].float().view(out_f, in_f // 16).to(dev)
    sc = sc.repeat_interleave(16, dim=1)
    return (CB[nib] * sc * sgn).to(torch.bfloat16)


def _main():
    t0 = time.time()
    blob = torch.load(BLOB_IN, map_location='cpu', mmap=True, weights_only=True)
    CB = blob['codebook'].float().cuda()
    H = 5120
    out = {'embed': blob['embed'], 'codebook': blob['codebook'],
           'layers': {}, 'format': 'ig32-g32-n16-nolloyd'}
    keys = list(blob['layers'].keys())
    n_done = 0
    for key in keys:
        p = blob['layers'][key]
        if key == 'lm_head':
            inf = H
            out_f = p['idx'].numel() * 2 // inf
        else:
            out_f = p['sign'].numel() // (p['idx'].numel() * 2 // p['idx'].numel()) if False else None
            # derive from shapes: idx = out*in/2; scale = out*in/16
            # in = 5120 for all linears in this model
            inf = H
            out_f = p['idx'].numel() * 2 // inf
        W = deq_udcq(p, out_f, inf)
        cc, ss, aa = [], [], []
        for r0 in range(0, out_f, CHUNK_ROWS):
            r1 = min(out_f, r0 + CHUNK_ROWS)
            c, s, a = pack_chunk(W[r0:r1])
            cc.append(c); ss.append(s); aa.append(a)
        out['layers'][key] = {
            'codes': torch.cat(cc).contiguous(),
            'sign': torch.cat(ss).contiguous(),
            'aux': torch.cat(aa).contiguous(),
            'out_f': out_f, 'in_f': inf}
        del W
        n_done += 1
        if n_done % 20 == 0 or n_done == len(keys):
            print(f'[{n_done}/{len(keys)}] {key} '
                  f'({time.time()-t0:.0f}s)', flush=True)

    torch.save(out, BLOB_OUT)
    print(f'SAVED {BLOB_OUT} in {time.time()-t0:.0f}s', flush=True)


if __name__ == "__main__":
    _main()
