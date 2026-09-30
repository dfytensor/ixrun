# -*- coding: utf-8 -*-
"""HPQ-x-scale runtime: packed storage + Triton decode+GEMV kernel.

Layout (m=8, k=64, levels=2, block 4x4=16 elems):
  codes int8  [nB, 2, 8] flat      (nB = out_f//4 * in_f//4)
  cb    fp16  [2, 8, 64, 2]   (per subspace 2-float centroid pair)
  scale fp16  [nB]            per-block max-abs
Decode: W[r, j*4+c] = (cb[0, r*2+c//2, c0, c%2] +
                       cb[1, r*2+c//2, c1, c%2]) * scale[j]
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _hpqs_gemv_kernel(x_ptr, codes_ptr, cb_ptr, sc_ptr, y_ptr,
                      n_bc, BLOCK_J: tl.constexpr):
    pid = tl.program_id(0)              # 4-output-row group
    js = tl.arange(0, BLOCK_J)
    duo = tl.arange(0, 2)
    acc0 = 0.0
    acc1 = 0.0
    acc2 = 0.0
    acc3 = 0.0
    for j0 in range(0, n_bc, BLOCK_J):
        jb = j0 + js
        mask = jb < n_bc
        bidx = pid * n_bc + jb
        m2 = tl.expand_dims(mask, 1)
        xa = tl.load(x_ptr + tl.expand_dims(jb * 4, 1)
                     + tl.expand_dims(duo, 0),
                     mask=m2, other=0.0).to(tl.float32)     # [BJ,2]
        xb = tl.load(x_ptr + tl.expand_dims(jb * 4 + 2, 1)
                     + tl.expand_dims(duo, 0),
                     mask=m2, other=0.0).to(tl.float32)     # [BJ,2]
        sc = tl.load(sc_ptr + bidx, mask=mask, other=0.0) \
            .to(tl.float32)
        scx = tl.expand_dims(sc, 1)                         # [BJ,1]
        for r in tl.static_range(4):
            cA = tl.load(codes_ptr + tl.expand_dims(
                bidx * 16 + r * 2, 1), mask=m2, other=0).to(tl.int32)
            cB = tl.load(codes_ptr + tl.expand_dims(
                bidx * 16 + 8 + r * 2, 1), mask=m2, other=0).to(tl.int32)
            cC = tl.load(codes_ptr + tl.expand_dims(
                bidx * 16 + r * 2 + 1, 1), mask=m2, other=0).to(tl.int32)
            cD = tl.load(codes_ptr + tl.expand_dims(
                bidx * 16 + 8 + r * 2 + 1, 1), mask=m2,
                other=0).to(tl.int32)
            v_lo = (tl.load(cb_ptr + ((r * 2) * 64 + cA) * 2
                            + tl.expand_dims(duo, 0), mask=m2,
                            other=0.0).to(tl.float32)
                    + tl.load(cb_ptr + 8 * 64 * 2
                              + ((r * 2) * 64 + cB) * 2
                              + tl.expand_dims(duo, 0), mask=m2,
                              other=0.0).to(tl.float32))    # [BJ,2]
            v_hi = (tl.load(cb_ptr + ((r * 2 + 1) * 64 + cC) * 2
                            + tl.expand_dims(duo, 0), mask=m2,
                            other=0.0).to(tl.float32)
                    + tl.load(cb_ptr + 8 * 64 * 2
                              + ((r * 2 + 1) * 64 + cD) * 2
                              + tl.expand_dims(duo, 0), mask=m2,
                              other=0.0).to(tl.float32))
            part = tl.sum(scx * (v_lo * xa + v_hi * xb))
            if r == 0:
                acc0 += part
            elif r == 1:
                acc1 += part
            elif r == 2:
                acc2 += part
            else:
                acc3 += part
    tl.store(y_ptr + pid * 4 + 0, tl.full((), acc0, tl.float32)
             .to(tl.bfloat16))
    tl.store(y_ptr + pid * 4 + 1, tl.full((), acc1, tl.float32)
             .to(tl.bfloat16))
    tl.store(y_ptr + pid * 4 + 2, tl.full((), acc2, tl.float32)
             .to(tl.bfloat16))
    tl.store(y_ptr + pid * 4 + 3, tl.full((), acc3, tl.float32)
             .to(tl.bfloat16))

def pack6(codes):
    """[nB, 16] uint8 codes (<64) -> [nB, 12] packed (4 codes/3 bytes;
    stored as lo48 | hi48 two little-endian uint64 halves overlapping at
    byte 6 for 4-byte-aligned loads)."""
    c = codes.long().cpu()
    ar = torch.arange(8)
    lo = (c[:, :8] << (6 * ar)).sum(1)
    hi = (c[:, 8:] << (6 * ar)).sum(1)
    out = torch.empty(c.shape[0], 12, dtype=torch.uint8)
    lb = lo.view(torch.uint8).reshape(-1, 8)
    hb = hi.view(torch.uint8).reshape(-1, 8)
    out[:, 0:6] = lb[:, 0:6]
    out[:, 6:8] = hb[:, 0:2]
    out[:, 8:12] = hb[:, 2:6]
    return out.contiguous()


def unpack6(codes6):
    """[nB, 12] -> [nB, 16] uint8 codes (inverse of pack6)."""
    b = codes6.long().cpu()
    nB = b.shape[0]
    bits = torch.zeros(nB, 96, dtype=torch.long)
    for k in range(12):
        bits[:, k * 8:(k + 1) * 8] = ((b[:, k:k + 1]
                                       >> torch.arange(8)) & 1)
    out = torch.empty(nB, 16, dtype=torch.uint8)
    w = torch.tensor([1, 2, 4, 8, 16, 32], dtype=torch.long)
    for i in range(16):
        out[:, i] = (bits[:, i * 6:(i + 1) * 6] * w).sum(1) \
            .to(torch.uint8)
    return out


def hpqs_pack(W, levels=2, k=64, m=8):
    """W [out_f, in_f] cuda float -> packed dict + reference recon."""
    import sys
    sys.path.insert(0, r'E:\IXRUN')
    from benchmarks.hpq_minicpm5 import pq_encode_decode
    of, inf = W.shape
    assert of % 4 == 0 and inf % 4 == 0
    blocks = W.reshape(of // 4, 4, inf // 4, 4).permute(0, 2, 1, 3) \
        .reshape(-1, 16)
    sc = blocks.abs().amax(dim=1, keepdim=True).clamp_min(1e-8)
    blocks_n = blocks / sc
    nB = blocks.shape[0]
    sub = 16 // m                                    # 2
    codes = torch.zeros(nB, levels, m, dtype=torch.uint8)
    cbs = torch.zeros(levels, m, k, sub, dtype=torch.float16)
    resid = blocks_n.clone()
    recon = torch.zeros_like(resid)
    for l in range(levels):
        recon_l, codes_l, cbs_l = pq_encode_decode(resid, None, k=k, m=m)
        for s in range(m):
            cbs[l, s] = cbs_l[s].half().cpu()
            codes[:, l, s] = codes_l[s].cpu().to(torch.uint8)
        recon += recon_l
        resid = blocks_n - recon
    rec = (recon * sc).reshape(of // 4, inf // 4, 4, 4) \
        .permute(0, 2, 1, 3).reshape(of, inf)
    return {
        'codes': codes.reshape(nB, levels * m).contiguous(),
        'codes6': pack6(codes.reshape(nB, levels * m)), 'cb': cbs,
        'scale': sc.squeeze(1).half().cpu().contiguous(),
        'out_f': of, 'in_f': inf, 'recon_ref': rec,
    }


def hpqs_gemv(x, packed, BLOCK_J=128):
    """x [in_f] bf16 -> y [out_f] bf16."""
    codes = packed['codes'].cuda()
    cb = packed['cb'].cuda()
    sc = packed['scale'].cuda()
    of, inf = packed['out_f'], packed['in_f']
    n_bc = inf // 4
    nB = codes.shape[0]
    y = torch.empty(of, dtype=torch.bfloat16, device=x.device)
    _hpqs_gemv_kernel[(of // 4,)](
        x.contiguous(), codes, cb, sc, y, n_bc,
        BLOCK_J=BLOCK_J, num_warps=4)
    return y


def _decode_ref(packed):
    """Reference decode straight from packed tensors (no kmeans)."""
    codes = packed['codes'].reshape(-1, 2, 8).float()
    cb = packed['cb'].float()
    sc = packed['scale'].float()
    of, inf = packed['out_f'], packed['in_f']
    nB = codes.shape[0]
    recon = torch.zeros(nB, 16)
    for l in range(2):
        for s in range(8):
            recon[:, s * 2:(s + 1) * 2] += cb[l, s][codes[:, l, s].long()]
    recon *= sc.unsqueeze(1)
    return recon.reshape(of // 4, inf // 4, 4, 4) \
        .permute(0, 2, 1, 3).reshape(of, inf)


if __name__ == '__main__':
    import time
    torch.manual_seed(0)
    for of, inf in [(512, 512), (1024, 512)]:
        W = (torch.randn(of, inf) * 0.02).cuda()
        t0 = time.time()
        pk = hpqs_pack(W)
        enc_s = time.time() - t0
        dref = _decode_ref(pk)                     # fp16-cb ground truth
        dmax = (pk['recon_ref'].cpu() - dref).abs().max().item()
        x = torch.randn(inf, dtype=torch.bfloat16, device='cuda')
        y = hpqs_gemv(x, pk)
        y_ref = (dref.cuda() @ x.float()).to(torch.bfloat16)
        gmax = (y.float() - y_ref.float()).abs().max().item()
        rel = ((y.float() - y_ref.float()).norm()
               / y_ref.float().norm()).item()
        print(f'[{of}x{inf}] enc={enc_s:.1f}s '
              f'decode-vs-fp16cb dmax={dmax:.2e} '
              f'gemv gmax={gmax:.4f} rel={rel:.4f}', flush=True)
    print('OK' if dmax == 0 else 'CHECK', flush=True)


class HpqsLinear(torch.nn.Module):
    """Deploy wrapper: M=1 -> hand-CUDA GEMV; M>1 -> decode+cublas."""

    def __init__(self, packed):
        super().__init__()
        self.packed = packed
        if 'codes6' not in self.packed:
            self.packed['codes6'] = pack6(self.packed['codes'])
        self.packed['codes6'] = self.packed['codes6'].cuda()
        self.packed['cb'] = packed['cb'].cuda()
        self.packed['scale'] = packed['scale'].cuda()
        self.out_features = packed['out_f']
        self.in_features = packed['in_f']

    def _decode(self):
        from experiments.hpqs_gemv_cuda.hpqs_gemv_cuda import _load
        ext = _load()
        return ext.decode(self.packed['codes6'],
                          self.packed['cb'],
                          self.packed['scale'],
                          self.packed['out_f'],
                          self.packed['in_f'])

    def forward(self, x):
        from experiments.hpqs_gemv_cuda.hpqs_gemv_cuda import _load
        ext = _load()
        if x.numel() == self.in_features:
            y = ext.gemv(x.reshape(-1).contiguous(),
                         self.packed['codes'], self.packed['cb'],
                         self.packed['scale'],
                         self.out_features, self.in_features)
            return y
        W = self._decode()
        return torch.nn.functional.linear(
            x.to(W.dtype), W)


def deploy_hpqs_selected(model, pred, verbose=True):
    """Wrap linears whose name matches pred() with HpqsLinear."""
    from ixrun.linear import iter_quantizable_linears, _set_parent_child
    n = 0
    for name, mod in list(iter_quantizable_linears(model)):
        if not pred(name):
            continue
        W = mod.weight.data.float().cuda()
        pk = hpqs_pack(W)
        del W
        torch.cuda.empty_cache()
        _set_parent_child(model, name, HpqsLinear(pk))
        n += 1
        if verbose:
            print(f'[hpqs-deploy] {name}', flush=True)
    return n