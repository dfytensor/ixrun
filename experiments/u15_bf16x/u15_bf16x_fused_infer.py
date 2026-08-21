# -*- coding: utf-8 -*-
"""SenseNova-U1.5 + BF16X FUSED decode+GEMM inference (fast path).

vs u15_bf16x_infer.py (two-segment: DMA -> decode-to-bf16-buf -> F.linear):
  * backbone packed streams (bf16 origin, ~12.6GB) RESIDENT ON GPU ->
    fused kernel decodes W tiles in registers and tl.dot's them: zero DMA on
    the hot path, no bf16 W ever materialized (reads ~12.3bpw packed instead
    of decode-write + GEMM-read = 32bpw round trip)
  * mot_gen packed streams (fp32 origin, ~12.6GB) stay CPU (don't fit):
    DMA per forward, then the SAME fused kernel consumes them directly
  * M==1 single-token forwards use the exact fused GEMV with CSR overflow
    overlay (ported from bf16x_fused.py)
  * multi-token fused mode decodes delta with saturation at 7 (near-lossless;
    delta>7 affects ~1% of elements). For bit-exact multi-token use the
    two-segment script (u15_bf16x_infer.py).

Usage:
  python u15_bf16x_fused_infer.py t2i [16:9]
"""
from __future__ import annotations
import sys, os, time, gc, json
sys.setrecursionlimit(20000)
sys.path.insert(0, r'E:\models\SenseNova-U1\src')
import numpy as np
import pandas
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from safetensors import safe_open

import sensenova_u1  # noqa
from sensenova_u1.utils import load_model_and_tokenizer
import triton
import triton.language as tl

MD = r'E:\models\SenseNova-U1.5-8B-MoT'
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'outputs_fused')
os.makedirs(OUT, exist_ok=True)
PACK = os.path.join(HERE, 'u15_bf16x_packed.safetensors')
META = os.path.join(HERE, 'u15_bf16x_meta.json')


# ------------------------------------------------------------------ #
#  fused multi-token GEMM: y[M,OUT] = x[M,IN] @ W.T, W decoded on the fly
# ------------------------------------------------------------------ #
@triton.jit
def _bf16x_gemm_fused_kernel(
    x_ptr, y_ptr,
    sign_ptr, mant_ptr, delta_ptr, emax_ptr,
    M, OUT_F: tl.constexpr, IN_F: tl.constexpr,
    GROUP: tl.constexpr,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_m = tl.program_id(1)
    offs_n = pid_n * BN + tl.arange(0, BN)
    offs_m = pid_m * BM + tl.arange(0, BM)
    mask_m = offs_m < M

    acc = tl.zeros((BM, BN), tl.float32)

    for k0 in tl.range(0, IN_F, BK):
        offs_k = k0 + tl.arange(0, BK)
        pos = offs_n[:, None] * IN_F + offs_k[None, :]        # [BN, BK] int32

        sw = tl.load(sign_ptr + pos // 32).to(tl.uint32)
        sign = ((sw >> (pos % 32)) & 1).to(tl.int32)

        mb = pos * 7
        mw = mb // 32
        ms = mb % 32
        w1 = tl.load(mant_ptr + mw).to(tl.uint32)
        cross = (ms + 7) > 32
        w2 = tl.where(cross, tl.load(mant_ptr + mw + 1).to(tl.uint32),
                      tl.zeros((BN, BK), tl.uint32))
        mant = tl.where(cross, ((w1 >> ms) | (w2 << (32 - ms))) & 0x7F,
                        (w1 >> ms) & 0x7F).to(tl.int32)

        db = pos * 3
        dw = db // 32
        ds = db % 32
        d1 = tl.load(delta_ptr + dw).to(tl.uint32)
        cd = (ds + 3) > 32
        d2 = tl.where(cd, tl.load(delta_ptr + dw + 1).to(tl.uint32),
                      tl.zeros((BN, BK), tl.uint32))
        delta = tl.where(cd, ((d1 >> ds) | (d2 << (32 - ds))) & 0x7,
                         (d1 >> ds) & 0x7).to(tl.int32)

        e8 = tl.load(emax_ptr + pos // GROUP).to(tl.int32)
        expo = tl.minimum(tl.maximum(e8 - delta, 0), 255)
        bits = ((sign << 15) | (expo << 7) | mant).to(tl.uint16)
        wt = bits.to(tl.bfloat16, bitcast=True)               # [BN, BK]

        xt = tl.load(x_ptr + offs_m[:, None] * IN_F + offs_k[None, :],
                     mask=mask_m[:, None], other=0.0)         # [BM, BK]
        acc += tl.dot(xt, tl.trans(wt))

    tl.store(y_ptr + offs_m[:, None] * OUT_F + offs_n[None, :],
             acc.to(tl.bfloat16), mask=mask_m[:, None])


def _fused_gemm(x, g, out_f, in_f):
    M = x.shape[0]
    y = torch.empty(M, out_f, dtype=torch.bfloat16, device=x.device)
    BM = 64 if M >= 64 else 16
    BN, BK = 128, 32          # decode tile [BN,BK] lives in smem: keep <= 100KB
    grid = (triton.cdiv(out_f, BN), triton.cdiv(M, BM))
    _bf16x_gemm_fused_kernel[grid](
        x, y, g['sign'], g['mant'], g['delta'], g['emax'],
        M, OUT_F=out_f, IN_F=in_f, GROUP=16,
        BM=BM, BN=BN, BK=BK, num_warps=8, num_stages=1,
    )
    return y


# ------------------------------------------------------------------ #
#  exact fused GEMV (M == 1) with CSR overflow overlay — from bf16x_fused.py
# ------------------------------------------------------------------ #
@triton.jit
def _bf16x_gemv_fused_kernel(
    x_ptr, y_ptr,
    sign_ptr, mant_ptr, delta_ptr, emax_ptr,
    olk_ptr, old_ptr, oloff_ptr,
    IN_F: tl.constexpr, GROUP: tl.constexpr, BK: tl.constexpr, R: tl.constexpr,
):
    pid = tl.program_id(0)
    rows = pid * R + tl.arange(0, R)
    base = rows.to(tl.int64) * IN_F
    acc = tl.zeros((R,), tl.float32)

    for k0 in tl.range(0, IN_F, BK):
        offs = base[:, None] + (k0 + tl.arange(0, BK))[None, :]
        sw = tl.load(sign_ptr + offs // 32).to(tl.uint32)
        sign = ((sw >> (offs % 32)) & 1).to(tl.int32)
        mb = offs * 7
        ms = mb % 32
        mw = mb // 32
        w1 = tl.load(mant_ptr + mw).to(tl.uint32)
        cross = (ms + 7) > 32
        w2 = tl.where(cross, tl.load(mant_ptr + mw + 1).to(tl.uint32),
                      tl.zeros((R, BK), tl.uint32))
        mant = tl.where(cross, ((w1 >> ms) | (w2 << (32 - ms))) & 0x7F,
                        (w1 >> ms) & 0x7F).to(tl.int32)
        db = offs * 3
        ds = db % 32
        dwd = db // 32
        d1 = tl.load(delta_ptr + dwd).to(tl.uint32)
        cd = (ds + 3) > 32
        d2 = tl.where(cd, tl.load(delta_ptr + dwd + 1).to(tl.uint32),
                      tl.zeros((R, BK), tl.uint32))
        delta = tl.where(cd, ((d1 >> ds) | (d2 << (32 - ds))) & 0x7,
                         (d1 >> ds) & 0x7).to(tl.int32)
        e8 = tl.load(emax_ptr + offs // GROUP).to(tl.int32)
        expo = tl.minimum(tl.maximum(e8 - delta, 0), 255)
        bits = ((sign << 15) | (expo << 7) | mant).to(tl.uint16)
        w = bits.to(tl.bfloat16, bitcast=True).to(tl.float32)
        xv = tl.load(x_ptr + k0 + tl.arange(0, BK)).to(tl.float32)
        acc += tl.sum(w * xv[None, :], axis=1)

    for r in tl.static_range(R):
        n = pid * R + r
        lo = tl.load(oloff_ptr + n)
        hi = tl.load(oloff_ptr + n + 1)
        part = 0.0
        for j in tl.range(lo, hi):
            k = tl.load(olk_ptr + j)
            td = tl.load(old_ptr + j).to(tl.int32)
            p = n.to(tl.int64) * IN_F + k
            sw = tl.load(sign_ptr + p // 32).to(tl.uint32)
            sgn = ((sw >> (p % 32)) & 1).to(tl.int32)
            ms = (p * 7) % 32
            mw = (p * 7) // 32
            m1 = tl.load(mant_ptr + mw).to(tl.uint32)
            m2 = tl.load(mant_ptr + mw + 1).to(tl.uint32)
            mant = ((m1 >> ms) | (m2 << (32 - ms))) & 0x7F
            e = tl.load(emax_ptr + p // GROUP).to(tl.int32)
            e7 = tl.minimum(tl.maximum(e - 7, 0), 255)
            ec = tl.minimum(tl.maximum(e - td, 0), 255)
            bw = ((sgn << 15) | (e7 << 7) | mant).to(tl.uint16)
            bc = ((sgn << 15) | (ec << 7) | mant).to(tl.uint16)
            part += (bc.to(tl.bfloat16, bitcast=True).to(tl.float32)
                     - bw.to(tl.bfloat16, bitcast=True).to(tl.float32)) \
                * tl.load(x_ptr + k).to(tl.float32)
        acc = tl.where(tl.arange(0, R) == r, acc + part, acc)

    tl.store(y_ptr + rows, acc.to(tl.bfloat16))


# ------------------------------------------------------------------ #
#  layer: resident (backbone) or DMA (mot) — same fused kernels
# ------------------------------------------------------------------ #
_CHUNK_MB = 1024


class _Pinner:
    def __init__(self):
        self.buf = None
        self.used = 0
        self.total = 0

    def add(self, t):
        es = t.element_size()
        n = t.numel() * es
        if self.buf is not None:
            self.used = (self.used + es - 1) // es * es
        if self.buf is None or self.used + n > self.buf.numel():
            sz = max(n + es, _CHUNK_MB << 20)
            self.buf = torch.empty(sz, dtype=torch.uint8).pin_memory()
            self.used = 0
            self.total += sz
        v = self.buf[self.used:self.used + n].view(t.dtype).view(t.shape)
        v.copy_(t)
        self.used += n
        return v


_PIN = _Pinner()
DMA_BYTES = [0]


class FusedBf16xLinear(nn.Module):
    """resident=True (backbone): streams stay on GPU, zero per-forward DMA.
    resident=False (mot fp32 branch): pinned CPU streams, DMA per forward."""

    def __init__(self, entry, blob, resident: bool):
        super().__init__()
        self.out_features = entry['out_f']
        self.in_features = entry['in_f']
        self.N = entry['N']
        self.n_fix = entry.get('n_fix', 0)
        self.resident = resident
        if resident:
            self.g = {k: blob[k].cuda() for k in
                      ('sign', 'mant', 'delta', 'emax')}
        else:
            self.cpu = {k: _PIN.add(blob[k]) for k in
                        ('sign', 'mant', 'delta', 'emax')}
            self.g = None
        # CSR overflow (for the exact M==1 GEMV path)
        if self.n_fix:
            oi, ov = blob['ovf_i'], blob['ovf_v']
            rows = oi // self.in_features
            ks = oi % self.in_features
            order = torch.argsort(rows, stable=True)
            rows, ks, ov = rows[order], ks[order], ov[order]
            counts = torch.bincount(rows, minlength=self.out_features)
            offs = torch.zeros(self.out_features + 1, dtype=torch.int32)
            offs[1:] = counts.cumsum(0).to(torch.int32)
            self._olk = ks.to(torch.int32).cuda()
            self._old = ov.to(torch.int32).cuda()
            self._oloff = offs.cuda()
        else:
            self._olk = torch.zeros(1, dtype=torch.int32, device='cuda')
            self._old = torch.zeros(1, dtype=torch.int32, device='cuda')
            self._oloff = torch.zeros(self.out_features + 1,
                                      dtype=torch.int32, device='cuda')
        self.bias = None

    def _streams(self, device):
        if self.resident:
            return self.g
        g = {k: self.cpu[k].to(device, non_blocking=True) for k in self.cpu}
        DMA_BYTES[0] += sum(t.numel() * t.element_size() for t in g.values())
        return g

    def forward(self, x):
        if not x.is_cuda:
            x = x.cuda()
        g = self._streams(x.device)
        M = x.numel() // self.in_features
        if M == 1:
            # exact single-token path with CSR overlay
            x1 = x.reshape(-1)
            y = torch.empty(self.out_features, dtype=torch.bfloat16,
                            device=x.device)
            r = 4
            while r > 1 and self.out_features % r:
                r //= 2
            _bf16x_gemv_fused_kernel[(self.out_features // r,)](
                x1, y, g['sign'], g['mant'], g['delta'], g['emax'],
                self._olk, self._old, self._oloff,
                IN_F=self.in_features, GROUP=16, BK=256, R=r, num_warps=2)
            return y.view(*x.shape[:-1], self.out_features)
        x2 = x.reshape(M, self.in_features)
        y = _fused_gemm(x2, g, self.out_features, self.in_features)
        return y.view(*x.shape[:-1], self.out_features)


def replace_linears(model, verbose=True):
    meta = json.load(open(META))['tensors']
    n_res = n_dma = 0
    t0 = time.time()
    with safe_open(PACK, framework='pt', device='cpu') as sf:
        for name, mod in list(model.named_modules()):
            if not isinstance(mod, nn.Linear):
                continue
            if name in meta:
                entry = meta[name]
                blob = {k: sf.get_tensor(entry[k]) for k in
                        ('sign', 'mant', 'delta', 'emax', 'ovf_i', 'ovf_v')}
                resident = 'mot_gen' not in name     # backbone fits on GPU
                new = FusedBf16xLinear(entry, blob, resident)
                parent = model
                parts = name.split('.')
                for q in parts[:-1]:
                    parent = getattr(parent, q)
                setattr(parent, parts[-1], new)
                mod.weight.data = torch.empty(0)
                if resident:
                    n_res += 1
                else:
                    n_dma += 1
                del blob
                gc.collect()
            else:
                mod.weight.data = mod.weight.data.to(torch.bfloat16)
    model.cuda()
    if hasattr(model, 'language_model') and hasattr(model.language_model, 'lm_head'):
        model.language_model.lm_head.cpu()
    torch.cuda.empty_cache()
    if verbose:
        print(f'[fused] {n_res} resident(GPU, zero-DMA) + {n_dma} mot(DMA) '
              f'linears in {time.time()-t0:.0f}s; pinned {_PIN.total/1e9:.1f}GB',
              flush=True)
    return n_res + n_dma


def save_img(im, path):
    if isinstance(im, Image.Image):
        im.save(path)
        return
    t = im.detach().float().cpu()
    while t.dim() > 3:
        t = t[0]
    if t.shape[0] == 3:
        arr = t.numpy().transpose(1, 2, 0)
    else:
        arr = t.numpy()
    arr = arr * 0.5 + 0.5 if arr.min() < -0.01 or arr.max() > 1.01 else arr
    Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8)).save(path)


SUPPORTED = {
    '1:1': (2048, 2048), '16:9': (2720, 1536), '9:16': (1536, 2720),
    '3:2': (2496, 1664), '2:3': (1664, 2496), '4:3': (2368, 1760),
    '3:4': (1760, 2368), '1:2': (1440, 2880), '2:1': (2880, 1440),
    '1:3': (1152, 3456), '3:1': (3456, 1152),
}


def main():
    task = sys.argv[1] if len(sys.argv) > 1 else 't2i'
    ar = sys.argv[2] if len(sys.argv) > 2 and sys.argv[2] in SUPPORTED else '1:1'
    W, H = SUPPORTED[ar]
    print('loading model (CPU) ...', flush=True)
    t0 = time.time()
    model, tok = load_model_and_tokenizer(MD, dtype=torch.bfloat16, device='cpu')
    print(f'  loaded {time.time()-t0:.0f}s; staging fused linears ...', flush=True)
    replace_linears(model)
    steps = 10
    print(f'[cfg] {ar} -> {W}x{H}, {steps} steps', flush=True)

    DMA_BYTES[0] = 0
    t0 = time.time()
    prompt = ('\u4e00\u53ea\u6234\u5b87\u822a\u5458\u5934\u76d4\u7684\u67ef\u57fa\u72ac'
              '\uff0c\u6f02\u6d6e\u5728\u592a\u7a7a\u4e2d\uff0c\u80cc\u666f\u662f\u84dd'
              '\u8272\u5730\u7403\uff0c\u7535\u5f71\u7ea7\u5149\u6548')
    out = model.t2i_generate(
        tok, prompt, cfg_scale=4.0, timestep_shift=3.0,
        image_size=(W, H), num_steps=steps, seed=0)
    dt = time.time() - t0
    imgs = out if isinstance(out, list) else [out]
    for i, im in enumerate(imgs):
        save_img(im, os.path.join(OUT, f't2i_{W}x{H}_{i}.png'))
    print(f'[t2i-fused] {len(imgs)} image(s) in {dt:.0f}s '
          f'(DMA {DMA_BYTES[0]/1e9:.1f}GB total) -> {OUT}', flush=True)


if __name__ == '__main__':
    main()
