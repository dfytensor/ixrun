# -*- coding: utf-8 -*-
"""BF16X-compressed SenseNova-U1.5 inference: t2i / edit smoke test (24GB GPU).

Memory plan:
  - RAM 68GB: packed streams ~24.9GB in CHUNKED pinned pools (~2GB each,
    allocated incrementally as layers pack; per-tensor pin x588 fragments
    the WDDM host allocator, one 25GB pin exceeds its limit)
  - GPU ~24GB: non-Linear params (~4GB) + shared decode buf + activations
  - host-RAM peak = packed + one layer bf16 (drop original right after pack)
"""
from __future__ import annotations
import sys, os, time, gc, json
sys.setrecursionlimit(20000)
sys.path.insert(0, r'E:\models\SenseNova-U1\src')
sys.path.insert(0, r'C:\Users\Administrator\AppData\Local\Temp\opencode\bfloat16x_repo')
import numpy as np
import pandas
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

MD = r'E:\models\SenseNova-U1.5-8B-MoT'
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'outputs')
os.makedirs(OUT, exist_ok=True)

import sensenova_u1  # noqa
from sensenova_u1.utils import load_model_and_tokenizer
from opqk_linear import bf16x_quantize
import triton
from bf16x_triton_test import _bf16x_decode_kernel, _bf16x_fix_ovf_kernel

_DEC_BUF = None
_CHUNK_MB = 256          # small chunks: WDDM pinned pool fragments after
                         # repeated large allocations (driver-level, not RAM)


def _dec_buf(N, device):
    global _DEC_BUF
    if _DEC_BUF is None or _DEC_BUF.numel() < N:
        _DEC_BUF = torch.empty(N, dtype=torch.bfloat16, device=device)
    return _DEC_BUF[:N]


class _Pinner:
    """Incremental pinned storage: buffers of <=2GB, grows as layers pack."""

    def __init__(self):
        self.buf = None      # current uint8 pinned chunk
        self.used = 0
        self.total = 0

    def add(self, t: torch.Tensor) -> torch.Tensor:
        es = t.element_size()
        n = t.numel() * es
        # align offset to the element size (view dtype requires it)
        if self.buf is not None:
            self.used = (self.used + es - 1) // es * es
        if self.buf is None or self.used + n > self.buf.numel():
            sz = max(n + es, _CHUNK_MB << 20)
            self.buf = torch.empty(sz, dtype=torch.uint8).pin_memory()
            self.used = 0
            self.total += sz
        v = (self.buf[self.used:self.used + n]
             .view(t.dtype).view(t.shape))
        v.copy_(t)
        self.used += n
        return v


_PIN = _Pinner()


class Bf16xStreamLinear(nn.Module):
    """Weight BF16X-packed in pinned pools; forward: DMA -> decode -> GEMM.

    Two construction paths:
      from_linear : quantize an nn.Linear in memory (first run, slow)
      from_packed : slice the pre-packed disk blob (u15_bf16x_packed.safetensors,
                    see pack_to_disk.py) ?no quantization, fast startup
    """

    @classmethod
    def from_linear(cls, lin: nn.Linear):
        obj = cls.__new__(cls)
        nn.Module.__init__(obj)
        obj.out_features = lin.out_features
        obj.in_features = lin.in_features
        w = lin.weight.data.detach().to(torch.bfloat16).cpu()
        p = bf16x_quantize(w, sub=16)
        pad1 = lambda t: torch.cat([t, torch.zeros(1, dtype=t.dtype)])
        oi = p['delta_ovf_idx'].to(torch.int64)
        ov = p['delta_ovf_val'].to(torch.int64)
        keep = ov > 7
        oi, ov = oi[keep], ov[keep]
        obj.pin = dict(
            sign=_PIN.add(pad1(p['sign_packed'])),
            mant=_PIN.add(pad1(p['mant_packed'])),
            delta=_PIN.add(pad1(p['delta_packed'])),
            emax=_PIN.add(p['emax']),
            pos=_PIN.add(oi) if oi.numel() else oi,
            val=_PIN.add(ov.to(torch.int32)) if oi.numel() else ov.to(torch.int32),
        )
        obj.n_fix = int(oi.numel())
        obj.N = int(w.numel())
        obj.bias = lin.bias.detach().clone().cpu() if lin.bias is not None else None
        return obj

    @classmethod
    def from_packed(cls, entry: dict, blob: dict, resident: bool = False):
        """entry: meta['tensors'][name]; blob: loaded safetensors dict.

        resident=True (bf16 backbone, fits in VRAM): streams staged to GPU
        once -> ZERO per-forward DMA; decode still uses the fast elementwise
        kernel + cublas F.linear (measured faster than a fused GEMM until a
        Marlin-class kernel exists). resident=False (fp32 mot branch): CPU
        pinned + DMA per forward.
        """
        obj = cls.__new__(cls)
        nn.Module.__init__(obj)
        obj.out_features = entry['out_f']
        obj.in_features = entry['in_f']
        obj.N = entry['N']
        obj.n_fix = entry['n_fix']
        obj.resident = resident
        obj.pin = {}
        if resident:
            obj.gpu = {k: blob[k].cuda() for k in ('sign', 'mant', 'delta', 'emax')}
            if obj.n_fix:
                obj.gpu['pos'] = blob['ovf_i'].cuda()
                obj.gpu['val'] = blob['ovf_v'].cuda()
        else:
            for key in ('sign', 'mant', 'delta', 'emax'):
                obj.pin[key] = _PIN.add(blob[key])
            if obj.n_fix:
                obj.pin['pos'] = _PIN.add(blob['ovf_i'])
                obj.pin['val'] = _PIN.add(blob['ovf_v'])
            else:
                obj.pin['pos'] = torch.zeros(0, dtype=torch.int64)
                obj.pin['val'] = torch.zeros(0, dtype=torch.int32)
        obj.bias = None
        return obj

    def _decode(self, device):
        if self.resident:
            g = self.gpu
        else:
            g = {k: self.pin[k].to(device, non_blocking=True) for k in self.pin}
        buf = _dec_buf(self.N, device)
        _bf16x_decode_kernel[(triton.cdiv(self.N, 1024),)](
            g['sign'], g['mant'], g['delta'], g['emax'], buf,
            self.N, BLOCK=1024)
        if self.n_fix:
            _bf16x_fix_ovf_kernel[(triton.cdiv(self.n_fix, 512),)](
                buf, g['emax'], g['pos'], g['val'], self.n_fix, SUB=16, BLOCK=512)
        return buf.view(self.out_features, self.in_features)

    def forward(self, x):
        if not x.is_cuda:
            x = x.cuda()
        w = self._decode(x.device)
        b = self.bias.to(x.device, x.dtype) if self.bias is not None else None
        return F.linear(x, w, b)


PACK = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                    'u15_bf16x_packed.safetensors')
META = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                    'u15_bf16x_meta.json')


def replace_linears(model, verbose=True):
    """Swap in Bf16xStreamLinear. Uses the on-disk packed blob when present
    (fast path: mmap-slice per layer, no quantization); falls back to
    in-memory quantization otherwise."""
    n = 0
    if os.path.exists(PACK) and os.path.exists(META):
        from safetensors import safe_open
        meta = json.load(open(META))['tensors']
        t0 = time.time()
        n_res = [0]
        n_dma = [0]
        with safe_open(PACK, framework='pt', device='cpu') as sf:
            for name, mod in list(model.named_modules()):
                if not isinstance(mod, nn.Linear):
                    continue
                if name in meta:
                    entry = meta[name]
                    blob = {k: sf.get_tensor(v) for k, v in
                            (('sign', entry['sign']), ('mant', entry['mant']),
                             ('delta', entry['delta']), ('emax', entry['emax']),
                             ('ovf_i', entry['ovf_i']), ('ovf_v', entry['ovf_v']))
                            if entry.get(k)}
                    # bf16 backbone -> GPU-resident packed (zero DMA);
                    # fp32 mot branch -> CPU pinned + DMA
                    resident = 'mot_gen' not in name
                    new = Bf16xStreamLinear.from_packed(entry, blob,
                                                        resident=resident)
                    if resident:
                        n_res[0] += 1
                    else:
                        n_dma[0] += 1
                    parent = model
                    parts = name.split('.')
                    for q in parts[:-1]:
                        parent = getattr(parent, q)
                    setattr(parent, parts[-1], new)
                    mod.weight.data = torch.empty(0)
                    n += 1
                else:
                    mod.weight.data = mod.weight.data.to(torch.bfloat16)
                if n % 100 == 0 and n:
                    gc.collect()
        if verbose:
            print(f'[bf16x] {n} Linears in {time.time()-t0:.0f}s | '
                  f'{n_res[0]} backbone GPU-resident (zero-DMA) + '
                  f'{n_dma[0]} mot DMA | pinned {_PIN.total/1e9:.1f}GB', flush=True)
    else:
        for name, mod in list(model.named_modules()):
            if isinstance(mod, nn.Linear):
                if mod.weight.numel() >= 4096 and 'embed' not in name and 'lm_head' not in name:
                    parent = model
                    parts = name.split('.')
                    for q in parts[:-1]:
                        parent = getattr(parent, q)
                    new = Bf16xStreamLinear.from_linear(mod)
                    setattr(parent, parts[-1], new)
                    mod.weight.data = torch.empty(0)      # drop bf16 original
                    n += 1
                else:
                    mod.weight.data = mod.weight.data.to(torch.bfloat16)
                gc.collect()
        if verbose:
            print(f'[bf16x] {n} Linears quantized+pinned '
                  f'({_PIN.total/1e9:.1f}GB pools)', flush=True)
    model.cuda()                                       # non-Linear params -> GPU
    # lm_head is only needed for text sampling; t2i path never samples from
    # the full vocab -> keep it on CPU, save 1.2GB GPU for KV cache
    if hasattr(model, 'language_model') and hasattr(model.language_model, 'lm_head'):
        model.language_model.lm_head.cpu()
    torch.cuda.empty_cache()
    return n


def save_img(im, path):
    if isinstance(im, Image.Image):
        im.save(path)
        return
    t = im.detach().float().cpu()
    while t.dim() > 3:
        t = t[0]
    if t.shape[0] == 3:                      # CHW
        arr = t.numpy().transpose(1, 2, 0)
    else:                                    # HWC
        arr = t.numpy()
    arr = arr * 0.5 + 0.5 if arr.min() < -0.01 or arr.max() > 1.01 else arr
    Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8)).save(path)


def main():
    task = sys.argv[1] if len(sys.argv) > 1 else 't2i'
    # official trained resolution buckets (examples/t2i/inference.py)
    SUPPORTED = {
        '1:1': (2048, 2048), '16:9': (2720, 1536), '9:16': (1536, 2720),
        '3:2': (2496, 1664), '2:3': (1664, 2496), '4:3': (2368, 1760),
        '3:4': (1760, 2368), '1:2': (1440, 2880), '2:1': (2880, 1440),
        '1:3': (1152, 3456), '3:1': (3456, 1152),
    }
    ar = sys.argv[2] if len(sys.argv) > 2 and sys.argv[2] in SUPPORTED else '1:1'
    W, H = SUPPORTED.get(ar, SUPPORTED['1:1'])
    print('loading model (CPU) ...', flush=True)
    t0 = time.time()
    model, tok = load_model_and_tokenizer(MD, dtype=torch.bfloat16, device='cpu')
    print(f'  loaded {time.time()-t0:.0f}s; packing linears ...', flush=True)
    t0 = time.time()
    replace_linears(model)
    print(f'  packed {time.time()-t0:.0f}s', flush=True)

    steps = 10
    print(f'[cfg] {ar} -> {W}x{H}, {steps} steps', flush=True)

    if task == 't2i':
        prompt = '一只戴宇航员头盔的柯基犬，漂浮在太空中，背景是蓝色地球，电影级光效'
        t0 = time.time()
        out = model.t2i_generate(
            tok, prompt, cfg_scale=4.0, timestep_shift=3.0,
            image_size=(W, H), num_steps=steps, seed=0)
        imgs = out if isinstance(out, list) else [out]
        for i, im in enumerate(imgs):
            save_img(im, os.path.join(OUT, f't2i_{W}x{H}_{i}.png'))
        print(f'[t2i] {len(imgs)} image(s) in {time.time()-t0:.0f}s -> {OUT}', flush=True)

    elif task == 'edit':
        # usage: edit <src.png> [prompt] [aspect]
        # argv: [0]=script [1]=edit [2]=src [3]=prompt [4]=aspect
        src = sys.argv[2] if len(sys.argv) > 2 and os.path.exists(sys.argv[2]) \
            else os.path.join(OUT, 't2i_0.png')
        prompt = sys.argv[3] if len(sys.argv) > 3 \
            else '\u4e00\u53ea\u6234\u5b87\u822a\u5458\u5934\u76d4\u7684\u67ef\u57fa\u72ac\uff0c\u6f02\u6d6e\u5728\u592a\u7a7a\u4e2d\uff0c\u80cc\u666f\u662f\u84dd\u8272\u5730\u7403\uff0c\u7535\u5f71\u7ea7\u5149\u6548'
        if len(sys.argv) > 4 and sys.argv[4] in SUPPORTED:
            W, H = SUPPORTED[sys.argv[4]]
        # 官方 it2i 推荐参数（examples/editing/inference.py + README）:
        #   cfg_scale=4.0, img_cfg_scale=1.0(=图像CFG关闭), cfg_norm=none, steps=50
        # 之前用双 cfg 3.0 会把图像条件也推开 -> 只信 prompt -> 雪画成死白过曝。
        print(f'[edit] src={src} prompt={prompt!r} size={W}x{H}', flush=True)
        t0 = time.time()
        imgs = model.it2i_generate(
            tok, prompt, images=[src], cfg_scale=4.0, img_cfg_scale=1.0,
            cfg_norm='none', timestep_shift=3.0, image_size=(W, H),
            num_steps=50, seed=0)
        for i, im in enumerate(imgs):
            save_img(im, os.path.join(OUT, f'edit_{W}x{H}_{i}.png'))
        print(f'[edit] {len(imgs)} image(s) in {time.time()-t0:.0f}s -> {OUT}', flush=True)


if __name__ == '__main__':
    main()


