# -*- coding: utf-8 -*-
"""PEAK-Q streaming inference for SenseNova-U1.5 (t2i/edit) ?24GB GPU.

Mirror of u15_bf16x_infer.py but decodes from the PEAK-Q rows-layout blob
(u15_peakq_packed.safetensors, 21.34GB) via ixrun's `_peakq_decode_v2_kernel`.
Pinned-pool DMA + single-kernel decode (no fixups) + F.linear.

Usage:
  python u15_peakq_infer.py t2i [16:9]
  python u15_peakq_infer.py edit <src.png> "prompt" [16:9]
"""
from __future__ import annotations
import sys, os, time, gc, json
sys.setrecursionlimit(20000)
sys.path.insert(0, r'E:\models\SenseNova-U1\src')
sys.path.insert(0, r'E:\IXRUN')
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
from ixrun.peakq import _peakq_decode_v2_kernel, _pick_v2_blk, PEAKQ_GROUP

MD = r'E:\models\SenseNova-U1.5-8B-MoT'
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'outputs_peakq')
os.makedirs(OUT, exist_ok=True)
PACK = os.path.join(HERE, 'u15_peakq_packed.safetensors')
META = os.path.join(HERE, 'u15_peakq_meta.json')

_DEC_BUF = None
_CHUNK_MB = 256          # small chunks: WDDM pinned pool fragments over sessions


def _dec_buf(N, device):
    global _DEC_BUF
    if _DEC_BUF is None or _DEC_BUF.numel() < N:
        _DEC_BUF = torch.empty(N, dtype=torch.bfloat16, device=device)
    return _DEC_BUF[:N]


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


class PeakQStreamLinear(nn.Module):
    """PEAK-Q rows-layout packed. resident=True (bf16 backbone): streams
    GPU-resident, zero per-forward DMA. resident=False (fp32 mot branch):
    CPU pinned + DMA per forward."""

    def __init__(self, entry: dict, blob: dict, resident: bool = False):
        super().__init__()
        self.out_features = entry['out_f']
        self.in_features = entry['in_f']
        self.N = entry['N']
        self.resident = resident
        KEYS = ('sign', 'emax', 'b1', 'b2r', 't1', 't2', 't3',
                't1o', 't2o', 't3o', 'b2o')
        if resident:
            self.gpu = {k: blob[k].cuda() for k in KEYS}
            self.pin = None
        else:
            self.pin = {k: _PIN.add(blob[k]) for k in KEYS}
            self.gpu = None
        self.bias = None

    def _decode(self, device):
        if self.resident:
            g = self.gpu
        else:
            g = {k: self.pin[k].to(device, non_blocking=True) for k in self.pin}
        buf = _dec_buf(self.N, device)
        bk = _pick_v2_blk(self.in_features)
        _peakq_decode_v2_kernel[(self.out_features,)](
            buf, g['sign'], g['emax'], g['b1'], g['b2r'],
            g['t1'], g['t2'], g['t3'],
            g['t1o'], g['t2o'], g['t3o'], g['b2o'],
            IN_F=self.in_features, GROUP=PEAKQ_GROUP, BK=bk,
        )
        return buf.view(self.out_features, self.in_features)

    def forward(self, x):
        if not x.is_cuda:
            x = x.cuda()
        w = self._decode(x.device)
        return F.linear(x, w, self.bias)


def replace_linears(model, verbose=True):
    meta = json.load(open(META))['tensors']
    n = 0
    n_res = n_dma = 0
    t0 = time.time()
    with safe_open(PACK, framework='pt', device='cpu') as sf:
        for name, mod in list(model.named_modules()):
            if not isinstance(mod, nn.Linear):
                continue
            if name in meta:
                entry = meta[name]
                blob = {k: sf.get_tensor(entry[k]) for k in
                        ('sign', 'emax', 'b1', 'b2r', 't1', 't2', 't3',
                         't1o', 't2o', 't3o', 'b2o')}
                resident = 'mot_gen' not in name
                new = PeakQStreamLinear(entry, blob, resident=resident)
                if resident:
                    n_res += 1
                else:
                    n_dma += 1
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
    model.cuda()
    if hasattr(model, 'language_model') and hasattr(model.language_model, 'lm_head'):
        model.language_model.lm_head.cpu()
    torch.cuda.empty_cache()
    if verbose:
        print(f'[peakq] {n} Linears from disk in {time.time()-t0:.0f}s, '
              f'pinned {_PIN.total/1e9:.1f}GB', flush=True)
    return n


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
    print(f'  loaded {time.time()-t0:.0f}s; loading packed linears ...', flush=True)
    replace_linears(model)
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
        src = sys.argv[2] if len(sys.argv) > 2 and os.path.exists(sys.argv[2]) \
            else os.path.join(HERE, 'outputs', 't2i_0.png')
        prompt = sys.argv[3] if len(sys.argv) > 3 else '\u628a\u80cc\u666f\u6362\u6210\u96ea\u5c71\u65e5\u51fa'
        if len(sys.argv) > 4 and sys.argv[4] in SUPPORTED:
            W, H = SUPPORTED[sys.argv[4]]
        print(f'[edit] src={src} prompt={prompt!r} size={W}x{H}', flush=True)
        t0 = time.time()
        imgs = model.it2i_generate(
            tok, prompt, images=[src], cfg_scale=3.0, img_cfg_scale=3.0,
            timestep_shift=3.0, image_size=(W, H), num_steps=steps, seed=0)
        for i, im in enumerate(imgs):
            save_img(im, os.path.join(OUT, f'edit_{W}x{H}_{i}.png'))
        print(f'[edit] {len(imgs)} image(s) in {time.time()-t0:.0f}s -> {OUT}', flush=True)


if __name__ == '__main__':
    main()


