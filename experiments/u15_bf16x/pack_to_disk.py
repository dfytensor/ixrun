# -*- coding: utf-8 -*-
"""One-time BF16X packing of SenseNova-U1.5 -> single safetensors on disk.

Loads the model layer-by-layer on CPU, packs every >=4096-elem Linear,
drops the original immediately (host-RAM peak ~= packed + 1 layer), saves:
  u15_bf16x_packed.safetensors  : flat stream blobs (sign/mant/delta/emax + overflow)
  u15_bf16x_meta.json           : per-tensor offsets/shapes so the loader can
                                  slice the blobs without any repacking.
"""
from __future__ import annotations
import sys, os, json, time, gc
sys.setrecursionlimit(20000)
sys.path.insert(0, r'E:\models\SenseNova-U1\src')
sys.path.insert(0, r'C:\Users\Administrator\AppData\Local\Temp\opencode\bfloat16x_repo')
import numpy as np
import pandas
import torch
import torch.nn as nn
from safetensors.torch import save_file

MD = r'E:\models\SenseNova-U1.5-8B-MoT'
OUT_DIR = r'E:\IXRUN\experiments\u15_bf16x'
PACK = os.path.join(OUT_DIR, 'u15_bf16x_packed.safetensors')
META = os.path.join(OUT_DIR, 'u15_bf16x_meta.json')

from sensenova_u1.utils import load_model_and_tokenizer
from opqk_linear import bf16x_quantize

pad1 = lambda t: torch.cat([t, torch.zeros(1, dtype=t.dtype)])


def main():
    print('loading model (CPU) ...', flush=True)
    t0 = time.time()
    model, tok = load_model_and_tokenizer(MD, dtype=torch.bfloat16, device='cpu')
    print(f'  loaded {time.time()-t0:.0f}s', flush=True)

    blobs = {}          # name -> tensor to save
    meta = {'tensors': {}, 'kind': 'bf16x-v1'}
    total_bytes = 0
    n = 0
    t0 = time.time()
    for name, mod in list(model.named_modules()):
        if not isinstance(mod, nn.Linear):
            continue
        if mod.weight.numel() < 4096 or 'embed' in name or 'lm_head' in name:
            continue
        w = mod.weight.data.detach().to(torch.bfloat16).cpu()
        p = bf16x_quantize(w, sub=16)
        oi = p['delta_ovf_idx'].to(torch.int64)
        ov = p['delta_ovf_val'].to(torch.int64)
        keep = ov > 7
        oi, ov = oi[keep], ov[keep]
        m = {}
        for key, val in (('sign', pad1(p['sign_packed'])),
                         ('mant', pad1(p['mant_packed'])),
                         ('delta', pad1(p['delta_packed'])),
                         ('emax', p['emax']),
                         ('ovf_i', oi), ('ovf_v', ov.to(torch.int32))):
            gk = f'{name}.{key}'
            blobs[gk] = val.contiguous()
            total_bytes += val.numel() * val.element_size()
            m[key] = gk
        m['out_f'], m['in_f'] = mod.out_features, mod.in_features
        m['N'] = int(mod.weight.numel())
        m['n_fix'] = int(oi.numel())
        meta['tensors'][name] = m
        mod.weight.data = torch.empty(0)      # drop bf16 original now
        del w, p
        n += 1
        if n % 50 == 0:
            print(f'  {n} packed, blob {total_bytes/1e9:.1f}GB, {time.time()-t0:.0f}s', flush=True)
        gc.collect()

    print(f'saving {n} packed linears ({total_bytes/1e9:.2f}GB) ...', flush=True)
    t0 = time.time()
    save_file(blobs, PACK, metadata={'format': 'bf16x'})
    json.dump(meta, open(META, 'w'))
    print(f'  saved in {time.time()-t0:.0f}s -> {PACK}', flush=True)
    print(f'  disk: {os.path.getsize(PACK)/1e9:.2f}GB (bf16 was 50.2GB fp32mix)', flush=True)


if __name__ == '__main__':
    main()


