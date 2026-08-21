# -*- coding: utf-8 -*-
"""One-time PEAK-Q packing of SenseNova-U1.5 -> safetensors on disk.

Same structure as pack_to_disk.py (BF16X) but uses ixrun's PEAK-Q
(layout='rows', 10.59 bpw, ~54 dB near-lossless vs bf16; fp32 mot branch
-> bf16 conversion same as BF16X path).

Stores per-tensor: sign_packed, emax, B1 bitmap, per-row B2, T1(uint8),
T2/T3(7-bit), 4 row-offset arrays.
"""
from __future__ import annotations
import sys, os, json, time, gc
sys.setrecursionlimit(20000)
sys.path.insert(0, r'E:\models\SenseNova-U1\src')
sys.path.insert(0, r'E:\IXRUN')
import numpy as np
import pandas
import torch
import torch.nn as nn
from safetensors.torch import save_file

from ixrun.peakq import peakq_quantize, PEAKQ_TIERS, PEAKQ_GROUP
from sensenova_u1.utils import load_model_and_tokenizer

MD = r'E:\models\SenseNova-U1.5-8B-MoT'
OUT_DIR = r'E:\IXRUN\experiments\u15_bf16x'
PACK = os.path.join(OUT_DIR, 'u15_peakq_packed.safetensors')
META = os.path.join(OUT_DIR, 'u15_peakq_meta.json')


def main():
    print('loading model (CPU) ...', flush=True)
    t0 = time.time()
    model, tok = load_model_and_tokenizer(MD, dtype=torch.bfloat16, device='cpu')
    print(f'  loaded {time.time()-t0:.0f}s', flush=True)

    blobs = {}
    meta = {'tensors': {}, 'kind': 'peakq-rows-v2',
            'tiers': str(PEAKQ_TIERS), 'group': PEAKQ_GROUP}
    total_bytes = 0
    n = 0
    t0 = time.time()
    for name, mod in list(model.named_modules()):
        if not isinstance(mod, nn.Linear):
            continue
        if mod.weight.numel() < 4096 or 'embed' in name or 'lm_head' in name:
            continue
        w = mod.weight.data.detach().to(torch.bfloat16).cpu()
        if w.shape[1] % 64 or w.shape[0] % 1:   # rows layout: no divisibility
            pass                                 # constraints beyond defaults
        p = peakq_quantize(w, group=PEAKQ_GROUP, tiers=PEAKQ_TIERS,
                           layout='rows')
        m = {}
        for key, val in (('sign', p['sign_packed']),
                         ('emax', p['emax']),
                         ('b1', p['bitmaps'][0]),
                         ('b2r', p['b2_rows']),
                         ('t1', p['streams'][0]),
                         ('t2', p['streams'][1]),
                         ('t3', p['streams'][2]),
                         ('t1o', p['t1_off']),
                         ('t2o', p['t2_bit_off']),
                         ('t3o', p['t3_bit_off']),
                         ('b2o', p['b2_bit_off'])):
            gk = f'{name}.{key}'
            blobs[gk] = val.contiguous()
            total_bytes += val.numel() * val.element_size()
            m[key] = gk
        m['out_f'], m['in_f'] = int(w.shape[0]), int(w.shape[1])
        m['N'] = int(w.numel())
        meta['tensors'][name] = m
        mod.weight.data = torch.empty(0)
        del w, p
        n += 1
        if n % 50 == 0:
            print(f'  {n} packed, blob {total_bytes/1e9:.1f}GB, {time.time()-t0:.0f}s', flush=True)
        gc.collect()

    print(f'saving {n} packed linears ({total_bytes/1e9:.2f}GB) ...', flush=True)
    t0 = time.time()
    save_file(blobs, PACK, metadata={'format': 'peakq'})
    json.dump(meta, open(META, 'w'))
    print(f'  saved in {time.time()-t0:.0f}s -> {PACK}', flush=True)
    print(f'  disk: {os.path.getsize(PACK)/1e9:.2f}GB '
          f'(bf16x blob was 25.21GB, bf16 equiv 32.4GB)', flush=True)


if __name__ == '__main__':
    main()


