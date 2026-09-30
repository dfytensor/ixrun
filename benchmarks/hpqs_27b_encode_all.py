# -*- coding: utf-8 -*-
"""Encode ALL down_proj+o_proj layers of Qwen3.8-27B to HPQ-x-scale
packs (one blob mmap pass, one layer resident at a time)."""
import sys
import time

sys.path.insert(0, r'E:\IXRUN')
import pandas  # noqa: F401
import torch
from ixrun.udcq import _decode_udcq_ref, UDCQ_G
from benchmarks.hpqs_runtime import hpqs_pack

BLOB = r'E:\IXRUN\experiments\qwen38_udcq\q38_blob.pt'
OUT = r'E:\IXRUN\experiments\hpqs_minicpm5\q38_down_o_packs.pt'
HID, INTER = 5120, 17408

blob = torch.load(BLOB, map_location='cpu', mmap=True,
                  weights_only=False)
names = sorted(n for n in blob['layers']
               if n.endswith('down_proj') or n.endswith('o_proj'))
packs = {}
if __import__('os').path.exists(OUT):
    packs = torch.load(OUT, map_location='cpu', weights_only=False)
    names = [n for n in names if n not in packs]
    print(f'resume: {len(packs)} done, {len(names)} to go', flush=True)
t00 = time.time()
for i, name in enumerate(names):
    t0 = time.time()
    e = blob['layers'][name]
    packed = {'g': UDCQ_G, 'N': e.get('N', 0),
              'out_f': e.get('out_f', HID), 'in_f': e.get('in_f', INTER),
              'idx': e['idx'], 'sign_packed': e['sign'],
              'scale': e['scale'], 'codebook': blob['codebook']}
    if name.endswith('o_proj'):
        packed['out_f'] = packed['in_f'] = HID
    W = _decode_udcq_ref(packed, device='cuda')
    pk = hpqs_pack(W.float())
    del W, packed, e
    packs[name] = {'codes6': pk['codes6'].cpu(), 'cb': pk['cb'].cpu(),
                   'scale': pk['scale'].cpu(),
                   'out_f': pk['out_f'], 'in_f': pk['in_f']}
    del pk
    gc = __import__('gc')
    gc.collect()
    torch.cuda.empty_cache()
    if i % 8 == 0:
        print(f'[{i+1}/{len(names)}] {name} ({time.time()-t0:.0f}s, '
              f'total {time.time()-t00:.0f}s)', flush=True)
torch.save(packs, OUT)
print(f'saved {len(packs)} packs -> {OUT} '
      f'({time.time()-t00:.0f}s total)', flush=True)
