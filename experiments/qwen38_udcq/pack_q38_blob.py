# -*- coding: utf-8 -*-
"""Pack the 27B UDCQ slim deploy to disk ONCE (torch.save), so debug
iterations load in ~2min instead of re-quantizing 55min.

Saves: experiments/qwen38_udcq/q38_blob.pt = {'codebook', 'layers':
 {name: {'idx','scale','sign'}}}. Also embed_tokens bf16 (CPU)."""
import sys, time, gc, os
sys.setrecursionlimit(20000)
sys.path.insert(0, r'E:\IXRUN')
import pandas
import torch
import torch.nn as nn
from ixrun.udcq import udcq_fit_codebook, udcq_quantize, UDCQ_G
from ixrun.config import QWEN38_PATH
from transformers import AutoModelForCausalLM

BLOB = r'E:\IXRUN\experiments\qwen38_udcq\q38_blob.pt'

m = AutoModelForCausalLM.from_pretrained(
    QWEN38_PATH, dtype=torch.bfloat16, low_cpu_mem_usage=True, device_map='cpu')

targets = []
for name, mod in m.named_modules():
    if isinstance(mod, nn.Linear) and mod.weight.numel() >= 1000 \
            and 'embed' not in name and 'mtp' not in name:
        targets.append((name, mod))
print(f'{len(targets)} targets', flush=True)

cb = None
out = {}
t0 = time.time()
for i, (name, mod) in enumerate(targets):
    w = mod.weight.data.cpu()
    if cb is None:
        cb = udcq_fit_codebook(w, g=UDCQ_G)
    p = udcq_quantize(w, cb, g=UDCQ_G)
    out[name] = {'idx': p['idx'], 'scale': p['scale'],
                 'sign': p['sign_packed']}
    mod.weight.data = torch.empty(0)
    del w, p
    if (i + 1) % 100 == 0:
        gc.collect()
        print(f'  [{i+1}] {time.time()-t0:.0f}s', flush=True)

# embed (CPU bf16)
emb = m.get_input_embeddings().weight.data.to(torch.bfloat16).cpu()
torch.save({'codebook': cb.half(), 'layers': out, 'embed': emb}, BLOB)
print(f'saved {os.path.getsize(BLOB)/1e9:.2f}GB in {time.time()-t0:.0f}s',
      flush=True)
