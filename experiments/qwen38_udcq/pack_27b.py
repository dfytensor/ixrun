# -*- coding: utf-8 -*-
"""Pack Qwen3.8-27B once to disk (UDCQ blob + meta), separate process so the
CUDA/pinned state fully releases afterwards. The inference process then
mmaps the blob with NO pinning (pageable DMA still overlaps with compute
thanks to per-layer staging copy)."""
import sys, os, time, gc, json
sys.setrecursionlimit(20000)
sys.path.insert(0, r'E:\IXRUN')
import pandas
import torch
import torch.nn as nn
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM

from ixrun.udcq import udcq_fit_codebook, udcq_quantize, UDCQ_G
from ixrun.config import QWEN38_PATH

MD = QWEN38_PATH
HERE = os.path.dirname(os.path.abspath(__file__))
PACK = os.path.join(HERE, 'qwen38_udcq_packed.safetensors')
META = os.path.join(HERE, 'qwen38_udcq_meta.json')


def main():
    print('loading (lazy cpu) ...', flush=True)
    t0 = time.time()
    m = AutoModelForCausalLM.from_pretrained(
        MD, dtype=torch.bfloat16, low_cpu_mem_usage=True, device_map='cpu')
    targets = []
    for name, mod in m.named_modules():
        if isinstance(mod, nn.Linear) and mod.weight.numel() >= 1000 \
                and 'embed' not in name:
            targets.append((name, mod))
    print(f'{len(targets)} target linears', flush=True)

    blobs = {}
    meta = {'tensors': {}, 'g': UDCQ_G}
    cb = None
    tot = 0
    for i, (name, mod) in enumerate(targets):
        w = mod.weight.data.cpu()
        if cb is None:
            cb = udcq_fit_codebook(w, g=UDCQ_G)
            torch.save(cb.half(), os.path.join(HERE, 'qwen38_udcq_cb.pt'))
        p = udcq_quantize(w, cb, g=UDCQ_G)
        ent = {}
        for k in ('idx', 'scale', 'sign_packed'):
            gk = f'{name}.{k}'
            blobs[gk] = p[k].contiguous()
            tot += p[k].numel() * p[k].element_size()
            ent[k] = gk
        ent.update(out_f=p['out_f'], in_f=p['in_f'], N=p['N'])
        meta['tensors'][name] = ent
        mod.weight.data = torch.empty(0)
        del w, p
        if (i + 1) % 100 == 0:
            gc.collect()
            print(f'  [{i+1}/{len(targets)}] blob {tot/1e9:.2f}GB '
                  f'{time.time()-t0:.0f}s', flush=True)
    meta['codebook'] = cb.tolist()
    print(f'saving {tot/1e9:.2f}GB ...', flush=True)
    save_file(blobs, PACK, metadata={'format': 'udcq'})
    json.dump(meta, open(META, 'w'))
    print(f'DONE {os.path.getsize(PACK)/1e9:.2f}GB in {time.time()-t0:.0f}s',
          flush=True)


if __name__ == '__main__':
    main()
