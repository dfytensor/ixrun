# -*- coding: utf-8 -*-
"""TTFT bench for Qwen3.8-27B UDCQ engines.

  python bench_prefill.py --engine spec --block 64 --plen 2048
  python bench_prefill.py --engine spec --block 1  --plen 2048   (legacy)
  python bench_prefill.py --engine graph --block 64 --plen 2048

Reports TTFT, prefill last-token top-8 ids (A/B correctness gate),
generated token ids, and decode tok/s. One engine build per run.
"""
import argparse
import os
import sys
import time

p = argparse.ArgumentParser()
p.add_argument('--engine', default='spec', choices=['spec', 'graph'])
p.add_argument('--block', type=int, default=64)
p.add_argument('--plen', type=int, default=2048)
p.add_argument('--gen', type=int, default=16)
p.add_argument('--max-ctx', type=int, default=4096)
a = p.parse_args()

os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
if a.engine == 'spec':
    os.environ['Q38_PREFILL_BLOCK'] = str(a.block)
else:
    os.environ['Q38_MAX_BLOCK'] = str(a.block)

sys.path.insert(0, r'E:\IXRUN')
import pandas  # noqa: F401  (must precede torch)
import torch
from transformers import AutoTokenizer

from ixrun.config import QWEN38_PATH
from ixrun.q38_spec import Q38SpecEngine
from ixrun.q38_graph import Q38GraphEngine

LOG = r'C:\Users\Administrator\AppData\Local\Temp\opencode\bp_stage.log'
_stage = open(LOG, 'w', encoding='utf-8')


def stage(msg):
    _stage.write(f'[{time.time()-T0:.0f}s] {msg}\n')
    _stage.flush()


T0 = time.time()
stage('imports done')

BLOB = r'E:\IXRUN\experiments\qwen38_udcq\q38_blob.pt'

free, total = torch.cuda.mem_get_info()
print(f'VRAM free {free/2**30:.2f} / {total/2**30:.1f} GiB', flush=True)

tok = AutoTokenizer.from_pretrained(QWEN38_PATH)
stage('tokenizer done')
base = ('The history of computing is a long and varied one. From the '
        'earliest mechanical calculators to modern processors, each '
        'generation built upon the ideas of the last. Engineers learned '
        'to pack more transistors into smaller spaces, and software grew '
        'in complexity alongside the hardware it ran on. ')
ids = tok(base, return_tensors='pt').input_ids[0].tolist()
while len(ids) < a.plen:
    ids = ids + ids
ids = ids[:a.plen]
text = tok.decode(ids)
print(f'prompt tokens: {len(ids)} | block={a.block}', flush=True)

t0 = time.time()
if a.engine == 'spec':
    eng = Q38SpecEngine.from_blob(BLOB, tokenizer=tok, max_ctx=a.max_ctx)
else:
    eng = Q38GraphEngine.from_blob(BLOB, tokenizer=tok, max_ctx=a.max_ctx)
stage(f'engine ready ({time.time()-t0:.1f}s)')
print(f'load+capture: {time.time()-t0:.1f}s', flush=True)

torch.cuda.synchronize()
t0 = time.time()
if a.engine == 'spec':
    h_last, logits_last = eng._prefill(ids)
else:
    logits_last = eng.prefill(ids)
torch.cuda.synchronize()
ttft = time.time() - t0
top = torch.topk(logits_last[0, -1].float(), 8).indices.tolist()
print(f'TTFT: {ttft*1000:.0f} ms ({len(ids)/ttft:.0f} tok/s prefill)')
print(f'prefill top-8: {top}', flush=True)

gen_ids = []
t0 = time.time()
if a.engine == 'spec':
    for chunk in eng.generate(text, max_new_tokens=a.gen):
        gen_ids.extend(chunk)
else:
    for piece in eng.stream(text, max_new_tokens=a.gen):
        pass
torch.cuda.synchronize()
tg = time.time() - t0
ids_out = tok(text, return_tensors='pt').input_ids[0].tolist()
print(f'gen {len(gen_ids)} tok in {tg:.2f}s = {len(gen_ids)/tg:.1f} tok/s')
print(f'gen ids: {gen_ids[:32]}')
