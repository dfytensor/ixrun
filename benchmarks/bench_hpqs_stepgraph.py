# -*- coding: utf-8 -*-
"""StepGraph decode tok/s: hpqs-mixed vs udcq-stream, same harness."""
import sys
import time

sys.path.insert(0, r'E:\IXRUN')
import pandas  # noqa: F401
import torch
from ixrun.step_graph import StepGraphEngine
from ixrun.config import MODEL_PATH, DATASET_CACHE
from ixrun.eval_utils import load_wikitext
from transformers import AutoTokenizer

codec = sys.argv[1] if len(sys.argv) > 1 else 'hpqs-mixed'
tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
texts = load_wikitext(cache_dir=DATASET_CACHE)
big = '\n'.join(texts)
ids = tok(big[20000:40000], return_tensors='pt').input_ids[0, :512]

eng = StepGraphEngine.from_pretrained(codec=codec, verbose=True)
t0 = time.time()
out = eng.generate(tok.decode(ids), max_new_tokens=128)
torch.cuda.synchronize()
t_tot = time.time() - t0
print(f'[{codec}] 128 tok in {t_tot:.2f}s wall (incl prefill+decode)')
# decode-only: rerun generation, prefill timed separately
eng.hard_reset()
t0 = time.time()
eng.prefill(ids)
torch.cuda.synchronize()
t_pf = time.time() - t0
t0 = time.time()
with torch.no_grad():
    for _ in range(128):
        eng._step()
torch.cuda.synchronize()
t_dec = (time.time() - t0) / 128 * 1000
print(f'[{codec}] prefill512={t_pf*1000:.0f}ms '
      f'decode={t_dec:.2f}ms/tok = {1000/t_dec:.1f} tok/s', flush=True)
