# -*- coding: utf-8 -*-
"""PY gsq pure-decode rate via differential timing (prefill cancels)."""
import sys, time
sys.path.insert(0, r'E:\IXRUN')
import pandas  # noqa: F401
MAXCTX = int(__import__('os').environ.get('PY1B_CTX', '2048'))
import torch
from ixrun.config import MODEL_PATH, DATASET_CACHE
from ixrun.eval_utils import load_wikitext
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
texts = load_wikitext(cache_dir=DATASET_CACHE)
prompt = '\n'.join(texts)[20000:20600]

from ixrun.step_graph import StepGraphEngine
sg = StepGraphEngine.from_pretrained(MODEL_PATH, codec='gsq',
                                     max_ctx=MAXCTX, verbose=False)


def run(n):
    t0 = time.perf_counter()
    sg.generate(prompt, max_new_tokens=n)
    torch.cuda.synchronize()
    return time.perf_counter() - t0


run(8)                      # warm (graph captured on first call)
t8 = min(run(8) for _ in range(3))
t136 = min(run(136) for _ in range(3))
dec = (t136 - t8) / 128
print(f't8={t8:.2f}s t136={t136:.2f}s')
out = sg.generate(prompt, max_new_tokens=136)
print('gen token count:', len(tok(out).input_ids), flush=True)
print(f'PY-gsq PURE decode: {1/dec:.1f} tok/s ({dec*1000:.1f} ms/tok)',
      flush=True)
