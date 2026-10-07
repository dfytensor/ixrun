# -*- coding: utf-8 -*-
"""E2E gate for CppGsqEngine: graph tokens == eager tokens, speed."""
import sys, time
sys.path.insert(0, r'E:\IXRUN')
import pandas  # noqa: F401
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from ixrun.config import MODEL_PATH, DATASET_CACHE
from ixrun.eval_utils import load_wikitext
from ixrun.cpp_engine import CppGsqEngine

m = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, dtype=torch.bfloat16, trust_remote_code=True
).eval().cuda()

t0 = time.perf_counter()
eng = CppGsqEngine(m, ctx=512)
print(f'engine ready, pack+init {time.perf_counter()-t0:.1f}s',
      flush=True)

tok = AutoTokenizer.from_pretrained(MODEL_PATH,
                                    trust_remote_code=True)
texts = load_wikitext(cache_dir=DATASET_CACHE)
big = '\n'.join(texts)
ids = tok(big[20000:21200],
          return_tensors='pt').input_ids[0, :256].tolist()

NGEN = 64

# A. graph path
eng.reset()
t0 = time.perf_counter()
nxt = eng.prefill(ids)
t_pf = time.perf_counter() - t0
t0 = time.perf_counter()
graph_toks = eng._gen_block(nxt, len(ids), NGEN)
t_g = time.perf_counter() - t0
print(f'prefill {len(ids)} tok: {t_pf:.2f}s '
      f'({len(ids)/t_pf:.0f} tok/s)', flush=True)
print(f'decode {NGEN} tok: {NGEN/t_g:.1f} tok/s', flush=True)

# B. eager reference (fresh state)
eng.reset()
nxt2 = eng.prefill(ids)
eager_toks = eng.generate_eager(nxt2, len(ids), NGEN)

match = sum(a == b for a, b in zip(graph_toks, eager_toks))
print(f'token match graph vs eager: {match}/{NGEN}', flush=True)
print(f'text: {tok.decode(graph_toks)[:150]}', flush=True)
assert match == NGEN, 'graph/eager token mismatch!'
print('E2E GATE PASSED', flush=True)
