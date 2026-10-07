# -*- coding: utf-8 -*-
"""Multi-instance gate: two engines generate independently (no
statics cross-contamination); each matches its own eager ref."""
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
tok = AutoTokenizer.from_pretrained(MODEL_PATH,
                                    trust_remote_code=True)
texts = load_wikitext(cache_dir=DATASET_CACHE)
big = '\n'.join(texts)
ids0 = tok(big[20000:20400],
           return_tensors='pt').input_ids[0, :64].tolist()
ids1 = tok(big[30000:30400],
           return_tensors='pt').input_ids[0, :64].tolist()

t0 = time.perf_counter()
e0 = CppGsqEngine(m, ctx=512, instance=0)
t_b0 = time.perf_counter() - t0
t0 = time.perf_counter()
e1 = CppGsqEngine(m, ctx=512, instance=1)
t_b1 = time.perf_counter() - t0
print(f'instances built: {t_b0:.0f}s + {t_b1:.0f}s', flush=True)

# interleaved generation: A, B, A, B — states must not mix
g0 = []
g1 = []
n0 = e0.prefill(ids0)
n1 = e1.prefill(ids1)
g0 += e0._gen_block(n0, len(ids0), 16)
g1 += e1._gen_block(n1, len(ids1), 16)
g0 += e0._replay_block(len(ids0) + 16, 16)
g1 += e1._replay_block(len(ids1) + 16, 16)

# refs (fresh, sequential)
e0.reset(); r0 = e0.prefill(ids0)
r0t = e0.generate_eager(r0, len(ids0), 32)
e1.reset(); r1 = e1.prefill(ids1)
r1t = e1.generate_eager(r1, len(ids1), 32)

m0 = sum(a == b for a, b in zip(g0, r0t))
m1 = sum(a == b for a, b in zip(g1, r1t))
cross = sum(a == b for a, b in zip(g0, g1))
print(f'eng0 graph vs eager: {m0}/32', flush=True)
print(f'eng1 graph vs eager: {m1}/32', flush=True)
print(f'text0: {tok.decode(g0)[:60]!r}', flush=True)
print(f'text1: {tok.decode(g1)[:60]!r}', flush=True)
assert m0 == 32 and m1 == 32, 'instance contamination!'
print('MULTI-INSTANCE GATE PASSED', flush=True)
