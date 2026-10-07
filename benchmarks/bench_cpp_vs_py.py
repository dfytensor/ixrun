# -*- coding: utf-8 -*-
"""Head-to-head: CppGsqEngine vs StepGraphEngine(gsq).
TTFT + decode tok/s + total wall on the same prompt."""
import sys, time
sys.path.insert(0, r'E:\IXRUN')
import pandas  # noqa: F401
import torch
from ixrun.config import MODEL_PATH, DATASET_CACHE
from ixrun.eval_utils import load_wikitext
from transformers import AutoTokenizer, AutoModelForCausalLM

N = 128
tok = AutoTokenizer.from_pretrained(MODEL_PATH,
                                    trust_remote_code=True)
texts = load_wikitext(cache_dir=DATASET_CACHE)
prompt = '\n'.join(texts)[20000:20600]
n_in = len(tok(prompt).input_ids)
print(f'prompt {n_in} tok, gen {N}', flush=True)

# ---- A. C++ engine ----
from ixrun.cpp_engine import CppGsqEngine
m = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, dtype=torch.bfloat16,
    trust_remote_code=True).eval().cuda()
eng = CppGsqEngine(m, ctx=512)
ids = tok(prompt, return_tensors='pt').input_ids[0].tolist()

eng.reset()
t0 = time.perf_counter()
nxt = eng.prefill(ids)
torch.cuda.synchronize()
t_ttft = time.perf_counter() - t0
t0 = time.perf_counter()
gtoks = eng._gen_block(nxt, len(ids), N)
torch.cuda.synchronize()
t_dec = time.perf_counter() - t0
print(f'[cpp-gsq] TTFT {t_ttft*1000:.0f}ms '
      f'({n_in/t_ttft:.0f} tok/s) | decode {N/t_dec:.1f} tok/s',
      flush=True)
del eng, m
torch.cuda.empty_cache()

# ---- B. Python StepGraph gsq ----
from ixrun.step_graph import StepGraphEngine
sg = StepGraphEngine.from_pretrained(MODEL_PATH, codec='gsq',
                                     verbose=False)
chunks = []
t0 = time.perf_counter()
out_py = sg.generate(prompt, max_new_tokens=N)
torch.cuda.synchronize()
t_total = time.perf_counter() - t0
print(f'[py-gsq ] total (prefill+decode) {t_total:.1f}s '
      f'-> {N/t_total:.1f} tok/s amortized', flush=True)
t0 = time.perf_counter()
out2 = sg.generate(prompt, max_new_tokens=N)
torch.cuda.synchronize()
print(f'          rerun {time.perf_counter()-t0:.1f}s '
      f'(warm)', flush=True)
print(f'cpp text : {tok.decode(gtoks)[:80]!r}', flush=True)
