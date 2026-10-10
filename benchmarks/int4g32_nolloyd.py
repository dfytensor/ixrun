# -*- coding: utf-8 -*-
"""Parametric-only (lloyd=0) vs warm quality on MiniCPM5-1B."""
import sys, time, gc, difflib
sys.path.insert(0, r'E:\IXRUN')
import pandas  # noqa: F401
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from ixrun.config import MODEL_PATH, DATASET_CACHE
from ixrun.eval_utils import eval_ppl, load_wikitext
from ixrun.linear import iter_quantizable_linears
from benchmarks.int4g32_sweep import quant_gw, QA, gen

tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
m = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, torch_dtype=torch.bfloat16, trust_remote_code=True)
m.eval().cuda()
texts = load_wikitext(cache_dir=DATASET_CACHE)

targets = list(iter_quantizable_linears(m))
orig = [(name, mod.weight.data.detach().cpu().clone())
        for name, mod in targets]
ppl_bf16 = eval_ppl(m, tok, texts)
ans_bf16 = [gen(m, tok, q) for q in QA]
print(f'[bf16] ppl {ppl_bf16:.2f}', flush=True)

CONFIGS = [
    (32, 8, 0),    # params-only int4
    (32, 16, 0),   # params-only int5
    (16, 8, 0),    # params-only int4 g16
    (32, 16, 25),  # warm anchor
]


def restore():
    with torch.no_grad():
        for (name, mod), (_, w0) in zip(targets, orig):
            mod.weight.data.copy_(w0.cuda())


print(f'{"group":>5} {"nlev":>4} {"lloyd":>5} {"bpw":>6} {"ppl":>9} '
      f'{"QA-sim":>7}', flush=True)
for group, nlev, lloyd in CONFIGS:
    with torch.no_grad():
        for name, mod in targets:
            mod.weight.data = quant_gw(mod.weight.data, group=group,
                                       nlev=nlev, lloyd=lloyd).to(
                                           mod.weight.dtype)
    ppl = eval_ppl(m, tok, texts)
    ans = [gen(m, tok, q) for q in QA]
    qa = sum(difflib.SequenceMatcher(None, a, b).ratio()
             for a, b in zip(ans, ans_bf16)) / len(QA) * 100
    bits = (nlev.bit_length() - 1) + 1 + 8 / group
    print(f'{group:>5} {nlev:>4} {lloyd:>5} {bits:>6.2f} {ppl:>9.2f} '
          f'{qa:>6.1f}%', flush=True)
    restore()

print('\nrefs: bf16 56.02 | GSQ 5.5bpw 57.2 | UDCQ 6bpw 58.06', flush=True)
print('DONE', flush=True)
