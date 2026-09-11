# -*- coding: utf-8 -*-
"""GMM 5-bit + per-32 scale (5.5bpw) precision check on MiniCPM5."""
import sys, time, gc
sys.path.insert(0, r'E:\IXRUN')
import pandas
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from ixrun.config import MODEL_PATH, DATASET_CACHE
from ixrun.eval_utils import eval_ppl, load_wikitext
from ixrun.linear import iter_quantizable_linears, _set_parent_child
from benchmarks.gmm5 import Gmm5Linear, gmm5_pack
from benchmarks.gmm5_spec_test import gmm5_pack_gpu
from benchmarks.gmm_stream_minicpm5 import fit_bayesian_gmm

tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
texts = load_wikitext(cache_dir=DATASET_CACHE)
m = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, torch_dtype=torch.bfloat16, trust_remote_code=True)
m.eval()
targets = list(iter_quantizable_linears(m))
samp = []
per = max(1, 3_000_000 // len(targets))
for _, mod in targets:
    w = mod.weight.data.reshape(-1).float()
    if w.numel() > per:
        w = w[torch.randint(0, w.numel(), (per,))]
    samp.append(w)
mu, _, _ = fit_bayesian_gmm(torch.cat(samp), K=32, iters=40)
print(f'[g5p] K=32 fitted', flush=True)

for group in (16, 32):
    m2 = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16, trust_remote_code=True)
    m2.eval()
    t0 = time.time()
    for name, mod in iter_quantizable_linears(m2):
        w = mod.weight.data
        packed = gmm5_pack(w, mu, group=group)
        _set_parent_child(m2, name, Gmm5Linear(packed))
    m2 = m2.cuda()
    ppl = eval_ppl(m2, tok, texts)
    bpw = 4 + 1 + 16 / group
    print(f'[g5p] group={group} ({bpw:.2f}bpw): ppl={ppl:.2f} '
          f'({time.time()-t0:.0f}s)', flush=True)
    del m2
    gc.collect(); torch.cuda.empty_cache()
print('[g5p] ref: bf16 56.02 | UDCQ 58.06 | GMM per-16 5-bit (prev) n/a')
