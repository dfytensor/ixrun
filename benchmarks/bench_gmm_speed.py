# -*- coding: utf-8 -*-
"""GMM (g32 config) inference speed: resident-decode forward vs baselines."""
import sys, time, gc
sys.path.insert(0, r'E:\IXRUN')
import pandas
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from ixrun.config import MODEL_PATH
from ixrun.eval_utils import bench_forward
from ixrun.linear import iter_quantizable_linears

from benchmarks.bench_gmm_minicpm5 import (fit_bayesian_gmm,
                                           quantize_layer_gmm)

tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

def load():
    m = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16, trust_remote_code=True)
    m.eval()
    return m

# bf16 baseline
m = load().cuda()
ms, mem = bench_forward(m, tok, warmup=3, n_runs=10)
print(f'[speed] bf16 resident: {ms:.0f}ms  {mem:.2f}GB', flush=True)
del m; gc.collect(); torch.cuda.empty_cache()

# GMM g32 (K=32, per-16, 5-bit idx) -> decode to bf16 resident
m = load()
targets = list(iter_quantizable_linears(m))
samp = []
per = max(1, 3_000_000 // len(targets))
for _, mod in targets:
    w = mod.weight.data.reshape(-1).float()
    if w.numel() > per:
        w = w[torch.randint(0, w.numel(), (per,))]
    samp.append(w)
xs = torch.cat(samp)
mu32, _, _ = fit_bayesian_gmm(xs, K=32, iters=40)
print(f'[speed] fitting done ({len(mu32)} components)', flush=True)
t0 = time.time()
for _, mod in targets:
    wd, _ = quantize_layer_gmm(mod.weight.data, mu32, group=16,
                               idx_bits=5)
    mod.weight.data = wd.to(torch.bfloat16)
print(f'[speed] GMM encode+decode (load-time): {time.time()-t0:.0f}s',
      flush=True)
m = m.cuda()
ms, mem = bench_forward(m, tok, warmup=3, n_runs=10)
print(f'[speed] GMM g32 resident (decode-at-load): {ms:.0f}ms  '
      f'{mem:.2f}GB', flush=True)
del m; gc.collect(); torch.cuda.empty_cache()
print('note: streaming fused-GEMV kernel for GMM (idx LUT + scale + '
      'residual) is NOT implemented yet — resident path equals bf16 '
      'speed; streaming would need the UDCQ kernel ported.', flush=True)
