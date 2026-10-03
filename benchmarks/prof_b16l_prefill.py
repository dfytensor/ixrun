# -*- coding: utf-8 -*-
"""Profile StepGraph prefill for the bf16xl codec."""
import sys
import time

sys.path.insert(0, r'E:\IXRUN')
import pandas  # noqa: F401
import torch
from ixrun.step_graph import StepGraphEngine
from ixrun.config import MODEL_PATH, DATASET_CACHE
from ixrun.eval_utils import load_wikitext
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
texts = load_wikitext(cache_dir=DATASET_CACHE)
big = '\n'.join(texts)
ids = tok(big[20000:40000], return_tensors='pt').input_ids[0, :512]

eng = StepGraphEngine.from_pretrained(codec='bf16xl', verbose=True)
eng.prefill(ids)   # warm (JIT etc.)
torch.cuda.synchronize()

from torch.profiler import profile, ProfilerActivity
with profile(activities=[ProfilerActivity.CPU,
                         ProfilerActivity.CUDA]) as prof:
    eng.prefill(ids)
    torch.cuda.synchronize()
print(prof.key_averages().table(sort_by='cuda_time_total',
                                row_limit=18))
