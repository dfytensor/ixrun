# -*- coding: utf-8 -*-
"""Debug: per-matrix rec error + progressive PPL to localize collapse."""
import sys, time
sys.path.insert(0, r'E:\IXRUN')
import pandas  # noqa: F401
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from ixrun.config import MODEL_PATH, DATASET_CACHE
from ixrun.eval_utils import eval_ppl, load_wikitext
from ixrun.linear import iter_quantizable_linears
from benchmarks.int4g32_bare import quant_g32_warm

tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
m = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, torch_dtype=torch.bfloat16, trust_remote_code=True)
m.eval().cuda()
texts = load_wikitext(cache_dir=DATASET_CACHE)

# 1) single-matrix reconstruction errors
with torch.no_grad():
    for name, mod in iter_quantizable_linears(m):
        if name in ('model.layers.0.self_attn.q_proj',
                    'model.layers.0.mlp.down_proj',
                    'model.layers.12.mlp.gate_proj'):
            W = mod.weight.data
            Wq = quant_g32_warm(W)
            rel = ((W.float() - Wq.float()).norm()
                   / W.float().norm()).item()
            print(f'{name}: rel-L2 {rel:.4f}', flush=True)

# 2) progressive PPL: quantize layers 0..K
layers_done = 0
with torch.no_grad():
    for name, mod in iter_quantizable_linears(m):
        L = int(name.split('.')[2])
        if L >= layers_done:
            layers_done = L + 1
            mod.weight.data = quant_g32_warm(mod.weight.data).to(
                mod.weight.dtype)
            if layers_done in (1, 4, 8, 16, 24):
                ppl = eval_ppl(m, tok, texts)
                print(f'layers 0..{layers_done-1} quantized: ppl {ppl:.2f}',
                      flush=True)
print('DONE', flush=True)
