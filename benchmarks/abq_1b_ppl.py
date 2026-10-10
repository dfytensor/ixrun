# -*- coding: utf-8 -*-
"""ABQ quality on MiniCPM5-1B: PPL + QA vs references."""
import sys, time, gc, difflib
sys.path.insert(0, r'E:\IXRUN')
import pandas  # noqa: F401
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from ixrun.config import MODEL_PATH, DATASET_CACHE
from ixrun.eval_utils import eval_ppl, load_wikitext
from ixrun.linear import iter_quantizable_linears
from benchmarks.abq_quantize import abq_train_books, abq_quantize

CFG = eval(sys.argv[1]) if len(sys.argv) > 1 else [(16, 16, 6)]

tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
m = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, torch_dtype=torch.bfloat16, trust_remote_code=True)
m.eval().cuda()
texts = load_wikitext(cache_dir=DATASET_CACHE)
targets = list(iter_quantizable_linears(m))
orig = [(n, mod.weight.data.detach().cpu().clone())
        for n, mod in targets]
ppl0 = eval_ppl(m, tok, texts)
print(f'[bf16] ppl {ppl0:.2f}', flush=True)

for (nb, nl, it) in CFG:
    with torch.no_grad():
        for (n, mod), (_, w0) in zip(targets, orig):
            mod.weight.data.copy_(w0.cuda())
    Gs = [mod.weight.data.float().reshape(-1, 32)
          for _, mod in targets]
    t0 = time.time()
    books = abq_train_books(Gs, K=nl, iters=it, n_books=nb,
                            verbose=True)
    del Gs
    torch.cuda.empty_cache()
    with torch.no_grad():
        for n, mod in targets:
            Wq, sel, codes, s, bpw = abq_quantize(
                mod.weight.data.float(), books)
            mod.weight.data = Wq.to(mod.weight.dtype)
    ppl = eval_ppl(m, tok, texts)
    print(f'[abq books={nb}x{nl}] bpw {bpw:.3f} | ppl {ppl:.2f} '
          f'({time.time()-t0:.0f}s)', flush=True)

print('\nrefs: bf16 56.02 | ig32(32,16)warm 56.13 | GSQ 5.5bpw 57.2 '
      '| UDCQ 6bpw 58.06', flush=True)
print('DONE', flush=True)
