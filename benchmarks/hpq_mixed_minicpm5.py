# -*- coding: utf-8 -*-
"""Mixed-precision: HPQ-x-scale (7bpw) on selected layers, UDCQ (6bpw)
elsewhere. Targets UDCQ-full ppl 58.06 at ~6.2-6.3 avg bpw, approaching
HPQ-full 56.93. Reconstructed bf16 weights are swapped in-place (offline
ppl study; no runtime kernels needed)."""
import gc
import re
import sys
import time

sys.path.insert(0, r'E:\IXRUN')
import pandas  # noqa: F401
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from ixrun.config import MODEL_PATH, DATASET_CACHE
from ixrun.eval_utils import eval_ppl, load_wikitext
from ixrun.linear import iter_quantizable_linears
from ixrun.udcq import (udcq_fit_codebook, udcq_quantize,
                        _decode_udcq_ref)
from benchmarks.hpq_minicpm5 import hpq_quantize

SELECTIONS = {
    'down': lambda n: 'down_proj' in n,
    'down+o': lambda n: 'down_proj' in n or 'o_proj' in n,
    'down+head2': None,     # filled below (needs block index)
    'down+o+head2': None,   # filled below
}


def head2(n):
    mt = re.search(r'layers\.(\d+)\.', n)
    if mt is None:
        return False
    b = int(mt.group(1))
    return b < 2 or b >= 22


SELECTIONS['down+head2'] = lambda n: ('down_proj' in n or head2(n))
SELECTIONS['down+o+head2'] = lambda n: ('down_proj' in n or 'o_proj' in n
                                        or head2(n))


def main():
    tok = AutoTokenizer.from_pretrained(MODEL_PATH,
                                        trust_remote_code=True)
    texts = load_wikitext(cache_dir=DATASET_CACHE)
    m = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, dtype=torch.bfloat16, trust_remote_code=True)
    m.eval()

    targets = list(iter_quantizable_linears(m))
    cb = udcq_fit_codebook(targets[0][1].weight.data.cpu())
    t0 = time.time()

    # phase 1: UDCQ recon for every layer (CPU bf16 kept)
    wu = {}
    bpw_udcq = {}
    n_elems = {}
    for name, mod in targets:
        W = mod.weight.data.float().cuda()
        packed = udcq_quantize(W, cb)
        wu[name] = _decode_udcq_ref(packed).to(torch.bfloat16).cpu()
        bpw_udcq[name] = packed['bits_per_weight']
        n_elems[name] = W.numel()
        del W, packed
        torch.cuda.empty_cache()
    print(f'[mixed] udcq recon done ({time.time()-t0:.0f}s)', flush=True)

    # phase 2: HPQ-x-scale recon for the union of selected layers
    sel_any = SELECTIONS['down+o+head2']
    wh = {}
    bpw_hpq = 7.0
    t1 = time.time()
    for name, mod in targets:
        if not sel_any(name):
            continue
        W = mod.weight.data.float().cuda()
        rec, _ = hpq_quantize(W, levels=2, k=64, m=8)
        wh[name] = rec.to(torch.bfloat16).cpu()
        del W, rec
        torch.cuda.empty_cache()
    print(f'[mixed] hpq recon done ({time.time()-t1:.0f}s)', flush=True)

    # phase 3: per-selection ppl (start from udcq, swap selected->hpq)
    for cname, pred in SELECTIONS.items():
        for name, mod in targets:
            src = wh if (sel_any(name) and pred(name)) else wu
            mod.weight.data = src[name].clone()
        m = m.cuda()
        ppl = eval_ppl(m, tok, texts)
        sel_e = sum(n_elems[n] for n, _ in targets if pred(n))
        tot_e = sum(n_elems.values())
        bpw = (bpw_hpq * sel_e + 6.0 * (tot_e - sel_e)) / tot_e
        print(f'[mixed] {cname}: bpw={bpw:.2f} ppl={ppl:.2f}',
              flush=True)
        m = m.cpu()
        torch.cuda.empty_cache()
    print('ref: bf16 56.02 | UDCQ-full 58.06 @6bpw | '
          'HPQ-full 56.93 @7bpw')


if __name__ == '__main__':
    main()
