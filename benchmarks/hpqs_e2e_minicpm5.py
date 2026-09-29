# -*- coding: utf-8 -*-
"""E2E: MiniCPM5-1B with down_proj+o_proj wrapped in HpqsLinear
(hand-CUDA GEMV, 7bpw), rest bf16. Measures eager decode tok/s vs the
all-bf16 baseline + sanity text. Packed codes cached on disk."""
import os
import sys
import time

sys.path.insert(0, r'E:\IXRUN')
import pandas  # noqa: F401
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from ixrun.config import MODEL_PATH, DATASET_CACHE
from ixrun.eval_utils import load_wikitext
from benchmarks.hpqs_runtime import HpqsLinear, hpqs_pack

CACHE = r'E:\IXRUN\experiments\hpqs_minicpm5\down_o_packs.pt'
N_NEW = 32


def gen_loop(m, ids, n_new):
    from transformers.cache_utils import DynamicCache
    cache = DynamicCache()
    t0 = time.time()
    n = 0
    with torch.no_grad():
        out = m(ids, past_key_values=cache, use_cache=True)
        nxt = out.logits[:, -1].argmax(-1, keepdim=True)
        for _ in range(n_new):
            out = m(nxt, past_key_values=cache, use_cache=True)
            nxt = out.logits[:, -1].argmax(-1, keepdim=True)
            n += 1
    torch.cuda.synchronize()
    return n / (time.time() - t0), int(nxt.item())


def main():
    tok = AutoTokenizer.from_pretrained(MODEL_PATH,
                                        trust_remote_code=True)
    texts = load_wikitext(cache_dir=DATASET_CACHE)
    big = '\n'.join(texts)
    prompt_ids = tok(big[20000:28000], return_tensors='pt') \
        .input_ids[:, :512].cuda()

    m = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, dtype=torch.bfloat16,
        trust_remote_code=True).eval().cuda()
    t0 = time.time()
    base_tps, base_tok = gen_loop(m, prompt_ids, N_NEW)
    print(f'bf16 baseline: {base_tps:.1f} tok/s ({time.time()-t0:.0f}s '
          f'load+run), last tok {base_tok}', flush=True)

    if os.path.exists(CACHE):
        packs = torch.load(CACHE, weights_only=False)
        print(f'loaded {len(packs)} packs from cache', flush=True)
    else:
        from ixrun.linear import iter_quantizable_linears
        packs = {}
        t0 = time.time()
        for name, mod in iter_quantizable_linears(m):
            if not ('down_proj' in name or 'o_proj' in name):
                continue
            W = mod.weight.data.float().cuda()
            packs[name] = hpqs_pack(W)
            del W
            torch.cuda.empty_cache()
        os.makedirs(os.path.dirname(CACHE), exist_ok=True)
        torch.save(packs, CACHE)
        print(f'encoded {len(packs)} layers in {time.time()-t0:.0f}s',
              flush=True)

    for name, mod in list(m.named_modules()):
        if name in packs:
            parent = m.get_submodule(name.rsplit('.', 1)[0])
            parent._modules[name.rsplit('.', 1)[1]] = \
                HpqsLinear(packs[name])
    torch.cuda.empty_cache()

    tps, tok_id = gen_loop(m, prompt_ids, N_NEW)
    print(f'HPQS down+o mixed: {tps:.1f} tok/s '
          f'({(base_tps - tps) / base_tps * 100:.1f}% slower than bf16), '
          f'last tok {tok_id}', flush=True)
    with torch.no_grad():
        out = m(prompt_ids)
    print('prefill logits finite:',
          bool(torch.isfinite(out.logits).all()), flush=True)


if __name__ == '__main__':
    main()
