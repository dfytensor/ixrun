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

    # ---- UDCQ for the rest (HpqsLinear is not nn.Linear -> skipped) ----
    from ixrun.udcq import deploy_udcq
    stats = deploy_udcq(m, cache='stream', verbose=True)

    hpq_bytes = sum(p['codes'].numel() + p['scale'].numel() * 2 + 4096
                    for p in packs.values())
    hpq_elems = sum(p['out_f'] * p['in_f'] for p in packs.values())
    tot_e = hpq_elems + stats['n_elems']
    bpw = (hpq_bytes + stats['total_bytes']) * 8 / tot_e
    bpw_packed6 = ((hpq_elems * 7 / 8) + stats['total_bytes']) * 8 / tot_e
    print(f'storage: HPQS {hpq_bytes/1e6:.0f}MB (unpack8) + '
          f'UDCQ {stats["total_bytes"]/1e6:.0f}MB -> {bpw:.2f} bpw '
          f'(6-bit codes would give {bpw_packed6:.2f})', flush=True)

    tps, tok_id = gen_loop(m, prompt_ids, N_NEW)
    print(f'HPQS down+o + UDCQ rest: {tps:.1f} tok/s '
          f'last tok {tok_id}', flush=True)
    with torch.no_grad():
        out = m(prompt_ids)
    print('prefill logits finite:',
          bool(torch.isfinite(out.logits).all()), flush=True)


if __name__ == '__main__':
    main()
