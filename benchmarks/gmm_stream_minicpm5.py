# -*- coding: utf-8 -*-
"""GMM streaming inference: encode Bayesian-GMM weights into the UDCQ
packed format (sign stream = all +1 since GMM component means carry the
sign) and deploy with UdcqLinear(cache='stream') -> fused decode+GEMV,
packed GPU-resident.

bpw = 4 (idx) + 1 (fp16 scale/16) + 0.03 (all-ones sign) = 5.03

Run: python -X utf8 -m benchmarks.gmm_stream_minicpm5
"""
import sys, time, gc
sys.path.insert(0, r'E:\IXRUN')
import pandas
import torch
import torch.nn.functional as F

from transformers import AutoTokenizer, AutoModelForCausalLM
from ixrun.config import MODEL_PATH, DATASET_CACHE
from ixrun.eval_utils import bench_forward, eval_ppl, load_wikitext
from ixrun.linear import iter_quantizable_linears, _set_parent_child
from ixrun.udcq import UdcqLinear
from benchmarks.bench_gmm_minicpm5 import fit_bayesian_gmm, _nearest

GROUP = 16
K = 16


def gmm_pack(w, mu, group=GROUP):
    """Bayesian-GMM weight -> UDCQ-compatible packed dict."""
    flat = w.reshape(-1).float().cpu()
    N = flat.numel()
    pad = (-N) % group
    if pad:
        flat = torch.cat([flat, flat.new_zeros(pad)])
    g = flat.view(-1, group)
    sc = (g.abs().amax(1) / float(mu.abs().max())).clamp_min(1e-12)
    y = g / sc[:, None]
    idx = _nearest(y, mu)                       # [ng, group] in 0..K-1
    idx = idx.reshape(-1)[:N].to(torch.uint8)
    scale = sc.to(torch.float16)
    # nibble pack (low nibble = even element, matching the UDCQ kernel)
    b = idx[0::2] | (idx[1::2] << 4) if N % 2 == 0 else \
        torch.cat([idx[0::2] | (idx[1::2] << 4), idx[-1:]])
    nwords = (N + 31) // 32
    sign_packed = torch.full((nwords,), -1, dtype=torch.int32)
    return {
        'g': group, 'out_f': w.shape[0], 'in_f': w.shape[1], 'N': N,
        'idx': b.to(torch.uint8), 'scale': scale,
        'sign_packed': sign_packed,
        'codebook': mu.to(torch.float16).clone(),   # SIGNED centers
        'bits_per_weight': 4 + 16 / group + 1 / 32,
    }


def main():
    tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    texts = load_wikitext(cache_dir=DATASET_CACHE)
    m = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16, trust_remote_code=True)
    m.eval()
    targets = list(iter_quantizable_linears(m))

    # fit GMM codebook (sample)
    samp = []
    per = max(1, 3_000_000 // len(targets))
    for _, mod in targets:
        w = mod.weight.data.reshape(-1).float()
        if w.numel() > per:
            w = w[torch.randint(0, w.numel(), (per,))]
        samp.append(w)
    mu, _, _ = fit_bayesian_gmm(torch.cat(samp), K=K, iters=40)
    print(f'[gmm-stream] codebook {len(mu)} components', flush=True)

    # encode + deploy as streaming UdcqLinear
    t0 = time.time()
    for name, mod in targets:
        packed = gmm_pack(mod.weight.data, mu)
        _set_parent_child(m, name,
                          UdcqLinear(packed, bias=None, cache='stream'))
    print(f'[gmm-stream] encoded {len(targets)} linears '
          f'({time.time()-t0:.0f}s)', flush=True)
    m = m.cuda()
    gc.collect(); torch.cuda.empty_cache()

    ppl = eval_ppl(m, tok, texts)
    ms, mem = bench_forward(m, tok, warmup=3, n_runs=10)
    print(f'[gmm-stream] ppl={ppl:.2f}  fwd={ms:.0f}ms  gpu={mem:.2f}GB '
          f'(packed streaming, 5.03 bpw)', flush=True)
    # single-token decode speed (M=1 -> fused GEMV / CUDA kernel path)
    ids = tok('The theory of relativity states that',
              return_tensors='pt').input_ids.cuda()
    with torch.no_grad():
        m.generate(ids, max_new_tokens=5, do_sample=False,
                   pad_token_id=tok.eos_token_id)
        torch.cuda.synchronize()
        t0 = time.time()
        m.generate(ids, max_new_tokens=30, do_sample=False,
                   pad_token_id=tok.eos_token_id)
        torch.cuda.synchronize()
    dt = (time.time() - t0) / 30
    print(f'[gmm-stream] decode: {1/dt:.1f} tok/s ({dt*1000:.1f}ms/tok) '
          f'[UDCQ_CUDA_GEMV={"1" if __import__("os").environ.get("UDCQ_CUDA_GEMV") else "0"}]',
          flush=True)
    print(f'[ref] resident GMM g16 ppl 61.26 @2.2GB | bf16 ppl 56.02 @2.2GB',
          flush=True)


if __name__ == '__main__':
    main()
