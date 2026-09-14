# -*- coding: utf-8 -*-
"""HPQ (Hierarchical Product Quantization) on MiniCPM5-1B — full-model
evaluation vs bf16 / UDCQ / GMM.

HPQ as in D:\\BaiduNetdiskDownload\\HPQ: 4x4 blocks flattened to 16-dim
vectors; PQ = m=4 subspaces x k-means codebook; L2 = second PQ on the
residual. Storage = m*levels*log2(k) bits per 16 elements:
  L1 k16 = 1.00 bpw | L2 k16 = 2.00 | L2 k64 = 3.00

Run: python -X utf8 -m benchmarks.hpq_minicpm5
"""
import sys, time, gc
sys.path.insert(0, r'E:\IXRUN')
import pandas
import torch

from transformers import AutoTokenizer, AutoModelForCausalLM
from ixrun.config import MODEL_PATH, DATASET_CACHE
from ixrun.eval_utils import eval_ppl, load_wikitext
from ixrun.linear import iter_quantizable_linears

BS = 4
DIM = BS * BS
M = 4
SUB = DIM // M


def kmeans_gpu(X, k, iters=25, blk=200_000):
    """Simple Lloyd k-means on GPU (X: [n, d])."""
    n = X.shape[0]
    idx = torch.randperm(n, device=X.device)[:k]
    C = X[idx].clone()
    for _ in range(iters):
        assign = torch.empty(n, dtype=torch.long, device=X.device)
        for b0 in range(0, n, blk):
            xb = X[b0:b0 + blk]
            assign[b0:b0 + blk] = torch.cdist(xb, C).argmin(1)
        for j in range(k):
            m = assign == j
            if m.any():
                C[j] = X[m].mean(0)
    return C


def pq_encode_decode(X, codes_list, m=M, k=16, sub=None):
    """One PQ level: fit codebooks on X, return (recon, codes, codebooks)."""
    codebooks = []
    recon = torch.empty_like(X)
    codes = []
    sub_dim = X.shape[1] // m
    for i in range(m):
        sub = X[:, i * sub_dim:(i + 1) * sub_dim]
        C = kmeans_gpu(sub, k)
        d = torch.cdist(sub, C)
        a = d.argmin(1)
        recon[:, i * sub_dim:(i + 1) * sub_dim] = C[a]
        codebooks.append(C)
        codes.append(a)
    return recon, codes, codebooks


def hpq_quantize(W, levels=2, k=16, m=M):
    """W: [out, in] gpu float. Returns recon, bpw."""
    of, inf = W.shape
    pad_r = (-of) % BS
    pad_c = (-inf) % BS
    if pad_r or pad_c:
        W = torch.nn.functional.pad(W, (0, pad_c, 0, pad_r))
    H, Wd = W.shape
    blocks = W.reshape(H // BS, BS, Wd // BS, BS).permute(0, 2, 1, 3) \
        .reshape(-1, DIM)
    # per-block scale (HPQ x group-scale hybrid): normalize each 4x4
    # block before PQ so the codebooks see unit-norm-ish data; the scale
    # costs 16 bit / 16 elem = +1.0 bpw (fp16) and is what UDCQ/GMM use
    # at group granularity (their 4x lower error at same bpw)
    bs_scale = blocks.abs().amax(dim=1, keepdim=True).clamp_min(1e-8)
    blocks = blocks / bs_scale
    recon = torch.zeros_like(blocks)
    resid = blocks.clone()
    for _ in range(levels):
        r, _, _ = pq_encode_decode(resid, None, k=k, m=m)
        recon += r
        resid = blocks - recon
    recon = recon * bs_scale
    out = recon.reshape(H // BS, Wd // BS, BS, BS).permute(0, 2, 1, 3) \
        .reshape(H, Wd)
    out = out[:of, :inf]
    bpw = m * levels * (k.bit_length() - 1) / DIM + \
        16.0 / DIM
    return out, bpw


def main():
    tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    texts = load_wikitext(cache_dir=DATASET_CACHE)
    results = []
    for levels, k, mq in [(2, 32, 8)]:
        m = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH, torch_dtype=torch.bfloat16, trust_remote_code=True)
        m.eval()
        t0 = time.time()
        bpw_acc = []
        err_acc = []
        for _, mod in iter_quantizable_linears(m):
            W = mod.weight.data.float().cuda()
            rec, bpw = hpq_quantize(W, levels=levels, k=k, m=mq)
            err = ((W - rec).norm() / W.norm()).item()
            err_acc.append(err)
            bpw_acc.append(bpw)
            mod.weight.data = rec.to(torch.bfloat16).cpu()
            del W, rec
            torch.cuda.empty_cache()
        m = m.cuda()
        ppl = eval_ppl(m, tok, texts)
        bpw = sum(bpw_acc) / len(bpw_acc)
        err = sum(err_acc) / len(err_acc)
        results.append((f'HPQ L{levels} k{k} m{mq}', bpw, err, ppl))
        print(f'[hpq] L{levels} k{k} m{mq}: bpw={bpw:.2f} '
              f'rel_err={err:.4f} ppl={ppl:.2f} ({time.time()-t0:.0f}s)',
              flush=True)
        del m
        gc.collect(); torch.cuda.empty_cache()
    print('\n=== MiniCPM5-1B: HPQ vs references ===')
    print(f'{"scheme":<14}{"bpw":>7}{"relerr":>9}{"ppl":>9}')
    for name, bpw, err, ppl in results:
        print(f'{name:<14}{bpw:>7.2f}{err:>9.4f}{ppl:>9.2f}')
    print('ref: bf16 56.02 | UDCQ 6bpw 58.06 | GMM K32 6bpw 57.92')


if __name__ == '__main__':
    main()
