# -*- coding: utf-8 -*-
"""bf16x shared-emax study on MiniCPM5-1B.

Format: groups of 16 elems share one emax (8b); emax is shared across
KG consecutive groups (KG=1 == per-16, PEAK-Q baseline). Per element:
delta = emax - expo in dbits (per-tensor chosen), mantissa truncation
to mb bits, sign 1b. Reports bpw / rel_err / SNR per config + ppl for
the promising points (offline weight swap, torch decode only).
"""
import gc
import sys
import time

sys.path.insert(0, r'E:\IXRUN')
import pandas  # noqa: F401
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from ixrun.config import MODEL_PATH, DATASET_CACHE
from ixrun.eval_utils import eval_ppl, load_wikitext
from ixrun.linear import iter_quantizable_linears

G = 16            # elems per emax-group (PEAK-Q granularity)
MB = 7            # mantissa bits kept (of bf16's 7) — try 7 and 5
CONFIGS = [(64, 2), (64, 1), (64, 0)]


def bf16x_decode(W, kg, mb):
    """W bf16 [out,in] -> reconstructed bf16 + storage bits/elem."""
    b = W.view(torch.uint16).int()
    s = (b >> 15) & 1
    e = (b >> 7) & 0xFF
    m = b & 0x7F
    e_nz = torch.where(e == 0, torch.ones_like(e), e)
    # supergroup emax over kg*G elems (flattened)
    sg = kg * G
    flat_e = e_nz.reshape(-1)
    pad = (-flat_e.numel()) % sg
    if pad:
        flat_e = torch.cat([flat_e,
                            torch.full((pad,), int(flat_e.min()),
                                       device=flat_e.device)])
    emax = flat_e.reshape(-1, sg).max(dim=1, keepdim=True).values
    delta = (emax - flat_e.reshape(-1, sg)).clamp_min(0)
    dbits = int(torch.log2(torch.tensor(
        float(delta.max().item()) + 1)).ceil().clamp_min(1))
    d_q = delta.clamp_max((1 << dbits) - 1)
    e_dq = emax - d_q
    e_dq = torch.where(flat_e.reshape(-1, sg) == 0,
                       torch.zeros_like(e_dq), e_dq)
    e_dq = e_dq.clamp(0, 0xFE).int()
    m_q = (m.reshape(-1) >> (7 - mb)) << (7 - mb)
    n = e.numel()
    em = e_dq.reshape(-1)[:n]
    bits = (s.reshape(-1) << 15) | (em << 7) | m_q
    W2 = bits.to(torch.uint16).view(torch.bfloat16) \
        .reshape(W.shape)
    bits_per_elem = (8.0 / kg + dbits + mb + 1)
    return W2, bits_per_elem, dbits


def int8_decode(W):
    qmax = 127
    s = W.float().abs().amax(dim=1, keepdim=True).clamp_min(1e-12) / qmax
    Wq = ((W.float() / s).round().clamp(-qmax, qmax)) * s
    return Wq.to(torch.bfloat16), 8.0 + 16.0 / W.shape[1], 0


def main():
    tok = AutoTokenizer.from_pretrained(MODEL_PATH,
                                        trust_remote_code=True)
    texts = load_wikitext(cache_dir=DATASET_CACHE)
    m = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, dtype=torch.bfloat16,
        trust_remote_code=True).eval().to('cuda')
    targets = list(iter_quantizable_linears(m))
    orig = {n: mod.weight.data.clone() for n, mod in targets}
    print(f'{len(targets)} linears', flush=True)

    for kg, mb in CONFIGS:
        t0 = time.time()
        errs = []
        snrs = []
        bpws = []
        dbits_max = 0
        for name, mod in targets:
            W = mod.weight.data
            if mb == 0:
                W2, bpw, dbits = int8_decode(W)
            else:
                W2, bpw, dbits = bf16x_decode(W, kg, mb)
            dbits_max = max(dbits_max, dbits)
            w = orig[name].float().cuda()
            r = W2.float().cuda()
            errs.append(((w - r).norm() / w.norm()).item())
            snrs.append(10 * torch.log10(
                w.var() / ((w - r) ** 2).mean()).item())
            bpws.append(bpw)
            mod.weight.data = W2
        m_gpu = m  # already on cuda
        ppl = eval_ppl(m_gpu, tok, texts)
        print(f'[bf16x KG={kg:>2} mb={mb}] bpw={sum(bpws)/len(bpws):.2f} '
              f'rel_err={sum(errs)/len(errs):.4f} '
              f'SNR={sum(snrs)/len(snrs):.1f}dB dbits={dbits_max} '
              f'ppl={ppl:.2f} ({time.time()-t0:.0f}s)', flush=True)
        for name, mod in targets:
            mod.weight.data = orig[name]
        gc.collect()
        torch.cuda.empty_cache()
    print('refs: bf16 56.02 | UDCQ 6bpw 58.06 | PEAK-Q per-16 ~10.5bpw')


if __name__ == '__main__':
    main()
