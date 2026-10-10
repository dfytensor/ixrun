# -*- coding: utf-8 -*-
"""UDCQ + ig32-style warm training (LS group scales + codebook refit).
Decode structure UNCHANGED (cb[nib]*scale*sign) -> zero speed cost."""
import sys
sys.path.insert(0, r'E:\IXRUN')
import torch

G = 16


def udcq_warm_quantize(W, nlev=16, iters=3, cb0=None):
    """W [out, in] fp32 -> (Wq, cb, scale) with UDCQ decode semantics:
    w = sign * cb[code] * scale[group]."""
    of, inf = W.shape
    a = W.abs().reshape(-1, G)                  # [nG, 16]
    if cb0 is None:
        # init codebook: kmeans over |w| normalized by group max
        s = a.amax(1, keepdim=True).clamp_min(1e-12)
        v = (a / s).reshape(-1)
        samp = v[torch.randperm(v.numel(), device=v.device)[:2_000_000]]
        qs = (torch.arange(nlev, device=v.device, dtype=torch.float32)
              + 0.5) / nlev
        cb = torch.quantile(samp, qs).clamp_min(1e-6).sort().values
    else:
        cb = cb0.clone()
    s = a.amax(1, keepdim=True).clamp_min(1e-12)
    for it in range(iters):
        # assign: code = argmin |a - cb[j]*s|  (== argmin |a/s - cb[j]|)
        v = a / s
        codes = (v.unsqueeze(-1) - cb).abs().argmin(-1)      # [nG,16]
        # LS scale per group: s = sum(a*cb[c]) / sum(cb[c]^2)
        cv = cb[codes]
        s = ((a * cv).sum(1, keepdim=True)
             / cv.pow(2).sum(1, keepdim=True).clamp_min(1e-12))
        s = s.clamp_min(1e-12)
    # end-of-loop codebook refit (in-loop alternation is NOT monotone)
    vf = (a / s).reshape(-1)
    cc = codes.reshape(-1)
    for j in range(nlev):
        m = cc == j
        if m.any():
            cb[j] = vf[m].mean()
    cb, _ = cb.sort()
    # final assign at converged cb
    v = a / s
    codes = (v.unsqueeze(-1) - cb).abs().argmin(-1)
    cv = cb[codes]
    s = ((a * cv).sum(1, keepdim=True)
         / cv.pow(2).sum(1, keepdim=True).clamp_min(1e-12))
    rec = (torch.sign(W.reshape(-1, G)) * cv * s).reshape(of, inf)
    return rec.to(torch.bfloat16), cb, s


if __name__ == '__main__':
    import time, gc
    import pandas  # noqa: F401
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from ixrun.config import MODEL_PATH, DATASET_CACHE
    from ixrun.eval_utils import eval_ppl, load_wikitext
    from ixrun.linear import iter_quantizable_linears
    from ixrun.udcq import udcq_fit_codebook, udcq_quantize, UDCQ_G as UG

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

    # global codebook init from a sample (like udcq_fit_codebook)
    samp = []
    for _, mod in targets:
        w = mod.weight.data.reshape(-1).float()
        n = min(w.numel(), 1_000_000)
        samp.append(w[torch.randint(0, w.numel(), (n,),
                                    device=w.device)])
    cb0 = udcq_fit_codebook(torch.cat(samp).unsqueeze(0).cpu(),
                            nlev=16, g=UG).cuda().float()
    del samp
    torch.cuda.empty_cache()
    print(f'[cb0] {[round(v,4) for v in cb0.tolist()]}', flush=True)

    with torch.no_grad():
        for n, mod in targets:
            Wq, cb, s = udcq_warm_quantize(mod.weight.data.float(),
                                           iters=6, cb0=cb0)
            mod.weight.data = Wq.to(mod.weight.dtype)
    ppl = eval_ppl(m, tok, texts)
    print(f'[UDCQ+warm] ppl {ppl:.2f}', flush=True)

    # standard UDCQ for the same-model reference
    with torch.no_grad():
        for (n, mod), (_, w0) in zip(targets, orig):
            mod.weight.data.copy_(w0.cuda())
    for n, mod in targets:
        p = udcq_quantize(mod.weight.data.float(), cb0, g=UG)
        from ixrun.udcq import _decode_udcq_ref
        rec = _decode_udcq_ref(p, device='cuda', dtype=torch.bfloat16)
        mod.weight.data = rec.to(mod.weight.dtype)
    ppl = eval_ppl(m, tok, texts)
    print(f'[UDCQ std] ppl {ppl:.2f}', flush=True)
    print('DONE', flush=True)
