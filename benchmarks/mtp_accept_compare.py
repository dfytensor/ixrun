# -*- coding: utf-8 -*-
"""MTP acceptance proxy on 27B: teacher-forcing agreement between the
MTP draft (MTP(emb(t_{i+1}), h_i)) and the main-model verification
argmax(logits_{i+1}) — directly comparable to the spec E metric.

Compares: UDCQ blob (E=2.8 baseline) vs GMM 5-bit 6bpw (new codec).
Run: $env:UDCQ_CUDA_GEMV='1'
     python -X utf8 -m benchmarks.mtp_accept_compare
"""
import sys, time, gc, json
sys.path.insert(0, r'E:\IXRUN')
import pandas
import torch

BLOB = r'E:\IXRUN\experiments\qwen38_udcq\q38_blob.pt'
MODEL = r'E:\models\Qwen3.8-27B'
TEXT = ('The theory of relativity states that the speed of light in '
        'vacuum is the same for all observers, regardless of their '
        'relative motion. This principle has profound implications for '
        'our understanding of space and time, and it forms the '
        'foundation of modern physics. In particular, the mass of a '
        'body moving with velocity v is given by the relativistic '
        'formula, where the rest mass increases with speed. Energy and '
        'mass are equivalent, as expressed by the famous equation.')


def agreement(m, mtp, ids, tok, dev='cuda'):
    """Teacher-forcing t+1 agreement: MTP(emb(t[i+1]), h[i]) vs
    argmax(logits[i+1]) over all positions."""
    from transformers.cache_utils import StaticCache
    tm = m.model.language_model if hasattr(m.model, 'language_model') \
        else m.model
    H = tm.config.hidden_size
    cache = StaticCache(config=m.config, max_cache_len=len(ids) + 8)
    emb_w = None
    for mod in m.modules():
        if type(mod).__name__ == '_CpuEmbed':
            emb_w = mod.weight_cpu
            break
    e = torch.zeros(1, 1, H, dtype=torch.bfloat16, device=dev)
    cos = torch.zeros(1, 1, 64, dtype=torch.bfloat16, device=dev)
    sin = torch.zeros_like(cos)
    pos = torch.zeros(1, dtype=torch.long, device=dev)
    pos_all = torch.arange(len(ids) + 8, device=dev).unsqueeze(0)
    with torch.no_grad():
        ca, sa = tm.rotary_emb(
            torch.zeros(1, len(ids) + 8, H, dtype=torch.bfloat16,
                        device=dev), pos_all)
    if ca.dim() == 4:
        ca = ca[:, :, 0]
    if ca.dim() == 3:
        ca, sa = ca[0], sa[0]
    hs, lgs = [], []
    with torch.no_grad():
        for i, tid in enumerate(ids):
            e.copy_(emb_w[tid].view(1, 1, H).to(dev, torch.bfloat16))
            cos.copy_(ca[i].view(1, 1, -1))
            sin.copy_(sa[i].view(1, 1, -1))
            pos.fill_(i)
            h = e
            for layer in tm.layers:
                h = layer(h, position_embeddings=(cos, sin),
                          attention_mask=None, position_ids=pos.view(1, 1),
                          past_key_values=cache)
                if isinstance(h, tuple):
                    h = h[0]
            hs.append(tm.norm(h))
            lgs.append(m.lm_head(tm.norm(h)))
    hit = tot = 0
    with torch.no_grad():
        for i in range(len(ids) - 2):
            lg, _ = mtp.forward2(
                emb_w[ids[i + 1]].view(1, 1, H).to(dev, torch.bfloat16),
                hs[i], pos.view(1, 1))
            a = int(lg[:, -1].argmax(-1).item())
            b = int(lgs[i + 1][:, -1].argmax(-1).item())
            hit += (a == b)
            tot += 1
    del cache, hs, lgs
    gc.collect()
    torch.cuda.empty_cache()
    return hit / tot


def main():
    from transformers import AutoTokenizer
    from ixrun.q38_graph import _build_from_blob, _apply_static_attention
    from ixrun.fla_patch import apply_fla_kernels
    from ixrun.gdn_seq_patch import apply_gdn_sequential_patch
    from ixrun.q38_spec import _load_mtp
    from ixrun.linear import _set_parent_child, iter_quantizable_linears

    tok = AutoTokenizer.from_pretrained(MODEL)
    apply_fla_kernels()
    apply_gdn_sequential_patch()
    _apply_static_attention()
    m = _build_from_blob(BLOB, MODEL, verbose=True)
    m.eval()
    mtp = _load_mtp(m, MODEL)
    ids = tok(TEXT, return_tensors='pt')['input_ids'][0].tolist()
    print(f'[mtp-cmp] {len(ids)} tokens', flush=True)

    t0 = time.time()
    r_udcq = agreement(m, mtp, ids, tok)
    print(f'[mtp-cmp] UDCQ 6bpw  agreement={r_udcq*100:.1f}%  '
          f'({time.time()-t0:.0f}s)', flush=True)

    # ---- re-encode to GMM 5-bit 6bpw ----
    from benchmarks.gmm5 import gmm5_pack, Gmm5Linear
    from benchmarks.gmm_stream_minicpm5 import fit_bayesian_gmm

    idxj = json.load(open(
        rf'{MODEL}\model.safetensors.index.json'))['weight_map']

    def ckpt_key(name):
        for cand in (name + '.weight',
                     name.replace('model.', 'model.language_model.', 1)
                     + '.weight'):
            if cand in idxj:
                return cand
        return None

    from safetensors import safe_open
    names = [n for n, mod in m.named_modules()
             if type(mod).__name__ == 'UdcqLinear']
    samp = []
    for name in names[:12]:
        key = ckpt_key(name)
        if key is None:
            continue
        with safe_open(rf'{MODEL}\{idxj[key]}', 'pt') as sf:
            w = sf.get_tensor(key).to(torch.bfloat16)
        wf = w.reshape(-1).float()
        samp.append(wf[torch.randint(0, wf.numel(), (300_000,))])
    mu, _, _ = fit_bayesian_gmm(torch.cat(samp), K=32, iters=40)
    print(f'[mtp-cmp] GMM K=32 fitted ({len(mu)} comps)', flush=True)
    t0 = time.time()
    for i, name in enumerate(names):
        key = ckpt_key(name)
        if key is None:
            continue
        with safe_open(rf'{MODEL}\{idxj[key]}', 'pt') as sf:
            w = sf.get_tensor(key).to(torch.bfloat16)
        packed = gmm5_pack(w, mu)
        del w
        _set_parent_child(m, name, Gmm5Linear(packed))
        if (i + 1) % 50 == 0:
            gc.collect(); torch.cuda.empty_cache()
            print(f'[mtp-cmp] {i+1}/{len(names)} '
                  f'({time.time()-t0:.0f}s, gpu '
                  f'{torch.cuda.memory_allocated()/1e9:.1f}GB)', flush=True)
    print(f'[mtp-cmp] re-encoded in {time.time()-t0:.0f}s', flush=True)
    gc.collect(); torch.cuda.empty_cache()
    t0 = time.time()
    r_gmm5 = agreement(m, mtp, ids, tok)
    print(f'[mtp-cmp] GMM-5bit 6bpw agreement={r_gmm5*100:.1f}%  '
          f'({time.time()-t0:.0f}s)', flush=True)
    print(f'[mtp-cmp] verdict: UDCQ {r_udcq*100:.1f}% vs GMM5 {r_gmm5*100:.1f}%',
          flush=True)


if __name__ == '__main__':
    main()
