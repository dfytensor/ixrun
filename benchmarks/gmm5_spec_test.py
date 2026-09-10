# -*- coding: utf-8 -*-
"""GMM 5-bit speculative decode test on 27B — full integration.

GPU-accelerated encoder (idx/scale/packing all on GPU: ~2min vs 102min
CPU) -> Gmm5Linear layers -> Q38SpecEngine(m, mtp) -> spec timing.

Run: $env:UDCQ_CUDA_GEMV='1'
     python -X utf8 -m benchmarks.gmm5_spec_test
"""
import sys, time, gc, json
sys.path.insert(0, r'E:\IXRUN')
import pandas
import torch

BLOB = r'E:\IXRUN\experiments\qwen38_udcq\q38_blob.pt'
MODEL = r'E:\models\Qwen3.8-27B'


def gmm5_pack_gpu(w, mu, group=16):
    """GPU encoder: nibble low 4 bits + 1-bit high bitmap + fp16 scale."""
    dev = w.device
    flat = w.reshape(-1).float()
    N = flat.numel()
    pad = (-N) % group
    if pad:
        flat = torch.cat([flat, flat.new_zeros(pad)])
    g = flat.view(-1, group)
    mmax = float(mu.abs().max())
    sc = (g.abs().amax(1) / mmax).clamp_min(1e-12)
    y = g / sc[:, None]
    ng = y.shape[0]
    idx = torch.empty(ng, group, dtype=torch.long, device=dev)
    BLK = 20000                       # chunked argmin: one kernel, low mem
    for b0 in range(0, ng, BLK):
        yb = y[b0:b0 + BLK]
        d = (yb[:, None, :] - mu[None, :, None]).abs()   # [nb, K, group]
        idx[b0:b0 + BLK] = d.argmin(1)
    idx = idx.reshape(-1)[:N]
    low = (idx & 0xF).to(torch.uint8)
    hi = (idx >> 4).to(torch.int32)
    if N % 2 == 0:
        b = low[0::2] | (low[1::2] << 4)
    else:
        b = torch.cat([low[0::2] | (low[1::2] << 4), low[-1:]])
    nw = (N + 31) // 32
    hp = torch.zeros(nw, dtype=torch.int32, device=dev)
    for bit in range(32):
        sel = hi[bit::32]
        hp[:sel.numel()] |= sel << bit
    return {
        'g': group, 'out_f': w.shape[0], 'in_f': w.shape[1], 'N': N,
        'idx': b.to(torch.uint8), 'sign_packed': hp,
        'scale': sc.to(torch.float16),
        'codebook': mu.to(torch.float16).clone(),
        'bits_per_weight': 4 + 1 + 16 / group,
    }


def main():
    from transformers import AutoTokenizer
    from safetensors import safe_open
    from ixrun.q38_graph import _build_from_blob, _apply_static_attention
    from ixrun.fla_patch import apply_fla_kernels
    from ixrun.gdn_seq_patch import apply_gdn_sequential_patch
    from ixrun.q38_spec import _load_mtp, Q38SpecEngine
    from ixrun.linear import _set_parent_child
    from benchmarks.gmm5 import Gmm5Linear
    from benchmarks.gmm_stream_minicpm5 import fit_bayesian_gmm

    tok = AutoTokenizer.from_pretrained(MODEL)
    apply_fla_kernels()
    apply_gdn_sequential_patch()
    _apply_static_attention()
    m = _build_from_blob(BLOB, MODEL, verbose=True)
    m.eval()
    mtp = _load_mtp(m, MODEL)
    idxj = json.load(open(
        rf'{MODEL}\model.safetensors.index.json'))['weight_map']

    def ckpt_key(name):
        for cand in (name + '.weight',
                     name.replace('model.', 'model.language_model.', 1)
                     + '.weight'):
            if cand in idxj:
                return cand
        return None

    names = [n for n, mod in m.named_modules()
             if type(mod).__name__ == 'UdcqLinear']
    print(f'[g5t] {len(names)} targets', flush=True)

    # fit K=32 on GPU samples
    samp = []
    for name in names[:12]:
        key = ckpt_key(name)
        with safe_open(rf'{MODEL}\{idxj[key]}', 'pt') as sf:
            w = sf.get_tensor(key).to(torch.bfloat16)
        wf = w.reshape(-1).float()
        samp.append(wf[torch.randint(0, wf.numel(), (300_000,))])
    mu = fit_bayesian_gmm(torch.cat(samp), K=32, iters=40)[0].cuda()
    print(f'[g5t] K=32 fitted', flush=True)

    # GPU-accelerated re-encode
    t0 = time.time()
    for i, name in enumerate(names):
        key = ckpt_key(name)
        with safe_open(rf'{MODEL}\{idxj[key]}', 'pt') as sf:
            w = sf.get_tensor(key).to(torch.bfloat16).cuda()
        packed = gmm5_pack_gpu(w, mu)
        del w
        _set_parent_child(m, name, Gmm5Linear(packed))
        if (i + 1) % 50 == 0:
            gc.collect(); torch.cuda.empty_cache()
            print(f'[g5t] {i+1}/{len(names)} ({time.time()-t0:.0f}s, gpu '
                  f'{torch.cuda.memory_allocated()/1e9:.1f}GB)', flush=True)
    print(f'[g5t] GPU re-encode done in {time.time()-t0:.0f}s', flush=True)
    gc.collect(); torch.cuda.empty_cache()

    # spec engine on the GMM5 model
    e = Q38SpecEngine(m, mtp, tokenizer=tok, verbose=True)
    ids = tok('The theory of relativity states that',
              return_tensors='pt')['input_ids'][0].tolist()
    out = []
    t0 = time.time()
    for batch in e._spec_iter(ids, 40):
        out.extend(batch)
    dt = time.time() - t0
    print(f'[g5t] GMM-5bit spec: {len(out)/dt:.1f} tok/s '
          f'({dt/len(out)*1000:.1f}ms/tok) | '
          f'gpu {torch.cuda.max_memory_allocated()/1e9:.2f}GB', flush=True)
    print(f'[g5t] -> {tok.decode(out)[:110]!r}', flush=True)


if __name__ == '__main__':
    main()
