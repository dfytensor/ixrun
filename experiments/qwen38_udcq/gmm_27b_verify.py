# -*- coding: utf-8 -*-
"""Qwen3.8-27B GMM verification: fit Bayesian-GMM codebook on real
weights, re-encode all 497 linears into UDCQ-compatible packed format,
generate + measure speed (UDCQ_CUDA_GEMV=1 for the hand CUDA kernel).

Run:  $env:UDCQ_CUDA_GEMV='1'
      python -X utf8 -m experiments.qwen38_udcq.gmm_27b_verify
"""
import sys, time, gc, json
sys.path.insert(0, r'E:\IXRUN')
import pandas
import torch

BLOB = r'E:\IXRUN\experiments\qwen38_udcq\q38_blob.pt'
MODEL = r'E:\models\Qwen3.8-27B'

from transformers import AutoTokenizer
from safetensors import safe_open
from ixrun.q38_graph import _build_from_blob, _apply_static_attention
from ixrun.fla_patch import apply_fla_kernels
from ixrun.gdn_seq_patch import apply_gdn_sequential_patch
from ixrun.linear import _set_parent_child
from ixrun.udcq import UdcqLinear
from benchmarks.gmm_stream_minicpm5 import gmm_pack, fit_bayesian_gmm

K = 16


def ckpt_key(name, idx):
    for cand in (name + '.weight',
                 name.replace('model.', 'model.language_model.', 1)
                 + '.weight'):
        if cand in idx:
            return cand
    return None


def main():
    tok = AutoTokenizer.from_pretrained(MODEL)
    apply_fla_kernels()
    apply_gdn_sequential_patch()
    _apply_static_attention()
    m = _build_from_blob(BLOB, MODEL, verbose=True)
    m.eval()
    idx = json.load(open(
        rf'{MODEL}\model.safetensors.index.json'))['weight_map']

    # ---- fit GMM on a sample of real layer weights ----
    targets = []
    for name, mod in m.named_modules():
        if type(mod).__name__ == 'UdcqLinear':
            targets.append((name, mod))
    print(f'[g27b] {len(targets)} UdcqLinear targets', flush=True)
    t0 = time.time()
    samp = []
    for name, mod in targets[:12]:
        key = ckpt_key(name, idx)
        if key is None:
            continue
        with safe_open(rf'{MODEL}\{idx[key]}', 'pt') as sf:
            w = sf.get_tensor(key).to(torch.bfloat16)
        wf = w.reshape(-1).float()
        n = min(300_000, wf.numel())
        samp.append(wf[torch.randint(0, wf.numel(), (n,))])
    mu, _, _ = fit_bayesian_gmm(torch.cat(samp), K=K, iters=40)
    print(f'[g27b] GMM fitted: {len(mu)} components, '
          f'{time.time()-t0:.0f}s', flush=True)

    # ---- re-encode every layer (release old packed streams as we go) ----
    names = [n for n, _ in targets]
    del targets                      # drop strong refs to old layers!
    gc.collect()
    t0 = time.time()
    for i, name in enumerate(names):
        key = ckpt_key(name, idx)
        if key is None:
            print(f'[g27b] SKIP {name}', flush=True)
            continue
        with safe_open(rf'{MODEL}\{idx[key]}', 'pt') as sf:
            w = sf.get_tensor(key).to(torch.bfloat16)
        packed = gmm_pack(w, mu)
        del w
        _set_parent_child(m, name,
                          UdcqLinear(packed, bias=None, cache='stream'))
        if (i + 1) % 25 == 0:
            gc.collect()
            torch.cuda.empty_cache()
            print(f'[g27b] {i+1}/{len(names)} '
                  f'({time.time()-t0:.0f}s, gpu '
                  f'{torch.cuda.memory_allocated()/1e9:.1f}GB)', flush=True)
    print(f'[g27b] re-encoded {len(names)} linears in '
          f'{time.time()-t0:.0f}s', flush=True)
    gc.collect(); torch.cuda.empty_cache()

    # ---- generate (eager per-token, CUDA fused GEMV) ----
    tm = m.model.language_model if hasattr(m.model, 'language_model') \
        else m.model
    from transformers.cache_utils import StaticCache
    MAX = 128
    cache = StaticCache(config=m.config, max_cache_len=MAX)
    H = tm.config.hidden_size
    emb_cpu = None
    for mod in m.modules():
        if type(mod).__name__ == '_CpuEmbed':
            emb_cpu = mod.weight_cpu
            break
    dev = 'cuda'

    def hard_reset():
        for lay in cache.layers:
            cum = getattr(lay, 'cumulative_length', None)
            if cum is not None:
                cum.zero_()
            hps = getattr(lay, 'has_previous_state', None)
            if hps is not None:
                if isinstance(hps, dict):
                    for k in hps:
                        hps[k] = False
                else:
                    lay.has_previous_state = [False] * len(hps)
            for cs in (getattr(lay, 'conv_states', None) or {}).values():
                if isinstance(cs, torch.Tensor):
                    cs.zero_()
            for rs in (getattr(lay, 'recurrent_states', None) or {}).values():
                if isinstance(rs, torch.Tensor):
                    rs.zero_()

    pos_all = torch.arange(MAX, device=dev).unsqueeze(0)
    with torch.no_grad():
        cos_all, sin_all = tm.rotary_emb(
            torch.zeros(1, MAX, H, dtype=torch.bfloat16, device=dev),
            pos_all)
    if cos_all.dim() == 4:
        cos_all = cos_all[:, :, 0]
    if cos_all.dim() == 3:
        cos_all, sin_all = cos_all[0], sin_all[0]

    emb1 = torch.zeros(1, 1, H, dtype=torch.bfloat16, device=dev)
    cos1 = torch.zeros(1, 1, cos_all.shape[-1], dtype=torch.bfloat16,
                       device=dev)
    sin1 = torch.zeros_like(cos1)
    pos1 = torch.zeros(1, dtype=torch.long, device=dev)

    def fwd():
        h = emb1
        for layer in tm.layers:
            h = layer(h, position_embeddings=(cos1, sin1),
                      attention_mask=None, position_ids=pos1.view(1, 1),
                      past_key_values=cache)
            if isinstance(h, tuple):
                h = h[0]
        return m.lm_head(tm.norm(h))

    def step(tid, t):
        emb1.copy_(emb_cpu[tid].view(1, 1, H).to(dev, torch.bfloat16))
        cos1.copy_(cos_all[t].view(1, 1, -1))
        sin1.copy_(sin_all[t].view(1, 1, -1))
        pos1.fill_(t)
        return fwd()

    ids = tok('The theory of relativity states that',
              return_tensors='pt')['input_ids'][0].tolist()

    # ---- CUDA-graph capture (g1 single-token decode) ----
    # seed the cache first (GDN branch selection), then capture
    for i in range(12):
        step(5 + i, i)
    hard_reset()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            fwd()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        logits_s = fwd()
    hard_reset()
    print('[g27b] g1 graph captured', flush=True)

    # prefill (eager) + graph decode
    logits = None
    for i, tid in enumerate(ids):
        logits = step(tid, i)
    nxt = int(logits[:, -1].argmax(-1).item())
    out = [nxt]
    t = len(ids)
    torch.cuda.synchronize()
    t0 = time.time()
    N = 30
    for _ in range(N - 1):
        emb1.copy_(emb_cpu[nxt].view(1, 1, H).to(dev, torch.bfloat16))
        cos1.copy_(cos_all[t].view(1, 1, -1))
        sin1.copy_(sin_all[t].view(1, 1, -1))
        pos1.fill_(t)
        g.replay()
        nxt = int(logits_s[:, -1].argmax(-1).item())
        out.append(nxt)
        t += 1
    torch.cuda.synchronize()
    dt = (time.time() - t0) / (N - 1)
    print(f'[g27b] GMM-27B (g1 graph): {1/dt:.2f} tok/s ({dt*1000:.0f}ms/tok) '
          f'| gpu {torch.cuda.max_memory_allocated()/1e9:.2f}GB', flush=True)
    print(f'[g27b] -> {tok.decode(out)[:110]!r}', flush=True)


if __name__ == '__main__':
    main()
