# -*- coding: utf-8 -*-
"""Qwen3.8-27B UDCQ inference from the on-disk blob (mmap, no pinning).

Per forward: safetensors mmap slice -> GPU staging (reused) -> fused GEMV
(single token) / decode+GEMM (multi-token). GPU budget: non-linear params
(~5.1GB) + staging (~150MB) + KV + activations."""
import sys, os, time, gc, json
sys.setrecursionlimit(20000)
sys.path.insert(0, r'E:\IXRUN')
import pandas
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors import safe_open
from transformers import AutoModelForCausalLM, AutoTokenizer

import triton
from ixrun.udcq import UDCQ_G, _udcq_gemv_kernel, _udcq_decode_kernel, udcq_fused_gemv
from ixrun.config import QWEN38_PATH

HERE = os.path.dirname(os.path.abspath(__file__))
PACK = os.path.join(HERE, 'qwen38_udcq_packed.safetensors')
META = os.path.join(HERE, 'qwen38_udcq_meta.json')
MD = QWEN38_PATH

tok = AutoTokenizer.from_pretrained(MD)

PROMPTS = [
    "The theory of relativity states that",
    "def quick_sort(arr):",
    "北京最值得游览的三个景点是",
]


class UdcqMmapLinear(nn.Module):
    """Streams read from the mmap'd blob per forward (pageable DMA) into a
    REUSED GPU staging buffer; then fused GEMV / decode+GEMM."""

    _SF = None          # shared safe_open handle
    _STAGE = {}

    def __init__(self, name, entry, cb_gpu):
        super().__init__()
        self.name = name
        self.out_features = entry['out_f']
        self.in_features = entry['in_f']
        self.N = entry['N']
        self.g = UDCQ_G
        self._cb = cb_gpu
        self.bias = None
        # hot caching: keep the most-recent stream tensors on GPU if small?
        # No — that's what made 34.8GB. Pure per-forward DMA.

    @classmethod
    def sf(cls):
        if cls._SF is None:
            cls._SF = safe_open(PACK, framework='pt', device='cpu')
        return cls._SF

    def forward(self, x):
        if not x.is_cuda:
            x = x.cuda()
        sf = UdcqMmapLinear.sf()
        idx_cpu = sf.get_tensor(f'{self.name}.idx')
        sc_cpu = sf.get_tensor(f'{self.name}.scale')
        sg_cpu = sf.get_tensor(f'{self.name}.sign_packed')
        assert idx_cpu.numel() == self.N, \
            f'{self.name}: idx {idx_cpu.numel()} != N {self.N} (blob key mismatch?)'
        st = UdcqMmapLinear._STAGE
        need = idx_cpu.numel() + sc_cpu.numel() * 2 + sg_cpu.numel() * 4
        if 'buf' not in st or st['buf'].numel() < need:
            st['buf'] = torch.empty(need, dtype=torch.uint8, device='cuda')
        b = st['buf']
        o = 0
        idx = b[o:o + idx_cpu.numel()].view(torch.uint8)
        if idx.shape != idx_cpu.shape:
            print(f'[bug] {self.name}: staging idx {idx.shape} vs cpu {idx_cpu.shape}',
                  flush=True)
        idx = idx.reshape(-1)[: idx_cpu.numel()]
        idx_cpu = idx_cpu.reshape(-1)
        idx.copy_(idx_cpu, non_blocking=True)
        sc = b[o + idx_cpu.numel(): o + idx_cpu.numel() + sc_cpu.numel() * 2] \
            .view(torch.float16).reshape(-1)[: sc_cpu.numel()]
        sc.copy_(sc_cpu.reshape(-1), non_blocking=True)
        sg = b[o + idx_cpu.numel() + sc_cpu.numel() * 2:
               o + idx_cpu.numel() + sc_cpu.numel() * 2 + sg_cpu.numel() * 4] \
            .view(torch.int32).reshape(-1)[: sg_cpu.numel()]
        sg.copy_(sg_cpu.reshape(-1), non_blocking=True)

        M = x.numel() // self.in_features
        if M == 1 and self.in_features % 256 == 0 and self.out_features % 2 == 0:
            y = udcq_fused_gemv(x, idx, sg, sc, self._cb,
                                self.out_features, self.in_features,
                                g=self.g)
            return y.view(*x.shape[:-1], self.out_features)
        total = self.out_features * self.in_features
        if 'w' not in st or st['w'].numel() < total:
            st['w'] = torch.zeros(total, dtype=torch.bfloat16, device='cuda')
        w_flat = st['w'][:total]
        w_flat.zero_()
        w_valid = w_flat[: self.N]
        _udcq_decode_kernel[(triton.cdiv(self.N, 1024),)](
            w_valid, idx, sg, sc, self._cb, self.N,
            GROUP=self.g, BLK=1024)
        return F.linear(x, w_flat.view(self.out_features, self.in_features))


@torch.no_grad()
def gen(m, prompt, n=40):
    ids = tok(prompt, return_tensors="pt")["input_ids"].cuda()
    out = m(ids, use_cache=True)
    past = out.past_key_values
    nxt = out.logits[:, -1].argmax(-1, keepdim=True)
    toks = [nxt[0, 0].item()]
    torch.cuda.synchronize(); t0 = time.time()
    for _ in range(n - 1):
        out = m(nxt, past_key_values=past, use_cache=True)
        past = out.past_key_values
        nxt = out.logits[:, -1].argmax(-1, keepdim=True)
        toks.append(nxt[0, 0].item())
    torch.cuda.synchronize()
    return tok.decode(toks, skip_special_tokens=True), (n - 1) / (time.time() - t0)


def main():
    meta = json.load(open(META))
    entries = meta['tensors']
    cb = torch.tensor(meta['codebook']).half().cuda()

    print('building model (meta) + swapping linears ...', flush=True)
    t0 = time.time()
    from accelerate import init_empty_weights
    from transformers import AutoConfig
    cfg = AutoConfig.from_pretrained(MD, trust_remote_code=True)
    with init_empty_weights():
        m = AutoModelForCausalLM.from_config(cfg, trust_remote_code=True)
    n = 0
    for name, mod in list(m.named_modules()):
        if isinstance(mod, nn.Linear) and name in entries:
            new = UdcqMmapLinear(name, entries[name], cb)
            parts = name.split('.')
            parent = m
            for q in parts[:-1]:
                parent = getattr(parent, q)
            setattr(parent, parts[-1], new)
            n += 1
    print(f'  swapped {n} linears in {time.time()-t0:.0f}s', flush=True)

    # materialize non-linear params from disk to GPU.
    # checkpoint keys carry model.language_model.* while the constructed
    # model's params are model.* (lm_head matches directly)
    from safetensors import safe_open as so
    idx = json.load(open(os.path.join(MD, 'model.safetensors.index.json')))['weight_map']

    def ckpt_key(name):
        if name in idx:
            return name
        alt = name.replace('model.', 'model.language_model.', 1)
        return alt if alt in idx else None

    moved = 0
    params = dict(m.named_parameters())
    missing = []
    with torch.no_grad():
        for name, p in params.items():
            if not p.numel() or not p.is_meta:
                continue
            key = ckpt_key(name)
            if key is None:
                missing.append(name)
                continue
            with so(os.path.join(MD, idx[key]), 'pt') as sf:
                t = sf.get_tensor(key)
            # replace the parameter object wholesale (set_data fails on
            # meta tensors with typed params like dt_bias)
            parts = name.split('.')
            parent = m
            for q in parts[:-1]:
                parent = getattr(parent, q)
            parent._parameters[parts[-1]] = torch.nn.Parameter(
                t.cuda(), requires_grad=False)
            moved += t.numel() * t.element_size()
            del t
    if missing:
        print(f'  [warn] {len(missing)} params without ckpt key: {missing[:3]}',
              flush=True)
    torch.cuda.empty_cache()
    print(f'  non-linear params moved: {moved/1e9:.2f}GB, '
          f'resident GPU = {torch.cuda.memory_allocated()/1e9:.2f}GB', flush=True)

    m.eval()
    for p in PROMPTS:
        txt, tps = gen(m, p)
        print(f'\n[{tps:.2f} tok/s] {p!r}\n  -> {txt!r}', flush=True)
    print(f'\npeak GPU = {torch.cuda.max_memory_allocated()/1e9:.2f}GB', flush=True)


if __name__ == '__main__':
    main()
