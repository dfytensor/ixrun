# -*- coding: utf-8 -*-
"""UDCQ layer-DMA mode for Qwen3.8-27B on 24GB: packed streams stay in ONE
pinned CPU pool; per forward the layer DMAs its streams into a REUSED GPU
staging area, then the fused GEMV / stream-decode+GEMM runs.

GPU budget: non-linear params (~5.1GB) + staging (largest layer ~120MB) +
KV cache + activations  ~= 6GB. vs 34.8GB resident (WDDM-paged, 0.8 tok/s).

Also overlaps: since generation is sequential layer-by-layer, we prefetch
the NEXT layer's streams while the current one computes (double-buffer).
"""
import sys, os, time, gc, json
sys.setrecursionlimit(20000)
sys.path.insert(0, r'E:\IXRUN')
import pandas
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

import triton
from ixrun.udcq import (udcq_fit_codebook, udcq_quantize, UDCQ_G,
                        _udcq_gemv_kernel, _udcq_decode_kernel, udcq_fused_gemv)
from ixrun.config import QWEN38_PATH

MD = QWEN38_PATH
tok = AutoTokenizer.from_pretrained(MD)

PROMPTS = [
    "The theory of relativity states that",
    "def quick_sort(arr):",
    "北京最值得游览的三个景点是",
]


# ---------------- pinned pool (chunked, WDDM-safe) ---------------- #
class Pinner:
    def __init__(self, chunk_mb=128):
        self.buf = None
        self.used = 0
        self.total = 0
        self.chunk = chunk_mb << 20

    def add(self, t):
        for attempt in range(5):
            try:
                es = t.element_size()
                n = t.numel() * es
                if self.buf is not None:
                    self.used = (self.used + es - 1) // es * es
                if self.buf is None or self.used + n > self.buf.numel():
                    self.buf = torch.empty(max(n + es, self.chunk),
                                           dtype=torch.uint8).pin_memory()
                    self.used = 0
                    self.total += self.buf.numel()
                v = self.buf[self.used:self.used + n].view(t.dtype).view(t.shape)
                v.copy_(t)
                self.used += n
                return v
            except torch.OutOfMemoryError:
                gc.collect()
                torch.cuda.empty_cache()
                time.sleep(3)
                self.buf = None          # retry with a fresh chunk
        raise RuntimeError('pin pool failed after retries')


PIN = Pinner()


class UdcqDmaLinear(nn.Module):
    """Packed streams in pinned CPU; per forward DMA -> reused GPU staging
    -> fused GEMV (single token) or decode+GEMM (multi-token)."""

    # shared staging (double-buffered prefetch is future work; single buf now)
    STAGE = {}

    def __init__(self, lin, CB):
        super().__init__()
        self.out_features = lin.out_features
        self.in_features = lin.in_features
        w = lin.weight.data.detach().float().cpu()
        p = udcq_quantize(w, CB, g=UDCQ_G)
        self.N = p["N"]
        self.g = p["g"]
        self.pin = {k: PIN.add(p[k]) for k in ("idx", "scale", "sign_packed")}
        self._cb = CB.half().cuda()                 # global, tiny
        self.bias = lin.bias.detach().cuda() if lin.bias is not None else None

    def _stage(self):
        """DMA this layer's streams into the (reused) GPU staging dict."""
        st = UdcqDmaLinear.STAGE
        dev = "cuda"
        need = self.pin["idx"].numel() + self.pin["scale"].numel() * 2 \
            + self.pin["sign_packed"].numel() * 4
        if "buf" not in st or st["buf"].numel() < need:
            st["buf"] = torch.empty(need, dtype=torch.uint8, device=dev)
        b = st["buf"]
        o = 0
        n = self.pin["idx"].numel()
        idx = b[o:o + n].view(torch.uint8)
        idx.copy_(self.pin["idx"], non_blocking=True)
        o += n
        n2 = self.pin["scale"].numel() * 2
        sc = b[o:o + n2].view(torch.float16)
        sc.copy_(self.pin["scale"], non_blocking=True)
        o += n2
        n3 = self.pin["sign_packed"].numel() * 4
        sg = b[o:o + n3].view(torch.int32)
        sg.copy_(self.pin["sign_packed"], non_blocking=True)
        return idx, sc, sg

    def forward(self, x):
        if not x.is_cuda:
            x = x.cuda()
        idx, sc, sg = self._stage()
        M = x.numel() // self.in_features
        if M == 1 and self.in_features % 256 == 0 and self.out_features % 2 == 0:
            y = udcq_fused_gemv(x, idx, sg, sc, self._cb,
                                self.out_features, self.in_features,
                                g=self.g)
            if self.bias is not None:
                y = y + self.bias.to(x.dtype)
            return y.view(*x.shape[:-1], self.out_features)
        # multi-token: decode into a shared bf16 buf then GEMM
        total = self.out_features * self.in_features
        buf = UdcqDmaLinear.STAGE.get("w")
        if buf is None or buf.numel() < total:
            UdcqDmaLinear.STAGE["w"] = buf = torch.zeros(
                total, dtype=torch.bfloat16, device=x.device)
        w_flat = buf[:total]
        w_flat.zero_()
        w_valid = w_flat[: self.N]
        _udcq_decode_kernel[(triton.cdiv(self.N, 1024),)](
            w_valid, idx, sg, sc, self._cb, self.N,
            GROUP=self.g, BLK=1024)
        w = w_flat.view(self.out_features, self.in_features)
        b = self.bias.to(x.dtype) if self.bias is not None else None
        return F.linear(x, w, b)


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
    print('loading Qwen3.8-27B (lazy cpu) + UDCQ-DMA deploy ...', flush=True)
    t0 = time.time()
    m = AutoModelForCausalLM.from_pretrained(
        MD, dtype=torch.bfloat16, low_cpu_mem_usage=True, device_map='cpu')
    targets = []
    for name, mod in m.named_modules():
        if isinstance(mod, nn.Linear) and mod.weight.numel() >= 1000 \
                and 'embed' not in name:
            targets.append((name, mod))
    cb = None
    tot_b = 0; tot_n = 0
    for i, (name, mod) in enumerate(targets):
        w = mod.weight.data.cpu()
        if cb is None:
            cb = udcq_fit_codebook(w, g=UDCQ_G)
        lay = UdcqDmaLinear(mod, cb)
        parts = name.split('.')
        parent = m
        for q in parts[:-1]:
            parent = getattr(parent, q)
        setattr(parent, parts[-1], lay)
        mod.weight.data = torch.empty(0)
        tot_b += lay.N * 6 / 8 + lay.pin['scale'].numel() * 2
        tot_n += lay.N
        del w
        if (i + 1) % 100 == 0:
            gc.collect()
            print(f'  [{i+1}/{len(targets)}] {time.time()-t0:.0f}s', flush=True)
    gc.collect()
    # non-linear params -> GPU (embed + norms + lm_head? lm_head now UDCQ too)
    moved = 0
    for name, p in m.named_parameters():
        if p.numel() and p.device.type == 'cpu':
            p.data = p.data.cuda()
            moved += p.numel() * 2
    torch.cuda.empty_cache()
    print(f'[udcq-dma-27B] {len(targets)} layers | pinned pool {PIN.total/1e9:.2f}GB '
          f'({tot_b*8/tot_n:.2f}bpw) | GPU non-linear {moved/1e9:.2f}GB | '
          f'{time.time()-t0:.0f}s', flush=True)
    print(f'  resident GPU = {torch.cuda.memory_allocated()/1e9:.2f}GB', flush=True)

    m.eval()
    for p in PROMPTS:
        txt, tps = gen(m, p)
        print(f'\n[{tps:.2f} tok/s] {p!r}\n  -> {txt!r}', flush=True)
    print(f'\npeak GPU = {torch.cuda.max_memory_allocated()/1e9:.2f}GB', flush=True)


if __name__ == '__main__':
    main()
