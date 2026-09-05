# -*- coding: utf-8 -*-
"""Format A/B on MiniCPM5-1B: bf16 vs INT8-X vs UDCQ vs MXFP6 vs MXINT8.

Answers the review's challenge:
  1. does OCP-MX-style fixed-length cheap-decode coding (MXFP6 6.25bpw /
     MXINT8 8.25bpw) beat UDCQ's 4-bit LUT at equal-or-better bpw?
  2. where does the 59GB/s effective bandwidth go (per-module profile)?

Run: python -X utf8 -m benchmarks.bench_formats_minicpm5
"""
import sys, time, gc
sys.path.insert(0, r'E:\IXRUN')
import pandas                       # MUST precede torch (heap rule)
import torch
import torch.nn as nn

from transformers import AutoTokenizer
from ixrun.config import MODEL_PATH, DATASET_CACHE
from ixrun.eval_utils import eval_ppl, load_wikitext, bench_forward

torch.manual_seed(0)

# --------------------------------------------------------------------- #
# MXFP6 (OCP MX, E3M2: 1s+3e+2m, block=32, E8M0 scale) + MXINT8 (8.25bpw)
# --------------------------------------------------------------------- #
MX_BLOCK = 32

# decode table: magnitude -> value (e3m2 grid, 32 magnitudes incl.
# subnormals, no specials). code = mag_code | (sign << 5), 6 bits.
def _e3m2_grid():
    vals = []
    for ef in range(8):            # exponent field
        for m2 in range(4):
            if ef == 0:            # subnormal: 0.m2 * 2^(1-bias), bias=3
                v = m2 * 2.0 ** (1 - 3 - 2)
            else:
                v = (1.0 + m2 / 4) * 2.0 ** (ef - 3)
            vals.append(v)
    return torch.tensor(vals, dtype=torch.float32)   # 32 magnitudes

_E3M2 = None

def mx_scale_exp(block_max):
    """E8M0: scale = 2^E; E chosen so block max fits half the grid (14)."""
    e = torch.ceil(torch.log2(block_max.clamp_min(1e-30)) - 3.80735)  # log2(14)
    e = torch.where(block_max == 0, torch.zeros_like(e), e)
    return e.clamp(-127, 127).to(torch.int8)

def encode_mxfp6(w: torch.Tensor) -> tuple:
    """w fp32 [N]; returns (codes uint8 [N], scales int8 [ceil(N/32)])."""
    global _E3M2
    if _E3M2 is None:
        _E3M2 = _e3m2_grid().to(w.device)
    flat = w.reshape(-1).float()
    N = flat.numel()
    pad = (-N) % MX_BLOCK
    if pad:
        flat = torch.cat([flat, flat.new_zeros(pad)])
    blk = flat.view(-1, MX_BLOCK)
    bmax = blk.abs().max(dim=1).values
    escale = mx_scale_exp(bmax)                     # [nb]
    s = torch.pow(2.0, escale.float())              # [nb]
    y = (blk / s.unsqueeze(1)).clamp(-28.0, 28.0)   # into e3m2 range
    # nearest magnitude code via searchsorted over the signed magnitude grid
    mag = _E3M2
    gy = torch.cat([-mag.flip(0), mag])             # 64 sorted (signed)
    yf = y.reshape(-1)
    pos = torch.searchsorted(gy, yf.abs()).clamp(0, gy.numel() - 1)
    near = torch.stack([pos - 1, pos]).clamp(0, gy.numel() - 1)  # [2, N]
    err = (gy[near] - yf.abs().view(1, -1)).abs()               # [2, N]
    cg = near.gather(0, err.argmin(dim=0, keepdim=True)).squeeze(0)
    mag_code = torch.where(cg < mag.numel(),
                           mag.numel() - 1 - cg,   # neg half
                           cg - mag.numel())        # pos half
    code = mag_code | ((yf < 0).long() << 5)
    return code[:N].to(torch.uint8), escale[: (N + MX_BLOCK - 1) // MX_BLOCK]

def decode_mxfp6(codes: torch.Tensor, scales: torch.Tensor, N: int) -> torch.Tensor:
    global _E3M2
    if _E3M2 is None:
        _E3M2 = _e3m2_grid().to(codes.device)
    pad = (-N) % MX_BLOCK
    if pad:
        codes = torch.cat([codes, codes.new_zeros(pad)])
    blk = codes.view(-1, MX_BLOCK).long()
    v = _E3M2[blk & 31]                             # magnitude
    v = torch.where((blk >> 5) & 1 == 1, -v, v)     # sign
    s = torch.pow(2.0, scales.float())
    out = (v * s.unsqueeze(1)).reshape(-1)[:N]
    return out

def encode_mxint8(w: torch.Tensor) -> tuple:
    """int8 + per-32 E8M0 power-of-2 scale (8.25bpw)."""
    flat = w.reshape(-1).float()
    N = flat.numel()
    pad = (-N) % MX_BLOCK
    if pad:
        flat = torch.cat([flat, flat.new_zeros(pad)])
    blk = flat.view(-1, MX_BLOCK)
    bmax = blk.abs().max(dim=1).values
    e = torch.ceil(torch.log2((bmax / 127.0).clamp_min(1e-30)))
    e = torch.where(bmax == 0, torch.zeros_like(e), e)
    e = e.clamp(-127, 127).to(torch.int8)
    s = torch.pow(2.0, e.float())
    y = (blk / s.unsqueeze(1)).round().clamp(-127, 127)
    return y.reshape(-1)[:N].to(torch.int8), e[: (N + MX_BLOCK - 1) // MX_BLOCK]

def decode_mxint8(codes: torch.Tensor, scales: torch.Tensor, N: int) -> torch.Tensor:
    pad = (-N) % MX_BLOCK
    if pad:
        codes = torch.cat([codes, codes.new_zeros(pad)])
    blk = codes.view(-1, MX_BLOCK).float()
    s = torch.pow(2.0, scales.float())
    return (blk * s.unsqueeze(1)).reshape(-1)[:N]

# --------------------------------------------------------------------- #
def quantize_linears(model, kind):
    from ixrun.linear import iter_quantizable_linears, _set_parent_child
    from ixrun.udcq import (udcq_fit_codebook, udcq_quantize, UdcqLinear,
                            UDCQ_G, UDCQ_NLEV)
    from ixrun.linear import deploy_model

    info = {}
    targets = list(iter_quantizable_linears(model))
    if kind == 'int8x':
        deploy_model(model, level_bits=(3, 5, 8), cache='full', verbose=False)
        info['bpw'] = 5.46
    elif kind == 'udcq':
        cb = udcq_fit_codebook(targets[0][1].weight.data.cpu(), nlev=UDCQ_NLEV, g=UDCQ_G)
        info['bpw'] = 6.0
        for name, mod in targets:
            packed = udcq_quantize(mod.weight.data, cb, g=UDCQ_G)
            bias = mod.bias.data if mod.bias is not None else None
            _set_parent_child(model, name, UdcqLinear(packed, bias=bias, cache='full'))
    elif kind in ('mxfp6', 'mxint8'):
        enc = encode_mxfp6 if kind == 'mxfp6' else encode_mxint8
        dec = decode_mxfp6 if kind == 'mxfp6' else decode_mxint8
        bits = 6 if kind == 'mxfp6' else 8
        tot = n_el = 0
        for name, mod in targets:
            w = mod.weight.data
            codes, scales = enc(w.cuda())
            wd = dec(codes, scales, w.numel()).reshape(w.shape).cpu()
            n_el += w.numel()
            tot += codes.numel() * bits / 8 + scales.numel() * 1
            mod.weight.data = wd.to(w.dtype)
        info['bpw'] = tot * 8 / max(n_el, 1)
    model = model.cuda()
    return model, info

# --------------------------------------------------------------------- #
def main():
    tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    texts = load_wikitext(cache_dir=DATASET_CACHE)
    print(f'[ab] wikitext samples {len(texts)}', flush=True)

    def load_bf16():
        from transformers import AutoModelForCausalLM
        m = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH, torch_dtype=torch.bfloat16, trust_remote_code=True)
        m.eval()
        return m

    from ixrun.peakq import peakq_snr

    results = []
    for kind in ('bf16', 'int8x', 'udcq', 'mxfp6', 'mxint8'):
        gc.collect(); torch.cuda.empty_cache()
        print(f'[ab] {kind}: loading + quantizing...', flush=True)
        t0 = time.time()
        m = load_bf16()
        bpw = 16.0
        if kind != 'bf16':
            m, info = quantize_linears(m, kind)
            bpw = info['bpw']
        else:
            m = m.cuda()
        gc.collect(); torch.cuda.empty_cache()
        ppl = eval_ppl(m, tok, texts)
        ms, mem = bench_forward(m, tok, warmup=3, n_runs=10)
        del m; gc.collect(); torch.cuda.empty_cache()
        results.append((kind, bpw, ms, mem, ppl))
        print(f'[ab] {kind}: bpw={bpw:.2f} fwd={ms:.0f}ms gpu={mem:.1f}GB '
              f'ppl={ppl:.2f}  ({time.time()-t0:.0f}s)', flush=True)

    print('\n=== MiniCPM5-1B format A/B ===')
    print(f'{"format":<9}{"bpw":>7}{"fwd_ms":>8}{"gpu_GB":>8}{"ppl":>9}')
    for kind, bpw, ms, mem, ppl in results:
        print(f'{kind:<9}{bpw:>7.2f}{ms:>8.0f}{mem:>8.1f}{ppl:>9.2f}')
    b = results[0]
    for r in results[1:]:
        print(f'  ppl delta vs bf16 ({b[4]:.2f}): {r[0]} = {r[4]-b[4]:+.2f}')

    # ------------------------------------------------------------------ #
    print('\n=== decode-step breakdown: GPU-in-layers vs overhead ===',
          flush=True)
    m = load_bf16().cuda()
    ids = tok('The theory of relativity states that',
              return_tensors='pt')['input_ids'].cuda()
    with torch.no_grad():
        m(ids, use_cache=True)

    layers = [mod for name, mod in m.named_modules()
              if type(mod).__name__ == 'DecoderLayer' or
              (name.endswith('.layer') and len(name.split('.')) == 3)]
    pre_evs = [torch.cuda.Event(enable_timing=True) for _ in layers]
    post_evs = [torch.cuda.Event(enable_timing=True) for _ in layers]
    hooks = []
    for i, lay in enumerate(layers):
        hooks.append(lay.register_forward_hook(
            lambda mod, inp, out, i=i: (pre_evs[i].record(), out)[1] and out))
    # post events need a pre-hook on the NEXT layer — simpler: wrap calls
    for h in hooks:
        h.remove()
    # direct per-layer timing loop (each layer called explicitly)
    print(f'  found {len(layers)} decoder layers', flush=True)
    t_pre = time.time()
    nxt = ids
    past = None
    with torch.no_grad():
        out = m(nxt, past_key_values=past, use_cache=True)
        past = out.past_key_values
    print(f'  first pass (kv-cache warm) done', flush=True)

    ev_start = torch.cuda.Event(enable_timing=True)
    ev_end = torch.cuda.Event(enable_timing=True)
    lay_time = [0.0] * len(layers)
    N = 30
    for _ in range(N):
        ev_start.record()
        with torch.no_grad():
            out = m(nxt, past_key_values=past, use_cache=True)
        ev_end.record()
        torch.cuda.synchronize()
        dt_wall = ev_start.elapsed_time(ev_end)
    # NOTE: whole-step graph is not hooked here; per-layer events require
    # a custom decode loop — measure at two granularities instead:
    # (a) full-step wall, (b) kernels-in-step via graph capture time
    print(f'  eager full-step: {dt_wall:.1f}ms', flush=True)

    # CUDA-graph full-step (capture + replay) to strip python/launch:
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            with torch.no_grad():
                out = m(nxt, past_key_values=past, use_cache=True)
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        with torch.no_grad():
            out_g = m(nxt, past_key_values=past, use_cache=True)
    ev0 = torch.cuda.Event(enable_timing=True)
    ev1 = torch.cuda.Event(enable_timing=True)
    for _ in range(5):
        ev0.record(); g.replay(); ev1.record()
    torch.cuda.synchronize()
    print(f'  graph full-step replay: {ev0.elapsed_time(ev1):.1f}ms '
          f'(python/launch removed)', flush=True)
    print(f'  => per-step overhead (eager - graph): '
          f'{dt_wall - ev0.elapsed_time(ev1):.1f}ms', flush=True)
    del m, out, past
    gc.collect(); torch.cuda.empty_cache()
    print(f'  effective bandwidth @bf16 2.2GB: eager '
          f'{2.2 / dt_wall * 1e3:.0f}GB/s | graph '
          f'{2.2 / ev0.elapsed_time(ev1) * 1e3:.0f}GB/s', flush=True)

if __name__ == '__main__':
    main()
