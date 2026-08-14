"""Qwen3.8-27B adaptation benchmark: bit-exact check + streaming deploy + generation.

The 27B bf16 weights (~55GB) don't fit on a 24GB GPU, so:
  - baseline ppl/logits comparison is impossible on this machine;
  - instead we verify layer-level bit-exact equivalence vs plain int8
    (the same proof used for MiniCPM5), then run INT8-X streaming inference
    (packed ~10-12GB GPU) and check generation quality.

Run:  python -m benchmarks.bench_qwen38
"""
from __future__ import annotations
import gc
import math
import sys
import time

sys.setrecursionlimit(10000)
import pandas  # noqa: F401  (stack overflow fix, must precede transformers)
import torch

from ixrun.config import QWEN38_PATH, DEFAULT_LEVELS
from ixrun.quantize import int8x_quantize
from ixrun.triton_kernels import decode_weight_triton
from ixrun.linear import iter_quantizable_linears
from ixrun.engine import Int8XEngine, StreamingLinear

N_EQUIV_LAYERS = 5
N_STASH = 5


def naive_int8_weight(w: torch.Tensor) -> torch.Tensor:
    scale_f32 = w.abs().max().clamp(min=1e-8) / 127.0
    i8 = (w.float() / scale_f32).round().clamp(-127, 127).to(torch.int8)
    return (i8.float() * scale_f32.bfloat16().float()).to(torch.bfloat16).view(w.shape)


def _packed_from_streaming(sl: StreamingLinear) -> dict:
    return {
        "level_bits": tuple(sl.level_bits),
        "out_f": sl.out_features, "in_f": sl.in_features, "N": sl.N,
        "scale": sl._scale.cpu(),
        "bitmaps": [sl._b1.cpu(), sl._b2.cpu()],
        "streams": [sl._l1.cpu(), sl._l2.cpu(), sl._l3.cpu()],
        "counts": list(sl.counts),
    }


@torch.no_grad()
def verify_deployed_equivalence(model, stash):
    """stash: {name: bf16 weight (GPU)} captured before deploy."""
    print(f"\n[1] Layer-level bit-exact check vs plain int8 ({len(stash)} layers) ...", flush=True)
    n = 0
    for name, mod in model.named_modules():
        if not isinstance(mod, StreamingLinear):
            continue
        if name not in stash:
            continue
        w_orig = stash[name]
        w_ix = decode_weight_triton(_packed_from_streaming(mod), device="cuda")
        w_ref = naive_int8_weight(w_orig)
        ok = torch.equal(w_ix.cpu(), w_ref.cpu())
        err = (w_orig.float() - w_ix.float()).pow(2).mean().item()
        sig = w_orig.float().pow(2).mean().item()
        snr = 10 * math.log10(sig / max(err, 1e-12))
        print(f"    {name:<58} bit-exact={ok}  SNR={snr:.2f}dB", flush=True)
        assert ok, f"{name} NOT bit-exact vs plain int8!"
        del w_ix, w_ref
        n += 1
    assert n == len(stash), f"verified {n} != stashed {len(stash)}"
    print(f"    -> {n} layers BIT-EXACT (INT8-X == plain int8)", flush=True)


def main():
    print("=" * 76)
    print("IXRUN — Qwen3.8-27B adaptation (multimodal, hybrid linear/full attention)")
    print("=" * 76)

    # manual deploy flow so we can stash original bf16 weights for verification
    from transformers import AutoTokenizer

    print("[load] model on CPU (bf16, low_cpu_mem) ...", flush=True)
    t0 = time.time()
    model = Int8XEngine._load_any(QWEN38_PATH, torch.bfloat16, low_cpu=True)
    tok = AutoTokenizer.from_pretrained(QWEN38_PATH, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    print(f"    loaded in {time.time()-t0:.0f}s", flush=True)

    # stash a few original weights (GPU) for bit-exact verification
    stash = {}
    for name, mod in iter_quantizable_linears(model):
        if len(stash) >= N_STASH:
            break
        stash[name] = mod.weight.data.cuda().clone()
    print(f"    stashed {len(stash)} original weights for verification", flush=True)

    t0 = time.time()
    stats = Int8XEngine._deploy_streaming(model, DEFAULT_LEVELS, verbose=True)
    t_dep = time.time() - t0
    eng = Int8XEngine(model, tok, stats)
    eng._finalize_device()
    print(f"\n[deploy] {stats['n_layers']} layers | packed={stats['total_bytes']/1e9:.2f}GB | "
          f"decode buf={stats['shared_gpu_MB']:.1f}MB | deploy={t_dep:.0f}s", flush=True)
    print(f"[vram] allocated={torch.cuda.memory_allocated()/1e9:.2f}GB "
          f"peak={torch.cuda.max_memory_allocated()/1e9:.2f}GB", flush=True)

    verify_deployed_equivalence(eng.model, stash)

    # generation smoke test (chat template)
    print("\n[2] Generation smoke test ...", flush=True)
    tok = eng.tokenizer
    messages = [{"role": "user", "content": "用一句话介绍相对论的核心思想"}]
    prompt = tok.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    t0 = time.time()
    out = eng.generate(prompt, max_new_tokens=48, do_sample=False)
    dt = time.time() - t0
    print(f"    prompt : {messages[0]['content']}", flush=True)
    print(f"    output : {out.strip()[:300]}", flush=True)
    print(f"    ({dt:.1f}s for 48 tokens = {dt/48*1000:.0f} ms/tok)", flush=True)

    # second prompt, English
    messages = [{"role": "user", "content": "The capital of France is"}]
    prompt = tok.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    out = eng.generate(prompt, max_new_tokens=32, do_sample=False)
    print(f"    en out : {out.strip()[:200]}", flush=True)

    print("\n" + "=" * 76)
    print(f"Qwen3.8-27B INT8-X streaming: packed={stats['total_bytes']/1e9:.2f}GB GPU "
          f"+ shared buf {stats['shared_gpu_MB']:.0f}MB — fits 24GB card", flush=True)
    print("done.", flush=True)


if __name__ == "__main__":
    main()
