"""PEAK-Q inference benchmark on MiniCPM5-1B: bf16 vs PEAK-Q cached/streaming.

Measures: forward ms (prefill), per-token decode latency + tok/s (generation
hot loop), peak GPU memory, ppl.

Run:  python -m benchmarks.bench_peakq
"""
from __future__ import annotations
import gc
import sys
import time
sys.setrecursionlimit(10000)
import pandas  # MUST before transformers (stack overflow fix on this env)
import torch

from ixrun.config import MODEL_PATH, DATASET_CACHE
from ixrun.peakq import peakq_quantize, PeakQLinear, PEAKQ_TIERS, PEAKQ_GROUP
from ixrun.linear import iter_quantizable_linears, _set_parent_child
from ixrun.eval_utils import bench_forward, eval_ppl, load_wikitext, format_bench


@torch.no_grad()
def bench_generate(model, tok, prompt, max_new_tokens=64, warmup=False):
    """KV-cached greedy generation (single-token steps — hits the fused GEMV
    path in streaming mode). Returns (tok/s, ms/token)."""
    ids = tok(prompt, return_tensors="pt")["input_ids"].cuda()
    model.eval()
    out = model(ids, use_cache=True)
    past = out.past_key_values
    nxt = out.logits[:, -1].argmax(-1, keepdim=True)
    if warmup:
        model(nxt, past_key_values=past, use_cache=True)
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(max_new_tokens):
        out = model(nxt, past_key_values=past, use_cache=True)
        past = out.past_key_values
        nxt = out.logits[:, -1].argmax(-1, keepdim=True)
    torch.cuda.synchronize()
    dt = time.time() - t0
    return max_new_tokens / dt, dt / max_new_tokens * 1000


def deploy_peakq_streaming(model, tiers=PEAKQ_TIERS, group=PEAKQ_GROUP):
    """GPU-resident packed + shared decode buffer + per-forward Triton decode."""
    targets = list(iter_quantizable_linears(model))
    max_N = 0
    total_bytes = 0
    for name, mod in targets:
        p = peakq_quantize(mod.weight.data, group=group, tiers=tiers)
        bias = mod.bias.data if mod.bias is not None else None
        ql = PeakQLinear(p, bias=bias, cache="none")
        total_bytes += p["total_bytes"]
        max_N = max(max_N, p["N"])
        _set_parent_child(model, name, ql)
    shared_w = torch.empty(max_N, dtype=torch.bfloat16, device="cuda")
    for _, mod in model.named_modules():
        if isinstance(mod, PeakQLinear):
            mod._set_shared_buf(shared_w)
    gc.collect(); torch.cuda.empty_cache()
    return total_bytes, shared_w.numel() * 2 / 1e6


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print("=" * 76)
    print("PEAK-Q inference benchmark — MiniCPM5-1B (RTX 4090)")
    print("=" * 76)

    tok = AutoTokenizer.from_pretrained(MODEL_PATH)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    texts = load_wikitext(cache_dir=DATASET_CACHE)
    prompt = "The theory of relativity states that"
    rows = []

    # ── 1. bf16 baseline ──────────────────────────────────────────────────
    print("\n[1] bf16 baseline ...", flush=True)
    gc.collect(); torch.cuda.reset_peak_memory_stats()
    m = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.bfloat16).cuda().eval()
    ms_b, mem_b = bench_forward(m, tok, warmup=3, n_runs=10)
    tps_b, mstok_b = bench_generate(m, tok, prompt, warmup=True)
    ppl_b = eval_ppl(m, tok, texts)
    del m; gc.collect(); torch.cuda.empty_cache()
    print(f"    fwd={ms_b:.0f}ms  decode={mstok_b:.1f}ms/tok ({tps_b:.1f} tok/s)  "
          f"gpu={mem_b:.1f}GB  ppl={ppl_b:.2f}", flush=True)
    rows.append(("bf16", f"{ms_b:.0f}ms", f"{mstok_b:.1f}ms", f"{tps_b:.1f}",
                 f"{mem_b:.1f}GB", f"{ppl_b:.2f}", "1359MB", "1.0x"))

    # ── 2. PEAK-Q cached (decode once -> F.linear) ────────────────────────
    print("\n[2] PEAK-Q cached deploy ...", flush=True)
    gc.collect(); torch.cuda.reset_peak_memory_stats()
    m = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.bfloat16).cuda().eval()
    from ixrun.peakq import deploy_peakq
    stats = deploy_peakq(m, cache="full")
    m.eval()
    ms_c, mem_c = bench_forward(m, tok, warmup=3, n_runs=10)
    tps_c, mstok_c = bench_generate(m, tok, prompt, warmup=True)
    ppl_c = eval_ppl(m, tok, texts)
    pk_mb = stats["total_bytes"] / 1e6
    del m; gc.collect(); torch.cuda.empty_cache()
    print(f"    fwd={ms_c:.0f}ms  decode={mstok_c:.1f}ms/tok ({tps_c:.1f} tok/s)  "
          f"gpu={mem_c:.1f}GB  ppl={ppl_c:.2f}  store={pk_mb:.0f}MB", flush=True)
    rows.append(("PEAK-Q cached", f"{ms_c:.0f}ms", f"{mstok_c:.1f}ms", f"{tps_c:.1f}",
                 f"{mem_c:.1f}GB", f"{ppl_c:.2f}", f"{pk_mb:.0f}MB",
                 f"{stats['compression_vs_bf16']:.2f}x"))

    # ── 3. PEAK-Q streaming (GPU-packed, shared buf, per-fwd decode) ──────
    print("\n[3] PEAK-Q streaming deploy ...", flush=True)
    gc.collect(); torch.cuda.reset_peak_memory_stats()
    m = AutoModelForCausalLM.from_pretrained(MODEL_PATH, dtype=torch.bfloat16).cuda().eval()
    total_bytes, shared_MB = deploy_peakq_streaming(m)
    m.eval()
    ms_s, mem_s = bench_forward(m, tok, warmup=3, n_runs=10)
    tps_s, mstok_s = bench_generate(m, tok, prompt, warmup=True)
    pk_mb_s = total_bytes / 1e6
    del m; gc.collect(); torch.cuda.empty_cache()
    print(f"    fwd={ms_s:.0f}ms  decode={mstok_s:.1f}ms/tok ({tps_s:.1f} tok/s)  "
          f"gpu={mem_s:.1f}GB  packed={pk_mb_s:.0f}MB  shared-buf={shared_MB:.0f}MB", flush=True)
    rows.append(("PEAK-Q stream", f"{ms_s:.0f}ms", f"{mstok_s:.1f}ms", f"{tps_s:.1f}",
                 f"{mem_s:.1f}GB", "—", f"{pk_mb_s:.0f}MB*",
                 f"{1359/pk_mb_s:.2f}x"))

    print("\n" + "=" * 76)
    print(format_bench(rows, ["Mode", "Fwd", "ms/tok", "tok/s", "GPU", "ppl", "Store", "comp"]))
    print(f"\n  ppl delta (cached vs bf16): {ppl_c - ppl_b:+.2f}")
    print("  * streaming: packed GPU-resident, single shared decode buffer")
    print("=" * 76)


if __name__ == "__main__":
    main()
