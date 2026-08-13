"""Full-pipeline benchmark: search + bf16 baseline + INT8-X (cached/streaming)
+ generation on MiniCPM5-1B.

Run:  python -m benchmarks.bench_minicpm5
"""
from __future__ import annotations
import gc
import sys
import time
sys.setrecursionlimit(10000)
import pandas  # MUST before transformers (stack overflow fix on this env)
import torch

from ixrun.config import MODEL_PATH, DATASET_CACHE, DEFAULT_LEVELS
from ixrun.search import search_optimal_levels
from ixrun.linear import deploy_model
from ixrun.engine import Int8XEngine, ResourceScheduler
from ixrun.eval_utils import bench_forward, eval_ppl, load_wikitext, format_bench


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print("=" * 72)
    print("IXRUN — INT8-X inference engine — MiniCPM5-1B full pipeline")
    print("=" * 72)

    tok = AutoTokenizer.from_pretrained(MODEL_PATH)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    texts = load_wikitext(cache_dir=DATASET_CACHE)
    print(f"[data] {len(texts)} wikitext samples loaded", flush=True)

    # ── 1. Search optimal levels ──────────────────────────────────────────
    print("\n[1] Searching optimal level scheme ...", flush=True)
    m_tmp = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True
    )
    results = search_optimal_levels(m_tmp, topk=6)
    del m_tmp; gc.collect()
    for r in results:
        marker = "  <-- default" if tuple(r["level_bits"]) == tuple(DEFAULT_LEVELS) else ""
        print(f"    {str(r['level_bits']):<14} bpw={r['bpw']:.2f}  "
              f"comp={r['compression']:.2f}x{marker}", flush=True)

    rows = []

    # ── 2. bf16 baseline ──────────────────────────────────────────────────
    print("\n[2] bf16 baseline ...", flush=True)
    gc.collect(); torch.cuda.reset_peak_memory_stats()
    m = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.bfloat16).cuda()
    ms_b, mem_b = bench_forward(m, tok)
    ppl_b = eval_ppl(m, tok, texts)
    wt_b = sum(p.numel() * 2 for p in m.parameters() if p.dtype == torch.bfloat16) / 1e6
    print(f"    fwd={ms_b:.0f}ms  gpu={mem_b:.1f}GB  ppl={ppl_b:.2f}  wt={wt_b:.0f}MB", flush=True)
    rows.append(("bf16", f"{ms_b:.0f}ms", f"{mem_b:.1f}GB", f"{ppl_b:.2f}", f"{wt_b:.0f}MB", "1.0x"))
    del m; gc.collect(); torch.cuda.empty_cache()

    # ── 3. INT8-X cached ──────────────────────────────────────────────────
    print("\n[3] INT8-X cached deploy ...", flush=True)
    gc.collect(); torch.cuda.reset_peak_memory_stats()
    m = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.bfloat16).cuda()
    est = ResourceScheduler.estimate(m, DEFAULT_LEVELS)
    stats = deploy_model(m, level_bits=DEFAULT_LEVELS, cache="full")
    m.eval()
    ms_c, mem_c = bench_forward(m, tok, warmup=3, n_runs=10)
    ppl_c = eval_ppl(m, tok, texts)
    pk_mb = stats["total_bytes"] / 1e6
    print(f"    fwd={ms_c:.0f}ms  gpu={mem_c:.1f}GB  ppl={ppl_c:.2f}  "
          f"store={pk_mb:.0f}MB ({wt_b/pk_mb:.2f}x)", flush=True)
    rows.append(("INT8-X cached", f"{ms_c:.0f}ms", f"{mem_c:.1f}GB",
                 f"{ppl_c:.2f}", f"{pk_mb:.0f}MB", f"{wt_b/pk_mb:.2f}x"))
    del m; gc.collect(); torch.cuda.empty_cache()

    # ── 4. INT8-X streaming (GPU-resident packed, no DMA) ─────────────────
    print("\n[4] INT8-X streaming deploy (GPU-packed + shared buf) ...", flush=True)
    gc.collect(); torch.cuda.reset_peak_memory_stats()
    m = AutoModelForCausalLM.from_pretrained(MODEL_PATH, dtype=torch.bfloat16).cuda()
    s_stats = Int8XEngine._deploy_streaming(m, DEFAULT_LEVELS, verbose=True)
    m.eval()
    ms_s, mem_s = bench_forward(m, tok, warmup=3, n_runs=10)
    print(f"    fwd={ms_s:.0f}ms  gpu={mem_s:.1f}GB  "
          f"(shared decode buf={s_stats['shared_gpu_MB']:.1f}MB)", flush=True)
    rows.append(("INT8-X stream", f"{ms_s:.0f}ms", f"{mem_s:.1f}GB",
                 "—", f"{pk_mb:.0f}MB*", f"{wt_b/pk_mb:.2f}x"))
    del m; gc.collect(); torch.cuda.empty_cache()

    # ── 4b. INT8-X graph (CUDA-Graph decode fusion) ───────────────────────
    print("\n[4b] INT8-X graph deploy (CUDA-Graph decode fusion) ...", flush=True)
    gc.collect(); torch.cuda.reset_peak_memory_stats()
    m = AutoModelForCausalLM.from_pretrained(MODEL_PATH, dtype=torch.bfloat16).cuda()
    g_stats, g_replay = Int8XEngine._deploy_graph(m, DEFAULT_LEVELS, verbose=True)
    m.eval()
    @torch.no_grad()
    def fwd_graph(ids):
        g_replay()
        torch.cuda.synchronize()
        return m(ids)
    ids_b = tok("Hello " * 20, return_tensors="pt")["input_ids"].cuda()
    for _ in range(3): fwd_graph(ids_b)
    torch.cuda.synchronize(); t0 = time.time()
    for _ in range(10): fwd_graph(ids_b)
    torch.cuda.synchronize()
    ms_g = (time.time() - t0) / 10 * 1000
    mem_g = torch.cuda.max_memory_allocated() / 1e9
    print(f"    fwd={ms_g:.0f}ms  gpu={mem_g:.1f}GB  (graph replay decode)", flush=True)
    rows.append(("INT8-X graph", f"{ms_g:.0f}ms", f"{mem_g:.1f}GB",
                 "—", f"{pk_mb:.0f}MB*", f"{wt_b/pk_mb:.2f}x"))
    del m; gc.collect(); torch.cuda.empty_cache()

    # ── 5. Summary ────────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print(format_bench(rows, ["Mode", "Fwd", "GPU", "ppl", "Store", "comp"]))
    print(f"\n  ppl delta (cached vs bf16): {ppl_c - ppl_b:+.2f}")
    print(f"  * streaming/graph: packed GPU-resident, shared/per-layer decode buf only")
    print("=" * 72)

    # ── 6. Generation smoke test ──────────────────────────────────────────
    print("\n[6] Generation smoke test (cached) ...", flush=True)
    eng = Int8XEngine.from_pretrained(MODEL_PATH, mode="cached", verbose=False)
    prompt = "The theory of relativity states that"
    print(f"    prompt: {prompt!r}", flush=True)
    t0 = time.time()
    out = eng.generate(prompt, max_new_tokens=40, do_sample=False)
    print(f"    output: {out.strip()!r}\n    ({time.time()-t0:.1f}s)", flush=True)
    del eng; gc.collect(); torch.cuda.empty_cache()
    print("\ndone.", flush=True)


if __name__ == "__main__":
    main()
