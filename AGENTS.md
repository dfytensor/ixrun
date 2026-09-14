# AGENTS.md — IXRUN project guide

## Environment
- **Python**: `F:\rwkv\.venv\Scripts\python.exe` (3.12, has torch/triton/transformers)
- Run all commands from `E:\IXRUN` (working directory).
- Offline mode required (no internet to HF): prefix with
  `HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1`
- **IMPORT ORDER (hard-won)**: `import torchvision` MUST happen before
  transformers imports a multimodal model module (qwen3_5). Loading the
  torchvision CPU pyd AFTER torch has initialized CUDA + its thread pool
  deadlocks the Windows loader lock (main thread blocked in
  LoadLibraryEx, 0% CPU, permanent). `ixrun/__init__.py` imports
  torchvision early as the fix. Symptom history: any `ixrun.q38_spec`
  import hung at ~712MB RSS forever; the opencode shell tool reported
  "ChildProcess.kill" (its own timeout cleaning up the hung tree, NOT
  an external watchdog).
- **Env-flag gotcha**: never test env flags with bare
  `os.environ.get(X)` — the string "0" is truthy (`UDCQ_CUDA_GEMV=0`
  enabled the CUDA path; udcq.py now checks `not in ("", "0")`).
- Model: MiniCPM5-1B (Llama-arch, 24 layers) at path in `ixrun/config.py:MODEL_PATH`.
- Model (large): Qwen3.8-27B (multimodal qwen3_5, 64 layers hybrid linear/full attn)
  at `ixrun/config.py:QWEN38_PATH` = `E:\models\Qwen3.8-27B`. Needs transformers>=5.8
  (installed: 5.15.0). 27B streaming: CPU lazy-load -> per-layer quantize -> packed
  to GPU, bf16 freed eagerly; 606 layers, packed 16.91GB, runs on 24GB card.

## Prefill (TTFT) — 2026/9/14 session results
- 27B graph engine, 2048-token prompt: 97s -> **10.3s** (9.4x) via
  (a) blocked prefill `Q38_MAX_BLOCK` (q38_graph MAX_BLOCK 8->256) and
  `Q38_PREFILL_BLOCK` (q38_spec; legacy per-token was 24ms/token -> ~49s
  TTFT at 2k), (b) ONE batched SDPA per full-attn layer instead of a
  per-token python loop (32k SDPA launches was the top profile item).
- Correctness gate: prefill last-token top-8 IDENTICAL between S=64 and
  S=256; vs S=8 only rank 3/4 swap (bf16-level kernel-order noise).
- The GDN seq-patch ELSE branch already routes prefill (no state AND
  state+seq_len>_MAX_S) through fla chunk kernel — no change needed there.
- Remaining 10s profile guess: GDN chunk launches at S=256 + 606-linear
  python dispatch per block; bigger blocks (512) may squeeze more.
- VALIDATED spec engine: Q38_PREFILL_BLOCK=64 TTFT 100.6s -> 4.4s @512tok
  (22.8x), generated tokens IDENTICAL to per-token path. Capture needs
  desktop VRAM lean; Q38_MIN_FREE_GB=1.2 override is safe (corruption
  was at 0.4GB; 1.5GB free captured cleanly with coherent output).
- HPQ x per-block-scale final ladder (MiniCPM5-1B ppl): m8k32+scale
  6bpw 60.17 vs UDCQ 58.06 / GMM 57.92 (same bpw -> we still win);
  m8k64+scale 7bpw 56.93 BEATS both (err 0.0245) — scale restores the
  per-element magnitude that block-PQ binding destroys, +1bpw. HPQ
  viable only at 7bpw; kept as research, not deployed.
- hpqs_runtime.py: Triton decode+GEMV for HPQ-x-scale, BIT-EXACT gemv
  (gmax 0.0000 vs fp16-cb ref). Triton gotcha: None-indexing silently
  drops dims - use tl.expand_dims. SEEDED mixed-precision ppl ladder
  (kmeans seed 42): down+o 6.36bpw 57.22 = best value (-0.84 vs UDCQ
  58.06);   unseeded runs jitter +-0.5 - always seed before comparing.
- hpqs kernel PERF verdict (f1d0376): bit-exact but 0.02-0.04x bf16
  (~10-15GB/s) across layouts (codes [2,8,nB] vs [nB,16]) and select
  vs select-free gather forms - bottleneck is gather latency chain +
  serial 4-row walk; parity needs hand-CUDA (UDCQ-class effort).
  Deploy stays UDCQ/GMM. Encoder mapping: d=r*4+c, s=d//2, pos=c%2;
  codes flat [nB, l*8+s]; same-subspace cols share ONE code (pos lives
  only in the cb offset - conflating them breaks bit-exactness).
- VRAM watch: Q38SpecEngine graph capture needs >=2GB free
  (Q38_MIN_FREE_GB guard); a busy desktop (QQ/Quark/Edge ~4GB) can push
  24GB cards under the guard at any ctx — engine itself unchanged.

## Commands
```powershell
# unit tests (fast, no model load)
$env:HF_HUB_OFFLINE='1'; & 'F:\rwkv\.venv\Scripts\python.exe' -m tests.test_core

# group-scale (ixgs) tests — lossless + SNR + kernel equivalence
$env:HF_HUB_OFFLINE='1'; & 'F:\rwkv\.venv\Scripts\python.exe' -m ixgs.test_gs

# full pipeline benchmark on MiniCPM5-1B (loads model 4x, ~3 min)
$env:HF_HUB_OFFLINE='1'; $env:TRANSFORMERS_OFFLINE='1'; & 'F:\rwkv\.venv\Scripts\python.exe' -m benchmarks.bench_minicpm5

# CLI
& 'F:\rwkv\.venv\Scripts\python.exe' -m ixrun.cli search
& 'F:\rwkv\.venv\Scripts\python.exe' -m ixrun.cli generate "Hello" --stream
& 'F:\rwkv\.venv\Scripts\python.exe' -m ixrun.cli bench
& 'F:\rwkv\.venv\Scripts\python.exe' -m ixrun.cli chat --model E:\models\Qwen3.8-27B --cache E:\models\qwen38_packed.pt
# OpenAI-compatible API server (fastapi+uvicorn, installed)
& 'F:\rwkv\.venv\Scripts\python.exe' -m ixrun.cli serve --model E:\models\Qwen3.8-27B --cache E:\models\qwen38_packed.pt --port 8000 --model-id qwen3.8-27b
```

## Architecture
- **quantize.py**: bf16 → per-tensor int8 (scale=max_abs/127) → (3,5,8) nested-bitmap
  packing. Lossless on the int8 representation.
- **triton_kernels.py**: fused decode via tl.cumsum + bitmap + cross-word bit extract
  (optimized: 2 cumsums, nl1/l3 ranks derived algebraically). Scatter fallback if
  scheme ≠ (3,5,8) or no CUDA.
- **fused.py**: fused decode+GEMV for single-token decode steps (generation hot
  loop). Per-row sequential walk, register-carried rank counters seeded from tiny
  per-row prefix arrays (compute_row_prefixes); bf16 weight never materializes.
  Shape heuristic (num_warps, BK): tall (2,512) / wide (2,1024) / square (4,512).
  Requires in_f % 512 == 0 (falls back otherwise). Wide layers (down_proj) use
  split-K: S=2, chunk-boundary prefixes, fp32 atomic accumulate (~233G elem/s).
- **fla_patch.py**: binds fla 0.5.2 Triton kernels (fused_recurrent_gdn /
  chunk_gated_delta_rule) into qwen3_5's delta-rule functions — without it the
  HF hub-kernel fallback silently uses an eager fp32 python loop (~120ms/token).
  Applied automatically in engine._load_any.
- **linear.py**: `Int8XLinear` (cache='full' decodes once; cache='none' re-decodes).
- **engine.py**: `Int8XEngine.from_pretrained(mode='cached'|'streaming'|'graph')`.
  Streaming: packed GPU-resident (463MB) + shared 14MB decode buffer, real-time
  Triton decode per forward (no DMA). Graph: CUDA-Graph captures all 168 decode
  kernels into one replay; GraphLinear does GEMM-only forward.
- **search.py**: exhaustive 2-5 level combo search by bit/w.
- **peakq.py**: PEAK-Q (Peak-Exact Adaptive K-bit) — exponent-group bf16
  re-encoding. Per 16-elem group: emax (8b) + sign stream + tiered payloads by
  delta = emax-expo: T1 (delta<=1, mant7+d1=8b, BIT-EXACT), T2 (delta<=3,
  mant6+d1=7b), T3 (delta<=7 saturated, mant5+d2=7b). Nested B1/B2 bitmaps +
  tl.cumsum rank algebra copied from `_ix_decode_kernel`; T1 stream is raw
  uint8 (kernel loads by rank, NOT by bit index). 10.50 bpw (1.52x), 69%
  elements bit-exact, SNR 54 dB vs INT8-X 20-33 dB on MiniCPM5. No sparse
  fixup kernel (saturation instead) → single-kernel decode, CUDA-Graph safe.
  Delta-field bits = ceil(log2(hi-lo+1)) where lo=prev_dmax+1 — off-by-one
  here silently disables the Triton path (guarded by payload_bits==[8,7,7]).
  Also hosts `_peakq_gemv_kernel` (+split-K variant): fused decode+GEMV for
  single-token steps, bf16 W never materialized; row prefixes reuse
  fused.compute_row_prefixes (same dict layout). Best in-context config
  (2 warps, BK=256), needs in_f % 256 == 0 (BK MUST divide in_f or the tail
  tile reads out of bounds → illegal memory access). MiniCPM5 end-to-end
  (KV-cache gen): cached 30ms/tok ≡ bf16 31; streaming fused 37ms/tok,
  1.8GB vs 2.2GB. Isolated min-of-reps kernel timing on this WDDM-shared
  desktop GPU is UNRELIABLE (picked (1,256) which is 15% slower in-model);
  always verify configs with deployed-model generation timing.
- **peakq.py v2 rows layout** (TPAB-inspired): `layout='rows'` restarts every
  row's T2/T3 streams + B2 bitmap at word boundaries (t1/t2/t3/b2 offsets
  int32[out_f+1]). Kernels `_peakq_decode_v2_kernel` (grid=out_f) and
  `_peakq_gemv_v2_kernel` (R rows/program, defaults R=4 BK=256 warps=2,
  out_f%R auto-halves fallback) need NO prefix tables — deploy skips
  compute_row_prefixes entirely, rows are randomly accessible, multi-row is
  free (rank state is per-row). Storage +1% (10.50 -> 10.59 bpw on MiniCPM5,
  offsets dominate on tiny mats). In-context: v2 35.2 ms/tok ≡ v1 35.5.
- **Host-RAM hygiene** (from tpab): PeakQLinear strips CPU packed bodies
  after GPU staging (`_strip_packed_bodies`); shared decode buffer singleton
  `_get_shared_w_buf`; `deploy_peakq_lazy` = big-model path (low_cpu_mem lazy
  load, per-layer CPU encode -> GPU stage -> drop, peak ~= resident + 1 layer).

## Key design decisions
- Scale computed as float32 (`max_abs/127.0`) then stored as bf16 — test ground truth
  must follow the same path or bf16 rounding mismatches occur.
- L3 (8-bit level) stored as raw uint8, not bit-packed (the Triton kernel loads it
  directly by index).
- Decode correctness verified bit-exact (max_err=0.0) for both Triton and scatter paths.

## Speculative decoding on Qwen3.8-27B (round4b_bisect.py) — hard-won pitfalls
- **transformers 5.15 cache `conv_states`/`recurrent_states` are DICTS** `{state_idx: tensor}`;
  iterating the layer object yields INT KEYS — `isinstance(r, Tensor)` guards silently
  disable ALL snapshot/rollback. Iterate `.values()`. This bug masqueraded as "accept-path
  corruption after 10-30 tokens" AND as "graph vs eager numerics differ" (both folklore
  died: g1-graph == g1-eager bit-exact once fixed).
- **Attention output reshape**: `[B,H,S,D]` needs `.transpose(1,2)` before
  `.reshape(B,S,-1)` — q_len=1 is accidentally fine without it, q_len>=2 garbles
  head/token layout (dmax 6.0 divergence).
- **CUDA graph pool aliasing**: tensors returned from a capture live in the shared pool;
  a LATER graph's replay (same pool) clobbers them. Copy outputs to pre-allocated static
  buffers INSIDE each capture. Symptom: garbage token ids (float bit patterns).
- **WDDM VRAM oversubscription** (>24GB on the 4090) silently pages to sysmem — the whole
  pipeline crawls at ~130GB/s. Keep peak < 24GB (int8+per-row-scale emb table = 1.27GB
  vs 2.54GB bf16).
- **Multi-token GEMV** (`udcq_fused_gemv_mt`, T∈{2,4,8}): same decode walk + per-token
  accumulators using the IDENTICAL `tl.sum` sub-expression => bit-exact vs sequential
  M=1 calls; bandwidth-bound so T=2/4 costs ~= one call. Makes layer-wise T-batched
  verify exact (GDN core per-token via gdn_seq_patch v3 S<=8; per-token SDPA).
- **Queue-architecture spec decode**: partial accepts roll back and RE-QUEUE the accepted
  prefix as the next block's known tokens (auto-reverified deterministically) — zero
  prefix-recompute replays, one sync per iteration. Commit only tokens BEYOND the known
  prefix (root position = pend_len-1) or text doubles.
- WDDM bottom line: there is NO sync tax. CUDA-event doorbell polling
  (ev.record + spin ev.query, no blocking synchronize) proved wall == GPU time;
  the earlier "~85ms sync tax" was VRAM exhaustion (leaked diagnostic clones,
  ~3.5GB of snapshot tensors) pushing into WDDM sysmem paging — in-context GPU
  time doubled (g4dec 105 -> 177ms, mem_get_info free=0.00GB). ALWAYS free
  diagnostics + empty_cache before timed runs; check torch.cuda.mem_get_info().
  Clean state: queue-k3 = 2 graph replays/iter, 120ms/iter GPU-bound, 2.22
  tok/iter (chain-draft quality decays: MTP recursion state drifts vs true h;
  zh worst at 1.25). k=1 (1.82 tok/iter, ~110ms) is at parity on WDDM; queue
  arch wins on low-sync platforms (Linux/TCC est. 17 at k=3, 27+ at k=7).
- mt-GEMV config: warps=4 WORSE than warps=2 in-context (g4 154 vs 105ms) — R4/BK256/W2
  is optimal; reconfirmed: always verify kernel configs with deployed-model timing.
- WDDM CUDA-graph CAPTURE ORDER: in Q38SpecEngine the S=1 greedy graph must be
  captured BEFORE the T=4 graphs — capturing it after several T=4 graphs HANGS
  the process silently at capture. (round4b always captured S=1 graphs first.)
- Speculative decoding vs sampling: temperature is compatible (scale both
  sides); top-p/top-k conceptually belong at verification only; repetition/
  frequency/min-p penalties are mutually exclusive with MTP — any sampling
  knob currently falls back to the greedy graph + CPU-side sampling
  (ixrun/sampling.py: HF-semantics penalty -> top-k -> nucleus -> temp).

## ixgs (Group-Scale, v3) — future direction
- `ixgs/` is a self-contained package: per-group scale (`group_max/15`, group=64)
  + the same (3,5,8) lossless encoding. Validated on MiniMax-H3 video DiT where
  per-tensor scales produce blocky output (coherent error accumulation across
  50 layers x 10 denoise steps); group scales reach 25.4 dB vs per-tensor 20.1 dB.
- Pitfalls encoded in its tests/docs: value-range tiers (not percentile), pad
  empty L3 stream to >=1 elem, pad l1/l2 streams with one zero word, fp16-round
  the group scales BEFORE quantizing (else decode doesn't close), ±0.0 sign is
  value-identical (use numeric equality for losslessness checks).
- Group-scale only wins on heavy-tailed weights (real LLM/DiT); pure gaussian
  synthetic data favors per-tensor.

- **NEW kernel variants MUST pass a bit-exact unit test before deployment** (per-group GMM: G=8 variant was never bit-exact-verified — it produced 27B text degeneration while G=16 was fine; severe VRAM paging did NOT corrupt text, so degeneration = numerical bug in the new variant). Rule: any new tl.constexpr configuration (GROUP/BK/R/T) gets a decode-vs-reference bit-exact check on real shapes before touching a model.
