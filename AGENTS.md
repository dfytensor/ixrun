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
- hpqs kernel PERF verdict REVISED (6c58852): the "0.02-0.04x bf16"
  Triton-era numbers were H2D-copy artifacts (codes tensor .cuda()
  inside the timing loop). Hand-CUDA kernel (warp-per-4-row-group,
  8 warps/block, split-K fp32 atomicAdd, uint4=16 codes, smem cb):
  0.90x bf16 matmul @2048x6144, gmax 0.0039 (reduction-order noise).
  HPQ-x-scale is DEPLOYABLE-TIER speed; line reopened. Smem level-1
  rows live at 8+s - mis-indexing breaks bit-exactness silently.
  ALWAYS pre-stage GPU tensors before kernel timing loops.
- **Extension kernels + CUDA graphs (294c163)**: torch extensions that
  launch kernels WITHOUT a stream arg land on the LEGACY stream, which
  silently BYPASSES cudaStreamCapture - the kernel runs once during
  capture (eager looks perfect) but is ABSENT from replay (pool
  garbage). ALWAYS launch on at::cuda::getCurrentCUDAStream() in
  extension code that can be captured. Symptom: eager correct + graph
  garbage. Also: UdcqLinear cache='full' re-decodes EVERY forward
  (docstring claims once) - stream is the deployed fast path.
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

## 27B C++ port (2026/10 session) — pieces all gated, Stage 4 = assembly
- **E2E INFERENCE WORKS (2026/10/09)**: coherent text both prompts
  ("The capital of France is" -> " Paris.\nThe capital of Germany is
  Berlin"; "北京最值得游览的三个景点是" -> "：\n1. 故宫（紫禁城...).
  TWO root-cause bugs, both in Python staging (zero kernel changes):
  (1) **Qwen3.5 RMSNorm is ZERO-CENTERED**: weight = Parameter(zeros),
  forward multiplies by (1.0 + weight) — NOT by weight. Affects
  input_layernorm / post_attention_layernorm / model.norm / q_norm /
  k_norm. Qwen3_5RMSNormGated (linear_attn.norm) is ones-init and
  multiplies DIRECTLY (direct w). Staging must feed (1+w) for the
  former. This one bug explained the entire 3.5x layer-0 norm mystery,
  the 60.4 xn2-norm "smoking gun" (post_w[3994]=-0.996 -> correct
  multiplier 0.0039 SUPPRESSES the channel-3994 spike; the wrong w
  amplified it), and every failed torch-ref gate (the refs copied the
  misread formula). LESSON: for every norm/math primitive, run the
  REAL module and compare element-wise BEFORE writing any gate.
  (2) **step27 slot stride**: T3(l,s) indexes (l*8+s)*3 — EVERY layer
  must stage exactly 8 slots x 3 tensors. Attn layers stage 7 packs +
  3 DUMMY tensors (zeros(8,uint8)+zeros(1,int32)+zeros(1,fp16));
  missing dummy shifts all later layers and OOB-crashes at layer 62
  (pack vector size 1488, first OOB index exactly 1488).
- **Verification tool: tests/test_27b_stage0.py** — zero-state
  single-token pos-0, hooks the REAL HF module (embed/in-ln/raw o/
  gated/core/h1/post-ln/mlp/layer-out), compares to C++ probes.
  After fix: all stages cos >= 0.998, norms within 0.5%.
  Build/run recipe: vcvars64 + CUDA_HOME/PATH=v12.6 + INCLUDE/LIB=v13.1
  (v12.6 install lacks headers; v13.1 nvcc has the cudafe++ AV;
  v13.4 (installed) nvcc is incompatible with torch cu126).
- **PERF (2026/10/09 late)**: 10 -> 30.8 tok/s decode in one session.
  (a) GEMV: the 6-bit decode was ALU-bound (~240GB/s even L2-hot),
  NOT bandwidth. udcq_gemv_v2 = warp-per-row (8 rows/block, x staged
  in smem; v1 re-read all of x per row), sign-folded 32-entry
  codebook (bit=1 -> +cb — per v1 semantics), float4 x, FMA loop:
  850-937GB/s on big shapes (gate 303->71us, lm_head 3719->1105us,
  qkv 228->64us). Dispatcher: v2 if out_f*in_f >= 8M else v1.
  Numeric gate vs torch dequant: rel 4.2e-6. Tools:
  tests/test_gemv27_bench.py (per-shape + dequant check).
  (b) CUDA GRAPH (WDDM = torch.cuda.CUDAGraph only): step27 split
  into capture-safe step27_impl (rope27/cache_write_b/attn_b read
  pos from a DEVICE scalar pointer; no .item inside) + eager step27
  wrapper + step27_g(h, dpos, theta); s27_reset() zeros
  conv/S/kv (multi-call hygiene). Warmup 2x -> reset -> capture ->
  reset -> replay per token. 36ms eager -> 32ms graph, TOKEN-EXACT
  vs eager. graph path is default in generate(graph=True).
  (c) Baselines measured same box: PY udcq-graph 12.2 tok/s;
  PY udcq-spec refused to load (2GB VRAM guard, desktop).
- **SPEC-PORT MATH VERDICT (2026/10/09)**: the PY spec engine's 2x
  gain (25.7 vs 12.2 tok/s measured; 32-53 documented) comes from
  amortizing a SLOW T=1 kernel (Triton ~234GB/s). Our v2 T=1 is
  already 815-938GB/s (near DRAM peak) so mt-amortization gains
  little: udcq_gemv_mt4_kernel (T=4, bit-exact vs 4x v2, x direct
  float4 reads; smem-chunked variant SLOWER — 1 blk/SM occupancy)
  measures only 1.3-1.65x per 4 tokens => spec net = LOSS with
  current kernels. mt4 kept (gated, tests/test_gemv_mt.py) for any
  future T=8/draft work. PY-spec needs Q38_MIN_FREE_GB=1.2 on a busy
  desktop (1.5GB captured clean).
- **PERF round 3 (2026/10/09 night)**: 30.8 -> 33.0 tok/s.
  (a) gdn_recurrent_v2_kernel: parallel over BOTH i and j
  (grid=(nv, dv/32), 128 thr = 4 i-chunks x 32 j, pairwise smem
  reduce; not bit-exact vs v1, deterministic) => 1.9 -> 0.59ms.
  (b) udcq_gemv_dual_kernel: b+a share one x => one launch
  (2*ceil(out_f/8) blocks, half -> pack0/pack1) used for b/a
  (48 calls saved) and attn k+v (16 saved) => 0.62ms total.
  Eager profile budget now: v2 GEMV 25.8 + rec 0.59 + dual 0.62 +
  rmsnorm 1.15 + rest 1.1 = ~29.3ms GPU, wall 30ms.
  DRAM floor for 19.2GB/token ~= 20-21ms => hard ceiling ~45-50
  tok/s with EVERYTHING else free; realistic ~35-38. The 1B's 2x
  over PY does NOT carry to 27B at 6bpw (weight bytes/token is the
  wall); PY-spec 25.7 vs our 33.0 = 1.28x.
- **PY-ENGINE OPTIMIZATION ROUND (2026/10/09 night)**: the "PY can't
  reach C++" verdict was WRONG — most of the gap was kernel design +
  operator fragmentation, both fixable in PY:
  (a) 27B: ported the C++ v2 GEMV design (sign-fold 32-entry cb +
  smem-x fp32 staging + float4 + fmaf) into experiments/
  udcq_gemv_cuda/udcq_gemv_cuda.py (ext name udcq_gemv_cuda_v2;
  mt kernel fold-only, x stays global). PY-graph engine 12.2 ->
  26.2 tok/s (2.1x), now UDCQ_CUDA_GEMV=1 by DEFAULT (udcq.py;
  =0 reverts; alignment guard %256 -> %16). PY-spec unchanged
  (25.2 — its bottleneck is draft/queue logic, not the GEMV).
  (b) 1B: same port for gsq (gsq_gemv_cuda v3) + torch.compile
  (default mode, wrapped before the manual graph capture; inductor
  fuses the HF norm/rotary/residual chains between graph breaks at
  the custom GEMV op): PY-gsq pure decode ~150 -> 174.5 -> 251.6
  tok/s = 95% of C++ (264.9). Test: tests/test_py1b_compile.py
  (differential timing; the old bench's "126" was prefill-diluted).
  Revised law: PY + hand-CUDA kernels + torch.compile reaches
  ~80-95% of the C++ engine; the residual is fused-op depth +
  zero-host-overhead, which is the C++ engine's reason to exist.
- **COMPILE ROUND + CTX CONFOUND (2026/10/09 late)**: the 1B PY-vs-C++
  comparison above was DOUBLE-CONFOUNDED (max_ctx 2048-vs-512 + no
  compile). Fair matrix at ctx=512, min-of-3 differential timing:
  PY-gsq 258.3 no-compile / **302.9 with torch.compile** vs C++
  264.9 => PY+compile BEATS the current C++ 1B engine by 14%.
  (1) attention cost scales ~linearly with max_ctx (StaticCache len)
  - ALWAYS ctx-match engine comparisons. (2) compile real gain is
  +8-17% (not +44% - that number had the ctx confound). (3) compile
  is SAFE on the 1B (Llama-arch StaticCache, text verified coherent)
  but CORRUPTS the 27B GDN engine (in-place conv/recurrent state
  mutations get functionalized -> token-repetition degeneration,
  +6.5% only) => q38_graph compile stays DEFAULT-OFF (Q38_COMPILE=1
  to experiment); step_graph gsq compiles by DEFAULT
  (STEP_GRAPH_COMPILE=0 disables; prefill routes through _orig_mod
  to avoid the long-sequence recompile). Deferred-sync note: the PY
  loop syncs every 16 tokens vs the C++ per-token .item() - part of
  the PY edge; C++ could reclaim via device-side token history.
- Known polish: generate() does not reset states between calls.
- All in engine_v5.cu, each with its own gate test, zero exceptions:
  UDCQ gemv/gemm (fp64 1e-7 tier; BUG: pack scale is f16 — kernel
  must take f32, convert at init, else rel-err 1.0); GDN recurrent
  core (3.97e-7 vs HF torch_recurrent_gated_delta_rule; q,k need
  L2norm + q pre-scale 1/sqrt(dk) HOST-side); l2norm (bit-exact);
  conv1d_update (state bit-exact); gated_rmsnorm (Qwen3_5RMSNormGated
  = (w*o*rsqrt(mean+eps))*silu(z)); FULL GDN LAYER assembled
  (test_gdn_layer.py, 2.5e-7 first try — spec in
  docs/gdn_layer_spec.md). UDCQ GEMM T-amortizes 55->11.7us/tok.
- split-K verdict: NO-DEPLOY (x-restaging/bandwidth-bound, not
  occupancy; order-preserving pipelining neutral — nvcc already
  hoists). GEMV ~22us = format floor; decode gains only via spec dec.
- Multi-instance engines: instance=N -> suffixed module = separate
  statics, gate PASSED (test_cpp_multi_instance.py). Weights dup
  0.7GB/instance.
- Stage 4 progress: step1 blob mmap loader + real-pack UDCQ GEMV
  gate 1.10e-7 (test_blob_l0.py); step2 REAL layer-0 GDN assembly
  gate (test_gdn_layer_real.py): vs bf16 5.09e-2 (format tier) BUT
  vs torch-same-quant-weights 9.92e-6 = math clean. Gate pattern:
  ALWAYS add the same-quant-weights torch chain to isolate math from
  format. HF notes: conv1d bias=False; rope_scaling=None (theta from
  text_config.rope_theta default); full-attn HAS q_norm/k_norm
  (256-dim RMSNorm, fp32, needed in attn path). Blob EXCLUDES mtp
  weights + all norms/conv/dt_bias/A_log (safetensors reads).
- BISECT BLOCKED by toolchain: nvcc cudafe++ dies 0xC0000005
  PERSISTENTLY (reboot did NOT fix; not RAM, not zombie procs,
  NOT size-linear — 4KB dead-code removal no effect; tiny exts
  compile fine; 5/5 deterministic on current source). Next
  unblock options: (a) repair/reinstall CUDA 13.1 toolkit or
  VS2022 BuildTools, (b) split source per refactor plan Step 3
  (move 27B section to engine_27b.cu, own compile unit),
  (c) bisect the SOURCE with direct nvcc -c runs on the cached
  cuda.cu (ninja bypass, locate the offending construct).
  Bisect logic + ground truth (HF top1 ' Paris') committed.
- nvcc AV FORENSICS round 2 (post-reboot): fwd-decl removal ->
  CLEAN "identifier undefined" error (no AV!) => the fwd DECL
  of attn_layer_step (36 torch::Tensor params) IS the cudafe
  trigger; single-decl + fresh cache dir (s4h) still AV =>
  not dir poisoning. PS IndexOf surgery on the .cu EMPTYED the
  file once (git checkout saved it) — NEVER do substring
  surgery on engine_v5.cu via PS, edit tool only.
  UNBLOCK = split source (Step 3): move 27B scheduler+layer fns
  to engine_27b.cu where step27 sits AFTER its callees (no fwd
  decls needed at all). That both fixes the AV and completes
  the hygiene plan.
- Step 4 (decode_64 scheduler) DESIGN CORRECTION before coding:
  the two layer fns have ASYMMETRIC contracts — attn_layer_step
  includes norms/residuals/mlp, gdn_layer_step is bare-core (norms/
  residual/mlp live OUTSIDE it). Draft attempt mixing them failed
  (reverted, never committed). REQUIRED: extend gdn_layer_step to
  full decoder semantics (add in_w/post_w norms + residual adds +
  gate/up/down GEMVs + silu — all pieces already gated) so both fns
  take (h, pos, weights) -> new h symmetrically. Then step27 = clean
  dispatch over layer_types {3,7,...,63} + final norm + lm_head
  GEMV (248320 out) + argmax. init27 static-init pattern (vectors
  of packs [64][7], norms, states) as designed. Gate ladder:
  (a) 4-layer schedule (3 GDN + 1 attn) bit-compare vs direct fn
  calls; (b) full-64 greedy tokens vs Python q38 pipeline (same
  blob -> expect >=90% match + coherent text).
- engine_v5.cu is 2100+ lines with ~350 dead/research lines; staged
  hygiene plan in docs/engine_v5_refactor_plan.md (execute Step 1
  first — zero-semantics dead code removal, no ext-name bump).

## C++ engine (ixrun/cpp) - 2026/10 session
- engine_v5.cu: GSQ GEMV (bit-exact port) + rmsnorm_f32 + rope + GQA S=1 attn
  + mlp_forward + layer_forward, all fp32-internal with bf16 boundaries.
  VALIDATED: 3-position chain memcheck-CLEAN, pos rel 0.062/0.079/0.091
  vs fp32 ref = GSQ inherent tier (bf16-activation + 5.5bpw), ACCEPTED.
- **SELF-FEEDING CUDA graph (2999a90) — the only correct + fastest decode**:
  in-graph loop argmax→g_tok_gpu→embed_lookup→24 layers→lm_head→argmax
  + tok_record(hist[d_pos]) + pos_incr(g_pos). Python = 1 replay()/token,
  zero sync/copy. 64/64 TOKEN-IDENTICAL vs eager per-token ref, 25.6 tok/s
  (pure GPU kernel time, WDDM PyTorch CUDAGraph). In-graph kernel writes
  PERSIST across replays; Python-written input buffers also visible.
- **BUG 1**: step_graph() had NO set_pos_kernel — replays ran at the stale
  d_pos from the last eager step() (pos 127) forever. The old "25 tok/s
  FULL graph coherent text" was position-corrupted (model robustness made
  it LOOK fine). LESSON: token-level comparison vs eager ref is MANDATORY;
  "coherent text" checks are worthless (positions wrong, cache clobbered,
  everything still looks fluent).
- **BUG 2**: lazy-init inside seed functions ran a full step at STALE g_pos,
  clobbering the last real cache slot (3/64 vs 24/24 mystery). LESSON:
  init helpers must be PURE (allocate only, launch zero kernels).
- **argmax_f32 was scanning only the first 256 logits** (single-element-
  per-thread, no grid-stride) for the full vocab — silently wrong since
  introduction; embed_lookup same. LESSON: single-block kernels over large
  arrays MUST grid-stride (or be bit-tested at real vocab size).
- **WDDM CUDA graphs (definitive)**: raw cudaStreamBeginCapture graphs read
  STALE data on replay (even with raw cudaMalloc + cudaMemcpy, zero PyTorch)
  — driver-level snapshot, UNFIXABLE on Windows. PyTorch torch.cuda.CUDAGraph
  (dedicated mempool) is the ONLY working graph path on WDDM. cudaGraphExec-
  KernelNodeSetParams untested (explicit-API construction would be needed).
- Iron rules: load_inline cpp_sources MUST declare every bound fn (empty
  = C2065); NO PYBIND11_MODULE in the .cu (load_inline generates it =
  LNK2005 PyInit); extension kernels launch on getCurrentCUDAStream
  (legacy stream silently escapes graph capture); after ANY kernel-sign
  or layout change bump the extension name (stale cache poisons);
  pack/ref roundtrip test BEFORE blaming the kernel; gs_pack lm_head
  reuses pks[-1] (weights are zeroed after pack - never re-pack).
- Open: GEMV kernel optimization (no vectorized uint4 loads, fp32 x
  re-reads; 25.6 tok/s is kernel-bound not launch-bound) → target 50+;
  prefill still per-token eager (~5s/128tok); C4 dual-format bf16xl.
- **ATTENTION V2 + full ladder (bf41f81/7409466/86eb40b/16b67a3/8e35088)**:
  attn_kernel_v2 = threads split the t-range, score computed ONCE to
  smem, block-max warp-reduce (rounding-free = exact), weighted-sum
  reads smem weights with identical t-order -> BIT-EXACT, 26 -> 295
  tok/s (11.4x). v1 was re-reading the whole KV 128x per head (9.6GB
  per token — THAT was the 38ms mystery, never the GEMVs). Batch
  prefill prefill_tokens(): one C++ call, no per-token pybind/.item()
  sync (240 tok/s). GEMV v2 (smem-x staging) bit-exact, 2.5x on
  lm_head shape only. CppGsqEngine (ixrun/cpp_engine.py) = production
  wrapper: generate/chat/serve all wired (--mode cpp-gsq), E2E gate
  tests/test_cpp_engine_e2e.py 64/64. Final: prefill 246, decode
  296 tok/s (~llama.cpp parity territory on this box).
- **Pinned-buffer race**: a host loop writing a reused pinned buffer
  + cudaMemcpyAsync = the copy reads AT EXECUTION TIME, next
  iteration's write races in. Kernel args are captured AT ENQUEUE —
  pass position values as kernel args (write_pos_kernel), never via
  shared pinned memory.
- **Warmup/throwaway-position rule (bug-2 class, structural)**: any
  warmup/dummy step must run at a position >= generation start
  (wrapper seeds ctx-16) — pollution there is rewritten-before-read;
  at a stale g_pos it clobbers the last REAL prefill slot silently.
- Polish queue for cpp-gsq: per-request max_tokens clamp in facade,
  chat-template stop tokens (<|im_end|> leaks; eos_token_id mismatch),
  streaming chunk-boundary token dedup; batched-GEMM prefill is the
  next big perf play (format-level, 10B-group packing blocks uint4).

- 27B INVESTIGATION POSITION MISMATCH: C++ h(L0)=5.915 was pos 0 (zero states, first token — tiny core CORRECT: only w[3] tap fires); HF 20.606 was pos 4 (states accumulated — larger core normal). The A/B comparison was INVALID. The e2e degeneration investigation needs complete restart with position-matched comparisons. All kernels/staging/decode verified clean. The real remaining question: does the blob-based C++ engine produce the same TEXT as the PY engine on the same prompt? If yes, both are correct (quantization tier). If no, bisect at MATCHED positions.

- CRITICAL CORRECTION to the A/B mismatch note above: test_27b_decisive.py used input_ids=[760] (SINGLE TOKEN) for the HF forward — so HF h(L0)=20.606 IS at pos 0 with zero states, SAME as C++ 5.915! The position mismatch theory is WRONG. The REAL finding: my torch ref formulas (used in every gate) produce h1 ~ 1.1-5.9-norm while the REAL HF module forward produces h1 ~ 20.6-norm on the same input — a ~3.5x core contribution difference. THE BUG IS IN MY TORCH REF FORMULAS (which the C++ kernels correctly implement): the formulas I wrote from reading the code differ from what HF actually computes. NEXT: (1) run the real HF module at(xn_bf16, None) on CUDA and compare its output vs my torch ref chain output element-wise, (2) the first divergent stage identifies the formula error (prime suspects: gated norm silu(z) application, conv1d state handling, or the l2norm+scaling interaction).

- 27B SMOKING GUN: rmsnorm_fw in-situ output 60.4-norm vs expected 6-16 (|post_w|=15.77). Standalone gate with randn w (71-norm) masked this. ALL 27B kernel gates MUST use real weight scales.

## 27B C++ engine — ULS format + perf marathon (2026/10/10-11) — 33.0 -> 37.5 decode / 37 -> 603 prefill
- **ULS (log2-scale UDCQ)**: scale stream fp16/16 -> uint8 log2/16 with the
  per-row (base,step) header INSIDE the stream (kernel signatures unchanged;
  base/step read from the passed pointer) => 6.0 -> 5.5 bpw, -8% weight
  bytes, -1.3GB VRAM. Worst row scale error 1.29% vs the 3.4% format tier.
  Repack = pure stream transform (tests/repack_uls.py, 497 mats/209s) ->
  F:\models\qwen38_uls_blob.pt. 5 scale-reading kernels templated <bool ULS>;
  udcq_set_uls(1) per process. PY track: UdcqLinear AUTO-DETECTS uint8
  scale (dtype == the flag; zero plumbing); ext v4; ULS non-{4,8} M must
  chunk through mt8/mt4/single GEMVs (the torch-dequant fallback = 75s/prompt!).
- **cudafe++ 0xC0000005 (recurring build AV)**: AVs ONLY when nvcc is spawned
  from the python process tree (ninja-from-cmd and manual nvcc succeed
  deterministically; ~50% flaky per attempt). WORKAROUND = tests/build_27b_ext.cmd
  (stage1 python regenerates cuda.cu, may AV harmlessly; stage2 ninja from cmd)
  + IXRUN_PREBUILT=1 makes _build_ext import the prebuilt .pyd directly.
  Ext q27k->q27l->q27m across the format/fusion changes.
- **Decode fusions**: add_norm_f32 (residual add + next norm in ONE launch,
  1024 threads; 161->33 norm calls/token; TOTAL 28.52->27.75ms);
  udcq_gemv_multi_kernel (qkv+z+b+a / q+k+v / gate+up each ONE launch; per-layer
  288B device tables with SYNC table-build memcpy - async would read a dead
  stack frame); v2 calls 369->129/token. 27->28ms wall.
- **attn_b_v3 split-K (long ctx)**: old attn_b_kernel swept t serially per
  thread (554us/call at pos 3800!). v3 = ATTN_V3_C=8 chunk-blocks/head +
  merge. BUG FOUND: den block-reduced across 256 threads that EACH walked
  all chunk entries (256x; the 0.996 = 1-1/256 signature).
- **Prefill (blocked, S=8 -> GEMM)**: mt8b (bf16-x, warp-per-row, L1-cached,
  the PY ext design) 149 tok/s -> dequant-to-bf16 + cuBLAS + segment
  (T<=256, layer-major) 307 -> batched T-row cores (conv1d_batch rl2n_t
  sigmoid_b ggate_b chunk_qgate_t rope27_t attn_batch_proj) 400 ->
  gdn_recurrent_v2t (state slice in SMEM across all T tokens; the per-token
  kernel RMW'd 6.3MB state/token = 77GB/segment!) 427 -> attn_b_v3t+merge_t
  (batched; 8192 per-token graph nodes had ~5us node overhead each) = 603.
  Gates: gdn_cmp_test A/B per-token-vs-batch (0.0 / 2.6e-7); gemm TEXT MATCH
  first-tok 561; 26-tok A/B TOKEN/TEXT EXACT. FOUND along the way: conv
  use_act=1 needs siLU; FLA l2norm is rsqrt(sum+eps) with NO /dk (I added
  a division); cudaMemcpy2DAsync for the vr slice.
- **GEMM segment budget (T=256, 252.6ms)**: dequant 83 (bf16 round-trip:
  packed 17.6GB read + 51GB bf16 write), cublas 99 (13.1 TFLOP @ ~132
  TFLOPS = near tensor peak), recurrent 36, attn 17, misc ~17. Dual-stream
  dequant overlap = NEUTRAL (segment is DRAM-saturated; overlap can't hide
  bandwidth behind bandwidth - reverted with data). int8 route (probe 2.06x
  on _int_mm) KILLED on quality: per-row int8 = 4.0% rel vs bf16 0.64%
  (6-octave within-row scale spread; IG32 lesson again) - code stays
  default-off (IXRUN_GEMM_I8). Remaining un-attempted lever: fused mma
  decode-GEMM (decode W into smem bf16 fragments per tile, skip the
  102GB/segment round trip; ~1.5-2x prefill, multi-hour).
- Handy hooks: IXRUN_NO_PREFILL=1 (legacy per-token prefill for A/B),
  IXRUN_GEMM_I8, IXRUN_BATCH_GDN=0, IXRUN_PREBUILT. Profiling the 27B decode
  graph with torch.profiler works (per-kernel attribution in graph replays).
- **v2c GEMV** (chunked-x, in_f>12288): smem 32.9KB vs v2's 69.6KB => 3
  blocks/SM; chunk boundaries at 32-group multiples keep the per-lane
  order => bit-exact. down 783->793GB/s (already 88% of practical; decode
  GEMVs are at their bandwidth ceiling - per-shape finals qkv 787 / z 720 /
  q 832 / gate 825 / down 793 / o 677 / lm_head 869 GB/s).
- **Silu epilogue fusion (decode)**: udcq_gemv_dual_silu_kernel - one warp
  walks BOTH gate/up packs for the same x row (x smem read once feeds both
  fma chains), epilogue writes act = v/(1+expf(-v))*up (identical to
  silu_kernel). TOKEN-EXACT; 27ms/step = 37.4 tok/s; eager 0.85->0.59s.
  LESSON: an aborted test run left a zombie python holding the GPU - the
  next test measured 63s/step of pure contention; ALWAYS kill zombies and
  re-test before believing a regression.
- **Long-ctx / KV-FP8 CLOSED with data**: at 4K the attention is COMPUTE-
  bound (attn v3 fixed the read inefficiency; KV bytes 268MB = ~0.3ms);
  FP8 KV saves ~0.15ms at 4K (0.5%) and only ~1-2ms at 16K where the
  O(pos) attention COMPUTE already caps us (~29 tok/s at 16K, ~23 at 32K
  extrapolated). The real 32K+ fix is attention-compute restructuring
  (chunked causal / paged), not KV bytes. VRAM: bf16 KV supports ~39K
  ctx in the free VRAM - already above the compute-capped useful range.
- **PY multi-pack CLOSED**: PY mt4 already runs at 0.96x a single call
  per 4 tokens (the L1-cached bf16-x design); the PY spec's bottleneck is
  the MTP draft quality (E=1.6) and the PY graph's gap to C++ is python
  host overhead - neither is fixed by packing.
- **Decode ladder final**: 37 tok/s = 72% of the absolute DRAM ceiling
  (51-53 at 900-940GB/s), ~95% of the 2-3h-reachable maximum. Every
  byte-reduction and kernel path measured and closed (per-32 16.5% err,
  int8 4.0%, fused-mma decode-ALU+register-ceiling, spec E-bound).
- **Fused mma decode-GEMM — CLOSED with measurements**: v1->v3 built and all
  gated (rel <= 2.2e-4, 5 shapes incl edges): v1 naive 1.12ms, v2
  double-buffer + 16-token sliver epilogue (smem 41.5KB, no attr), v3
  ldmatrix (PLAIN .x2/.x4 - our xs is [token][k] = the B^T layout; .trans
  swaps k/n => rel 1.6; k/n-swap is the .trans-bug signature) + 32-entry
  sign-fold cb_fold. Perf stuck at ~1.14ms vs the dequant+cublas pair
  0.68ms: the DECODE ALU (~8 instr/elem = >= the mma issue) + 25%
  occupancy (a full [128x128] fp32 acc tile = 64K regs = the whole SM
  register file). The economics close only with a perfect free decode;
  at T=1024 the pair amortizes (dequant once) while the fused re-decodes
  per token-tile => the design loses MORE at large T. Do not re-open
  without a different mma input strategy (e.g. 4-bit tensor ops).

## 1B StepGraph decode marathon (2026/10/13) — 161 -> 373 tok/s in one session
- PROFILE FIRST: the "1B is overhead-bound" folklore was WRONG at the
  kernel level - torch.profiler on the captured graph showed attention
  2.25ms/step (38%): transformers 5.15 StaticCache hands SDPA the
  FULL-WIDTH 2048-slot k/v + bool mask, and the cutlass mem-efficient
  kernel burns ~94us/layer regardless of position. Backend switching
  (flash/math/cudnn) does nothing; a dedicated S=1 kernel is the fix.
- ixrun/fast_attn.py: intercept F.scaled_dot_product_attention (call-
  time attribute lookup in sdpa_attention_forward), two-phase split-KV
  Triton (8 chunks, local online softmax + merge), 12x faster than
  cutlass. Wrapper needs @torch.compiler.disable or Dynamo inlines it
  and inductor tries to recompile the Triton kernel (fails).
- GQA TRAP: mask=None decode takes the enable_gqa path - k/v stay at
  2 kv-heads (no repeat_kv). Indexing k/v by the Q head id = OOB IMA
  at capture. REP constexpr (h_kv = hq // REP); mask mode receives
  pre-repeated kv (rep=1). Partials must be stored per Q head.
- Triton loop-carried state MUST be tensors: a python-scalar -inf m_i
  changes type after iteration 1 -> silently wrong code (the flash
  tutorial initializes m_i as a zeros-tensor minus inf for a reason).
  Fully-masked block: s - (-inf) = nan -> clamp m_new via m_safe.
- Positional mode: kernel reads the engine's self.pos device tensor
  (offs <= pos; pointer captured, value read at execution - the 27B
  write_pos_kernel pattern), HF mask build stubbed via
  masking_utils.create_causal_mask + the modeling_llama binding
  (transformers 5.15 has NO _update_causal_mask method). Saves the
  per-step cat/neg mask kernels (~0.2ms). Prefill goes is_causal -
  DIFFERENT rounding than the masked path: near-tie first tokens can
  flip vs mask mode (1 of 4 prompts). Engine deterministic per mode.
- bf16xl GEMV v4: torch::zeros + yf.to(bf16) per call = 2 wasted
  kernels x168/step. n_sp==1 stores bf16 directly; n_sp>1 writes
  disjoint fp32 partial slots + one tiny reduce. REJECTED en route:
  (a) single global arrival counter - 1728 atomics serialize on ONE
  L2 address = 4x slower; (b) per-row counters + __threadfence -
  device-scope fences ~50ns x blocks dominated (95us vs 25us).
  Split-K path now deterministic (bonus - atomicAdd was not).
- UDCQ lm_head (IXRUN_Q_LMHEAD=1 default; skipped for bf16/bf16xl by
  design): cublas 0.4GB bf16 gemv 0.47ms -> fused stream GEMV ~0.2ms,
  -0.26GB VRAM. Gate: 4/4 prompts BIT-IDENTICAL vs bf16 lm_head on
  the deterministic bf16 codec (noise below every argmax margin).
- Env flags: IXRUN_FAST_ATTN=0 (revert attention), IXRUN_POS_ATTN=0
  (bool-mask path), IXRUN_POS_STUB=0 (keep HF mask build), 
  IXRUN_Q_LMHEAD=0 (bf16 lm_head), STEP_GRAPH_COMPILE=0.
- Final: gsq 373 / bf16xl 312 / udcq 289 tok/s (48-step decode,
  4090/WDDM). gsq >= llama.cpp bf16 (350) / BF16X (380) at 5.5bpw.

## bf16xl GEMV ~582GB/s wall — three hypotheses tested and FALSIFIED (2026/10/13)
- smem-x staging (27B-v2 style, fp32 +1-pad tiles): isolated wide-shape
  neutral, small shapes WORSE (staging/sync tax), in-model 306 vs 312.
  The x path was never the bottleneck (L1 broadcast was already fine).
- aligned-uint4 multi-group walk (8 groups = 112B = 7x16B; per-lane
  16B-aligned when n_gr%8==0): bit-exact, in-model 313-319 = v4 within
  noise.  The unaligned ld_u32 L1-double-booking theory: falsified.
- occupancy splits for small-row layers (k/v out_f=256 -> 32 blocks):
  in-model 306-307 = slightly WORSE (the extra reduce-kernel dependency
  costs more than the occupancy gains).
- Desktop noise NOTE: same-code reruns spanned 231-321 tok/s today
  (QQ/Quark/Edge background load); ±15% swings make sub-5% kernel
  deltas UNFALSIFIABLE in-model on this box.  All three attempts
  reverted; v4 stands.  Remaining lever: NCU-grade profiling or a
  different decode strategy - do NOT reopen without one.

## ULS on the 1B + the vacuous-gate lesson (2026/10/13, acbd2d8)
- udcq-uls codec: deploy_udcq(stream) -> lmhead-q -> udcq_scale_to_uls
  walk (AFTER lmhead-q so the lm_head converts too).  6.0 -> 5.5bpw,
  udcq 289 -> 324 tok/s (+12%).  Gates: dequant rel 0.0029, ext
  single/mt8/mt4 uls=1 all ~0.0024 vs torch dequant, 4-quadrant text
  (fp16/ULS x lmhead on/off) coherent.
- VACUOUS GATE: the earlier "lm_head 4/4 BIT-IDENTICAL" gate compared
  codec=bf16 against itself - bf16 SKIPS lmhead-q on both sides, so
  nothing was gated.  The real bug it would have caught: lmhead-q
  fitted a per-module codebook while the ext kernels read the ONE
  __constant__ codebook installed by the first module (the per-call cb
  arg is IGNORED) -> the lm_head decoded with the wrong codebook.
  fp16 masked it (universal fits near-identical); ULS pushed text into
  '<s>' degeneracy.  Fix: quantize lm_head with stats['codebook'].
  RULE: a gate must verify the CONFIG IT SHIPS - check the side
  actually changed (log the lmhead-q/uls deploy lines in BOTH runs).

## 1B vs llama.cpp: parity campaign (2026/10/13 late, 8875543..34f4acb)
- The llama.cpp table (their own numbers, 4090): F16 371t/s @801GB/s
  79% | BF16X 389 @606 60% | Q8_0 508 @584 58%.  Read it as: the
  SIMPLEST-format-wins hierarchy + the decode-ALU tax visible in the
  falling effective bandwidth.  Our bf16xl GEMV was ALREADY at their
  BF16X kernel rate (582 vs 606 GB/s) - the whole gap was the
  everything-else around the GEMVs, not the GEMVs.
- The fusion ladder that closed it (PY side, each with a gate):
  cached rotary tables (build [max_ctx,hd] ONCE, forward = gather;
  the HF chains were 0.27ms/step) -> fused rope application (ONE
  triton kernel for all q+k rows per layer, rotate_half via the
  partner-element load + sign; replaces 5 kernels/tensor = 0.3ms)
  -> explicit codebook staging BEFORE capture (the lazy install
  leaked into the graph and the lm_head decoded with a stale
  constant = '<s>' spam that looked codec-dependent) -> lmhead-q
  for bf16xl too -> fused add+rmsnorm layer (one kernel emits the
  new residual AND the normed output; in-place on the attn buffer).
- int8g codec: int8 + per-group f16 scale (g=32 = exactly Q8_0's
  block; 8.5bpw).  The GEMV = a uint4 load + 16 FMAs - 901GB/s on
  the lm_head shape isolated.  rel 0.0056.  169 linears incl lm_head.
- LATENT BUG the campaign flushed out: Int8GLinear/Bf16xlLinear M=1
  paths returned 1-D tensors; HF's post-o_proj reshape masked it
  until the fused add+norm consumed o_proj output directly.
  RULE: every flat-returning GEMV module must view() back to x's
  shape (UdcqLinear always did).
- RESULT (48-step decode, sync-per-token = the CONSERVATIVE protocol;
  deferred-sync generate runs faster):
  bf16xl 384+-5 vs BF16X 389  = parity
  gsq    463    vs BF16X 389  = +19% at 5.5bpw
  int8g  487+-4 vs Q8_0  508  = 96% conservative, >= parity deferred
- Desktop noise remains +-10-15%: sub-5% claims need median-of-3+
  runs interleaved, and even then one 350-vs-490 outlier/day is
  normal (background load bursts).

## 1B lessons applied to the 27B PY graph (2026/10/13 night, f57b694): 26.2 -> 31.5 tok/s (+20%)
- PROFILE FIRST held again, and HUMBLED the hypothesis: the 27B PY
  graph attention is NOT the 1B pathology - 16 full-attn layers x
  cutlass @ ctx 256 = 0.25ms/step only (the 1B's 94us/layer was the
  2048-wide cache; the 27B graph runs max_ctx=256).  The real split:
  GEMV 30.7ms (573GB/s) + ~6ms norm/copy chains.
- fused add+norm qwen3_5 variant: output = rmsnorm(sum) * (1+weight)
  (ZERO-CENTERED - the AGENTS lesson).  (1+w) buffers must be built
  POST-LOAD: an __init__-time build captures the META/unloaded weight
  (IMA at the first layer).  -1.1ms/step.  IXRUN_FUSED_ADDNORM=0.
- chunked-x GEMV = the big one (+16%): the PY v2 staged the WHOLE x
  in dynamic smem (IN_F*4B; 17408 -> 69.6KB = 1 block/SM = 12.5%
  occupancy).  The C++ v2c already knew this ("smem 32.9KB -> 3
  blocks/SM"); the PY ext now does the same: 128-group-multiple
  chunks, static 8.3KB, per-lane order preserved = bit-exact.  GEMV
  30.7 -> 25.6ms = 660GB/s = the C++ engine's 655.  Ext v4 -> v5.
- PY graph now 31.5 t/s.  The remaining ~5ms vs the C++ 37.4 = the
  HF orchestration (867 copies, norm tails, GDN python) - the C++
  engine's reason to exist, per the revised law.
- NOT applied (measured dead for the 27B): fast attention (0.25ms
  total), int8g (the 4% int8 quality rejection stands), deferred-sync
  (the graph replay is already zero-sync).

## MTP/spec baseline after the GEMV v5 (2026/10/13 night)
- The spec engine AUTO-benefited from the chunked-x GEMV (the draft
  path = single-token MTP-head forwards): 29.8 -> 32.4 (en) / 28.4
  (zh) tok/s measured with the accept histogram (E=2.09/1.66 at
  TOK=4; hist en {1:10, 2:4, 3:6, 4:3} - the chain dies at 1-2 most
  iterations).
- TOK sweep VERDICT: T=8 en 32.7 (E=2.40) = a wash vs T=4; zh 24.7
  (E=1.78) = WORSE.  The verify cost is flat in T (mt8 = one weight
  pass) but the ACCEPT does not scale - the marginal drafts 5-8 are
  worthless.  TOK=4 stands; the lever is E itself (draft quality),
  not the chain length.
- The structural math for a future C++ spec port: the C++ verify =
  mt4 ~17.2ms (the 1.5x amortization) + the C++ MTP-head draft ~1ms
  per step (0.65GB: fc + one full layer + the shared lm_head at
  660GB/s); at the measured E=1.7-2.1 that is ~10-12ms/tok = 80-100
  t/s.  The old "spec net = LOSS" verdict (mt4 1.3-1.65x) predates
  the E histograms and the ULS blob - worth re-examining before
  believing either way.  The port needs: the MTP head in C++ (fc +
  one decoder layer + norms + the shared lm_head read), the
  chain-draft graphs, the rollback ring - a multi-hour project.
- Env forensics note: Q38_MIN_FREE_GB set in a .cmd did not reach
  python on one box run (read 1.2 while the file said 0.9; a probe
  cmd with the same pattern propagated fine); forced in-script
  os.environ assignment as the workaround.  Root cause unidentified.

## MTP round 2: the in-graph accept (2026/10/13 late, e822c77)
- The greedy accept moved INTO g4dec_body (captured): kills the
  per-iter int() hard sync + 4-5 python kernels.  en 32.4 -> 36.5
  t/s (+13%) = within 3% of the C++ GREEDY engine; the PY spec is
  now the fastest 27B config on this box.  zh flat (28.5): zh rolls
  back 27/29 iters and the _fast_rollback state ring dominates.
- The E ceiling is P1 (the one-step MTP accuracy ~0.57 en): the
  measured E=2.09 = 1 + P1(1 + P2|1(...)) exactly; raising E means
  touching the MTP weights, not inference-time code.
- C++ spec port floor math (a dedicated session): the PY iter =
  the mt4 25.6 + the GDN chunk ~14 + host; the C++ GDN core is 0.6
  for the same tokens -> ~70 t/s at the SAME E.  Needs: the MTP
  weights in C++ (fc = a 2H->H GEMV, one full-attn layer with
  use_cache=False = the degenerate per-token attention = the
  existing kernels, 3 norms, the shared lm_head), the chain graphs,
  the rollback ring.

## Server concurrency state (2026/10/13 night)
- The serialized mode (gen_lock) is CONCURRENCY-SAFE: 6-8 parallel
  HTTP clients queue correctly, zero corruption, ~1.78s/req at
  max_tokens 24 (the aggregate ~0.56 req/s).  That is the shipped
  behavior; the fastapi layer handles the connections, the
  generations serialize on the single decode graph.
- --batched (BatchedGreedyGenerator, continuous batching) is
  INCOMPATIBLE with the step-graph custom-module stack: the batched
  forward hits a device-side assert (the batching core predates
  UdcqLinear-stream/Bf16xlLinear/fused add+norm/pos-attention and
  drives the raw model at M=8 through paths those modules do not
  cover).  Reverted to the serialized server; the flag stays off.
  Fixing it = port the batching core to the custom modules (a
  dedicated session; the economics are proven at B>=8 = 3.5x).
- bf16xl served answers at temp 0: 'Germany -> Dresden' happened in
  the serialized path and 'Berlin' in the batched path (different
  kernel rounding) - a 1B knowledge error near the tie, NOT a
  quantization issue (the tier is lossless).

## Fused sampling kernel: WIP parked (2026/10/14)
- experiments/sample_fused/: ONE-kernel penalties+top-k+top-p+temp+
  softmax+multinomial (v6).  Greedy+penalties = EXACT vs the torch
  reference; 33us/launch isolated.  The SAMPLING path is NOT correct:
  u reads negative with a positive total, thr/total alternate by the
  curand subsequence parity, supp=1.  Verified-not-the-cause: the
  int/float s_bins, the pass-3 per-thread walk (replaced with the
  correct exclusive-prefix-scan walk), the smem sizes, the stale
  binary (v6 rebuilt clean, float out).  REMAINING SUSPECTS: the
  block reduction (fixed once but the symptom persisted -> re-verify
  gmax in-kernel vs torch), the curand_init/Philox in a
  fast_math TU, or an smem alias the compiler hid.  Debug protocol
  that works: the float out[4] dump (tok/thr/u/total), the python
  mimic of the threshold logic (the logic IS sound: 93k tokens pass
  the hist path at the wide test distribution).
- The SHIPPED path stays: the torch-op GPU sampling in step_graph
  (_gen_tokens): the top-k mask + the sort/nucleus + the multinomial
  on-device (~10 launches, 8.7ms/tok served overhead) - the server
  round-trip 15 -> 134 tok/s after the needs_sampling routing fix.
- The fused kernel payoff when finished: the sampling overhead
  8.7ms -> ~0.15ms per token (the L2-resident row, 3 passes,
  one launch) + the penalties move on-GPU (the CPU sampling path
  retires).  Do NOT reopen without the in-kernel gmax/denom dump
  vs the torch first.

## Sampling fast path (2026/10/14, 300af17): served sampled decode 134 -> 262 tok/s
- The per-token sampling ran ~12 FULL-VOCAB kernels (the 130k sort
  ~2-4ms + the multinomial ~1ms) = ~8.7ms/token of the served
  overhead.  Replacement: everything within a top-1024 subset
  (topk -> temp -> the sort/cumsum within 1024 -> the nucleus + the
  top-k mask -> the Gumbel-max pick).  The Gumbel-max IS an exact
  categorical sample.  ~0.5ms/token.  No host coverage sync (the
  top-1024 truncation at top_p <= 0.95 is statistically negligible;
  documented).
- Server round-trip (the official 0.9/0.95): 30/60/120-token
  requests = 219/256/262 tok/s.  The gap to the direct greedy
  (312-384) = the sampling launches + the HTTP/SSE.
- The WIP fused kernel (sample_fused) remains parked: its payoff
  (8.7 -> 0.15ms) is now mostly captured by the subset trick (8.7 ->
  ~0.5ms) - the reopen bar is higher.
- Still on the CPU path: repetition/presence/frequency penalties
  (the GPU version needs the counts scatter + the in-kernel logits
  adjust - fold it into the parked kernel or a small standalone).

## Sampling fast path (2026/10/14, 300af17): served sampled decode 134 -> 262 tok/s
- The per-token sampling ran ~12 FULL-VOCAB kernels (the 130k sort
  ~2-4ms + the multinomial ~1ms) = ~8.7ms/token of the served
  overhead.  Replacement: everything within a top-1024 subset
  (topk -> temp -> the sort/cumsum within 1024 -> the nucleus + the
  top-k mask -> the Gumbel-max pick).  The Gumbel-max IS an exact
  categorical sample.  ~0.5ms/token.  No host coverage sync (the
  top-1024 truncation at top_p <= 0.95 is statistically negligible;
  documented).
- Server round-trip (the official 0.9/0.95): 30/60/120-token
  requests = 219/256/262 tok/s.  The gap to the direct greedy
  (312-384) = the sampling launches + the HTTP/SSE.
- The WIP fused kernel (sample_fused) remains parked: its payoff
  (8.7 -> 0.15ms) is now mostly captured by the subset trick (8.7 ->
  ~0.5ms) - the reopen bar is higher.
- Still on the CPU path: repetition/presence/frequency penalties
  (the GPU version needs the counts scatter + the in-kernel logits
  adjust - fold it into the parked kernel or a small standalone).

## The SELF-FEEDING SAMPLING GRAPH (2026/10/14 late, 8e31c9a): 6x the
## concurrent throughput, WDDM-noise-immune
- The python sampling (even the top-1024 subset) = ~13 python
  launches per token; a loaded WDDM desktop swings the submission
  cost 10-30x (the same hour: 3.6ms/tok -> 25ms/tok).  The fix is
  structural: the Gumbel-max sampling over the top-1024 CAPTURED
  with the forward into a second shared-pool graph.  The graph READS
  nxt_g as its input and writes the pick back - the self-feeding
  loop; the host per token = pos.fill_ + one replay.  Concurrent x6:
  1.0s wall (was 10.7s).
- Capture-safe bisected findings (each verified in isolation):
  - device-indexed tensor[idx] = x writes BREAK the capture
    (use scatter_/gather)
  - fill_(0-dim CUDA tensor) = a hidden D2H sync -> BREAKS it
    (use gather + copy_)
  - torch.sort BREAKS it (the 4096-bin histogram nucleus instead)
  - the warmup MUST run the FULL body before the capture (the
    kernel selection/workspace allocs during a capture invalidate it)
  - the readback trigger = an ITERATION counter (len(out) % period
    never fires - out is static between readbacks -> the loop ran to
    max_ctx, 27s/request)
- The greedy = the SAME graph with temp clamped to 1e-6 (the exact
  argmax); the penalties unconditionally applied in-graph (the
  defaults are the identity).  One graph serves EVERY config.
- do_sample probabilistic requests still take the python loop
  (IXRUN_GRAPH_SAMPLE=0 reverts everything).
