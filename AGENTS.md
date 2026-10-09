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
