# AGENTS.md — IXRUN project guide

## Environment
- **Python**: `F:\rwkv\.venv\Scripts\python.exe` (3.12, has torch/triton/transformers)
- Run all commands from `E:\IXRUN` (working directory).
- Offline mode required (no internet to HF): prefix with
  `HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1`
- Model: MiniCPM5-1B (Llama-arch, 24 layers) at path in `ixrun/config.py:MODEL_PATH`.
- Model (large): Qwen3.8-27B (multimodal qwen3_5, 64 layers hybrid linear/full attn)
  at `ixrun/config.py:QWEN38_PATH` = `E:\models\Qwen3.8-27B`. Needs transformers>=5.8
  (installed: 5.15.0). 27B streaming: CPU lazy-load -> per-layer quantize -> packed
  to GPU, bf16 freed eagerly; 606 layers, packed 16.91GB, runs on 24GB card.

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
