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
- **triton_kernels.py**: fused decode via tl.cumsum + bitmap + cross-word bit extract.
  Precomputed per-block prefix sums (`precompute_block_offsets`). Scatter fallback if
  scheme ≠ (3,5,8) or no CUDA.
- **linear.py**: `Int8XLinear` (cache='full' decodes once; cache='none' re-decodes).
- **engine.py**: `Int8XEngine.from_pretrained(mode='cached'|'streaming'|'graph')`.
  Streaming: packed GPU-resident (463MB) + shared 14MB decode buffer, real-time
  Triton decode per forward (no DMA). Graph: CUDA-Graph captures all 168 decode
  kernels into one replay; GraphLinear does GEMM-only forward.
- **search.py**: exhaustive 2-5 level combo search by bit/w.

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
