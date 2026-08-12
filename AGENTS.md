# AGENTS.md — IXRUN project guide

## Environment
- **Python**: `F:\rwkv\.venv\Scripts\python.exe` (3.12, has torch/triton/transformers)
- Run all commands from `E:\IXRUN` (working directory).
- Offline mode required (no internet to HF): prefix with
  `HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1`
- Model: MiniCPM5-1B (Llama-arch, 24 layers) at path in `ixrun/config.py:MODEL_PATH`.

## Commands
```powershell
# unit tests (fast, no model load)
$env:HF_HUB_OFFLINE='1'; & 'F:\rwkv\.venv\Scripts\python.exe' -m tests.test_core

# full pipeline benchmark on MiniCPM5-1B (loads model 4x, ~3 min)
$env:HF_HUB_OFFLINE='1'; $env:TRANSFORMERS_OFFLINE='1'; & 'F:\rwkv\.venv\Scripts\python.exe' -m benchmarks.bench_minicpm5

# CLI
& 'F:\rwkv\.venv\Scripts\python.exe' -m ixrun.cli search
& 'F:\rwkv\.venv\Scripts\python.exe' -m ixrun.cli generate "Hello" --stream
& 'F:\rwkv\.venv\Scripts\python.exe' -m ixrun.cli bench
```

## Architecture
- **quantize.py**: bf16 → per-tensor int8 (scale=max_abs/127) → (3,5,8) nested-bitmap
  packing. Lossless on the int8 representation.
- **triton_kernels.py**: fused decode via tl.cumsum + bitmap + cross-word bit extract.
  Precomputed per-block prefix sums (`precompute_block_offsets`). Scatter fallback if
  scheme ≠ (3,5,8) or no CUDA.
- **linear.py**: `Int8XLinear` (cache='full' decodes once; cache='none' re-decodes).
- **engine.py**: `Int8XEngine.from_pretrained(mode='cached'|'streaming')`.
  Streaming keeps packed on pinned host, decodes into shared 22MB GPU buffer.
- **search.py**: exhaustive 2-5 level combo search by bit/w.

## Key design decisions
- Scale computed as float32 (`max_abs/127.0`) then stored as bf16 — test ground truth
  must follow the same path or bf16 rounding mismatches occur.
- L3 (8-bit level) stored as raw uint8, not bit-packed (the Triton kernel loads it
  directly by index).
- Decode correctness verified bit-exact (max_err=0.0) for both Triton and scatter paths.
