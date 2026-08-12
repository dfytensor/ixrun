"""Default paths and constants for the IXRUN engine."""
import os

# --- Python interpreter (the verified env with torch/triton/transformers) ---
PYTHON = r"F:\rwkv\.venv\Scripts\python.exe"

# --- Model ---
MODEL_PATH = (
    r"F:\dg_minicpm5\hf_cache\models--openbmb--MiniCPM5-1B\snapshots"
    r"\4e9de7a0778dc1c362e983e6858f0e77542cbdca"
)

# --- Dataset cache (for wikitext ppl eval) ---
DATASET_CACHE = r"F:\hf_cache\datasets"

# --- Default quantization scheme ---
# (3,5,8) nested bitmap: 3-bit(|v|<=3, ~55%) + 5-bit(|v|<=15, ~40%) + 8-bit(rest, ~4%)
DEFAULT_LEVELS = (3, 5, 8)

# int8 magnitude threshold covered by `b` bits: |v| <= 2^b - 1 ... but sign-aware
# the meaningful signed range for b bits is [-(2^(b-1)-1), 2^(b-1)-1]; we pack
# unsigned offset so the threshold is (2^(b-1)-1) for sign + magnitude encoding.
# In the (3,5,8) scheme we store abs-thresholds: 3bit->|v|<=3, 5bit->|v|<=15, 8bit->|v|<=127
BIT_TO_THRESHOLD = {
    1: 0,
    2: 1,
    3: 3,
    4: 7,
    5: 15,
    6: 31,
    7: 63,
    8: 127,
}

# Layers to skip during quantization (kept in bf16 / native dtype)
SKIP_PATTERNS = ("lm_head", "embed", "shared", "wte", "wte_emb")

# Module classes considered a quantizable Linear (name guard + size guard)
MIN_LINEAR_ELEMS = 1000

# Device
import torch

CUDA = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
CPU = torch.device("cpu")

# Triton decode block size (elements per program)
TRITON_BLOCK = 1024
