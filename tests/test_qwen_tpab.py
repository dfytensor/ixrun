"""Qwen3.8-27B: TPAB backend vs INT8-X streaming — VRAM / speed / quality.

Deploys TPAB on the CPU-side lazy-loaded model layer-by-layer (same pattern
as Int8XEngine._deploy_streaming), then runs generation A/B against the
INT8-X packed cache.
"""
import sys, time, gc, os
sys.setrecursionlimit(10000)
import pandas
import torch
from transformers import AutoTokenizer
from ixrun.config import QWEN38_PATH, DEFAULT_LEVELS
from ixrun.engine import Int8XEngine, StreamingLinear
from ixrun.linear import iter_quantizable_linears, _set_parent_child
from ixrun.tpab_linear import TpabLinear
from ixrun.hybrid import deploy_model_hybrid
from ixrun.generate import generate_text

LOG = open(r"E:\IXRUN\tests\qwen_tpab_out.txt", "w", encoding="utf-8", errors="replace")
def P(*a):
    s = " ".join(str(x) for x in a)
    LOG.write(s + "\n"); LOG.flush(); print(s, flush=True)

SNR_DB = float(os.environ.get("TPAB_SNR", "24.0"))
tok = AutoTokenizer.from_pretrained(QWEN38_PATH, trust_remote_code=True)
if tok.pad_token is None: tok.pad_token = tok.eos_token

prompt = tok.apply_chat_template(
    [{"role": "user",
      "content": "Write a 150-word story about a robot learning to paint. Do not stop early."}],
    tokenize=False, add_generation_prompt=True, enable_thinking=False)

@torch.no_grad()
def gen(model, n=150, warm=12):
    ids = tok(prompt, return_tensors="pt")["input_ids"].cuda()
    model.generate(ids, max_new_tokens=warm, do_sample=False, pad_token_id=tok.pad_token_id)
    torch.cuda.synchronize()
    t0 = time.time()
    out = model.generate(ids, max_new_tokens=n, do_sample=False, pad_token_id=tok.pad_token_id)
    torch.cuda.synchronize()
    dt = time.time() - t0
    return dt, out.shape[1] - ids.shape[1], tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True)

# ---------------- HYBRID deploy (lazy CPU load, per-layer encode) ----------------
P(f"=== TPAB@{SNR_DB:.0f}dB + split-K deploy on Qwen3.8-27B (all-TPAB) ===")
t0 = time.time()
m = Int8XEngine._load_any(QWEN38_PATH, torch.bfloat16, low_cpu=True)
P(f"load: {time.time()-t0:.0f}s")

from ixrun.tpab_linear import deploy_model_tpab as _dep
stats = _dep(m, snr_target_db=SNR_DB, verbose=True, lazy=True)
gc.collect(); torch.cuda.empty_cache()

eng = Int8XEngine(m, tok, stats)
eng._finalize_device()
P(f"VRAM alloc={torch.cuda.memory_allocated()/1e9:.2f}GB "
  f"peak={torch.cuda.max_memory_allocated()/1e9:.2f}GB")

dt, ntok, txt_t = gen(m)
P(f"\nTPAB-SK: {dt:.1f}s / {ntok} tok = {dt/ntok*1000:.0f} ms/tok")
P(f"  out: {txt_t.strip()[:110]!r}")
P(f"  VRAM peak: {torch.cuda.max_memory_allocated()/1e9:.2f}GB")
del m, eng; gc.collect(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()

# ---------------- INT8-X streaming reference ----------------
P("\n=== INT8-X streaming reference ===")
eng2 = Int8XEngine.from_pretrained(QWEN38_PATH, mode="streaming",
                                   cache_path=r"E:\models\qwen38_packed.pt", verbose=False)
m2 = eng2.model
dt2, ntok2, txt_x = gen(m2)
P(f"INT8-X : {dt2:.1f}s / {ntok2} tok = {dt2/ntok2*1000:.0f} ms/tok")
P(f"  out: {txt_x.strip()[:110]!r}")
P(f"  VRAM peak: {torch.cuda.max_memory_allocated()/1e9:.2f}GB")
P(f"\nspeedup: {dt2/max(dt,1e-9):.2f}x")
P(f"prefix-match: {txt_t[:80] == txt_x[:80]}")
P("DONE")
LOG.close()
