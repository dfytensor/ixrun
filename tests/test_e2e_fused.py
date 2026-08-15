"""End-to-end: fused GEMV generation speed + output sanity on MiniCPM5."""
import sys, time, gc
sys.setrecursionlimit(10000)
import pandas
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from ixrun.config import MODEL_PATH, DEFAULT_LEVELS
from ixrun.engine import Int8XEngine, StreamingLinear

tok = AutoTokenizer.from_pretrained(MODEL_PATH)
if tok.pad_token is None: tok.pad_token = tok.eos_token

def build():
    m = AutoModelForCausalLM.from_pretrained(MODEL_PATH, dtype=torch.bfloat16).cuda()
    Int8XEngine._deploy_streaming(m, DEFAULT_LEVELS, verbose=False)
    m.eval()
    return m

# fused on/off toggle for A/B
m = build()
n_fused = sum(1 for _ in m.named_modules() if isinstance(_[1], StreamingLinear) and _[1]._use_fused)
print(f"layers with fused path: {n_fused}")

ids = tok("The theory of relativity states that", return_tensors="pt")["input_ids"].cuda()
prompt = "The theory of relativity states that"

@torch.no_grad()
def gen_time(model, n=40):
    t0 = time.time()
    out = model.generate(
        ids, max_new_tokens=n, do_sample=False,
        pad_token_id=tok.pad_token_id,
    )
    torch.cuda.synchronize()
    dt = time.time() - t0
    txt = tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True)
    return dt, txt

# --- A: fused enabled (new) — warmup first (Triton JIT compile) ---
gen_time(m, n=5)
dt_a, txt_a = gen_time(m)
print(f"fused   : {dt_a:.2f}s for 40 tok = {dt_a/40*1000:.0f} ms/tok")
print(f"  out: {txt_a[:90]!r}")

# --- B: fused disabled (previous behavior) — warmup too ---
for _, sl in m.named_modules():
    if isinstance(sl, StreamingLinear):
        sl._use_fused = False
gen_time(m, n=5)
dt_b, txt_b = gen_time(m)
print(f"decode+ : {dt_b:.2f}s for 40 tok = {dt_b/40*1000:.0f} ms/tok")
print(f"  out: {txt_b[:90]!r}")
print(f"\nspeedup: {dt_b/dt_a:.2f}x   same output: {txt_a == txt_b}")
