"""Test MTP head + speculative generation on Qwen3.8-27B."""
import sys, time
sys.setrecursionlimit(10000)
import pandas
import torch
from transformers import AutoTokenizer
from ixrun.config import QWEN38_PATH, DEFAULT_LEVELS
from ixrun.engine import Int8XEngine
from ixrun.mtp import build_mtp_head, spec_generate

CACHE = r"E:\models\qwen38_packed.pt"
tok = AutoTokenizer.from_pretrained(QWEN38_PATH, trust_remote_code=True)
if tok.pad_token is None: tok.pad_token = tok.eos_token

m = Int8XEngine._load_any(QWEN38_PATH, torch.bfloat16, low_cpu=True)
stats = Int8XEngine._deploy_streaming(m, DEFAULT_LEVELS, cache_path=CACHE, verbose=False)
eng = Int8XEngine(m, tok, stats); eng._finalize_device()
m.eval()

mtp = build_mtp_head(m, QWEN38_PATH, verbose=True)
assert mtp is not None

prompt = tok.apply_chat_template(
    [{"role": "user", "content": "Write a 150-word story about a robot learning to paint. Do not stop early."}],
    tokenize=False, add_generation_prompt=True, enable_thinking=False)
ids = tok(prompt, return_tensors="pt")["input_ids"].cuda()

# --- baseline greedy ---
@torch.no_grad()
def gen_baseline(n=200):
    t0 = time.time()
    out = m.generate(ids, max_new_tokens=n, do_sample=False, pad_token_id=tok.pad_token_id)
    torch.cuda.synchronize()
    return time.time()-t0, out[0][ids.shape[1]:], out.shape[1]-ids.shape[1]

m.generate(ids, max_new_tokens=16, pad_token_id=tok.pad_token_id)
t0, base_ids, n0 = gen_baseline()
print(f"baseline: {t0:.1f}s / {n0} tok = {t0/n0*1000:.0f} ms/tok", flush=True)

# --- speculative ---
def gen_spec(n=200):
    t0 = time.time()
    out = spec_generate(m, tok, ids, mtp, max_new_tokens=n, pad_token_id=tok.pad_token_id)
    torch.cuda.synchronize()
    return time.time()-t0, out[0], getattr(out, "_mtp_stats", {})

gen_spec(16)
t1, spec_ids, mtp_stats = gen_spec()
print(f"spec    : {t1:.1f}s / {spec_ids.numel()} tok = {t1/spec_ids.numel()*1000:.0f} ms/tok")
print(f"mtp stats: {mtp_stats}")

base_txt = tok.decode(base_ids, skip_special_tokens=True)
spec_txt = tok.decode(spec_ids, skip_special_tokens=True)
print(f"prefix-match(80ch): {base_txt[:80] == spec_txt[:80]}")
print(f"base[:100]: {base_txt.strip()[:100]!r}")
print(f"spec[:100]: {spec_txt.strip()[:100]!r}")
