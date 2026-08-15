"""Qwen3.8-27B fused vs old: correct token counting + long output."""
import sys, time
sys.setrecursionlimit(10000)
import pandas
import torch
from transformers import AutoTokenizer
from ixrun.config import QWEN38_PATH, DEFAULT_LEVELS
from ixrun.engine import Int8XEngine, StreamingLinear

CACHE = r"E:\models\qwen38_packed.pt"
tok = AutoTokenizer.from_pretrained(QWEN38_PATH, trust_remote_code=True)
if tok.pad_token is None: tok.pad_token = tok.eos_token

m = Int8XEngine._load_any(QWEN38_PATH, torch.bfloat16, low_cpu=True)
stats = Int8XEngine._deploy_streaming(m, DEFAULT_LEVELS, cache_path=CACHE, verbose=True)
eng = Int8XEngine(m, tok, stats); eng._finalize_device()

msgs = [{"role": "user",
         "content": "Write a 150-word story about a robot learning to paint. Do not stop early."}]
prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                 enable_thinking=False)
ids = tok(prompt, return_tensors="pt")["input_ids"].cuda()
N = 200

@torch.no_grad()
def gen_time(n=N):
    t0 = time.time()
    out = m.generate(ids, max_new_tokens=n, do_sample=False,
                     pad_token_id=tok.pad_token_id)
    torch.cuda.synchronize()
    dt = time.time() - t0
    ntok = out.shape[1] - ids.shape[1]
    return dt, ntok, tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True)

gen_time(16)  # warmup/JIT
dt_a, na, txt_a = gen_time()
print(f"fused   : {dt_a:.2f}s / {na} tok = {dt_a/na*1000:.0f} ms/tok", flush=True)

for _, sl in m.named_modules():
    if isinstance(sl, StreamingLinear):
        sl._use_fused = False
gen_time(16)
dt_b, nb, txt_b = gen_time()
print(f"decode+ : {dt_b:.2f}s / {nb} tok = {dt_b/nb*1000:.0f} ms/tok")
print(f"speedup: {dt_b/max(dt_a,1e-9):.2f}x   prefix-match: {txt_a[:80] == txt_b[:80]}")
print(f"out[:140]: {txt_a.strip()[:140]!r}")
