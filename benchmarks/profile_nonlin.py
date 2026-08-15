"""Measure non-linear cost: attention + norms + python/launch overhead.

Replaces every StreamingLinear forward with a zero-returning stub (same
shape), so the timed forward = everything EXCEPT our linear layers.
"""
import sys, time
sys.setrecursionlimit(10000)
import pandas
import torch
from transformers import AutoTokenizer
from ixrun.config import QWEN38_PATH, DEFAULT_LEVELS
from ixrun.engine import Int8XEngine, StreamingLinear

CACHE = r"E:\models\qwen38_packed.pt"
tok = AutoTokenizer.from_pretrained(QWEN38_PATH, trust_remote_code=True)
m = Int8XEngine._load_any(QWEN38_PATH, torch.bfloat16, low_cpu=True)
stats = Int8XEngine._deploy_streaming(m, DEFAULT_LEVELS, cache_path=CACHE, verbose=False)
eng = Int8XEngine(m, tok, stats); eng._finalize_device()
m.eval()

prompt = tok.apply_chat_template([{"role": "user", "content": "hi"}], tokenize=False,
                                 add_generation_prompt=True, enable_thinking=False)
ids = tok(prompt, return_tensors="pt")["input_ids"].cuda()

@torch.no_grad()
def step_time(warm=4, rep=10):
    out = m(ids, use_cache=True)
    past = out.past_key_values
    nxt = ids[:, -1:]
    for _ in range(warm): m(nxt, past_key_values=past, use_cache=True)
    torch.cuda.synchronize(); t0 = time.time()
    for _ in range(rep): m(nxt, past_key_values=past, use_cache=True)
    torch.cuda.synchronize()
    return (time.time()-t0)/rep*1000

t_real = step_time()

# stub out all StreamingLinears
orig_fwd = {}
for name, sl in m.named_modules():
    if isinstance(sl, StreamingLinear):
        orig_fwd[name] = sl.forward
        of, inf_ = sl.out_features, sl.in_features
        def make_zero(of=of):
            def f(x):
                return x.new_zeros(x.shape[:-1] + (of,))
            return f
        sl.forward = make_zero(of)
t_nonlin = step_time()

# restore
for name, sl in m.named_modules():
    if isinstance(sl, StreamingLinear):
        sl.forward = orig_fwd[name]

print(f"full step          : {t_real:.1f} ms")
print(f"non-linear (stub)  : {t_nonlin:.1f} ms  <- attention+norms+overhead")
print(f"linear (difference): {t_real - t_nonlin:.1f} ms")
