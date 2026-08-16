"""Profile inside Qwen3_5GatedDeltaNet: which ops make up the 0.78ms/layer?

Uses torch profiler on a single decode step, aggregated by op name, for the
linear-attention layers only.
"""
import sys, time
sys.setrecursionlimit(10000)
import pandas
import torch
from transformers import AutoTokenizer
from ixrun.config import QWEN38_PATH, DEFAULT_LEVELS
from ixrun.engine import Int8XEngine

tok = AutoTokenizer.from_pretrained(QWEN38_PATH, trust_remote_code=True)
m = Int8XEngine._load_any(QWEN38_PATH, torch.bfloat16, low_cpu=True)
stats = Int8XEngine._deploy_streaming(m, DEFAULT_LEVELS, cache_path=r"E:\models\qwen38_packed.pt", verbose=False)
eng = Int8XEngine(m, tok, stats); eng._finalize_device(); m.eval()

prompt = tok.apply_chat_template([{"role": "user", "content": "hi"}], tokenize=False,
                                 add_generation_prompt=True, enable_thinking=False)
ids = tok(prompt, return_tensors="pt")["input_ids"].cuda()
with torch.no_grad():
    out = m(ids, use_cache=True)
    past = out.past_key_values
    nxt = ids[:, -1:]
    for _ in range(3): m(nxt, past_key_values=past, use_cache=True)

from torch.profiler import profile, ProfilerActivity
with torch.no_grad():
    with profile(activities=[ProfilerActivity.CUDA], record_shapes=False) as prof:
        for _ in range(5):
            m(nxt, past_key_values=past, use_cache=True)
    torch.cuda.synchronize()

# aggregate cuda kernel time by name
agg = {}
for evt in prof.key_averages():
    if evt.device_type == torch.autograd.DeviceType.CUDA or evt.self_device_time_total > 0:
        agg[evt.key] = (evt.self_device_time_total, evt.count)

total = sum(t for t, _ in agg.values())
print(f"total CUDA time / 5 steps = {total/1000:.1f}ms  ({total/5/1000:.2f} ms/step)")
rows = sorted(agg.items(), key=lambda kv: -kv[1][0])
for name, (t, c) in rows[:28]:
    print(f"{t/5/1000:8.4f} ms/step  x{c//5:4d}  {name[:88]}")
