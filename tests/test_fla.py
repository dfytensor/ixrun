"""Try enabling fla kernels for Qwen3.5 linear attention + measure decode step."""
import sys, time
sys.setrecursionlimit(10000)
import pandas
import torch
from transformers import AutoTokenizer
from ixrun.config import QWEN38_PATH, DEFAULT_LEVELS
from ixrun.engine import Int8XEngine

CACHE = r"E:\models\qwen38_packed.pt"
tok = AutoTokenizer.from_pretrained(QWEN38_PATH, trust_remote_code=True)

import fla  # noqa
print("fla version:", getattr(fla, "__version__", "?"), flush=True)

m = Int8XEngine._load_any(QWEN38_PATH, torch.bfloat16, low_cpu=True)
stats = Int8XEngine._deploy_streaming(m, DEFAULT_LEVELS, cache_path=CACHE, verbose=False)
eng = Int8XEngine(m, tok, stats); eng._finalize_device()
m.eval()

# check kernel decorators picked up
layer0 = m.model.layers[0] if hasattr(m.model, "layers") else m.model.language_model.layers[0]
print("layer0 attn types:", [type(v).__name__ for k, v in layer0.named_children()])

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

t = step_time()
print(f"decode step with fla available: {t:.1f} ms/tok", flush=True)

# generation sanity
out = m.generate(ids, max_new_tokens=32, do_sample=False, pad_token_id=tok.eos_token_id)
print("gen:", tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True)[:120])
