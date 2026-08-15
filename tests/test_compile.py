"""Try torch.compile on GatedDeltaNet / MLP / Attention modules (glue fusion)."""
import sys, time
sys.setrecursionlimit(10000)
import pandas
import torch
from transformers import AutoTokenizer
from ixrun.config import QWEN38_PATH, DEFAULT_LEVELS
from ixrun.engine import Int8XEngine

tok = AutoTokenizer.from_pretrained(QWEN38_PATH, trust_remote_code=True)
if tok.pad_token is None: tok.pad_token = tok.eos_token
m = Int8XEngine._load_any(QWEN38_PATH, torch.bfloat16, low_cpu=True)
stats = Int8XEngine._deploy_streaming(m, DEFAULT_LEVELS, cache_path=r"E:\models\qwen38_packed.pt", verbose=False)
eng = Int8XEngine(m, tok, stats); eng._finalize_device(); m.eval()

prompt = tok.apply_chat_template(
    [{"role": "user", "content": "Write a 150-word story about a robot learning to paint. Do not stop early."}],
    tokenize=False, add_generation_prompt=True, enable_thinking=False)
ids = tok(prompt, return_tensors="pt")["input_ids"].cuda()

@torch.no_grad()
def gen(n=200):
    t0 = time.time()
    out = m.generate(ids, max_new_tokens=n, do_sample=False, pad_token_id=tok.pad_token_id)
    torch.cuda.synchronize()
    dt = time.time()-t0
    return dt, out.shape[1]-ids.shape[1], tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True)

gen(16)
dt0, n0, txt0 = gen()
print(f"before compile: {dt0:.1f}s / {n0} = {dt0/n0*1000:.0f} ms/tok", flush=True)

torch._dynamo.config.suppress_errors = True
import torch._dynamo as dyn
dyn.reset()
layers = m.model.layers if hasattr(m.model, "layers") else m.model.language_model.layers
n_comp = 0
for layer in layers:
    for cname, child in layer.named_children():
        if "norm" in cname.lower():
            continue
        try:
            child.forward = torch.compile(child.forward, dynamic=False)
            n_comp += 1
        except Exception:
            pass
print(f"compiled {n_comp} modules", flush=True)

try:
    gen(16)
    dt1, n1, txt1 = gen()
    print(f"after compile : {dt1:.1f}s / {n1} = {dt1/n1*1000:.0f} ms/tok")
    print(f"speedup {dt0/max(dt1,1e-9):.2f}x  prefix-match {txt0[:80]==txt1[:80]}")
except Exception as e:
    print("compile path failed:", type(e).__name__, str(e)[:200])
