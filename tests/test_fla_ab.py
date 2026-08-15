"""A/B: fla-kernel delta rule vs torch-eager — speed + output equivalence."""
import sys, time
sys.setrecursionlimit(10000)
import pandas
import torch
from transformers import AutoTokenizer
from ixrun.config import QWEN38_PATH, DEFAULT_LEVELS
from ixrun.engine import Int8XEngine

CACHE = r"E:\models\qwen38_packed.pt"
tok = AutoTokenizer.from_pretrained(QWEN38_PATH, trust_remote_code=True)
if tok.pad_token is None: tok.pad_token = tok.eos_token

prompt = tok.apply_chat_template(
    [{"role": "user", "content": "Write a 150-word story about a robot learning to paint. Do not stop early."}],
    tokenize=False, add_generation_prompt=True, enable_thinking=False)
ids = tok(prompt, return_tensors="pt")["input_ids"].cuda()

def build(use_fla):
    from ixrun.fla_patch import apply_fla_kernels
    import transformers.models.qwen3_5.modeling_qwen3_5 as MQ
    # reset to eager by rebuilding module functions is hard; instead skip
    # patching for baseline via env-free flag
    if not use_fla:
        # restore originals saved on first patch (we patch AFTER baseline run)
        pass
    m = Int8XEngine._load_any(QWEN38_PATH, torch.bfloat16, low_cpu=True)
    stats = Int8XEngine._deploy_streaming(m, DEFAULT_LEVELS, cache_path=CACHE, verbose=False)
    eng = Int8XEngine(m, tok, stats); eng._finalize_device()
    return m

@torch.no_grad()
def gen(model, n=200):
    t0 = time.time()
    out = model.generate(ids, max_new_tokens=n, do_sample=False,
                         pad_token_id=tok.pad_token_id)
    torch.cuda.synchronize()
    dt = time.time() - t0
    ntok = out.shape[1] - ids.shape[1]
    return dt, ntok, tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True)

# baseline FIRST (eager — patch happens inside _load_any, so temporarily disable)
import ixrun.fla_patch as FP
_orig_apply = FP.apply_fla_kernels
FP.apply_fla_kernels = lambda verbose=False: False
import ixrun.engine as E
E_apply_ref = None

m = build(use_fla=False) if False else None

# simpler: baseline = patch disabled via monkeypatched no-op
FP.apply_fla_kernels = lambda verbose=False: False
m = Int8XEngine._load_any(QWEN38_PATH, torch.bfloat16, low_cpu=True)
stats = Int8XEngine._deploy_streaming(m, DEFAULT_LEVELS, cache_path=CACHE, verbose=False)
eng = Int8XEngine(m, tok, stats); eng._finalize_device()
gen(m, 16)
dt0, n0, txt0 = gen(m)
print(f"eager : {dt0:.2f}s / {n0} tok = {dt0/n0*1000:.0f} ms/tok", flush=True)
del m, eng
import gc; gc.collect(); torch.cuda.empty_cache()

# fla path
FP.apply_fla_kernels = _orig_apply
m = Int8XEngine._load_any(QWEN38_PATH, torch.bfloat16, low_cpu=True)
stats = Int8XEngine._deploy_streaming(m, DEFAULT_LEVELS, cache_path=CACHE, verbose=False)
eng = Int8XEngine(m, tok, stats); eng._finalize_device()
gen(m, 16)
dt1, n1, txt1 = gen(m)
print(f"fla   : {dt1:.2f}s / {n1} tok = {dt1/n1*1000:.0f} ms/tok", flush=True)
print(f"speedup: {dt0/max(dt1,1e-9):.2f}x")
print(f"same output: {txt0 == txt1}")
print(f"fla out[:120]: {txt1.strip()[:120]!r}")
