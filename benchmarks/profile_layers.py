"""Per-module timing of one decode step via CUDA events on each decoder layer
and its attention submodule."""
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

# hook every submodule of each decoder layer: capture per-call GPU time
from collections import defaultdict
timings = defaultdict(float)
events = {}

def make_pre(name):
    e = torch.cuda.Event(enable_timing=True)
    def pre(mod, inp):
        e.record()
    return pre, e, name

hooks = []
layers_module = m.model.layers if hasattr(m.model, "layers") else m.model.language_model.layers
for li, layer in enumerate(layers_module):
    for sub_name, sub in [("self_attn", None), ]:
        pass
    for child_name, child in layer.named_children():
        if child_name in ("input_layernorm", "post_attention_layernorm"):
            continue
        full = f"L{li}.{child_name}"
        pre, e, nm = make_pre(full)
        post_e = torch.cuda.Event(enable_timing=True)
        def make_post(e_start, e_end, nm):
            def post(mod, inp, outp):
                e_end.record()
                timings[nm] += torch.cuda.Event.elapsed_time  # can't call here
            return post
        # simpler: store events, sync at end — use closure list
        hooks.append((child, pre, post_e, full))

# simpler approach: manual event pairs recorded via forward pre-hooks only
recs = []  # (start_ev, end_ev, name)
def pre_hook(name):
    def f(mod, inp):
        s = torch.cuda.Event(enable_timing=True); s.record()
        recs.append((s, None, name))
    return f
def post_hook(name):
    def f(mod, inp, outp):
        s, _, nm = recs[-1]
        e = torch.cuda.Event(enable_timing=True); e.record()
        recs[-1] = (s, e, nm)
    return f

for li, layer in enumerate(layers_module):
    for child_name, child in layer.named_children():
        if "norm" in child_name.lower():
            continue
        child.register_forward_pre_hook(pre_hook(f"L{li}.{child_name}"))
        child.register_forward_hook(post_hook(f"L{li}.{child_name}"))

with torch.no_grad():
    torch.cuda.synchronize()
    for _ in range(10):
        recs.clear()
        m(nxt, past_key_values=past, use_cache=True)
    torch.cuda.synchronize()
    # accumulate last run only
agg = defaultdict(float)
for s, e, nm in recs:
    if e is not None:
        agg[nm] += s.elapsed_time(e)

cat = defaultdict(float)
cnt = defaultdict(int)
for nm, t in agg.items():
    base = nm.split(".", 1)[1]
    cat[base] += t
    cnt[base] += 1
print("\nper-call-type totals over one decode step (ms):")
for base, t in sorted(cat.items(), key=lambda kv: -kv[1]):
    print(f"  {base:<28} {t:7.2f} ms  ({cnt[base]} calls, {t/max(cnt[base],1):.3f} avg)")
print(f"  SUM: {sum(cat.values()):.2f} ms")
