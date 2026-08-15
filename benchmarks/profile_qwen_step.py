"""Deep profile of Qwen3.8-27B single-token decode step (per-component)."""
import sys, time
sys.setrecursionlimit(10000)
import pandas
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from ixrun.config import QWEN38_PATH, DEFAULT_LEVELS
from ixrun.engine import Int8XEngine, StreamingLinear
from ixrun.fused import fused_gemv

CACHE = r"E:\models\qwen38_packed.pt"
tok = AutoTokenizer.from_pretrained(QWEN38_PATH, trust_remote_code=True)
m = Int8XEngine._load_any(QWEN38_PATH, torch.bfloat16, low_cpu=True)
stats = Int8XEngine._deploy_streaming(m, DEFAULT_LEVELS, cache_path=CACHE, verbose=False)
eng = Int8XEngine(m, tok, stats); eng._finalize_device()
m.eval()

lins = [(n, s) for n, s in m.named_modules() if isinstance(s, StreamingLinear)]
fused = [(n, s) for n, s in lins if s._use_fused]
unfused = [(n, s) for n, s in lins if not s._use_fused]
print(f"total={len(lins)} fused={len(fused)} unfused={len(unfused)}")
for n, s in unfused[:10]:
    print(f"  unfused: {n} {s.out_features}x{s.in_features}")

# --- time fused GEMV kernels only (decode step shapes) ---
xs = {}
for n, s in fused:
    xs.setdefault((s.out_features, s.in_features),
                  torch.randn(s.in_features, dtype=torch.bfloat16, device="cuda"))
torch.cuda.synchronize(); t0 = time.time()
REP = 20
for _ in range(REP):
    for n, s in fused:
        fused_gemv(xs[(s.out_features, s.in_features)], s._b1, s._b2, s._l1, s._l2,
                   s._l3, s._q1, s._q2, s._scale, s.out_features, s.in_features)
torch.cuda.synchronize()
t_gemv = (time.time()-t0)/REP*1000

# --- time unfused layers (decode + linear, batched input) ---
xb = {}
for n, s in unfused:
    xb.setdefault((s.out_features, s.in_features),
                  torch.randn(1, 1, s.in_features, dtype=torch.bfloat16, device="cuda"))
for n, s in unfused: s._decode()
torch.cuda.synchronize(); t0 = time.time()
for _ in range(REP):
    for n, s in unfused:
        s(xb[(s.out_features, s.in_features)])
torch.cuda.synchronize()
t_unf = (time.time()-t0)/REP*1000

# --- full decode step ---
prompt = tok.apply_chat_template([{"role":"user","content":"hi"}], tokenize=False,
                                 add_generation_prompt=True, enable_thinking=False)
ids = tok(prompt, return_tensors="pt")["input_ids"].cuda()
with torch.no_grad():
    out = m(ids, use_cache=True)
    past = out.past_key_values
    nxt = ids[:, -1:]
    for _ in range(3): m(nxt, past_key_values=past, use_cache=True)
    torch.cuda.synchronize(); t0 = time.time()
    for _ in range(10): m(nxt, past_key_values=past, use_cache=True)
    torch.cuda.synchronize()
    t_step = (time.time()-t0)/10*1000

# --- attention-only estimate: step - gemv - unfused ---
t_attn_other = t_step - t_gemv - t_unf
print(f"\ndecode step total     : {t_step:7.1f} ms")
print(f"  fused GEMV kernels  : {t_gemv:7.1f} ms  ({len(fused)} layers)")
print(f"  unfused linears     : {t_unf:7.1f} ms  ({len(unfused)} layers)")
print(f"  attn+norms+overhead : {t_attn_other:7.1f} ms")

# breakdown of fused by shape
from collections import Counter, defaultdict
shape_time = defaultdict(float)
for (of, inf_), x in xs.items():
    s = next(s for n, s in fused if (s.out_features, s.in_features) == (of, inf_))
    torch.cuda.synchronize(); t0 = time.time()
    for _ in range(50):
        fused_gemv(x, s._b1, s._b2, s._l1, s._l2, s._l3, s._q1, s._q2, s._scale, of, inf_)
    torch.cuda.synchronize()
    shape_time[(of, inf_)] = (time.time()-t0)/50*1000
cnt = Counter((s.out_features, s.in_features) for n, s in fused)
print("\nfused shapes (per-token cost = n_layers x per-call):")
for (of, inf_), c in cnt.most_common(12):
    print(f"  {of}x{inf_} x{c:3d}: {shape_time[(of,inf_)]*c:7.2f} ms total")
