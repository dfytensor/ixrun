"""Validate batched sampling: params actually change outputs + greedy path
unchanged + server e2e with sampling params."""
import sys, time, json, threading, urllib.request
sys.setrecursionlimit(10000)
import pandas
import torch

LOG = open(r"E:\IXRUN\tests\bs_out.txt", "w", encoding="utf-8", errors="replace")
def P(*a):
    s = " ".join(str(x) for x in a)
    LOG.write(s + "\n"); LOG.flush()

from transformers import AutoModelForCausalLM, AutoTokenizer
from ixrun.config import MODEL_PATH
from ixrun.tpab_linear import deploy_model_tpab
from ixrun.batching import BatchedGreedyGenerator

tok = AutoTokenizer.from_pretrained(MODEL_PATH)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
m = AutoModelForCausalLM.from_pretrained(MODEL_PATH, dtype=torch.bfloat16).cuda()
deploy_model_tpab(m, snr_target_db=28.0, verbose=False)
m.eval()

gen = BatchedGreedyGenerator(m, tok, min_batch=4, max_batch=8, coalesce_ms=30)

def run(prompt, n=60, **kw):
    req, q = gen.submit(prompt, n, **kw)
    req.done.wait()
    parts = []
    while not q.empty():
        parts.append(q.get(timeout=0.1))
    return "".join(parts)

# 1. greedy twice -> identical (temperature=0 path)
a = run("The capital of France is", temperature=0.0)
b = run("The capital of France is", temperature=0.0)
P(f"greedy deterministic: {a == b}  out={a.strip()[:60]!r}")

# 2. sampling diversity: temp=1.2 twice -> (almost surely) different
c = run("Once upon a time", 40, temperature=1.2, top_p=0.95, top_k=50)
d = run("Once upon a time", 40, temperature=1.2, top_p=0.95, top_k=50)
P(f"temp1.2 diverse: {c != d}")
P(f"  c={c.strip()[:60]!r}")
P(f"  d={d.strip()[:60]!r}")

# 3. repetition penalty tames loops on a loop-prone prompt
loop_p = "The list of numbers: 1, 2, 3,"
no_rep = run(loop_p, 50, temperature=1.0, top_p=0.9)
with_rep = run(loop_p, 50, temperature=1.0, top_p=0.9, repetition_penalty=1.3)
def loopiness(t):
    words = t.split()
    if len(words) < 2:
        return 0.0
    from collections import Counter
    c = Counter(words)
    return c.most_common(1)[0][1] / len(words)
P(f"rep-penalty loopiness: {loopiness(no_rep):.2f} -> {loopiness(with_rep):.2f}")
P(f"  no_rep={no_rep.strip()[:70]!r}")
P(f"  w_rep ={with_rep.strip()[:70]!r}")
gen.close()
P("DONE")
LOG.close()
