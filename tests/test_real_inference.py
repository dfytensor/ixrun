"""MiniCPM5 real-inference validation: bf16 vs INT8-X vs TPAB@28 on
multiple practical tasks — factual QA, arithmetic, code, translation,
multi-turn coherence, long-form structure. Reports per-task match and
side-by-side outputs.
"""
import sys, gc
sys.setrecursionlimit(10000)
import pandas
import torch

LOG = open(r"E:\IXRUN\tests\real_out.txt", "w", encoding="utf-8", errors="replace")
def P(*a):
    s = " ".join(str(x) for x in a)
    LOG.write(s + "\n"); LOG.flush()

from transformers import AutoModelForCausalLM, AutoTokenizer
from ixrun.config import MODEL_PATH, DEFAULT_LEVELS
from ixrun.linear import deploy_model
from ixrun.tpab_linear import deploy_model_tpab

tok = AutoTokenizer.from_pretrained(MODEL_PATH)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

TASKS = [
    ("fact-1",  "The capital of France is", 24, "Paris"),
    ("fact-2",  "The chemical formula of water is", 16, "H2O"),
    ("fact-3",  "The largest planet in the solar system is", 16, "Jupiter"),
    ("arith-1", "Calculate: 12 + 27 =", 16, "39"),
    ("arith-2", "Calculate: 9 times 6 =", 12, "54"),
    ("code-1",  "In Python, to print a message you write: print(", 24, None),
    ("trans-1", "The English word 'cat' translates to French as", 12, None),
    ("logic-1", "If all birds can fly and a penguin is a bird, then", 30, None),
    ("story-1", "Once upon a time in a small village,", 60, None),
    ("struct-1", "The three primary colors are", 20, "red"),
]

@torch.no_grad()
def gen(model, prompt, n):
    ids = tok(prompt, return_tensors="pt")["input_ids"].cuda()
    out = model.generate(ids, max_new_tokens=n, do_sample=False,
                         pad_token_id=tok.pad_token_id)
    return tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True).strip()

def build(kind):
    gc.collect(); torch.cuda.empty_cache()
    m = AutoModelForCausalLM.from_pretrained(MODEL_PATH, dtype=torch.bfloat16).cuda()
    if kind == "int8x":
        deploy_model(m, level_bits=DEFAULT_LEVELS, cache="none", verbose=False)
    elif kind == "tpab":
        deploy_model_tpab(m, snr_target_db=28.0, verbose=False)
    m.eval()
    return m

results = {}
for kind in ("bf16", "int8x", "tpab"):
    m = build(kind)
    outs = {}
    for name, prompt, n, _ in TASKS:
        outs[name] = gen(m, prompt, n)
    results[kind] = outs
    del m; gc.collect(); torch.cuda.empty_cache()

# ---- scoring ----
P(f"{'task':<9} {'bf16==tpab':<10} {'bf16==int8x':<11} expected")
same_tp = same_ix = 0
for name, prompt, n, expect in TASKS:
    b, t, x = results["bf16"][name], results["tpab"][name], results["int8x"][name]
    m_t = b == t
    m_x = b == x
    same_tp += m_t; same_ix += m_x
    hit = ""
    if expect:
        hit = " HIT" if expect.lower() in b.lower() else " MISS(bf16)"
    P(f"{name:<9} {str(m_t):<10} {str(m_x):<11}{hit}")

P(f"\nexact-match totals: tpab {same_tp}/{len(TASKS)}  int8x {same_ix}/{len(TASKS)}")
P("\n---- side by side (bf16 | tpab | int8x) ----")
for name, prompt, n, _ in TASKS:
    P(f"\n[{name}] {prompt!r}")
    P(f"  bf16 : {results['bf16'][name][:110]!r}")
    P(f"  tpab : {results['tpab'][name][:110]!r}")
    P(f"  i8x  : {results['int8x'][name][:110]!r}")
P("DONE")
LOG.close()
