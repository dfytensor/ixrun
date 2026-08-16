"""Validate BatchedGreedyGenerator: batched (B=8) vs serial (B=1) outputs
on MiniCPM5 TPAB — correctness + throughput."""
import sys, time, threading
sys.setrecursionlimit(10000)
import pandas
import torch

LOG = open(r"E:\IXRUN\tests\cb_out.txt", "w", encoding="utf-8", errors="replace")
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
P("model ready (TPAB@28)")

PROMPTS = [
    "The theory of relativity states that",
    "The capital of France is",
    "Once upon a time in a distant land,",
    "Machine learning is a field of",
    "The three laws of robotics were",
    "In the beginning of the universe,",
    "Water boils at a temperature of",
    "The fastest land animal is the",
]
N = 40

# --- serial reference (B=1 each) ---
t0 = time.time()
serial_texts = []
for p in PROMPTS:
    ids = tok(p, return_tensors="pt")["input_ids"].cuda()
    out = m.generate(ids, max_new_tokens=N, do_sample=False,
                     pad_token_id=tok.pad_token_id)
    serial_texts.append(tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True))
torch.cuda.synchronize()
t_serial = time.time() - t0
P(f"serial  : {t_serial:.1f}s for {len(PROMPTS)} reqs "
  f"({t_serial/len(PROMPTS)/N*1000:.0f} ms/tok, "
  f"{len(PROMPTS)*N/t_serial:.1f} tok/s total)")

# --- batched B=8 ---
gen = BatchedGreedyGenerator(m, tok, min_batch=8, max_batch=8, coalesce_ms=0)
t0 = time.time()
results = [gen.submit(p, N) for p in PROMPTS]
texts = {}
for i, (req, q) in enumerate(results):
    parts = []
    req.done.wait()
    while not q.empty():
        parts.append(q.get(timeout=0.1))
    texts[i] = "".join(parts)
torch.cuda.synchronize()
t_batch = time.time() - t0
gen.close()
batch_texts = [texts[i] for i in range(len(PROMPTS))]
P(f"batched : {t_batch:.1f}s ({len(PROMPTS)*N/t_batch:.1f} tok/s total, "
  f"{t_serial/t_batch:.2f}x throughput)")

n_match = sum(1 for a, b in zip(serial_texts, batch_texts) if a.strip() == b.strip())
P(f"exact match: {n_match}/{len(PROMPTS)}")
for i, (a, b) in enumerate(zip(serial_texts, batch_texts)):
    if a.strip() != b.strip():
        P(f"  [{i}] serial={a.strip()[:60]!r}")
        P(f"       batch ={b.strip()[:60]!r}")
P("DONE")
LOG.close()
