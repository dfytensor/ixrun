"""End-to-end: TPAB backend on MiniCPM5 鈥?quality (ppl) + generation speed.

Compares: bf16 / INT8-X streaming / TPAB streaming.
"""
import sys, time, gc, math
sys.setrecursionlimit(10000)
import pandas
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from ixrun.config import MODEL_PATH, DATASET_CACHE, DEFAULT_LEVELS
from ixrun.engine import Int8XEngine
from ixrun.linear import deploy_model
from ixrun.tpab_linear import deploy_model_tpab
from ixrun.eval_utils import eval_ppl, load_wikitext

LOG = open(r"E:\IXRUN\tests\tpab_e2e_out.txt", "w", encoding="utf-8", errors="replace")
def P(*a):
    s = " ".join(str(x) for x in a)
    LOG.write(s + "\n"); LOG.flush(); print(s, flush=True)

tok = AutoTokenizer.from_pretrained(MODEL_PATH)
if tok.pad_token is None: tok.pad_token = tok.eos_token
texts = load_wikitext(cache_dir=DATASET_CACHE)

prompt_ids = tok("The theory of relativity states that", return_tensors="pt")["input_ids"].cuda()

@torch.no_grad()
def gen(model, n=64):
    t0 = time.time()
    out = model.generate(prompt_ids, max_new_tokens=n, do_sample=False,
                         pad_token_id=tok.pad_token_id)
    torch.cuda.synchronize()
    return time.time()-t0, tok.decode(out[0][prompt_ids.shape[1]:], skip_special_tokens=True)

def run(label):
    gc.collect(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    m = AutoModelForCausalLM.from_pretrained(MODEL_PATH, dtype=torch.bfloat16).cuda()
    if label == "bf16":
        pass
    elif label == "int8x":
        deploy_model(m, level_bits=DEFAULT_LEVELS, cache="none", verbose=False)
    elif label == "tpab":
        stats = deploy_model_tpab(m, snr_target_db=28.0, verbose=True)
    elif label == "tpab26":
        stats = deploy_model_tpab(m, snr_target_db=26.0, verbose=True)
    m.eval()
    ppl = eval_ppl(m, tok, texts, max_samples=30)
    gen(m, 8)  # warmup (Triton JIT)
    dt, txt = gen(m)
    mem = torch.cuda.max_memory_allocated()/1e9
    P(f"{label:>6}: ppl={ppl:.2f}  gen={dt*1000/64:.0f} ms/tok  peak={mem:.2f}GB")
    P(f"        {txt.strip()[:90]!r}")
    del m; gc.collect(); torch.cuda.empty_cache()
    return ppl

p_bf = run("bf16")
p_ix = run("int8x")
p28 = run("tpab")
p26 = run("tpab26")
P(f"\ndelta: int8x {p_ix-p_bf:+.2f}  tpab28 {p28-p_bf:+.2f}  tpab26 {p26-p_bf:+.2f}")
P("DONE")
LOG.close()

