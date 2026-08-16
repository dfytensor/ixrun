"""MiniCPM5 VRAM: deploy-time transient vs inference-time resident/peak."""
import sys, gc
sys.setrecursionlimit(10000)
import pandas
import torch

LOG = open(r"E:\IXRUN\tests\vram2_out.txt", "w", encoding="utf-8", errors="replace")
def P(*a):
    s = " ".join(str(x) for x in a)
    LOG.write(s + "\n"); LOG.flush()

from transformers import AutoModelForCausalLM, AutoTokenizer
from ixrun.config import MODEL_PATH, DEFAULT_LEVELS
from ixrun.tpab_linear import deploy_model_tpab
from ixrun.engine import Int8XEngine

tok = AutoTokenizer.from_pretrained(MODEL_PATH)
ids = tok("Hello world, this is a VRAM measurement prompt for peak testing.",
          return_tensors="pt")["input_ids"].cuda()

for label, loader, deploy in [
    ("bf16", "full", None),
    ("tpab28", "full", lambda m: deploy_model_tpab(m, snr_target_db=28.0, verbose=False)),
    ("tpab28-lazy", "lazy", lambda m: deploy_model_tpab(m, snr_target_db=28.0, verbose=False, lazy=True)),
]:
    gc.collect(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    if loader == "lazy":
        m = Int8XEngine._load_any(MODEL_PATH, torch.bfloat16, low_cpu=True)
    else:
        m = AutoModelForCausalLM.from_pretrained(MODEL_PATH, dtype=torch.bfloat16).cuda()
    a_load = torch.cuda.memory_allocated()/1e9
    p_load = torch.cuda.max_memory_allocated()/1e9
    if deploy:
        deploy(m)
        if loader == "lazy":
            eng_tmp = Int8XEngine(m, tok, {})
            eng_tmp._finalize_device()
        gc.collect(); torch.cuda.empty_cache()
        a_dep = torch.cuda.memory_allocated()/1e9
        p_dep = torch.cuda.max_memory_allocated()/1e9
    else:
        a_dep, p_dep = a_load, p_load
    # inference-time peak measured FRESH (reset after deploy)
    torch.cuda.reset_peak_memory_stats()
    m.eval()
    with torch.no_grad():
        for _ in range(3): m(ids)
        torch.cuda.synchronize()
    a_inf = torch.cuda.memory_allocated()/1e9
    p_inf = torch.cuda.max_memory_allocated()/1e9
    P(f"{label:>12}: after-load alloc={a_load:.2f} deploy-peak={p_dep:.2f} "
      f"| resident={a_dep:.2f} | inference-peak={p_inf:.2f}")
    del m; gc.collect(); torch.cuda.empty_cache()
P("DONE")
LOG.close()
