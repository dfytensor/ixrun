"""After the VRAM fixes: deploy time + inference speed regression check."""
import sys, time, gc
sys.setrecursionlimit(10000)
import pandas
import torch

LOG = open(r"E:\IXRUN\tests\spd_out.txt", "w", encoding="utf-8", errors="replace")
def P(*a):
    s = " ".join(str(x) for x in a)
    LOG.write(s + "\n"); LOG.flush()

from transformers import AutoModelForCausalLM, AutoTokenizer
from ixrun.config import MODEL_PATH
from ixrun.tpab_linear import deploy_model_tpab
from ixrun.engine import Int8XEngine

tok = AutoTokenizer.from_pretrained(MODEL_PATH)
if tok.pad_token is None: tok.pad_token = tok.eos_token
ids = tok("The theory of relativity states that", return_tensors="pt")["input_ids"].cuda()

@torch.no_grad()
def gen(model, n=64, warm=8):
    model.generate(ids, max_new_tokens=warm, do_sample=False, pad_token_id=tok.pad_token_id)
    torch.cuda.synchronize()
    t0 = time.time()
    out = model.generate(ids, max_new_tokens=n, do_sample=False, pad_token_id=tok.pad_token_id)
    torch.cuda.synchronize()
    return (time.time()-t0)/n*1000

# --- A: GPU encode (old path, reference for deploy speed) ---
gc.collect(); torch.cuda.empty_cache()
m = AutoModelForCausalLM.from_pretrained(MODEL_PATH, dtype=torch.bfloat16).cuda()
# temporarily force GPU encode for timing comparison
import ixrun.tpab_linear as TL
from ixrun.tpab import encode_tpab, stage_gpu
from ixrun.tpab_gemv import prepare_gemv_stage
orig_init = TL.TpabLinear.__init__
def gpu_init(self, weight, snr_target_db=26.0, outlier_frac=0.004):
    import torch.nn as nn
    nn.Module.__init__(self)
    self.out_features, self.in_features = weight.shape
    self.packed = encode_tpab(weight.detach().cuda() if not weight.is_cuda else weight.detach(),
                              snr_target_db=snr_target_db, outlier_frac=outlier_frac)
    self.tile_r = self.packed["tile_r"]
    self.staged = stage_gpu(self.packed, "cuda")
    self.gemv_stage = prepare_gemv_stage(self.packed, "cuda", staged=self.staged)
TL.TpabLinear.__init__ = gpu_init
t0 = time.time()
deploy_model_tpab(m, snr_target_db=28.0, verbose=False)
t_gpu_dep = time.time()-t0
TL.TpabLinear.__init__ = orig_init
m.eval()
ms_gpu = gen(m)
P(f"GPU-encode : deploy={t_gpu_dep:.0f}s  gen={ms_gpu:.0f} ms/tok")
del m; gc.collect(); torch.cuda.empty_cache()

# --- B: CPU encode + lazy (new path) ---
m = Int8XEngine._load_any(MODEL_PATH, torch.bfloat16, low_cpu=True)
t0 = time.time()
deploy_model_tpab(m, snr_target_db=28.0, verbose=False, lazy=True)
t_cpu_dep = time.time()-t0
eng = Int8XEngine(m, tok, {})
eng._finalize_device()
m.eval()
ms_cpu = gen(m)
P(f"CPU+lazy   : deploy={t_cpu_dep:.0f}s  gen={ms_cpu:.0f} ms/tok")
P(f"\ndeploy ratio: {t_cpu_dep/t_gpu_dep:.2f}x  |  gen ratio: {ms_cpu/ms_gpu:.2f}x")
P("DONE")
LOG.close()
