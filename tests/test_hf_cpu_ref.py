import sys, time
sys.modules['fla'] = None  # force torch fallbacks (CPU forward)
sys.path.insert(0, r'E:\IXRUN')
sys.path.insert(0, r'E:\IXRUN\experiments\qwen38_udcq')
import pandas  # noqa: F401
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from ixrun.config import QWEN38_PATH as MD

m = AutoModelForCausalLM.from_pretrained(
    MD, dtype=torch.bfloat16, low_cpu_mem_usage=True,
    device_map='cpu')
m.eval()
tok = AutoTokenizer.from_pretrained(MD)

ids = tok("The capital of France is",
          return_tensors='pt')["input_ids"][0].tolist()
print('ids:', ids, flush=True)
with torch.no_grad():
    out = m(input_ids=torch.tensor([ids]))
top = torch.topk(out.logits[0, -1].float(), 5)
print('HF-CPU top5:', top.indices.tolist(), flush=True)
print('HF-CPU vals:', [round(v, 2)
                       for v in top.values.tolist()], flush=True)
print('HF-CPU first tok:', top.indices[0].item(),
      repr(tok.decode([top.indices[0].item()])), flush=True)
