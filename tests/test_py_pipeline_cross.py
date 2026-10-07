import sys, time
sys.path.insert(0, r'E:\IXRUN')
sys.path.insert(0, r'E:\IXRUN\experiments\qwen38_udcq')
import pandas  # noqa: F401
import torch
import qwen38_udcq_infer as Q
from transformers import AutoModelForCausalLM, AutoTokenizer

m = AutoModelForCausalLM.from_pretrained(
    Q.MD, dtype=torch.bfloat16, low_cpu_mem_usage=True,
    device_map='cpu')
Q.deploy_stream_lazy(m)
m.eval()
Q.to_gpu_selective(m)
tok = AutoTokenizer.from_pretrained(Q.MD)

ids = tok("The capital of France is",
          return_tensors='pt')["input_ids"][0].tolist()
with torch.no_grad():
    out = m(torch.tensor([ids]).cuda(), use_cache=True)
top = torch.topk(out.logits[0, -1].float(), 5)
print('PY top5:', top.indices.tolist(), flush=True)
print('PY vals:', [round(v, 2)
                   for v in top.values.tolist()], flush=True)
print('PY first tok:', top.indices[0].item(),
      tok.decode([top.indices[0].item()]), flush=True)
txt, tps = Q.gen(m, "The capital of France is", n=12)
print(f'PY text: {txt!r} ({tps:.1f} tok/s)', flush=True)
