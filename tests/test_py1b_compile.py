# -*- coding: utf-8 -*-
"""Experiment: does torch.compile close the PY 1B architecture gap?
Same GSQ deploy as StepGraphEngine(codec='gsq'), + torch.compile before
the engine's manual graph capture."""
import sys, time
sys.path.insert(0, r'E:\IXRUN')
import pandas  # noqa: F401
import torch
from ixrun.config import MODEL_PATH, DATASET_CACHE
from ixrun.eval_utils import load_wikitext
from ixrun.engine import Int8XEngine
from ixrun.linear import iter_quantizable_linears, _set_parent_child
from transformers import AutoTokenizer

MODE = sys.argv[1] if len(sys.argv) > 1 else 'default'

tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
model = Int8XEngine._load_any(MODEL_PATH, torch.bfloat16, low_cpu=True)

# GSQ deploy (mirrors step_graph.py codec='gsq')
from benchmarks.gsq_runtime import GsqLinear, gs_pack
for name, mod in list(iter_quantizable_linears(model)):
    W = mod.weight.data.float().cuda()
    pk = gs_pack(W)
    del W
    torch.cuda.empty_cache()
    _set_parent_child(model, name, GsqLinear(pk))
print('[gsq] deployed', flush=True)

model = model.cuda()
model.eval()
if MODE != 'none':
    t0 = time.perf_counter()
    model = torch.compile(model, mode=MODE)
    print(f'[compile {MODE}] wrapped in {time.perf_counter()-t0:.0f}s '
          '(kernels compile lazily)', flush=True)

from ixrun.step_graph import StepGraphEngine
sg = StepGraphEngine(model, tok, max_ctx=1024, verbose=False)
print('[engine] graph captured', flush=True)

from ixrun.eval_utils import load_wikitext as _lw
texts = _lw(cache_dir=DATASET_CACHE)
prompt = '\n'.join(texts)[20000:20600]


def run(n):
    t0 = time.perf_counter()
    sg.generate(prompt, max_new_tokens=n)
    torch.cuda.synchronize()
    return time.perf_counter() - t0


run(8)                    # triggers lazy inductor compilation
print('[first run done]', flush=True)
run(8)
t8 = run(8)
t136 = run(136)
dec = (t136 - t8) / 128
print(f'PY-gsq+compile({MODE}) PURE decode: {1/dec:.1f} tok/s '
      f'({dec*1000:.1f} ms/tok)', flush=True)
out = sg.generate(prompt, max_new_tokens=48)
print('text:', repr(out[-80:]), flush=True)
