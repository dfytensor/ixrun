import os, sys, time
os.environ['UDCQ_CUDA_GEMV'] = '1'
os.environ['Q38_GREEDY_ONLY'] = '1'
sys.path.insert(0, r'E:\IXRUN')
import pandas  # noqa
import torch

from ixrun.q38_spec import Q38SpecEngine
BLOB = r'E:\IXRUN\experiments\qwen38_udcq\q38_blob.pt'
MODEL = r'E:\models\Qwen3.8-27B'
CTX = int(os.environ.get('SPEC_CTX', '128'))
eng = Q38SpecEngine.from_blob(BLOB, MODEL, max_ctx=CTX)
torch.cuda.synchronize()
prompt = "The capital of France is"
eng.generate(prompt, max_new_tokens=4)

orig = eng._spec_iter
cnt = {'n': 0}
def wrapper(ids, mn, **kw):
    for batch in orig(ids, mn, **kw):
        cnt['n'] += 1
        yield batch
eng._spec_iter = wrapper
torch.cuda.synchronize()
t0 = time.perf_counter()
r = eng.generate(prompt, max_new_tokens=64)
torch.cuda.synchronize()
dt = time.perf_counter() - t0
n_tok = len(eng.tokenizer(r).input_ids)
n_iter = cnt['n']
print(f'[ctx={CTX}] tok {n_tok} iters {n_iter} | E {n_tok/n_iter:.2f} '
      f'| iter {dt/n_iter*1000:.1f}ms | {n_tok/dt:.1f} tok/s', flush=True)