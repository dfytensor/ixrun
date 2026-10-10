# -*- coding: utf-8 -*-
"""Blocked-prefill gate: same engine, A/B between the mt8 blocked prefill
(IXRUN_NO_PREFILL=0) and the legacy per-token path (=1). Tokens must match
exactly (mt8 is bit-exact vs v2 per token)."""
import os
os.environ['IXRUN_PREBUILT'] = '1'
import sys, time
sys.path.insert(0, r'E:\IXRUN')
import pandas  # noqa
import torch
from ixrun.cpp_engine_27b import CppQwen27bEngine

eng = CppQwen27bEngine.from_blob(
    r'F:\models\qwen38_uls_blob.pt', r'E:\models\Qwen3.8-27B',
    ctx=512, verbose=True)

P = ("The capital of France is Paris. The capital of Germany is Berlin. "
     "The capital of Italy is")
ids = eng.tok(P, return_tensors='pt').input_ids[0].tolist()
print(f'prompt tokens: {len(ids)}', flush=True)


def run(no_prefill):
    os.environ['IXRUN_NO_PREFILL'] = '1' if no_prefill else '0'
    t0 = time.perf_counter()
    out = eng.generate(P, max_new_tokens=16, graph=True)
    torch.cuda.synchronize()
    return out, time.perf_counter() - t0


out_blk, t_blk = run(False)
out_leg, t_leg = run(True)
print(f'blocked: [{t_blk:.2f}s] {out_blk!r}', flush=True)
print(f'legacy : [{t_leg:.2f}s] {out_leg!r}', flush=True)
print('TOKEN/TEXT EXACT MATCH' if out_blk == out_leg
      else '*** MISMATCH ***', flush=True)
assert out_blk == out_leg, 'blocked prefill differs from legacy path'

# mt8 kernel direct gate vs 8 sequential single calls (bit-exact expected)
ub = torch.load(r'F:\models\qwen38_uls_blob.pt', map_location='cpu',
                mmap=True, weights_only=True)
p = ub['layers']['model.layers.0.mlp.gate_proj']
cb = ub['codebook'].float().cuda()
OF, INF = 17408, 5120
torch.manual_seed(0)
x8 = (torch.randn(8, INF, device='cuda') * 0.5).to(torch.bfloat16).float()
y8 = eng.ext.udcq_gemv_mt8_out(
    x8.contiguous(), p['idx'].cuda(), p['sign'].cuda(), p['scale'].cuda(),
    cb, OF, INF, 16)
ys = torch.stack([
    eng.ext.udcq_gemv_out(x8[i].contiguous(), p['idx'].cuda(),
                          p['sign'].cuda(), p['scale'].cuda(), cb,
                          OF, INF, 16) for i in range(8)])
d = (y8 - ys).abs().max().item()
rel = d / ys.abs().max().item()
print(f'mt8 vs 8x single: max abs {d:.3e} rel {rel:.3e}', flush=True)
assert rel < 1e-6, 'mt8 kernel not matching singles'
print('PASS', flush=True)
