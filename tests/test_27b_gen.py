# -*- coding: utf-8 -*-
"""27B C++ e2e text generation + speed measurement."""
import sys, time
sys.path.insert(0, r'E:\IXRUN')
from ixrun.cpp_engine_27b import CppQwen27bEngine

eng = CppQwen27bEngine.from_blob(
    r'E:\IXRUN\experiments\qwen38_udcq\q38_blob.pt',
    r'E:\models\Qwen3.8-27B', ctx=256, verbose=True)

for prompt in ["The capital of France is",
               "北京最值得游览的三个景点是"]:
    t0 = time.perf_counter()
    text = eng.generate(prompt, max_new_tokens=8)
    dt = time.perf_counter() - t0
    nstep = len(eng.tok(prompt).input_ids) + 8
    print(f'TEXT: {text!r}  [{dt:.2f}s for {nstep} steps = '
          f'{dt/nstep*1000:.0f}ms/step, {nstep/dt:.1f} tok/s]',
          flush=True)
