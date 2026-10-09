# -*- coding: utf-8 -*-
"""First-ever 27B C++ e2e text generation via CUDA 12.6 nvcc."""
import sys
sys.path.insert(0, r'E:\IXRUN')
from ixrun.cpp_engine_27b import CppQwen27bEngine

import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
eng = CppQwen27bEngine.from_blob(
    r'E:\IXRUN\experiments\qwen38_udcq\q38_blob.pt',
    r'E:\models\Qwen3.8-27B', ctx=256, verbose=True)

for prompt in ["The capital of France is",
               "北京最值得游览的三个景点是"]:
    text = eng.generate(prompt, max_new_tokens=8)
    print(f'TEXT: {text!r}', flush=True)
