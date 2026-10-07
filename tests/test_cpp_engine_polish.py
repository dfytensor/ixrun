# -*- coding: utf-8 -*-
"""Polish gate: stop tokens cut, max_tokens honored, streaming
dedup (no duplicated/garbled text across chunk boundaries)."""
import sys, time
sys.path.insert(0, r'E:\IXRUN')
import pandas  # noqa: F401
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from ixrun.config import MODEL_PATH
from ixrun.cpp_engine import CppGsqEngine

m = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, dtype=torch.bfloat16, trust_remote_code=True
).eval().cuda()
eng = CppGsqEngine(m, ctx=512)
eng.tok = AutoTokenizer.from_pretrained(MODEL_PATH,
                                        trust_remote_code=True)

# chat-template prompt (im_end stop path)
msgs = [{"role": "user", "content": "What is 2+3? Answer briefly."}]
try:
    prompt = eng.tok.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=True)
except Exception:
    prompt = "What is 2+3? Answer briefly."

# A. max_tokens honored
out = eng.generate(prompt, max_tokens=8)
n_toks = len(eng.tok(out, add_special_tokens=False).input_ids)
print(f'[A] max_tokens=8 -> {n_toks} tokens: {out!r}', flush=True)
assert n_toks <= 8 + 2, 'max_tokens not honored'

# B. no <|im_end|> leak
assert '<|im_end|>' not in out, 'im_end leaked!'
print(f'[B] no im_end leak: {out!r}', flush=True)

# C. streaming: concatenated chunks == generate
chunks = list(eng.stream(prompt, max_tokens=40))
joined = ''.join(chunks)
full = eng.generate(prompt, max_tokens=40)
# greedy + same seed -> identical text
print(f'[C] stream==generate: {joined == full}', flush=True)
print(f'    reply: {joined[:90]!r}', flush=True)
assert joined == full, 'stream/generate mismatch'

print('POLISH GATE PASSED', flush=True)
