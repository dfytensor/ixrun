# -*- coding: utf-8 -*-
"""S=8 verify forward vs 4+4 sequential: token-stream equivalence."""
import os, sys
os.environ['Q38_TOK'] = '8'
os.environ['Q38_GREEDY_ONLY'] = '1'
sys.path.insert(0, r'E:\IXRUN')
import pandas  # noqa
import torch

from ixrun.q38_spec import Q38SpecEngine

BLOB = r'E:\IXRUN\experiments\qwen38_udcq\q38_blob.pt'
MODEL = r'E:\models\Qwen3.8-27B'
eng = Q38SpecEngine.from_blob(BLOB, MODEL, max_ctx=128)
torch.cuda.synchronize()

ids = eng.tokenizer("The capital of France is").input_ids
ids = ids + [0] * (8 - len(ids)) if len(ids) < 8 else ids[:8]
print('block tokens:', ids, flush=True)

H = eng.H

@torch.no_grad()
def fwd(tok_block, pos0):
    t = torch.tensor(tok_block, dtype=torch.long, device='cuda')
    emb = eng.emb_rows(t).view(1, len(tok_block), H)
    idx = torch.arange(pos0, pos0 + len(tok_block), device='cuda')
    cos = eng._cos_all[idx].unsqueeze(0)
    sin = eng._sin_all[idx].unsqueeze(0)
    h = emb
    for layer in eng.layers:
        h = layer(h, position_embeddings=(cos, sin),
                  attention_mask=None, position_ids=idx.view(1, -1),
                  past_key_values=eng.cache)
        if isinstance(h, tuple):
            h = h[0]
    return eng.model.lm_head(eng.final_norm(h))

eng.hard_reset()
lgA = fwd(ids, 0)
aA = lgA[0].argmax(-1).tolist()

eng.hard_reset()
lgB1 = fwd(ids[:4], 0)
lgB2 = fwd(ids[4:], 4)
aB = (lgB1[0].argmax(-1).tolist() + lgB2[0].argmax(-1).tolist())

eng.hard_reset()
lgC1 = fwd(ids[:2], 0)
lgC2 = fwd(ids[2:], 2)
aC = (lgC1[0].argmax(-1).tolist() + lgC2[0].argmax(-1).tolist())

print('S=8 one-shot :', aA, flush=True)
print('4+4 sequential:', aB, flush=True)
print('2+2+... (2+6):', aC, flush=True)
print('MATCH 8-vs-4+4:', aA == aB, flush=True)
print('MATCH 8-vs-2+6:', aA == aC, flush=True)
