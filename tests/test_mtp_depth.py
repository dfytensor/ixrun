# -*- coding: utf-8 -*-
"""Pure-eager MTP chain depth curve: drafts vs the main model's truth.
No queue, no graphs — isolates the MTP recursion quality."""
import os, sys
os.environ['Q38_GREEDY_ONLY'] = '1'
sys.path.insert(0, r'E:\IXRUN')
import pandas  # noqa
import torch

from ixrun.q38_spec import Q38SpecEngine

BLOB = r'E:\IXRUN\experiments\qwen38_udcq\q38_blob.pt'
eng = Q38SpecEngine.from_blob(BLOB, r'E:\models\Qwen3.8-27B', max_ctx=128)
torch.cuda.synchronize()
H = eng.H
ND = 7

@torch.no_grad()
def main_step(tok, pos):
    t = torch.tensor([tok], dtype=torch.long, device='cuda')
    emb = eng.emb_rows(t).view(1, 1, H)
    idx = torch.tensor([pos], device='cuda')
    cos = eng._cos_all[idx].unsqueeze(0)
    sin = eng._sin_all[idx].unsqueeze(0)
    h, lg = eng._step(emb, cos, sin, idx)
    return int(lg[0, -1].argmax()), h

@torch.no_grad()
def mtp_chain(root, h0, n):
    tok = torch.tensor(root, dtype=torch.long, device='cuda')
    h = h0
    out = []
    for _ in range(n):
        e = eng.emb_rows(tok.view(1)).view(1, 1, H)
        lg, h = eng.mtp.forward2(e, h, eng.mtp_pos_buf)
        tok = lg[:, -1].argmax()
        out.append(int(tok))
    return out

prompt = "The history of the Roman Empire spans more than a thousand years, and its influence"
ids = eng.tokenizer(prompt).input_ids
print('prompt len', len(ids), flush=True)

for rep in range(3):
    eng.hard_reset()
    # eager single-token prefill (slow but honest state)
    h = None
    for pos, t in enumerate(ids):
        nxt, h = main_step(t, pos)
    t = len(ids)
    root = nxt
    # truth: continue the main model ND+1 steps
    truth = []
    ht = h
    for s in range(ND + 1):
        nt, ht = main_step(root if s == 0 else truth[-1], t + s)
        truth.append(nt)
    # drafts from the MTP chain seeded at (h before root, root)
    drafts = mtp_chain(root, h, ND)
    hit = [drafts[s] == truth[s] for s in range(ND)]
    print(f'rep{rep}: depth-hits {hit}', flush=True)
    print(f'  drafts {drafts}', flush=True)
    print(f'  truth  {truth[:ND]}', flush=True)
print('DONE', flush=True)
