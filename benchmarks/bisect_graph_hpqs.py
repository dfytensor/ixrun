# -*- coding: utf-8 -*-
"""Bisect: StepGraph 'hpqs-mixed' with a FILTERED pack set.
argv[1] = 'one' (only L0 down_proj) or 'all' (48, known garbage)."""
import os
import sys

sys.path.insert(0, r'E:\IXRUN')
import pandas  # noqa: F401
import torch
from ixrun.step_graph import StepGraphEngine

CACHE = r'E:\IXRUN\experiments\hpqs_minicpm5\down_o_packs.pt'
mode = sys.argv[1] if len(sys.argv) > 1 else 'one'
packs = torch.load(CACHE, weights_only=False)
if mode == 'one':
    keep = 'model.layers.0.mlp.down_proj'
    packs = {keep: packs[keep]}
tmp = rf'C:\Users\Administrator\AppData\Local\Temp\opencode\packs_{mode}.pt'
torch.save(packs, tmp)
os.environ['HPQS_PACK_CACHE'] = tmp

eng = StepGraphEngine.from_pretrained(codec='hpqs-mixed', verbose=True)
out = eng.generate('The history of computing is', max_new_tokens=20)
print(f'graph+{mode}:', repr(out), flush=True)
