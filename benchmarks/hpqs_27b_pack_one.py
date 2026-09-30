# -*- coding: utf-8 -*-
"""Pre-build a 27B HpqsLinear pack: mmap-load one UDCQ layer from the
blob, decode to bf16, hpqs_pack, save. Runs standalone (low VRAM)."""
import sys

sys.path.insert(0, r'E:\IXRUN')
import pandas  # noqa: F401
import torch
from ixrun.udcq import _decode_udcq_ref, UDCQ_G
from benchmarks.hpqs_runtime import hpqs_pack

NAME = 'model.layers.0.mlp.down_proj'
BLOB = r'E:\IXRUN\experiments\qwen38_udcq\q38_blob.pt'
OUT = r'E:\IXRUN\experiments\hpqs_minicpm5\q38_l0_down_pack.pt'

blob = torch.load(BLOB, map_location='cpu', mmap=True,
                  weights_only=False)
e = blob['layers'][NAME]
of, inf = e.get('out_f', 5120), e.get('in_f', 17408)
N = of * inf
packed = {'g': UDCQ_G, 'N': e.get('N', N),
          'out_f': of, 'in_f': inf,
          'idx': e['idx'], 'sign_packed': e['sign'],
          'scale': e['scale'], 'codebook': blob['codebook']}
W = _decode_udcq_ref(packed, device='cuda')
print(f'decoded {NAME}: {tuple(W.shape)}', flush=True)
pk = hpqs_pack(W.float())
del W
torch.cuda.empty_cache()
torch.save({'codes': pk['codes'].cpu(), 'cb': pk['cb'].cpu(),
            'scale': pk['scale'].cpu(),
            'out_f': pk['out_f'], 'in_f': pk['in_f']}, OUT)
print(f'saved {OUT} '
      f'({(pk["codes"].numel()+pk["scale"].numel()*2+4096)/1e6:.0f}MB)',
      flush=True)
