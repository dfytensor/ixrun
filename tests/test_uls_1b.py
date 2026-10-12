import os
os.environ['IXRUN_PREBUILT'] = '1'
os.environ['IXRUN_FAST_ATTN'] = '0'
os.environ['IXRUN_Q_LMHEAD'] = '0'
import sys
sys.path.insert(0, r'E:\IXRUN')
import torch
from ixrun.step_graph import StepGraphEngine
from ixrun.udcq import UdcqLinear, udcq_scale_to_uls

eng = StepGraphEngine.from_pretrained(codec='udcq-stream')
mods = [m for m in eng.model_raw.modules() if isinstance(m, UdcqLinear)]
m = mods[3]
old = m._scale
udcq_scale_to_uls(m)

from ixrun.udcq import _cuda_gemv_mod
cg = _cuda_gemv_mod()
cg.install_codebook(m._cb.float())

x = torch.randn(8, m.in_features, device='cuda',
                dtype=torch.bfloat16)
W = m._uls_dequant_torch(m._scale.device).float()
y_ref = x.float() @ W.T

for tag, fn, sl in [('mt8', cg.cuda_gemv_mt8, slice(0, 8)),
                    ('mt4', cg.cuda_gemv_mt, slice(0, 4))]:
    y = fn(x[sl].contiguous(), m._idx, m._sign, m._scale,
           m.out_features, m.in_features, uls=1).float()
    r = (y - y_ref[sl]).norm() / y_ref[sl].norm()
    print(f'{tag}: rel={r.item():.5f} '
          f'gmax={(y - y_ref[sl]).abs().max().item():.5f}', flush=True)
    print('  y[0,:4]:', [round(v, 3) for v in y[0, :4].tolist()],
          flush=True)
    print('  ref[0,:4]:', [round(v, 3) for v in y_ref[0, :4].tolist()],
          flush=True)
m._scale = old
