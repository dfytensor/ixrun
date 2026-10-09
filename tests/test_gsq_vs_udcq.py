# -*- coding: utf-8 -*-
"""GSQ vs UDCQ on real 27B weights: bpw + GEMV-output error vs bf16."""
import sys, json, time
sys.path.insert(0, r'E:\IXRUN')
import torchvision  # noqa: F401
import torch

sys.path.insert(0, r'E:\IXRUN')
from ixgs import int8gs_quantize, decode_weight_scatter
from safetensors import safe_open

BLOB = r'E:\IXRUN\experiments\qwen38_udcq\q38_blob.pt'
MODEL = r'E:\models\Qwen3.8-27B'
blob = torch.load(BLOB, map_location='cpu', mmap=True, weights_only=True)
cb = blob['codebook'].float()
index = json.load(open(MODEL + r'\model.safetensors.index.json',
                       encoding='utf-8'))['weight_map']

def load_bf16(key):
    with safe_open(MODEL + '\\' + index[key], framework='pt') as f:
        return f.get_tensor(key).float()

def deq_udcq(p, out_f, in_f):
    idx = p['idx'].long().view(out_f, in_f // 2)
    nib = torch.empty(out_f, in_f, dtype=torch.long)
    nib[:, 0::2] = idx & 0xF
    nib[:, 1::2] = idx >> 4
    sg = p['sign'].int().view(out_f, in_f // 32, 1)
    bits = (sg >> torch.arange(32, dtype=torch.int)) & 1
    sgn = bits.reshape(out_f, in_f).float() * 2 - 1
    sc = p['scale'].float().view(out_f, in_f // 16).repeat_interleave(16, dim=1)
    return cb[nib] * sc * sgn

CASES = [
    ('model.language_model.layers.0.mlp.gate_proj.weight',
     'model.layers.0.mlp.gate_proj', 17408, 5120),
    ('model.language_model.layers.0.linear_attn.in_proj_qkv.weight',
     'model.layers.0.linear_attn.in_proj_qkv', 10240, 5120),
    ('model.language_model.layers.3.self_attn.q_proj.weight',
     'model.layers.3.self_attn.q_proj', 12288, 5120),
]
torch.manual_seed(0)
xr = torch.randn(5120)
for skey, bkey, out_f, in_f in CASES:
    t0 = time.perf_counter()
    Wb = load_bf16(skey)
    p = blob['layers'][bkey]
    Wu = deq_udcq(p, out_f, in_f)
    ub = p['idx'].numel() + p['sign'].numel() * 4 + p['scale'].numel() * 2
    g = int8gs_quantize(Wb)
    gb = sum(v.numel() * v.element_size()
             for v in g.values() if torch.is_tensor(v))
    Wg = decode_weight_scatter(g, device='cpu').float()
    yb, yu, yg = Wb @ xr, Wu @ xr, Wg @ xr
    e_u = (yu - yb).norm() / yb.norm()
    e_g = (yg - yb).norm() / yb.norm()
    print(f'{bkey}\n  UDCQ {ub / out_f / in_f * 8:.3f}bpw  gemv rel-err {e_u*100:.3f}%'
          f'\n  GSQ  {gb / out_f / in_f * 8:.3f}bpw  gemv rel-err {e_g*100:.3f}%'
          f'   [{time.perf_counter()-t0:.0f}s]', flush=True)
print('DONE', flush=True)
