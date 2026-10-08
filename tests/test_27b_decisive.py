# -*- coding: utf-8 -*-
"""DECISIVE: HF forward with layer-0 weights replaced by DECODED
blob weights. h(L0) ~ 5.9 => packs bad; ~ 20.6 => C++ kernels bad."""
import sys
sys.modules['fla'] = None
sys.path.insert(0, r'E:\IXRUN')
import pandas  # noqa: F401
import torch
from ixrun.udcq import decode_udcq_triton

BLOB = r'E:\IXRUN\experiments\qwen38_udcq\q38_blob.pt'
blob = torch.load(BLOB, map_location='cpu', mmap=True,
                  weights_only=True)
cb = blob['codebook']

from transformers import AutoModelForCausalLM, AutoTokenizer
m = AutoModelForCausalLM.from_pretrained(
    r'E:\models\Qwen3.8-27B', dtype=torch.bfloat16,
    low_cpu_mem_usage=True, device_map='cpu')
m.eval()
tok = AutoTokenizer.from_pretrained(r'E:\models\Qwen3.8-27B')
ids = [760]
captured = []
def mkhook():
    def hk(mod, args):
        captured.append(args[0].detach())
    return hk
hk = m.model.layers[1].register_forward_pre_hook(mkhook())

def triton_decode(pre, nm, of, inf):
    p = dict(blob['layers'][pre + nm])
    p.update({'g': 16, 'sign_packed': p['sign'],
              'N': p['idx'].numel() * 2, 'out_f': of, 'in_f': inf,
              'codebook': cb})
    return decode_udcq_triton(p)

with torch.no_grad():
    # baseline bf16 forward
    out = m(input_ids=torch.tensor([ids]))
    h_base = captured[0][0, 0].float().norm().item()
    top0 = torch.topk(out.logits[0, -1].float(), 3)
    print(f'BASELINE h(L0) norm {h_base:.3f} | top1 '
          f'{top0.indices[0].item()} '
          f'{tok.decode([top0.indices[0].item()])!r}', flush=True)

    # replace layer-0 quantizable weights with decoded blob weights
    pre = 'model.layers.0.'
    la = m.model.layers[0].linear_attn
    ml = m.model.layers[0].mlp
    repl = [
        (la.in_proj_qkv, 'linear_attn.in_proj_qkv', 10240, 5120),
        (la.in_proj_z, 'linear_attn.in_proj_z', 6144, 5120),
        (la.in_proj_b, 'linear_attn.in_proj_b', 48, 5120),
        (la.in_proj_a, 'linear_attn.in_proj_a', 48, 5120),
        (la.out_proj, 'linear_attn.out_proj', 5120, 6144),
        (ml.gate_proj, 'mlp.gate_proj', 17408, 5120),
        (ml.up_proj, 'mlp.up_proj', 17408, 5120),
        (ml.down_proj, 'mlp.down_proj', 5120, 17408),
    ]
    for mod, nm, of, inf in repl:
        W = triton_decode(pre, nm, of, inf).to(torch.bfloat16).cpu()
        mod.weight.data = W
    print('layer-0 weights replaced with decoded blob', flush=True)

    captured.clear()
    out = m(input_ids=torch.tensor([ids]))
    h_dec = captured[0][0, 0].float().norm().item()
    top1 = torch.topk(out.logits[0, -1].float(), 3)
    print(f'DECODED h(L0) norm {h_dec:.3f} | top1 '
          f'{top1.indices[0].item()} '
          f'{tok.decode([top1.indices[0].item()])!r}', flush=True)
    print(f'VERDICT: decoded-weights h = {h_dec:.1f} '
          f'(C++ says 5.9, bf16 says 20.6) -> '
          + ('PACKS BAD' if h_dec < 12 else 'C++ KERNELS BAD'),
          flush=True)
hk.remove()
