import sys
sys.path.insert(0, r'E:\IXRUN')
import pandas
import torch
from ixrun.udcq import decode_udcq_triton

BLOB = r'E:\IXRUN\experiments\qwen38_udcq\q38_blob.pt'
blob = torch.load(BLOB, map_location='cpu', mmap=True, weights_only=True)
cb = blob['codebook']
GROUP = 16
L = 'model.layers.0.linear_attn.'

from transformers import AutoModelForCausalLM
m = AutoModelForCausalLM.from_pretrained(
    r'E:\models\Qwen3.8-27B', dtype=torch.bfloat16,
    low_cpu_mem_usage=True, device_map='cpu')
at = m.model.layers[0].linear_attn
at.to('cuda')
m.model.embed_tokens.to('cuda')

emb = blob['embed']
he = emb[760].cuda().float()
in_w = m.model.layers[0].input_layernorm.weight.data.float().cuda()
xn = in_w * he * torch.rsqrt(he.pow(2).mean() + 1e-6)

for nm, mod in [('in_proj_qkv', at.in_proj_qkv),
                ('in_proj_z', at.in_proj_z),
                ('in_proj_b', at.in_proj_b),
                ('in_proj_a', at.in_proj_a)]:
    of = mod.weight.data.shape[0]
    inf = mod.weight.data.shape[1]
    # PY bf16 result
    x_in = xn.to(torch.bfloat16)
    py_out = (x_in.float() @ mod.weight.data.cuda().float().T).float()
    # C++ decoded result (Triton decode == my decode == truth)
    p = dict(blob['layers'][L + nm])
    p.update({'g': 16, 'sign_packed': p['sign'],
              'N': p['idx'].numel() * 2, 'out_f': of, 'in_f': inf,
              'codebook': cb})
    W = decode_udcq_triton(p).float().cuda()
    cpp_out = W @ xn
    rel = ((cpp_out - py_out).norm() / py_out.norm().clamp_min(1e-9)).item()
    cs = torch.nn.functional.cosine_similarity(
        cpp_out, py_out, dim=0).item()
    print(f'{nm}: PY {py_out.norm().item():.4f} cpp {cpp_out.norm().item():.4f} '
          f'rel {rel:.4f} cos {cs:.4f}', flush=True)
