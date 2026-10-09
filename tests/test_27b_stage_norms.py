import sys
sys.modules['fla'] = None
sys.path.insert(0, r'E:\IXRUN')
import pandas
import torch
import torch.nn.functional as F

from ixrun.config import QWEN38_PATH
from transformers import AutoModelForCausalLM

m = AutoModelForCausalLM.from_pretrained(
    QWEN38_PATH, dtype=torch.bfloat16,
    low_cpu_mem_usage=True, device_map='cpu')
m.eval()
at = m.model.layers[0].linear_attn
at.to('cuda')

emb = None  # load from blob
from torch.utils.cpp_extension import load_inline
# just use HF embed
he = m.model.embed_tokens.weight.data[760].cuda().float()
in_w = m.model.layers[0].input_layernorm.weight.data.float().cuda()
post_w = m.model.layers[0].post_attention_layernorm.weight.data.float().cuda()
xn = in_w * he * torch.rsqrt(he.pow(2).mean() + 1e-6)
print(f'xn norm: {xn.norm().item():.4f}', flush=True)

caps = {}
def mk(name):
    def hk(mod, inp, out):
        caps[name] = out.detach()
    return hk

hooks = []
hooks.append(at.in_proj_qkv.register_forward_hook(mk('qkv')))
hooks.append(at.in_proj_z.register_forward_hook(mk('z')))
hooks.append(at.in_proj_b.register_forward_hook(mk('b')))
hooks.append(at.in_proj_a.register_forward_hook(mk('a')))
hooks.append(at.norm.register_forward_hook(mk('gated_norm')))
hooks.append(at.register_forward_hook(mk('mod_out')))

with torch.no_grad():
    out = at(xn.view(1, 1, -1).to(torch.bfloat16), None)

for hk in hooks:
    hk.remove()

for nm in ('qkv', 'z', 'b', 'a', 'gated_norm', 'mod_out'):
    if nm in caps:
        t = caps[nm]
        print(f'{nm}: shape {tuple(t.shape)} norm {t.float().norm().item():.4f} '
              f'max {t.float().abs().max().item():.4f}', flush=True)

# Also: print z, b, a values to check gates
if 'b' in caps:
    b = caps['b']
    print(f'b (pre-sigmoid): min {b.min().item():.4f} max {b.max().item():.4f} '
          f'norm {b.norm().item():.4f}', flush=True)
    print(f'beta = sigmoid(b): min {(b*0+1).sigmoid().min().item():.4f}', flush=True)
if 'a' in caps:
    a = caps['a']
    A_log = at.A_log.data.float()
    dt_bias = at.dt_bias.data.float()
    g = -torch.exp(A_log) * F.softplus(a + dt_bias)
    print(f'g (decay): min {g.min().item():.4f} max {g.max().item():.4f} '
          f'exp(g) range [{g.exp().min().item():.6f}, {g.exp().max().item():.6f}]', flush=True)
print('DONE', flush=True)
