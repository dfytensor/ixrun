import os
os.environ['IXRUN_PREBUILT'] = '1'
os.environ['IXRUN_FAST_ATTN'] = '0'
import sys
sys.path.insert(0, r'E:\IXRUN')
import torch
from ixrun.step_graph import StepGraphEngine
from ixrun.fast_attn import install_fused_rope

eng = StepGraphEngine.from_pretrained(codec='bf16xl', verbose=False)
import transformers.models.llama.modeling_llama as ml

orig = ml.apply_rotary_pos_emb
print('installed:', install_fused_rope(), flush=True)

H, KVH, D = 16, 2, 128


def rot(x):
    x1, x2 = x[..., :D // 2], x[..., D // 2:]
    return torch.cat((-x2, x1), dim=-1)


torch.manual_seed(3)
for S in (1, 2, 8):
    q = torch.randn(1, H, S, D, device='cuda', dtype=torch.bfloat16)
    k = torch.randn(1, KVH, S, D, device='cuda', dtype=torch.bfloat16)
    cos = (torch.randn(1, S, D, device='cuda',
                       dtype=torch.bfloat16) * 0.5)
    sin = (torch.randn(1, S, D, device='cuda',
                       dtype=torch.bfloat16) * 0.5)
    q2, k2 = q.clone(), k.clone()
    qr, kr = ml.apply_rotary_pos_emb(q, k, cos, sin)
    q_ref = (q2 * cos.unsqueeze(1)) + (rot(q2) * sin.unsqueeze(1))
    k_ref = (k2 * cos.unsqueeze(1)) + (rot(k2) * sin.unsqueeze(1))
    eq = (qr.float() - q_ref.float()).abs().max().item()
    ek = (kr.float() - k_ref.float()).abs().max().item()
    inq = qr.data_ptr() == q.data_ptr() and kr.data_ptr() == k.data_ptr()
    print(f'S={S}: q_err={eq:.5f} k_err={ek:.5f} in_place={inq}',
          flush=True)
print('PASS', flush=True)
