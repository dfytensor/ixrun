import os
os.environ['IXRUN_PREBUILT'] = '1'
os.environ['IXRUN_FAST_ATTN'] = '0'
import sys
sys.path.insert(0, r'E:\IXRUN')
import torch
import torch.nn.functional as F

from ixrun.fast_attn import (_s1_attention, _s1_attention_pos,
                             _HAS_TRITON)

assert _HAS_TRITON
torch.manual_seed(7)

# real call pattern: FULL-WIDTH k/v [1,KVH,W,D] + bool mask / pos
H, KG, D, W = 16, 2, 128, 2048
REP = H // KG
kc = torch.randn(1, KG, W, D, device='cuda', dtype=torch.bfloat16)
vc = torch.randn(1, KG, W, D, device='cuda', dtype=torch.bfloat16)
worst = 0.0
for n in (1, 2, 7, 31, 64, 100, 128, 129, 333, 512, 1024, 2047, 2048):
    q = (torch.randn(1, H, 1, D, device='cuda', dtype=torch.bfloat16)
         * 1.7)
    mask = torch.zeros(1, 1, 1, W, device='cuda', dtype=torch.bool)
    mask[..., :n] = True
    # HF mask-mode reference: kv repeated to H heads, sliced to n
    kr = kc.repeat_interleave(REP, dim=1)[:, :, :n].contiguous()
    vr = vc.repeat_interleave(REP, dim=1)[:, :, :n].contiguous()
    ref = F.scaled_dot_product_attention(q, kr, vr,
                                         scale=0.08838834764831845)
    out = _s1_attention(q, kc, vc, mask, 0.08838834764831845)
    err = (out.float() - ref.float()).abs().max().item()
    worst = max(worst, err)
    print(f'mask n={n:5d}  max_err={err:.5f}', flush=True)
print(f'MASK WORST {worst:.5f}  -> '
      f'{"PASS" if worst < 0.02 else "FAIL"}', flush=True)

worst = 0.0
for n in (0, 1, 2, 7, 64, 333, 512, 1024, 2047):
    q = (torch.randn(1, H, 1, D, device='cuda', dtype=torch.bfloat16)
         * 1.7)
    pos = torch.full((1, 1), n, dtype=torch.int64, device='cuda')
    kr = kc.repeat_interleave(REP, dim=1)[:, :, :n + 1].contiguous()
    vr = vc.repeat_interleave(REP, dim=1)[:, :, :n + 1].contiguous()
    ref = F.scaled_dot_product_attention(q, kr, vr,
                                         scale=0.08838834764831845)
    out = _s1_attention_pos(q, kc, vc, pos, 0.08838834764831845)
    err = (out.float() - ref.float()).abs().max().item()
    worst = max(worst, err)
    print(f'pos  n={n:5d}  max_err={err:.5f}', flush=True)
print(f'POS WORST {worst:.5f}  -> '
      f'{"PASS" if worst < 0.02 else "FAIL"}', flush=True)
