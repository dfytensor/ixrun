import os
os.environ['IXRUN_PREBUILT'] = '1'
os.environ['IXRUN_FAST_ATTN'] = '0'
import sys, time
sys.path.insert(0, r'E:\IXRUN')
import torch
import torch.nn.functional as F

from ixrun.fast_attn import _s1_attention, _HAS_TRITON

assert _HAS_TRITON
torch.manual_seed(7)

# real call pattern: FULL-WIDTH k/v [1,H,W,D] + bool mask [1,1,1,W]
H, D, W = 16, 128, 2048
kc = torch.randn(1, H, W, D, device='cuda', dtype=torch.bfloat16)
vc = torch.randn(1, H, W, D, device='cuda', dtype=torch.bfloat16)
worst = 0.0
for n in (1, 2, 7, 31, 64, 100, 128, 129, 333, 512, 1024, 2047, 2048):
    q = (torch.randn(1, H, 1, D, device='cuda', dtype=torch.bfloat16)
         * 1.7)
    mask = torch.zeros(1, 1, 1, W, device='cuda', dtype=torch.bool)
    mask[..., :n] = True
    ref = F.scaled_dot_product_attention(
        q, kc[:, :, :n].contiguous(), vc[:, :, :n].contiguous(),
        scale=0.08838834764831845)
    out = _s1_attention(q, kc, vc, mask, 0.08838834764831845)
    err = (out.float() - ref.float()).abs().max().item()
    worst = max(worst, err)
    print(f'n={n:5d}  max_err={err:.5f}', flush=True)
print(f'WORST {worst:.5f}  -> {"PASS" if worst < 0.02 else "FAIL"}',
      flush=True)

# timing at n=64 vs n=1024
for n in (64, 512, 1024, 2048):
    q = torch.randn(1, H, 1, D, device='cuda', dtype=torch.bfloat16)
    mask = torch.zeros(1, 1, 1, W, device='cuda', dtype=torch.bool)
    mask[..., :n] = True
    for _ in range(10):
        _s1_attention(q, kc, vc, mask, 0.088388)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(100):
        _s1_attention(q, kc, vc, mask, 0.088388)
    torch.cuda.synchronize()
    us = (time.perf_counter() - t0) / 100 * 1e6
    print(f'n={n:5d}  {us:6.1f} us/layer  (x24 = {us*24/1000:.3f} ms/step)',
          flush=True)
