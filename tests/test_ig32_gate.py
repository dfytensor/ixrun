# -*- coding: utf-8 -*-
"""Gate + bench for ig32 (int5-g32+warm) GEMV kernel."""
import sys, time
sys.path.insert(0, r'E:\IXRUN')
import pandas  # noqa: F401
import torch
from benchmarks.ig32_runtime import ig32_pack, ig32_decode_ref
from experiments.ig32_gemv_cuda import ig32_gemv, ig32p_gemv, install_tables, _load

torch.manual_seed(0)
_load()
install_tables()
print('tables readback:', _load().get_tables().cpu().tolist())
print('ext built', flush=True)

ok = True
for of, inf in [(512, 512), (2048, 6144), (5504, 1536)]:
    W = (torch.randn(of, inf) * 0.02).cuda()
    t0 = time.time()
    pk = ig32_pack(W)
    tk = time.time() - t0
    dref = ig32_decode_ref(pk).float()
    rel = ((dref - W).norm() / W.norm()).item()
    x = torch.randn(inf, dtype=torch.bfloat16, device='cuda')
    y = ig32_gemv(x, pk)
    y_ref = (dref @ x.float()).to(torch.bfloat16)
    gmax = (y.float() - y_ref.float()).abs().max().item()
    yp = ig32p_gemv(x, pk)
    gmaxp = (yp.float() - y_ref.float()).abs().max().item()
    ok &= gmax < 0.05 and gmaxp < 0.06
    bpw = 4 + 1 + (8 + 16) / 32
    for _ in range(20):
        ig32_gemv(x, pk)
    torch.cuda.synchronize()
    best = 1e9
    ev0 = torch.cuda.Event(enable_timing=True)
    ev1 = torch.cuda.Event(enable_timing=True)
    for _ in range(100):
        ev0.record()
        ig32_gemv(x, pk)
        ev1.record()
        torch.cuda.synchronize()
        best = min(best, ev0.elapsed_time(ev1))
    print(f'[{of}x{inf}] pack {tk:.1f}s rec-rel {rel:.4f} gmax {gmax:.4f}/{gmaxp:.4f} '
          f'{bpw:.2f}bpw {best:.3f}ms', flush=True)
print('IG32 GATE:', 'PASS' if ok else 'FAIL', flush=True)
