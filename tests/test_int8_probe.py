# -*- coding: utf-8 -*-
"""int8 GEMM feasibility probe: torch._int_mm speed/layout vs bf16 cublas."""
import sys, time
sys.path.insert(0, r'E:\IXRUN')
import pandas  # noqa
import torch

torch.manual_seed(0)
M, K, N = 256, 5120, 17408


def bench(fn, n=20):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n * 1e3


a8 = torch.randint(-127, 127, (M, K), dtype=torch.int8, device='cuda')
b8 = torch.randint(-127, 127, (K, N), dtype=torch.int8, device='cuda')
ab = (torch.randn(M, K, device='cuda') * 0.1).bfloat16()
bb = torch.randn(K, N, device='cuda').bfloat16()
bbt_src = torch.randn(N, K, device='cuda').bfloat16()
wb8 = torch.randint(-127, 127, (N, K), dtype=torch.int8, device='cuda')

try:
    c = torch._int_mm(a8, b8)
    print('int_mm [M,K]x[K,N] ok', tuple(c.shape), c.dtype, flush=True)
    t_i8 = bench(lambda: torch._int_mm(a8, b8))
    fl = 2 * M * K * N / 1e12
    print(f'int8  : {t_i8:.2f}ms ({fl/(t_i8/1e3):.0f} TOPS)', flush=True)
except Exception as e:
    print('int_mm [K,N] FAILED:', e, flush=True)
    t_i8 = None

try:
    c2 = torch._int_mm(a8, wb8.t())
    print('int_mm with .t() operand ok', flush=True)
    t_i8t = bench(lambda: torch._int_mm(a8, wb8.t()))
    print(f'int8t : {t_i8t:.2f}ms', flush=True)
except Exception as e:
    print('int_mm .t() FAILED:', e, flush=True)

t_bf = bench(lambda: torch.matmul(ab, bbt_src.t()))
fl = 2 * M * K * N / 1e12
print(f'bf16  : {t_bf:.2f}ms ({fl/(t_bf/1e3):.0f} TFLOPS)', flush=True)

# capture test
try:
    g = torch.cuda.CUDAGraph()
    c3 = torch.empty(M, N, dtype=torch.int32, device='cuda')
    for _ in range(2):
        torch._int_mm(a8, b8, out=c3)
    torch.cuda.synchronize()
    with torch.cuda.graph(g):
        torch._int_mm(a8, b8, out=c3)
    g.replay()
    torch.cuda.synchronize()
    print('int_mm graph capture: OK', flush=True)
except Exception as e:
    print('int_mm graph capture FAILED:', type(e).__name__, str(e)[:200],
          flush=True)

# epilogue cost estimate
acc = torch.zeros(M, N, dtype=torch.int32, device='cuda')
sx = torch.rand(M, device='cuda')
sw = torch.rand(N, device='cuda')
t_ep = bench(lambda: acc.float().mul_(sx[:, None]).mul_(sw[None, :]))
print(f'epilogue float+mul+mul [256x17408]: {t_ep:.3f}ms', flush=True)
