# -*- coding: utf-8 -*-
"""int8g: int8 weights + per-16-group fp16 scale (8.5bpw, the Q8_0
class).  The decode is ~3 instr/elem (cvt, cvt, FMA) vs ~12-20 for the
LUT/bitmap codecs - llama.cpp's own table shows this tier WINS:
Q8_0 508 t/s > BF16X 389 > F16 371 (decode ALU eats bandwidth).
Per-16 scales (finer than Q8_0's per-32) at the same bpw."""
import sys

import torch
from torch.utils.cpp_extension import load_inline

_CUDA_SRC = r"""
#include <torch/extension.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <ATen/cuda/CUDAContext.h>

__global__ void int8g_gemv_kernel(
    const __nv_bfloat16* __restrict__ x,
    const int8_t* __restrict__ q,       // [out_f, in_f]
    const __half* __restrict__ scale,   // [out_f, n_gr]
    __nv_bfloat16* __restrict__ y,      // [out_f]
    int n_gr, int out_f, int sdiv)
{
    const int lane = threadIdx.x & 31;
    const int r = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    if (r >= out_f) {
        return;
    }
    const int8_t* row = q + (long long)r * n_gr * 16;
    const __half* srow = scale + (long long)r * ((n_gr + sdiv) >> sdiv);
    float acc = 0.f;
    for (int jb = lane; jb < n_gr; jb += 32) {
        const uint4 v = *reinterpret_cast<const uint4*>(row + jb * 16);
        const __nv_bfloat16* x16 = x + jb * 16;
        const float sc = __half2float(srow[jb >> sdiv]);
        const int8_t* b = reinterpret_cast<const int8_t*>(&v);
        float inner = 0.f;
        #pragma unroll
        for (int j = 0; j < 16; ++j) {
            inner += (float)b[j] * __bfloat162float(x16[j]);
        }
        acc += inner * sc;
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        acc += __shfl_down_sync(0xffffffff, acc, off);
    }
    if (lane == 0) {
        y[r] = __float2bfloat16(acc);
    }
}

torch::Tensor gemv(torch::Tensor x, torch::Tensor q, torch::Tensor scale,
                   int64_t out_f, int64_t in_f, int64_t sdiv)
{
    const int n_gr = (int)(in_f / 16);
    const int wpb = 8;
    const unsigned gx = (unsigned)((out_f + wpb - 1) / wpb);
    auto y = torch::empty({out_f},
        torch::dtype(torch::kBFloat16).device(x.device()));
    int8g_gemv_kernel<<<gx, wpb * 32, 0,
                        at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()),
        q.data_ptr<int8_t>(),
        reinterpret_cast<const __half*>(scale.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(y.data_ptr()),
        n_gr, (int)out_f, (int)sdiv);
    return y;
}
"""

_CPP_SRC = """
torch::Tensor gemv(torch::Tensor x, torch::Tensor q, torch::Tensor scale,
                   int64_t out_f, int64_t in_f, int64_t sdiv);
"""

_EXT = None


def _load():
    global _EXT
    if _EXT is None:
        _EXT = load_inline(
            name='int8g_gemv_cuda_v2',
            cpp_sources=[_CPP_SRC],
            cuda_sources=[_CUDA_SRC],
            functions=['gemv'],
            extra_cuda_cflags=['-O3', '--use_fast_math',
                               '-allow-unsupported-compiler'],
            verbose=False)
    return _EXT


def int8g_pack(W, g=32):
    """W [out, in] bf16 -> dict(q i8 [out,in], scale fp16 [out, in/g])."""
    of, inf = W.shape
    assert inf % g == 0
    wg = W.float().view(of, inf // g, g)
    scale = wg.abs().amax(dim=2, keepdim=True) / 127.0
    scale = scale.clamp_min(1e-12).half().float()      # fp16-rounded
    q = torch.round(wg / scale).clamp(-127, 127).to(torch.int8)
    return {'q': q.view(of, inf).contiguous().cpu(),
            'scale': scale.view(of, -1).half().contiguous().cpu(),
            'out_f': of, 'in_f': inf, 'g': g}


def int8g_decode_ref(pk):
    of, inf, g = pk['out_f'], pk['in_f'], pk['g']
    w = (pk['q'].float().view(of, inf // g, g)
         * pk['scale'].float().unsqueeze(-1))
    return w.view(of, inf).to(torch.bfloat16)


def int8g_gemv_cuda(x, pk):
    ext = _load()
    return ext.gemv(x.contiguous(), pk['q'].cuda(), pk['scale'].cuda(),
                    pk['out_f'], pk['in_f'],
                    1 if pk['g'] == 32 else 0)


if __name__ == '__main__':
    import time
    sys.path.insert(0, r'E:\IXRUN')
    torch.manual_seed(0)
    for of, inf in [(512, 512), (1536, 1536), (4608, 1536),
                    (1536, 4608), (130560, 1536)]:
        W = (torch.randn(of, inf) * 0.02).to(torch.bfloat16).cuda()
        pk = int8g_pack(W)
        for k in ('q', 'scale'):
            pk[k] = pk[k].cuda()
        d = int8g_decode_ref(pk)
        rel = ((d.float() - W.float()).norm()
               / W.float().norm()).item()
        x = torch.randn(inf, dtype=torch.bfloat16, device='cuda')
        y = int8g_gemv_cuda(x, pk)
        y_ref = (d.float() @ x.float()).to(torch.bfloat16)
        gmax = (y.float() - y_ref.float()).abs().max().item()
        for _ in range(10):
            int8g_gemv_cuda(x, pk)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(200):
            int8g_gemv_cuda(x, pk)
        torch.cuda.synchronize()
        us = (time.perf_counter() - t0) / 200 * 1e6
        bpw = pk['q'].numel() * 8 + pk['scale'].numel() * 16
        print(f'[{of}x{inf}] rel={rel:.5f} gmax={gmax:.4f} '
              f'{us:6.1f}us {bpw / inf / of:.2f}bpw '
              f'{of * inf * 1.0625 / us / 1e3:.0f}GB/s', flush=True)
