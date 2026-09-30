# -*- coding: utf-8 -*-
"""GSQ hand-CUDA GEMV: K32 scalar codebook (smem) + per-16 int8 scale
+ 5-bit packed codes (10B per 16-elem group, one warp per row)."""
import sys

import torch
from torch.utils.cpp_extension import load_inline

_CUDA_SRC = r"""
#include <torch/extension.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cstdint>
#include <ATen/cuda/CUDAContext.h>

__global__ void gs_gemv_kernel(
    const __nv_bfloat16* __restrict__ x,
    const uint8_t* __restrict__ codes,     // [nR*nGr, 10]
    const float* __restrict__ cb,          // [32]
    const uint8_t* __restrict__ s_i8,      // [nR*nGr]
    float s_base, float s_step,            // exp2(base + v*step)
    float* __restrict__ yf,                // [out_f]
    int n_gr)                              // groups per row
{
    __shared__ float cb_sm[32];
    for (int i = threadIdx.x; i < 32; i += blockDim.x) {
        cb_sm[i] = cb[i];
    }
    __syncthreads();
    int lane = threadIdx.x & 31;
    int r = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    float acc = 0.f;
    for (int jb = lane; jb < n_gr; jb += 32) {
        long long gidx = (long long)r * n_gr + jb;
        const uint8_t* p = codes + gidx * 10;
        uint32_t d0 = (uint32_t)p[0] | ((uint32_t)p[1] << 8)
            | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
        uint32_t d1 = (uint32_t)p[4] | ((uint32_t)p[5] << 8)
            | ((uint32_t)p[6] << 16) | ((uint32_t)p[7] << 24);
        uint16_t d2 = (uint16_t)(p[8] | (p[9] << 8));
        unsigned long long lo = (unsigned long long)d0
            | ((unsigned long long)d1 << 32);
        float s = exp2f(s_base + (float)s_i8[gidx] * s_step);
        const __nv_bfloat16* x16 = x + jb * 16;
        float inner = 0.f;
        #pragma unroll
        for (int i = 0; i < 12; ++i) {
            int c = (int)((lo >> (5 * i)) & 0x1F);
            inner += cb_sm[c] * __bfloat162float(x16[i]);
        }
        int c12 = (int)(((lo >> 60)
            | ((unsigned long long)d2 << 4)) & 0x1F);
        inner += cb_sm[c12] * __bfloat162float(x16[12]);
        #pragma unroll
        for (int i = 0; i < 3; ++i) {
            int c = (int)((d2 >> (5 * i + 1)) & 0x1F);
            inner += cb_sm[c] * __bfloat162float(x16[13 + i]);
        }
        acc += inner * s;
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        acc += __shfl_down_sync(0xffffffff, acc, off);
    }
    if (lane == 0) {
        yf[r] = acc;
    }
}

torch::Tensor gemv(torch::Tensor x, torch::Tensor codes,
                   torch::Tensor cb, torch::Tensor s_i8,
                   double s_base, double s_step,
                   int64_t out_f, int64_t in_f)
{
    auto yf = torch::empty({out_f},
        torch::dtype(torch::kFloat32).device(x.device()));
    int n_gr = (int)(in_f / 16);
    int wpb = 8;
    unsigned gx = (unsigned)(out_f / wpb);
    auto s0 = at::cuda::getCurrentCUDAStream();
    gs_gemv_kernel<<<gx, wpb * 32, 0, s0>>>(
        reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()),
        codes.data_ptr<uint8_t>(), cb.data_ptr<float>(),
        s_i8.data_ptr<uint8_t>(), (float)s_base, (float)s_step,
        yf.data_ptr<float>(), n_gr);
    return yf.to(torch::kBFloat16);
}
"""

_CPP_SRC = """
torch::Tensor gemv(torch::Tensor x, torch::Tensor codes,
                   torch::Tensor cb, torch::Tensor s_i8,
                   double s_base, double s_step,
                   int64_t out_f, int64_t in_f);
"""

_EXT = None


def _load():
    global _EXT
    if _EXT is None:
        _EXT = load_inline(
            name='gsq_gemv_cuda_v2',
            cpp_sources=[_CPP_SRC],
            cuda_sources=[_CUDA_SRC],
            functions=['gemv'],
            extra_cuda_cflags=['-O3', '--use_fast_math',
                               '-allow-unsupported-compiler'],
            verbose=False)
    return _EXT


def gs_gemv_cuda(x, pk):
    ext = _load()
    return ext.gemv(x.contiguous(), pk['codes5'].cuda(),
                    pk['cb'].cuda(), pk['s_i8'].cuda(),
                    pk['s_base'], pk['s_step'],
                    pk['out_f'], pk['in_f'])


if __name__ == '__main__':
    import time
    sys.path.insert(0, r'E:\IXRUN')
    from benchmarks.gsq_runtime import gs_pack, gs_decode_ref
    torch.manual_seed(0)
    for of, inf in [(512, 512), (2048, 6144)]:
        W = (torch.randn(of, inf) * 0.02).cuda()
        pk = gs_pack(W)
        pk['codes5'] = pk['codes5'].cuda()
        pk['cb'] = pk['cb'].cuda()
        pk['s_i8'] = pk['s_i8'].cuda()
        dref = gs_decode_ref(pk)
        x = torch.randn(inf, dtype=torch.bfloat16, device='cuda')
        y = gs_gemv_cuda(x, pk)
        y_ref = (dref.float() @ x.float()).to(torch.bfloat16)
        gmax = (y.float() - y_ref.float()).abs().max().item()
        for _ in range(5):
            gs_gemv_cuda(x, pk)
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(200):
            gs_gemv_cuda(x, pk)
        torch.cuda.synchronize()
        t_g = (time.time() - t0) / reps_dummy() if False else \
            (time.time() - t0) / 200 * 1000
        Wb = dref.float()
        for _ in range(5):
            Wb @ x.float()
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(200):
            Wb @ x.float()
        torch.cuda.synchronize()
        t_b = (time.time() - t0) / 200 * 1000
        print(f'[{of}x{inf}] gmax={gmax:.4f} gsq={t_g:.3f}ms '
              f'bf16={t_b:.3f}ms ({t_b/t_g:.2f}x)', flush=True)
