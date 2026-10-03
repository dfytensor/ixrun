# -*- coding: utf-8 -*-
"""bf16x-lossless GEMV: 14b packed elements (delta6|mant7|sign1,
delta==63 zero) + per-64-elem emax byte. Warp-per-row, byte-assembly
loads (28B groups, 4B-aligned), bf16 register reassembly (no dequant
ALU beyond shifts). TRUE lossless: output bits == bf16 matmul input."""
import sys

import torch
from torch.utils.cpp_extension import load_inline

_CUDA_SRC = r"""
#include <torch/extension.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cstdint>
#include <ATen/cuda/CUDAContext.h>

__global__ void bf16xl_gemv_kernel(
    const __nv_bfloat16* __restrict__ x,
    const uint8_t* __restrict__ stream,   // [nR*nGr, 28]
    const uint8_t* __restrict__ emax,     // [nSuper]
    float* __restrict__ yf,
    int n_gr, int sg)                     // groups/row, groups/supergroup
{
    __syncthreads();                       // no smem state; keep layout
    __shared__ float red[8][32];
    int lane = threadIdx.x & 31;
    int r = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    int jb0 = blockIdx.y * ((n_gr + gridDim.y - 1) / gridDim.y);
    int jb1 = min(jb0 + (n_gr + gridDim.y - 1) / gridDim.y, n_gr);
    float acc = 0.f;
    for (int jb = jb0 + lane; jb < jb1; jb += 32) {
        long long gidx = (long long)r * n_gr + jb;
        const uint8_t* p = stream + gidx * 28;
        uint32_t d[7];
        #pragma unroll
        for (int k = 0; k < 7; ++k) {
            d[k] = *(const uint32_t*)(p + k * 4);
        }
        int e = emax[gidx / sg];
        const __nv_bfloat16* x16 = x + jb * 16;
        float inner = 0.f;
        #pragma unroll
        for (int i = 0; i < 16; ++i) {
            int bo = i * 14;
            int w = bo >> 5;
            int sh = bo & 31;
            uint32_t v24 = sh ? ((d[w] >> sh) | (d[w + 1] << (32 - sh)))
                              : d[w];
            int val = (int)(v24 & 0x3FFF);
            int delta = val >> 8;
            int m = (val >> 1) & 0x7F;
            int sign = val & 1;
            int ee = (delta == 63) ? 0 : (e - delta);
            uint16_t w16 = (uint16_t)((sign << 15) | (ee << 7) | m);
            float wv = __bfloat162float(
                *reinterpret_cast<__nv_bfloat16*>(&w16));
            inner += wv * __bfloat162float(x16[i]);
        }
        acc += inner;
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        acc += __shfl_down_sync(0xffffffff, acc, off);
    }
    if (lane == 0) {
        atomicAdd(&yf[r], acc);
    }
}

torch::Tensor gemv(torch::Tensor x, torch::Tensor stream,
                   torch::Tensor emax, int64_t out_f, int64_t in_f,
                   int64_t kg)
{
    auto yf = torch::zeros({out_f},
        torch::dtype(torch::kFloat32).device(x.device()));
    int n_gr = (int)(in_f / 16);
    int sg = (int)(kg / 16);
    int wpb = 8;
    int n_sp = n_gr / 96; if (n_sp < 1) n_sp = 1;
    if (n_sp > 6) n_sp = 6;
    unsigned gx = (unsigned)(out_f / wpb);
    dim3 grid(gx, (unsigned)n_sp);
    auto s0 = at::cuda::getCurrentCUDAStream();
    bf16xl_gemv_kernel<<<grid, wpb * 32, 0, s0>>>(
        reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()),
        stream.data_ptr<uint8_t>(), emax.data_ptr<uint8_t>(),
        yf.data_ptr<float>(), n_gr, sg);
    return yf.to(torch::kBFloat16);
}
"""

_CPP_SRC = """
torch::Tensor gemv(torch::Tensor x, torch::Tensor stream,
                   torch::Tensor emax, int64_t out_f, int64_t in_f,
                   int64_t kg);
"""

_EXT = None


def _load():
    global _EXT
    if _EXT is None:
        _EXT = load_inline(
            name='bf16xl_gemv_cuda_v1',
            cpp_sources=[_CPP_SRC],
            cuda_sources=[_CUDA_SRC],
            functions=['gemv'],
            extra_cuda_cflags=['-O3', '--use_fast_math',
                               '-allow-unsupported-compiler'],
            verbose=False)
    return _EXT


def bf16xl_gemv_cuda(x, pk):
    ext = _load()
    return ext.gemv(x.contiguous(), pk['stream'].cuda(),
                    pk['emax'].cuda(), pk['out_f'], pk['in_f'],
                    pk['kg'])


if __name__ == '__main__':
    import time
    sys.path.insert(0, r'E:\IXRUN')
    from benchmarks.bf16xl_runtime import bf16xl_pack, \
        bf16xl_decode_ref
    torch.manual_seed(0)
    for of, inf in [(512, 512), (1536, 4608)]:
        W = (torch.randn(of, inf) * 0.02).to(torch.bfloat16).cuda()
        pk = bf16xl_pack(W)
        pk['stream'] = pk['stream'].cuda()
        pk['emax'] = pk['emax'].cuda()
        d = bf16xl_decode_ref(pk)
        exact = bool((d.view(torch.uint16) ==
                      W.view(torch.uint16)).all())
        x = torch.randn(inf, dtype=torch.bfloat16, device='cuda')
        y = bf16xl_gemv_cuda(x, pk)
        y_ref = (d.float() @ x.float()).to(torch.bfloat16)
        gmax = (y.float() - y_ref.float()).abs().max().item()
        for _ in range(5):
            bf16xl_gemv_cuda(x, pk)
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(200):
            bf16xl_gemv_cuda(x, pk)
        torch.cuda.synchronize()
        t_g = (time.time() - t0) / 200 * 1000
        Wb = W.float()
        for _ in range(5):
            Wb @ x.float()
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(200):
            Wb @ x.float()
        torch.cuda.synchronize()
        t_b = max((time.time() - t0) / 200 * 1000, 1e-4)
        mb = pk['stream'].numel() + pk['emax'].numel()
        print(f'[{of}x{inf}] lossless={exact} gmax={gmax:.4f} '
              f'bf16xl={t_g:.3f}ms bf16={t_b:.3f}ms ({t_b/t_g:.2f}x) '
              f'{mb*8/(of*inf):.2f}bpw', flush=True)
