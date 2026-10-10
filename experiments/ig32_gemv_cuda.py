# -*- coding: utf-8 -*-
"""int5-g32+warm hand-CUDA GEMV (group=32, nlev=16, 5b code + 1b sign +
per-group 16-entry fp16 level table).

Layout per group (32 weights along in_f):
  codes : 16 B (uint4, 32 x 4-bit, low nibble = even element)
  sign  : 4 B (uint32, bit j = element j; bit=1 -> +)
  levels: 32 B fp16 [16] (materialized at pack time from family/b/beta/gmax)
Levels table is shared across all rows -> L2-resident (few MB/matrix).
"""
import sys

import torch
from torch.utils.cpp_extension import load_inline

_CUDA_SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cstdint>

#define WARPS 8

__device__ __constant__ float c_brel[15];
__device__ __constant__ float c_beta[7];
__device__ __constant__ float c_tanh[7];

__global__ __launch_bounds__(WARPS * 32) void ig32_gemv_kernel(
    const __nv_bfloat16* __restrict__ x,     // [IN_F]
    __nv_bfloat16* __restrict__ y,           // [OUT_F]
    const uint8_t* __restrict__ codes,       // [OUT_F * IN_F / 2]
    const int* __restrict__ sign,            // [OUT_F * IN_F / 32]
    const __half* __restrict__ levels,       // [nG, 16]
    int OUT_F, int IN_F)
{
    extern __shared__ float smem[];
    float* x_sm = smem;                      // IN_F floats
    float* lv_sm = smem + IN_F;              // [WARPS][32][16] private slices
    for (int i = threadIdx.x; i < IN_F; i += blockDim.x)
        x_sm[i] = __bfloat162float(x[i]);
    __syncthreads();
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    int r = blockIdx.x * WARPS + warp;
    if (r >= OUT_F) return;
    int n_gr = IN_F / 32;
    const uint8_t* irow = codes + (size_t)r * (IN_F / 2);
    const int* srow = sign + (size_t)r * (IN_F / 32);
    const __half* lrow = levels + (size_t)r * n_gr * 16;
    float* my = lv_sm + warp * (32 * 16) + lane * 16;
    float acc = 0.f;
    for (int g = lane; g < n_gr; g += 32) {
        uint4 cb = *reinterpret_cast<const uint4*>(irow + (size_t)g * 16);
        uint32_t sw = (uint32_t)srow[g];
        const __half* lg = lrow + g * 16;
        #pragma unroll
        for (int q = 0; q < 8; ++q) {
            __half2 h = *reinterpret_cast<const __half2*>(lg + 2 * q);
            my[2 * q] = __low2float(h);
            my[2 * q + 1] = __high2float(h);
        }
        const float* xs = x_sm + (size_t)g * 32;
        float xr[32];
        #pragma unroll
        for (int q = 0; q < 8; ++q)
            *reinterpret_cast<float4*>(&xr[q * 4]) =
                *reinterpret_cast<const float4*>(xs + q * 4);
        float inner = 0.f;
        #pragma unroll
        for (int i = 0; i < 32; ++i) {
            uint32_t w = (i >> 3) == 0 ? cb.x : (i >> 3) == 1 ? cb.y :
                         (i >> 3) == 2 ? cb.z : cb.w;
            int nib = (int)((w >> (4 * (i & 7))) & 0xF);
            float sgn = ((sw >> i) & 1u) ? -1.f : 1.f;
            inner = fmaf(my[nib], xr[i] * sgn, inner);
        }
        acc += inner;
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        acc += __shfl_down_sync(0xffffffffu, acc, off);
    if (lane == 0) y[r] = __float2bfloat16_rn(acc);
}

torch::Tensor ig32_gemv(torch::Tensor x, torch::Tensor codes,
                        torch::Tensor sign, torch::Tensor levels,
                        int64_t out_f, int64_t in_f) {
    auto y = torch::empty({out_f}, x.options());
    int grid = (int)((out_f + WARPS - 1) / WARPS);
    size_t sm = (size_t)in_f * sizeof(float)
              + (size_t)WARPS * 32 * 16 * sizeof(float);
    ig32_gemv_kernel<<<grid, WARPS * 32, sm,
                       at::cuda::getCurrentCUDAStream()>>>(
        (const __nv_bfloat16*)x.data_ptr(),
        (__nv_bfloat16*)y.data_ptr(),
        codes.data_ptr<uint8_t>(), sign.data_ptr<int>(),
        (const __half*)levels.data_ptr(), (int)out_f, (int)in_f);
    return y;
}

// ---------------- params variant (27B-deployable) --------------------- //
// levels are COMPUTED per group from (family, b_rel idx, beta idx, gmax):
//  prm uint8 [nG]: bit7 = tanh family, bits 4..6 = b_rel idx, bits 0..2 = beta idx
//   gmax __half [nG]
// logical storage: 4b code + 1b sign + 24b/32 params = bpw + 0.75
__device__ __forceinline__ void ig32_levels_prm(
    float gmx, int prm, float* lv, int nlev)
{
    bool tanh_f = (prm & 0x80) != 0;
    float brev = (tanh_f ? c_tanh : c_brel)[(prm >> 3) & 0xF];
    float b = brev * gmx;
    float beta = c_beta[prm & 7];
    float l2g = __log2f(gmx);
    float gb = exp2f(beta * l2g);
    float fmax = (tanh_f) ? tanhf(gmx / b) : gb / (b + gb);
    for (int k = 0; k < nlev; ++k) {
        float fd = fminf((float)k / (nlev - 1) * fmax, 1.0f - 1e-6f);
        float w;
        if (tanh_f) {
            w = b * atanhf(fd);
        } else {
            float t = fd * b / (1.0f - fd);      // = w^beta
            w = (beta == 1.0f) ? t : exp2f(__log2f(t) / beta);
        }
        lv[k] = fminf(fmaxf(w, 0.f), gmx);
    }
}

__global__ __launch_bounds__(WARPS * 32) void ig32p_gemv_kernel(
    const __nv_bfloat16* __restrict__ x,
    __nv_bfloat16* __restrict__ y,
    const uint8_t* __restrict__ codes,
    const int* __restrict__ sign,
    const __half* __restrict__ gmax,         // [nG]
    const uint8_t* __restrict__ prm,         // [nG]
    int OUT_F, int IN_F)
{
    extern __shared__ float smem[];
    float* x_sm = smem;
    float* lv_sm = smem + IN_F;
    for (int i = threadIdx.x; i < IN_F; i += blockDim.x)
        x_sm[i] = __bfloat162float(x[i]);
    __syncthreads();
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    int r = blockIdx.x * WARPS + warp;
    if (r >= OUT_F) return;
    int n_gr = IN_F / 32;
    const uint8_t* irow = codes + (size_t)r * (IN_F / 2);
    const int* srow = sign + (size_t)r * (IN_F / 32);
    const __half* grow = gmax + (size_t)r * n_gr;
    const uint8_t* prow = prm + (size_t)r * n_gr;
    float* my = lv_sm + warp * (32 * 16) + lane * 16;
    float acc = 0.f;
    for (int g = lane; g < n_gr; g += 32) {
        uint4 cb = *reinterpret_cast<const uint4*>(irow + (size_t)g * 16);
        uint32_t sw = (uint32_t)srow[g];
        ig32_levels_prm(__half2float(grow[g]), prow[g], my, 16);
        const float* xs = x_sm + (size_t)g * 32;
        float xr[32];
        #pragma unroll
        for (int q = 0; q < 8; ++q)
            *reinterpret_cast<float4*>(&xr[q * 4]) =
                *reinterpret_cast<const float4*>(xs + q * 4);
        float inner = 0.f;
        #pragma unroll
        for (int i = 0; i < 32; ++i) {
            uint32_t w = (i >> 3) == 0 ? cb.x : (i >> 3) == 1 ? cb.y :
                         (i >> 3) == 2 ? cb.z : cb.w;
            int nib = (int)((w >> (4 * (i & 7))) & 0xF);
            float sgn = ((sw >> i) & 1u) ? -1.f : 1.f;
            inner = fmaf(my[nib], xr[i] * sgn, inner);
        }
        acc += inner;
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        acc += __shfl_down_sync(0xffffffffu, acc, off);
    if (lane == 0) y[r] = __float2bfloat16_rn(acc);
}

void install_tables(torch::Tensor brel, torch::Tensor beta) {
    float tanhb[7] = {0.25f, 0.4f, 0.6f, 0.85f, 1.2f, 1.8f, 3.0f};
    cudaMemcpyToSymbol(c_brel, brel.data_ptr<float>(), 15 * sizeof(float));
    cudaMemcpyToSymbol(c_beta, beta.data_ptr<float>(), 7 * sizeof(float));
    cudaMemcpyToSymbol(c_tanh, tanhb, 7 * sizeof(float));
}

torch::Tensor get_tables() {
    auto t = torch::zeros({22}, torch::dtype(torch::kFloat32)
                                       .device(torch::kCUDA));
    cudaMemcpyFromSymbol(t.data_ptr(), c_brel, 15 * sizeof(float));
    cudaMemcpyFromSymbol((char*)t.data_ptr() + 15 * sizeof(float),
                         c_beta, 7 * sizeof(float));
    return t;
}

torch::Tensor ig32p_gemv(torch::Tensor x, torch::Tensor codes,
                         torch::Tensor sign, torch::Tensor gmax,
                         torch::Tensor prm, int64_t out_f, int64_t in_f) {
    auto y = torch::empty({out_f}, x.options());
    int grid = (int)((out_f + WARPS - 1) / WARPS);
    size_t sm = (size_t)in_f * sizeof(float)
              + (size_t)WARPS * 32 * 16 * sizeof(float);
    ig32p_gemv_kernel<<<grid, WARPS * 32, sm,
                        at::cuda::getCurrentCUDAStream()>>>(
        (const __nv_bfloat16*)x.data_ptr(),
        (__nv_bfloat16*)y.data_ptr(),
        codes.data_ptr<uint8_t>(), sign.data_ptr<int>(),
        (const __half*)gmax.data_ptr(), prm.data_ptr<uint8_t>(),
        (int)out_f, (int)in_f);
    return y;
}
"""

_CPP_SRC = """
torch::Tensor get_tables();
torch::Tensor ig32_gemv(torch::Tensor x, torch::Tensor codes,
                        torch::Tensor sign, torch::Tensor levels,
                        int64_t out_f, int64_t in_f);
torch::Tensor ig32p_gemv(torch::Tensor x, torch::Tensor codes,
                         torch::Tensor sign, torch::Tensor gmax,
                         torch::Tensor prm, int64_t out_f, int64_t in_f);
void install_tables(torch::Tensor brel, torch::Tensor beta);
"""

_EXT = None


def _load():
    global _EXT
    if _EXT is None:
        _EXT = load_inline(
            name='ig32_gemv_v2',
            cpp_sources=[_CPP_SRC],
            cuda_sources=[_CUDA_SRC],
            functions=['ig32_gemv', 'ig32p_gemv', 'install_tables',
                       'get_tables'],
            extra_cuda_cflags=['-O3', '--use_fast_math',
                               '-allow-unsupported-compiler'],
            verbose=False)
    return _EXT


def ig32_gemv(x, pk):
    ext = _load()
    return ext.ig32_gemv(x.contiguous().view(-1), pk['codes'].cuda(),
                         pk['sign'].cuda(), pk['levels'].cuda(),
                         pk['out_f'], pk['in_f'])


def install_tables():
    ext = _load()
    B_REL = torch.logspace(-3, 2, 15).float()
    BETAS = torch.tensor([0.50, 0.55, 0.65, 0.75, 0.85, 0.95, 1.00],
                         dtype=torch.float32)
    ext.install_tables(B_REL.cuda(), BETAS.cuda())


def ig32p_gemv(x, pk):
    ext = _load()
    return ext.ig32p_gemv(x.contiguous().view(-1), pk['codes'].cuda(),
                          pk['sign'].cuda(), pk['gmax'].cuda(),
                          pk['prm'].cuda(), pk['out_f'], pk['in_f'])
