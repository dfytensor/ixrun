# -*- coding: utf-8 -*-
"""bf16xl v2 GEMV/decode: three-plane layout (mant 14B | delta 6B |
sgn 2B | emax 1B per 16-elem group = 23B, 11.5bpw). 7x unaligned
uint32 loads per thread, funnel-shift compile-time extraction,
__uint_as_float(bf16bits<<16) weight assembly. Warp-per-row GEMV with
adaptive split-K; thread-per-group decode."""
import sys

import torch
from torch.utils.cpp_extension import load_inline

_CUDA_SRC = r"""
#include <torch/extension.h>
#include <cuda_bf16.h>
#include <cstdint>
#include <ATen/cuda/CUDAContext.h>

__device__ __forceinline__ uint32_t ld_u32(const uint8_t* p) {
    const uintptr_t a = (uintptr_t) p;
    const uint32_t* pa = (const uint32_t*)(a & ~(uintptr_t)3);
    int sh = (int)(a & 3) * 8;
    return sh == 0 ? pa[0] : __funnelshift_r(pa[0], pa[1], sh);
}

__device__ __forceinline__ float w_from_bits(uint32_t sgn, uint32_t e,
                                             uint32_t m) {
    uint32_t bits = ((sgn << 15) | (e << 7) | m) << 16;
    return __uint_as_float(bits);
}

__global__ void bf16xl_gemv_kernel(
    const __nv_bfloat16* __restrict__ x,
    const uint8_t* __restrict__ mant,   // [nG, 14]
    const uint8_t* __restrict__ delta,  // [nG, 6]
    const uint8_t* __restrict__ sgn,    // [nG, 2]
    const uint8_t* __restrict__ emax,   // [nSuper]
    float* __restrict__ yf,
    int n_gr, int sg, int n_sp)
{
    int lane = threadIdx.x & 31;
    int r = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    int per = (n_gr + n_sp - 1) / n_sp;
    int jb0 = blockIdx.y * per;
    int jb1 = min(jb0 + per, n_gr);
    float acc = 0.f;
    for (int jb = jb0 + lane; jb < jb1; jb += 32) {
        long long gidx = (long long)r * n_gr + jb;
        const uint8_t* mb = mant + gidx * 14;
        const uint8_t* db = delta + gidx * 6;
        const uint8_t* sb = sgn + gidx * 2;
        uint32_t mw0 = ld_u32(mb);      uint32_t mw1 = ld_u32(mb + 4);
        uint32_t mw2 = ld_u32(mb + 8);  uint32_t mw3 = ld_u32(mb + 12);
        uint32_t dw0 = ld_u32(db);      uint32_t dw1 = ld_u32(db + 4);
        uint32_t sw = ld_u32(sb);
        uint8_t em = emax[gidx / sg];
        const __nv_bfloat16* x16 = x + jb * 16;
        float inner = 0.f;
        #pragma unroll
        for (int j = 0; j < 16; ++j) {
            int b7 = 7 * j, b3 = 3 * j;
            const uint32_t mw_sel[4] = {mw0, mw1, mw2, mw3};
            const uint32_t dw_sel[2] = {dw0, dw1};
            uint32_t mv = (b7 % 32 + 7 <= 32)
                ? (mw_sel[b7 / 32] >> (b7 % 32)) & 0x7F
                : __funnelshift_r(mw_sel[b7 / 32], mw_sel[b7 / 32 + 1],
                                  b7 % 32) & 0x7F;
            uint32_t dv = (b3 % 32 + 3 <= 32)
                ? (dw_sel[b3 / 32] >> (b3 % 32)) & 0x7
                : __funnelshift_r(dw_sel[b3 / 32], dw_sel[b3 / 32 + 1],
                                  b3 % 32) & 0x7;
            uint8_t ee = em > dv ? (uint8_t)(em - dv) : (uint8_t)0;
            float wv = w_from_bits((sw >> j) & 1u, (uint32_t)ee, mv);
            inner += wv * __bfloat162float(x16[j]);
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

__global__ void bf16xl_decode_kernel(
    const uint8_t* __restrict__ mant,
    const uint8_t* __restrict__ delta,
    const uint8_t* __restrict__ sgn,
    const uint8_t* __restrict__ emax,
    __nv_bfloat16* __restrict__ W,
    int n_gr, int sg, long long nG)
{
    long long gidx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (gidx >= nG) {
        return;
    }
    const uint8_t* mb = mant + gidx * 14;
    const uint8_t* db = delta + gidx * 6;
    const uint8_t* sb = sgn + gidx * 2;
    uint32_t mw0 = ld_u32(mb);      uint32_t mw1 = ld_u32(mb + 4);
    uint32_t mw2 = ld_u32(mb + 8);  uint32_t mw3 = ld_u32(mb + 12);
    uint32_t dw0 = ld_u32(db);      uint32_t dw1 = ld_u32(db + 4);
    uint32_t sw = ld_u32(sb);
    uint8_t em = emax[gidx / sg];
    int r = (int)(gidx / n_gr);
    int jb = (int)(gidx % n_gr);
    __nv_bfloat16* row = W + (long long)r * ((long long)n_gr * 16)
                       + jb * 16;
    #pragma unroll
    for (int j = 0; j < 16; ++j) {
        int b7 = 7 * j, b3 = 3 * j;
        const uint32_t mw_sel[4] = {mw0, mw1, mw2, mw3};
        const uint32_t dw_sel[2] = {dw0, dw1};
        uint32_t mv = (b7 % 32 + 7 <= 32)
            ? (mw_sel[b7 / 32] >> (b7 % 32)) & 0x7F
            : __funnelshift_r(mw_sel[b7 / 32], mw_sel[b7 / 32 + 1],
                              b7 % 32) & 0x7F;
        uint32_t dv = (b3 % 32 + 3 <= 32)
            ? (dw_sel[b3 / 32] >> (b3 % 32)) & 0x7
            : __funnelshift_r(dw_sel[b3 / 32], dw_sel[b3 / 32 + 1],
                              b3 % 32) & 0x7;
        uint8_t ee = em > dv ? (uint8_t)(em - dv) : (uint8_t)0;
        uint32_t sgnb = ((sw >> j) & 1u) << 15;
        uint16_t w16 = (uint16_t)(sgnb | ((uint32_t)ee << 7) | mv);
        row[j] = *reinterpret_cast<__nv_bfloat16*>(&w16);
    }
}

torch::Tensor gemv(torch::Tensor x, torch::Tensor mant,
                   torch::Tensor delta, torch::Tensor sgn,
                   torch::Tensor emax, int64_t out_f, int64_t in_f,
                   int64_t kg)
{
    auto yf = torch::zeros({out_f},
        torch::dtype(torch::kFloat32).device(x.device()));
    int n_gr = (int)(in_f / 16);
    int sg = (int)(kg / 16);
    int n_sp = n_gr / 96; if (n_sp < 1) n_sp = 1;
    if (n_sp > 6) n_sp = 6;
    int wpb = 8;
    unsigned gx = (unsigned)(out_f / wpb);
    dim3 grid(gx, (unsigned)n_sp);
    auto s0 = at::cuda::getCurrentCUDAStream();
    bf16xl_gemv_kernel<<<grid, wpb * 32, 0, s0>>>(
        reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()),
        mant.data_ptr<uint8_t>(), delta.data_ptr<uint8_t>(),
        sgn.data_ptr<uint8_t>(), emax.data_ptr<uint8_t>(),
        yf.data_ptr<float>(), n_gr, sg, n_sp);
    return yf.to(torch::kBFloat16);
}

torch::Tensor decode(torch::Tensor mant, torch::Tensor delta,
                     torch::Tensor sgn, torch::Tensor emax,
                     int64_t out_f, int64_t in_f, int64_t kg)
{
    auto W = torch::empty({out_f, in_f},
        torch::dtype(torch::kBFloat16).device(mant.device()));
    int n_gr = (int)(in_f / 16);
    int sg = (int)(kg / 16);
    long long nG = (long long)out_f * n_gr;
    long long thr = 256;
    unsigned gx = (unsigned)((nG + thr - 1) / thr);
    auto s0 = at::cuda::getCurrentCUDAStream();
    bf16xl_decode_kernel<<<gx, (unsigned)thr, 0, s0>>>(
        mant.data_ptr<uint8_t>(), delta.data_ptr<uint8_t>(),
        sgn.data_ptr<uint8_t>(), emax.data_ptr<uint8_t>(),
        reinterpret_cast<__nv_bfloat16*>(
            W.view(torch::kUInt16).data_ptr()),
        n_gr, sg, nG);
    return W;
}
"""

_CPP_SRC = """
torch::Tensor gemv(torch::Tensor x, torch::Tensor mant,
                   torch::Tensor delta, torch::Tensor sgn,
                   torch::Tensor emax, int64_t out_f, int64_t in_f,
                   int64_t kg);
torch::Tensor decode(torch::Tensor mant, torch::Tensor delta,
                     torch::Tensor sgn, torch::Tensor emax,
                     int64_t out_f, int64_t in_f, int64_t kg);
"""

_EXT = None


def _load():
    global _EXT
    if _EXT is None:
        _EXT = load_inline(
            name='bf16xl_gemv_cuda_v3',
            cpp_sources=[_CPP_SRC],
            cuda_sources=[_CUDA_SRC],
            functions=['gemv', 'decode'],
            extra_cuda_cflags=['-O3', '--use_fast_math',
                               '-allow-unsupported-compiler'],
            verbose=False)
    return _EXT


def bf16xl_gemv_cuda(x, pk):
    ext = _load()
    return ext.gemv(x.contiguous(), pk['mant'].cuda(),
                    pk['delta'].cuda(), pk['sgn'].cuda(),
                    pk['emax'].cuda(), pk['out_f'], pk['in_f'],
                    pk['kg'])


if __name__ == '__main__':
    import time
    sys.path.insert(0, r'E:\IXRUN')
    from benchmarks.bf16xl_runtime import bf16xl_pack, bf16xl_decode_ref
    torch.manual_seed(0)
    for of, inf in [(512, 512), (1536, 4608)]:
        W = (torch.randn(of, inf) * 0.02).to(torch.bfloat16).cuda()
        pk = bf16xl_pack(W)
        for k in ('mant', 'delta', 'sgn', 'emax'):
            pk[k] = pk[k].cuda()
        d = bf16xl_decode_ref(pk)
        rel = ((d.float() - W.float()).norm()
               / W.float().norm()).item()
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
        t_g = max((time.time() - t0) / 200 * 1000, 1e-4)
        Wb = W.float()
        for _ in range(5):
            Wb @ x.float()
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(200):
            Wb @ x.float()
        torch.cuda.synchronize()
        t_b = max((time.time() - t0) / 200 * 1000, 1e-4)
        mb = pk['mant'].numel() + pk['delta'].numel() \
            + pk['sgn'].numel() + pk['emax'].numel()
        print(f'[{of}x{inf}] rel={rel:.4f} gmax={gmax:.4f} '
              f'v3={t_g:.3f}ms bf16={t_b:.3f}ms ({t_b/t_g:.2f}x) '
              f'{mb*8/(of*inf):.2f}bpw', flush=True)
