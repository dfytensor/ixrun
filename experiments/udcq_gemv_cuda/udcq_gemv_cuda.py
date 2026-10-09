# -*- coding: utf-8 -*-
"""Hand-written CUDA fused decode+GEMV (M=1 + MT=4) for UDCQ.

v2 design (ported from the C++ engine_v5 udcq_gemv_v2, 4x over v1):
  - sign-folded 32-entry codebook in smem: cb[nib | bit<<4], bit=1 -> +cb
    (kills the per-element sign select + one multiply)
  - x staged in smem once per block (fp32), float4 reads per group
  - single FMA chain per element: inner = fmaf(cb2, x, inner)
  - group-strided walk (lane g, g += 32), warp shuffle reduce
  - mt kernel: same fold, x stays global uint4 (4 tokens share the walk)

Requires IN_F % 16 == 0. Sign/scale/idx layout identical to v1.
"""
import sys, time, torch
sys.path.insert(0, r'E:\IXRUN')
import pandas
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

CUDA_SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#define WARPS 8           // warps per block (8 rows)

__device__ __constant__ float c_cb[16];

__global__ __launch_bounds__(WARPS * 32) void udcq_gemv_cuda_kernel(
    const __nv_bfloat16* __restrict__ x,     // [IN_F]
    __nv_bfloat16* __restrict__ y,           // [OUT_F]
    const uint8_t* __restrict__ idx,         // [OUT_F * IN_F / 2]
    const int* __restrict__ sign,            // int32 [OUT_F * IN_F / 32]
    const __half* __restrict__ scale,        // [OUT_F * IN_F / 16]
    int OUT_F, int IN_F)
{
    extern __shared__ float smem[];
    float* x_sm = smem;                      // IN_F floats
    float* cb_sm = smem + IN_F;              // 32 floats, sign-folded
    for (int i = threadIdx.x; i < 16; i += blockDim.x) {
        cb_sm[i] = -c_cb[i];                 // bit=0 -> negative
        cb_sm[i + 16] = c_cb[i];             // bit=1 -> positive
    }
    for (int i = threadIdx.x; i < IN_F; i += blockDim.x)
        x_sm[i] = __bfloat162float(x[i]);
    __syncthreads();
    int lane = threadIdx.x & 31;
    int r = blockIdx.x * WARPS + (threadIdx.x >> 5);
    if (r >= OUT_F) return;
    int n_gr = IN_F / 16;
    const uint8_t* irow = idx + (size_t)r * (IN_F / 2);
    const int* srow = sign + (size_t)r * (IN_F / 32);
    const __half* crow = scale + (size_t)r * n_gr;
    float acc = 0.f;
    for (int g = lane; g < n_gr; g += 32) {
        uint2 b2 = *reinterpret_cast<const uint2*>(irow + (size_t)g * 8);
        uint32_t sw = (uint32_t)srow[g >> 1] >> (16 * (g & 1));
        float sc = __half2float(crow[g]);
        const float* xs = x_sm + (size_t)g * 16;
        float xr[16];
        #pragma unroll
        for (int q = 0; q < 4; ++q)
            *reinterpret_cast<float4*>(&xr[q * 4]) =
                *reinterpret_cast<const float4*>(xs + q * 4);
        float inner = 0.f;
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            int b = (int)((b2.x >> (8 * i)) & 0xFF);
            int s0 = (int)((sw >> (2 * i)) & 1u) << 4;
            int s1 = (int)((sw >> (2 * i + 1)) & 1u) << 4;
            inner = fmaf(cb_sm[(b & 0xF) | s0], xr[2 * i], inner);
            inner = fmaf(cb_sm[(b >> 4) | s1], xr[2 * i + 1], inner);
        }
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            int b = (int)((b2.y >> (8 * i)) & 0xFF);
            int s0 = (int)((sw >> (8 + 2 * i)) & 1u) << 4;
            int s1 = (int)((sw >> (8 + 2 * i + 1)) & 1u) << 4;
            inner = fmaf(cb_sm[(b & 0xF) | s0], xr[8 + 2 * i], inner);
            inner = fmaf(cb_sm[(b >> 4) | s1], xr[8 + 2 * i + 1], inner);
        }
        acc += inner * sc;
    }
    #pragma unroll
    for (int o = 16; o; o >>= 1)
        acc += __shfl_down_sync(0xffffffffu, acc, o);
    if (lane == 0) y[r] = __float2bfloat16_rn(acc);
}

// ------------------------------------------------------------------ //
// Multi-token (TOK=4): decode each weight ONCE, FMA across tokens.
// v2 fold applied; x stays global uint4 (staging 4 planes exceeds smem
// on down_proj shapes).
// ------------------------------------------------------------------ //
#define TOK 4

__global__ __launch_bounds__(WARPS * 32) void udcq_gemv_mt_cuda_kernel(
    const __nv_bfloat16* __restrict__ x,     // [TOK, IN_F]
    __nv_bfloat16* __restrict__ y,           // [TOK, OUT_F]
    const uint8_t* __restrict__ idx,
    const int* __restrict__ sign,
    const __half* __restrict__ scale,
    int OUT_F, int IN_F)
{
    __shared__ float cb_sm[32];
    for (int i = threadIdx.x; i < 16; i += blockDim.x) {
        cb_sm[i] = -c_cb[i];
        cb_sm[i + 16] = c_cb[i];
    }
    __syncthreads();
    const int row = blockIdx.x * WARPS + threadIdx.x / 32;
    const int t = threadIdx.x & 31;
    if (row >= OUT_F) return;
    const int NSTEP = IN_F / 256;
    const uint8_t* irow = idx + (size_t)row * (IN_F / 2);
    const int* srow = sign + (size_t)row * (IN_F / 32);
    const __half* crow = scale + (size_t)row * (IN_F / 16);
    float a[TOK][4];
    #pragma unroll
    for (int tk = 0; tk < TOK; tk++)
        #pragma unroll
        for (int u = 0; u < 4; u++) a[tk][u] = 0.f;
    int j = 0;
    for (; j + 3 < NSTEP; j += 4) {
        #pragma unroll
        for (int u = 0; u < 4; u++) {
            const int k0 = (j + u) * 256;
            const uint32_t b = *(const uint32_t*)(irow + k0 / 2 + (size_t)t * 4);
            const uint32_t sw = (uint32_t)*(const int*)(srow + (k0 >> 5) + (t >> 2));
            const float sc = __half2float(crow[(k0 >> 4) + (t >> 1)]);
            const int sb = (t & 3) * 8;
            float wv[8];
            #pragma unroll
            for (int i = 0; i < 8; i++)
                wv[i] = cb_sm[((b >> (4 * i)) & 0xF)
                              | (int)((sw >> (sb + i)) & 1u) << 4] * sc;
            #pragma unroll
            for (int tk = 0; tk < TOK; tk++) {
                const uint4 xv = *(const uint4*)(x + (size_t)tk * IN_F +
                                                 k0 + (size_t)t * 8);
                float acc = 0.f;
                #pragma unroll
                for (int i = 0; i < 8; i++)
                    acc = fmaf(wv[i], __bfloat162float(
                        ((const __nv_bfloat16*)&xv)[i]), acc);
                a[tk][u] += acc;
            }
        }
    }
    for (; j < NSTEP; j++) {
        const int k0 = j * 256;
        const uint32_t b = *(const uint32_t*)(irow + k0 / 2 + (size_t)t * 4);
        const uint32_t sw = (uint32_t)*(const int*)(srow + (k0 >> 5) + (t >> 2));
        const float sc = __half2float(crow[(k0 >> 4) + (t >> 1)]);
        const int sb = (t & 3) * 8;
        float wv[8];
        #pragma unroll
        for (int i = 0; i < 8; i++)
            wv[i] = cb_sm[((b >> (4 * i)) & 0xF)
                          | (int)((sw >> (sb + i)) & 1u) << 4] * sc;
        #pragma unroll
        for (int tk = 0; tk < TOK; tk++) {
            float acc = 0.f;
            #pragma unroll
            for (int i = 0; i < 8; i++)
                acc = fmaf(wv[i], __bfloat162float(
                    x[(size_t)tk * IN_F + k0 + (size_t)t * 8 + i]), acc);
            a[tk][0] += acc;
        }
    }
    float s[TOK];
    #pragma unroll
    for (int tk = 0; tk < TOK; tk++) {
        s[tk] = (a[tk][0] + a[tk][1]) + (a[tk][2] + a[tk][3]);
        #pragma unroll
        for (int o = 16; o; o >>= 1)
            s[tk] += __shfl_xor_sync(0xffffffffu, s[tk], o);
    }
    if (t == 0) {
        #pragma unroll
        for (int tk = 0; tk < TOK; tk++)
            y[(size_t)tk * OUT_F + row] = __float2bfloat16_rn(s[tk]);
    }
}

torch::Tensor gemv_mt_cuda(torch::Tensor x, torch::Tensor idx,
                           torch::Tensor sign, torch::Tensor scale,
                           int64_t out_f, int64_t in_f) {
    auto y = torch::empty({TOK, out_f}, x.options());
    int grid = (out_f + WARPS - 1) / WARPS;
    udcq_gemv_mt_cuda_kernel<<<grid, WARPS * 32, 0,
                               at::cuda::getCurrentCUDAStream()>>>(
        (const __nv_bfloat16*)x.data_ptr(),
        (__nv_bfloat16*)y.data_ptr(),
        (const uint8_t*)idx.data_ptr(), (const int*)sign.data_ptr(),
        (const __half*)scale.data_ptr(), (int)out_f, (int)in_f);
    return y;
}

torch::Tensor gemv_cuda(torch::Tensor x, torch::Tensor idx,
                        torch::Tensor sign, torch::Tensor scale,
                        torch::Tensor cb, int64_t out_f, int64_t in_f) {
    auto y = torch::empty({out_f}, x.options());
    int grid = (out_f + WARPS - 1) / WARPS;
    size_t sm = (size_t)in_f * sizeof(float) + 32 * sizeof(float);
    udcq_gemv_cuda_kernel<<<grid, WARPS * 32, sm,
                            at::cuda::getCurrentCUDAStream()>>>(
        (const __nv_bfloat16*)x.data_ptr(),
        (__nv_bfloat16*)y.data_ptr(),
        (const uint8_t*)idx.data_ptr(), (const int*)sign.data_ptr(),
        (const __half*)scale.data_ptr(), (int)out_f, (int)in_f);
    return y;
}

void install_codebook(torch::Tensor cb) {
    // call ONCE outside any graph capture; kernels trust c_cb
    cudaMemcpyToSymbol(c_cb, cb.data_ptr<float>(), 16 * sizeof(float));
}

void install_attr(int64_t max_sm) {
    // x_sm staging can exceed the 48KB default (IN_F=17408 -> 69.6KB)
    cudaFuncSetAttribute(udcq_gemv_cuda_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize, (int)max_sm);
}
"""

_EXT = None


def _load():
    global _EXT
    if _EXT is None:
        _EXT = load_inline(
            name='udcq_gemv_cuda_v2',
            cpp_sources=['torch::Tensor gemv_cuda(torch::Tensor x, '
                         'torch::Tensor idx, torch::Tensor sign, '
                         'torch::Tensor scale, torch::Tensor cb, '
                         'int64_t out_f, int64_t in_f);',
                         'torch::Tensor gemv_mt_cuda(torch::Tensor x, '
                         'torch::Tensor idx, torch::Tensor sign, '
                         'torch::Tensor scale, '
                         'int64_t out_f, int64_t in_f);',
                         'void install_codebook(torch::Tensor cb);',
                         'void install_attr(int64_t max_sm);'],
            cuda_sources=CUDA_SRC,
            functions=['gemv_cuda', 'gemv_mt_cuda', 'install_codebook',
                       'install_attr'],
            extra_cuda_cflags=['-O3', '--use_fast_math',
                               '-allow-unsupported-compiler'],
            verbose=False)
        # x_sm can exceed 48KB (IN_F=17408 -> 69.6KB); opt in once
        import torch as _t
        dev = _t.cuda.current_device()
        max_sm = _t.cuda.get_device_properties(dev).shared_memory_per_block_optin
        _EXT.install_attr(max_sm)
    return _EXT


def cuda_gemv(x, idx, sign, scale, cb_f32, out_f, in_f):
    """cb_f32: float32 [16]. Returns bf16 [out_f].
    install_codebook() must have been called once with the model codebook."""
    ext = _load()
    return ext.gemv_cuda(x.reshape(-1), idx, sign, scale, cb_f32,
                         out_f, in_f)


def install_codebook(cb_f32):
    """Upload the model-wide codebook to __constant__ memory (once)."""
    ext = _load()
    ext.install_codebook(cb_f32.cuda())


def cuda_gemv_mt(x, idx, sign, scale, out_f, in_f):
    """x: [4, in_f] bf16 -> y: [4, out_f]. Decodes each weight once."""
    ext = _load()
    return ext.gemv_mt_cuda(x.reshape(4, in_f), idx, sign, scale,
                            out_f, in_f)


if __name__ == '__main__':
    torch.manual_seed(0)
    dev = 'cuda'
    print('building + testing...', flush=True)
    ext = _load()
    print('built.', flush=True)

    from ixrun.udcq import udcq_fit_codebook, udcq_quantize, UDCQ_G
    from ixrun.udcq import udcq_fused_gemv

    for out_f, in_f in [(5120, 5120), (17408, 5120), (5120, 17408),
                        (248320, 5120), (13824, 4096)]:
        if in_f % 16 or out_f % 8:
            print(f'skip {out_f}x{in_f}', flush=True)
            continue
        W = (torch.randn(out_f, in_f, device='cpu') * 0.02)
        cb = udcq_fit_codebook(W, nlev=16, g=UDCQ_G)
        packed = udcq_quantize(W, cb, g=UDCQ_G)
        x = (torch.randn(in_f, device=dev) * 0.5).to(torch.bfloat16)
        for k in ('idx', 'scale', 'sign_packed', 'codebook'):
            packed[k] = packed[k].cuda() if torch.is_tensor(packed[k]) else packed[k]
        cb_f = packed['codebook'].float()
        install_codebook(cb_f)
        y_ref = udcq_fused_gemv(x, packed['idx'], packed['sign_packed'],
                                packed['scale'], packed['codebook'],
                                out_f, in_f, g=UDCQ_G)
        y_cu = cuda_gemv(x, packed['idx'], packed['sign_packed'],
                         packed['scale'], cb_f, out_f, in_f)
        d = (y_cu.float() - y_ref.float()).abs().max().item()
        ref = y_ref.float().abs().mean().item()
        snr = 20 * torch.log10(torch.tensor(ref / (d + 1e-9))).item()
        # timing (min-of-reps via events)
        for _ in range(20):
            cuda_gemv(x, packed['idx'], packed['sign_packed'],
                      packed['scale'], cb_f, out_f, in_f)
        torch.cuda.synchronize()
        best = 1e9
        ev0 = torch.cuda.Event(enable_timing=True)
        ev1 = torch.cuda.Event(enable_timing=True)
        for _ in range(100):
            ev0.record()
            cuda_gemv(x, packed['idx'], packed['sign_packed'],
                      packed['scale'], cb_f, out_f, in_f)
            ev1.record()
            torch.cuda.synchronize()
            best = min(best, ev0.elapsed_time(ev1))
        gb = (out_f * in_f * 0.5625) / 1e9
        print(f'{out_f}x{in_f}: max {d:.6f} snr {snr:.1f}dB | '
              f'{best:.3f}ms ({gb / (best / 1e3):.0f}GB/s)', flush=True)
