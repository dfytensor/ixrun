# -*- coding: utf-8 -*-
"""Hand-written CUDA fused decode+GEMV (M=1) for UDCQ — the higher-ILP
attack on the ~320GB/s Triton ceiling.

Design (warp-per-row):
  - 1 warp = 1 output row; thread t owns elements [j*256 + t*8, +8) per
    wide step j -> 32x4B idx loads coalesce to 128B/warp-instruction
  - one uint4 (16B) x-load per thread per step (8-aligned bf16)
  - sign word: one uint32 per thread per step (4x L1 redundancy)
  - scale: two __half per thread per step
  - codebook in __constant__ memory (global per model, set once)
  - 4 independent fp32 accumulators (unroll 4 wide steps) + warp shuffle
    reduce at the end

DRAM ~0.56B/elem (idx 0.5 + scale 0.06) vs Triton path's ~1.03B -> ceiling
~2x if load-throughput bound.
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

#define WIDE 256          // elements per warp-wide step (32 threads x 8)
#define WARPS 8           // warps per block

__device__ __constant__ float c_cb[16];
__device__ unsigned long long g_last_cb = 0;   // avoid per-call symbol copy

__device__ __forceinline__ float bit_to_sgn(unsigned w, int bit) {
    // matches the Triton convention: sgn = bit*2-1 (bit=1 -> +1)
    return (w >> bit) & 1 ? 1.0f : -1.0f;
}

__global__ __launch_bounds__(WARPS * 32) void udcq_gemv_cuda_kernel(
    const __nv_bfloat16* __restrict__ x,     // [IN_F]
    __nv_bfloat16* __restrict__ y,           // [OUT_F]
    const uint8_t* __restrict__ idx,         // [OUT_F * IN_F / 2]
    const int* __restrict__ sign,            // int32 [OUT_F * IN_F / 32]
    const __half* __restrict__ scale,        // [OUT_F * IN_F / 16]
    int OUT_F, int IN_F)
{
    // codebook -> shared: random per-thread gather from __constant__ would
    // serialize (data-dependent divergent access); smem is banked
    __shared__ float s_cb[16];
    if (threadIdx.x < 16) s_cb[threadIdx.x] = c_cb[threadIdx.x];
    __syncthreads();

    const int row = blockIdx.x * WARPS + threadIdx.x / 32;
    const int t = threadIdx.x & 31;
    if (row >= OUT_F) return;
    const int NSTEP = IN_F / WIDE;

    const uint8_t* irow = idx + (size_t)row * (IN_F / 2);
    const int* srow = sign + (size_t)row * (IN_F / 32);
    const __half* crow = scale + (size_t)row * (IN_F / 16);
    const __nv_bfloat16* xb = x + (size_t)t * 8;

    float a[4] = {0.f, 0.f, 0.f, 0.f};
    int j = 0;
    for (; j + 3 < NSTEP; j += 4) {
        // 4 independent accumulator chains, 4 wide steps unrolled
        #pragma unroll
        for (int u = 0; u < 4; u++) {
            const int k0 = (j + u) * WIDE;
            const uint32_t b = *(const uint32_t*)(irow + k0 / 2 + (size_t)t * 4);
            const uint32_t sw = *(const uint32_t*)(srow + (k0 >> 5) + (t >> 2));
            // thread t's 8 elements always lie in ONE scale group
            const float sc = __half2float(crow[(k0 >> 4) + (t >> 1)]);
            const uint4 xv = *(const uint4*)(xb + (size_t)k0);   // 8 bf16
            float acc = 0.f;
            const int sb = (t & 3) * 8;          // sign bits within word
            #pragma unroll
            for (int i = 0; i < 8; i++) {
                const int n = (b >> (4 * i)) & 0xF;
                const float sgn = bit_to_sgn(sw, sb + i);
                const float xf = __bfloat162float(
                    ((const __nv_bfloat16*)&xv)[i]);
                acc += s_cb[n] * sc * sgn * xf;
            }
            if (u == 0) a[0] += acc;
            else if (u == 1) a[1] += acc;
            else if (u == 2) a[2] += acc;
            else a[3] += acc;
        }
    }
    for (; j < NSTEP; j++) {
        const int k0 = j * WIDE;
        const uint32_t b = *(const uint32_t*)(irow + k0 / 2 + (size_t)t * 4);
        const uint32_t sw = *(const uint32_t*)(srow + (k0 >> 5) + (t >> 2));
        const float sc = __half2float(crow[(k0 >> 4) + (t >> 1)]);
        float acc = 0.f;
        const int sb = (t & 3) * 8;
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            const int n = (b >> (4 * i)) & 0xF;
            const float xf = __bfloat162float(x[k0 + t * 8 + i]);
            acc += s_cb[n] * sc * bit_to_sgn(sw, sb + i) * xf;
        }
        a[0] += acc;
    }
    float s = (a[0] + a[1]) + (a[2] + a[3]);
    #pragma unroll
    for (int o = 16; o; o >>= 1)
        s += __shfl_xor_sync(0xffffffffu, s, o);
    if (t == 0) y[row] = __float2bfloat16_rn(s);
}

torch::Tensor gemv_cuda(torch::Tensor x, torch::Tensor idx,
                        torch::Tensor sign, torch::Tensor scale,
                        torch::Tensor cb, int64_t out_f, int64_t in_f) {
    auto y = torch::empty({out_f}, x.options());
    int grid = (out_f + WARPS - 1) / WARPS;
    udcq_gemv_cuda_kernel<<<grid, WARPS * 32, 0,
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
"""

_EXT = None


def _load():
    global _EXT
    if _EXT is None:
        _EXT = load_inline(
            name='udcq_gemv_cuda',
            cpp_sources=['torch::Tensor gemv_cuda(torch::Tensor x, '
                         'torch::Tensor idx, torch::Tensor sign, '
                         'torch::Tensor scale, torch::Tensor cb, '
                         'int64_t out_f, int64_t in_f);',
                         'void install_codebook(torch::Tensor cb);'],
            cuda_sources=CUDA_SRC,
            functions=['gemv_cuda', 'install_codebook'],
            extra_cuda_cflags=['-O3', '--use_fast_math',
                               '-allow-unsupported-compiler'],
            verbose=False)
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
        if in_f % 256 or out_f % 8:
            print(f'skip {out_f}x{in_f}', flush=True)
            continue
        W = (torch.randn(out_f, in_f, device='cpu') * 0.02)
        cb = udcq_fit_codebook(W, nlev=16, g=UDCQ_G)
        packed = udcq_quantize(W, cb, g=UDCQ_G)
        x = (torch.randn(in_f, device=dev) * 0.5).to(torch.bfloat16)
        for k in ('idx', 'scale', 'sign_packed', 'codebook'):
            packed[k] = packed[k].cuda() if torch.is_tensor(packed[k]) else packed[k]
        cb_f = packed['codebook'].float()
        y_ref = udcq_fused_gemv(x, packed['idx'], packed['sign_packed'],
                                packed['scale'], packed['codebook'],
                                out_f, in_f, g=UDCQ_G)
        y_cu = cuda_gemv(x, packed['idx'], packed['sign_packed'],
                         packed['scale'], cb_f, out_f, in_f)
        d = (y_cu.float() - y_ref.float()).abs().max().item()
        ref = y_ref.float().abs().mean().item()
        snr = 20 * torch.log10(torch.tensor(ref / (d + 1e-9))).item()
        print(f'{out_f}x{in_f}: cuda-vs-triton max {d:.6f} (|y| {ref:.4f}) '
              f'snr {snr:.1f}dB', flush=True)

    # timing (min-of-reps, standalone)
    out_f, in_f = 5120, 5120
    W = (torch.randn(out_f, in_f, device='cpu') * 0.02)
    cb = udcq_fit_codebook(W, nlev=16, g=UDCQ_G)
    packed = udcq_quantize(W, cb, g=UDCQ_G)
    for k in ('idx', 'scale', 'sign_packed', 'codebook'):
        packed[k] = packed[k].cuda() if torch.is_tensor(packed[k]) else packed[k]
    x = (torch.randn(in_f, device=dev) * 0.5).to(torch.bfloat16)
    cb_f = packed['codebook'].float()
    for name, fn in [
        ('cuda', lambda: cuda_gemv(x, packed['idx'], packed['sign_packed'],
                                   packed['scale'], cb_f, out_f, in_f)),
        ('triton', lambda: udcq_fused_gemv(
            x, packed['idx'], packed['sign_packed'], packed['scale'],
            packed['codebook'], out_f, in_f, g=UDCQ_G)),
    ]:
        for _ in range(20):
            fn()
        torch.cuda.synchronize()
        best = 1e9
        ev0 = torch.cuda.Event(enable_timing=True)
        ev1 = torch.cuda.Event(enable_timing=True)
        for _ in range(100):
            ev0.record()
            fn()
            ev1.record()
            torch.cuda.synchronize()
            best = min(best, ev0.elapsed_time(ev1))
        gb = (out_f * in_f * 0.56) / 1e9
        print(f'{name}: {best:.3f}ms  ({gb / (best / 1e3):.0f}GB/s)',
              flush=True)
