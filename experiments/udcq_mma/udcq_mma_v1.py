# -*- coding: utf-8 -*-
"""Hand-written CUDA mma kernel for UDCQ (Windows: nvcc + VS2022 BuildTools).

v1 design (target: large-M, where Triton fused GEMM loses):
  - decode compressed streams -> SMEM bf16 W tile ONCE per CTA/k-tile
    (removes the x(M/BM) decode redundancy that killed the Triton version)
  - mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 via inline PTX
  - fragments loaded with direct 32-bit SMEM accesses (v1; ldmatrix later)
  - CTA tile 128(M) x 128(N) x 64(K), 8 warps (2 x 4), each warp 64 x 32
Build -> correctness vs decode-ref GEMM -> bench vs cublas bf16.
"""
import sys, os, time, torch
sys.path.insert(0, r'E:\IXRUN')
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline
from ixrun.udcq import udcq_fit_codebook, udcq_quantize, _decode_udcq_ref

CUDA_SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>

#define BM 128
#define BN 128
#define BK 64
#define NWARPS 8

// ---------------------------------------------------------------------------
// mma.sync m16n8k16 bf16: D(f32x4) += A(b32x4) * B(b32x2)
// ---------------------------------------------------------------------------
__device__ __forceinline__ void mma_bf16(float* d, const unsigned* a,
                                         const unsigned* b) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

// ---------------------------------------------------------------------------
// fused UDCQ decode + GEMM kernel
//   y[M,OUT_F] = x[M,IN_F] @ W.T ;  W[n,k] = sign * scale[n*K/16+..] * CB[idx]
//   idx: uint8 [OUT_F*IN_F] byte-aligned, W row-major
//   sign: int32 flat bitstream ; scale: __nv_bfloat16 [ng] ; cb: bf16 [16]
// ---------------------------------------------------------------------------
__global__ void udcq_mma_kernel(
    const __nv_bfloat16* __restrict__ x,   // [M, IN_F]
    __nv_bfloat16* __restrict__ y,         // [M, OUT_F]
    const uint8_t* __restrict__ idx,
    const int* __restrict__ sign,
    const __half* __restrict__ scale,     // f16 storage (packer uses .half())
    const __half* __restrict__ cb,
    int M, int OUT_F, int IN_F)
{
    const int n0 = blockIdx.x * BN;        // W row tile
    const int m0 = blockIdx.y * BM;        // token tile
    const int tid = threadIdx.x;           // 0..255
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int wm = warp >> 2;              // 0..1
    const int wn = warp & 3;               // 0..3

    __shared__ __nv_bfloat16 Xs[BM][BK];
    __shared__ __nv_bfloat16 Ws[BN][BK];
    __shared__ float CBs[16];
    if (tid < 16) CBs[tid] = __half2float(cb[tid]);
    __syncthreads();

    // accumulators: warp tile 64 x 32 = (4 mma m-tiles) x (4 n-tiles)
    float acc[4][4][4];
#pragma unroll
    for (int i = 0; i < 4; i++)
#pragma unroll
        for (int j = 0; j < 4; j++)
#pragma unroll
            for (int r = 0; r < 4; r++) acc[i][j][r] = 0.f;

    const int n_elems = BN * BK;           // 8192 per CTA per k-tile
    for (int k0 = 0; k0 < IN_F; k0 += BK) {
        __syncthreads();
        // ---- decode W tile: compressed gmem -> smem bf16 (once per CTA) ----
        for (int i = tid; i < n_elems; i += 256) {
            const int n = i >> 6, k = i & 63;         // BN=128,BK=64
            const long pos = (long)(n0 + n) * IN_F + k0 + k;
            const uint8_t code = idx[pos];
            const unsigned sgn_bit = (sign[pos >> 5] >> (pos & 31)) & 1;
            float v = CBs[code] * __half2float(scale[(unsigned)(pos >> 4)]);
            unsigned short bits = __bfloat16_as_ushort(__float2bfloat16(v));
            Ws[n][k] = __ushort_as_bfloat16(bits ^ ((sgn_bit ^ 1u) << 15));  // bit=1 == positive
        }
        // ---- load x tile -> smem (row-major [BM][BK]) ----
        // x is [M, IN_F]; tile rows m0..m0+BM, cols k0..k0+BK
        for (int i = tid; i < BM * BK; i += 256) {
            const int m = i >> 6, k = i & 63;
            const int gm = m0 + m;
            Xs[m][k] = (gm < M) ? x[(long)gm * IN_F + k0 + k]
                                : __float2bfloat16(0.f);
        }
        __syncthreads();

        // ---- mma over the tile: warp computes [wm*64..+64) x [wn*32..+32)
#pragma unroll
        for (int ki = 0; ki < 4; ki++) {              // BK=64 -> 4 k-steps
            const int kc = ki * 16;
#pragma unroll
            for (int mi = 0; mi < 4; mi++) {          // m16 tiles
                // A frag: 4 b32; lane l: g=l>>2, c=l&3
                const int g = lane >> 2, c = lane & 3;
                const int ar = wm * 64 + mi * 16 + g;
                unsigned a[4];
                // PTX m16n8k16 A layout: R0={A[g][2c],A[g][2c+1]},
                // R1={A[g+8][2c],..+1}, R2={A[g][2c+8],..+9},
                // R3={A[g+8][2c+8],..+9}
                a[0] = *(const unsigned*)&Xs[ar][kc + 2 * c];
                a[1] = *(const unsigned*)&Xs[ar + 8][kc + 2 * c];
                a[2] = *(const unsigned*)&Xs[ar][kc + 2 * c + 8];
                a[3] = *(const unsigned*)&Xs[ar + 8][kc + 2 * c + 8];
#pragma unroll
                for (int ni = 0; ni < 4; ni++) {      // n8 tiles
                    // B frag (KxN): lane l holds B[k=2c..][n=g]
                    const int br = wn * 32 + ni * 8 + g;
                    unsigned b[2];
                    b[0] = *(const unsigned*)&Ws[br][kc + 2 * c];
                    b[1] = *(const unsigned*)&Ws[br][kc + 2 * c + 8];
                    mma_bf16(acc[mi][ni], a, b);
                }
            }
        }
    }

    // ---- epilogue: acc -> y ; D m16n8: lane l: D[g][2c],[g][2c+1],[g+8][..]
    const int g = lane >> 2, c = lane & 3;
#pragma unroll
    for (int mi = 0; mi < 4; mi++) {
#pragma unroll
        for (int ni = 0; ni < 4; ni++) {
            const int mr = m0 + wm * 64 + mi * 16 + g;
            const int nc = n0 + wn * 32 + ni * 8 + 2 * c;
            if (mr < M) {
                float* yp;
                yp = (float*)&y[(long)mr * OUT_F + nc];
                y[(long)mr * OUT_F + nc]       = __float2bfloat16(acc[mi][ni][0]);
                y[(long)mr * OUT_F + nc + 1]   = __float2bfloat16(acc[mi][ni][1]);
                y[(long)(mr + 8) * OUT_F + nc]     = __float2bfloat16(acc[mi][ni][2]);
                y[(long)(mr + 8) * OUT_F + nc + 1] = __float2bfloat16(acc[mi][ni][3]);
                (void)yp;
            }
        }
    }
}

torch::Tensor udcq_mma(torch::Tensor x, torch::Tensor idx, torch::Tensor sign,
                       torch::Tensor scale, torch::Tensor cb,
                       int64_t out_f, int64_t in_f) {
    const int M = x.size(0);
    auto y = torch::empty({M, out_f}, x.options());
    dim3 grid((out_f + BN - 1) / BN, (M + BM - 1) / BM);
    udcq_mma_kernel<<<grid, NWARPS * 32, 0, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(y.data_ptr()),
        idx.data_ptr<uint8_t>(),
        sign.data_ptr<int>(),
        reinterpret_cast<const __half*>(scale.data_ptr()),
        reinterpret_cast<const __half*>(cb.data_ptr()),
        M, (int)out_f, (int)in_f);
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "launch failed: ", cudaGetErrorString(err));
    return y;
}
"""

CPP_SRC = "torch::Tensor udcq_mma(torch::Tensor x, torch::Tensor idx, torch::Tensor sign, torch::Tensor scale, torch::Tensor cb, int64_t out_f, int64_t in_f);"

mod = load_inline(
    name='udcq_mma_v1d',
    cpp_sources=CPP_SRC,
    cuda_sources=CUDA_SRC,
    functions=['udcq_mma'],
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    verbose=False,
)


def heavy(shape, seed):
    g = torch.Generator().manual_seed(seed)
    b = torch.randn(shape, generator=g) * 0.02
    o = torch.zeros(shape)
    n = int(o.numel() * 0.03)
    f = o.view(-1)
    i = torch.randperm(o.numel(), generator=g)[:n]
    f[i] = torch.randn(n, generator=g) * 0.15
    return (b + o).bfloat16()


def tmin(fn, it=30, reps=3):
    best = float('inf')
    e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
    for _ in range(reps):
        for _ in range(5):
            fn()
        e0.record()
        for _ in range(it):
            fn()
        e1.record()
        torch.cuda.synchronize()
        best = min(best, e0.elapsed_time(e1) / it)
    return best


print('== correctness ==')
for shape in [(2048, 1536), (6144, 1536), (1536, 4096)]:
    o_, i_ = shape
    w = heavy(shape, o_ + i_)
    cb = udcq_fit_codebook(w)
    p = udcq_quantize(w, cb)
    dev = 'cuda'
    w_ref = _decode_udcq_ref(p, device=dev)
    for M in (33, 256, 1024, 4096):
        x = torch.randn(M, i_, dtype=torch.bfloat16, device=dev)
        y = mod.udcq_mma(
            x, p['idx'].to(dev).reshape(-1),
            p['sign_packed'].to(dev).to(torch.int32),
            p['scale'].to(dev), p['codebook'].to(dev), o_, i_)
        y_ref = F.linear(x, w_ref)
        rel = ((y.float() - y_ref.float()).norm() / y_ref.float().norm()).item()
        assert rel < 5e-2, f'{shape} M={M}: rel {rel:.4f}'
    print(f'  {str(shape):>14s}: M=33..4096 OK')

print('\n== speed vs cublas bf16 (min-of-reps) ==')
print(f"{'shape':>14s} {'M':>5s} | {'cublas':>9s} | {'udcq-mma':>9s} | {'x':>5s}")
for shape in [(2048, 1536), (6144, 1536), (1536, 4096), (1536, 12288),
              (12288, 4096)]:
    o_, i_ = shape
    w = heavy(shape, o_ + i_).cuda()
    cb = udcq_fit_codebook(w.cpu())
    p = udcq_quantize(w.cpu(), cb)
    idx = p['idx'].cuda().reshape(-1)
    sg = p['sign_packed'].cuda().to(torch.int32)
    sc = p['scale'].cuda()
    cbg = p['codebook'].cuda()
    for M in (256, 1024, 4096):
        x = torch.randn(M, i_, dtype=torch.bfloat16, device='cuda')
        tc = tmin(lambda: F.linear(x, w))
        tu = tmin(lambda: mod.udcq_mma(x, idx, sg, sc, cbg, o_, i_))
        print(f'{str(shape):>14s} {M:5d} | {tc:7.3f}ms | {tu:7.3f}ms | {tc/tu:4.2f}')
    del w
    torch.cuda.empty_cache()
