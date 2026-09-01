# -*- coding: utf-8 -*-
"""v2 (rewritten): cp.async double-buffer for the BIG streams only.

Staging via cp.async 16B chunks, double-buffered:
  - x tile [BM][BK] bf16 (row-contiguous, 16B = 8 elems chunks)  2x16KB
  - idx tile per W row (64B = 4 chunks/row, row-contiguous)      2x8KB
sign/scale are NOT staged: a 16B chunk of them spans W rows (BK=64 ->
4 scale groups / 2 sign words per row) whose global storage is not
contiguous -> the v2a crash. They are read directly from gmem in the
decode loop (tiny, broadcast-heavy, L1-cached).
Decode: smem idx + gmem sign/scale -> Ws bf16 smem (once per CTA/k-tile).
mma: same as v1. Total smem = 2*16 + 2*8 + 16 + 64B = ~56.1KB.
"""
import sys, time, torch
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

__device__ __forceinline__ void mma_bf16(float* d, const unsigned* a,
                                         const unsigned* b) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

__device__ __forceinline__ void cp_async16(void* smem, const void* gmem) {
    unsigned s = (unsigned)__cvta_generic_to_shared(smem);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(s), "l"(gmem));
}

__global__ __launch_bounds__(256) void udcq_mma_v2_kernel(
    const __nv_bfloat16* __restrict__ x,
    __nv_bfloat16* __restrict__ y,
    const uint8_t* __restrict__ idx,
    const int* __restrict__ sign,
    const __half* __restrict__ scale,
    const __half* __restrict__ cb,
    int M, int OUT_F, int IN_F)
{
    extern __shared__ unsigned char smem_raw[];
    __nv_bfloat16* Xs[2] = {(__nv_bfloat16*)smem_raw,
                            (__nv_bfloat16*)(smem_raw + BM * BK * 2)};
    uint8_t* Is[2] = {smem_raw + 2 * BM * BK * 2,
                      smem_raw + 2 * BM * BK * 2 + BN * BK};
    __nv_bfloat16* Ws = (__nv_bfloat16*)(smem_raw + 2 * BM * BK * 2 + 2 * BN * BK);
    float* CBs = (float*)(Ws + BN * BK);

    const int tid = threadIdx.x;
    const int n0 = blockIdx.x * BN;
    const int m0 = blockIdx.y * BM;
    const int warp = tid >> 5, lane = tid & 31;
    const int wm = warp >> 2, wn = warp & 3;

    if (tid < 16) CBs[tid] = __half2float(cb[tid]);
    __syncthreads();

    // stage x tile (16B chunks, row-contiguous) into buf s at k-tile k0
    auto stage_x = [&](int k0, int s) {
        for (int i = tid * 16; i < BM * BK * 2; i += 256 * 16) {
            const int m = i / (BK * 2);
            const int gm = m0 + m;
            if (gm < M) {
                const __nv_bfloat16* g = x + (long)gm * IN_F + k0 + (i % (BK * 2)) / 2;
                cp_async16(((unsigned char*)Xs[s]) + i, g);
            } else {
                *(uint4*)(((unsigned char*)Xs[s]) + i) = make_uint4(0, 0, 0, 0);
            }
        }
    };
    // stage idx tile (row-contiguous 64B rows) into buf s
    auto stage_idx = [&](int k0, int s) {
        for (int i = tid * 16; i < BN * BK; i += 256 * 16) {
            const int n = i / BK;
            cp_async16(Is[s] + i, idx + (long)(n0 + n) * IN_F + k0 + (i % BK));
        }
    };
    auto commit = [&]() { asm volatile("cp.async.commit_group;\n"); };
    auto wait = [&]() { asm volatile("cp.async.wait_group 0;\n"); };

    const int KT = IN_F / BK;
    stage_x(0, 0);
    stage_idx(0, 0);
    commit();

    float acc[4][4][4];
#pragma unroll
    for (int i = 0; i < 4; i++)
#pragma unroll
        for (int j = 0; j < 4; j++)
#pragma unroll
            for (int r = 0; r < 4; r++) acc[i][j][r] = 0.f;

    for (int kt = 0; kt < KT; kt++) {
        const int s = kt & 1;
        const int k0 = kt * BK;
        if (kt + 1 < KT) {
            stage_x(k0 + BK, s ^ 1);
            stage_idx(k0 + BK, s ^ 1);
        }
        commit();                       // (prefetch group OR empty group)
        wait();                         // wait ALL (both groups done)
        __syncthreads();

        // ---- decode: smem idx + gmem sign/scale -> Ws ----
        for (int i = tid; i < BN * BK; i += 256) {
            const int n = i >> 6, k = i & 63;
            const long pos = (long)(n0 + n) * IN_F + k0 + k;
            const float v = CBs[Is[s][i]] * __half2float(scale[(unsigned)(pos >> 4)]);
            const unsigned sgn = ((unsigned)sign[pos >> 5] >> (pos & 31)) & 1;
            const unsigned bits = __bfloat16_as_ushort(__float2bfloat16(v));
            Ws[i] = __ushort_as_bfloat16(bits ^ ((sgn ^ 1u) << 15));
        }
        __syncthreads();

        // ---- mma ----
        const int g = lane >> 2, c = lane & 3;
#pragma unroll
        for (int ki = 0; ki < 4; ki++) {
            const int kc = ki * 16;
#pragma unroll
            for (int mi = 0; mi < 4; mi++) {
                const int ar = (wm * 64 + mi * 16 + g) * BK + kc + 2 * c;
                unsigned a[4];
                a[0] = *(const unsigned*)&Xs[s][ar];
                a[1] = *(const unsigned*)&Xs[s][ar + 8 * BK];
                a[2] = *(const unsigned*)&Xs[s][ar + 8];
                a[3] = *(const unsigned*)&Xs[s][ar + 8 * BK + 8];
#pragma unroll
                for (int ni = 0; ni < 4; ni++) {
                    const int br = (wn * 32 + ni * 8 + g) * BK + kc + 2 * c;
                    unsigned b[2];
                    b[0] = *(const unsigned*)&Ws[br];
                    b[1] = *(const unsigned*)&Ws[br + 8];
                    mma_bf16(acc[mi][ni], a, b);
                }
            }
        }
        __syncthreads();
    }

    const int g2 = lane >> 2, c2 = lane & 3;
#pragma unroll
    for (int mi = 0; mi < 4; mi++)
#pragma unroll
        for (int ni = 0; ni < 4; ni++) {
            const int mr = m0 + wm * 64 + mi * 16 + g2;
            const int nc = n0 + wn * 32 + ni * 8 + 2 * c2;
            if (mr < M) {
                y[(long)mr * OUT_F + nc] = __float2bfloat16(acc[mi][ni][0]);
                y[(long)mr * OUT_F + nc + 1] = __float2bfloat16(acc[mi][ni][1]);
                y[(long)(mr + 8) * OUT_F + nc] = __float2bfloat16(acc[mi][ni][2]);
                y[(long)(mr + 8) * OUT_F + nc + 1] = __float2bfloat16(acc[mi][ni][3]);
            }
        }
}

torch::Tensor udcq_mma_v2(torch::Tensor x, torch::Tensor idx, torch::Tensor sign,
                          torch::Tensor scale, torch::Tensor cb,
                          int64_t out_f, int64_t in_f) {
    const int M = x.size(0);
    auto y = torch::empty({M, out_f}, x.options());
    const int smem = 2 * BM * BK * 2 + 2 * BN * BK + BN * BK * 2 + 64;
    static bool cfg = false;
    if (!cfg) {
        cudaFuncSetAttribute(udcq_mma_v2_kernel,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
        cfg = true;
    }
    dim3 grid((out_f + BN - 1) / BN, (M + BM - 1) / BM);
    udcq_mma_v2_kernel<<<grid, 256, smem, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(y.data_ptr()),
        idx.data_ptr<uint8_t>(), sign.data_ptr<int>(),
        reinterpret_cast<const __half*>(scale.data_ptr()),
        reinterpret_cast<const __half*>(cb.data_ptr()),
        M, (int)out_f, (int)in_f);
    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "v2 launch failed");
    return y;
}
"""

CPP_SRC = "torch::Tensor udcq_mma_v2(torch::Tensor x, torch::Tensor idx, torch::Tensor sign, torch::Tensor scale, torch::Tensor cb, int64_t out_f, int64_t in_f);"

mod = load_inline(name='udcq_mma_v2r', cpp_sources=CPP_SRC, cuda_sources=CUDA_SRC,
                  functions=['udcq_mma_v2'],
                  extra_cuda_cflags=['-O3', '--use_fast_math'], verbose=False)


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


print('== v2 correctness ==')
for shape in [(2048, 1536), (6144, 1536), (1536, 4096)]:
    o_, i_ = shape
    w = heavy(shape, o_ + i_)
    cb = udcq_fit_codebook(w)
    p = udcq_quantize(w, cb)
    w_ref = _decode_udcq_ref(p, device='cuda')
    for M in (33, 256, 1024, 4096):
        x = torch.randn(M, i_, dtype=torch.bfloat16, device='cuda')
        y = mod.udcq_mma_v2(x, p['idx'].cuda().reshape(-1),
                            p['sign_packed'].cuda().to(torch.int32),
                            p['scale'].cuda(), p['codebook'].cuda(), o_, i_)
        y_ref = F.linear(x, w_ref)
        rel = ((y.float() - y_ref.float()).norm() / y_ref.float().norm()).item()
        assert rel < 5e-2, f'{shape} M={M}: rel {rel:.4f}'
    print(f'  {str(shape):>14s}: OK')
    del w_ref
    torch.cuda.empty_cache()

print('\n== v2 speed vs cublas ==')
print(f"{'shape':>14s} {'M':>5s} | {'cublas':>9s} | {'v2':>8s} | {'x':>5s}")
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
        tu = tmin(lambda: mod.udcq_mma_v2(x, idx, sg, sc, cbg, o_, i_))
        print(f'{str(shape):>14s} {M:5d} | {tc:7.3f}ms | {tu:6.3f}ms | {tc/tu:4.2f}')
    del w
    torch.cuda.empty_cache()


print('\n== M crossover sweep ==')
for shape in [(6144, 1536), (12288, 4096)]:
    o_, i_ = shape
    w = heavy(shape, o_ + i_).cuda()
    cb = udcq_fit_codebook(w.cpu())
    p = udcq_quantize(w.cpu(), cb)
    idx = p['idx'].cuda().reshape(-1)
    sg = p['sign_packed'].cuda().to(torch.int32)
    sc = p['scale'].cuda()
    cbg = p['codebook'].cuda()
    for M in (128, 512, 768):
        x = torch.randn(M, i_, dtype=torch.bfloat16, device='cuda')
        tc = tmin(lambda: F.linear(x, w))
        tu = tmin(lambda: mod.udcq_mma_v2(x, idx, sg, sc, cbg, o_, i_))
        print(f'  {str(shape):>14s} M={M:4d}: cublas {tc:.3f}ms  v2 {tu:.3f}ms  {tc/tu:.2f}x')
    del w
    torch.cuda.empty_cache()
