# -*- coding: utf-8 -*-
"""Hand-CUDA GEMV for HPQ-x-scale (m=8 k=64 L2, 4x4 blocks + scale).

codes uint8 [nB,16] (bidx*16 + l*8 + s) | cb fp16 [2,8,64,2] -> smem
float2 [16][64] | scale fp16 [nB] | x/y bf16 (use __nv_bfloat16).
Grid = (out_f/4/wpb, n_sp): 8 warps/block, each warp one 4-row group,
split-K over col-blocks with fp32 atomicAdd; one uint4 = 16 codes.
"""
import sys

import torch
from torch.utils.cpp_extension import load_inline

_CUDA_SRC = r"""
#include <torch/extension.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cstdint>

__global__ void hpqs_gemv_kernel(
    const __nv_bfloat16* __restrict__ x,
    const uint8_t* __restrict__ codes,
    const __half2* __restrict__ cb,
    const __half* __restrict__ scale,
    float* __restrict__ yf,
    int n_bc, int n_sp, int warps_per_blk)
{
    __shared__ float2 cb_sm[16][64];
    int lane = threadIdx.x & 31;
    for (int i = threadIdx.x; i < 16 * 64; i += blockDim.x) {
        cb_sm[i / 64][i % 64] = __half22float2(cb[i]);
    }
    __syncthreads();
    int g = blockIdx.x * warps_per_blk + (threadIdx.x >> 5);
    int jb0 = blockIdx.y * n_bc;
    int jb1 = jb0 + n_bc;
    float acc0 = 0.f, acc1 = 0.f, acc2 = 0.f, acc3 = 0.f;
    for (int jb = jb0 + lane; jb < jb1; jb += 32) {
        int bidx = g * (n_bc * n_sp) + jb;
        uint4 cw = *reinterpret_cast<const uint4*>(codes + bidx * 16);
        uint8_t cA[8], cB[8];
        memcpy(cA, &cw.x, 4); memcpy(cA + 4, &cw.y, 4);
        memcpy(cB, &cw.z, 4); memcpy(cB + 4, &cw.w, 4);
        float sc = __half2float(scale[bidx]);
        const __nv_bfloat16* x4 = x + jb * 4;
        float xa0 = __bfloat162float(x4[0]);
        float xa1 = __bfloat162float(x4[1]);
        float xa2 = __bfloat162float(x4[2]);
        float xa3 = __bfloat162float(x4[3]);
        #pragma unroll
        for (int r = 0; r < 4; ++r) {
            int s0 = r * 2, s1 = r * 2 + 1;
            float2 a0 = cb_sm[s0][cA[s0]];
            float2 a1 = cb_sm[s1][cA[s1]];
            float2 b0v = cb_sm[8 + s0][cB[s0]];
            float2 b1v = cb_sm[8 + s1][cB[s1]];
            float t = ((a0.x + b0v.x) * xa0
                     + (a0.y + b0v.y) * xa1
                     + (a1.x + b1v.x) * xa2
                     + (a1.y + b1v.y) * xa3) * sc;
            if (r == 0) acc0 += t;
            else if (r == 1) acc1 += t;
            else if (r == 2) acc2 += t;
            else acc3 += t;
        }
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        acc0 += __shfl_down_sync(0xffffffff, acc0, off);
        acc1 += __shfl_down_sync(0xffffffff, acc1, off);
        acc2 += __shfl_down_sync(0xffffffff, acc2, off);
        acc3 += __shfl_down_sync(0xffffffff, acc3, off);
    }
    if (lane == 0) {
        atomicAdd(&yf[g * 4 + 0], acc0);
        atomicAdd(&yf[g * 4 + 1], acc1);
        atomicAdd(&yf[g * 4 + 2], acc2);
        atomicAdd(&yf[g * 4 + 3], acc3);
    }
}

torch::Tensor gemv(torch::Tensor x, torch::Tensor codes,
                   torch::Tensor cb, torch::Tensor scale,
                   int64_t out_f, int64_t in_f)
{
    auto yf = torch::zeros({out_f},
        torch::dtype(torch::kFloat32).device(x.device()));
    int n_bc_tot = (int)(in_f / 4);
    int n_sp = n_bc_tot / 256; if (n_sp < 1) n_sp = 1;
    if (n_sp > 8) n_sp = 8;
    int n_bc = n_bc_tot / n_sp;
    int wpb = 8;
    unsigned gx = (unsigned)(out_f / 4 / wpb);
    dim3 grid(gx, (unsigned)n_sp);
    hpqs_gemv_kernel<<<grid, wpb * 32>>>(
        reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()),
        codes.data_ptr<uint8_t>(),
        reinterpret_cast<const __half2*>(cb.data_ptr()),
        reinterpret_cast<const __half*>(scale.data_ptr()),
        yf.data_ptr<float>(), n_bc, n_sp, wpb);
    return yf.to(torch::kBFloat16);
}
"""

_CPP_SRC = """
torch::Tensor gemv(torch::Tensor x, torch::Tensor codes,
                   torch::Tensor cb, torch::Tensor scale,
                   int64_t out_f, int64_t in_f);
"""

_EXT = None


def _load():
    global _EXT
    if _EXT is None:
        _EXT = load_inline(
            name='hpqs_gemv_cuda_v2',
            cpp_sources=[_CPP_SRC],
            cuda_sources=[_CUDA_SRC],
            functions=['gemv'],
            extra_cuda_cflags=['-O3', '--use_fast_math',
                               '-allow-unsupported-compiler'],
            verbose=False)
    return _EXT


def hpqs_gemv_cuda(x, packed):
    """x [in_f] bf16 -> y [out_f] bf16. out_f % 32 == 0 required."""
    ext = _load()
    return ext.gemv(x.contiguous(), packed['codes'].cuda(),
                    packed['cb'].cuda(), packed['scale'].cuda(),
                    packed['out_f'], packed['in_f'])


if __name__ == '__main__':
    import time
    sys.path.insert(0, r'E:\IXRUN')
    from benchmarks.hpqs_runtime import hpqs_pack, _decode_ref
    torch.manual_seed(0)
    for of, inf in [(512, 512), (2048, 6144)]:
        W = (torch.randn(of, inf) * 0.02).cuda()
        pk = hpqs_pack(W)
        dref = _decode_ref(pk).cuda()
        pk['codes'] = pk['codes'].cuda()
        pk['cb'] = pk['cb'].cuda()
        pk['scale'] = pk['scale'].cuda()
        x = torch.randn(inf, dtype=torch.bfloat16, device='cuda')
        y = hpqs_gemv_cuda(x, pk)
        y_ref = (dref.float() @ x.float()).to(torch.bfloat16)
        gmax = (y.float() - y_ref.float()).abs().max().item()
        for _ in range(5):
            hpqs_gemv_cuda(x, pk)
        torch.cuda.synchronize()
        t0 = time.time()
        reps = 200
        for _ in range(reps):
            hpqs_gemv_cuda(x, pk)
        torch.cuda.synchronize()
        t_h = (time.time() - t0) / reps * 1000
        Wb = dref.float()
        for _ in range(5):
            Wb @ x.float()
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(reps):
            Wb @ x.float()
        torch.cuda.synchronize()
        t_b = (time.time() - t0) / reps * 1000
        nB = (of // 4) * (inf // 4)
        print(f'[{of}x{inf}] gmax={gmax:.4f} '
              f'cuda={t_h:.3f}ms bf16={t_b:.3f}ms '
              f'({t_b/t_h:.2f}x, {nB*18/t_h/1e6:.0f}GB/s eff)',
              flush=True)
