# -*- coding: utf-8 -*-
"""Compile-chain smoke test: torch load_inline CUDA on Windows."""
import torch
from torch.utils.cpp_extension import load_inline

cuda_src = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>

__global__ void add_one_kernel(const __nv_bfloat16* x, __nv_bfloat16* y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = __hadd(x[i], __float2bfloat16(1.0f));
}

torch::Tensor add_one(torch::Tensor x) {
    auto y = torch::empty_like(x);
    int n = x.numel();
    add_one_kernel<<<(n + 255) / 256, 256, 0,
                     at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(y.data_ptr()), n);
    return y;
}
"""

cpp_src = "torch::Tensor add_one(torch::Tensor x);"

mod = load_inline(
    name='smoke_test',
    cpp_sources=cpp_src,
    cuda_sources=cuda_src,
    functions=['add_one'],
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    verbose=False,
)
x = torch.tensor([1.0, 2.0, 3.0], dtype=torch.bfloat16, device='cuda')
y = mod.add_one(x)
print('smoke OK:', y.float().tolist())
