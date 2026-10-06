# -*- coding: utf-8 -*-
"""Minimal CUDA graph test: single kernel, verify input update works."""
import sys
sys.path.insert(0, r'E:\IXRUN')
import pandas  # noqa: F401
import torch
from torch.utils.cpp_extension import load_inline

cuda_src = r"""
#include <torch/extension.h>
#include <cuda_bf16.h>
#include <ATen/cuda/CUDAContext.h>

__global__ void copy_kernel(const float* src, float* dst, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = src[i] * 2.0f;  // scale by 2 for visibility
}

static cudaGraphExec_t g_exec = nullptr;
static torch::Tensor g_out;

void mini_capture(torch::Tensor input) {
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    g_out = torch::empty({input.numel()},
        torch::TensorOptions().dtype(torch::kFloat32)
            .device(input.device()));
    // warmup
    copy_kernel<<<1, 256, 0, stream>>>(
        input.data_ptr<float>(), g_out.data_ptr<float>(),
        (int)input.numel());
    cudaStreamSynchronize(stream);
    // capture
    cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
    copy_kernel<<<1, 256, 0, stream>>>(
        input.data_ptr<float>(), g_out.data_ptr<float>(),
        (int)input.numel());
    cudaGraph_t graph;
    cudaStreamEndCapture(stream, &graph);
    cudaGraphInstantiate(&g_exec, graph, NULL, NULL, 0);
}

torch::Tensor mini_replay() {
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    cudaGraphLaunch(g_exec, stream);
    return g_out;
}
"""

ext = load_inline(
    name='mini_graph_test',
    cpp_sources=['void mini_capture(torch::Tensor input);\n'
                 'torch::Tensor mini_replay();'],
    cuda_sources=[cuda_src],
    functions=['mini_capture', 'mini_replay'],
    extra_cuda_cflags=['-O3'],
    verbose=False)

# Test
inp = torch.tensor([1.0, 2.0, 3.0, 4.0], device='cuda')
ext.mini_capture(inp)
print(f'capture-time input: {inp.tolist()}')

r = ext.mini_replay()
torch.cuda.synchronize()
print(f'replay 1 output:    {r.tolist()}')  # should be [2,4,6,8]

# Update input in-place
inp.copy_(torch.tensor([10.0, 20.0, 30.0, 40.0], device='cuda'))
print(f'updated input:      {inp.tolist()}')

r = ext.mini_replay()
torch.cuda.synchronize()
print(f'replay 2 output:    {r.tolist()}')  # should be [20,40,60,80]

ok = r[0].item() == 20.0
print('GRAPH INPUT UPDATE:', 'WORKS' if ok else 'BROKEN')
