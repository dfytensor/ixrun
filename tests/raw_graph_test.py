# -*- coding: utf-8 -*-
"""Test: raw cudaMalloc CUDA graph input update on WDDM."""
import sys
sys.path.insert(0, r'E:\IXRUN')
import pandas  # noqa: F401
import torch
from torch.utils.cpp_extension import load_inline

src = open(r'E:\IXRUN\ixrun\cpp\engine_v5.cu',
           encoding='utf-8').read()
# Just extract the raw graph section
proto = '''
void raw_graph_init(int64_t n);
void raw_graph_set_input(torch::Tensor data);
torch::Tensor raw_graph_get_output();
void raw_graph_capture();
void raw_graph_replay();
'''
ext = load_inline(name='raw_graph_test', cpp_sources=[proto],
                  cuda_sources=[src],
                  functions=['raw_graph_init', 'raw_graph_set_input',
                             'raw_graph_get_output',
                             'raw_graph_capture',
                             'raw_graph_replay'],
                  extra_cuda_cflags=['-O3'],
                  verbose=False)

N = 4
ext.raw_graph_init(N)

# Set initial data [1,2,3,4]
inp = torch.tensor([1.0, 2.0, 3.0, 4.0], device='cuda')
ext.raw_graph_set_input(inp)
ext.raw_graph_capture()

r = ext.raw_graph_get_output()
torch.cuda.synchronize()
print(f'capture input [1,2,3,4] -> output: {r.tolist()}')  # [2,4,6,8]

# Update to [10,20,30,40]
new = torch.tensor([10.0, 20.0, 30.0, 40.0], device='cuda')
ext.raw_graph_set_input(new)
r = ext.raw_graph_replay()
torch.cuda.synchronize()
out = ext.raw_graph_get_output()
print(f'updated [10,20,30,40] -> output: {out.tolist()}')  # [20,40,60,80]?

ok = out[0].item() == 20.0
print(f'RAW cudaMalloc GRAPH: {"WORKS!" if ok else "STILL BROKEN"}')
