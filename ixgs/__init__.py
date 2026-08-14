"""ixgs — Group-Scale INT8-X: per-group scales + (3,5,8) lossless encoding.

See README.md for the design rationale (MiniMax-H3 case study).
"""
from .quantize import int8gs_quantize, decode_weight_scatter, per_tensor_int8_reference
from .kernels import decode_weight_triton, has_triton
from .linear import Int8GSLinear, deploy_model_gs, iter_quantizable_linears

__all__ = [
    "int8gs_quantize",
    "decode_weight_scatter",
    "decode_weight_triton",
    "has_triton",
    "per_tensor_int8_reference",
    "Int8GSLinear",
    "deploy_model_gs",
    "iter_quantizable_linears",
]
