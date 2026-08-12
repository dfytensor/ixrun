"""IXRUN: INT8-X inference engine for LLM text generation.

Subpackages:
  bitpack       bit-stream packing / unpacking
  quantize      bf16 -> int8 -> nested-bitmap (3,5,8) packed representation
  search        exhaustive analysis of optimal level-bit combinations
  triton_kernels fused GPU decode kernel + PyTorch fallback
  linear        Int8XLinear deploy layer (cached + live decode)
  engine        streaming inference + resource scheduler
  generate      text generation with streaming token output
"""
from .quantize import int8x_quantize, DEFAULT_LEVELS
from .search import search_optimal_levels
from .linear import Int8XLinear, deploy_model
from .engine import Int8XEngine
from .generate import generate_text, stream_generate

__version__ = "0.1.0"

__all__ = [
    "int8x_quantize",
    "DEFAULT_LEVELS",
    "search_optimal_levels",
    "Int8XLinear",
    "deploy_model",
    "Int8XEngine",
    "generate_text",
    "stream_generate",
    "__version__",
]
