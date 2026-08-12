"""Int8XLinear: drop-in replacement for nn.Linear with INT8-X packed weights.

Two decode strategies:

  * ``cache='full'`` (default): decode the whole weight to bf16 once at deploy
    time and run plain F.linear afterwards. Fastest inference; the compression
    benefit is in *storage* (you can keep the packed dict and drop the bf16).

  * ``cache='none'``: re-decode every forward into a caller-supplied shared
    GPU buffer. Minimises resident GPU memory at the cost of decode bandwidth;
    used by the streaming engine (:class:`ixrun.engine.Int8XEngine`).
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

from .quantize import int8x_quantize
from .triton_kernels import decode_weight_triton, decode_weight_scatter
from .config import SKIP_PATTERNS, MIN_LINEAR_ELEMS

__all__ = ["Int8XLinear", "deploy_model", "iter_quantizable_linears"]


def iter_quantizable_linears(model: nn.Module):
    """Yield (name, module) for every Linear that should be quantized."""
    for name, mod in model.named_modules():
        if not isinstance(mod, nn.Linear):
            continue
        if mod.weight.numel() < MIN_LINEAR_ELEMS:
            continue
        if any(s in name for s in SKIP_PATTERNS):
            continue
        yield name, mod


def _set_parent_child(model: nn.Module, dotted_name: str, new_module: nn.Module):
    parts = dotted_name.split(".")
    parent = model
    for p in parts[:-1]:
        parent = getattr(parent, p)
    setattr(parent, parts[-1], new_module)


class Int8XLinear(nn.Module):
    """nn.Linear backed by an INT8-X packed weight.

    Parameters
    ----------
    packed : dict from :func:`int8x_quantize`.
    bias : optional bias tensor.
    cache : 'full' to decode once and keep bf16 weight; 'none' to re-decode
        every forward (needs ``shared_buf`` set).
    use_triton : prefer the fused Triton decode kernel.
    """

    def __init__(
        self,
        packed: dict,
        bias=None,
        cache: str = "full",
        use_triton: bool = True,
    ):
        super().__init__()
        self.out_features = packed["out_f"]
        self.in_features = packed["in_f"]
        self.packed = packed
        self.use_triton = use_triton
        self._cache = cache
        self._bias = None
        if bias is not None:
            self.register_buffer(
                "_bias_buf", bias.detach().clone()
            )
        else:
            self._bias_buf = None

        if cache == "full":
            self._w = self._decode(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        else:
            self._w = None

    def _decode(self, device):
        if self.use_triton and torch.cuda.is_available():
            return decode_weight_triton(self.packed, device)
        return decode_weight_scatter(self.packed, device)

    def forward(self, x):
        if self._cache == "full":
            w = self._w
        else:
            w = self._decode(x.device)
        bias = self._bias_buf.to(x.dtype) if self._bias_buf is not None else None
        return F.linear(x, w, bias)

    def extra_repr(self) -> str:
        return (
            f"in={self.in_features}, out={self.out_features}, "
            f"levels={self.packed['level_bits']}, bpw={self.packed['bits_per_weight']:.2f}"
        )


@torch.no_grad()
def deploy_model(
    model: nn.Module,
    level_bits=(3, 5, 8),
    cache: str = "full",
    use_triton: bool = True,
    verbose: bool = True,
) -> dict:
    """Replace every quantizable Linear in *model* with Int8XLinear (in-place).

    Returns a stats dict: n_layers, total_bytes, counts, bits_per_weight.
    """
    targets = list(iter_quantizable_linears(model))
    total_bytes = 0
    n_elems = 0
    counts = [0, 0, 0]
    for name, mod in targets:
        packed = int8x_quantize(mod.weight.data, level_bits)
        bias = mod.bias.data if mod.bias is not None else None
        new_layer = Int8XLinear(packed, bias=bias, cache=cache, use_triton=use_triton)
        _set_parent_child(model, name, new_layer)
        total_bytes += packed["total_bytes"]
        n_elems += packed["N"]
        for i, c in enumerate(packed["counts"]):
            counts[i] += c
    stats = {
        "n_layers": len(targets),
        "total_bytes": total_bytes,
        "n_elems": n_elems,
        "counts": counts,
        "bits_per_weight": (total_bytes * 8) / max(n_elems, 1),
        "compression_vs_bf16": (n_elems * 2) / max(total_bytes, 1),
    }
    if verbose:
        N = sum(counts)
        print(
            f"[deploy] {stats['n_layers']} layers | "
            f"{total_bytes / 1e6:.0f}MB packed ({stats['compression_vs_bf16']:.2f}x vs bf16) | "
            f"bpw={stats['bits_per_weight']:.2f} | "
            f"L1={counts[0]/N*100:.1f}% L2={counts[1]/N*100:.1f}% L3={counts[2]/N*100:.1f}%",
            flush=True,
        )
    return stats
