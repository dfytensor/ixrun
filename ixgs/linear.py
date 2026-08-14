"""Int8GSLinear: nn.Linear replacement backed by Group-Scale INT8-X weights."""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

from .quantize import int8gs_quantize
from .kernels import decode_weight_triton, has_triton
from .quantize import decode_weight_scatter

__all__ = ["Int8GSLinear", "deploy_model_gs", "iter_quantizable_linears"]

SKIP_PATTERNS = ("lm_head", "embed", "shared", "wte")
MIN_LINEAR_ELEMS = 1000


def iter_quantizable_linears(model: nn.Module):
    for name, mod in model.named_modules():
        if not isinstance(mod, nn.Linear):
            continue
        if mod.weight.numel() < MIN_LINEAR_ELEMS:
            continue
        if any(s in name for s in SKIP_PATTERNS):
            continue
        yield name, mod


def _decode(packed: dict, device, use_triton: bool = True) -> torch.Tensor:
    if use_triton and has_triton():
        return decode_weight_triton(packed, device)
    return decode_weight_scatter(packed, device)


class Int8GSLinear(nn.Module):
    """Drop-in nn.Linear with group-scale INT8-X packed weights.

    cache='full': decode once at deploy, plain F.linear afterwards.
    cache='none': re-decode every forward (streaming / low-VRAM mode).
    """

    def __init__(self, packed: dict, bias=None, cache: str = "full", use_triton: bool = True):
        super().__init__()
        self.out_features = packed["out_f"]
        self.in_features = packed["in_f"]
        self.packed = packed
        self._cache = cache
        self._use_triton = use_triton
        if bias is not None:
            self.register_buffer("_bias_buf", bias.detach().clone())
        else:
            self._bias_buf = None
        if cache == "full":
            dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self._w = _decode(packed, dev, use_triton)
        else:
            self._w = None

    def forward(self, x):
        w = self._w if self._cache == "full" else _decode(self.packed, x.device, self._use_triton)
        bias = self._bias_buf.to(x.dtype) if self._bias_buf is not None else None
        return F.linear(x, w, bias)

    def extra_repr(self) -> str:
        p = self.packed
        return (
            f"in={self.in_features}, out={self.out_features}, "
            f"gs={p['group_size']}, counts={p['counts']}, "
            f"bpw={p['bits_per_weight']:.2f}"
        )


@torch.no_grad()
def deploy_model_gs(
    model: nn.Module,
    group_size: int = 64,
    cache: str = "full",
    use_triton: bool = True,
    verbose: bool = True,
) -> dict:
    """Replace every quantizable Linear with Int8GSLinear (in-place)."""
    targets = list(iter_quantizable_linears(model))
    total_bytes = 0
    n_elems = 0
    counts = [0, 0, 0]
    for name, mod in targets:
        packed = int8gs_quantize(mod.weight.data, group_size)
        bias = mod.bias.data if mod.bias is not None else None
        new_layer = Int8GSLinear(packed, bias=bias, cache=cache, use_triton=use_triton)
        parts = name.split(".")
        parent = model
        for p in parts[:-1]:
            parent = getattr(parent, p)
        setattr(parent, parts[-1], new_layer)
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
    if verbose and n_elems:
        print(
            f"[deploy-gs] {stats['n_layers']} layers | "
            f"{total_bytes / 1e6:.0f}MB ({stats['compression_vs_bf16']:.2f}x vs bf16) | "
            f"bpw={stats['bits_per_weight']:.2f} | "
            f"L1={counts[0]/n_elems*100:.1f}% L2={counts[1]/n_elems*100:.1f}% L3={counts[2]/n_elems*100:.1f}%",
            flush=True,
        )
    return stats
