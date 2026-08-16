"""TpabLinear: nn.Linear replacement using TPAB-compressed weights.

Two paths (mirrors StreamingLinear):
  * multi-token / prefill: full decode into a persistent f32 workspace via
    the tile-parallel kernel + outlier scatter, then F.linear
  * single-token decode step (generation hot loop): fused_gemv_tpab —
    weights decoded in-registers per row, never materialized
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

from .tpab import encode_tpab, decode_tpab_triton, stage_gpu
from .tpab_gemv import fused_gemv_tpab, prepare_gemv_stage

__all__ = ["TpabLinear", "deploy_model_tpab"]

# shared f32 prefill-decode workspace: one buffer sized to the largest layer
# serves ALL TpabLinear instances (prefill decodes sequentially anyway)
_SHARED_WS = None
_SHARED_WS_SIZE = 0


def _get_shared_ws(n_elems: int, device) -> torch.Tensor:
    global _SHARED_WS, _SHARED_WS_SIZE
    if _SHARED_WS is None or _SHARED_WS_SIZE < n_elems:
        _SHARED_WS = torch.zeros(n_elems, dtype=torch.float32, device=device)
        _SHARED_WS_SIZE = n_elems
    return _SHARED_WS


class TpabLinear(nn.Module):
    def __init__(self, weight: torch.Tensor, snr_target_db=26.0, outlier_frac=0.004):
        super().__init__()
        self.out_features, self.in_features = weight.shape
        self.packed = encode_tpab(weight.detach(), snr_target_db=snr_target_db,
                                  outlier_frac=outlier_frac)
        self.staged = stage_gpu(self.packed, "cuda")
        self.gemv_stage = prepare_gemv_stage(self.packed, "cuda")

    def forward(self, x):
        if x.dtype == torch.bfloat16 and x.numel() == self.in_features:
            y = fused_gemv_tpab(x, self.gemv_stage, self.out_features, self.in_features)
            return y.view(x.shape[:-1] + (self.out_features,))
        # multi-token: decode into the SHARED workspace, then F.linear
        ws = _get_shared_ws(self.packed["T"] * self.packed["n_per"], x.device)
        w = decode_tpab_triton(self.packed, device=x.device, out_f32=ws,
                               staged=self.staged)
        return F.linear(x, w, None)

    def extra_repr(self):
        return (f"{self.in_features}x{self.out_features}, "
                f"bpw={self.packed['bpw']:.2f}")


@torch.no_grad()
def deploy_model_tpab(model, snr_target_db=26.0, outlier_frac=0.004, verbose=True):
    """Replace every quantizable Linear with TpabLinear (in-place)."""
    from .linear import iter_quantizable_linears, _set_parent_child

    targets = list(iter_quantizable_linears(model))
    total_bytes = 0
    n_elems = 0
    n_skip = 0
    for name, mod in targets:
        O, I = mod.weight.shape
        if O % 64 or I % 64:
            n_skip += 1
            continue
        tl = TpabLinear(mod.weight.data, snr_target_db, outlier_frac)
        _set_parent_child(model, name, tl)
        total_bytes += tl.packed["total_bytes"]
        n_elems += O * I
    stats = {
        "n_layers": len(targets) - n_skip,
        "skipped": n_skip,
        "total_bytes": total_bytes,
        "bpw": total_bytes * 8 / max(n_elems, 1),
    }
    if verbose:
        print(f"[tpab-deploy] {stats['n_layers']} layers "
              f"({n_skip} skipped non-64-divisible) | "
              f"{total_bytes/1e6:.0f}MB | bpw={stats['bpw']:.2f}", flush=True)
    return stats
