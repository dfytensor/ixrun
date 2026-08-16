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
        # encode on CPU: deploy-time VRAM stays flat (the search temporaries
        # are ~3x the packed size per layer; on GPU they stacked on top of
        # the still-resident bf16 weights and pushed peaks +1.7GB)
        w = weight.detach()
        was_cuda = w.is_cuda
        self.packed = encode_tpab(w.cpu() if was_cuda else w,
                                  snr_target_db=snr_target_db,
                                  outlier_frac=outlier_frac)
        self.tile_r = self.packed["tile_r"]
        self.staged = stage_gpu(self.packed, "cuda")
        self.gemv_stage = prepare_gemv_stage(self.packed, "cuda", staged=self.staged)
        # free the CPU-side packed bodies (GPU copies exist; keeping them
        # costs ~20x the model's packed size in host RAM on 27B and OOMs)
        self.packed = {k: v for k, v in self.packed.items() if k != "bodies"}
        # split-K routing: only LARGE layers benefit (sweep data: tall
        # 17408x5120 -> 1.74x, wide 5120x17408 -> 1.27x, but small layers
        # lose ~2x to the 4-op overhead of zero/atomics/overlay/cast —
        # MiniCPM5 regressed 35 -> 62 ms/tok when split unconditionally)
        t_c = self.in_features // 64
        if self.in_features >= 8192 and t_c % 2 == 0:
            self._split = 2
        elif self.out_features >= 16000 and self.in_features >= 4096 and t_c % 4 == 0:
            self._split = 4
        else:
            self._split = 1
        self._y32 = None

    def forward(self, x):
        if x.dtype == torch.bfloat16 and x.numel() == self.in_features:
            if self._split > 1:
                # split-K: fp32 atomics + external outlier overlay
                from .tpab_gemv_splitk import fused_gemv_tpab_splitk
                if self._y32 is None:
                    self._y32 = torch.zeros(
                        self.out_features, dtype=torch.float32, device=x.device)
                gs = self.gemv_stage
                y32 = fused_gemv_tpab_splitk(
                    x, gs, self.out_features, self.in_features,
                    tile_r=self.tile_r, split=self._split, y32=self._y32)
                if gs.get("ol_rows_idx") is not None and gs["ol_row_k"].numel():
                    xf = x.reshape(-1)
                    contrib = (gs["ol_row_v"].to(torch.bfloat16).float()
                               * xf[gs["ol_row_k"].long()].float())
                    y32.index_add_(0, gs["ol_rows_idx"], contrib)
                y = y32.to(torch.bfloat16)
            else:
                y = fused_gemv_tpab(x, self.gemv_stage, self.out_features,
                                    self.in_features, tile_r=self.tile_r)
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
def deploy_model_tpab(model, snr_target_db=26.0, outlier_frac=0.004, verbose=True,
                      lazy=False):
    """Replace every quantizable Linear with TpabLinear (in-place).

    lazy=True: the model was loaded with low_cpu_mem_usage (weights stay on
    disk/mmap until touched) — encode each layer from its CPU tensor then
    drop it, so the full bf16 weights NEVER materialize on GPU. This is the
    big-model path (deploy peak ~= final resident + one layer's bf16).
    """
    from .linear import iter_quantizable_linears, _set_parent_child

    targets = list(iter_quantizable_linears(model))
    total_bytes = 0
    n_elems = 0
    n_skip = 0
    for name, mod in targets:
        O, I = mod.weight.shape
        if I % 64:
            n_skip += 1
            continue
        tr = 64
        while O % tr and tr > 1:
            tr //= 2
        if O % tr:
            n_skip += 1
            continue
        w = mod.weight.data
        if lazy:
            # force materialization of just this layer to CPU RAM
            # (low_cpu_mem keeps tensors meta until first touch)
            w = w.to("cpu", torch.bfloat16)
        tl = TpabLinear(w, snr_target_db, outlier_frac)
        mod.weight.data = torch.empty(0)      # release the bf16 reference
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
