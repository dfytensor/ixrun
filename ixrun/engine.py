"""INT8-X streaming inference engine + resource scheduler.

``Int8XEngine`` is the high-level entry point: load a model, deploy INT8-X
quantization, and generate text. Two deployment modes:

  * ``mode='cached'`` — decode every packed weight to bf16 once and keep it
    resident on GPU. Fastest inference; GPU memory ~= original bf16 weights
    (the compression benefit is in *storage*, not runtime resident memory).

  * ``mode='streaming'`` — keep the packed bitstreams on pinned host memory and
    re-decode each layer's weight into a single shared GPU buffer on every
    forward. GPU weight memory ~= one layer's worth (~tens of MB), enabling
    very large models on small GPUs at the cost of decode bandwidth.

The :class:`ResourceScheduler` estimates packed / resident / peak memory and
recommends a mode for a given GPU budget.
"""
from __future__ import annotations
import gc
import torch
import torch.nn as nn
import torch.nn.functional as F

from .quantize import int8x_quantize
from .triton_kernels import decode_weight_triton, decode_weight_scatter, precompute_block_offsets
from .linear import iter_quantizable_linears, _set_parent_child, Int8XLinear
from .config import TRITON_BLOCK
from .generate import generate_text, stream_generate

__all__ = ["Int8XEngine", "StreamingLinear", "ResourceScheduler"]


# --------------------------------------------------------------------------- #
#  Streaming linear layer (live decode into a shared GPU buffer)
# --------------------------------------------------------------------------- #
class StreamingLinear(nn.Module):
    """Per-layer streaming decode: pinned-CPU packed -> shared GPU buf -> GEMM.

    Holds only the *packed* bitstreams (on pinned host memory) plus tiny
    per-layer precomputed offsets on GPU. On each forward it copies the
    bitstreams into pre-allocated GPU staging slices, runs the fused Triton
    decode into the shared weight buffer, and calls F.linear.
    """

    def __init__(self, packed: dict, bias=None):
        super().__init__()
        self.out_features = packed["out_f"]
        self.in_features = packed["in_f"]
        self.N = packed["N"]
        self.level_bits = packed["level_bits"]
        self.counts = packed["counts"]

        # keep packed bitstreams on pinned host RAM
        self._bitmaps = [b.pin_memory() for b in packed["bitmaps"]]
        self._streams = []
        for i, s in enumerate(packed["streams"]):
            self._streams.append(s.pin_memory() if s.numel() > 0 else s)
        self._scale_cpu = packed["scale"].detach().clone()
        # GPU-resident scale (tiny)
        self.register_buffer("_scale", self._scale_cpu.cuda())

        # GPU staging for bitmaps + streams (slices of engine-managed buffers)
        self._staging_b = [None] * len(self._bitmaps)
        self._staging_s = [None] * len(self._streams)
        self._w_slice = None  # view into the shared decode buffer

        # triton block offsets (GPU-resident, tiny)
        if tuple(self.level_bits) == (3, 5, 8) and torch.cuda.is_available():
            self._b1_blk, self._b2_blk = precompute_block_offsets(packed)
            self._use_triton = True
        else:
            self._b1_blk = self._b2_blk = None
            self._use_triton = False

        if bias is not None:
            self.register_buffer("_bias", bias.detach().clone().cuda())
        else:
            self._bias = None

    # the engine calls these to wire up the shared buffers
    def _attach_staging(self, staging_b, staging_s, w_slice):
        self._staging_b = staging_b
        self._staging_s = staging_s
        self._w_slice = w_slice

    def forward(self, x):
        # 1. DMA packed data host -> GPU staging slices
        for i, b in enumerate(self._bitmaps):
            if self._staging_b[i] is not None and b.numel() > 0:
                self._staging_b[i].copy_(b, non_blocking=True)
        for i, s in enumerate(self._streams):
            if self._staging_s[i] is not None and s.numel() > 0:
                self._staging_s[i].copy_(s, non_blocking=True)

        w = self._w_slice[: self.N]
        if self._use_triton:
            self._triton_decode(w)
        else:
            decoded = decode_weight_scatter(
                {
                    "level_bits": self.level_bits,
                    "out_f": self.out_features,
                    "in_f": self.in_features,
                    "N": self.N,
                    "scale": self._scale,
                    "bitmaps": self._staging_b,
                    "streams": self._staging_s,
                    "counts": self.counts,
                },
                device=x.device,
            )
            w.copy_(decoded.reshape(-1))

        weight = w.view(self.out_features, self.in_features)
        bias = self._bias.to(x.dtype) if self._bias is not None else None
        return F.linear(x, weight, bias)

    def _triton_decode(self, out_flat):
        import triton

        b1, b2 = self._staging_b[0], self._staging_b[1]
        l1, l2, l3 = self._staging_s[0], self._staging_s[1], self._staging_s[2]
        from .triton_kernels import _ix_decode_kernel

        blk = TRITON_BLOCK
        n_blk = (self.N + blk - 1) // blk
        _ix_decode_kernel[(n_blk,)](
            out_flat, b1, b2, l1, l2, l3,
            self._b1_blk, self._b2_blk, self._scale,
            self.N, BLK=blk,
        )


# --------------------------------------------------------------------------- #
#  Resource scheduler
# --------------------------------------------------------------------------- #
class ResourceScheduler:
    """Estimate memory footprints and recommend a deployment mode."""

    @staticmethod
    @torch.no_grad()
    def estimate(model: nn.Module, level_bits=(3, 5, 8)) -> dict:
        bf16_bytes = 0
        packed_bytes = 0
        n_elems = 0
        max_layer_elems = 0
        for name, mod in iter_quantizable_linears(model):
            w = mod.weight.data
            packed = int8x_quantize(w, level_bits)
            bf16_bytes += w.numel() * 2
            packed_bytes += packed["total_bytes"]
            n_elems += w.numel()
            max_layer_elems = max(max_layer_elems, w.numel())
        return {
            "bf16_weight_MB": bf16_bytes / 1e6,
            "packed_storage_MB": packed_bytes / 1e6,
            "compression": bf16_bytes / max(packed_bytes, 1),
            "max_layer_elems": max_layer_elems,
            "shared_decode_buf_MB": max_layer_elems * 2 / 1e6,  # bf16
            "n_elems": n_elems,
        }

    @staticmethod
    def recommend(model: nn.Module, gpu_budget_GB: float, level_bits=(3, 5, 8)) -> str:
        est = ResourceScheduler.estimate(model, level_bits)
        bf16_MB = est["bf16_weight_MB"]
        # leave headroom for activations + KV cache (~1.5GB for 1B-scale models)
        headroom_MB = 1500
        if bf16_MB + headroom_MB <= gpu_budget_GB * 1000:
            return "cached"
        return "streaming"


# --------------------------------------------------------------------------- #
#  Main engine
# --------------------------------------------------------------------------- #
class Int8XEngine:
    """High-level INT8-X inference engine.

    Typical usage::

        eng = Int8XEngine.from_pretrained(MODEL_PATH, mode="cached")
        print(eng.generate("Hello", max_new_tokens=64))
    """

    def __init__(self, model, tokenizer, stats=None):
        self.model = model
        self.tokenizer = tokenizer
        self.stats = stats or {}
        self.model.eval()

    # ----- construction -----------------------------------------------------
    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        mode: str = "cached",
        level_bits=(3, 5, 8),
        dtype=torch.bfloat16,
        gpu_budget_GB: float | None = None,
        verbose: bool = True,
    ) -> "Int8XEngine":
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # auto-pick mode if a budget is given
        if gpu_budget_GB is not None and mode == "auto":
            tmp = AutoModelForCausalLM.from_pretrained(
                model_path, dtype=dtype, low_cpu_mem_usage=True
            )
            mode = ResourceScheduler.recommend(tmp, gpu_budget_GB, level_bits)
            del tmp
            gc.collect()
            if verbose:
                print(f"[engine] auto-selected mode='{mode}'", flush=True)

        model = AutoModelForCausalLM.from_pretrained(
            model_path, dtype=dtype, low_cpu_mem_usage=True
        ).cuda()

        if verbose:
            est = ResourceScheduler.estimate(model, level_bits)
            print(
                f"[engine] estimate: bf16={est['bf16_weight_MB']:.0f}MB "
                f"packed={est['packed_storage_MB']:.0f}MB "
                f"({est['compression']:.2f}x)",
                flush=True,
            )

        if mode == "streaming":
            stats = cls._deploy_streaming(model, level_bits, verbose=verbose)
        else:
            stats = cls._deploy_cached(model, level_bits, verbose=verbose)

        return cls(model, tokenizer, stats)

    # ----- deploy helpers ---------------------------------------------------
    @staticmethod
    def _deploy_cached(model, level_bits, verbose=True):
        from .linear import deploy_model

        return deploy_model(model, level_bits=level_bits, cache="full", verbose=verbose)

    @staticmethod
    @torch.no_grad()
    def _deploy_streaming(model, level_bits, verbose=True):
        targets = list(iter_quantizable_linears(model))

        # 1. quantize every layer and collect sizes
        layer_packed = []
        max_bm = [0, 0]
        max_st = [0, 0, 0]
        max_N = 0
        total_bytes = 0
        for name, mod in targets:
            p = int8x_quantize(mod.weight.data, level_bits)
            bias = mod.bias.data if mod.bias is not None else None
            layer_packed.append((name, p, bias))
            total_bytes += p["total_bytes"]
            max_N = max(max_N, p["N"])
            for i, b in enumerate(p["bitmaps"]):
                max_bm[i] = max(max_bm[i], b.numel())
            for i, s in enumerate(p["streams"]):
                max_st[i] = max(max_st[i], s.numel())

        # 2. shared GPU buffers (one set, reused by every layer)
        device = torch.device("cuda")
        staging_b = [torch.empty(max_bm[i], dtype=torch.int32, device=device) for i in range(len(max_bm))]
        staging_s = [
            torch.empty(
                max_st[i],
                dtype=torch.uint8 if (i == 2) else torch.int32,
                device=device,
            )
            for i in range(len(max_st))
        ]
        shared_w = torch.empty(max_N, dtype=torch.bfloat16, device=device)

        # 3. replace layers, wiring each into the shared buffers
        for name, p, bias in layer_packed:
            sl = StreamingLinear(p, bias=bias)
            w_slice = shared_w
            # wire staging slices sized for this layer
            sb = [staging_b[i][: p["bitmaps"][i].numel()] if p["bitmaps"][i].numel() > 0 else None
                  for i in range(len(p["bitmaps"]))]
            ss = [staging_s[i][: p["streams"][i].numel()] if p["streams"][i].numel() > 0 else None
                  for i in range(len(p["streams"]))]
            sl._attach_staging(sb, ss, w_slice)
            _set_parent_child(model, name, sl)

        shared_MB = (
            sum(b.numel() * 4 for b in staging_b)
            + sum(s.numel() * s.element_size() for s in staging_s)
            + shared_w.numel() * 2
        ) / 1e6
        if verbose:
            print(
                f"[engine][streaming] {len(targets)} layers | "
                f"packed={total_bytes/1e6:.0f}MB on pinned host | "
                f"shared GPU decode buf={shared_MB:.1f}MB",
                flush=True,
            )
        return {
            "mode": "streaming",
            "n_layers": len(targets),
            "total_bytes": total_bytes,
            "shared_gpu_MB": shared_MB,
        }

    # ----- inference --------------------------------------------------------
    def generate(self, prompt, **kw):
        return generate_text(self.model, self.tokenizer, prompt, **kw)

    def stream(self, prompt, **kw):
        yield from stream_generate(self.model, self.tokenizer, prompt, **kw)
