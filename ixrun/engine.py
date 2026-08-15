"""INT8-X streaming inference engine + resource scheduler.

``Int8XEngine`` is the high-level entry point: load a model, deploy INT8-X
quantization, and generate text. Three deployment modes:

  * ``mode='cached'`` — decode every packed weight to bf16 once and keep it
    resident on GPU. Fastest inference; GPU memory ~= full bf16 weights.

  * ``mode='streaming'`` — keep packed bitstreams **GPU-resident** (no DMA!)
    and re-decode each layer into a single shared GPU weight buffer every
    forward via the fused Triton kernel. GPU weight memory ~= packed(463MB)
    + one shared decode buf(22MB), a ~4.5x reduction vs cached.

  * ``mode='graph'`` — streaming + CUDA-Graph capture of all decode kernels.
    One ``graph.replay()`` replaces 168 individual kernel launches. Same
    memory as streaming, lower decode latency.

The :class:`ResourceScheduler` estimates packed / resident / peak memory and
recommends a mode for a given GPU budget.
"""
from __future__ import annotations
import gc
import pandas  # MUST before transformers (stack overflow fix on this env)
import torch
import torch.nn as nn
import torch.nn.functional as F

from .quantize import int8x_quantize
from .triton_kernels import decode_weight_scatter, precompute_block_offsets
from .linear import iter_quantizable_linears, _set_parent_child
from .config import TRITON_BLOCK
from .generate import generate_text, stream_generate

__all__ = ["Int8XEngine", "StreamingLinear", "ResourceScheduler"]


# --------------------------------------------------------------------------- #
#  Streaming linear: GPU-resident packed + shared decode buf + Triton decode
# --------------------------------------------------------------------------- #
class StreamingLinear(nn.Module):
    """Per-layer real-time GPU decode: GPU-packed -> shared buf -> GEMM.

    Packed bitmaps/streams are GPU-resident (no host DMA). On each forward the
    fused Triton kernel decodes the packed data into the shared weight buffer,
    then F.linear runs the matmul. The shared buffer is overwritten by the next
    layer (transformer layers execute sequentially so this is safe).
    """

    def __init__(self, packed: dict, bias=None):
        super().__init__()
        self.out_features = packed["out_f"]
        self.in_features = packed["in_f"]
        self.N = packed["N"]
        self.level_bits = packed["level_bits"]
        self.counts = packed["counts"]

        # GPU-resident packed data (no host, no DMA)
        for key, val in [
            ("_b1", packed["bitmaps"][0]),
            ("_b2", packed["bitmaps"][1]),
            ("_l1", packed["streams"][0]),
            ("_l2", packed["streams"][1]),
            ("_l3", packed["streams"][2]),
            ("_scale", packed["scale"]),
        ]:
            self.register_buffer(key, val.cuda())

        # precomputed per-block prefix sums (GPU, tiny)
        if tuple(self.level_bits) == (3, 5, 8) and torch.cuda.is_available():
            self.register_buffer("_b1_blk", precompute_block_offsets(packed)[0])
            self.register_buffer("_b2_blk", precompute_block_offsets(packed)[1])
            self._use_triton = True
        else:
            self._b1_blk = self._b2_blk = None
            self._use_triton = False

        # fused decode+GEMV for single-token decode steps (generation hot
        # loop): reads only packed streams, bf16 weight never materializes
        self._use_fused = False
        if (
            self._use_triton
            and self.in_features % 512 == 0
            and torch.cuda.is_available()
        ):
            from .fused import compute_row_prefixes

            q1, q2 = compute_row_prefixes(packed)
            self.register_buffer("_q1", q1.cuda())
            self.register_buffer("_q2", q2.cuda())
            self._use_fused = True

        if bias is not None:
            self.register_buffer("_bias", bias.detach().clone().cuda())
        else:
            self._bias = None

        self._w_buf = None  # set by engine

    def _set_shared_buf(self, w_buf):
        self._w_buf = w_buf

    def _decode(self):
        """Decode packed weight into shared buffer (Triton fused kernel)."""
        w_flat = self._w_buf[: self.N]
        if self._use_triton:
            import triton
            from .triton_kernels import _ix_decode_kernel

            blk = TRITON_BLOCK
            n_blk = (self.N + blk - 1) // blk
            _ix_decode_kernel[(n_blk,)](
                w_flat, self._b1, self._b2, self._l1, self._l2, self._l3,
                self._b1_blk, self._b2_blk, self._scale, self.N, BLK=blk,
            )
        else:
            decoded = decode_weight_scatter(
                {
                    "level_bits": self.level_bits, "out_f": self.out_features,
                    "in_f": self.in_features, "N": self.N, "scale": self._scale,
                    "bitmaps": [self._b1, self._b2],
                    "streams": [self._l1, self._l2, self._l3], "counts": self.counts,
                },
                device=w_flat.device,
            )
            w_flat.copy_(decoded.reshape(-1))
        return w_flat.view(self.out_features, self.in_features)

    def forward(self, x):
        # single-token decode step (KV-cached generation): fused GEMV path,
        # weights decoded in-register — no shared-buffer roundtrip at all
        if (
            self._use_fused
            and x.dtype == torch.bfloat16
            and x.numel() == self.in_features
        ):
            from .fused import fused_gemv

            y = fused_gemv(
                x, self._b1, self._b2, self._l1, self._l2, self._l3,
                self._q1, self._q2, self._scale,
                self.out_features, self.in_features,
            )
            y = y.view(x.shape[:-1] + (self.out_features,))
            if self._bias is not None:
                y = y + self._bias.to(y.dtype)
            return y

        w = self._decode()
        bias = self._bias.to(x.dtype) if self._bias is not None else None
        return F.linear(x, w, bias)


# --------------------------------------------------------------------------- #
#  Graph-mode layer: decode is captured in a CUDA graph, forward = GEMM only
# --------------------------------------------------------------------------- #
class GraphLinear(nn.Module):
    """Light GEMM-only layer; weight is decoded by the engine's graph replay."""

    def __init__(self, packed: dict, bias=None, decode_buf=None):
        super().__init__()
        self.out_features = packed["out_f"]
        self.in_features = packed["in_f"]
        self.N = packed["N"]
        self._w_buf = decode_buf  # per-layer GPU buffer (decoded by graph)
        if bias is not None:
            self.register_buffer("_bias", bias.detach().clone().cuda())
        else:
            self._bias = None

    def forward(self, x):
        w = self._w_buf[: self.N].view(self.out_features, self.in_features)
        bias = self._bias.to(x.dtype) if self._bias is not None else None
        return F.linear(x, w, bias)


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
            "shared_decode_buf_MB": max_layer_elems * 2 / 1e6,
            "n_elems": n_elems,
        }

    @staticmethod
    def recommend(model: nn.Module, gpu_budget_GB: float, level_bits=(3, 5, 8)) -> str:
        est = ResourceScheduler.estimate(model, level_bits)
        bf16_MB = est["bf16_weight_MB"]
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

        eng = Int8XEngine.from_pretrained(MODEL_PATH, mode="streaming")
        print(eng.generate("Hello", max_new_tokens=64))
    """

    def __init__(self, model, tokenizer, stats=None, graph_replay=None):
        self.model = model
        self.tokenizer = tokenizer
        self.stats = stats or {}
        self._graph_replay = graph_replay
        self.model.eval()

    # ----- construction -----------------------------------------------------
    @classmethod
    def _load_any(cls, model_path: str, dtype, low_cpu: bool = True):
        """Load with the right auto-class (CausalLM / multimodal / generic)."""
        from transformers import (
            AutoConfig,
            AutoModel,
            AutoModelForCausalLM,
            AutoModelForImageTextToText,
        )

        cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        arch = (getattr(cfg, "architectures", None) or [""])[0]
        try:
            if "ConditionalGeneration" in arch or "ForConditionalGeneration" in arch:
                return AutoModelForImageTextToText.from_pretrained(
                    model_path, dtype=dtype, low_cpu_mem_usage=low_cpu
                )
            return AutoModelForCausalLM.from_pretrained(
                model_path, dtype=dtype, low_cpu_mem_usage=low_cpu
            )
        except Exception:
            return AutoModel.from_pretrained(
                model_path, dtype=dtype, low_cpu_mem_usage=low_cpu
            )

    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        mode: str = "cached",
        level_bits=(3, 5, 8),
        dtype=torch.bfloat16,
        verbose: bool = True,
        cache_path: str | None = None,
    ) -> "Int8XEngine":
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # For big models (streaming/graph): stay on CPU during quantization;
        # packed data moves to GPU per-layer, bf16 weights are freed as we go.
        big = mode in ("streaming", "graph")
        model = cls._load_any(model_path, dtype, low_cpu=True)
        if not big:
            model = model.cuda()

        if verbose and not big:
            est = ResourceScheduler.estimate(model, level_bits)
            print(
                f"[engine] estimate: bf16={est['bf16_weight_MB']:.0f}MB "
                f"packed={est['packed_storage_MB']:.0f}MB ({est['compression']:.2f}x)",
                flush=True,
            )

        if mode == "streaming":
            stats = cls._deploy_streaming(model, level_bits, verbose=verbose,
                                          cache_path=cache_path)
        elif mode == "graph":
            stats, replay_fn = cls._deploy_graph(model, level_bits, verbose=verbose)
            return cls(model, tokenizer, stats, graph_replay=replay_fn)
        else:
            stats = cls._deploy_cached(model, level_bits, verbose=verbose)

        eng = cls(model, tokenizer, stats)
        if big:
            eng._finalize_device()
        return eng

    # ----- deploy helpers ---------------------------------------------------
    @staticmethod
    def _deploy_cached(model, level_bits, verbose=True):
        from .linear import deploy_model

        return deploy_model(model, level_bits=level_bits, cache="full", verbose=verbose)

    @staticmethod
    @torch.no_grad()
    def _deploy_streaming(model, level_bits, verbose=True, cache_path=None):
        """GPU-resident packed + shared decode buffer + Triton real-time decode.

        cache_path: if set and the file exists, load the previously quantized
        packed weights (skips the ~minutes quantization AND the bf16 disk read,
        since lazy-loaded weights are never touched). If it doesn't exist,
        quantize now and save for next time.
        """
        import os

        targets = list(iter_quantizable_linears(model))
        layer_data = None

        if cache_path is not None and os.path.exists(cache_path):
            if verbose:
                print(f"[engine][cache] loading packed weights from {cache_path}", flush=True)
            blob = torch.load(cache_path, map_location="cpu", weights_only=False)
            if tuple(blob["level_bits"]) != tuple(level_bits):
                raise ValueError(
                    f"cache was built with level_bits={blob['level_bits']}, "
                    f"requested {tuple(level_bits)}; delete the cache or change level_bits"
                )
            layer_data = blob["layers"]
            # sanity: names must match this model
            have = {n for n, _ in model.named_modules()}
            missing = [n for n, _, _ in layer_data if n not in have]
            if missing:
                raise ValueError(f"cache/model mismatch: {len(missing)} unknown layers")
            if verbose:
                print(f"[engine][cache] {len(layer_data)} layers loaded", flush=True)

        if layer_data is None:
            layer_data = []
            for name, mod in targets:
                p = int8x_quantize(mod.weight.data, level_bits)
                bias = mod.bias.data if mod.bias is not None else None
                layer_data.append((name, p, bias))
            if cache_path is not None:
                if verbose:
                    print(f"[engine][cache] saving packed weights to {cache_path}", flush=True)
                torch.save({"level_bits": tuple(level_bits), "layers": layer_data}, cache_path)

        max_N = 0
        total_bytes = 0
        stream_layers = []
        for name, p, bias in layer_data:
            sl = StreamingLinear(p, bias=bias)
            stream_layers.append((name, sl))
            total_bytes += p["total_bytes"]
            max_N = max(max_N, p["N"])

        # single shared decode buffer (max layer weight, bf16)
        shared_w = torch.empty(max_N, dtype=torch.bfloat16, device="cuda")
        for name, sl in stream_layers:
            sl._set_shared_buf(shared_w)
            _set_parent_child(model, name, sl)

        # free original bf16 weights (now replaced by packed GPU data)
        gc.collect(); torch.cuda.empty_cache()

        shared_MB = shared_w.numel() * 2 / 1e6
        if verbose:
            print(
                f"[engine][streaming] {len(targets)} layers | "
                f"packed GPU-resident={total_bytes/1e6:.0f}MB | "
                f"shared decode buf={shared_MB:.1f}MB",
                flush=True,
            )
        return {
            "mode": "streaming",
            "n_layers": len(targets),
            "total_bytes": total_bytes,
            "shared_gpu_MB": shared_MB,
        }

    @staticmethod
    @torch.no_grad()
    def _deploy_graph(model, level_bits, verbose=True):
        """Streaming + CUDA-Graph capture of all decode kernels.

        NOTE: graph mode needs one persistent decode buffer per layer
        (= full bf16 weights on GPU) — only viable for small models.
        """
        targets = list(iter_quantizable_linears(model))
        total_params = sum(m.weight.numel() for _, m in targets)
        if total_params * 2 > torch.cuda.get_device_properties(0).total_memory * 0.9:
            raise RuntimeError(
                f"graph mode needs {total_params*2/1e9:.1f}GB of per-layer decode "
                f"buffers — too large for this GPU; use mode='streaming'"
            )

        total_bytes = 0
        decode_specs = []  # (name, StreamingLinear, per_layer_decode_buf)
        for name, mod in targets:
            p = int8x_quantize(mod.weight.data, level_bits)
            bias = mod.bias.data if mod.bias is not None else None
            sl = StreamingLinear(p, bias=bias)
            # per-layer decode buffer (graph needs fixed addresses)
            buf = torch.empty(p["N"], dtype=torch.bfloat16, device="cuda")
            sl._set_shared_buf(buf)
            decode_specs.append((name, sl, buf))
            total_bytes += p["total_bytes"]

        # warmup on a side stream (standard CUDA-graph pattern), then capture
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for name, sl, buf in decode_specs:
                sl._decode()
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for name, sl, buf in decode_specs:
                sl._decode()
        torch.cuda.synchronize()
        if verbose:
            print(f"[engine][graph] captured {len(decode_specs)} decode kernels", flush=True)

        # swap in GraphLinear (GEMM-only forward)
        for name, sl, buf in decode_specs:
            gl = GraphLinear(
                {"out_f": sl.out_features, "in_f": sl.in_features, "N": sl.N},
                bias=sl._bias,
                decode_buf=buf,
            )
            _set_parent_child(model, name, gl)

        # keep StreamingLinear objects (and their GPU packed buffers) alive
        # for as long as the graph exists — default-arg captures the reference
        def replay(_keep_alive=decode_specs, _graph=graph):
            _graph.replay()

        packed_MB = total_bytes / 1e6
        if verbose:
            print(
                f"[engine][graph] {len(decode_specs)} layers | "
                f"packed={packed_MB:.0f}MB | decode=graph.replay()",
                flush=True,
            )
        return (
            {"mode": "graph", "n_layers": len(decode_specs), "total_bytes": total_bytes},
            replay,
        )

    # ----- inference --------------------------------------------------------
    def _finalize_device(self):
        """After CPU-side streaming deploy, move leftover params to GPU.

        Iterates per-module (a dotted-name setattr on the root model would
        create a stray attribute instead of updating nested modules).
        """
        for _, mod in self.model.named_modules():
            for p in mod.parameters(recurse=False):
                if p.data.device.type != "cuda":
                    p.data = p.data.cuda()
            for bname, b in mod.named_buffers(recurse=False):
                if b.device.type != "cuda":
                    setattr(mod, bname, b.cuda())
        gc.collect(); torch.cuda.empty_cache()
        self.model.eval()

    def _pre_forward(self):
        if self._graph_replay is not None:
            self._graph_replay()

    def generate(self, prompt, **kw):
        self._pre_forward()
        return generate_text(self.model, self.tokenizer, prompt, **kw)

    def stream(self, prompt, **kw):
        self._pre_forward()
        yield from stream_generate(self.model, self.tokenizer, prompt, **kw)
