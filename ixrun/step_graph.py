# -*- coding: utf-8 -*-
"""StepGraphEngine: whole-decode-step CUDA-Graph engine (Llama-arch).

Benchmark finding (bench_formats_minicpm5): a 1B eager decode step is
~77% Python/launch overhead — full-step CUDA graph replay is 13.2ms vs
57.4ms eager. This engine captures the ENTIRE one-token forward in one
graph (StaticCache-backed) and replays per token.

Formats:
  codec='bf16'        resident bf16 (fastest possible step)
  codec='int8x'       INT8-X cache='full' (bf16 resident after decode)
  codec='udcq'        UDCQ cache='full' (bf16 resident) — default trade
  codec='udcq-stream' UDCQ streaming: packed GPU-resident, fused-GEMV
                      decode kernels INSIDE the captured step (~1GB VRAM)

Interface identical to Int8XEngine: tokenizer / generate / stream.
"""
from __future__ import annotations

import gc
import os
import time

import torch
import torch.nn.functional as F

from .config import MODEL_PATH

__all__ = ["StepGraphEngine"]

DEFAULT_MAX_CTX = 2048


class StepGraphEngine:
    def __init__(self, model, tokenizer, stats=None, max_ctx=DEFAULT_MAX_CTX,
                 verbose=True):
        self.model = model
        # raw (un-compiled) module for eager prefill: avoids a Dynamo
        # shape-recompile of the long-sequence path at startup
        self.model_raw = getattr(model, '_orig_mod', model)
        self.tokenizer = tokenizer
        self.stats = stats or {}
        self.max_ctx = max_ctx
        model.eval()
        self._cfg = model.config
        self._deploy_static(max_ctx, verbose)

    # ------------------------------------------------------------------ #
    @classmethod
    def from_pretrained(cls, model_path=MODEL_PATH, codec='udcq',
                        max_ctx=DEFAULT_MAX_CTX, verbose=True):
        from transformers import AutoTokenizer

        from .engine import Int8XEngine

        tokenizer = AutoTokenizer.from_pretrained(model_path,
                                                  trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = Int8XEngine._load_any(model_path, torch.bfloat16,
                                      low_cpu=True)
        # S=1 decode attention via the fused Triton kernel (the cutlass
        # fmha path costs ~94us/layer on these shapes); must be installed
        # BEFORE graph capture.  IXRUN_FAST_ATTN=0 reverts.
        from .fast_attn import install_fast_decode_attention

        install_fast_decode_attention()
        stats = {'codec': codec}
        if codec == 'int8x':
            from .linear import deploy_model

            deploy_model(model, level_bits=(3, 5, 8), cache='full',
                         verbose=verbose)
        elif codec == 'udcq':
            from .udcq import deploy_udcq

            stats.update(deploy_udcq(model, cache='full', verbose=verbose))
        elif codec in ('udcq', 'udcq-stream', 'udcq-uls'):
            from .udcq import deploy_udcq

            stats.update(deploy_udcq(model, cache='stream', verbose=verbose))
        elif codec == 'gmm-stream':
            # Bayesian-GMM codebook encoded into UDCQ-compatible packed
            # format (signed codebook, all-ones sign stream) -> the fused
            # decode+GEMV kernels run unchanged inside the whole-step graph
            from .linear import iter_quantizable_linears, _set_parent_child
            from .udcq import UdcqLinear
            from benchmarks.gmm_stream_minicpm5 import (gmm_pack,
                                                        fit_bayesian_gmm)

            targets = list(iter_quantizable_linears(model))
            samp = []
            per = max(1, 3_000_000 // len(targets))
            for _, mod in targets:
                w = mod.weight.data.reshape(-1).float()
                if w.numel() > per:
                    w = w[torch.randint(0, w.numel(), (per,))]
                samp.append(w)
            mu, _, _ = fit_bayesian_gmm(torch.cat(samp), K=16, iters=40)
            for name, mod in targets:
                packed = gmm_pack(mod.weight.data, mu)
                bias = mod.bias.data if mod.bias is not None else None
                _set_parent_child(model, name,
                                  UdcqLinear(packed, bias=bias,
                                             cache='stream'))
            stats.update({'gmm_components': len(mu), 'bpw': 5.03})
            if verbose:
                print(f'[gmm-stream] {len(targets)} linears | '
                      f'{len(mu)} components | 5.03 bpw', flush=True)
        elif codec == 'hpqs-mixed':
            # HPQ-x-scale (7bpw, hand-CUDA GEMV) on down_proj+o_proj,
            # UDCQ-stream (6bpw) on the rest; packs from disk cache
            import os as _os
            from .linear import iter_quantizable_linears, _set_parent_child
            from .udcq import deploy_udcq
            from benchmarks.hpqs_runtime import HpqsLinear

            cache_p = _os.environ.get(
                'HPQS_PACK_CACHE',
                r'E:\IXRUN\experiments\hpqs_minicpm5\down_o_packs.pt')
            packs = torch.load(cache_p, weights_only=False)
            for name, mod in list(iter_quantizable_linears(model)):
                if name in packs:
                    _set_parent_child(model, name, HpqsLinear(packs[name]))
            stats.update(deploy_udcq(model, cache='stream',
                                     verbose=verbose))
            stats['hpqs_layers'] = len(packs)
            if verbose:
                print(f'[hpqs-mixed] {len(packs)} HpqsLinear + '
                      f'{stats["n_layers"]} UdcqLinear', flush=True)
        elif codec == 'gsq':
            # GSQ: K32 signed codebook (5b) + log-i8 per-16 scales
            # (5.50bpw), hand-CUDA GEMV in-graph
            from .linear import iter_quantizable_linears, _set_parent_child
            from benchmarks.gsq_runtime import GsqLinear

            for name, mod in list(iter_quantizable_linears(model)):
                W = mod.weight.data.float().cuda()
                pk = {}
                from benchmarks.gsq_runtime import gs_pack
                pk = gs_pack(W)
                del W
                torch.cuda.empty_cache()
                _set_parent_child(model, name, GsqLinear(pk))
            stats.update({'bpw': 5.5, 'codec': 'gsq'})
            if verbose:
                print('[gsq] all linears wrapped (5.50bpw)', flush=True)
        elif codec == 'ig32':
            # int5-g32+warm (user scheme): 4b code + 1b sign + per-group
            # 16-level table; 5.75bpw, PPL 56.13 on MiniCPM5-1B
            from .linear import iter_quantizable_linears, _set_parent_child
            from benchmarks.ig32_runtime import Ig32Linear, ig32_pack

            for name, mod in list(iter_quantizable_linears(model)):
                W = mod.weight.data.float().cuda()
                pk = ig32_pack(W)
                del W
                torch.cuda.empty_cache()
                _set_parent_child(model, name, Ig32Linear(pk))
            stats.update({'bpw': 5.75, 'codec': 'ig32'})
            if verbose:
                print('[ig32] all linears wrapped (5.75bpw)', flush=True)
        elif codec == 'bf16xl':
            from benchmarks.bf16xl_runtime import deploy_bf16xl

            stats['n_layers'] = deploy_bf16xl(model, verbose=verbose)
            stats['bpw'] = 11.5
        elif codec == 'int8g':
            # int8 + per-16 fp16 scale (the Q8_0 class): the decode is
            # ~3 instr/elem so the GEMV runs near the DRAM ceiling
            # (860GB/s isolated on the lm_head shape) - the llama.cpp
            # table shows this tier beats every complex format.
            from benchmarks.int8g_runtime import deploy_int8g

            stats['n_layers'] = deploy_int8g(model, verbose=verbose)
            stats['bpw'] = 9.0
        elif codec != 'bf16':
            raise ValueError(f'unknown codec: {codec}')
        # quantize lm_head with the UDCQ stream path (fused decode+GEMV
        # over packed streams): the cublas bf16 gemv costs ~0.47ms/step
        # (vocab 130560 x 1536 = 0.4GB reads), the fused kernel ~0.2ms
        # and -0.26GB VRAM.  The 27B ships the same way (int8
        # embed/lm_head); llama.cpp's BF16X tier quantizes the lm_head
        # too.  Only the pure bf16 lossless codec skips it.
        # IXRUN_Q_LMHEAD=0 reverts.
        lm = getattr(model, 'lm_head', None)
        if (codec != 'bf16' and type(lm).__name__
                == 'Linear' and lm.weight.numel() > 1_000_000
                and os.environ.get('IXRUN_Q_LMHEAD', '1')
                not in ('', '0')):
            from .udcq import (UDCQ_G, UdcqLinear, udcq_fit_codebook,
                               udcq_quantize)
            from .linear import _set_parent_child

            _t0 = time.perf_counter()
            W = lm.weight.data
            # MUST quantize with the BODY's global codebook: the ext
            # kernels read the one __constant__ codebook installed by the
            # first module - a per-module refit here silently decodes the
            # lm_head with the wrong codebook (-> degenerate '<s>' text;
            # the kernels ignore the per-call cb argument).
            cb_l = stats.get('codebook')
            cb = (torch.tensor(cb_l, dtype=torch.float32)
                  if cb_l is not None
                  else udcq_fit_codebook(W.cpu(), nlev=16, g=UDCQ_G))
            packed = udcq_quantize(W, cb, g=UDCQ_G)
            bias = lm.bias.data if lm.bias is not None else None
            _set_parent_child(model, 'lm_head',
                              UdcqLinear(packed, bias=bias, cache='stream'))
            lm.weight.data = torch.empty(0)
            stats['lm_head_q'] = True
            if verbose:
                mb = packed['total_bytes'] / 1e6
                print(f'[lmhead-q] UDCQ stream {mb:.0f}MB '
                      f'({time.perf_counter()-_t0:.0f}s)', flush=True)
        # ULS transform: fp16 scales -> i8 log2 per-row streams (5.5bpw).
        # Runs AFTER lmhead-q so the quantized lm_head is converted too.
        if codec == 'udcq-uls':
            from .udcq import UdcqLinear, udcq_scale_to_uls

            _t0 = time.perf_counter()
            _n = 0
            for _m in model.modules():
                if isinstance(_m, UdcqLinear):
                    _n += udcq_scale_to_uls(_m)
            stats['uls_layers'] = _n
            if verbose:
                print(f'[uls] {_n} matrices -> i8 log2 scales (5.5bpw, '
                      f'{time.perf_counter()-_t0:.0f}s)', flush=True)
        model = model.cuda()
        gc.collect()
        torch.cuda.empty_cache()
        # inductor fusion of the HF op chains (norm/rotary/residual/silu)
        # between graph breaks at the custom GEMV kernels: +44% decode on
        # gsq (174.5 -> 251.6 tok/s measured, text verified coherent).
        # Llama-arch StaticCache only - the 27B GDN in-place states BREAK
        # under Dynamo functionalization (do NOT enable there).
        # STEP_GRAPH_COMPILE=0 disables. Prefill runs through the raw
        # (_orig_mod) model to avoid a shape-recompile at startup.
        if (codec in ('gsq', 'bf16xl', 'int8g') and
                os.environ.get('STEP_GRAPH_COMPILE', '1')
                not in ('', '0')):
            _t0 = time.perf_counter()
            model = torch.compile(model)
            if verbose:
                print(f'[step-graph] torch.compile wrapped '
                      f'({time.perf_counter()-_t0:.0f}s; kernels compile '
                      'at capture warmup)', flush=True)
        eng = cls(model, tokenizer, stats=stats, max_ctx=max_ctx,
                  verbose=verbose)
        return eng

    # ------------------------------------------------------------------ #
    def _deploy_static(self, max_ctx, verbose):
        from transformers.cache_utils import StaticCache

        self.cache = StaticCache(
            config=self._cfg, max_batch_size=1, max_cache_len=max_ctx)
        dev = 'cuda'
        self.in_ids = torch.zeros(1, 1, dtype=torch.long, device=dev)
        self.pos = torch.zeros(1, 1, dtype=torch.long, device=dev)
        self.logits = torch.zeros(
            1, 1, self.model.lm_head.out_features
            if hasattr(self.model, 'lm_head')
            else self._cfg.vocab_size,
            dtype=torch.bfloat16, device=dev)

        # positional fast attention: the S=1 kernel reads self.pos
        # (device int, kernel-arg pointer + value-at-execution = replay
        # safe), so HF's per-step causal-mask construction (cat/neg
        # elementwise kernels ~0.2ms/step) is skipped entirely.  The
        # eager prefill still gets is_causal=True (q_len>1, mask=None).
        # IXRUN_POS_ATTN=0 keeps the bool-mask path.
        if os.environ.get('IXRUN_POS_ATTN', '1') not in ('', '0'):
            from . import fast_attn as _fa

            if _fa._patched:
                _fa.attach_pos_tensor(self.pos)
                _n = 0
                for _m in self.model.modules():
                    if hasattr(_m, '_update_causal_mask'):
                        _m._update_causal_mask = lambda *a, **k: None
                        _n += 1
                # transformers 5.15 builds the decode bool mask in
                # masking_utils.create_causal_mask, bound into the
                # modeling module (llama for MiniCPM5-1B); stub it so the
                # S=1 kernel's positional path replaces the mask kernels
                # (IXRUN_POS_STUB=0 isolates pos-attach from the stub)
                import sys as _sys

                _stub_on = os.environ.get('IXRUN_POS_STUB', '1') != '0'
                _hook_modules = ('transformers.masking_utils',
                                 f'{type(self.model_raw).__module__}'
                                 ) if _stub_on else ()
                for _mn in _hook_modules:
                    _mo = _sys.modules.get(_mn)
                    if _mo is not None and hasattr(
                            _mo, 'create_causal_mask'):
                        _mo.create_causal_mask = lambda *a, **k: None
                        _n += 1
                if verbose and _n:
                    print(f'[step-graph] pos-attention mode '
                          f'(mask build skipped on {_n} hooks)',
                          flush=True)
        # cached rotary: HF rebuilds cos/sin (matmul + cat + cos/sin
        # chains) every step = ~0.27ms of clone/expand kernels; build the
        # full [max_ctx, hd] tables ONCE and leave a positional gather
        # (bit-identical: the table IS the original forward's output).
        # IXRUN_CACHED_ROPE=0 reverts.
        if os.environ.get('IXRUN_CACHED_ROPE', '1') not in ('', '0'):
            import types as _types

            _nr = 0
            for _m in self.model.modules():
                if type(_m).__name__ == 'LlamaRotaryEmbedding' \
                        and hasattr(_m, 'inv_freq'):
                    _ids = torch.arange(max_ctx, device=dev).view(1, -1)
                    _xb = torch.zeros(1, 8, device=dev,
                                      dtype=torch.bfloat16)
                    _cos, _sin = _m.forward(_xb, _ids)
                    _m._cos_cache = _cos[0].contiguous()
                    _m._sin_cache = _sin[0].contiguous()

                    def _rope_fwd(self, x, position_ids):
                        idx = position_ids.reshape(-1)
                        return (self._cos_cache[idx].unsqueeze(0),
                                self._sin_cache[idx].unsqueeze(0))

                    _m.forward = _types.MethodType(_rope_fwd, _m)
                    _rope_mod = _m
                    _nr += 1
            if verbose and _nr:
                print(f'[step-graph] cached rope on {_nr} modules, '
                      f'rows={_rope_mod._cos_cache.shape[0]}',
                      flush=True)
        # fused rotary application: one Triton kernel per layer replaces
        # HF's 5-kernel mul/neg/cat/add chain per tensor (~0.3ms/step).
        # IXRUN_FUSED_ROPE=0 reverts.
        if os.environ.get('IXRUN_FUSED_ROPE', '1') not in ('', '0'):
            from .fast_attn import install_fused_rope

            if install_fused_rope() and verbose:
                print('[step-graph] fused rope application', flush=True)
        # fused residual-add + rmsnorm at the attention->MLP boundary
        # (one kernel emits the sum AND the normed output)
        if os.environ.get('IXRUN_FUSED_ADDNORM', '1') not in ('', '0'):
            from .fast_attn import install_fused_add_norm

            if install_fused_add_norm() and verbose:
                print('[step-graph] fused add+norm layers', flush=True)

        # install the UDCQ codebook EXPLICITLY once, before any graph
        # capture: the ext kernels read the __constant__ c_cb staged by
        # install_codebook ("call ONCE outside any graph capture").
        # Relying on the lazy per-forward install leaks a memcpy into
        # the capture (the _cb_installed flag does not survive across
        # the warmup/capture module instances) and the lm_head decoded
        # with a stale codebook -> '<s>' degeneration.
        from .udcq import UdcqLinear as _Udcq, _cuda_gemv_mod as _cgm

        _um = [m for m in self.model.modules()
               if isinstance(m, _Udcq)]
        if _um and os.environ.get('UDCQ_CUDA_GEMV', '1') not in ('', '0'):
            _cg = _cgm()
            _cg.install_codebook(_um[0]._cb.float().contiguous())
            _cg._cb_installed = True
            if verbose:
                print(f'[step-graph] codebook staged for '
                      f'{len(_um)} UdcqLinear modules', flush=True)

        # warmup on a side stream then capture the full decode step
        if verbose:
            print('[step-graph] capturing full decode step...', flush=True)
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                self._step()
        torch.cuda.current_stream().wait_stream(s)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            lg = self._step()
            self.logits.copy_(lg)          # static output (pool-safe)
        self.graph = g
        self.hard_reset()
        if verbose:
            print('[step-graph] captured; cache reset', flush=True)

    @torch.no_grad()
    def _step(self):
        return self.model(self.in_ids, position_ids=self.pos,
                          past_key_values=self.cache, use_cache=True,
                          attention_mask=None).logits

    # ------------------------------------------------------------------ #
    def hard_reset(self):
        for layer in self.cache.layers:
            for attr in ('keys', 'values', 'key_cache', 'value_cache'):
                t = getattr(layer, attr, None)
                if isinstance(t, torch.Tensor):
                    t.zero_()
            for attr in ('cumulative_length', 'cache_position'):
                t = getattr(layer, attr, None)
                if isinstance(t, torch.Tensor):
                    t.zero_()

    @torch.no_grad()
    def prefill(self, ids):
        """Eager prefill of the full prompt; returns next-token logits."""
        t = torch.tensor(ids, dtype=torch.long, device='cuda').view(1, -1)
        pos = torch.arange(len(ids), device='cuda').view(1, -1)
        out = self.model_raw(t, position_ids=pos,
                             past_key_values=self.cache,
                             use_cache=True, attention_mask=None)
        return out.logits

    def _first_token(self, ids):
        logits = self.prefill(ids)
        return int(logits[:, -1].argmax(-1).item())

    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def _gen_tokens(self, ids, max_new_tokens, temperature=0.0,
                    do_sample=False, top_p=1.0, top_k=0,
                    repetition_penalty=1.0, sync_period=16):
        from .sampling import needs_sampling, sample_token

        knobs = needs_sampling(do_sample, temperature, top_p, top_k,
                               repetition_penalty)
        if knobs:
            # CPU-side sampling needs per-token decisions: fall back to the
            # synchronous loop (MiniCPM5 steps are ~7ms, fine)
            self.hard_reset()
            first = self._first_token(ids)
            eos_ids = {self.tokenizer.eos_token_id}
            gc = getattr(self.model.config, "generation_config", None)
            gids = getattr(gc, "eos_token_id", None) if gc else None
            if gids is not None:
                eos_ids |= {gids} if isinstance(gids, int) else set(gids)
            eos = eos_ids
            out = [first]
            t = len(ids)
            while len(out) < max_new_tokens and t < self.max_ctx - 1:
                self.in_ids.fill_(out[-1])
                self.pos.fill_(t)
                self.graph.replay()
                nxt = sample_token(
                    self.logits[0, 0].float().cpu(),
                    temperature=temperature if do_sample else 0.0,
                    top_p=top_p, top_k=top_k,
                    repetition_penalty=repetition_penalty, past_ids=out)
                if nxt in eos:
                    break
                out.append(nxt)
                t += 1
            return out
        # deferred-sync decode: argmax stays on GPU; token ids accumulate
        # in a GPU buffer and are read back every sync_period tokens (one
        # WDDM device sync per period instead of per token).
        self.hard_reset()
        first = self._first_token(ids)          # single sync (prefill)
        eos_ids = {self.tokenizer.eos_token_id}
        gc = getattr(self.model.config, "generation_config", None)
        gids = getattr(gc, "eos_token_id", None) if gc else None
        if gids is not None:
            eos_ids |= {gids} if isinstance(gids, int) else set(gids)
        eos = eos_ids
        t = len(ids)
        nxt_g = torch.full((1,), first, dtype=torch.long, device='cuda')
        out_g = torch.zeros(max_new_tokens, dtype=torch.long, device='cuda')
        period = min(sync_period, max_new_tokens)
        pin = torch.zeros(period, dtype=torch.long, pin_memory=True)
        out = [first]          # the prefill argmax IS the first generated
        n = 0
        while n < max_new_tokens - 1 and t < self.max_ctx - 1:
            self.in_ids.copy_(nxt_g.view(1, 1))
            self.pos.fill_(t)
            self.graph.replay()
            if temperature and temperature > 0 or top_p < 1.0 \
                    or top_k > 0:
                # on-GPU sampling (temp + top-k + nucleus): the logits
                # row NEVER leaves the device - the CPU sampling path
                # costs a full-vocab D2H copy per token (~65ms on WDDM)
                lg = self.logits[:, -1].float()
                if top_k and top_k > 0:
                    k = min(top_k, lg.shape[-1])
                    vals = torch.topk(lg, k).values
                    lg = torch.where(
                        lg >= vals[..., -1:],
                        lg, torch.tensor(float('-inf'), device=lg.device))
                if top_p < 1.0:
                    sorted_l, order = torch.sort(lg, descending=True)
                    cum = torch.cumsum(
                        torch.softmax(sorted_l, dim=-1), dim=-1)
                    keep = cum <= top_p
                    keep[..., 0] = True
                    mask = torch.zeros_like(lg, dtype=torch.bool)
                    mask.scatter_(-1, order, keep)
                    lg = torch.where(
                        mask, lg,
                        torch.tensor(float('-inf'), device=lg.device))
                if temperature and temperature > 0:
                    lg = lg / temperature
                torch.multinomial(F.softmax(lg, dim=-1), 1, out=nxt_g)
            else:
                torch.argmax(self.logits[:, -1], out=nxt_g)
            out_g[n] = nxt_g
            n += 1
            t += 1
            if n % period == 0:
                pin.copy_(out_g[n - period:n], non_blocking=True)
                torch.cuda.synchronize()        # one sync per period
                vals = pin.tolist()
                hit = [i for i, v in enumerate(vals) if v in eos]
                if hit:
                    out.extend(vals[:hit[0] + 1])
                    return out
                out.extend(vals)
        r = n % period
        if r:                                   # tail readback
            pin2 = torch.zeros(r, dtype=torch.long, pin_memory=True)
            pin2.copy_(out_g[n - r:n], non_blocking=True)
            torch.cuda.synchronize()
            vals = pin2.tolist()
            hit = [i for i, v in enumerate(vals) if v in eos]
            if hit:
                vals = vals[:hit[0] + 1]
            out.extend(vals)
        return out

    def generate(self, prompt, max_new_tokens=64, temperature=0.0,
                 do_sample=False, top_p=1.0, top_k=0,
                 repetition_penalty=1.0, **kw):
        if not do_sample:
            temperature = 0.0
        top_p = float(kw.pop('top_p', top_p))
        top_k = int(kw.pop('top_k', top_k))
        repetition_penalty = float(kw.pop('repetition_penalty',
                                          repetition_penalty))
        ids = self.tokenizer(prompt, return_tensors='pt')['input_ids'][0] \
            .tolist()
        out = self._gen_tokens(ids, max_new_tokens, temperature, do_sample,
                               top_p, top_k, repetition_penalty)
        return self.tokenizer.decode(out)

    def stream(self, prompt, max_new_tokens=64, temperature=0.0,
               do_sample=False, top_p=1.0, top_k=0,
               repetition_penalty=1.0, **kw):
        if not do_sample:
            temperature = 0.0
        top_p = float(kw.pop('top_p', top_p))
        top_k = int(kw.pop('top_k', top_k))
        repetition_penalty = float(kw.pop('repetition_penalty',
                                          repetition_penalty))
        ids = self.tokenizer(prompt, return_tensors='pt')['input_ids'][0] \
            .tolist()
        for v in self._gen_tokens(ids, max_new_tokens, temperature,
                                  do_sample, top_p, top_k,
                                  repetition_penalty, sync_period=8):
            yield self.tokenizer.decode([v])


if __name__ == '__main__':
    # quick self-check: bf16 vs udcq-stream token agreement + speed
    import sys
    sys.path.insert(0, r'E:\IXRUN')
    import pandas
    import time as _t

    for codec in ('bf16', 'udcq', 'udcq-stream'):
        e = StepGraphEngine.from_pretrained(codec=codec)
        ids = e.tokenizer('The theory of relativity states that',
                          return_tensors='pt')['input_ids'][0].tolist()
        t0 = _t.time()
        toks = e._gen_tokens(ids, 40, 0.0)
        torch.cuda.synchronize()
        dt = (_t.time() - t0) / len(toks) * 1e3
        print(f'[{codec}] {1000/dt:.1f} tok/s ({dt:.1f}ms/tok) -> '
              f'{e.tokenizer.decode(toks)[:80]!r} | gpu '
              f'{torch.cuda.memory_allocated()/1e9:.2f}GB', flush=True)
        del e
        gc.collect()
        torch.cuda.empty_cache()
