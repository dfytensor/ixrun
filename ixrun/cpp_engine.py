# -*- coding: utf-8 -*-
"""C++ GSQ engine — production wrapper around ixrun/cpp/engine_v5.cu.

Architecture (validated 2026/10, token-exact vs eager ref):
  prefill: prefill_tokens()  whole prompt in ONE C++ call (240 tok/s)
  decode:  self-feeding CUDA graph (295 tok/s, zero Python per token)
           argmax->embed->24 layers->lm_head->argmax + hist + pos_incr
"""
import os
import torch
from torch.utils.cpp_extension import load_inline

_CPP_DIR = os.path.join(os.path.dirname(__file__), 'cpp')
_EXT_NAME = 'ixrun_cpp_v5eng1'

_PROTO = '''
void init_model(
    torch::Tensor embed_table, torch::Tensor pos_gpu,
    std::vector<torch::Tensor> kv_caches,
    std::vector<torch::Tensor> in_norms,
    std::vector<torch::Tensor> post_norms,
    torch::Tensor final_norm_w,
    std::vector<torch::Tensor> codes,
    std::vector<torch::Tensor> cbs,
    std::vector<torch::Tensor> s_i8s,
    std::vector<double> bases, std::vector<double> steps,
    std::vector<int64_t> out_fs, std::vector<int64_t> in_fs,
    torch::Tensor lh_codes, torch::Tensor lh_cb,
    torch::Tensor lh_s, double lh_base, double lh_step,
    int64_t lh_out_f, int64_t lh_in_f,
    int64_t n_heads, int64_t n_kv_heads,
    int64_t head_dim, int64_t ctx, double theta);
int64_t step(int64_t token_id);
void prefill_tokens(std::vector<int64_t> toks, int64_t start_pos);
void sf_step_graph();
void sf_seed(int64_t token, int64_t pos);
torch::Tensor sf_get_hist(int64_t from, int64_t n);
'''


def _build_ext():
    src = open(os.path.join(_CPP_DIR, 'engine_v5.cu'),
               encoding='utf-8').read()
    return load_inline(
        name=_EXT_NAME, cpp_sources=[_PROTO], cuda_sources=[src],
        functions=['init_model', 'step', 'prefill_tokens',
                   'sf_step_graph', 'sf_seed', 'sf_get_hist'],
        extra_cuda_cflags=['-O3', '--use_fast_math',
                           '-allow-unsupported-compiler'],
        verbose=False)


class CppGsqEngine:
    """GSQ 5.5bpw C++ engine. Build from a HF bf16 causal LM."""

    def __init__(self, model, ctx=512):
        import time
        from ixrun.linear import iter_quantizable_linears
        from benchmarks.gsq_runtime import gs_pack

        self.model = model
        self.ctx = ctx
        cfg = model.config
        self.nh = cfg.num_attention_heads
        self.nkv = cfg.num_key_value_heads
        self.hd = getattr(cfg, 'head_dim',
                          cfg.hidden_size // self.nh)
        rs = getattr(cfg, 'rope_scaling', None) or {}
        self.theta = float(rs.get(
            'rope_theta', getattr(cfg, 'rope_theta', 10000.0)))
        self.nl = cfg.num_hidden_layers

        t0 = time.perf_counter()
        sd = dict(model.named_modules())
        targets = list(iter_quantizable_linears(model))
        lay0 = targets[0][0].rsplit('.', 3)[0]

        pks = []
        for _, mod in targets:
            pk = gs_pack(mod.weight.data.cuda())
            for k in ('codes5', 'cb', 's_i8'):
                pk[k] = pk[k].cuda()
            pks.append(pk)
        self.lh_pk = gs_pack(model.lm_head.weight.data.cuda())
        for k in ('codes5', 'cb', 's_i8'):
            self.lh_pk[k] = self.lh_pk[k].cuda()
        self.embed_w = model.model.embed_tokens.weight.data.cuda()
        fn_w = sd['model.norm'].weight.data.cuda()
        in_ws = [sd[f'{lay0}.{l}.input_layernorm'].weight.data.cuda()
                 for l in range(self.nl)]
        post_ws = [sd[f'{lay0}.{l}.post_attention_layernorm']
                   .weight.data.cuda() for l in range(self.nl)]

        self.pos_gpu = torch.zeros(1, dtype=torch.int32,
                                   device='cuda')
        self.kcs = [torch.zeros(2 * self.nkv, ctx, self.hd,
                                dtype=torch.bfloat16,
                                device='cuda')
                    for _ in range(self.nl)]
        self.ext = _build_ext()
        self.ext.init_model(
            self.embed_w, self.pos_gpu, self.kcs, in_ws, post_ws,
            fn_w,
            [p['codes5'] for p in pks], [p['cb'] for p in pks],
            [p['s_i8'] for p in pks],
            [p['s_base'] for p in pks], [p['s_step'] for p in pks],
            [p['out_f'] for p in pks], [p['in_f'] for p in pks],
            self.lh_pk['codes5'], self.lh_pk['cb'],
            self.lh_pk['s_i8'], self.lh_pk['s_base'],
            self.lh_pk['s_step'], self.lh_pk['out_f'],
            self.lh_pk['in_f'],
            self.nh, self.nkv, self.hd, ctx, self.theta)
        self._graph = None
        self.pack_time = time.perf_counter() - t0

    def reset(self):
        for kc in self.kcs:
            kc.zero_()

    def prefill(self, ids):
        """Process prompt ids; returns first generated token."""
        self.ext.prefill_tokens(ids[:-1], 0)
        self.pos_gpu.fill_(len(ids) - 1)
        return self.ext.step(ids[-1])

    def _ensure_graph(self):
        if self._graph is not None:
            return
        # warmup at a THROWAWAY position: warmup pollution at slots
        # >= generation start is harmless (rewritten before read),
        # but at stale g_pos it would clobber the last REAL prefill
        # slot (the bug-2 class — init must never touch real state).
        self.ext.sf_seed(0, self.ctx - 16)
        torch.cuda.synchronize()
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                self.ext.sf_step_graph()
        torch.cuda.current_stream().wait_stream(s)
        self._graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self._graph):
            self.ext.sf_step_graph()

    def generate(self, seed_token, start_pos, n, out=None):
        """Greedy generation: seed + n replays, batch hist read.

        out: optional list to append tokens into (streaming ready —
        hist can be read in chunks later)."""
        self._ensure_graph()
        self.ext.sf_seed(seed_token, start_pos)
        torch.cuda.synchronize()
        for _ in range(n):
            self._graph.replay()
        torch.cuda.synchronize()
        toks = self.ext.sf_get_hist(start_pos, n).tolist()
        if out is not None:
            out.extend(toks)
        return toks

    def generate_eager(self, seed_token, start_pos, n, out=None):
        """Per-token reference path (for validation / fallback)."""
        toks = []
        t = seed_token
        for i in range(n):
            self.pos_gpu.fill_(start_pos + i)
            t = self.ext.step(t)
            toks.append(t)
        if out is not None:
            out.extend(toks)
        return toks

    # ------------- CLI engine facade (greedy, llama.cpp-style) ------ //
    @classmethod
    def from_pretrained(cls, model_path, ctx=512, verbose=True):
        import gc
        import time
        from transformers import (AutoModelForCausalLM,
                                  AutoTokenizer)
        t0 = time.perf_counter()
        tok = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True)
        m = AutoModelForCausalLM.from_pretrained(
            model_path, dtype=torch.bfloat16, trust_remote_code=True
        ).eval().cuda()
        eng = cls(m, ctx=ctx)
        eng.tok = tok
        eng.model = None
        gc.collect()
        torch.cuda.empty_cache()
        if verbose:
            print(f'[cpp-gsq] packed+init '
                  f'{time.perf_counter()-t0:.1f}s', flush=True)
        return eng

    def _gen_block(self, seed, pos, n):
        self._ensure_graph()
        self.ext.sf_seed(seed, pos)
        torch.cuda.synchronize()
        for _ in range(n):
            self._graph.replay()
        torch.cuda.synchronize()
        return self.ext.sf_get_hist(pos, n).tolist()

    def _replay_block(self, pos, n):
        # continuation: tok/pos state already advanced in-graph
        for _ in range(n):
            self._graph.replay()
        torch.cuda.synchronize()
        return self.ext.sf_get_hist(pos, n).tolist()

    def _prep(self, prompt, max_new_tokens):
        self.reset()
        ids = self.tok(prompt, return_tensors='pt')\
            .input_ids[0].tolist()
        budget = self.ctx - len(ids) - 2
        n = max(1, min(max_new_tokens, budget))
        return ids, n

    def _stop_ids(self):
        ids = []
        eos = getattr(self.tok, 'eos_token_id', None)
        if eos is not None:
            ids.append(eos)
        try:
            im = self.tok.convert_tokens_to_ids('<|im_end|>')
            if im is not None and im != eos:
                ids.append(im)
        except Exception:
            pass
        return ids

    def _cut_stops(self, toks):
        cut = len(toks)
        for s in self._stop_ids():
            if s in toks:
                cut = min(cut, toks.index(s))
        return toks[:cut]

    def generate(self, prompt, max_new_tokens=128, max_tokens=None,
                 **kw):
        ids, n = self._prep(prompt, max_tokens or max_new_tokens)
        nxt = self.prefill(ids)
        return self.tok.decode(
            self._cut_stops(self._gen_block(nxt, len(ids), n)))

    def stream(self, prompt, max_new_tokens=128, chunk=16,
               max_tokens=None, **kw):
        ids, n = self._prep(prompt, max_tokens or max_new_tokens)
        nxt = self.prefill(ids)
        hist, prev, done = [], "", 0
        while done < n:
            k = min(chunk, n - done)
            if done == 0:
                toks = self._gen_block(nxt, len(ids), k)
            else:
                toks = self._replay_block(len(ids) + done, k)
            done += k
            hist.extend(toks)
            cut_toks = self._cut_stops(hist)
            text = self.tok.decode(cut_toks)
            if len(text) > len(prev):
                yield text[len(prev):]
                prev = text
            if len(cut_toks) < len(hist):
                return
