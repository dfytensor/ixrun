# -*- coding: utf-8 -*-
"""C++ 27B Qwen3.8 engine — blob packs + init27 + step27."""
import os, time, gc
import torch
from torch.utils.cpp_extension import load_inline

_DIR = os.path.dirname(__file__)

def _build_ext():
    name = 'ixrun_cpp_q27m'
    if os.environ.get('IXRUN_PREBUILT', '') not in ('', '0'):
        # workaround: cudafe++ AVs when nvcc is spawned from the python
        # process tree (flaky, source-independent); build via
        # tests/build_27b_ext.cmd (regen cuda.cu from python, then ninja
        # from cmd) and load the prebuilt .pyd directly.
        import importlib, sys
        from torch.utils.cpp_extension import _get_build_directory
        bd = _get_build_directory(name, False)
        sys.path.insert(0, bd)
        m = importlib.import_module(name)
        pyd = os.path.join(bd, name + ('.pyd' if os.name == 'nt' else '.so'))
        print(f'[cpp-27b] IXRUN_PREBUILT: {pyd} '
              f'({time.strftime("%H:%M:%S", time.localtime(os.path.getmtime(pyd)))})',
              flush=True)
        return m
    src = open(os.path.join(_DIR, 'cpp', 'engine_v5.cu'), encoding='utf-8').read()
    src27 = open(os.path.join(_DIR, 'cpp', 'engine_27b.cu'), encoding='utf-8').read()
    proto = '''
void udcq_set_uls(int64_t on);
torch::Tensor attn_v3_test(torch::Tensor q, torch::Tensor kv,
    int64_t nh, int64_t nkv, int64_t hd, int64_t ctx, int64_t pos);
torch::Tensor udcq_gemv_mt8_out(torch::Tensor x, torch::Tensor idx,
    torch::Tensor sign, torch::Tensor scale, torch::Tensor cb,
    int64_t out_f, int64_t in_f, int64_t group);
torch::Tensor udcq_dequant_test(torch::Tensor idx, torch::Tensor sign,
    torch::Tensor scale, torch::Tensor cb,
    int64_t out_f, int64_t in_f, int64_t group);
torch::Tensor udcq_gemv_out(torch::Tensor x,
    torch::Tensor idx, torch::Tensor sign,
    torch::Tensor scale, torch::Tensor cb,
    int64_t out_f, int64_t in_f, int64_t group);
void init27(torch::Tensor cb,
    std::vector<torch::Tensor> packs,
    std::vector<torch::Tensor> nw1,
    std::vector<torch::Tensor> nw2,
    std::vector<torch::Tensor> gex,
    std::vector<torch::Tensor> gnorm,
    std::vector<torch::Tensor> aex,
    torch::Tensor fnw,
    torch::Tensor lh_i, torch::Tensor lh_s, torch::Tensor lh_sc,
    std::vector<int64_t> attn_layers,
    int64_t hidden, int64_t inter, int64_t ctx);
int64_t step27(torch::Tensor h, int64_t pos, double theta);
void step27_g(torch::Tensor h, torch::Tensor dpos, double theta);
void step27_prefill(torch::Tensor h8, torch::Tensor dpos8, double theta,
                    int64_t final_block);
void step27_prefill_gemm(torch::Tensor hT, torch::Tensor dposT, double theta);
void s27_reset();
torch::Tensor s27_get_tok();
void s27_set_probe(int64_t l);
torch::Tensor s27d_get_h1();
'''
    return load_inline(name='ixrun_cpp_q27m', cpp_sources=[proto],
                       cuda_sources=[src, src27],
                       functions=['init27', 'step27', 'step27_g',
                                  'step27_prefill', 'step27_prefill_gemm',
                                  's27_reset', 's27_get_tok',
                                  's27_set_probe', 's27d_get_h1',
                                  'udcq_set_uls', 'udcq_gemv_out',
                                  'attn_v3_test', 'udcq_gemv_mt8_out', 'udcq_dequant_test'],
                       extra_cuda_cflags=['-O3', '--use_fast_math',
                                          '-allow-unsupported-compiler'],
                       verbose=False)

class CppQwen27bEngine:
    @classmethod
    def from_blob(cls, blob_path, model_path, ctx=512, verbose=True):
        t0 = time.perf_counter()
        blob = torch.load(blob_path, map_location='cpu', mmap=True,
                          weights_only=True)
        cb = blob['codebook'].float().cuda()
        from transformers import AutoModelForCausalLM, AutoTokenizer
        m = AutoModelForCausalLM.from_pretrained(
            model_path, dtype=torch.bfloat16, low_cpu_mem_usage=True,
            device_map='cpu')
        tok = AutoTokenizer.from_pretrained(model_path)
        cfg = m.config
        NL = cfg.num_hidden_layers
        ATTN = sorted(i for i, x in enumerate(cfg.layer_types)
                      if x != 'linear_attention')
        hidden, inter = cfg.hidden_size, cfg.intermediate_size
        nh, nkv, hd = cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim
        nv, nk = cfg.linear_num_value_heads, cfg.linear_num_key_heads
        vdim, kdim = cfg.linear_value_head_dim, cfg.linear_key_head_dim
        conv_dim = 2 * nk * kdim + nv * vdim
        ext = _build_ext()
        if blob.get('uls'):
            ext.udcq_set_uls(1)
            if verbose:
                print('[cpp-27b] ULS log-scale blob (5.5bpw)', flush=True)

        packs, nw1, nw2, gex, gnorm, aex = [], [], [], [], [], []
        ig, ia = 0, 0
        for l in range(NL):
            pre = f'model.layers.{l}.'
            is_attn = l in ATTN
            if is_attn:
                for nm, of, inf in [('self_attn.q_proj', nh*hd, hidden),
                                    ('self_attn.k_proj', nkv*hd, hidden),
                                    ('self_attn.v_proj', nkv*hd, hidden),
                                    ('self_attn.o_proj', hidden, nh*hd),
                                    ('mlp.gate_proj', inter, hidden),
                                    ('mlp.up_proj', inter, hidden),
                                    ('mlp.down_proj', hidden, inter)]:
                    p = blob['layers'][pre + nm]
                    packs += [p['idx'].cuda(), p['sign'].cuda(), p['scale'].cuda()]
                packs += [torch.zeros(8, dtype=torch.uint8, device='cuda'),
                          torch.zeros(1, dtype=torch.int32, device='cuda'),
                          torch.zeros(1, dtype=torch.float16, device='cuda')]
                a = m.model.layers[l].self_attn
                aex += [(a.q_norm.weight.data.float() + 1.0).cuda(),
                        (a.k_norm.weight.data.float() + 1.0).cuda()]
                sa = ia; ia += 1
            else:
                for nm in ('linear_attn.in_proj_qkv', 'linear_attn.in_proj_z',
                           'linear_attn.in_proj_b', 'linear_attn.in_proj_a',
                           'linear_attn.out_proj', 'mlp.gate_proj',
                           'mlp.up_proj', 'mlp.down_proj'):
                    p = blob['layers'][pre + nm]
                    packs += [p['idx'].cuda(), p['sign'].cuda(), p['scale'].cuda()]
                la = m.model.layers[l].linear_attn
                gex += [la.conv1d.weight.data.squeeze(1).float().cuda().flatten(),
                        torch.zeros(conv_dim, device='cuda'),
                        la.A_log.data.float().cuda(),
                        la.dt_bias.data.float().cuda()]
                gnorm.append(la.norm.weight.data.float().cuda())
                ig += 1
            nw1.append((m.model.layers[l].input_layernorm.weight.data.float() + 1.0).cuda())
            nw2.append((m.model.layers[l].post_attention_layernorm.weight.data.float() + 1.0).cuda())
            if verbose and (l+1) % 16 == 0:
                free, _ = torch.cuda.mem_get_info()
                print(f'  layer {l+1}/64 staged, free {free/1e9:.1f}GB ({time.perf_counter()-t0:.0f}s)', flush=True)

        fnw = (m.model.norm.weight.data.float() + 1.0).cuda()
        lh = blob['layers']['lm_head']
        lh_t = (lh['idx'].cuda(), lh['sign'].cuda(), lh['scale'].cuda())
        emb = blob['embed']

        ext.init27(cb, packs, nw1, nw2, gex, gnorm, aex, fnw,
                   *lh_t, ATTN, hidden, inter, ctx)
        eng = cls.__new__(cls)
        eng.ext = ext
        eng.tok = tok
        eng.ctx = ctx
        eng.hidden = hidden
        eng.emb = emb
        eng.blob = blob
        eng.model = m
        eng.NL = NL
        eng.theta = float(cfg.rope_parameters.get('rope_theta', 1e6)) if cfg.rope_parameters else 1e6
        eng._graph = None
        gc.collect()
        torch.cuda.empty_cache()
        if verbose:
            free, _ = torch.cuda.mem_get_info()
            print(f'[cpp-27b] init done {time.perf_counter()-t0:.0f}s | '
                  f'VRAM free {free/1e9:.1f}GB', flush=True)
        return eng

    def _graph_generate(self, ids, max_new_tokens):
        if self._graph is None:
            self._he_buf = torch.empty(self.hidden, dtype=torch.float32,
                                       device='cuda')
            self._dpos = torch.zeros(1, dtype=torch.int32, device='cuda')
            # warmup (allocations, func attrs) then wipe state
            for p in (0, 1):
                self._dpos.fill_(p)
                self.ext.step27_g(self._he_buf, self._dpos, self.theta)
            torch.cuda.synchronize()
            self.ext.s27_reset()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                self.ext.step27_g(self._he_buf, self._dpos, self.theta)
            self._graph = g
            print('[cpp-27b] decode graph captured', flush=True)
            # blocked-prefill graphs: h8 + dpos8 written between replays
            self._h8 = torch.empty(8, self.hidden, dtype=torch.float32,
                                   device='cuda')
            self._dpos8 = torch.zeros(8, dtype=torch.int32, device='cuda')
            self._pos8 = torch.zeros(8, dtype=torch.int32)
            for _ in range(2):
                self.ext.step27_prefill(self._h8, self._dpos8, self.theta, 0)
                self.ext.step27_prefill(self._h8, self._dpos8, self.theta, 1)
            torch.cuda.synchronize()
            self.ext.s27_reset()
            gp = torch.cuda.CUDAGraph()
            with torch.cuda.graph(gp):
                self.ext.step27_prefill(self._h8, self._dpos8, self.theta, 0)
            gpf = torch.cuda.CUDAGraph()
            with torch.cuda.graph(gpf):
                self.ext.step27_prefill(self._h8, self._dpos8, self.theta, 1)
            self._graph_pf = gp
            self._graph_pf_final = gpf
            print('[cpp-27b] prefill graphs captured', flush=True)
            # gemm-prefill graphs (T=256 / T=64 segments; cuBLAS + cores,
            # text-identical to the legacy path at 3x+ throughput)
            self._graph_gemm = {}
            try:
                self._hT = torch.empty(256, self.hidden, dtype=torch.float32,
                                       device='cuda')
                self._dposT = torch.zeros(256, dtype=torch.int32,
                                          device='cuda')
                for tsz in (256, 64):
                    hv = self._hT.narrow(0, 0, tsz)
                    dv = self._dposT.narrow(0, 0, tsz)
                    for _ in range(2):
                        self.ext.step27_prefill_gemm(hv, dv, self.theta)
                    torch.cuda.synchronize()
                    self.ext.s27_reset()
                    gg = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(gg):
                        self.ext.step27_prefill_gemm(hv, dv, self.theta)
                    self._graph_gemm[tsz] = gg
                print('[cpp-27b] gemm prefill graphs captured', flush=True)
            except Exception as e:
                print(f'[cpp-27b] gemm prefill capture failed: {e}',
                      flush=True)
                self._graph_gemm = None
        self.ext.s27_reset()
        torch.cuda.synchronize()
        toks = []
        n = len(ids)
        # blocked prefill: full 8-token blocks via the captured graphs (mt8q
        # projections, one weight read per 8 tokens); last block emits the
        # first new token; a 0-7 token tail and the decode loop use the S=1
        # graph as before. IXRUN_NO_PREFILL=1 forces the legacy path (gate).
        M = (n // 8) * 8 if os.environ.get(
            'IXRUN_NO_PREFILL', '0') in ('', '0') else 0
        pos = 0
        use_gemm = bool(self._graph_gemm) and os.environ.get(
            'IXRUN_GEMM_PREFILL', '1') not in ('', '0')
        if M > 0 and use_gemm:
            # gemm segments first (256, then 64s): weights dequantized once
            # per layer per segment + cuBLAS matmuls over the segment
            for tsz in (256, 64):
                g = self._graph_gemm[tsz]
                while n - pos >= tsz:
                    self._hT[:tsz].copy_(
                        self.emb[ids[pos:pos + tsz]].float())
                    self._dposT[:tsz].copy_(torch.arange(
                        pos, pos + tsz, dtype=torch.int32))
                    g.replay()
                    pos += tsz
        while n - pos >= 8:
            # mt8b 8-blocks for the remainder
            self._h8.copy_(self.emb[ids[pos:pos + 8]].float())
            self._pos8.copy_(torch.tensor(
                [pos + i for i in range(8)], dtype=torch.int32))
            self._dpos8.copy_(self._pos8)
            if pos + 8 == n:
                self._graph_pf_final.replay()
            else:
                self._graph_pf.replay()
            pos += 8
        if pos > 0 and pos == n:
            toks.append(int(self.ext.s27_get_tok().item()))
            print(f'  prefill {n} tok -> first tok {toks[-1]}', flush=True)
        # remaining 0-7 prompt tokens + decode via the S=1 graph
        for pos in range(pos, n + max_new_tokens):
            t = ids[pos] if pos < n else (toks[-1] if toks else ids[0])
            self._he_buf.copy_(self.emb[t].float())
            self._dpos.fill_(pos)
            self._graph.replay()
            nxt = int(self.ext.s27_get_tok().item())
            if pos >= n - 1:
                toks.append(nxt)
            if pos < 3 or pos % 8 == 0:
                print(f'  pos {pos} ok tok {nxt}', flush=True)
        return toks

    def generate(self, prompt, max_new_tokens=32, graph=True):
        ids = self.tok(prompt, return_tensors='pt').input_ids[0].tolist()
        if graph:
            try:
                return self.tok.decode(self._graph_generate(ids, max_new_tokens))
            except Exception as e:
                print(f'[graph path failed: {e}] -> eager fallback', flush=True)
                self._graph = None
        self.ext.s27_reset()
        torch.cuda.synchronize()
        toks = []
        he_buf = torch.empty(self.hidden, dtype=torch.float32, device='cuda')
        for pos in range(len(ids) + max_new_tokens):
            t = ids[pos] if pos < len(ids) else (toks[-1] if toks else ids[0])
            he_buf.copy_(self.emb[t].float())
            try:
                nxt = self.ext.step27(he_buf, pos, self.theta)
            except RuntimeError as e:
                print(f'ERROR at pos {pos}: {e}', flush=True)
                raise
            if pos >= len(ids) - 1:
                toks.append(nxt)
            if pos < 3 or pos % 8 == 0:
                print(f'  pos {pos} ok tok {nxt}', flush=True)
        return self.tok.decode(toks)
