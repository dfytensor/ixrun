# -*- coding: utf-8 -*-
"""C++ 27B Qwen3.8 engine — blob packs + init27 + step27."""
import os, time, gc
import torch
from torch.utils.cpp_extension import load_inline

_DIR = os.path.dirname(__file__)

def _build_ext():
    src = open(os.path.join(_DIR, 'cpp', 'engine_v5.cu'), encoding='utf-8').read()
    src27 = open(os.path.join(_DIR, 'cpp', 'engine_27b.cu'), encoding='utf-8').read()
    proto = '''
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
void s27_set_probe(int64_t l);
torch::Tensor s27d_get_h1();
'''
    return load_inline(name='ixrun_cpp_q27g', cpp_sources=[proto],
                       cuda_sources=[src, src27],
                       functions=['init27', 'step27', 's27_set_probe',
                                  's27d_get_h1'],
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

    def generate(self, prompt, max_new_tokens=32):
        ids = self.tok(prompt, return_tensors='pt').input_ids[0].tolist()
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
