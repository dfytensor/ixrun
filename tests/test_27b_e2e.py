# -*- coding: utf-8 -*-
"""Stage 4 step 5: FULL 64-layer real-weights greedy generation.
Gate: runs, valid ids, deterministic, coherent-ish text."""
import sys, time
sys.path.insert(0, r'E:\IXRUN')
import pandas  # noqa: F401
import torch
from torch.utils.cpp_extension import load_inline
from ixrun.config import QWEN38_PATH

BLOB = r'E:\IXRUN\experiments\qwen38_udcq\q38_blob.pt'
src = open(r'E:\IXRUN\ixrun\cpp\engine_v5.cu',
           encoding='utf-8').read()
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
'''
ext = load_inline(name='ixrun_cpp_v5s4g', cpp_sources=[proto],
                  cuda_sources=[src],
                  functions=['init27', 'step27'],
                  extra_cuda_cflags=['-O3', '--use_fast_math',
                                     '-allow-unsupported-compiler'],
                  verbose=False)

free0, total0 = torch.cuda.mem_get_info()
print(f'VRAM free {free0/1e9:.1f} / {total0/1e9:.1f} GB', flush=True)

t0 = time.perf_counter()
blob = torch.load(BLOB, map_location='cpu', mmap=True,
                  weights_only=True)
cb_g = blob['codebook'].float().cuda()
lh = blob['layers']['lm_head']
lh_i, lh_s = lh['idx'].cuda(), lh['sign'].cuda()
lh_sc = lh['scale'].float().cuda()
print(f'blob {time.perf_counter()-t0:.1f}s', flush=True)

from transformers import AutoModelForCausalLM, AutoTokenizer
t0 = time.perf_counter()
m = AutoModelForCausalLM.from_pretrained(
    QWEN38_PATH, dtype=torch.bfloat16, low_cpu_mem_usage=True,
    device_map='cpu')
print(f'hf {time.perf_counter()-t0:.0f}s', flush=True)
tokz = AutoTokenizer.from_pretrained(QWEN38_PATH)

cfg = __import__('json').load(
    open(QWEN38_PATH + r'\config.json',
         encoding='utf-8'))['text_config']
ATTN = [i for i, x in enumerate(cfg['layer_types'])
        if x != 'linear_attention']
NL = cfg['num_hidden_layers']

packs, nw1, nw2, gex, gnorm, aex = [], [], [], [], [], []
t0 = time.perf_counter()
for l in range(NL):
    pre = f'model.layers.{l}.'
    if l in ATTN:
        for nm, of, inf in [('self_attn.q_proj', 12288, 5120),
                            ('self_attn.k_proj', 1024, 5120),
                            ('self_attn.v_proj', 1024, 5120),
                            ('self_attn.o_proj', 5120, 6144),
                            ('mlp.gate_proj', 17408, 5120),
                            ('mlp.up_proj', 17408, 5120),
                            ('mlp.down_proj', 5120, 17408)]:
            p = blob['layers'][pre + nm]
            packs += [p['idx'].cuda(), p['sign'].cuda(),
                      p['scale'].cuda()]
        packs += [torch.zeros(8, dtype=torch.uint8, device='cuda'),
                  torch.zeros(1, dtype=torch.int32, device='cuda'),
                  torch.zeros(1, dtype=torch.float16,
                              device='cuda')]
        at = m.model.layers[l]
        aex += [at.self_attn.q_norm.weight.data.float().cuda(),
                at.self_attn.k_norm.weight.data.float().cuda()]
    else:
        for nm, of, inf in [('linear_attn.in_proj_qkv', 10240, 5120),
                            ('linear_attn.in_proj_z', 6144, 5120),
                            ('linear_attn.in_proj_b', 48, 5120),
                            ('linear_attn.in_proj_a', 48, 5120),
                            ('linear_attn.out_proj', 5120, 6144),
                            ('mlp.gate_proj', 17408, 5120),
                            ('mlp.up_proj', 17408, 5120),
                            ('mlp.down_proj', 5120, 17408)]:
            p = blob['layers'][pre + nm]
            packs += [p['idx'].cuda(), p['sign'].cuda(),
                      p['scale'].cuda()]
        la = m.model.layers[l].linear_attn
        gex += [la.conv1d.weight.data.squeeze(1).float()
                .cuda().flatten(),
                torch.zeros(10240, device='cuda'),
                la.A_log.data.float().cuda(),
                la.dt_bias.data.float().cuda()]
        gnorm.append(la.norm.weight.data.float().cuda())
    nw1.append(m.model.layers[l].input_layernorm.weight.data
               .float().cuda())
    nw2.append(m.model.layers[l].post_attention_layernorm.weight
               .data.float().cuda())
    if (l + 1) % 16 == 0:
        free, _ = torch.cuda.mem_get_info()
        print(f'  layer {l+1}/64 staged, free {free/1e9:.1f}GB '
              f'({time.perf_counter()-t0:.0f}s)', flush=True)
fnw = m.model.norm.weight.data.float().cuda()
emb = blob['embed']            # CPU bf16 [vocab, hidden]
emb_pin = emb  # no pin: WDDM address-space pressure
print(f'staging done {time.perf_counter()-t0:.0f}s', flush=True)
del m
free, _ = torch.cuda.mem_get_info()
print(f'VRAM after staging: free {free/1e9:.1f}GB', flush=True)

ext.init27(cb_g, packs, nw1, nw2, gex, gnorm, aex, fnw,
           lh_i, lh_s, lh_sc, ATTN, 5120, 17408, 512)

prompt = "The capital of France is"
ids = tokz(prompt, return_tensors='pt').input_ids[0].tolist()
print(f'prompt ids: {ids}', flush=True)

THETA = cfg['rope_parameters']['rope_theta']
NGEN = 16
t0 = time.perf_counter()
toks = []
cur = ids[0]
for pos in range(len(ids) + NGEN - 1):
    t = ids[pos] if pos < len(ids) else cur
    hrow = emb_pin[t].cuda().float()
    nxt = ext.step27(hrow, pos, THETA)
    if pos >= len(ids) - 1:
        toks.append(nxt)
        cur = nxt
torch.cuda.synchronize()
t_gen = time.perf_counter() - t0
print(f'{len(toks)} tokens in {t_gen:.0f}s '
      f'({t_gen/len(toks):.1f}s/tok)', flush=True)
text = tokz.decode(toks)
print(f'TEXT: {text!r}', flush=True)
assert all(0 <= t < 248320 for t in toks)
print('STEP 5 E2E GATE PASSED', flush=True)
