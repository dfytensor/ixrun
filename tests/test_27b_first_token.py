# -*- coding: utf-8 -*-
"""First-token logit probe: top-5 at pos 4 after feeding the prompt."""
import sys, json
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
torch::Tensor s27_get_lg();
'''
ext = load_inline(name='ixrun_cpp_v5s4g', cpp_sources=[proto],
                  cuda_sources=[src],
                  functions=['init27', 'step27', 's27_get_lg'],
                  extra_cuda_cflags=['-O3', '--use_fast_math',
                                     '-allow-unsupported-compiler'],
                  verbose=False)

blob = torch.load(BLOB, map_location='cpu', mmap=True,
                  weights_only=True)
cb_g = blob['codebook'].float().cuda()
lh = blob['layers']['lm_head']
lh_t = (lh['idx'].cuda(), lh['sign'].cuda(), lh['scale'].cuda())

from transformers import AutoModelForCausalLM, AutoTokenizer
m = AutoModelForCausalLM.from_pretrained(
    QWEN38_PATH, dtype=torch.bfloat16, low_cpu_mem_usage=True,
    device_map='cpu')
tokz = AutoTokenizer.from_pretrained(QWEN38_PATH)
cfg = json.load(open(QWEN38_PATH + r'\config.json',
                     encoding='utf-8'))['text_config']
ATTN = [i for i, x in enumerate(cfg['layer_types'])
        if x != 'linear_attention']

packs, nw1, nw2, gex, gnorm, aex = [], [], [], [], [], []
for l in range(64):
    pre = f'model.layers.{l}.'
    if l in ATTN:
        for nm in ('self_attn.q_proj', 'self_attn.k_proj',
                   'self_attn.v_proj', 'self_attn.o_proj',
                   'mlp.gate_proj', 'mlp.up_proj',
                   'mlp.down_proj'):
            p = blob['layers'][pre + nm]
            packs += [p['idx'].cuda(), p['sign'].cuda(),
                      p['scale'].cuda()]
        packs += [torch.zeros(8, dtype=torch.uint8,
                              device='cuda'),
                  torch.zeros(1, dtype=torch.int32,
                              device='cuda'),
                  torch.zeros(1, dtype=torch.float16,
                              device='cuda')]
        at = m.model.layers[l]
        aex += [at.self_attn.q_norm.weight.data.float().cuda(),
                at.self_attn.k_norm.weight.data.float().cuda()]
    else:
        for nm in ('linear_attn.in_proj_qkv',
                   'linear_attn.in_proj_z',
                   'linear_attn.in_proj_b',
                   'linear_attn.in_proj_a',
                   'linear_attn.out_proj',
                   'mlp.gate_proj', 'mlp.up_proj',
                   'mlp.down_proj'):
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
fnw = m.model.norm.weight.data.float().cuda()
emb = blob['embed']

ext.init27(cb_g, packs, nw1, nw2, gex, gnorm, aex, fnw,
           *lh_t, ATTN, 5120, 17408, 512)

ids = [760, 6511, 314, 9338, 369]   # "The capital of France is"
for pos, t in enumerate(ids):
    h = emb[t].cuda().float()
    nxt = ext.step27(h, pos, 1e7)
    if pos == len(ids) - 1:
        lg = ext.s27_get_lg()
        top = torch.topk(lg, 5)
        print(f'logits: max {lg.max():.3f} min {lg.min():.3f} '
              f'nan {torch.isnan(lg).any().item()}',
              flush=True)
        print(f'top5: {top.indices.tolist()}', flush=True)
        print(f'vals: {[round(v, 2) for v in top.values.tolist()]}',
              flush=True)
        print(f'argmax={nxt}  decode: {tokz.decode([nxt])!r}',
              flush=True)
        # HF CPU reference forward (bf16)
        with torch.no_grad():
            out = m(input_ids=torch.tensor([ids]))
        hf_lg = out.logits[0, -1].float()
        htop = torch.topk(hf_lg, 5)
        print(f'HF  top5: {htop.indices.tolist()}', flush=True)
        print(f'HF  vals: {[round(v, 2) for v in htop.values.tolist()]}',
              flush=True)
        cos = torch.nn.functional.cosine_similarity(
            lg, hf_lg, dim=0).item()
        print(f'cosine(cpp, hf): {cos:.4f}', flush=True)
