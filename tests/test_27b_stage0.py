# -*- coding: utf-8 -*-
"""Layer-0 single-token stage-by-stage C++ vs REAL HF module comparison.
Zero state, pos 0, token 760. First divergent stage = the bug."""
import sys, json
sys.modules['fla'] = None
sys.path.insert(0, r'E:\IXRUN')
import pandas  # noqa: F401
import torch
from torch.utils.cpp_extension import load_inline
from ixrun.config import QWEN38_PATH

BLOB = r'E:\IXRUN\experiments\qwen38_udcq\q38_blob.pt'
src = open(r'E:\IXRUN\ixrun\cpp\engine_v5.cu',
           encoding='utf-8').read()
src27 = open(r'E:\IXRUN\ixrun\cpp\engine_27b.cu',
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
void s27_set_probe(int64_t l);
torch::Tensor s27_get_probe_h();
torch::Tensor s27_get_lg();
    torch::Tensor s27d_get_xn();
    torch::Tensor s27d_get_core();
    torch::Tensor s27d_get_h1();
    torch::Tensor s27d_get_xn2();
    torch::Tensor s27d_get_gated();
    torch::Tensor s27d_get_o();
torch::Tensor udcq_gemv_out(torch::Tensor x, torch::Tensor idx,
    torch::Tensor sign, torch::Tensor scale, torch::Tensor cb,
    int64_t out_f, int64_t in_f, int64_t group);
torch::Tensor conv1d_update_out(torch::Tensor x,
    torch::Tensor conv_state, torch::Tensor w, torch::Tensor bias,
    int64_t use_act);
torch::Tensor l2norm_out(torch::Tensor x2d);
torch::Tensor gdn_recurrent_out(torch::Tensor q, torch::Tensor k,
    torch::Tensor v, torch::Tensor g, torch::Tensor beta,
    torch::Tensor S);
torch::Tensor rmsnorm_fw_out(torch::Tensor x2d, torch::Tensor w,
    double eps);
torch::Tensor gated_rmsnorm_out(torch::Tensor o, torch::Tensor z,
    torch::Tensor w, double eps);
'''
ext = load_inline(name='ixrun_cpp_v5s4k7', cpp_sources=[proto],
                  cuda_sources=[src, src27],
                  functions=['init27', 'step27', 's27_set_probe',
                             's27_get_probe_h', 's27_get_lg',
                             's27d_get_xn', 's27d_get_core', 's27d_get_h1', 's27d_get_h1', 's27d_get_xn2',
                             's27d_get_gated', 's27d_get_o',
                             'udcq_gemv_out', 'conv1d_update_out',
                             'l2norm_out', 'gdn_recurrent_out',
                             'gated_rmsnorm_out', 'rmsnorm_fw_out'],
                  extra_cuda_cflags=['-O3', '--use_fast_math',
                                     '-allow-unsupported-compiler'],
                  verbose=False)

blob = torch.load(BLOB, map_location='cpu', mmap=True,
                  weights_only=True)
cb_g = blob['codebook'].float().cuda()
lh = blob['layers']['lm_head']
lh_t = (lh['idx'].cuda(), lh['sign'].cuda(), lh['scale'].cuda())
emb = blob['embed']

from transformers import AutoModelForCausalLM, AutoTokenizer
m = AutoModelForCausalLM.from_pretrained(
    QWEN38_PATH, dtype=torch.bfloat16, low_cpu_mem_usage=True,
    device_map='cpu')
m.eval()
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
        aex += [(at.self_attn.q_norm.weight.data.float() + 1.0).cuda(),
                (at.self_attn.k_norm.weight.data.float() + 1.0).cuda()]
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
    nw1.append((m.model.layers[l].input_layernorm.weight.data
                .float() + 1.0).cuda())
    nw2.append((m.model.layers[l].post_attention_layernorm.weight
                .data.float() + 1.0).cuda())
fnw = (m.model.norm.weight.data.float() + 1.0).cuda()
print('staged', flush=True)
ext.init27(cb_g, packs, nw1, nw2, gex, gnorm, aex, fnw,
           *lh_t, ATTN, 5120, 17408, 512)
print('INIT DONE', flush=True)

cap = {}
def mkpre(name):
    def h(mod, args, kwargs=None):
        cap[name] = args
    return h
def mkpost(name):
    def h(mod, inp, out):
        cap[name] = out
    return h
hs = [
    m.model.embed_tokens.register_forward_hook(mkpost('emb')),
    m.model.layers[0].input_layernorm.register_forward_hook(mkpost('xn')),
    m.model.layers[0].linear_attn.norm.register_forward_pre_hook(mkpre('o_z')),
    m.model.layers[0].linear_attn.norm.register_forward_hook(mkpost('gated')),
    m.model.layers[0].linear_attn.register_forward_hook(mkpost('core')),
    m.model.layers[0].mlp.register_forward_hook(mkpost('mlp')),
    m.model.layers[0].register_forward_hook(mkpost('layerout')),
]
with torch.no_grad():
    m(input_ids=torch.tensor([[760]]))
for h in hs:
    h.remove()
print('HF forward done', flush=True)

def nm(x):
    return x.float().norm().item()
def cs(a, b):
    a = a.float().reshape(-1)
    b = b.float().reshape(-1)
    if a.device != b.device:
        b = b.to(a.device)
    return torch.nn.functional.cosine_similarity(a, b, dim=0).item()

emb_row = cap['emb'][0, 0].float()
he0 = emb_row.clone().cuda()
ext.s27_set_probe(0)
tok = ext.step27(he0, 0, 1e7)
torch.cuda.synchronize()
print('step27 done tok', tok, flush=True)

xn_c = ext.s27d_get_xn().cuda().float()
o_c = ext.s27d_get_o().cuda().float()
g_c = ext.s27d_get_gated().cuda().float()
core_c = ext.s27d_get_core().cuda().float()
h1_c = ext.s27d_get_h1().cuda().float()
xn2_c = ext.s27d_get_xn2().cuda().float()
hout_c = ext.s27_get_probe_h().cuda().float()

xn_p = cap['xn'][0, 0].float()
o_p = cap['o_z'][0].float().reshape(-1)
z_p = cap['o_z'][1].float().reshape(-1)
gated_p = cap['gated'].float().reshape(-1)
core_p = cap['core'][0, 0].float()
mlp_p = cap['mlp'][0, 0].float()
lout_p = cap['layerout'][0, 0].float()

print(f'{"stage":8} {"cpp_norm":>9} {"py_norm":>9} {"cos":>9}', flush=True)
print(f'{"xn":8} {nm(xn_c):9.4f} {nm(xn_p):9.4f} {cs(xn_c, xn_p):9.5f}', flush=True)
print(f'{"raw o":8} {nm(o_c):9.4f} {nm(o_p):9.4f} {cs(o_c, o_p):9.5f}', flush=True)
print(f'{"gated":8} {nm(g_c):9.4f} {nm(gated_p):9.4f} {cs(g_c, gated_p):9.5f}', flush=True)
print(f'{"core":8} {nm(core_c):9.4f} {nm(core_p):9.4f} {cs(core_c, core_p):9.5f}', flush=True)
h1_p = emb_row + core_p
print(f'{"h1":8} {nm(h1_c):9.4f} {nm(h1_p):9.4f} {cs(h1_c, h1_p):9.5f}', flush=True)
np_ = m.model.layers[0].post_attention_layernorm
with torch.no_grad():
    xn2_p = np_(h1_p.reshape(1, -1).to(torch.bfloat16)).float().reshape(-1)
print(f'{"xn2":8} {nm(xn2_c):9.4f} {nm(xn2_p):9.4f} {cs(xn2_c, xn2_p):9.5f}', flush=True)
print(f'{"mlp":8} {"-":>9} {nm(mlp_p):9.4f}', flush=True)
print(f'{"layer":8} {nm(hout_c):9.4f} {nm(lout_p):9.4f} {cs(hout_c, lout_p):9.5f}', flush=True)
print(f'z norm py {nm(z_p):.4f}', flush=True)

lg = ext.s27_get_lg()
top = torch.topk(lg, 5)
print('cpp top5', top.indices.tolist(),
      [round(v, 2) for v in top.values.tolist()], flush=True)
print('DONE', flush=True)
