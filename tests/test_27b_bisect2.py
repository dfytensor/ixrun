# -*- coding: utf-8 -*-
"""Per-layer bisection via probe-layer mechanism (no layer_h vector).
For each layer L: set_probe(L), run prompt forward, capture h after L.
Compare vs HF CPU hooks. First divergent layer = crime scene."""
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
ext = load_inline(name='ixrun_cpp_v5s4k5', cpp_sources=[proto],
                  cuda_sources=[src, src27],
                  functions=['init27', 'step27', 's27_set_probe',
                             's27_get_probe_h', 's27_get_lg',
                             's27d_get_xn', 's27d_get_core', 's27d_get_h1', 's27d_get_h1', 's27d_get_xn2',
                             's27d_get_gated', 's27d_get_o',
                             'udcq_gemv_out', 'conv1d_update_out',
                             'l2norm_out', 'gdn_recurrent_out',
                             'gated_rmsnorm_out', 'rmsnorm_fw_out'],
                  extra_cuda_cflags=['-O3', '--use_fast_math', '--use_fast_math',
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
m.eval()
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

slots = ['in_proj_qkv','in_proj_z','in_proj_b','in_proj_a','out_proj','mlp.gate_proj','mlp.up_proj','mlp.down_proj']
for s, nm in enumerate(slots):
    pi = packs[(0*8+s)*3]; ps = packs[(0*8+s)*3+1]; psc = packs[(0*8+s)*3+2]
    bp = blob['layers']['model.layers.0.' + ('linear_attn.' + nm if s < 5 else nm)]
    ok = (torch.equal(pi, bp['idx'].cuda()) and torch.equal(ps, bp['sign'].cuda()) and torch.equal(psc, bp['scale'].cuda()))
    print(f'slot {s} {nm}: staging match {ok}', flush=True)
print(f'STAGING CHECK: nw1[0] norm {nw1[0].float().norm():.3f} | hf input_layernorm norm ' + str(m.model.layers[0].input_layernorm.weight.float().norm().item()) + ' | hf post norm ' + str(m.model.layers[0].post_attention_layernorm.weight.float().norm().item()) + f' | nw2[0] {nw2[0].float().norm():.3f} | fnw {fnw.float().norm():.3f}', flush=True)
ext.init27(cb_g, packs, nw1, nw2, gex, gnorm, aex, fnw,
           *lh_t, ATTN, 5120, 17408, 512)
ext.s27_set_probe(0)
he0 = emb[760].cuda().float()
ext.step27(he0, 0, 1e7)
h1_pos0 = ext.s27d_get_h1().cuda().float().norm().item()
print(f'SINGLE-TOKEN h(L0) norm: {h1_pos0:.3f}', flush=True)

ids = [760, 6511, 314, 9338, 369]

# HF CPU per-layer h (last position)
captured = []
def mkhook(i):
    def hook(mod, args, kw=None):
        captured.append(args[0].detach())
    return hook
hooks = [layer.register_forward_pre_hook(mkhook(i), with_kwargs=True)
         for i, layer in enumerate(m.model.layers)]
py_core_caps = []
py_mlp_caps = []
def mkcore(i):
    def hk2(mod, inp, out):
        if i == 0:
            py_core_caps.append(out.detach())
    return hk2
def mkpre(i):
    def hk3(mod, args):
        py_mlp_caps.append(args[0].detach())
    return hk3
hooks2 = [m.model.layers[0].linear_attn.register_forward_hook(mkcore(0)),
          m.model.layers[0].mlp.register_forward_pre_hook(mkpre(1))]
with torch.no_grad():
    out = m(input_ids=torch.tensor([ids]))
for hk in hooks + hooks2:
    hk.remove()
hf_h_l0_pos0 = captured[1][0, 0].float().norm().item()
print(f'HF single-token h(L0): {hf_h_l0_pos0:.3f}', flush=True)

# C++ per-layer h via probe-layer sweep
cpp_layers = []
for L in range(64):
    ext.s27_set_probe(L)
    for pos, t in enumerate(ids):
        h = emb[t].cuda().float()
        ext.step27(h, pos, 1e7)
    torch.cuda.synchronize()
    cpp_layers.append(ext.s27_get_probe_h().clone())
# xn probe: last sweep L=63 (attn layer — xn not captured there, so
# rerun one more sweep at the last GDN layer for the xn check)
ext.s27_set_probe(0)
for pos, t in enumerate(ids):
    h = emb[t].cuda().float()
    ext.step27(h, pos, 1e7)
torch.cuda.synchronize()
xn = ext.s27d_get_xn().cuda().float()
core0 = ext.s27d_get_core().cuda().float()
h1_c = ext.s27d_get_h1().cuda().float()
he0 = captured[0][0, -1].float().cuda()
h1_ref = he0 + core0
print(f'h1(L0): cpp norm {h1_c.float().norm().item():.3f} vs embed+core {h1_ref.float().norm().item():.3f} | cos {torch.nn.functional.cosine_similarity(h1_c.float(), h1_ref, dim=0).item():.4f}', flush=True)
s27d_core_ref = core0.clone()
he62 = he0   # HF input to layer 0 = embed
w62 = nw1[0]
ln_ref = w62 * he62 * torch.rsqrt(he62.pow(2).mean() + 1e-6)
py_core = py_core_caps[0][0, -1].float().cuda()
xn2 = ext.s27d_get_xn2().cuda().float()
print(f'py caps: core {len(py_core_caps)} mlp {len(py_mlp_caps)}', flush=True)
py_mlp_in = py_mlp_caps[-1][0, -1].float().cuda()
gated_s = ext.s27d_get_gated().cuda().float()
o_s = ext.s27d_get_o().cuda().float()
print(f'sched gated norm {gated_s.norm():.3f} | sched o norm {o_s.norm():.3f}', flush=True)
gated_s = ext.s27d_get_gated().cuda().float()
o_s = ext.s27d_get_o().cuda().float()
print(f'sched gated norm {gated_s.float().norm().item():.3f} | sched o norm {o_s.float().norm().item():.3f}', flush=True)
xn2_direct = ext.rmsnorm_fw_out(h1_c.view(1, -1), nw2[0], 1e-6).view(-1)
xn2_torch_ref = nw2[0] * (h1_c / torch.rsqrt(h1_c.pow(2).mean() + 1e-6))
xn_c2 = ext.rmsnorm_fw_out(xn.view(1, -1), nw2[0], 1e-6).view(-1)
xn_c2_ref = nw2[0] * (xn / torch.rsqrt(xn.pow(2).mean() + 1e-6))
print(f'A) kernel(xn): {xn_c2.float().norm().item():.3f} vs torch {xn_c2_ref.float().norm().item():.3f}', flush=True)
print(f'B) kernel(h1): {xn2_direct.float().norm().item():.3f} vs torch {xn2_torch_ref.float().norm().item():.3f}', flush=True)
topv, topi = torch.topk(h1_c.abs().flatten(), 5)
print(f'C) h1 top5 abs: {[round(v, 2) for v in topv.tolist()]} at {topi.tolist()}', flush=True)
print(f'xn2 refs: DIRECT-kernel {xn2_direct.float().norm().item():.3f} | torch-formula {xn2_torch_ref.float().norm().item():.3f} | PY {py_mlp_in.norm().item():.3f} | cos(direct,torch) {torch.nn.functional.cosine_similarity(xn2_direct.float(), xn2_torch_ref, dim=0).item():.4f}', flush=True)
print(f'DIRECT rmsnorm_fw_out(h1_c, nw2[0]): norm {xn2_direct.float().norm().item():.3f} | cos vs cpp-xn2 ' + str(torch.nn.functional.cosine_similarity(xn2_direct.float(), xn2, dim=0).item())[:6] + ' | cos vs PY ' + str(torch.nn.functional.cosine_similarity(xn2_direct.float(), py_mlp_in, dim=0).item())[:6], flush=True)
print(f'xn2(L0) vs PY mlp-input: norm {xn2.norm():.3f} vs {py_mlp_in.norm():.3f} | cos {torch.nn.functional.cosine_similarity(xn2, py_mlp_in, dim=0).item():.4f}', flush=True)
print(f'core(L0): cpp norm {core0.norm():.3f} vs PY linear_attn out norm {py_core.norm():.3f} | cos {torch.nn.functional.cosine_similarity(core0, py_core, dim=0).item():.4f}', flush=True)
print(f'xn(L0) vs torch-LN: norm {xn.norm():.3f} vs {ln_ref.norm():.3f} | cos {torch.nn.functional.cosine_similarity(xn, ln_ref, dim=0).item():.4f}', flush=True)
lg = ext.s27_get_lg()
top = torch.topk(lg, 3)
print(f'cpp top3: {top.indices.tolist()} '
      f'max {lg.max():.2f}', flush=True)

print(f'{"L":>3} {"hf_norm":>8} {"cpp_norm":>8} {"cos":>8}',
      flush=True)
first_bad = None
for i in range(64):
    if i >= 63:
        continue
    hf = captured[i + 1][0, -1].float()   # input to L+1 = output of L
    cpp = cpp_layers[i]
    cos = torch.nn.functional.cosine_similarity(
        hf, cpp, dim=0).item()
    if i < 6 or i % 8 == 0 or i > 60:
        print(f'{i:>3} {hf.norm():8.1f} {cpp.norm():8.1f} '
              f'{cos:8.4f}', flush=True)
    if cos < 0.9 and first_bad is None:
        first_bad = i
print(f'first layer with cos<0.9: {first_bad}', flush=True)

# ---- FINAL EXPERIMENT: isolated chain on the scheduler's own xn ----
import torch.nn.functional as F
def pack0(nm):
    p = blob['layers']['model.layers.0.linear_attn.' + nm]
    return (p['idx'].cuda(), p['sign'].cuda(), p['scale'].cuda())
st_x = torch.zeros(10240, 3, device='cuda')
S_x = torch.zeros(48, 128, 128, device='cuda')
qkv_x = ext.udcq_gemv_out(xn, *pack0('in_proj_qkv'), cb_g, 10240, 5120, 16)
z_x = ext.udcq_gemv_out(xn, *pack0('in_proj_z'), cb_g, 6144, 5120, 16)
b_x = ext.udcq_gemv_out(xn, *pack0('in_proj_b'), cb_g, 48, 5120, 16)
a_x = ext.udcq_gemv_out(xn, *pack0('in_proj_a'), cb_g, 48, 5120, 16)
mq_x = ext.conv1d_update_out(qkv_x, st_x, gex[0].view(10240, 4),
                             torch.zeros(10240, device='cuda'), 1)
q_c, k_c, v_c = mq_x.split([2048, 2048, 6144])
q_c = q_c.reshape(16, 128).repeat_interleave(3, 0).contiguous()
k_c = k_c.reshape(16, 128).repeat_interleave(3, 0).contiguous()
v_c = v_c.reshape(48, 128).contiguous()
q_c = ext.l2norm_out(q_c)
k_c = ext.l2norm_out(k_c)
q_c = q_c / (128 ** 0.5)
beta = torch.sigmoid(b_x)
gv = -torch.exp(gex[2]) * F.softplus(a_x + gex[3])
S_x2 = S_x.clone()
o_c = ext.gdn_recurrent_out(q_c, k_c, v_c, gv.contiguous(),
                            beta.contiguous(), S_x2)
o_x = ext.gated_rmsnorm_out(o_c, z_x.reshape(48, 128).contiguous(),
                            gnorm[0], 1e-6)
core_x = ext.udcq_gemv_out(o_x.reshape(-1).contiguous(),
                           *pack0('out_proj'), cb_g, 5120, 6144, 16)
torch.cuda.synchronize()
print(f'FINAL: isolated-core(xn_sched) norm {core_x.float().norm().item():.3f} '
      f'vs scheduler-captured norm {s27d_core_ref.float().norm().item():.3f} | cos '
      + str(torch.nn.functional.cosine_similarity(
          core_x, s27d_core_ref, dim=0).item()), flush=True)