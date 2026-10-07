# -*- coding: utf-8 -*-
"""Bisect SF-graph token divergence: ref-vs-ref, eager-vs-ref,
graph-vs-eager."""
import sys, time
sys.path.insert(0, r'E:\IXRUN')
import pandas  # noqa: F401
import torch
from torch.utils.cpp_extension import load_inline
from transformers import AutoTokenizer, AutoModelForCausalLM
from ixrun.config import MODEL_PATH, DATASET_CACHE
from ixrun.eval_utils import load_wikitext
from ixrun.linear import iter_quantizable_linears
from benchmarks.gsq_runtime import gs_pack

m = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, dtype=torch.bfloat16, trust_remote_code=True
).eval().cuda()
cfg = m.config
H, nh = cfg.hidden_size, cfg.num_attention_heads
nkv = cfg.num_key_value_heads
hd = getattr(cfg, 'head_dim', H // nh)
rs = getattr(cfg, 'rope_scaling', None) or {}
theta = float(rs.get('rope_theta',
                     getattr(cfg, 'rope_theta', 10000.0)))
CTX = 512

sd = dict(m.named_modules())
targets = list(iter_quantizable_linears(m))
names = [n for n, _ in targets]
lay0 = names[0].rsplit('.', 3)[0]

pks = []
for name, mod in targets:
    pk = gs_pack(mod.weight.data.cuda())
    for k in ('codes5', 'cb', 's_i8'):
        pk[k] = pk[k].cuda()
    pks.append(pk)
lh_pk = gs_pack(m.lm_head.weight.data.cuda())
for k in ('codes5', 'cb', 's_i8'):
    lh_pk[k] = lh_pk[k].cuda()
embed_w = m.model.embed_tokens.weight.data.cuda()
fn_w = sd['model.norm'].weight.data.cuda()
in_ws = [sd[lay0 + f'.{l}.input_layernorm'].weight.data.cuda()
         for l in range(24)]
post_ws = [sd[lay0 + f'.{l}.post_attention_layernorm'].weight
            .data.cuda() for l in range(24)]

src = open(r'E:\IXRUN\ixrun\cpp\engine_v5.cu',
           encoding='utf-8').read()
proto = '''
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
void sf_step_graph();
void sf_seed(int64_t token, int64_t pos);
torch::Tensor sf_get_hist(int64_t from, int64_t n);
torch::Tensor sf_get_tok();
torch::Tensor sf_get_emb();
'''
ext = load_inline(name='ixrun_cpp_v5sf2', cpp_sources=[proto],
                  cuda_sources=[src],
                  functions=['init_model', 'step', 'sf_step_graph',
                             'sf_seed', 'sf_get_hist',
                             'sf_get_tok', 'sf_get_emb'],
                  extra_cuda_cflags=['-O3', '--use_fast_math',
                                     '-allow-unsupported-compiler'],
                  verbose=False)

all_codes = [p['codes5'] for p in pks]
all_cbs = [p['cb'] for p in pks]
all_s = [p['s_i8'] for p in pks]
all_bases = [p['s_base'] for p in pks]
all_steps = [p['s_step'] for p in pks]
all_out_f = [p['out_f'] for p in pks]
all_in_f = [p['in_f'] for p in pks]

tok = AutoTokenizer.from_pretrained(MODEL_PATH,
                                    trust_remote_code=True)
texts = load_wikitext(cache_dir=DATASET_CACHE)
big = '\n'.join(texts)
prompt_ids = tok(big[20000:21200],
                 return_tensors='pt').input_ids[0, :128].cuda().tolist()

pos_gpu = torch.zeros(1, dtype=torch.int32, device='cuda')
kcs = [torch.zeros(2*nkv, CTX, hd, dtype=torch.bfloat16,
                   device='cuda') for _ in range(24)]

ext.init_model(embed_w, pos_gpu, kcs, in_ws, post_ws, fn_w,
               all_codes, all_cbs, all_s, all_bases, all_steps,
               all_out_f, all_in_f,
               lh_pk['codes5'], lh_pk['cb'], lh_pk['s_i8'],
               lh_pk['s_base'], lh_pk['s_step'],
               lh_pk['out_f'], lh_pk['in_f'],
               nh, nkv, hd, CTX, theta)

NGEN = 24

def prefill():
    for kc in kcs:
        kc.zero_()
    nxt = prompt_ids[0]
    for i, t in enumerate(prompt_ids):
        pos_gpu.fill_(i)
        nxt = ext.step(t)
    return nxt

# ---- A1: ref gen 1 ----
nxt = prefill()
ref1 = []
t_ = nxt
for g in range(NGEN):
    pos_gpu.fill_(len(prompt_ids) + g)
    t_ = ext.step(t_)
    ref1.append(t_)

# ---- A2: ref gen 2 (determinism check) ----
nxt = prefill()
ref2 = []
t_ = nxt
for g in range(NGEN):
    pos_gpu.fill_(len(prompt_ids) + g)
    t_ = ext.step(t_)
    ref2.append(t_)
m_ref = sum(a == b for a, b in zip(ref1, ref2))
print(f'[1] ref vs ref determinism: {m_ref}/{NGEN}', flush=True)

# ---- B: eager sf (no CUDA graph, same kernels as graph) ----
nxt = prefill()
ext.sf_seed(nxt, len(prompt_ids))
torch.cuda.synchronize()
eager = []
for g in range(NGEN):
    ext.sf_step_graph()
    torch.cuda.synchronize()
    eager.append(int(ext.sf_get_tok().item()))
m_e = sum(a == b for a, b in zip(eager, ref1))
print(f'[2] eager-sf vs ref: {m_e}/{NGEN}', flush=True)
d = next((i for i, (a, b) in enumerate(zip(eager, ref1))
          if a != b), None)
if d is not None:
    print(f'    first divergence at tok {d}: '
          f'eager={eager[d]} ref={ref1[d]}', flush=True)

# ---- C: graph sf vs ref ----
nxt = prefill()
ext.sf_seed(nxt, len(prompt_ids))
torch.cuda.synchronize()
s2 = torch.cuda.Stream()
s2.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(s2):
    for _ in range(3):
        ext.sf_step_graph()
torch.cuda.current_stream().wait_stream(s2)
g3 = torch.cuda.CUDAGraph()
ext.sf_seed(nxt, len(prompt_ids))
torch.cuda.synchronize()
with torch.cuda.graph(g3):
    ext.sf_step_graph()
print('graph captured', flush=True)

ext.sf_seed(nxt, len(prompt_ids))
torch.cuda.synchronize()
for _ in range(NGEN):
    g3.replay()
torch.cuda.synchronize()
hist = ext.sf_get_hist(len(prompt_ids), NGEN).tolist()
m_g = sum(a == b for a, b in zip(hist, ref1))
print(f'[3] graph-sf vs ref: {m_g}/{NGEN}', flush=True)
m_ge = sum(a == b for a, b in zip(hist, eager))
print(f'[4] graph-sf vs eager-sf: {m_ge}/{NGEN}', flush=True)

# ---- D: embedding path bit-check ----
# compare sf emb_buf (kernel lookup) vs memcpy (ref path uses)
ext.sf_seed(7, 300)   # arbitrary token
torch.cuda.synchronize()
ext.sf_step_graph()   # runs embed_lookup into g_emb_buf
torch.cuda.synchronize()
emb_kernel = ext.sf_get_emb().clone()
emb_memcpy = embed_w[7].cpu()
same = torch.equal(emb_kernel.view(torch.uint16),
                   emb_memcpy.cpu().view(torch.uint16))
print(f'[5] embed kernel vs table row bit-exact: {same}',
      flush=True)
