# -*- coding: utf-8 -*-
"""Control experiment: old-style graph (Python feedback per token,
PROVEN pattern) vs SF graph (in-graph feedback) vs per-token ref.
Isolates whether cross-replay kernel-written data handoff is the bug."""
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
void step_graph();
void set_input_embedding(torch::Tensor embedding);
int64_t get_last_token();
void sf_step_graph();
void sf_seed(int64_t token, int64_t pos);
torch::Tensor sf_get_hist(int64_t from, int64_t n);
torch::Tensor sf_get_tok();
'''
ext = load_inline(name='ixrun_cpp_v5sf3', cpp_sources=[proto],
                  cuda_sources=[src],
                  functions=['init_model', 'step', 'step_graph',
                             'set_input_embedding',
                             'get_last_token', 'sf_step_graph',
                             'sf_seed', 'sf_get_hist',
                             'sf_get_tok'],
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
                 return_tensors='pt').input_ids[0, :256].cuda().tolist()

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

NGEN = 40

def prefill():
    for kc in kcs:
        kc.zero_()
    nxt = prompt_ids[0]
    for i, t in enumerate(prompt_ids):
        pos_gpu.fill_(i)
        nxt = ext.step(t)
    return nxt

# ---- A. reference ----
nxt = prefill()
ref = []
t_ = nxt
for g in range(NGEN):
    pos_gpu.fill_(len(prompt_ids) + g)
    t_ = ext.step(t_)
    ref.append(t_)

# ---- B. OLD-style graph: Python feeds embedding + reads token ----
nxt = prefill()
ext.set_input_embedding(embed_w[nxt])
torch.cuda.synchronize()
pos_gpu.fill_(len(prompt_ids))
s1 = torch.cuda.Stream()
s1.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(s1):
    for _ in range(3):
        ext.step_graph()
torch.cuda.current_stream().wait_stream(s1)
g_old = torch.cuda.CUDAGraph()
pos_gpu.fill_(len(prompt_ids))
torch.cuda.synchronize()
with torch.cuda.graph(g_old):
    ext.step_graph()

ext.set_input_embedding(embed_w[nxt])
old_toks = []
for g in range(NGEN):
    pos_gpu.fill_(len(prompt_ids) + g)
    g_old.replay()
    torch.cuda.synchronize()
    tk = ext.get_last_token()
    old_toks.append(tk)
    ext.set_input_embedding(embed_w[tk])
m_old = sum(a == b for a, b in zip(old_toks, ref))
print(f'[old-style graph vs ref]: {m_old}/{NGEN}', flush=True)
if m_old < NGEN:
    d = next(i for i, (a, b) in enumerate(zip(old_toks, ref))
             if a != b)
    print(f'  first div tok {d}: old={old_toks[d]} ref={ref[d]}',
          flush=True)

# ---- C. SF graph: all feedback on GPU ----
nxt = prefill()
ext.sf_seed(nxt, len(prompt_ids))
torch.cuda.synchronize()
s2 = torch.cuda.Stream()
s2.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(s2):
    for _ in range(3):
        ext.sf_step_graph()
torch.cuda.current_stream().wait_stream(s2)
g_sf = torch.cuda.CUDAGraph()
ext.sf_seed(nxt, len(prompt_ids))
torch.cuda.synchronize()
with torch.cuda.graph(g_sf):
    ext.sf_step_graph()
ext.sf_seed(nxt, len(prompt_ids))
torch.cuda.synchronize()
for _ in range(NGEN):
    g_sf.replay()
torch.cuda.synchronize()
sf_toks = ext.sf_get_hist(len(prompt_ids), NGEN).tolist()
m_sf = sum(a == b for a, b in zip(sf_toks, ref))
print(f'[SF graph vs ref]: {m_sf}/{NGEN}', flush=True)
if m_sf < NGEN:
    d = next(i for i, (a, b) in enumerate(zip(sf_toks, ref))
             if a != b)
    print(f'  first div tok {d}: sf={sf_toks[d]} ref={ref[d]}',
          flush=True)

# ---- D. SF graph AGAIN (same graph, re-seed) — self-consistency ----
nxt = prefill()
ext.sf_seed(nxt, len(prompt_ids))
torch.cuda.synchronize()
for _ in range(NGEN):
    g_sf.replay()
torch.cuda.synchronize()
sf2_toks = ext.sf_get_hist(len(prompt_ids), NGEN).tolist()
m_self = sum(a == b for a, b in zip(sf2_toks, sf_toks))
print(f'[SF run2 vs SF run1]: {m_self}/{NGEN}', flush=True)
