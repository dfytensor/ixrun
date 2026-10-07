# -*- coding: utf-8 -*-
"""Self-feeding CUDA graph: argmax->embed->pos++ all in-graph.
Python does ONE replay() per token, zero sync/copy per token."""
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
print(f'packed {len(pks)} + lm_head', flush=True)

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
'''
ext = load_inline(name='ixrun_cpp_v5sf7', cpp_sources=[proto],
                  cuda_sources=[src],
                  functions=['init_model', 'step', 'sf_step_graph',
                             'sf_seed', 'sf_get_hist'],
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
print('model initialized', flush=True)

NGEN = 64

# ---------- A. reference generation (per-token step) ----------
def prefill_and_gen():
    for kc in kcs:
        kc.zero_()
    nxt = prompt_ids[0]
    for i, t in enumerate(prompt_ids):
        pos_gpu.fill_(i)
        nxt = ext.step(t)
    out = []
    tok_cur = nxt
    for g in range(NGEN):
        pos_gpu.fill_(len(prompt_ids) + g)
        tok_cur = ext.step(tok_cur)
        out.append(tok_cur)
    return nxt, out

t0 = time.perf_counter()
seed_tok, ref_toks = prefill_and_gen()
t_ref = time.perf_counter() - t0
print(f'ref prefill+gen: {t_ref:.1f}s', flush=True)

# ---------- B. reset state, warmup + capture self-feeding graph ----------
for kc in kcs:
    kc.zero_()
nxt = prompt_ids[0]
for i, t in enumerate(prompt_ids):
    pos_gpu.fill_(i)
    nxt = ext.step(t)
print(f're-prefilled, nxt={nxt}', flush=True)

ext.sf_seed(nxt, len(prompt_ids))   # seed before warmup
torch.cuda.synchronize()

s2 = torch.cuda.Stream()
s2.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(s2):
    for _ in range(3):
        ext.sf_step_graph()
torch.cuda.current_stream().wait_stream(s2)

g3 = torch.cuda.CUDAGraph()
ext.sf_seed(nxt, len(prompt_ids))   # re-seed after warmup advanced pos
torch.cuda.synchronize()
with torch.cuda.graph(g3):
    ext.sf_step_graph()
print('self-feeding graph captured!', flush=True)

# ---------- C. generate: seed + N replays, batch read ----------
ext.sf_seed(nxt, len(prompt_ids))
torch.cuda.synchronize()

t0 = time.perf_counter()
for _ in range(NGEN):
    g3.replay()
torch.cuda.synchronize()
t_g = time.perf_counter() - t0

hist = ext.sf_get_hist(len(prompt_ids), NGEN).tolist()
print(f'SF-graph {NGEN} tok in {t_g:.3f}s = {NGEN/t_g:.1f} tok/s',
      flush=True)

match = sum(1 for a, b in zip(hist, ref_toks) if a == b)
print(f'token match vs per-token ref: {match}/{NGEN}', flush=True)
print(f'REF  text: {tok.decode(ref_toks)[:200]}', flush=True)
print(f'GRAPH text: {tok.decode(hist)[:200]}', flush=True)
d = next((i for i, (a, b) in enumerate(zip(hist, ref_toks))
          if a != b), None)
if d is not None:
    print(f'first divergence tok {d}: '
          f'graph={hist[d]} ref={ref_toks[d]}', flush=True)

# ---------- C2. ref generated AFTER graph (transient check) ----------
for kc in kcs:
    kc.zero_()
nxt2 = prompt_ids[0]
for i, t in enumerate(prompt_ids):
    pos_gpu.fill_(i)
    nxt2 = ext.step(t)
ref2 = []
t_ = nxt2
for g in range(NGEN):
    pos_gpu.fill_(len(prompt_ids) + g)
    t_ = ext.step(t_)
    ref2.append(t_)
m2 = sum(a == b for a, b in zip(ref2, ref_toks) if True)
m_g2 = sum(a == b for a, b in zip(hist, ref2))
print(f'ref-after vs ref-before: {m2}/{NGEN}; '
      f'graph vs ref-after: {m_g2}/{NGEN}', flush=True)

# ---------- D. longer benchmark (128 tok, replay-only) ----------
ext.sf_seed(nxt, len(prompt_ids))
torch.cuda.synchronize()
t0 = time.perf_counter()
for _ in range(128):
    g3.replay()
torch.cuda.synchronize()
t_b = time.perf_counter() - t0
print(f'SF-graph 128 tok bench: {128/t_b:.1f} tok/s', flush=True)
h2 = ext.sf_get_hist(len(prompt_ids), 128).tolist()
print(f'text: {tok.decode(h2)[:300]}', flush=True)
