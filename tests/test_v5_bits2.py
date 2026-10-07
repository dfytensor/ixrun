# -*- coding: utf-8 -*-
"""Clean 3-way bisect at ONE position, no warmup pollution:
  kv_ref  = eager step()          at pos 128
  kv_sg   = eager step_graph()    at pos 128
  kv_g    = graph-replayed step   at pos 128
Warmup runs at pos 511 (isolated slot, zeroed by re-prefill)."""
import sys
sys.path.insert(0, r'E:\IXRUN')
import pandas  # noqa: F401
import torch
from torch.utils.cpp_extension import load_inline
from transformers import AutoModelForCausalLM, AutoTokenizer
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
torch::Tensor sf_get_lg();
'''
ext = load_inline(name='ixrun_cpp_v5sf5', cpp_sources=[proto],
                  cuda_sources=[src],
                  functions=['init_model', 'step', 'step_graph',
                             'set_input_embedding',
                             'get_last_token', 'sf_get_lg'],
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

tokz = AutoTokenizer.from_pretrained(MODEL_PATH,
                                     trust_remote_code=True)
texts = load_wikitext(cache_dir=DATASET_CACHE)
big = '\n'.join(texts)
prompt_ids = tokz(big[20000:21200],
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

def prefill():
    for kc in kcs:
        kc.zero_()
    nxt = prompt_ids[0]
    for i, t in enumerate(prompt_ids):
        pos_gpu.fill_(i)
        nxt = ext.step(t)
    return nxt

def bits(t):
    return t.view(torch.uint16)

# ---- 1. eager step() reference ----
nxt = prefill()
pos_gpu.fill_(128)
tok_ref = ext.step(nxt)
torch.cuda.synchronize()
kv_ref = kcs[0][:, 128, :].clone()
lg_ref = ext.sf_get_lg().clone()

# ---- 2. eager step_graph() (no graph, no warmup yet) ----
nxt2 = prefill()
assert nxt2 == nxt
ext.set_input_embedding(embed_w[nxt])
pos_gpu.fill_(128)
torch.cuda.synchronize()
ext.step_graph()
torch.cuda.synchronize()
tok_sg = ext.get_last_token()
kv_sg = kcs[0][:, 128, :].clone()
lg_sg = ext.sf_get_lg().clone()

print(f'[eager step vs eager step_graph]', flush=True)
print(f'  token: {tok_ref} vs {tok_sg} '
      f'{"MATCH" if tok_ref == tok_sg else "DIFF"}', flush=True)
print(f'  kv slot bit-exact: '
      f'{torch.equal(bits(kv_ref), bits(kv_sg))}', flush=True)
print(f'  logits bit-exact: '
      f'{torch.equal(bits(lg_ref), bits(lg_sg))}', flush=True)
if not torch.equal(bits(kv_ref), bits(kv_sg)):
    d = (kv_ref.float() - kv_sg.float()).abs()
    print(f'  kv maxdiff: {d.max().item():.4f}', flush=True)

# ---- 3. graph replay (warmup at pos 511, isolated) ----
nxt3 = prefill()
assert nxt3 == nxt
ext.set_input_embedding(embed_w[nxt])
torch.cuda.synchronize()
pos_gpu.fill_(511)
s1 = torch.cuda.Stream()
s1.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(s1):
    for _ in range(3):
        ext.step_graph()
torch.cuda.current_stream().wait_stream(s1)
torch.cuda.synchronize()

pos_gpu.fill_(128)
ext.set_input_embedding(embed_w[nxt])
torch.cuda.synchronize()
g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g):
    ext.step_graph()
print('captured', flush=True)

ext.set_input_embedding(embed_w[nxt])
pos_gpu.fill_(128)
torch.cuda.synchronize()
g.replay()
torch.cuda.synchronize()
tok_g = ext.get_last_token()
kv_g = kcs[0][:, 128, :].clone()
lg_g = ext.sf_get_lg().clone()

print(f'[graph replay vs eager step]', flush=True)
print(f'  token: {tok_ref} vs {tok_g} '
      f'{"MATCH" if tok_ref == tok_g else "DIFF"}', flush=True)
print(f'  kv slot bit-exact: '
      f'{torch.equal(bits(kv_ref), bits(kv_g))}', flush=True)
print(f'  kv slot vs sg: '
      f'{torch.equal(bits(kv_sg), bits(kv_g))}', flush=True)
if not torch.equal(bits(kv_ref), bits(kv_g)):
    d = (kv_ref.float() - kv_g.float()).abs()
    print(f'  kv maxdiff: {d.max().item():.4f}', flush=True)
    print(f'  kv_g abs scale: '
          f'{kv_g.float().abs().max().item():.4f}', flush=True)
print(f'  logits bit-exact: '
      f'{torch.equal(bits(lg_ref), bits(lg_g))}', flush=True)
