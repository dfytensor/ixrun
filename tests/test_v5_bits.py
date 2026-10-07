# -*- coding: utf-8 -*-
"""Bit-level bisect: eager step() vs graph replay at ONE position.
Compare: KV cache slot write, final-norm out, logits."""
import sys
sys.path.insert(0, r'E:\IXRUN')
import pandas  # noqa: F401
import torch
from torch.utils.cpp_extension import load_inline
from transformers import AutoModelForCausalLM
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

src = open(r'E:\RUN'.replace('RUN', 'IXRUN') + r'\ixrun\cpp\engine_v5.cu',
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
torch::Tensor sf_get_fn();
'''
ext = load_inline(name='ixrun_cpp_v5sf4', cpp_sources=[proto],
                  cuda_sources=[src],
                  functions=['init_model', 'step', 'step_graph',
                             'set_input_embedding',
                             'get_last_token',
                             'sf_get_lg', 'sf_get_fn'],
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

from transformers import AutoTokenizer
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

def prefill():
    for kc in kcs:
        kc.zero_()
    nxt = prompt_ids[0]
    for i, t in enumerate(prompt_ids):
        pos_gpu.fill_(i)
        nxt = ext.step(t)
    return nxt

# ---- eager ref step at pos 128 ----
nxt = prefill()
pos_gpu.fill_(128)
ref_tok = ext.step(nxt)   # eager, syncs inside via .item()
torch.cuda.synchronize()
ref_lg = ext.sf_get_lg().clone()    # logits from step's buffer
ref_fn = ext.sf_get_fn().clone()    # final-norm out
ref_kv0 = kcs[0][:, 128, :].clone() # layer0 k+v at slot 128

# ---- graph replay at pos 128 (fresh prefill, same state) ----
nxt2 = prefill()
assert nxt2 == nxt, f'prefill nondeterministic {nxt2} vs {nxt}'
ext.set_input_embedding(embed_w[nxt])
torch.cuda.synchronize()
pos_gpu.fill_(128)
s1 = torch.cuda.Stream()
s1.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(s1):
    for _ in range(3):
        ext.step_graph()
torch.cuda.current_stream().wait_stream(s1)
g_old = torch.cuda.CUDAGraph()
pos_gpu.fill_(128)
torch.cuda.synchronize()
with torch.cuda.graph(g_old):
    ext.step_graph()

# replay at pos 128: re-seed caches at 128..130 were polluted by
# warmup; replay rewrites slot 128 before attention reads it.
ext.set_input_embedding(embed_w[nxt])
pos_gpu.fill_(128)
torch.cuda.synchronize()
g_old.replay()
torch.cuda.synchronize()
g_tok = ext.get_last_token()
g_lg = ext.sf_get_lg().clone()
g_fn = ext.sf_get_fn().clone()
g_kv0 = kcs[0][:, 128, :].clone()

print(f'token: ref={ref_tok} graph={g_tok} '
      f'{"MATCH" if ref_tok == g_tok else "DIFF"}', flush=True)
kv_same = torch.equal(ref_kv0.view(torch.uint16),
                      g_kv0.view(torch.uint16))
fn_d = (ref_fn - g_fn).abs().max().item()
fn_n = ref_fn.abs().max().item()
lg_same_ids = (ref_lg == g_lg).float().mean().item()
lg_d = (ref_lg - g_lg).abs().max().item()
print(f'kv slot128 bit-exact: {kv_same}', flush=True)
print(f'final-norm maxdiff: {fn_d:.6e} (scale {fn_n:.3f})',
      flush=True)
print(f'logits: {lg_same_ids*100:.2f}% equal, '
      f'maxdiff {lg_d:.6e}', flush=True)
if not kv_same:
    d = (ref_kv0.float() - g_kv0.float()).abs()
    print(f'kv diff per-head max: '
          f'{d.max(dim=1).values.tolist()}', flush=True)
