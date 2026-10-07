# -*- coding: utf-8 -*-
"""Batched prefill gate: kcs bit-exact vs per-token prefill_tokens."""
import sys, time
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
lay0 = targets[0][0].rsplit('.', 3)[0]
pks = []
for _, mod in targets:
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
void prefill_tokens(std::vector<int64_t> toks, int64_t start_pos);
torch::Tensor prefill_batch(torch::Tensor ids_gpu,
                            int64_t start_pos);
'''
ext = load_inline(name='ixrun_cpp_v5pb1', cpp_sources=[proto],
                  cuda_sources=[src],
                  functions=['init_model', 'step', 'prefill_tokens',
                             'prefill_batch'],
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
                 return_tensors='pt').input_ids[0, :192].tolist()

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

def snap():
    return [kc.clone() for kc in kcs]

# A. per-token reference
for kc in kcs:
    kc.zero_()
ext.prefill_tokens(prompt_ids[:-1], 0)
torch.cuda.synchronize()
ref_kcs = snap()
pos_gpu.fill_(len(prompt_ids) - 1)
ref_nxt = ext.step(prompt_ids[-1])

# B. batched
for kc in kcs:
    kc.zero_()
ids_gpu = torch.tensor(prompt_ids[:-1], dtype=torch.int64,
                       device='cuda')
t0 = time.perf_counter()
ext.prefill_batch(ids_gpu, 0)
torch.cuda.synchronize()
t_b = time.perf_counter() - t0
bat_kcs = snap()
pos_gpu.fill_(len(prompt_ids) - 1)
bat_nxt = ext.step(prompt_ids[-1])

same_all = all(torch.equal(a.view(torch.uint16),
                           b.view(torch.uint16))
               for a, b in zip(ref_kcs, bat_kcs))
print(f'kcs bit-exact: {same_all}', flush=True)
print(f'nxt: ref={ref_nxt} batch={bat_nxt} '
      f'{"MATCH" if ref_nxt == bat_nxt else "DIFF"}', flush=True)
if not same_all:
    for l, (a, b) in enumerate(zip(ref_kcs, bat_kcs)):
        if not torch.equal(a.view(torch.uint16),
                           b.view(torch.uint16)):
            d = (a.float() - b.float()).abs().max().item()
            print(f'  layer {l} maxdiff {d:.4f}', flush=True)
            break
print(f'batch prefill {len(prompt_ids)-1} tok in {t_b*1000:.0f}ms '
      f'({(len(prompt_ids)-1)/t_b:.0f} tok/s)', flush=True)
