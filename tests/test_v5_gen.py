# -*- coding: utf-8 -*-
"""v5 multi-position + full gen: C++ engine end-to-end token generation."""
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
theta = float(rs.get('rope_theta', getattr(cfg, 'rope_theta', 10000.0)))
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
print(f'packed {len(pks)} linears + lm_head', flush=True)

src = open(r'E:\IXRUN\ixrun\cpp\engine_v5.cu',
           encoding='utf-8').read()
proto = '''
void graph_capture(
    torch::Tensor h_bf16, torch::Tensor pos_gpu,
    std::vector<torch::Tensor> kv_caches,
    std::vector<torch::Tensor> in_norms,
    std::vector<torch::Tensor> post_norms,
    std::vector<torch::Tensor> codes,
    std::vector<torch::Tensor> cbs,
    std::vector<torch::Tensor> s_i8s,
    std::vector<double> bases,
    std::vector<double> steps,
    std::vector<int64_t> out_fs,
    std::vector<int64_t> in_fs,
    int64_t n_heads, int64_t n_kv_heads,
    int64_t head_dim, int64_t ctx, double theta);
void graph_set_input(torch::Tensor embedding);
torch::Tensor graph_replay();
torch::Tensor decode_24(
    torch::Tensor h_bf16, torch::Tensor pos_gpu,
    std::vector<torch::Tensor> kv_caches,
    std::vector<torch::Tensor> in_norms,
    std::vector<torch::Tensor> post_norms,
    std::vector<torch::Tensor> codes,
    std::vector<torch::Tensor> cbs,
    std::vector<torch::Tensor> s_i8s,
    std::vector<double> bases,
    std::vector<double> steps,
    std::vector<int64_t> out_fs,
    std::vector<int64_t> in_fs,
    int64_t n_heads, int64_t n_kv_heads,
    int64_t head_dim, int64_t ctx, double theta);
torch::Tensor layer_forward(
    torch::Tensor h, torch::Tensor in_nw, torch::Tensor post_nw,
    torch::Tensor qc, torch::Tensor qcb, torch::Tensor qs,
    double qb, double qst, int64_t qo, int64_t qi,
    torch::Tensor kc, torch::Tensor kcb, torch::Tensor ks,
    double kb, double kst, int64_t ko, int64_t ki,
    torch::Tensor vc, torch::Tensor vcb, torch::Tensor vs,
    double vb, double vst, int64_t vo, int64_t vi,
    torch::Tensor oc, torch::Tensor ocb, torch::Tensor os8,
    double ob, double ost, int64_t oo, int64_t oi,
    torch::Tensor gc, torch::Tensor gcb, torch::Tensor gs8,
    double gb, double gst, int64_t go, int64_t gi,
    torch::Tensor uc, torch::Tensor ucb, torch::Tensor us8,
    double ub, double ust, int64_t uo, int64_t ui,
    torch::Tensor dc, torch::Tensor dcb, torch::Tensor ds8,
    double db, double dst, int64_t dof, int64_t dif,
    torch::Tensor kv_cache,
    int64_t n_heads, int64_t n_kv_heads,
    int64_t head_dim, int64_t ctx, double theta);
torch::Tensor rmsnorm_out(torch::Tensor x, torch::Tensor w);
torch::Tensor gemv_out(torch::Tensor x, torch::Tensor codes,
    torch::Tensor cb, torch::Tensor s_i8,
    double s_base, double s_step, int64_t out_f, int64_t in_f);
'''
ext = load_inline(name='ixrun_cpp_v5b', cpp_sources=[proto],
                  cuda_sources=[src],
                  functions=['layer_forward', 'rmsnorm_out',
                             'gemv_out', 'decode_24',
                             'graph_capture', 'graph_set_input',
                             'graph_replay'],
                  extra_cuda_cflags=['-O3', '--use_fast_math',
                                     '-allow-unsupported-compiler'],
                  verbose=False)


def pk_args(i):
    p = pks[i]
    return (p['codes5'], p['cb'], p['s_i8'],
            p['s_base'], p['s_step'], p['out_f'], p['in_f'])


def run_layer(h, l, pos, kc):
    b = l * 7
    return ext.layer_forward(
        h, in_ws[l], post_ws[l],
        *pk_args(b), *pk_args(b+1), *pk_args(b+2), *pk_args(b+3),
        *pk_args(b+4), *pk_args(b+5), *pk_args(b+6),
        kc, pos, nh, nkv, hd, CTX, theta)


def argmax_lm(h):
    hnm = ext.rmsnorm_out(h, fn_w)
    yf = ext.gemv_out(hnm, lh_pk['codes5'], lh_pk['cb'],
                      lh_pk['s_i8'], lh_pk['s_base'],
                      lh_pk['s_step'], lh_pk['out_f'], lh_pk['in_f'])
    return int(yf.float().argmax().item())


# reference: Python GSQ StepGraph
from ixrun.step_graph import StepGraphEngine
tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
texts = load_wikitext(cache_dir=DATASET_CACHE)
big = '\n'.join(texts)
prompt_ids = tok(big[20000:21200],
                 return_tensors='pt').input_ids[0, :256].cuda().tolist()
print(f'prompt: {len(prompt_ids)} tokens', flush=True)

eng = StepGraphEngine.from_pretrained(codec='gsq', verbose=False)
ref_text = eng.generate(tok.decode(prompt_ids), max_new_tokens=12)
del eng
torch.cuda.empty_cache()
ref_ids = tok(ref_text, return_tensors='pt'
              ).input_ids[0].tolist()[-12:]
print(f'py ref tail: {ref_ids}', flush=True)

# C++ chain: prefill + gen (per-layer KV caches!)
kcs = [torch.zeros(2*nkv, CTX, hd, dtype=torch.bfloat16,
                   device='cuda') for _ in range(24)]
# pre-flatten pack args for decode_24
all_codes = [p['codes5'] for p in pks]
all_cbs = [p['cb'] for p in pks]
all_s = [p['s_i8'] for p in pks]
all_bases = [p['s_base'] for p in pks]
all_steps = [p['s_step'] for p in pks]
all_out_f = [p['out_f'] for p in pks]
all_in_f = [p['in_f'] for p in pks]

# pos_gpu: int32 [1] tensor, updated between graph replays
pos_gpu = torch.zeros(1, dtype=torch.int32, device='cuda')

# --- prefill with eager decode_24 ---
t0 = time.time()
hh = None
for i, t in enumerate(prompt_ids):
    hh = embed_w[t].clone()
    pos_gpu.fill_(i)
    hh = ext.decode_24(hh, pos_gpu, kcs, in_ws, post_ws,
                       all_codes, all_cbs, all_s,
                       all_bases, all_steps, all_out_f, all_in_f,
                       nh, nkv, hd, CTX, theta)
t_prefill = time.time() - t0
print(f'prefill {len(prompt_ids)} tok in {t_prefill:.2f}s '
      f'({len(prompt_ids)/t_prefill:.1f} tok/s)', flush=True)

# --- CUDA graph capture + generation ---
# Save cache state (graph capture warmup pollutes it)
cache_snapshots = [kc.clone() for kc in kcs]

pos_gpu.fill_(256)
ext.graph_capture(hh, pos_gpu, kcs, in_ws, post_ws,
                  all_codes, all_cbs, all_s,
                  all_bases, all_steps, all_out_f, all_in_f,
                  nh, nkv, hd, CTX, theta)
print('graph captured', flush=True)

# Restore cache to post-prefill state
for kc, snap in zip(kcs, cache_snapshots):
    kc.copy_(snap)
del cache_snapshots
print('cache restored, graph ready', flush=True)

# Generation with CUDA graph replay
t0 = time.perf_counter()
nxt = argmax_lm(hh)  # first token from prefill (eager, no graph needed)
gen = [nxt]

for step in range(1, 12):
    pos = len(prompt_ids) + step - 1
    pos_gpu.fill_(pos)
    ext.graph_set_input(embed_w[nxt])
    hh = ext.graph_replay()
    torch.cuda.synchronize()
    if step <= 2:
        in_n = embed_w[nxt].float().norm().item()
        out_n = hh.float().norm().item()
        out_nan = bool(hh.float().isnan().any())
        print('  step%d: input_norm=%.4f out_norm=%.4f nan=%s'
              % (step, in_n, out_n, out_nan), flush=True)
    nxt = argmax_lm(hh)
    gen.append(nxt)
t_gen = time.perf_counter() - t0
print(f'gen 12 tok in {t_gen:.2f}s = {12/t_gen:.1f} tok/s', flush=True)
print(f'C++ gen:    {gen}', flush=True)
match = sum(1 for a, b_ in zip(gen, ref_ids) if a == b_)
print(f'token match: {match}/12', flush=True)
print(f'text: {tok.decode(gen)}', flush=True)
