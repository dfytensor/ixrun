# -*- coding: utf-8 -*-
"""v5 static-init generation: init_model once, then step(token_id)."""
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
void step_graph();
void set_start_token(int64_t token);
int64_t get_last_token();
void set_input_embedding(torch::Tensor embedding);
torch::Tensor decode_24(
    torch::Tensor h_bf16, torch::Tensor pos_gpu,
    std::vector<torch::Tensor> kv_caches,
    std::vector<torch::Tensor> in_norms,
    std::vector<torch::Tensor> post_norms,
    std::vector<torch::Tensor> codes,
    std::vector<torch::Tensor> cbs,
    std::vector<torch::Tensor> s_i8s,
    std::vector<double> bases, std::vector<double> steps,
    std::vector<int64_t> out_fs, std::vector<int64_t> in_fs,
    int64_t n_heads, int64_t n_kv_heads,
    int64_t head_dim, int64_t ctx, double theta);
torch::Tensor rmsnorm_out(torch::Tensor x, torch::Tensor w);
torch::Tensor gemv_out(torch::Tensor x, torch::Tensor codes,
    torch::Tensor cb, torch::Tensor s_i8,
    double s_base, double s_step,
    int64_t out_f, int64_t in_f);
std::vector<int64_t> generate_batch(
    int64_t start_token, int64_t n_tokens);
std::vector<int64_t> generate_batch_fast(
    int64_t start_token, int64_t n_tokens);
'''
ext = load_inline(name='ixrun_cpp_v5o', cpp_sources=[proto],
                  cuda_sources=[src],
                  functions=['init_model', 'step',
                             'generate_batch',
                             'generate_batch_fast',
                             'decode_24', 'rmsnorm_out',
                             'gemv_out', 'step_graph',
                             'set_start_token',
                             'get_last_token',
                             'set_input_embedding'],
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

# Init once
ext.init_model(embed_w, pos_gpu, kcs, in_ws, post_ws, fn_w,
               all_codes, all_cbs, all_s, all_bases, all_steps,
               all_out_f, all_in_f,
               lh_pk['codes5'], lh_pk['cb'], lh_pk['s_i8'],
               lh_pk['s_base'], lh_pk['s_step'],
               lh_pk['out_f'], lh_pk['in_f'],
               nh, nkv, hd, CTX, theta)
print('model initialized', flush=True)

# Prefill (one step per token, no tensor args!)
t0 = time.perf_counter()
nxt = prompt_ids[0]
for i, t in enumerate(prompt_ids):
    pos_gpu.fill_(i)
    nxt = ext.step(t)
t_pf = time.perf_counter() - t0
print(f'prefill {len(prompt_ids)} tok in {t_pf:.2f}s '
      f'({len(prompt_ids)/t_pf:.1f} tok/s)', flush=True)

# --- PyTorch CUDA Graph wrapping C++ decode_24 (StepGraph architecture) ---
# PyTorch handles WDDM memory pools; C++ handles 24-layer compute.
# The key: decode_24's layer_forward uses STATIC buffers (no alloc after
# warmup) + we update input/pos between replays via static Python tensors.

static_input = torch.zeros(H, dtype=torch.bfloat16, device='cuda')
graph_out = None

# Warmup on side stream (initializes static buffers in layer_forward)
s = torch.cuda.Stream()
s.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(s):
    for _ in range(3):
        graph_out = ext.decode_24(static_input, pos_gpu, kcs,
                                  in_ws, post_ws,
                                  all_codes, all_cbs, all_s,
                                  all_bases, all_steps,
                                  all_out_f, all_in_f,
                                  nh, nkv, hd, CTX, theta)
torch.cuda.current_stream().wait_stream(s)

# Capture with PyTorch CUDA graph
g = torch.cuda.CUDAGraph()
pos_gpu.fill_(256)
with torch.cuda.graph(g):
    graph_out = ext.decode_24(static_input, pos_gpu, kcs,
                              in_ws, post_ws,
                              all_codes, all_cbs, all_s,
                              all_bases, all_steps,
                              all_out_f, all_in_f,
                              nh, nkv, hd, CTX, theta)
print('pytorch CUDA graph captured!', flush=True)

# Verify replay works with new input
static_input.copy_(embed_w[nxt])
pos_gpu.fill_(256)
g.replay()
torch.cuda.synchronize()
print(f'graph out norm = {graph_out.float().norm().item():.4f}',
      flush=True)

# Benchmark: graph replay per token
tokens = []
t0 = time.perf_counter()
for step_i in range(64):
    pos_gpu.fill_(len(prompt_ids) + step_i)
    g.replay()
    torch.cuda.synchronize()
    # argmax from graph output
    hnm = ext.rmsnorm_out(graph_out, fn_w)
    lg = ext.gemv_out(hnm, lh_pk['codes5'], lh_pk['cb'],
                      lh_pk['s_i8'], lh_pk['s_base'],
                      lh_pk['s_step'], lh_pk['out_f'], lh_pk['in_f'])
    nxt2 = int(lg.float().argmax().item())
    tokens.append(nxt2)
    static_input.copy_(embed_w[nxt2])
t_g = time.perf_counter() - t0
print(f'py-graph 64 tok in {t_g:.3f}s = {64/t_g:.1f} tok/s', flush=True)
print(f'text: {tok.decode(tokens)}', flush=True)

# --- FULL-GRAPH: 24 layers + norm + lm_head + argmax (no feedback loop) ---
del g
torch.cuda.synchronize()
torch.cuda.empty_cache()

# Initialize buffers on default stream
ext.set_input_embedding(embed_w[nxt])
torch.cuda.synchronize()

# Warmup on side stream
pos_gpu.fill_(256)
s2 = torch.cuda.Stream()
s2.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(s2):
    for _ in range(3):
        ext.step_graph()
torch.cuda.current_stream().wait_stream(s2)

# Capture
g2 = torch.cuda.CUDAGraph()
pos_gpu.fill_(256)
ext.set_input_embedding(embed_w[nxt])
torch.cuda.synchronize()
with torch.cuda.graph(g2):
    ext.step_graph()
print('FULL graph captured!', flush=True)

# Generate: Python sets input embedding, graph does the rest
gen_full = [nxt]
t0 = time.perf_counter()
for step_i in range(64):
    pos_gpu.fill_(len(prompt_ids) + step_i)
    g2.replay()
    torch.cuda.synchronize()
    tok = ext.get_last_token()  # one sync read
    gen_full.append(tok)
    ext.set_input_embedding(embed_w[tok])  # set next input
t_full = time.perf_counter() - t0
print(f'FULL-graph 64 tok in {t_full:.3f}s = '
      f'{64/t_full:.1f} tok/s', flush=True)
print(f'text: {tok.decode(gen_full)}', flush=True)
