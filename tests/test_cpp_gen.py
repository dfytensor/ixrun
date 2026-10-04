# -*- coding: utf-8 -*-
"""C2-final: token-level generation — python-glued 24x C++ layer_forward
+ GSQ lm_head, greedy, vs the Python GSQ StepGraph engine."""
import sys

sys.path.insert(0, r'E:\IXRUN')
import pandas  # noqa: F401
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from ixrun.config import MODEL_PATH, DATASET_CACHE
from ixrun.eval_utils import load_wikitext
from ixrun.linear import iter_quantizable_linears
from ixrun.step_graph import StepGraphEngine
from benchmarks.gsq_runtime import gs_pack, gs_decode_ref

m = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, dtype=torch.bfloat16,
    trust_remote_code=True).eval().cuda()
cfg = m.config
H = cfg.hidden_size
nh, nkv = cfg.num_attention_heads, cfg.num_key_value_heads
hd = getattr(cfg, 'head_dim', H // nh)
theta = float(getattr(cfg, 'rope_theta', 10000.0))
CTX = 256

sd = dict(m.named_modules())
targets = list(iter_quantizable_linears(m))
print(f'packing {len(targets)} linears...', flush=True)
pks = []
for name, mod in targets:
    pks.append(gs_pack(mod.weight.data.cuda()))
    mod.weight.data = torch.empty(0)
    torch.cuda.empty_cache()
print('packed', flush=True)

tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
texts = load_wikitext(cache_dir=DATASET_CACHE)
big = '\n'.join(texts)
prompt_ids = tok(big[20000:21200], return_tensors='pt') \
    .input_ids[0, :256].cuda().tolist()

src = open(r'E:\IXRUN\ixrun\cpp\engine.cu', encoding='utf-8').read()
proto = open(r'E:\IXRUN\tests\cpp_proto.h', encoding='utf-8').read()
ext = load_inline(name='ixrun_cpp_v4', cpp_sources=[proto],
                  cuda_sources=[src],
                  functions=['layer_forward', 'rmsnorm_out',
                             'gsq_gemv_out'],
                  extra_cuda_cflags=['-O3', '--use_fast_math',
                                     '-allow-unsupported-compiler'],
                  verbose=False)

n_layers = len(targets) // 7
names = [n for n, _ in targets]
in_ws = [sd[names[l * 7] .rsplit('.', 1)[0].rsplit('.', 1)[0]
          + '.input_layernorm'].weight.data.cuda()
         for l in range(n_layers)]
post_ws = [sd[names[l * 7 + 3].rsplit('.', 1)[0]
            + '.post_attention_layernorm'].weight.data.cuda()
           for l in range(n_layers)]
print(f'{n_layers} layers', flush=True)


def argmax_pack(pk, h):
    yf = ext.gsq_gemv_out(h, pk['codes5'].cuda(), pk['cb'].cuda(),
                          pk['s_i8'].cuda(), pk['s_base'], pk['s_step'],
                          pk['out_f'], pk['in_f'])
    return int(yf.float().argmax().item())


eng = StepGraphEngine.from_pretrained(codec='gsq', verbose=False)
out = eng.generate(tok.decode(tok(prompt_ids)), max_new_tokens=12)
ref_ids = tok(out, return_tensors='pt').input_ids[0].tolist()
print('py-gsq ref ids:', ref_ids[:12], flush=True)
del eng
torch.cuda.empty_cache()

embed_w = m.model.embed_tokens.weight.data.cuda()
fn_w = sd['model.norm'].weight.data.cuda()
lh_pk = gs_pack(targets[-1][1].weight.data.cuda())
lh_pk['codes5'] = lh_pk['codes5'].cuda()
lh_pk['cb'] = lh_pk['cb'].cuda()
lh_pk['s_i8'] = lh_pk['s_i8'].cuda()
for pk in pks:
    for k in ('codes5', 'cb', 's_i8'):
        pk[k] = pk[k].cuda()

n_layers = len(targets) // 7


def cpp_next(h, pos):
    hh = h.clone()
    for l in range(n_layers):
        b = l * 7
        hh = ext.layer_forward(
            hh, in_ws[l], post_ws[l],
            pks[b + 0]['codes5'], pks[b + 0]['cb'], pks[b + 0]['s_i8'],
            pks[b + 0]['s_base'], pks[b + 0]['s_step'],
            pks[b + 0]['out_f'], pks[b + 0]['in_f'],
            pks[b + 1]['codes5'], pks[b + 1]['cb'], pks[b + 1]['s_i8'],
            pks[b + 1]['s_base'], pks[b + 1]['s_step'],
            pks[b + 1]['out_f'], pks[b + 1]['in_f'],
            pks[b + 2]['codes5'], pks[b + 2]['cb'], pks[b + 2]['s_i8'],
            pks[b + 2]['s_base'], pks[b + 2]['s_step'],
            pks[b + 2]['out_f'], pks[b + 2]['in_f'],
            pks[b + 3]['codes5'], pks[b + 3]['cb'], pks[b + 3]['s_i8'],
            pks[b + 3]['s_base'], pks[b + 3]['s_step'],
            pks[b + 3]['out_f'], pks[b + 3]['in_f'],
            pks[b + 4]['codes5'], pks[b + 4]['cb'], pks[b + 4]['s_i8'],
            pks[b + 4]['s_base'], pks[b + 4]['s_step'],
            pks[b + 4]['out_f'], pks[b + 4]['in_f'],
            pks[b + 5]['codes5'], pks[b + 5]['cb'], pks[b + 5]['s_i8'],
            pks[b + 5]['s_base'], pks[b + 5]['s_step'],
            pks[b + 5]['out_f'], pks[b + 5]['in_f'],
            pks[b + 6]['codes5'], pks[b + 6]['cb'], pks[b + 6]['s_i8'],
            pks[b + 6]['s_base'], pks[b + 6]['s_step'],
            pks[b + 6]['out_f'], pks[b + 6]['in_f'],
            kc, pos, nh, nkv, hd, CTX, theta)
    hnf = ext.rmsnorm_out(hh, fn_w)
    return argmax_pack(lh_pk, hnf)


h = embed_w[prompt_ids].squeeze(0)
for i, t in enumerate(prompt_ids):
    h = embed_w[t].clone()
    pos = i
    hh = h
    for l in range(n_layers):
        b = l * 7
        hh = ext.layer_forward(
            hh, in_ws[l], post_ws[l],
            pks[b + 0]['codes5'], pks[b + 0]['cb'], pks[b + 0]['s_i8'],
            pks[b + 0]['s_base'], pks[b + 0]['s_step'],
            pks[b + 0]['out_f'], pks[b + 0]['in_f'],
            pks[b + 1]['codes5'], pks[b + 1]['cb'], pks[b + 1]['s_i8'],
            pks[b + 1]['s_base'], pks[b + 1]['s_step'],
            pks[b + 1]['out_f'], pks[b + 1]['in_f'],
            pks[b + 2]['codes5'], pks[b + 2]['cb'], pks[b + 2]['s_i8'],
            pks[b + 2]['s_base'], pks[b + 2]['s_step'],
            pks[b + 2]['out_f'], pks[b + 2]['in_f'],
            pks[b + 3]['codes5'], pks[b + 3]['cb'], pks[b + 3]['s_i8'],
            pks[b + 3]['s_base'], pks[b + 3]['s_step'],
            pks[b + 3]['out_f'], pks[b + 3]['in_f'],
            pks[b + 4]['codes5'], pks[b + 4]['cb'], pks[b + 4]['s_i8'],
            pks[b + 4]['s_base'], pks[b + 4]['s_step'],
            pks[b + 4]['out_f'], pks[b + 4]['in_f'],
            pks[b + 5]['codes5'], pks[b + 5]['cb'], pks[b + 5]['s_i8'],
            pks[b + 5]['s_base'], pks[b + 5]['s_step'],
            pks[b + 5]['out_f'], pks[b + 5]['in_f'],
            pks[b + 6]['codes5'], pks[b + 6]['cb'], pks[b + 6]['s_i8'],
            pks[b + 6]['s_base'], pks[b + 6]['s_step'],
            pks[b + 6]['out_f'], pks[b + 6]['in_f'],
            kc, pos, nh, nkv, hd, CTX, theta)
gen = []
hh = h
for step in range(12):
    pos = len(prompt_ids) + step - 1
    nxt = cpp_next(hh, pos)
    gen.append(nxt)
    hh = embed_w[nxt].clone()
print('C++ gen ids:', gen, flush=True)
print('py ref ids :', ref_ids[:12], flush=True)
match = sum(1 for a, b in zip(gen, ref_ids) if a == b)
print(f'token match: {match}/12', flush=True)
print(tok.decode(gen), flush=True)