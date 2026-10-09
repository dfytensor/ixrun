# -*- coding: utf-8 -*-
"""int4-g32+warm parameter sweep on MiniCPM5-1B: group x nlev -> PPL/QA."""
import sys, time, gc, difflib
sys.path.insert(0, r'E:\IXRUN')
import pandas  # noqa: F401
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from ixrun.config import MODEL_PATH, DATASET_CACHE
from ixrun.eval_utils import eval_ppl, load_wikitext
from ixrun.linear import iter_quantizable_linears

torch.manual_seed(42)
DEV = 'cuda'

B_REL = torch.logspace(-3, 2, 15).tolist()
BETAS = [0.50, 0.55, 0.65, 0.75, 0.85, 0.95, 1.00]
TANH_B = [0.25, 0.4, 0.6, 0.85, 1.2, 1.8, 3.0]


def quant_gw(W, group=32, nlev=8, lloyd=25):
    of, inf = W.shape
    G = W.float().view(-1, group)
    sign = (G < 0)
    A = G.abs()
    gmax = A.amax(1, keepdim=True).clamp_min(1e-12)
    nm1 = nlev - 1

    best_mse = torch.full((A.shape[0],), float('inf'), device=A.device)
    best_lv = torch.zeros(A.shape[0], nlev, device=A.device)

    for br in B_REL:
        b = br * gmax
        for beta in BETAS:
            f = A ** beta / (b + A ** beta)
            fmax = gmax ** beta / (b + gmax ** beta)
            q = torch.round(f / fmax * nm1).clamp_(0, nm1)
            fd = (q / nm1 * fmax).clamp_(max=1 - 1e-6)
            wr = (fd * b / (1 - fd)).clamp_min(0)
            if beta != 1.0:
                wr = wr ** (1.0 / beta)
            wr = torch.minimum(wr, gmax)
            mse = ((A - wr) ** 2).mean(dim=1)
            imp = mse < best_mse
            if imp.any():
                best_mse = torch.where(imp, mse, best_mse)
                fdi = ((torch.arange(nlev, device=A.device,
                                     dtype=torch.float32)
                        / nm1).view(1, -1) * fmax).clamp(max=1 - 1e-6)
                lv = (fdi * b / (1 - fdi)).clamp_min(0)
                if beta != 1.0:
                    lv = lv ** (1.0 / beta)
                lv = torch.minimum(lv, gmax)
                best_lv = torch.where(imp.view(-1, 1), lv, best_lv)

    for br in TANH_B:
        b = br * gmax
        f = torch.tanh(A / b)
        fmax = torch.tanh(gmax / b)
        q = torch.round(f / fmax * nm1).clamp_(0, nm1)
        fd = (q / nm1 * fmax).clamp_(max=1 - 1e-6)
        wr = (b * torch.atanh(fd)).clamp_min(0)
        wr = torch.minimum(wr, gmax)
        mse = ((A - wr) ** 2).mean(dim=1)
        imp = mse < best_mse
        if imp.any():
            best_mse = torch.where(imp, mse, best_mse)
            fdi = ((torch.arange(nlev, device=A.device,
                                 dtype=torch.float32)
                    / nm1).view(1, -1) * fmax).clamp(max=1 - 1e-6)
            lv = (b * torch.atanh(fdi)).clamp_min(0)
            lv = torch.minimum(lv, gmax)
            best_lv = torch.where(imp.view(-1, 1), lv, best_lv)

    lv = best_lv.clone()
    for it in range(lloyd):
        d = (A.unsqueeze(-1) - lv.unsqueeze(1)).abs()
        idx = d.argmin(-1)
        new = lv.clone()
        for k in range(nlev):
            mm = (idx == k)
            cnt = mm.sum(1, keepdim=True)
            s = (A * mm).sum(1, keepdim=True)
            new[:, k:k + 1] = torch.where(cnt > 0, s / cnt.clamp_min(1),
                                          lv[:, k:k + 1])
        d_new = (A.unsqueeze(-1) - new.unsqueeze(1)).abs()
        mse_new = d_new.min(-1).values.mean(1)
        mse_old = d.min(-1).values.mean(1)
        better = (mse_new <= mse_old).view(-1, 1)
        lv = torch.where(better, new, lv)
    d = (A.unsqueeze(-1) - lv.unsqueeze(1)).abs()
    idx = d.argmin(-1)
    sgn = torch.where(sign, -1.0, 1.0)
    rec = sgn * lv.gather(1, idx)
    return rec.view(of, inf).to(W.dtype)


QA = ['用一句话介绍长城。', '中国的四大发明是什么？', 'Explain photosynthesis',
      'Python 最大公约数函数', '9.11 和 9.9 哪个大', '天空为什么是蓝色',
      'Capital of France', '秋天五言绝句']


@torch.no_grad()
def gen(m, tok, q, n=64):
    ids = tok(q, return_tensors='pt').input_ids.cuda()
    out = m.generate(ids, max_new_tokens=n, do_sample=False,
                     pad_token_id=tok.eos_token_id)
    return tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)


CONFIGS = [(32, 8), (64, 8), (16, 8), (64, 16), (32, 16), (16, 16)]

tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
m = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, torch_dtype=torch.bfloat16, trust_remote_code=True)
m.eval().cuda()
texts = load_wikitext(cache_dir=DATASET_CACHE)

targets = list(iter_quantizable_linears(m))
orig = [(name, mod.weight.data.detach().cpu().clone())
        for name, mod in targets]
ppl_bf16 = eval_ppl(m, tok, texts)
ans_bf16 = [gen(m, tok, q) for q in QA]
print(f'[bf16] ppl {ppl_bf16:.2f}', flush=True)


def restore():
    with torch.no_grad():
        for (name, mod), (_, w0) in zip(targets, orig):
            mod.weight.data.copy_(w0.cuda())


print(f'{"group":>5} {"nlev":>4} {"bpw":>6} {"ppl":>9} {"QA-sim":>7} '
      f'{"t(s)":>5}', flush=True)
for group, nlev in CONFIGS:
    t0 = time.time()
    with torch.no_grad():
        for name, mod in targets:
            mod.weight.data = quant_gw(mod.weight.data, group=group,
                                       nlev=nlev).to(mod.weight.dtype)
    torch.cuda.synchronize()
    tq = time.time() - t0
    ppl = eval_ppl(m, tok, texts)
    ans = [gen(m, tok, q) for q in QA]
    qa = sum(difflib.SequenceMatcher(None, a, b).ratio()
             for a, b in zip(ans, ans_bf16)) / len(QA) * 100
    bits = (nlev.bit_length() - 1) + 1 + 8 / group
    print(f'{group:>5} {nlev:>4} {bits:>6.2f} {ppl:>9.2f} {qa:>6.1f}% '
          f'{tq:>5.0f}', flush=True)
    restore()

print('\nrefs: bf16 16bpw ppl 56.02 | GSQ 5.5bpw ~57.2 | GMM 5.5bpw 57.92 '
      '| UDCQ 6bpw 58.06', flush=True)
print('DONE', flush=True)
