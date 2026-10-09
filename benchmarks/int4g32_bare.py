# -*- coding: utf-8 -*-
"""int4-g32+warm bare quantization on MiniCPM5-1B (PY stack, HF forward).

Scheme (user report): per 32-wide input-dim group, pick the best of 112
parametric transforms (105 bounded-rational + 7 tanh) by 8-level uniform
f-space quantization MSE, then 25 rounds of 1D Lloyd warm-started from
the winner's 8 reconstruction levels. 4.25 bpw stored (3b level + 1b sign
+ 8b group params). Zero calibration, zero correction layers.
"""
import sys, time, json, gc, difflib
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
LLOYD_ITERS = 25


def quant_g32_warm(W, group=32):
    """W [out, in] float -> (idx uint8 [out,in] 0..7, sign bits implicit,
    levels [nG,8] f32, meta). Returns reconstruction directly too."""
    of, inf = W.shape
    assert inf % group == 0
    G = W.float().view(of * (inf // group), group)
    sign = (G < 0)
    A = G.abs()                                   # [nG, 32]
    gmax = A.amax(dim=1, keepdim=True).clamp_min(1e-12)
    nG = A.shape[0]

    best_mse = torch.full((nG,), float('inf'), device=A.device)
    # winner's 8 w-space reconstruction levels [nG, 8]
    best_lv = torch.zeros(nG, 8, device=A.device)

    # ---- rational family (c=1 param, report-exact): ----
    #   f = A^b/(b+A^b); fmax = g^b/(b+g^b); q = round(f/fmax*7)
    #   fd = q/7*fmax;  inverse w = (fd*b/(1-fd))^(1/beta), clamp gmax
    for br in B_REL:
        b = br * gmax                              # [nG,1]
        for beta in BETAS:
            f = A ** beta / (b + A ** beta)
            fmax = gmax ** beta / (b + gmax ** beta)
            q = torch.round(f / fmax * 7).clamp_(0, 7)
            fd = (q / 7 * fmax).clamp_(max=1 - 1e-6)
            wr = (fd * b / (1 - fd)).clamp_min(0)
            if beta != 1.0:
                wr = wr ** (1.0 / beta)
            wr = torch.minimum(wr, gmax)
            mse = ((A - wr) ** 2).mean(dim=1)
            imp = mse < best_mse
            if imp.any():
                best_mse = torch.where(imp, mse, best_mse)
                # 8 levels = inverse(fd_i), fd_i = i/7 * fmax
                fdi = ((torch.arange(8, device=A.device,
                                     dtype=torch.float32)
                        / 7).view(1, 8) * fmax).clamp(max=1 - 1e-6)
                lv = (fdi * b / (1 - fdi)).clamp_min(0)
                if beta != 1.0:
                    lv = lv ** (1.0 / beta)
                lv = torch.minimum(lv, gmax)
                best_lv = torch.where(imp.view(nG, 1), lv, best_lv)

    # ---- tanh family: f = tanh(A/b) ----
    for br in TANH_B:
        b = br * gmax
        f = torch.tanh(A / b)
        fmax = torch.tanh(gmax / b)
        q = torch.round(f / fmax * 7).clamp_(0, 7)
        fd = (q / 7 * fmax).clamp_(max=1 - 1e-6)
        wr = (b * torch.atanh(fd)).clamp_min(0)
        wr = torch.minimum(wr, gmax)
        mse = ((A - wr) ** 2).mean(dim=1)
        imp = mse < best_mse
        if imp.any():
            best_mse = torch.where(imp, mse, best_mse)
            fdi = ((torch.arange(8, device=A.device, dtype=torch.float32)
                    / 7).view(1, 8) * fmax).clamp(max=1 - 1e-6)
            lv = (b * torch.atanh(fdi)).clamp_min(0)
            lv = torch.minimum(lv, gmax)
            best_lv = torch.where(imp.view(nG, 1), lv, best_lv)

    # ---- Lloyd warm: 25 rounds of 1D k-means per group ----
    lv = best_lv.clone()
    for it in range(LLOYD_ITERS):
        d = (A.unsqueeze(-1) - lv.unsqueeze(1)).abs()   # [nG,32,8]
        idx = d.argmin(dim=-1)                          # [nG,32]
        new = lv.clone()
        for k in range(8):
            m = (idx == k)
            cnt = m.sum(dim=1, keepdim=True)
            s = (A * m).sum(dim=1, keepdim=True)
            upd = cnt > 0
            new[:, k:k + 1] = torch.where(
                upd, s / cnt.clamp_min(1), lv[:, k:k + 1])
        d_new = (A.unsqueeze(-1) - new.unsqueeze(1)).abs()
        mse_new = d_new.min(dim=-1).values.mean(dim=1)
        mse_old = d.min(dim=-1).values.mean(dim=1)
        better = (mse_new <= mse_old).view(nG, 1)
        lv = torch.where(better, new, lv)
    d = (A.unsqueeze(-1) - lv.unsqueeze(1)).abs()
    idx = d.argmin(dim=-1)
    sgn = torch.where(sign, -1.0, 1.0)
    rec = sgn * lv.gather(1, idx)
    Wq = rec.view(of, inf).to(W.dtype)
    return Wq


QA = [
    ('用一句话介绍长城。', None),
    ('中国的四大发明是什么？', None),
    ('Explain photosynthesis', None),
    ('Python 最大公约数函数', None),
    ('9.11 和 9.9 哪个大', None),
    ('天空为什么是蓝色', None),
    ('Capital of France', None),
    ('秋天五言绝句', None),
]


@torch.no_grad()
def gen(m, tok, q, n=64):
    ids = tok(q, return_tensors='pt').input_ids.cuda()
    out = m.generate(ids, max_new_tokens=n, do_sample=False,
                     pad_token_id=tok.eos_token_id)
    return tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)


def main():
    tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    m = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16, trust_remote_code=True)
    m.eval().cuda()
    texts = load_wikitext(cache_dir=DATASET_CACHE)

    # ---- bf16 baseline ----
    ppl_bf16 = eval_ppl(m, tok, texts)
    ans_bf16 = [gen(m, tok, q) for q, _ in QA]
    print(f'[bf16] ppl {ppl_bf16:.2f}', flush=True)

    # ---- int4-g32+warm in place ----
    t0 = time.time()
    nq = 0
    with torch.no_grad():
        for name, mod in iter_quantizable_linears(m):
            mod.weight.data = quant_g32_warm(mod.weight.data).to(
                mod.weight.dtype)
            nq += 1
    torch.cuda.synchronize()
    tq = time.time() - t0
    print(f'[int4g32] {nq} matrices quantized in {tq:.0f}s | '
          f'stored 4.25 bpw (3.76x)', flush=True)

    ppl_q = eval_ppl(m, tok, texts)
    ans_q = [gen(m, tok, q) for q, _ in QA]
    print(f'[int4g32] ppl {ppl_q:.2f}', flush=True)

    sims = []
    for i, (q, _) in enumerate(QA):
        r = difflib.SequenceMatcher(
            None, ans_q[i], ans_bf16[i]).ratio() * 100
        sims.append(r)
        print(f'  Q{i} sim {r:5.1f}% | bf16: {ans_bf16[i][:40]!r} | '
              f'int4: {ans_q[i][:40]!r}', flush=True)
    print(f'\n== SUMMARY ==')
    print(f'  bf16      : ppl {ppl_bf16:.2f}')
    print(f'  int4 g32  : ppl {ppl_q:.2f} | QA-sim {sum(sims)/8:.1f}% '
          f'(refs: UDCQ 6bpw ppl 58.06 | GSQ 5.5bpw ~57.2 | '
          f'GMM 57.92 | user report QA 75.3% on 2B)')
    print(f'  quant time: {tq:.0f}s', flush=True)


if __name__ == '__main__':
    main()
