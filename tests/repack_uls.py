# -*- coding: utf-8 -*-
"""Repack UDCQ fp16-scale blob -> ULS (8-bit log2 per-row scale) blob.

Scale stream per row: uint8 [8-byte header (float base, float step)][i8 codes]
  scale_decoded(g) = 2 ** (base + step * i8[g])
Codes/sign streams are copied UNTOUCHED - no weight requantization.
Usage: python tests/repack_uls.py [--check N] [--dst PATH]
"""
import sys, time, argparse
sys.path.insert(0, r'E:\IXRUN')
import pandas  # noqa
import torch

SRC = r'E:\IXRUN\experiments\qwen38_udcq\q38_blob.pt'
DST = r'F:\models\qwen38_uls_blob.pt'
MODEL = r'E:\models\Qwen3.8-27B'


def shape_map():
    from transformers import AutoConfig
    cfg = AutoConfig.from_pretrained(MODEL)
    cfg = getattr(cfg, 'text_config', cfg)
    NL = cfg.num_hidden_layers
    ATTN = sorted(i for i, x in enumerate(cfg.layer_types)
                  if x != 'linear_attention')
    hidden, inter = cfg.hidden_size, cfg.intermediate_size
    nh, nkv, hd = (cfg.num_attention_heads, cfg.num_key_value_heads,
                   cfg.head_dim)
    nv, nk = cfg.linear_num_value_heads, cfg.linear_num_key_heads
    vdim, kdim = cfg.linear_value_head_dim, cfg.linear_key_head_dim
    m = {}
    for l in range(NL):
        pre = f'model.layers.{l}.'
        if l in ATTN:
            for nm, of, inf in [('self_attn.q_proj', 2*nh*hd, hidden),
                                ('self_attn.k_proj', nkv*hd, hidden),
                                ('self_attn.v_proj', nkv*hd, hidden),
                                ('self_attn.o_proj', hidden, nh*hd),
                                ('mlp.gate_proj', inter, hidden),
                                ('mlp.up_proj', inter, hidden),
                                ('mlp.down_proj', hidden, inter)]:
                m[pre + nm] = (of, inf)
        else:
            for nm, of, inf in [('linear_attn.in_proj_qkv', 2*nk*kdim + nv*vdim, hidden),
                                ('linear_attn.in_proj_z', nv*vdim, hidden),
                                ('linear_attn.in_proj_b', nv, hidden),
                                ('linear_attn.in_proj_a', nv, hidden),
                                ('linear_attn.out_proj', hidden, nv*vdim),
                                ('mlp.gate_proj', inter, hidden),
                                ('mlp.up_proj', inter, hidden),
                                ('mlp.down_proj', hidden, inter)]:
                m[pre + nm] = (of, inf)
    m['lm_head'] = (cfg.vocab_size, hidden)
    return m


def encode_row(s):
    """s: [out, n_gr] fp32 -> (buf [out, 8+n_gr] uint8)"""
    out, n_gr = s.shape
    pos = s > 0
    lg = torch.log2(torch.where(pos, s, torch.ones_like(s)))
    lg = torch.where(pos, lg, torch.full_like(lg, -30.0))   # zeros -> below base
    real = torch.where(pos, lg, torch.full_like(lg, float('inf')))
    base = real.min(dim=1).values
    base = torch.where(torch.isfinite(base), base, torch.full_like(base, -30.0))
    mx = lg.max(dim=1).values
    mx = torch.where(torch.isfinite(mx), mx, torch.full_like(mx, -30.0))
    step = (mx - base) / 255.0
    i8 = ((lg - base[:, None]) / torch.where(step > 0, step, torch.ones_like(step))[:, None]
          ).round().clamp(0, 255).to(torch.uint8)
    i8 = torch.where(pos, i8, torch.zeros_like(i8))         # zero groups -> i8=0
    dec = torch.exp2(base[:, None] + step[:, None] * i8.float())
    rel = (dec - s).abs() / s.clamp_min(2 ** -24)
    rel = torch.where(pos, rel, torch.zeros_like(rel))
    hdr = torch.stack([base, step], dim=1).contiguous()      # [out, 2] fp32
    buf = torch.empty(out, 8 + n_gr, dtype=torch.uint8)
    buf[:, :8] = hdr.view(torch.uint8).view(out, 8)
    buf[:, 8:] = i8
    return buf, rel.max().item(), rel.mean().item(), int((~pos).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--check', type=int, default=0)
    ap.add_argument('--dst', default=DST)
    a = ap.parse_args()

    t0 = time.perf_counter()
    blob = torch.load(SRC, map_location='cpu', mmap=True, weights_only=True)
    smap = shape_map()
    print(f'loaded mmap {time.perf_counter()-t0:.0f}s', flush=True)
    keys = list(blob['layers'].keys())
    for k in keys:
        of, inf = smap[k]
        p = blob['layers'][k]
        assert p['idx'].numel() == of * inf // 2, (k, 'idx', p['idx'].numel(), of*inf//2)
        assert p['scale'].numel() == of * inf // 16, (k, 'scale', p['scale'].numel(), of*inf//16)
    print('shape map verified for all 497', flush=True)

    if a.check:
        wk, wmax, wmean, nz = None, 0.0, 0.0, 0
        for k in keys[:a.check]:
            of, inf = smap[k]
            s2 = blob['layers'][k]['scale'].float().view(of, inf // 16)
            _, rmax, rmean, z = encode_row(s2)
            nz += z
            if rmax > wmax:
                wmax, wk = rmax, k
            wmean = max(wmean, rmean)
        print(f'CHECK {a.check} mats: worst row-rel {wmax:.5f} ({wk}) | '
              f'worst mean {wmean:.5f} | zero scales {nz}', flush=True)
        return

    out = {'uls': True, 'codebook': blob['codebook'], 'embed': blob['embed'],
           'layers': {}}
    wk, wmax, nz = None, 0.0, 0
    for i, k in enumerate(keys):
        of, inf = smap[k]
        p = blob['layers'][k]
        s2 = p['scale'].float().view(of, inf // 16)
        buf, rmax, rmean, z = encode_row(s2)
        nz += z
        if rmax > wmax:
            wmax, wk = rmax, k
        out['layers'][k] = {'idx': p['idx'], 'sign': p['sign'], 'scale': buf}
        if (i + 1) % 100 == 0 or i + 1 == len(keys):
            print(f'  {i+1}/{len(keys)} worst row-rel {wmax:.5f} '
                  f'({wk}) zeros {nz} ({time.perf_counter()-t0:.0f}s)', flush=True)
    t1 = time.perf_counter()
    torch.save(out, a.dst)
    print(f'saved {a.dst} in {time.perf_counter()-t1:.0f}s | '
          f'worst row-rel {wmax:.5f} ({wk}) | zero scales {nz} | '
          f'total {time.perf_counter()-t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
