"""Correctness tests for PEAK-Q (ixrun.peakq).

Run:  python -m tests.test_peakq
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

from ixrun.peakq import (
    peakq_quantize,
    decode_peakq_scatter,
    decode_peakq_triton,
    precompute_peakq_offsets,
    PeakQLinear,
    deploy_peakq,
    peakq_snr,
    peakq_exact_pct,
    PEAKQ_TIERS,
    PEAKQ_GROUP,
)
from ixrun.quantize import int8x_quantize
from ixrun.triton_kernels import decode_weight_triton


def _heavy_tail(shape, seed=0):
    """Realistic LLM-ish weight: gaussian core + sparse large outliers."""
    torch.manual_seed(seed)
    base = torch.randn(shape) * 0.02
    out = torch.zeros(shape)
    n = int(out.numel() * 0.03)
    flat = out.view(-1)
    idx = torch.randperm(out.numel())[:n]
    flat[idx] = torch.randn(n) * 0.15
    return (base + out).bfloat16()


def _expo(w):
    return (w.reshape(-1).view(torch.int16).to(torch.int64) >> 7) & 0xFF


def test_tier1_bit_exact():
    w = _heavy_tail((128, 512), seed=1)
    p = peakq_quantize(w)
    dec = decode_peakq_scatter(p, device="cpu")

    expo = _expo(w)
    N = w.numel()
    e_pad = torch.cat([expo, torch.zeros((-N) % 16, dtype=torch.int64)])
    emax = e_pad.view(-1, 16).amax(-1)
    delta = (emax.unsqueeze(1) - e_pad.view(-1, 16)).reshape(-1)[:N]
    t1 = delta <= 1

    a = w.reshape(-1).view(torch.int16)
    b = dec.reshape(-1).view(torch.int16)
    assert (a[t1] == b[t1]).all(), "T1 (delta<=1) elements must be bit-exact"
    assert (a[delta == 0] == b[delta == 0]).all(), "group peaks must be bit-exact"

    pct = peakq_exact_pct(w, dec)
    assert pct >= 45.0, f"expected >=45% bit-exact (T1 fraction), got {pct:.1f}%"

    # per-element error bound: |dec-w| <= 2^-5 * 2^(emax_g - 127)
    peak_lo = torch.pow(2.0, (emax[(torch.arange(N) // 16)].double() - 127))
    err = (dec.reshape(-1).float() - w.reshape(-1).float()).abs().double()
    assert (err <= peak_lo * 2**-5).all(), "per-element error exceeds tier bound"
    print(f"[ok] T1 bit-exact; overall bit-exact={pct:.1f}% (T1 frac={t1.float().mean()*100:.1f}%)")


def test_zero_groups_preserved():
    w = _heavy_tail((64, 256), seed=2)
    w[3, :] = 0.0                       # whole row zero -> whole groups emax=0
    w[5, 7] = 0.0                       # isolated exact zero (sat path)
    p = peakq_quantize(w)
    dec = decode_peakq_scatter(p, device="cpu")
    assert (dec[3] == 0).all(), "all-zero groups must decode to exact zeros"
    # isolated zero: saturates into T3 -> decodes to ~2^-7 of its group peak
    gpeak = w[5, 0:16].abs().max().float()
    assert dec[5, 7].abs().float() < gpeak * 2**-5, "isolated zero error exceeds tier bound"
    print("[ok] zero groups / isolated zeros preserved")


def test_snr_vs_int8x():
    w = _heavy_tail((128, 512), seed=3)
    p = peakq_quantize(w)
    dec = decode_peakq_scatter(p, device="cpu")
    snr_pk = peakq_snr(w, dec)

    p8 = int8x_quantize(w)
    from ixrun.triton_kernels import decode_weight_scatter
    dec8 = decode_weight_scatter(p8, device="cpu")
    snr_i8 = peakq_snr(w, dec8)

    assert snr_pk > snr_i8 + 8.0, (
        f"PEAK-Q SNR {snr_pk:.1f}dB should beat INT8-X {snr_i8:.1f}dB by >8dB"
    )
    print(f"[ok] SNR: PEAK-Q {snr_pk:.1f} dB @ {p['bits_per_weight']:.2f}bpw "
          f"vs INT8-X {snr_i8:.1f} dB @ {p8['bits_per_weight']:.2f}bpw")


def test_bpw_and_compression():
    w = _heavy_tail((128, 512), seed=4)
    p = peakq_quantize(w)
    # synthetic heavy-tail is wider than real weights (T1 ~37% vs 46% real),
    # so bpw is higher than the ~9.45 measured on MiniCPM5 — see calibration
    assert p["bits_per_weight"] < 11.6, f"bpw too high: {p['bits_per_weight']:.2f}"
    assert p["bits_per_weight"] > 8.0
    assert p["compression_vs_bf16"] > 1.38
    print(f"[ok] bpw={p['bits_per_weight']:.2f} compression={p['compression_vs_bf16']:.2f}x")


def test_triton_matches_scatter():
    if not torch.cuda.is_available():
        print("[skip] no CUDA — triton match test skipped")
        return
    w = _heavy_tail((128, 512), seed=5)          # 65536 elems = 64 blocks
    p = peakq_quantize(w)
    ref = decode_peakq_scatter(p, device="cuda")
    tri = decode_peakq_triton(p, device="cuda")
    assert torch.equal(
        ref.view(torch.int16), tri.view(torch.int16)
    ), "triton decode != scatter decode"
    print("[ok] triton decode bit-identical to scatter decode")

    # non-multiple-of-block N + padded group path through the kernel too
    w2 = _heavy_tail((100, 130), seed=6)          # 13000 elems, pad=8
    p2 = peakq_quantize(w2)
    ref2 = decode_peakq_scatter(p2, device="cuda")
    tri2 = decode_peakq_triton(p2, device="cuda")
    assert torch.equal(ref2.view(torch.int16), tri2.view(torch.int16))
    print("[ok] triton decode bit-identical on padded/unaligned shape")


def test_linear_and_deploy():
    torch.manual_seed(7)
    w = _heavy_tail((96, 128), seed=7)
    lin = nn.Linear(128, 96, bias=True).to(torch.bfloat16)
    with torch.no_grad():
        lin.weight.copy_(w)
    packed = peakq_quantize(lin.weight.data)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ql = PeakQLinear(packed, bias=lin.bias.data, cache="full").to(dev)

    x = torch.randn(2, 5, 128, dtype=torch.bfloat16, device=dev)
    y_ref = F.linear(x, decode_peakq_scatter(packed, device=dev),
                     lin.bias.data.to(dev))
    y_got = ql(x)
    assert torch.allclose(y_ref.float(), y_got.float()), "PeakQLinear forward mismatch"
    print("[ok] PeakQLinear forward == F.linear(decoded)")

    class Toy(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(128, 256, bias=False)
            self.small = nn.Linear(4, 4, bias=False)   # below MIN_LINEAR_ELEMS
            self.fc2 = nn.Linear(256, 64, bias=False)

        def forward(self, x):
            return self.fc2(F.silu(self.fc1(x)))

    toy = Toy().to(torch.bfloat16).to(dev)
    x = torch.randn(3, 128, dtype=torch.bfloat16, device=dev)
    y0 = toy(x)
    stats = deploy_peakq(toy, verbose=False)
    assert stats["n_layers"] == 2, "small linear should be skipped"
    y1 = toy(x)
    rel = ((y0.float() - y1.float()).norm() / y0.float().norm()).item()
    assert rel < 0.01, f"deploy changed outputs too much: rel={rel:.4f}"
    print(f"[ok] deploy_peakq: {stats['n_layers']} layers, "
          f"bpw={stats['bits_per_weight']:.2f}, forward rel-err={rel:.5f}")


def test_fused_gemv():
    if not torch.cuda.is_available():
        print("[skip] no CUDA — fused gemv test skipped")
        return
    from ixrun.peakq import peakq_fused_gemv
    from ixrun.fused import compute_row_prefixes, _pick_split

    for shape, split_expected in [((128, 512), 1),      # tall/square
                                  ((96, 1024), 2),      # wide -> split-K
                                  ((32, 2048), 2)]:
        w = _heavy_tail(shape, seed=hash(shape) % 1000)
        p = peakq_quantize(w)
        w_ref = decode_peakq_scatter(p, device="cuda")   # exact reference

        out_f, in_f = shape
        x = torch.randn(in_f, dtype=torch.bfloat16, device="cuda")

        split = _pick_split(in_f) if out_f < in_f else 1
        assert split == split_expected, f"{shape}: split={split} != {split_expected}"
        if split > 1:
            chunk = in_f // split
            q1, q2 = compute_row_prefixes(p, chunk)
        else:
            chunk = 0
            q1, q2 = compute_row_prefixes(p)
        q1, q2 = q1.cuda(), q2.cuda()

        if chunk > 0:
            y32 = torch.zeros(out_f, dtype=torch.float32, device="cuda")
            y = peakq_fused_gemv(x, p["sign_packed"].cuda(), p["emax"].cuda(),
                                 p["bitmaps"][0].cuda(), p["bitmaps"][1].cuda(),
                                 p["streams"][0].cuda(), p["streams"][1].cuda(),
                                 p["streams"][2].cuda(), q1, q2,
                                 out_f, in_f, chunk=chunk, y32=y32).to(torch.bfloat16)
        else:
            y = peakq_fused_gemv(x, p["sign_packed"].cuda(), p["emax"].cuda(),
                                 p["bitmaps"][0].cuda(), p["bitmaps"][1].cuda(),
                                 p["streams"][0].cuda(), p["streams"][1].cuda(),
                                 p["streams"][2].cuda(), q1, q2,
                                 out_f, in_f)

        # reference: decode to bf16 then fp32 dot (same weights, torch order)
        y_ref = torch.mv(w_ref.float(), x.float())
        rel = ((y.float() - y_ref).norm() / y_ref.norm()).item()
        assert rel < 2e-2, f"fused gemv {shape}: rel-err {rel:.4f}"
        print(f"[ok] fused gemv {shape} (split={split}): rel-err={rel:.5f}")


def test_v2_rows_layout():
    """v2 row-restart streams decode bit-identical to v1 (same values, new layout)."""
    if not torch.cuda.is_available():
        print("[skip] no CUDA — v2 test skipped")
        return
    from ixrun.peakq import decode_peakq_triton

    for shape in [(128, 512), (100, 130), (96, 1024), (33, 640)]:
        w = _heavy_tail(shape, seed=abs(hash(shape)) % 1000)
        p1 = peakq_quantize(w, layout="global")
        p2 = peakq_quantize(w, layout="rows")

    for shape in [(128, 512), (100, 130), (96, 1024), (33, 640)]:
        w = _heavy_tail(shape, seed=abs(hash(shape)) % 1000)
        p1 = peakq_quantize(w, layout="global")
        p2 = peakq_quantize(w, layout="rows")

        # storage sanity only where v2 is meant to be used (real shapes);
        # degenerate tiny mats (e.g. 100x130) pay double-digit % in padding
        if shape[1] >= 512 and shape[0] >= 96:
            ratio = p2["bits_per_weight"] / p1["bits_per_weight"]
            assert ratio < 1.06, f"{shape}: v2/v1 bpw ratio {ratio:.3f}"

        d1 = decode_peakq_scatter(p1, device="cpu")
        d2s = decode_peakq_scatter(p2, device="cpu")
        assert torch.equal(d1.view(torch.int16), d2s.view(torch.int16)), \
            f"{shape}: v2 scatter != v1 decode"
        d2t = decode_peakq_triton(p2, device="cuda").cpu()
        assert torch.equal(d2s.view(torch.int16), d2t.view(torch.int16)), \
            f"{shape}: v2 triton != v2 scatter"

    # realistic-shape overhead check
    w = _heavy_tail((512, 1536), seed=42)
    pa = peakq_quantize(w, layout="global")
    pb = peakq_quantize(w, layout="rows")
    assert pb["bits_per_weight"] < pa["bits_per_weight"] * 1.02
    print(f"[ok] v2 rows layout: scatter==v1, triton==scatter across 4 shapes "
          f"(real-shape bpw {pa['bits_per_weight']:.2f} -> {pb['bits_per_weight']:.2f}, "
          f"+{(pb['bits_per_weight']/pa['bits_per_weight']-1)*100:.1f}%)")


def test_v2_gemv_multirow():
    if not torch.cuda.is_available():
        print("[skip] no CUDA — v2 gemv test skipped")
        return
    from ixrun.peakq import peakq_fused_gemv_v2

    for shape in [(128, 512), (64, 1024)]:
        out_f, in_f = shape
        w = _heavy_tail(shape, seed=abs(hash(shape + (7,))) % 1000)
        p = peakq_quantize(w, layout="rows")
        w_ref = decode_peakq_scatter(p, device="cuda")
        x = torch.randn(in_f, dtype=torch.bfloat16, device="cuda")
        y_ref = torch.mv(w_ref.float(), x.float())

        dev = "cuda"
        args = (p["sign_packed"].to(dev), p["emax"].to(dev),
                p["bitmaps"][0].to(dev), p["b2_rows"].to(dev),
                p["streams"][0].to(dev), p["streams"][1].to(dev),
                p["streams"][2].to(dev),
                p["t1_off"].to(dev), p["t2_bit_off"].to(dev),
                p["t3_bit_off"].to(dev), p["b2_bit_off"].to(dev))
        for r in (1, 2, 4):
            y = peakq_fused_gemv_v2(x, *args, out_f, in_f, r=r)
            rel = ((y.float() - y_ref).norm() / y_ref.norm()).item()
            assert rel < 2e-2, f"v2 gemv {shape} R={r}: rel-err {rel:.4f}"
        print(f"[ok] v2 gemv {shape}: R=1/2/4 all match (rel<2e-2)")


def test_lazy_deploy_and_hygiene():
    """deploy_peakq_lazy: works, strips CPU bodies, layers share one buffer."""
    from ixrun.peakq import deploy_peakq_lazy, PeakQLinear

    class Toy(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(128, 256, bias=False)
            self.fc2 = nn.Linear(256, 128, bias=False)

        def forward(self, x):
            return self.fc2(torch.nn.functional.silu(self.fc1(x)))

    torch.manual_seed(11)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if dev == "cpu":
        print("[skip] no CUDA — lazy deploy test skipped")
        return
    toy = Toy().to(torch.bfloat16).to(dev)
    x = torch.randn(3, 128, dtype=torch.bfloat16, device=dev)
    y0 = toy(x)                       # reference on the bf16 linears
    stats = deploy_peakq_lazy(toy, verbose=False)
    y1 = toy(x)
    rel = ((y0.float() - y1.float()).norm() / y0.float().norm()).item()
    assert rel < 0.01, f"lazy deploy rel-err {rel:.4f}"
    n_lazy = 0
    for _, mod in toy.named_modules():
        if isinstance(mod, PeakQLinear):
            assert not any(k in mod.packed for k in
                           ("streams", "bitmaps", "sign_packed", "emax", "b2_rows")), \
                "CPU bodies not stripped"
            n_lazy += 1
    assert n_lazy == 2
    print(f"[ok] deploy_peakq_lazy: {stats['n_layers']} layers, bodies stripped, "
          f"rel-err={rel:.5f}")


def main():
    print("Running PEAK-Q tests ...\n")
    test_tier1_bit_exact()
    test_zero_groups_preserved()
    test_snr_vs_int8x()
    test_bpw_and_compression()
    test_triton_matches_scatter()
    test_linear_and_deploy()
    test_fused_gemv()
    test_v2_rows_layout()
    test_v2_gemv_multirow()
    test_lazy_deploy_and_hygiene()
    print("\nAll PEAK-Q tests passed.")


if __name__ == "__main__":
    main()
