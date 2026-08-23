# -*- coding: utf-8 -*-
"""Correctness tests for UDCQ (ixrun.udcq).

Run:  python -m tests.test_udcq
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

from ixrun.udcq import (
    udcq_fit_codebook,
    udcq_quantize,
    decode_udcq_triton,
    _decode_udcq_ref,
    UdcqLinear,
    deploy_udcq,
    udcq_snr,
    UDCQ_G,
    UDCQ_NLEV,
)


def _heavy_tail(shape, seed=0):
    torch.manual_seed(seed)
    base = torch.randn(shape) * 0.02
    out = torch.zeros(shape)
    n = int(out.numel() * 0.03)
    flat = out.view(-1)
    idx = torch.randperm(out.numel())[:n]
    flat[idx] = torch.randn(n) * 0.15
    return (base + out).bfloat16()


def test_codebook_and_snr():
    w = _heavy_tail((128, 512), seed=1)
    cb = udcq_fit_codebook(w, nlev=UDCQ_NLEV, g=UDCQ_G)
    assert cb.numel() == UDCQ_NLEV
    assert (cb >= 0).all() and (cb <= 1).all()
    assert (cb[1:] > cb[:-1]).all(), "codebook must be strictly increasing"
    p = udcq_quantize(w, cb)
    # decode-path SNR (scale f16 rounding included)
    wq = _decode_udcq_ref(p)
    s = udcq_snr(w, wq)
    assert s > 20.0, f"4-bit decode SNR too low: {s:.1f} dB"
    # bpw accounting: 4 (idx) + ~1 (sign packed) + 1 (scale f16/16) ~= 6.0
    assert 5.8 < p["bits_per_weight"] < 6.1
    print(f"[ok] codebook + quantize: decode SNR {s:.1f} dB @ "
          f"{p['bits_per_weight']:.2f} bpw")


def test_triton_matches_ref():
    if not torch.cuda.is_available():
        print("[skip] no CUDA")
        return
    w = _heavy_tail((128, 512), seed=2)          # N % 16 == 0
    w2 = _heavy_tail((100, 130), seed=3)         # N % 16 != 0 (pad path)
    cb = udcq_fit_codebook(w)
    for wt in (w, w2):
        p = udcq_quantize(wt, cb)
        ref = _decode_udcq_ref(p, device="cuda").float()
        tri = decode_udcq_triton(p, device="cuda").float()
        n_valid = p["N"]
        diff = (ref.reshape(-1)[:n_valid] - tri.reshape(-1)[:n_valid]).abs()
        # both decode the same math (f16 scale, f16 CB); allow 1-ulp bf16
        rel = diff.max().item()
        assert rel <= 2e-3, f"triton != ref (max abs {rel})"
    print("[ok] triton LUT decode matches reference (aligned + padded)")


def test_linear_forward():
    torch.manual_seed(4)
    w = _heavy_tail((96, 128), seed=4)
    lin = nn.Linear(128, 96, bias=True).to(torch.bfloat16)
    with torch.no_grad():
        lin.weight.copy_(w)
    cb = udcq_fit_codebook(w)
    p = udcq_quantize(lin.weight.data, cb)
    ul = UdcqLinear(p, bias=lin.bias.data).cuda()
    x = torch.randn(2, 5, 128, dtype=torch.bfloat16, device="cuda")
    y_ref = F.linear(x, _decode_udcq_ref(p, device="cuda"), lin.bias.data.cuda())
    y = ul(x)
    assert torch.allclose(y_ref.float(), y.float(), atol=1e-2)
    print("[ok] UdcqLinear forward == F.linear(decoded)")


def test_deploy():
    class Toy(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(128, 256, bias=False)
            self.small = nn.Linear(4, 4, bias=False)
            self.fc2 = nn.Linear(256, 64, bias=False)

        def forward(self, x):
            return self.fc2(F.silu(self.fc1(x)))

    torch.manual_seed(5)
    toy = Toy().to(torch.bfloat16)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    toy = toy.to(dev)
    x = torch.randn(3, 128, dtype=torch.bfloat16, device=dev)
    y0 = toy(x)
    stats = deploy_udcq(toy, verbose=False)
    assert stats["n_layers"] == 2
    y1 = toy(x)
    rel = ((y0.float() - y1.float()).norm() / y0.float().norm()).item()
    assert rel < 0.05, f"deploy rel-err {rel:.4f}"
    print(f"[ok] deploy_udcq: {stats['n_layers']} layers "
          f"bpw={stats['bits_per_weight']:.2f}, forward rel-err={rel:.4f}")


def test_stream_mode():
    """cache='stream': shared buffer decode == full-mode decode."""
    if not torch.cuda.is_available():
        print("[skip] no CUDA")
        return
    torch.manual_seed(6)
    w1 = _heavy_tail((96, 128), seed=6)
    w2 = _heavy_tail((64, 256), seed=7)          # different (smaller) layer
    cb = udcq_fit_codebook(w1)
    p1, p2 = udcq_quantize(w1, cb), udcq_quantize(w2, cb)
    l1 = UdcqLinear(p1, cache="full").cuda()
    l2 = UdcqLinear(p2, cache="stream").cuda()
    x1 = torch.randn(2, 4, 128, dtype=torch.bfloat16, device="cuda")
    x2 = torch.randn(2, 4, 256, dtype=torch.bfloat16, device="cuda")
    y1 = l1(x) if False else l1(x1)
    y2 = l2(x2)
    # stream decode of layer2 == its full decode
    l2_full = UdcqLinear(p2, cache="full").cuda()
    y2_ref = l2_full(x2)
    assert torch.allclose(y2.float(), y2_ref.float(), atol=1e-2), \
        "stream decode != full decode"
    # layer1 still correct AFTER layer2 used the (reused) shared buffer
    y1b = l1(x1)
    assert torch.allclose(y1.float(), y1b.float(), atol=1e-6)
    print("[ok] stream mode: shared-buffer decode correct + reusable")


def main():
    print("Running UDCQ tests ...\n")
    test_codebook_and_snr()
    test_triton_matches_ref()
    test_linear_forward()
    test_deploy()
    test_stream_mode()
    print("\nAll UDCQ tests passed.")


if __name__ == "__main__":
    main()
