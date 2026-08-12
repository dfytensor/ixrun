"""Correctness tests for the IXRUN INT8-X engine.

Run:  python -m tests.test_core
"""
from __future__ import annotations
import torch

from ixrun.bitpack import pack_bits_stream, unpack_bits_stream
from ixrun.quantize import int8x_quantize, decode_to_weight, compute_positions
from ixrun.triton_kernels import decode_weight_triton, decode_weight_scatter
from ixrun.search import eval_scheme
from ixrun.config import DEFAULT_LEVELS, BIT_TO_THRESHOLD


def test_bitpack_roundtrip():
    torch.manual_seed(0)
    vals = torch.randint(0, 8, (1000,), dtype=torch.int32)
    packed = pack_bits_stream(vals, 3)
    decoded = unpack_bits_stream(packed, 1000, 3)
    assert torch.equal(vals, decoded), "3-bit roundtrip failed"
    print("[ok] bitpack 3-bit roundtrip")

    vals5 = torch.randint(0, 32, (777,), dtype=torch.int32)
    packed5 = pack_bits_stream(vals5, 5)
    decoded5 = unpack_bits_stream(packed5, 777, 5)
    assert torch.equal(vals5, decoded5), "5-bit roundtrip failed"
    print("[ok] bitpack 5-bit roundtrip")


def test_quantize_decode_lossless():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(42)
    # realistic-ish distribution (many small values)
    w = (torch.randn(128, 256) * 0.03).bfloat16().to(dev)
    p = int8x_quantize(w, DEFAULT_LEVELS)

    # int8 ground truth (the (3,5,8) is lossless on the int8)
    # match quantize's scale path: max_abs/127.0 (float32) then bfloat16
    scale_f32 = w.abs().max().clamp(min=1e-8) / 127.0
    i8_gt = (w.float() / scale_f32).round().clamp(-127, 127).to(torch.int8).reshape(-1)
    scale_bf16 = scale_f32.bfloat16()
    w_gt = (i8_gt.float() * scale_bf16.float()).to(torch.bfloat16).reshape(w.shape).to(dev)

    w_scatter = decode_weight_scatter(p, device=dev)
    assert torch.equal(w_scatter, w_gt), "scatter decode not lossless"
    print("[ok] scatter decode is lossless on int8")

    if torch.cuda.is_available():
        w_triton = decode_weight_triton(p, device=dev)
        assert torch.equal(w_triton, w_gt), "triton decode not lossless"
        print("[ok] triton decode is lossless on int8")

    # also check decode_to_weight helper
    w_helper = decode_to_weight(p, device=dev)
    assert torch.equal(w_helper, w_gt), "decode_to_weight not lossless"
    print("[ok] decode_to_weight is lossless")


def test_compression_bounds():
    # realistic LLM weight distribution: many small values, few large outliers
    torch.manual_seed(1)
    base = torch.randn(256, 512) * 0.005        # most values tiny (-> L1/L2)
    outliers = torch.zeros(256, 512)
    n_out = int(256 * 512 * 0.04)                # ~4% large outliers (-> L3)
    flat = outliers.view(-1)
    idx = torch.randperm(flat.numel())[:n_out]
    flat[idx] = torch.randn(n_out) * 0.05
    w = (base + outliers).bfloat16()
    p = int8x_quantize(w, DEFAULT_LEVELS)
    # with realistic distribution should get >2x compression
    assert p["compression_vs_bf16"] > 2.0, f"unexpected compression {p['compression_vs_bf16']}"
    assert p["bits_per_weight"] < 8.0, f"unexpected bpw {p['bits_per_weight']}"
    print(f"[ok] compression={p['compression_vs_bf16']:.2f}x bpw={p['bits_per_weight']:.2f}")


def test_positions_consistency():
    torch.manual_seed(2)
    w = (torch.randn(64, 128) * 0.04).bfloat16()
    p = int8x_quantize(w, (3, 5, 8))
    positions = compute_positions(p)
    # positions should partition all N elements with no overlap
    all_pos = torch.cat(positions)
    N = p["N"]
    assert all_pos.numel() == N, f"positions cover {all_pos.numel()} != {N}"
    assert all_pos.unique().numel() == N, "positions have duplicates"
    print("[ok] positions partition is correct")


def test_search_eval():
    # synthetic CDF: 60% |v|<=3, 35% 3<|v|<=15, 5% |v|>15
    cum = {t: min(1.0, t / 15 * 0.95) for t in range(128)}
    cum[3] = 0.60
    cum[15] = 0.95
    r = eval_scheme((3, 5, 8), cum)
    assert r is not None
    assert r["compression"] > 2.0
    print(f"[ok] eval_scheme (3,5,8): bpw={r['bpw']:.2f} comp={r['compression']:.2f}x")


def main():
    print("Running IXRUN core tests ...\n")
    test_bitpack_roundtrip()
    test_quantize_decode_lossless()
    test_compression_bounds()
    test_positions_consistency()
    test_search_eval()
    print("\nAll tests passed.")


if __name__ == "__main__":
    main()
