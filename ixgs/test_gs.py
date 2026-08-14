"""Correctness tests for ixgs (Group-Scale INT8-X).

Run:  python -m ixgs.test_gs
"""
import math
import torch

from .quantize import (
    int8gs_quantize,
    decode_weight_scatter,
    per_tensor_int8_reference,
)
from .kernels import decode_weight_triton, has_triton


def _snr(a: torch.Tensor, b: torch.Tensor) -> float:
    mse = ((a.float() - b.float()) ** 2).mean().item()
    if mse == 0:
        return float("inf")
    return 10 * math.log10((b.float() ** 2).mean().item() / mse)


def _heavy_tail(shape: torch.Size) -> torch.Tensor:
    """Realistic LLM-like weight: gaussian bulk + sparse large outliers.

    The outliers inflate the per-tensor max, which is exactly the regime where
    group-scale beats per-tensor. On pure gaussian data (no outliers) the
    per-tensor /127 scale is finer and wins — that is expected, not a bug.
    """
    w = torch.randn(shape) * 0.02
    outlier = (torch.rand(shape) < 0.0005).float()
    w = w + outlier * torch.randn(shape) * 1.5
    return w.to(torch.bfloat16)


def test_roundtrip_lossless():
    torch.manual_seed(0)
    for shape in [(512, 256), (1000, 640), (21504, 5376)]:
        w = _heavy_tail(shape)
        packed = int8gs_quantize(w)
        w_rec = decode_weight_scatter(packed)
        # encoding layer lossless <=> reconstructed bf16 == reference (int * scale -> bf16)
        gs = packed["group_scales"].float().to(w_rec.device)
        scale_per = gs.repeat_interleave(packed["group_size"])[: packed["N"]]
        i_true = (w.float().reshape(-1).to(w_rec.device) / scale_per).round().clamp(-127, 128)
        w_ref = (i_true * scale_per).to(torch.bfloat16).view_as(w_rec)
        # numeric equality (encoding layer lossless); +0/-0 sign bits may differ
        # (round(-0.4)=-0.0 in float, int path returns +0.0) — value-identical
        numeq = (w_rec.float() == w_ref.float()) | (w_rec.float() == 0) & (w_ref.float() == 0)
        bitexact = numeq.float().mean().item()
        assert bitexact == 1.0, f"{shape}: lossless {bitexact*100:.4f}% < 100%"
        snr = _snr(w_rec, w.to(w_rec.device))
        ref = per_tensor_int8_reference(w)
        snr_ref = _snr(ref.to(w_rec.device), w.to(w_rec.device))
        print(f"  {shape}: lossless=100% bpw={packed['bits_per_weight']:.2f} "
              f"SNR={snr:.1f}dB (per-tensor {snr_ref:.1f}dB)")
        assert snr > snr_ref, "group-scale must beat per-tensor on heavy-tailed weights"


def test_triton_matches_scatter():
    if not has_triton():
        print("  [skip] no triton/cuda")
        return
    torch.manual_seed(1)
    for shape in [(512, 256), (5376, 4096)]:
        w = (torch.randn(shape) * 0.05).to(torch.bfloat16)
        packed = int8gs_quantize(w)
        w_scatter = decode_weight_scatter(packed)
        w_triton = decode_weight_triton(packed)
        diff = (w_scatter.float() - w_triton.float()).abs().max().item()
        assert diff == 0.0, f"{shape}: triton vs scatter max diff {diff}"
        print(f"  {shape}: triton == scatter (bit-exact)")


def test_forward_equivalence():
    torch.manual_seed(2)
    w = _heavy_tail((256, 128))
    x = torch.randn(8, 128, dtype=torch.bfloat16)
    packed = int8gs_quantize(w)
    w_rec = decode_weight_scatter(packed)
    x = x.to(w_rec.device)
    y_ref = torch.nn.functional.linear(x, w.to(x.device))
    y_rec = torch.nn.functional.linear(x, w_rec.to(x.dtype))
    corr = torch.corrcoef(torch.stack([y_ref.float().flatten(), y_rec.float().flatten()]))[0, 1]
    # per-tensor baseline on the same extreme-outlier layer
    ref = per_tensor_int8_reference(w).to(x.device)
    y_pt = torch.nn.functional.linear(x, ref)
    corr_pt = torch.corrcoef(torch.stack([y_ref.float().flatten(), y_pt.float().flatten()]))[0, 1]
    assert corr > 0.99, f"output correlation {corr}"
    assert corr > corr_pt, "group-scale forward must beat per-tensor"
    print(f"  forward corr={corr:.6f} (per-tensor {corr_pt:.6f})")


if __name__ == "__main__":
    print("== test_roundtrip_lossless ==")
    test_roundtrip_lossless()
    print("== test_triton_matches_scatter ==")
    test_triton_matches_scatter()
    print("== test_forward_equivalence ==")
    test_forward_equivalence()
    print("ALL PASSED")
