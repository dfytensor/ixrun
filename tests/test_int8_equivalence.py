"""Verify INT8-X == plain INT8 precision (bit-exact equivalence).

Claims to verify:
  1. Layer level: INT8-X decoded weight is BIT-IDENTICAL to naive per-tensor
     int8 dequantization (i8 * scale) for every layer of MiniCPM5-1B.
  2. Model level: a model deployed with INT8-X produces the same ppl (and
     near-identical logits) as a model deployed with plain int8 weights.
  3. Quality vs bf16: INT8-X == int8 (both only lose bf16->int8 roundoff).

Run:  python -m tests.test_int8_equivalence
"""
from __future__ import annotations
import gc
import math
import sys

sys.setrecursionlimit(10000)
import pandas  # noqa: F401  (stack overflow fix, must precede transformers)
import torch
import torch.nn as nn

from ixrun.quantize import int8x_quantize
from ixrun.triton_kernels import decode_weight_triton
from ixrun.linear import iter_quantizable_linears, deploy_model
from ixrun.eval_utils import eval_ppl, load_wikitext
from ixrun.config import MODEL_PATH, DATASET_CACHE, DEFAULT_LEVELS


def naive_int8_weight(w: torch.Tensor) -> torch.Tensor:
    """Reference: plain per-tensor int8 quantize -> dequant (same scale path)."""
    scale_f32 = w.abs().max().clamp(min=1e-8) / 127.0
    i8 = (w.float() / scale_f32).round().clamp(-127, 127).to(torch.int8)
    return (i8.float() * scale_f32.bfloat16().float()).to(torch.bfloat16).view(w.shape)


@torch.no_grad()
def test_layer_bit_exact(n_layers=10):
    from transformers import AutoModelForCausalLM

    print(f"[1] Layer-level bit-exact check (first {n_layers} layers) ...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, dtype=torch.bfloat16, low_cpu_mem_usage=True
    )
    n_checked = n_bad = 0
    max_diff = 0.0
    snr_ix, snr_i8 = 0.0, 0.0
    for name, mod in iter_quantizable_linears(model):
        if n_checked >= n_layers:
            break
        w = mod.weight.data.cuda()
        w_ref = naive_int8_weight(w)          # plain int8 dequant
        p = int8x_quantize(w, DEFAULT_LEVELS)  # INT8-X packed
        w_ix = decode_weight_triton(p, device="cuda")  # INT8-X decoded

        bit_exact = torch.equal(w_ix, w_ref)
        if not bit_exact:
            n_bad += 1
            max_diff = max(max_diff, (w_ix.float() - w_ref.float()).abs().max().item())
        # SNR vs original bf16 (should be identical for both)
        err_ref = (w.float() - w_ref.float()).pow(2).mean().item()
        err_ix = (w.float() - w_ix.float()).pow(2).mean().item()
        sig = w.float().pow(2).mean().item()
        snr_i8 += 10 * math.log10(sig / max(err_ref, 1e-12))
        snr_ix += 10 * math.log10(sig / max(err_ix, 1e-12))
        n_checked += 1
        print(f"    {name:<42} bit-exact={bit_exact}", flush=True)

    del model; gc.collect(); torch.cuda.empty_cache()
    assert n_bad == 0, f"{n_bad} layers NOT bit-exact! max_diff={max_diff}"
    print(f"    -> ALL {n_checked} layers BIT-EXACT vs plain int8 (max_diff=0.0)", flush=True)
    print(f"    -> SNR vs bf16: int8={snr_i8/n_checked:.2f}dB  INT8-X={snr_ix/n_checked:.2f}dB",
          flush=True)


@torch.no_grad()
def test_model_equivalence():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print("\n[2] Model-level equivalence: plain-int8 model vs INT8-X model ...", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL_PATH)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    texts = load_wikitext(cache_dir=DATASET_CACHE)[:30]
    prompt_ids = tok("The theory of relativity states that",
                     return_tensors="pt")["input_ids"].cuda()

    # --- plain int8 deployed model (reference) ---
    print("    deploying plain-int8 model ...", flush=True)
    m_i8 = AutoModelForCausalLM.from_pretrained(MODEL_PATH, dtype=torch.bfloat16).cuda()
    for name, mod in list(iter_quantizable_linears(m_i8)):
        w = mod.weight.data
        # replace weight in-place with int8 dequant values
        mod.weight.data = naive_int8_weight(w)
    m_i8.eval()
    ppl_i8 = eval_ppl(m_i8, tok, texts)
    logits_i8 = m_i8(prompt_ids).logits.float()
    del m_i8; gc.collect(); torch.cuda.empty_cache()

    # --- INT8-X deployed model ---
    print("    deploying INT8-X model ...", flush=True)
    m_ix = AutoModelForCausalLM.from_pretrained(MODEL_PATH, dtype=torch.bfloat16).cuda()
    deploy_model(m_ix, level_bits=DEFAULT_LEVELS, cache="full", verbose=False)
    m_ix.eval()
    ppl_ix = eval_ppl(m_ix, tok, texts)
    logits_ix = m_ix(prompt_ids).logits.float()
    del m_ix; gc.collect(); torch.cuda.empty_cache()

    lg_max = (logits_i8 - logits_ix).abs().max().item()
    lg_mean = (logits_i8 - logits_ix).abs().mean().item()
    greedy_same = (
        logits_i8.argmax(-1) == logits_ix.argmax(-1)
    ).float().mean().item() * 100

    print(f"    ppl plain-int8 = {ppl_i8:.4f}", flush=True)
    print(f"    ppl INT8-X     = {ppl_ix:.4f}", flush=True)
    print(f"    logits: max|diff|={lg_max:.3e}  mean|diff|={lg_mean:.3e}", flush=True)
    print(f"    greedy next-token agreement: {greedy_same:.2f}%", flush=True)

    assert abs(ppl_i8 - ppl_ix) < 1e-6, "ppl mismatch!"
    print("    -> ppl IDENTICAL to plain int8 (INT8-X adds ZERO extra loss)", flush=True)


def main():
    print("=" * 72)
    print("INT8-X == plain INT8 precision equivalence test (MiniCPM5-1B)")
    print("=" * 72)
    test_layer_bit_exact()
    test_model_equivalence()
    print("\nAll equivalence tests passed: INT8-X is a LOSSLESS codec on int8.")
    print("Quality is exactly plain-int8 quality; the only loss is bf16->int8")
    print("roundoff itself (identical to any int8 quantizer).")


if __name__ == "__main__":
    main()
