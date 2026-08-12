"""Exhaustive search for the optimal nested-bitmap level combination.

Analyzes the int8 magnitude distribution of a model's Linear weights and ranks
every candidate level-bit scheme (2-level .. 5-level nested bitmaps) by
bits-per-weight (lower is better), so you can pick the scheme with the best
compression before committing to a full quantization run.
"""
from __future__ import annotations
from itertools import combinations

import torch
import torch.nn as nn

from .bitpack import pack_bits_stream  # noqa: F401  (re-export convenience)
from .config import BIT_TO_THRESHOLD, SKIP_PATTERNS, MIN_LINEAR_ELEMS

__all__ = ["analyze_distribution", "eval_scheme", "search_optimal_levels"]


@torch.no_grad()
def analyze_distribution(model: nn.Module) -> dict:
    """Collect the int8 magnitude CDF across all quantizable Linear layers.

    Returns a dict with:
      cum : {threshold: fraction of |v| <= threshold}
      total_elems : int
      hist : raw histogram of |v| for 0..127
    """
    chunks = []
    for name, mod in model.named_modules():
        if not isinstance(mod, nn.Linear):
            continue
        if mod.weight.numel() < MIN_LINEAR_ELEMS:
            continue
        if any(s in name for s in SKIP_PATTERNS):
            continue
        w = mod.weight.data
        s = w.abs().max().clamp(min=1e-8) / 127.0
        i8 = (w.float() / s).round().clamp(-127, 127).to(torch.int8).reshape(-1)
        chunks.append(i8.abs())
    abs_v = torch.cat(chunks)
    N = abs_v.numel()
    hist = torch.zeros(128, dtype=torch.int64)
    for t in range(128):
        hist[t] = (abs_v == t).sum().item()
    cum_counts = hist.cumsum(0)
    cum = {t: cum_counts[t].item() / N for t in range(128)}
    return {"cum": cum, "total_elems": N, "hist": hist.tolist()}


def eval_scheme(level_bits: tuple, cum: dict) -> dict:
    """Estimate bits-per-weight for a given level scheme using the magnitude CDF.

    The nested-bitmap flag overhead is:
      flag = 1 + (1-p1) + (1-p1-p2) + ... + pN
    where p_i is the fraction of weights in level i.
    """
    thresholds = [BIT_TO_THRESHOLD[b] for b in level_bits]
    if thresholds[-1] < 127:
        return None  # scheme does not cover the full range
    # fractions per level
    ps = []
    prev_t = -1
    for t in thresholds:
        if prev_t < 0:
            p = cum[t]
        else:
            p = cum[t] - cum[prev_t]
        ps.append(p)
        prev_t = t
    # the last level should capture everything beyond the second-to-last threshold
    ps[-1] = 1.0 - sum(ps[:-1])

    # data bits per weight
    data = sum(p * b for p, b in zip(ps, level_bits))
    # flag bits per weight (nested bitmaps)
    flag = 0.0
    remaining = 1.0
    for i in range(len(level_bits) - 1):
        flag += remaining  # B_i has one bit per element still unassigned
        remaining -= ps[i]
    # final bitmap not needed; remaining elements all go to last level
    bpw = flag + data
    return {
        "level_bits": level_bits,
        "fractions": ps,
        "flag_bpw": flag,
        "data_bpw": data,
        "bpw": bpw,
        "compression": 16.0 / bpw,
    }


def search_optimal_levels(
    model: nn.Module,
    max_levels: int = 5,
    topk: int = 10,
    min_bits: int = 2,
) -> list:
    """Search all nested-bitmap schemes (2..max_levels levels) and rank by bpw.

    Returns a sorted list of scheme dicts (best first).
    """
    dist = analyze_distribution(model)
    cum = dist["cum"]
    results = []
    for n_levels in range(2, max_levels + 1):
        for combo in combinations(range(min_bits, 9), n_levels):
            if combo[-1] != 8:
                continue  # last level must be 8-bit
            r = eval_scheme(combo, cum)
            if r is not None:
                r["n_levels"] = n_levels
                results.append(r)
    results.sort(key=lambda r: r["bpw"])
    return results[:topk]
