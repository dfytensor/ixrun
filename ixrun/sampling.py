# -*- coding: utf-8 -*-
"""CPU-side token sampling for the custom-loop engines (Q38/StepGraph).

Applies, in order: repetition penalty (HF semantics) -> top-k truncation
-> top-p (nucleus) truncation -> temperature softmax -> multinomial.
When no sampling knob is active the caller keeps the GPU argmax path.

The 27B vocab is 248K: the logits slice (~1MB fp32) is copied CPU-side
once per token (~1ms) — negligible against the ~30ms decode step.
"""
from __future__ import annotations

import torch

__all__ = ["needs_sampling", "sample_token"]


def needs_sampling(do_sample: bool, temperature: float, top_p: float,
                   top_k: int, repetition_penalty: float) -> bool:
    return bool(do_sample and temperature > 0) or top_p < 1.0 \
        or top_k > 0 or repetition_penalty != 1.0


def gpu_sample_token(logits1, temperature: float = 0.0, top_p: float = 1.0,
                     top_k: int = 0) -> torch.Tensor:
    """Sample one token from a GPU logits row [V] (spec-verify stage).

    top-p/top-k are applied ONLY here (verification); the draft stays a
    full-vocab argmax. temperature scales before the softmax. Returns a
    0-dim int64 GPU tensor.
    """
    lg = logits1.float()
    if top_k and top_k > 0:
        k = min(top_k, lg.numel())
        vals = torch.topk(lg, k).values
        lg = torch.where(lg >= vals.min(), lg,
                         torch.tensor(float('-inf'), device=lg.device))
    if top_p < 1.0:
        sorted_l, order = torch.sort(lg, descending=True)
        cum = torch.cumsum(torch.softmax(sorted_l, dim=0), dim=0)
        keep = cum <= top_p
        keep[0] = True
        mask = torch.zeros_like(lg, dtype=torch.bool)
        mask[order[keep]] = True
        lg = torch.where(mask, lg,
                         torch.tensor(float('-inf'), device=lg.device))
    if temperature and temperature > 0:
        lg = lg / temperature
    return torch.multinomial(torch.softmax(lg, dim=0), 1).squeeze()


def sample_token(logits_v, temperature: float = 0.0, top_p: float = 1.0,
                 top_k: int = 0, repetition_penalty: float = 1.0,
                 past_ids=()) -> int:
    """logits_v: 1-D CPU tensor (fp32/bf16) over the vocab.

    Returns the sampled token id (or argmax when no knob is active).
    """
    logits = logits_v.float().clone()
    if repetition_penalty != 1.0 and past_ids:
        penalize = set(int(t) for t in past_ids)
        penalize.discard(-1)
        if penalize:
            idx = torch.tensor(sorted(penalize), dtype=torch.long)
            g = logits[idx]
            # HF semantics: score / penalty when positive else score * p
            logits[idx] = torch.where(
                g > 0, g / repetition_penalty, g * repetition_penalty)
    if top_k and top_k > 0:
        k = min(top_k, logits.numel())
        thr = torch.topk(logits, k).values.min()
        logits = torch.where(logits >= thr, logits,
                             torch.tensor(float('-inf')))
    if top_p < 1.0:
        sorted_l, order = torch.sort(logits, descending=True)
        cum = torch.cumsum(torch.softmax(sorted_l, dim=0), dim=0)
        keep = cum <= top_p
        # always keep the top token (nucleus rule)
        keep[0] = True
        mask = torch.zeros_like(logits, dtype=torch.bool)
        mask[order[keep]] = True
        logits = torch.where(mask, logits,
                             torch.tensor(float('-inf')))
    if temperature and temperature > 0:
        logits = logits / temperature
    probs = torch.softmax(logits, dim=0)
    if not torch.isfinite(probs).all() or probs.sum() <= 0:
        return int(torch.argmax(logits_v.float()))
    return int(torch.multinomial(probs, 1).item())
