"""Benchmark (forward speed + GPU memory) and perplexity evaluation helpers."""
from __future__ import annotations
import math
import time

import torch
import torch.nn.functional as F

__all__ = ["bench_forward", "eval_ppl", "load_wikitext", "format_bench"]


@torch.no_grad()
def bench_forward(model, tokenizer, prompt="Hello " * 20, warmup=5, n_runs=20):
    """Return (ms_per_forward, peak_gpu_GB)."""
    device = next(model.parameters()).device
    ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)
    model.eval()
    torch.cuda.reset_peak_memory_stats()
    for _ in range(warmup):
        model(ids)
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(n_runs):
        model(ids)
    torch.cuda.synchronize()
    ms = (time.time() - t0) / n_runs * 1000
    mem = torch.cuda.max_memory_allocated() / 1e9
    return ms, mem


def load_wikitext(cache_dir=None, split="test"):
    """Load wikitext-2 test texts (cached, offline)."""
    import os

    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    from datasets import load_dataset

    ds = load_dataset("wikitext", "wikitext-2-raw-v1", cache_dir=cache_dir)
    return [d["text"] for d in ds[split] if d["text"].strip() and len(d["text"]) > 100]


@torch.no_grad()
def eval_ppl(model, tokenizer, texts, max_samples=30, max_length=1024, device=None):
    """Compute perplexity on a list of texts."""
    if device is None:
        device = next(model.parameters()).device
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    for i, t in enumerate(texts):
        if i >= max_samples:
            break
        ids = tokenizer(t, return_tensors="pt", truncation=True, max_length=max_length)[
            "input_ids"
        ].to(device)
        if ids.size(1) < 2:
            continue
        out = model(ids)
        logits = out.logits[:, :-1].reshape(-1, out.logits.size(-1))
        total_loss += F.cross_entropy(
            logits, ids[:, 1:].reshape(-1), reduction="sum"
        ).item()
        total_tokens += ids[:, 1:].numel()
    return math.exp(total_loss / max(total_tokens, 1))


def format_bench(rows, headers):
    """Pretty-print a benchmark table."""
    widths = [max(len(str(h)), *(len(str(r[j])) for r in rows)) for j, h in enumerate(headers)]
    sep = "  ".join("-" * w for w in widths)
    lines = ["  ".join(h.ljust(widths[j]) for j, h in enumerate(headers)), sep]
    for r in rows:
        lines.append("  ".join(str(c).ljust(widths[j]) for j, c in enumerate(r)))
    return "\n".join(lines)
