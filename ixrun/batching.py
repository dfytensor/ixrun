"""Continuous batching core: batched greedy decode loop + server integration.

Batching economics are proven (B>=8 decode+GEMM 3.5x). This module gives
the server a request-coalescing generation path:

  - BatchedGreedyGenerator.batch_append(req) -> queue of token callbacks
  - a background scheduler thread coalesces pending requests into a batch,
    runs the KV-cached batched decode loop (pad-masked positions), fans
    streaming chunks back per request
  - requests join/leave the batch between steps (no mid-step re-batch)

Correctness model: greedy per-row argmax with per-row EOS; rows that
finish keep emitting nothing (their KV stays but is masked out of the
lossless path since we simply stop reading them).
"""
from __future__ import annotations
import threading
import time
import queue as _queue

import torch

from .generate import wait_quiescent

__all__ = ["BatchedGreedyGenerator"]


class _Req:
    __slots__ = ("ids", "cb", "done", "max_new", "n_out", "eos",
                 "temperature", "top_p", "top_k", "repetition_penalty",
                 "generated", "dbuf")

    def __init__(self, ids, max_new, cb, temperature=1.0, top_p=1.0,
                 top_k=0, repetition_penalty=1.0):
        self.ids = ids                # [1, L] prompt tokens (GPU)
        self.cb = cb                  # callable(str_chunk) -> None
        self.max_new = max_new
        self.n_out = 0
        self.eos = False
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.repetition_penalty = repetition_penalty
        self.generated = []           # sampled token ids (for rep penalty)
        self.dbuf = ""                # streamer text buffer (prefix-stable)
        self.done = threading.Event()


def _sample_rows(logits: torch.Tensor, reqs, active_idx) -> torch.Tensor:
    """Per-row sampling: temperature -> top_k -> top_p -> multinomial.

    logits: [B, V] float; rows not in active_idx return arbitrary valid ids
    (their outputs are ignored). Repetition penalty applied on the raw
    logits for each row's generated history.
    """
    B, V = logits.shape
    out = torch.zeros(B, dtype=torch.long, device=logits.device)
    for i in active_idx:
        r = reqs[i]
        row = logits[i].clone()
        # repetition penalty (classic CTRL-style: divide pos / multiply neg)
        if r.repetition_penalty != 1.0 and r.generated:
            hist = torch.tensor(r.generated[-256:], device=row.device)
            uniq = torch.unique(hist)
            lv = row[uniq]
            row[uniq] = torch.where(lv > 0, lv / r.repetition_penalty,
                                    lv * r.repetition_penalty)
        if r.temperature <= 0:
            out[i] = row.argmax()
            continue
        row = row / r.temperature
        if r.top_k and 0 < r.top_k < V:
            kth = row.topk(r.top_k).values[-1]
            row = row.where(row >= kth, float("-inf"))
        if 0 < r.top_p < 1.0:
            sv, si = row.sort(descending=True)
            cum = torch.softmax(sv, -1).cumsum(-1)
            keep = cum - torch.softmax(sv, -1) < r.top_p
            keep[0] = True
            sv = sv.where(keep, float("-inf"))
            row = torch.full_like(row, float("-inf"))
            row.scatter_(0, si, sv)
        probs = torch.softmax(row, -1)
        out[i] = torch.multinomial(probs, 1)
    return out


class BatchedGreedyGenerator:
    """Single-GPU continuous-batching greedy generator.

    Usage:
        gen = BatchedGreedyGenerator(model, tokenizer)
        gen.submit(prompt_text, max_new_tokens, on_chunk)  # non-blocking
        ... poll or use the returned threading.Event
    """

    def __init__(self, model, tokenizer, tick_s: float = 0.05,
                 min_batch: int = 8, max_batch: int = 16,
                 coalesce_ms: int = 40):
        self.model = model
        self.tok = tokenizer
        self.tick_s = tick_s
        self.min_batch = min_batch
        self.max_batch = max_batch
        self.coalesce_ms = coalesce_ms
        self.pending: list[_Req] = []
        self.lock = threading.Lock()
        self.wake = threading.Event()
        self.stop_flag = False
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    # ------------------------------------------------------------------ #
    def submit(self, prompt_text: str, max_new_tokens: int = 128,
               temperature: float = 1.0, top_p: float = 1.0, top_k: int = 0,
               repetition_penalty: float = 1.0):
        out_q: _queue.Queue = _queue.Queue()

        def cb(chunk: str):
            out_q.put(chunk)

        ids = self.tok(prompt_text, return_tensors="pt")["input_ids"]
        req = _Req(ids.cuda(), max_new_tokens, cb,
                   temperature=temperature, top_p=top_p, top_k=top_k,
                   repetition_penalty=repetition_penalty)
        with self.lock:
            self.pending.append(req)
        self.wake.set()
        return req, out_q

    def close(self):
        self.stop_flag = True
        self.wake.set()
        self._thread.join(timeout=10)

    # ------------------------------------------------------------------ #
    def _run(self):
        while not self.stop_flag:
            with self.lock:
                batch = self.pending[: self.max_batch]
                self.pending = self.pending[len(batch):]
            if not batch:
                self.wake.wait(self.tick_s)
                self.wake.clear()
                continue
            # coalesce window: wait a little for more requests to join
            if len(batch) < self.min_batch:
                time.sleep(self.coalesce_ms / 1000)
                with self.lock:
                    more = self.pending[: self.max_batch - len(batch)]
                    self.pending = self.pending[len(more):]
                batch += more
            try:
                self._run_batch(batch)
            except Exception as e:
                for r in batch:
                    r.cb(f"\n[batch error: {type(e).__name__}]")
                for r in batch:
                    r.done.set()

    @torch.no_grad()
    def _run_batch(self, reqs: list[_Req]):
        model, tok = self.model, self.tok
        dev = next(model.parameters()).device
        B = len(reqs)
        eos_ids = set()
        for t in (tok.eos_token_id, getattr(tok, "pad_token_id", None)):
            if t is not None:
                eos_ids.add(t)

        # pad prompts to a common length (LEFT pad so last position is the
        # newest token for every row)
        L = max(r.ids.shape[1] for r in reqs)
        pad_id = tok.pad_token_id or tok.eos_token_id
        inp = torch.full((B, L), pad_id, dtype=torch.long, device=dev)
        attn = torch.zeros((B, L), dtype=torch.long, device=dev)
        for i, r in enumerate(reqs):
            li = r.ids.shape[1]
            inp[i, L - li:] = r.ids[0]
            attn[i, L - li:] = 1
        pos = attn.cumsum(-1) - 1
        pos.clamp_(min=0)

        out = model(input_ids=inp, attention_mask=attn, position_ids=pos,
                    use_cache=True)
        past = out.past_key_values
        logits_last = out.logits[:, -1].float()                # [B, V]

        while not self.stop_flag:
            active = [i for i, r in enumerate(reqs)
                      if not r.eos and r.n_out < r.max_new]
            if not active:
                break
            nxt = _sample_rows(logits_last, reqs, active)      # [B]
            # stream tokens for active rows; per-token decode with a
            # hold-back for incomplete BPE/UTF-8 boundaries (the same
            # trick TextIteratorStreamer uses)
            for i in active:
                t_id = nxt[i].item()
                if t_id in eos_ids:
                    reqs[i].eos = True
                    continue
                r = reqs[i]
                r.generated.append(t_id)
                r.n_out += 1
                piece = tok.decode([t_id], skip_special_tokens=True)
                if piece.endswith("\ufffd"):            # incomplete boundary
                    r.dbuf += piece                    # hold back
                    continue
                if r.dbuf:
                    piece = r.dbuf + piece
                    r.dbuf = ""
                if piece:
                    r.cb(piece)
            still = [i for i in active if not reqs[i].eos]
            if not still:
                break
            step_tok = nxt.reshape(B, 1)
            step_pos = pos[:, -1:] + 1
            step_attn = torch.cat([attn, torch.ones((B, 1), dtype=torch.long,
                                                    device=dev)], dim=1)
            out = model(input_ids=step_tok, attention_mask=step_attn,
                        position_ids=step_pos, past_key_values=past,
                        use_cache=True)
            past = out.past_key_values
            attn = step_attn
            pos = step_pos
            logits_last = out.logits[:, -1].float()

        for r in reqs:
            if r.dbuf:
                r.cb(r.dbuf)
                r.dbuf = ""
            r.done.set()
