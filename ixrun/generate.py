"""Text generation on INT8-X deployed models.

Works with any HuggingFace causal LM (LlamaForCausalLM and friends) whose
Linear layers have been replaced by Int8XLinear. Supports greedy / sampling
and token-by-token streaming via TextIteratorStreamer.
"""
from __future__ import annotations
import threading

import torch

__all__ = ["generate_text", "stream_generate", "wait_quiescent"]

# Registry of live generation threads. The engine decodes weights into SHARED
# buffers, so two concurrent generate() calls corrupt each other 鈥?every new
# generation must wait for all previously started threads to be gone, no
# matter which code path (server abort, GC on the worker thread, ...) ended
# the previous one.
_ACTIVE: set = set()
_ACTIVE_LOCK = threading.Lock()


def wait_quiescent(poll_s: float = 2.0):
    """Block until no generation threads from earlier requests are alive."""
    while True:
        with _ACTIVE_LOCK:
            alive = [t for t in _ACTIVE if t.is_alive()]
            # prune finished entries (their cleanup may have run on the
            # worker thread itself via GC, which skips the join+discard)
            for t in list(_ACTIVE):
                if not t.is_alive():
                    _ACTIVE.discard(t)
        if not alive:
            return
        alive[0].join(timeout=poll_s)


@torch.no_grad()
def generate_text(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 128,
    temperature: float = 0.7,
    top_p: float = 0.9,
    do_sample: bool = True,
    top_k: int = 50,
    presence_penalty: float = 0.0,
    frequency_penalty: float = 0.0,
    repetition_penalty: float = 1.1,
    device=None,
) -> str:
    """Generate text and return the full decoded string."""
    if device is None:
        device = next(model.parameters()).device
    # never overlap with a live generation from an earlier request
    wait_quiescent()
    ids = tokenizer(prompt, return_tensors="pt").to(device)
    # Llama tokenizer may emit token_type_ids which generate() rejects
    ids.pop("token_type_ids", None)
    out = model.generate(
        **ids,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature if do_sample else 1.0,
        top_p=top_p if do_sample else 1.0,
        top_k=top_k if do_sample else 0,
        repetition_penalty=repetition_penalty,
        presence_penalty=presence_penalty,
        frequency_penalty=frequency_penalty,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
    )
    new_tokens = out[0][ids["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


@torch.no_grad()
def stream_generate(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 128,
    temperature: float = 0.7,
    top_p: float = 0.9,
    do_sample: bool = True,
    top_k: int = 50,
    presence_penalty: float = 0.0,
    frequency_penalty: float = 0.0,
    repetition_penalty: float = 1.1,
    device=None,
):
    """Yield decoded text chunks as they are produced (generator).

    Cancellation-safe: closing this generator stops the background generation
    thread at the next decode step (StoppingCriteria flag) and joins it 鈥?    required by the API server when a client disconnects mid-stream, since a
    stray generation would otherwise corrupt the shared decode buffers of a
    concurrently-started request.
    """
    from threading import Thread

    from transformers import StoppingCriteria, TextIteratorStreamer

    if device is None:
        device = next(model.parameters()).device

    # never overlap with a live generation from an earlier request
    wait_quiescent()

    class _Cancel(StoppingCriteria):
        """Flag checked every decode step; set on consumer close."""

        def __init__(self):
            self.stop = False

        def __call__(self, input_ids, scores, **kw):
            return self.stop

    stopper = _Cancel()

    streamer = TextIteratorStreamer(
        tokenizer,
        skip_prompt=True,
        skip_special_tokens=True,
    )
    ids = tokenizer(prompt, return_tensors="pt").to(device)
    ids.pop("token_type_ids", None)
    gen_kwargs = dict(
        **ids,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature if do_sample else 1.0,
        top_p=top_p if do_sample else 1.0,
        top_k=top_k if do_sample else 0,
        repetition_penalty=repetition_penalty,
        presence_penalty=presence_penalty,
        frequency_penalty=frequency_penalty,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        streamer=streamer,
        stopping_criteria=[stopper],
    )
    thread = Thread(target=model.generate, kwargs=gen_kwargs)
    with _ACTIVE_LOCK:
        _ACTIVE.add(thread)
    thread.start()
    try:
        for text in streamer:
            yield text
    finally:
        # consumer gone (GeneratorExit) or exhausted: stop the worker promptly
        stopper.stop = True
        cur = threading.current_thread()
        if thread is not cur:
            # normal deterministic close (server finally / exhausted loop):
            # join, THEN discard 鈥?no next generation can overlap.
            thread.join(timeout=60)
            with _ACTIVE_LOCK:
                _ACTIVE.discard(thread)
        # else: cleanup is running ON the generate thread (GC finalizer path).
        # Cannot self-join; keep the registration so the next request's
        # wait_quiescent() joins this thread before starting (it prunes the
        # entry once dead).
