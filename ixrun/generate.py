"""Text generation on INT8-X deployed models.

Works with any HuggingFace causal LM (LlamaForCausalLM and friends) whose
Linear layers have been replaced by Int8XLinear. Supports greedy / sampling
and token-by-token streaming via TextIteratorStreamer.
"""
from __future__ import annotations
import torch

__all__ = ["generate_text", "stream_generate"]


@torch.no_grad()
def generate_text(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 128,
    temperature: float = 0.7,
    top_p: float = 0.9,
    do_sample: bool = True,
    repetition_penalty: float = 1.1,
    device=None,
) -> str:
    """Generate text and return the full decoded string."""
    if device is None:
        device = next(model.parameters()).device
    ids = tokenizer(prompt, return_tensors="pt").to(device)
    # Llama tokenizer may emit token_type_ids which generate() rejects
    ids.pop("token_type_ids", None)
    out = model.generate(
        **ids,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature if do_sample else 1.0,
        top_p=top_p if do_sample else 1.0,
        repetition_penalty=repetition_penalty,
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
    repetition_penalty: float = 1.1,
    device=None,
):
    """Yield decoded text chunks as they are produced (generator)."""
    from threading import Thread
    from transformers import TextIteratorStreamer

    if device is None:
        device = next(model.parameters()).device
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
        repetition_penalty=repetition_penalty,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        streamer=streamer,
    )
    thread = Thread(target=model.generate, kwargs=gen_kwargs)
    thread.start()
    for text in streamer:
        yield text
    thread.join()
