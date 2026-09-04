"""OpenAI-compatible HTTP API server for Int8XEngine.

Endpoints:
  GET  /v1/models
  POST /v1/chat/completions   (stream=true SSE / stream=false JSON)
  GET  /health

Works with any OpenAI-compatible client — including opencode, which can be
pointed at this server via an openai-compatible provider config.

Run:
  python -m ixrun.cli serve --model E:/models/Qwen3.8-27B --cache E:/models/qwen38_packed.pt
"""
from __future__ import annotations
import json
import threading
import time
import uuid

from fastapi import FastAPI
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field

from .engine import Int8XEngine

__all__ = ["create_app"]


def _split_think(text: str) -> tuple[str, str]:
    """Return (reasoning, answer). Handles both explicit <think>...</think>
    blocks and prompt-side auto-opened blocks (output starts mid-thinking)."""
    if "</think>" in text:
        head, tail = text.split("</think>", 1)
        reasoning = head
        if reasoning.lstrip().startswith("<think>"):
            reasoning = reasoning.split("<think>", 1)[1]
        return reasoning.strip(), tail.strip()
    return "", text.strip()


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str | None = None
    messages: list[ChatMessage]
    stream: bool = False
    max_tokens: int | None = Field(default=None, alias="max_tokens")
    max_completion_tokens: int | None = None
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    repetition_penalty: float | None = None
    frequency_penalty: float | None = None   # mapped to repetition (1+f)
    presence_penalty: float | None = None    # ignored (kept for compat)
    stream_options: dict | None = None


class _ThinkSplitter:
    """Streaming state machine that suppresses <think> reasoning.

    ``expect_think`` is set when the applied prompt auto-opened a think block
    (Qwen3-style: prompt ends with '<think>'), in which case ALL output before
    </think> is reasoning and must be buffered, not emitted.
    """

    def __init__(self, expect_think: bool = False):
        self.expect_think = expect_think
        self.mode = "think" if expect_think else None
        self.buf = ""

    def feed(self, chunk: str) -> tuple[str, str]:
        """Returns (reasoning_delta, answer_delta) to emit."""
        self.buf += chunk
        if self.mode is None:
            if self.buf.lstrip().startswith("<"):
                self.mode = "think"
            elif self.buf.strip() or len(self.buf) > 8:
                self.mode = "plain"
            else:
                return "", ""
        if self.mode == "plain":
            out, self.buf = self.buf, ""
            return "", out
        # think mode: buffer until </think>
        if "</think>" in self.buf:
            _, tail = self.buf.split("</think>", 1)
            self.buf = ""
            self.mode = "plain"
            return "", tail
        return "", ""


def create_app(eng: Int8XEngine, model_id: str, enable_thinking: bool = False,
               batched: bool = False, min_batch: int = 8,
               max_batch: int = 16) -> FastAPI:
    app = FastAPI(title="ixrun inference server")
    gen_lock = threading.Lock()  # single GPU: serialize generations
    _bgen = None
    if batched:
        from .batching import BatchedGreedyGenerator

        _bgen = BatchedGreedyGenerator(eng.model, eng.tokenizer,
                                       min_batch=min_batch,
                                       max_batch=max_batch)

    def _prompt(req: ChatCompletionRequest) -> tuple[str, bool]:
        """Returns (prompt, expect_think)."""
        msgs = [{"role": m.role, "content": m.content} for m in req.messages]
        try:
            try:
                p = eng.tokenizer.apply_chat_template(
                    msgs, tokenize=False, add_generation_prompt=True,
                    enable_thinking=enable_thinking,
                )
            except TypeError:
                p = eng.tokenizer.apply_chat_template(
                    msgs, tokenize=False, add_generation_prompt=True
                )
        except Exception:
            p = "".join(f"{m['role']}: {m['content']}\n" for m in msgs)
        return p, p.rstrip().endswith("<think>")

    def _gen_kwargs(req: ChatCompletionRequest) -> dict:
        max_new = req.max_completion_tokens or req.max_tokens or 512
        kw = dict(max_new_tokens=int(max_new))
        if req.temperature is not None:
            kw["temperature"] = req.temperature
            kw["do_sample"] = True
        else:
            kw["do_sample"] = False
        if req.top_p is not None:
            kw["top_p"] = req.top_p
        if req.top_k:
            kw["top_k"] = int(req.top_k)
        rep = req.repetition_penalty
        if rep is None and req.frequency_penalty is not None:
            rep = 1.0 + max(req.frequency_penalty, 0.0)
        if rep is not None and rep != 1.0:
            kw["repetition_penalty"] = rep
        return kw

    @app.get("/health")
    def health():
        return {"status": "ok", "model": model_id}

    @app.get("/v1/models")
    def list_models():
        return {
            "object": "list",
            "data": [
                {
                    "id": model_id,
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": "ixrun",
                }
            ],
        }

    @app.post("/v1/chat/completions")
    def chat_completions(req: ChatCompletionRequest):
        prompt, expect_think = _prompt(req)
        kw = _gen_kwargs(req)
        rid = "chatcmpl-" + uuid.uuid4().hex[:12]
        created = int(time.time())

        if not req.stream:
            with gen_lock:
                text = eng.generate(prompt, **kw)
            reasoning, answer = _split_think(text)
            msg = {"role": "assistant", "content": answer}
            if reasoning:
                msg["reasoning_content"] = reasoning
            return {
                "id": rid,
                "object": "chat.completion",
                "created": created,
                "model": req.model or model_id,
                "choices": [
                    {"index": 0, "finish_reason": "stop", "message": msg}
                ],
                "usage": {
                    "prompt_tokens": -1,
                    "completion_tokens": -1,
                    "total_tokens": -1,
                },
            }

        # ---- streaming (SSE) ----
        if _bgen is not None:
            # continuous-batching path: coalesces concurrent requests into
            # batch>=8 forwards — ~3-5x aggregate throughput; supports
            # temperature / top_p / top_k / repetition_penalty per row
            def sse_batched():
                rep = req.repetition_penalty
                if rep is None and req.frequency_penalty is not None:
                    rep = 1.0 + max(req.frequency_penalty, 0.0)
                req_obj, out_q = _bgen.submit(
                    prompt, kw.get("max_new_tokens", 512),
                    temperature=req.temperature or 1.0,
                    top_p=req.top_p or 1.0,
                    top_k=int(req.top_k or 0),
                    repetition_penalty=rep or 1.0,
                )
                splitter = _ThinkSplitter(expect_think=expect_think)
                try:
                    while True:
                        try:
                            chunk = out_q.get(timeout=0.25)
                        except Exception:
                            if req_obj.done.is_set():
                                break
                            continue
                        _, a_delta = splitter.feed(chunk)
                        if a_delta:
                            yield _sse_chunk(rid, created, req.model or model_id,
                                             content=a_delta)
                    yield _sse_chunk(rid, created, req.model or model_id,
                                     content=None, finish="stop")
                    yield "data: [DONE]\n\n"
                except GeneratorExit:
                    raise

            return StreamingResponse(sse_batched(), media_type="text/event-stream")

        def sse():
            with gen_lock:
                splitter = _ThinkSplitter(expect_think=expect_think)
                it = eng.stream(prompt, **kw)
                try:
                    for chunk in it:
                        r_delta, a_delta = splitter.feed(chunk)
                        if a_delta:
                            yield _sse_chunk(rid, created, req.model or model_id,
                                             content=a_delta)
                        # note: unclosed think block (max_tokens hit mid-
                        # reasoning) is intentionally not emitted as answer
                except GeneratorExit:
                    # client disconnected mid-stream: unwind WITHOUT yielding
                    # (yielding after GeneratorExit raises RuntimeError)
                    raise
                finally:
                    # deterministic cleanup even if we're being closed:
                    # it.close() cancels the generation thread (StoppingCriteria
                    # flag) and joins it, all while we still hold gen_lock —
                    # prevents a stray generation from corrupting the shared
                    # decode buffers of the next request
                    it.close()
                # normal completion only (client still connected)
                if req.stream_options and req.stream_options.get("include_usage"):
                    yield _sse_chunk(rid, created, req.model or model_id,
                                     content=None, finish="stop", usage=True)
                else:
                    yield _sse_chunk(rid, created, req.model or model_id,
                                     content=None, finish="stop")
                yield "data: [DONE]\n\n"

        return StreamingResponse(sse(), media_type="text/event-stream")

    def _sse_chunk(rid, created, model, content=None, finish=None, usage=False):
        delta = {}
        if content is not None:
            delta["content"] = content
        payload = {
            "id": rid,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [
                {"index": 0, "delta": delta, "finish_reason": finish}
            ],
        }
        if usage:
            payload["usage"] = {
                "prompt_tokens": -1, "completion_tokens": -1, "total_tokens": -1
            }
        return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

    return app


def serve(
    model_path: str,
    mode: str = "streaming",
    cache_path: str | None = None,
    level_bits=(3, 5, 8),
    host: str = "127.0.0.1",
    port: int = 8000,
    model_id: str | None = None,
    enable_thinking: bool = False,
    batched: bool = False,
    min_batch: int = 8,
    max_batch: int = 16,
    codec: str = "int8x",
):
    """Load engine + run uvicorn (blocking).

    mode='udcq-graph' + cache_path=<q38_blob.pt>: Qwen3.8-27B UDCQ 6bpw
    StaticCache + CUDA-Graph engine (~15 tok/s, 24GB card).
    codec='peakq': near-lossless 10.6bpw engine.
    """
    import uvicorn

    if mode == "udcq-graph":
        from .q38_graph import Q38GraphEngine

        eng = Q38GraphEngine.from_blob(cache_path, model_path)
    elif codec == "peakq":
        from .peakq_engine import PeakQEngine

        eng = PeakQEngine.from_pretrained(model_path, mode=mode)
    else:
        eng = Int8XEngine.from_pretrained(
            model_path, mode=mode, level_bits=tuple(level_bits),
            cache_path=cache_path, verbose=True,
        )
    if model_id is None:
        model_id = (model_path.rstrip("/\\").replace("\\", "/").split("/")[-1]
                    .lower().replace(".", "-"))
    app = create_app(eng, model_id, enable_thinking=enable_thinking,
                     batched=batched, min_batch=min_batch, max_batch=max_batch)
    print(f"[server] listening on http://{host}:{port}/v1 (model={model_id}"
          f"{', batched' if batched else ''})", flush=True)
    uvicorn.run(app, host=host, port=port, log_level="warning")
