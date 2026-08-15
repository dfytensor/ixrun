"""Interactive chat REPL for any Int8XEngine-deployed model.

Usage:
    python -m ixrun.cli chat --model E:\\models\\Qwen3.8-27B
    python -m ixrun.cli chat --model <minicpm5-path>
Commands inside the chat: /exit /new /len N /think on|off
"""
from __future__ import annotations

__all__ = ["chat_repl"]


def _apply_template(tokenizer, messages, enable_thinking: bool | None):
    kwargs = {}
    if enable_thinking is not None:
        kwargs["enable_thinking"] = enable_thinking
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, **kwargs
        )
    except (TypeError, Exception) as e:
        # template doesn't accept enable_thinking (e.g. MiniCPM5)
        if kwargs and ("enable_thinking" in str(e) or isinstance(e, TypeError)):
            return tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        raise


def _strip_think(text: str) -> str:
    """Drop a leading <think>...</think> block for display (kept in history)."""
    if "</think>" in text:
        return text.split("</think>", 1)[1].lstrip("\n")
    return text


def chat_repl(eng, max_new_tokens=256, temperature=0.7, do_sample=True):
    tok = eng.tokenizer
    history = []
    enable_thinking = None  # model default
    max_new = max_new_tokens
    print("ixrun chat — /exit quit, /new reset, /len N tokens, /think on|off\n")
    while True:
        try:
            user = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user:
            continue
        if user.startswith("/"):
            cmd = user.split()[0].lower()
            if cmd in ("/exit", "/quit", "/q"):
                break
            if cmd == "/new":
                history = []
                print("(history cleared)\n")
                continue
            if cmd == "/len":
                try:
                    max_new = int(user.split()[1])
                    print(f"(max_new_tokens={max_new})\n")
                except (IndexError, ValueError):
                    print("usage: /len N\n")
                continue
            if cmd == "/think":
                arg = user.split()[1].lower() if len(user.split()) > 1 else ""
                enable_thinking = {"on": True, "off": False}.get(arg)
                print(f"(enable_thinking={enable_thinking})\n")
                continue
            print(f"unknown command {cmd}\n")
            continue

        history.append({"role": "user", "content": user})
        prompt = _apply_template(tok, history, enable_thinking)
        print("ai  > ", end="", flush=True)
        full = ""
        mode = None  # None=undecided, "think"=inside <think>, "plain"=visible
        try:
            for chunk in eng.stream(
                prompt,
                max_new_tokens=max_new,
                temperature=temperature,
                do_sample=do_sample,
            ):
                full += chunk
                if mode is None:
                    mode = "think" if full.lstrip().startswith("<") else "plain"
                if mode == "plain":
                    print(chunk, end="", flush=True)
                elif "</think>" in full:
                    mode = "plain"
                    part = chunk.split("</think>", 1)[1]
                    if part:
                        print(part, end="", flush=True)
        except KeyboardInterrupt:
            print("\n(interrupted)")
        print("\n")
        history.append({"role": "assistant", "content": full})
    print("bye.")
