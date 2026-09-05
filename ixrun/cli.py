"""IXRUN command-line interface.

Usage examples:
  python -m ixrun.cli search        -- analyze optimal level scheme
  python -m ixrun.cli generate "Hello, my name is" --max-new-tokens 64
  python -m ixrun.cli bench         -- full bf16 vs INT8-X benchmark
  python -m ixrun.cli bench --mode streaming
"""
from __future__ import annotations
import argparse
import gc
import sys
import torch

from .config import MODEL_PATH, DATASET_CACHE, DEFAULT_LEVELS


def _cmd_search(args):
    from transformers import AutoModelForCausalLM
    from .search import search_optimal_levels

    print(f"[search] loading {args.model} ...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True
    )
    results = search_optimal_levels(model, max_levels=args.max_levels, topk=args.topk)
    print(f"\nTop {len(results)} level schemes (lower bpw = better compression):\n")
    print(f"{'Levels':<16}{'bpw':>8}{'flag':>8}{'data':>8}{'comp':>8}{'fractions'}")
    print("-" * 70)
    for r in results:
        frac_str = " ".join(f"{p*100:.0f}%" for p in r["fractions"])
        marker = "  <-- default" if tuple(r["level_bits"]) == tuple(DEFAULT_LEVELS) else ""
        print(
            f"{str(r['level_bits']):<16}{r['bpw']:>8.2f}{r['flag_bpw']:>8.2f}"
            f"{r['data_bpw']:>8.2f}{r['compression']:>7.2f}x  [{frac_str}]{marker}"
        )


def _build_engine(args):
    """Engine factory shared by generate/chat.

    --mode step-graph  whole-step CUDA-Graph decode (Llama-arch, ~50 tok/s
                       on MiniCPM5); --codec bf16|int8x|udcq|udcq-stream
    --codec peakq     PeakQEngine (cached/streaming, 10.6bpw 54dB)
    --mode udcq-graph + --cache <blob>  Q38GraphEngine (27B 6bpw, 15 tok/s)
    """
    if getattr(args, "mode", None) == "udcq-graph":
        from .q38_graph import Q38GraphEngine

        if not args.cache:
            raise SystemExit("udcq-graph requires --cache <q38_blob.pt>")
        return Q38GraphEngine.from_blob(args.cache, args.model)
    if getattr(args, "mode", None) == "udcq-spec":
        from .q38_spec import Q38SpecEngine

        if not args.cache:
            raise SystemExit("udcq-spec requires --cache <q38_blob.pt>")
        return Q38SpecEngine.from_blob(args.cache, args.model)
    if getattr(args, "mode", None) == "step-graph":
        from .step_graph import StepGraphEngine

        return StepGraphEngine.from_pretrained(
            args.model, codec=args.codec, verbose=True)
    if getattr(args, "codec", "int8x") == "peakq":
        from .peakq_engine import PeakQEngine

        return PeakQEngine.from_pretrained(args.model, mode=args.mode)
    from .engine import Int8XEngine

    return Int8XEngine.from_pretrained(
        args.model, mode=args.mode, level_bits=tuple(args.levels),
        cache_path=args.cache, verbose=True,
    )


def _cmd_generate(args):
    eng = _build_engine(args)
    print(f"\n[prompt] {args.prompt}\n", flush=True)
    if args.stream:
        for chunk in eng.stream(args.prompt, max_new_tokens=args.max_new_tokens,
                                temperature=args.temperature, do_sample=args.do_sample):
            sys.stdout.write(chunk)
            sys.stdout.flush()
        print()
    else:
        out = eng.generate(args.prompt, max_new_tokens=args.max_new_tokens,
                           temperature=args.temperature, do_sample=args.do_sample)
        print(out)


def _cmd_chat(args):
    from .chat import chat_repl

    eng = _build_engine(args)
    chat_repl(eng, max_new_tokens=args.max_new_tokens,
              temperature=args.temperature, do_sample=args.do_sample)


def _cmd_bench(args):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from .linear import deploy_model
    from .eval_utils import bench_forward, eval_ppl, load_wikitext

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    texts = load_wikitext(cache_dir=DATASET_CACHE)

    rows = []
    # bf16 baseline
    print("[bench] bf16 baseline ...", flush=True)
    gc.collect()
    torch.cuda.reset_peak_memory_stats()
    m = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16).cuda()
    ms_b, mem_b = bench_forward(m, tok)
    ppl_b = eval_ppl(m, tok, texts)
    wt_mb = sum(p.numel() * 2 for p in m.parameters() if p.dtype == torch.bfloat16) / 1e6
    rows.append(("bf16", f"{ms_b:.0f}ms", f"{mem_b:.1f}GB", f"{ppl_b:.2f}", f"{wt_mb:.0f}MB", "1.0x"))
    del m
    gc.collect()
    torch.cuda.empty_cache()

    # INT8-X
    print(f"[bench] INT8-X deploy (mode={args.mode}) ...", flush=True)
    gc.collect()
    torch.cuda.reset_peak_memory_stats()
    m = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16).cuda()
    if args.mode == "streaming":
        from .engine import Int8XEngine

        Int8XEngine._deploy_streaming(m, tuple(args.levels), verbose=True)
    elif args.mode == "graph":
        from .engine import Int8XEngine

        Int8XEngine._deploy_graph(m, tuple(args.levels), verbose=True)
    else:
        deploy_model(m, level_bits=tuple(args.levels), cache="full", verbose=True)
    m.eval()
    ms_x, mem_x = bench_forward(m, tok, warmup=3, n_runs=10)
    ppl_x = eval_ppl(m, tok, texts)
    est_bytes = sum(
        getattr(mod, "packed", {}).get("total_bytes", 0)
        for _, mod in m.named_modules()
    )
    pk_mb = est_bytes / 1e6 if est_bytes > 0 else wt_mb / 2.9
    rows.append((f"INT8-X({args.mode})", f"{ms_x:.0f}ms", f"{mem_x:.1f}GB",
                 f"{ppl_x:.2f}", f"{pk_mb:.0f}MB", f"{wt_mb/pk_mb:.1f}x"))
    del m
    gc.collect()
    torch.cuda.empty_cache()

    from .eval_utils import format_bench

    print("\n" + format_bench(rows, ["Mode", "Fwd", "GPU", "ppl", "Store", "comp"]))
    print(f"\nppl delta: {float(rows[1][3]) - float(rows[0][3]):+.2f}")


def _cmd_serve(args):
    from .server import serve

    serve(
        args.model,
        mode=args.mode,
        cache_path=args.cache,
        level_bits=tuple(args.levels),
        host=args.host,
        port=args.port,
        model_id=args.model_id,
        enable_thinking=args.think,
        batched=args.batched,
        codec=args.codec,
    )


def main():
    p = argparse.ArgumentParser(prog="ixrun", description="INT8-X inference engine")
    sub = p.add_subparsers(dest="cmd", required=True)

    ps = sub.add_parser("search", help="analyze optimal level-bit scheme")
    ps.add_argument("--model", default=MODEL_PATH)
    ps.add_argument("--max-levels", type=int, default=5)
    ps.add_argument("--topk", type=int, default=12)
    ps.set_defaults(func=_cmd_search)

    pg = sub.add_parser("generate", help="generate text")
    pg.add_argument("prompt")
    pg.add_argument("--model", default=MODEL_PATH)
    pg.add_argument("--mode", default="cached",
                    choices=["cached", "streaming", "graph", "udcq-graph", "udcq-spec", "step-graph"])
    pg.add_argument("--codec", default="int8x", choices=["int8x", "peakq", "bf16", "udcq", "udcq-stream"])
    pg.add_argument("--levels", type=int, nargs="+", default=list(DEFAULT_LEVELS))
    pg.add_argument("--max-new-tokens", type=int, default=128)
    pg.add_argument("--temperature", type=float, default=0.7)
    pg.add_argument("--do-sample", action="store_true", default=True)
    pg.add_argument("--no-sample", dest="do_sample", action="store_false")
    pg.add_argument("--stream", action="store_true")
    pg.add_argument("--cache", default=None, help="packed-weight cache file / UDCQ blob")
    pg.set_defaults(func=_cmd_generate)

    pc = sub.add_parser("chat", help="interactive chat REPL")
    pc.add_argument("--model", default=MODEL_PATH)
    pc.add_argument("--mode", default="streaming",
                    choices=["cached", "streaming", "udcq-graph", "udcq-spec", "step-graph"])
    pc.add_argument("--codec", default="int8x", choices=["int8x", "peakq", "bf16", "udcq", "udcq-stream"])
    pc.add_argument("--levels", type=int, nargs="+", default=list(DEFAULT_LEVELS))
    pc.add_argument("--max-new-tokens", type=int, default=256)
    pc.add_argument("--temperature", type=float, default=0.7)
    pc.add_argument("--no-sample", dest="do_sample", action="store_false")
    pc.add_argument("--cache", default=None, help="packed-weight cache file / UDCQ blob")
    pc.set_defaults(func=_cmd_chat, do_sample=True)

    pv = sub.add_parser("serve", help="OpenAI-compatible API server")
    pv.add_argument("--model", default=MODEL_PATH)
    pv.add_argument("--mode", default="streaming",
                    choices=["cached", "streaming", "udcq-graph", "udcq-spec", "step-graph"])
    pv.add_argument("--codec", default="int8x", choices=["int8x", "peakq", "bf16", "udcq", "udcq-stream"])
    pv.add_argument("--levels", type=int, nargs="+", default=list(DEFAULT_LEVELS))
    pv.add_argument("--cache", default=None, help="packed-weight cache file / UDCQ blob")
    pv.add_argument("--host", default="127.0.0.1")
    pv.add_argument("--port", type=int, default=8000)
    pv.add_argument("--model-id", default=None, help="model id advertised via /v1/models")
    pv.add_argument("--think", action="store_true",
                    help="enable thinking mode (default: direct answers)")
    pv.add_argument("--batched", action="store_true",
                    help="continuous batching: coalesce concurrent greedy requests "
                         "into batch forwards (~3x aggregate throughput)")
    pv.set_defaults(func=_cmd_serve)

    pb = sub.add_parser("bench", help="benchmark bf16 vs INT8-X")
    pb.add_argument("--model", default=MODEL_PATH)
    pb.add_argument("--mode", default="cached", choices=["cached", "streaming", "graph"])
    pb.add_argument("--levels", type=int, nargs="+", default=list(DEFAULT_LEVELS))
    pb.set_defaults(func=_cmd_bench)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
