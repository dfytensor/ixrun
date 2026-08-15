"""Profile streaming inference: decode-kernel time vs GEMM time vs attention.

Run on MiniCPM5 (fast) then Qwen3.8. Measures a single decode-step forward
(1 token with KV cache) broken into: Triton decode (all layers), F.linear,
everything else (attention/norms).
"""
import sys, time
sys.setrecursionlimit(10000)
import pandas
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from ixrun.config import MODEL_PATH, DEFAULT_LEVELS
from ixrun.engine import Int8XEngine, StreamingLinear
from ixrun.linear import iter_quantizable_linears

torch.cuda.synchronize()

def profile(model_path, label):
    from transformers import AutoConfig
    from ixrun.engine import Int8XEngine as E
    print(f"\n=== {label} ===", flush=True)
    tok = AutoTokenizer.from_pretrained(model_path)
    m = E._load_any(model_path, torch.bfloat16, low_cpu=True).cuda()
    stats = E._deploy_streaming(m, DEFAULT_LEVELS, verbose=False)
    m.eval()
    eng = Int8XEngine(m, tok, stats); eng._finalize_device()

    # collect streaming layers in execution order
    lins = [(n, mod) for n, mod in m.named_modules() if isinstance(mod, StreamingLinear)]
    print(f"layers={len(lins)} packed={stats['total_bytes']/1e9:.2f}GB", flush=True)

    ids = tok("Hello world, this is a profiling prompt.", return_tensors="pt")["input_ids"].cuda()

    # --- time decode-only (sum of kernels) ---
    torch.cuda.synchronize(); t0 = time.time()
    NITER = 5
    for _ in range(NITER):
        for n, sl in lins:
            sl._decode()
    torch.cuda.synchronize()
    t_decode = (time.time()-t0)/NITER*1000

    # --- time GEMM-only (reuse last decoded buffer content) ---
    import torch.nn.functional as F
    xs = {sl.in_features: torch.randn(1, 1, sl.in_features, dtype=torch.bfloat16, device="cuda")
          for _, sl in lins}
    for n, sl in lins:
        sl._decode()  # ensure valid buffer
    torch.cuda.synchronize(); t0 = time.time()
    for _ in range(NITER):
        for n, sl in lins:
            w = sl._w_buf[:sl.N].view(sl.out_features, sl.in_features)
            F.linear(xs[sl.in_features], w)
    torch.cuda.synchronize()
    t_gemm = (time.time()-t0)/NITER*1000

    # --- full forward (prefill, 8 tokens) ---
    ids8 = ids[:, :8] if ids.shape[1] >= 8 else ids
    with torch.no_grad():
        for _ in range(2): m(ids8)
        torch.cuda.synchronize(); t0 = time.time()
        for _ in range(NITER): m(ids8)
        torch.cuda.synchronize()
        t_fwd8 = (time.time()-t0)/NITER*1000

    # --- single-token decode step with cache (generation regime) ---
    with torch.no_grad():
        out = m(ids8, use_cache=True)
        past = out.past_key_values
        nxt = ids8[:, -1:].clone()
        for _ in range(2): m(nxt, past_key_values=past, use_cache=True)
        torch.cuda.synchronize(); t0 = time.time()
        for _ in range(NITER): m(nxt, past_key_values=past, use_cache=True)
        torch.cuda.synchronize()
        t_tok = (time.time()-t0)/NITER*1000

    print(f"decode-kernels total : {t_decode:8.1f} ms", flush=True)
    print(f"GEMM-only total      : {t_gemm:8.1f} ms", flush=True)
    print(f"full fwd (8-tok)     : {t_fwd8:8.1f} ms", flush=True)
    print(f"decode-step (1 tok)  : {t_tok:8.1f} ms  <-- generation speed", flush=True)
    print(f"decode share of step : {t_decode/t_tok*100:.0f}%", flush=True)
    del m, eng; torch.cuda.empty_cache()
    import gc; gc.collect(); torch.cuda.empty_cache()

if __name__ == "__main__":
    profile(MODEL_PATH, "MiniCPM5-1B streaming")
