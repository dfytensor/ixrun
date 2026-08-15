"""Check which implementation (fla vs torch) is bound for the delta-rule kernels."""
import sys
sys.setrecursionlimit(10000)
import pandas
import torch
from ixrun.config import QWEN38_PATH, DEFAULT_LEVELS
from ixrun.engine import Int8XEngine
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained(QWEN38_PATH, trust_remote_code=True)
m = Int8XEngine._load_any(QWEN38_PATH, torch.bfloat16, low_cpu=True)
stats = Int8XEngine._deploy_streaming(m, DEFAULT_LEVELS, cache_path=r"E:\models\qwen38_packed.pt", verbose=False)
eng = Int8XEngine(m, tok, stats); eng._finalize_device()

import transformers.models.qwen3_5.modeling_qwen3_5 as MQ
import inspect
for fn_name in ("torch_recurrent_gated_delta_rule", "torch_chunk_gated_delta_rule"):
    f = getattr(MQ, fn_name, None)
    if f is None:
        print(f"{fn_name}: NOT FOUND"); continue
    mod = getattr(f, "__module__", "?")
    try:
        src = inspect.getsource(f)
        is_torch = "def torch_" in src[:2000]
    except Exception:
        is_torch = None
    print(f"{fn_name}: bound module={mod} torch_eager={is_torch}")

# what does the fla package expose?
try:
    from fla.ops.gated_delta_rule import recurrent_gated_delta_rule as fla_rec  # noqa
    print("fla recurrent_gated_delta_rule: import OK")
except Exception as e:
    print("fla recurrent import FAIL:", type(e).__name__, str(e)[:120])
try:
    import fla
    print("fla version:", getattr(fla, "__version__", "?"))
except Exception as e:
    print(e)

# kernelization status from transformers side
try:
    from transformers.integrations import hub_kernels as hk
    print("kernels_available:", hk.is_kernels_available(), "| kernels_enabled:", hk._kernels_enabled)
except Exception as e:
    print("hk introspect fail:", e)
