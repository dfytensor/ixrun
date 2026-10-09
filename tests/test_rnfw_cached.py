import sys
sys.path.insert(0, r'E:\IXRUN')
import pandas
import torch
# Use the already-compiled rnfw3 binary (no rebuild needed)
from torch.utils.cpp_extension import load_inline
# Just import the cached module
ext = torch.ops.load_library.__class__  # won't work; use importlib
# Actually: load_inline with SAME name + SAME source = cached, no rebuild
# But source has changed... Instead: use importlib on the cached .pyd
import importlib.util
ext_path = r'C:\Users\Administrator\AppData\Local\torch_extensions\torch_extensions\Cache\py312_cu126\ixrun_rnfw3\ixrun_rnfw3.pyd'
spec = importlib.util.spec_from_file_location('rnfw3', ext_path)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

d = torch.load(r'C:\Users\Administrator\AppData\Local\Temp\opencode\h1_dump.pt')
h1 = d['h1'].cuda().float()
w = d['w'].cuda().float()
y = mod.rmsnorm_fw_out(h1.view(1, -1), w, 1e-6).view(-1)
y_ref = w * h1 * torch.rsqrt(h1.pow(2).mean() + 1e-6)
e = ((y - y_ref).norm() / y_ref.norm().clamp_min(1e-9)).item()
print(f'rnfw3(cached) on REAL h1: kernel {y.norm().item():.4f} vs torch {y_ref.norm().item():.4f} rel-err {e:.2e}', flush=True)
