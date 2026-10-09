import sys
sys.path.insert(0, r'E:\IXRUN')
import pandas
import torch

d = torch.load(r'C:\Users\Administrator\AppData\Local\Temp\opencode\h1_dump.pt')
h1 = d['h1'].float()
w = d['w'].float()

# Correct rmsnorm (matching HF Qwen3_5RMSNorm)
rms = torch.rsqrt(h1.pow(2).mean() + 1e-6)
y_ref = w * h1 * rms
print(f'input norm: {h1.norm().item():.4f}', flush=True)
print(f'w norm: {w.norm().item():.4f} | rms: {rms.item():.4f}', flush=True)
print(f'torch output norm: {y_ref.norm().item():.4f}', flush=True)

topv, topi = torch.topk(y_ref.abs(), 3)
print(f'torch top3: {[(round(v.item(), 2), int(i)) for v, i in zip(topv, topi)]}', flush=True)

# Check: does h1 have an outlier?
toph, tophi = torch.topk(h1.abs(), 3)
print(f'h1 top3 abs: {[(round(v.item(), 2), int(i)) for v, i in zip(toph, tophi)]}', flush=True)

# HF rms_norm (from qwen3_5 code): rsqrt(sum(x^2)/d + eps)
# vs my formula: rsqrt(sum(x^2)/d + eps) — same? Let me verify:
# HF: variance = hidden_states.pow(2).mean(-1, keepdim=True)
#     hidden_states * rsqrt(variance + eps)
# Same as: x * rsqrt(mean(x²) + eps) ✓

# So what does 5.519 mean? Let me print the C++ measurement:
print(f'C++ measured output norm: 60.399 (from bisect2)', flush=True)
print(f'ratio: {60.399 / y_ref.norm().item():.2f}', flush=True)

# The kernel should produce y = w * h1 * rsqrt(mean(h1²) + eps)
# Same formula → same result. If kernel gives 60.4 and torch gives 5.5,
# either the kernel reads different w, or different h1, or there's a
# fundamental bug in the kernel's reduction for this specific input.
print(f'h1 mean: {h1.mean().item():.6f} | h1 std: {h1.std().item():.4f}', flush=True)
print(f'w mean: {w.mean().item():.6f} | w std: {w.std().item():.4f}', flush=True)
print(f'h1*w (no norm) norm: {(h1 * w).norm().item():.4f}', flush=True)
