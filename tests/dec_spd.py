"""Full-decode kernel before/after int32 fix (quick throughput check)."""
import sys, time
sys.setrecursionlimit(10000)
import torch
from ixrun.tpab import encode_tpab, decode_tpab_triton, stage_gpu, decode_tpab_ref

torch.manual_seed(0)
for O, I in [(17408, 5120), (5120, 5120)]:
    base = torch.randn(O, I) * 0.005
    flat = base.view(-1)
    idx = torch.randperm(flat.numel())[: int(O*I*0.02)]
    flat[idx] = torch.randn(len(idx)) * 0.05
    w = base.bfloat16().cuda()
    p = encode_tpab(w, snr_target_db=26.0)
    st = stage_gpu(p, "cuda")
    a = decode_tpab_ref(p, device="cuda")
    b = decode_tpab_triton(p, device="cuda", staged=st)
    same = torch.equal(a, b)
    out = torch.zeros(p["T"]*p["n_per"], dtype=torch.float32, device="cuda")
    for _ in range(5): decode_tpab_triton(p, device="cuda", out_f32=out, staged=st)
    torch.cuda.synchronize(); t0 = time.time()
    for _ in range(50): decode_tpab_triton(p, device="cuda", out_f32=out, staged=st)
    torch.cuda.synchronize()
    ms = (time.time()-t0)/50*1e3
    print(f"{O}x{I}: {ms:.3f}ms ({O*I/ms/1e6:.0f}G/s)  triton==ref: {same}")
