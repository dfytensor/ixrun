"""int8g runtime: int8 + per-16 fp16 scale linear (8.9-9bpw) for the
StepGraph 1B engine.  The decode is ~3 instr/elem - the Q8_0 class
that beats every complex format in llama.cpp's own table."""
import torch

from experiments.int8g_cuda.int8g_cuda import (_load, int8g_decode_ref,
                                               int8g_pack)


class Int8GLinear(torch.nn.Module):
    def __init__(self, pk):
        super().__init__()
        self.q = pk['q'].cuda()
        self.scale = pk['scale'].cuda()
        self.out_features = pk['out_f']
        self.in_features = pk['in_f']
        self.packed = pk

    def _dequant(self):
        of, inf, g = self.out_features, self.in_features, self.packed['g']
        w = (self.q.float().view(of, inf // g, g)
             * self.scale.float().unsqueeze(-1))
        return w.view(of, inf).to(torch.bfloat16)

    def forward(self, x):
        if x.numel() == self.in_features:
            ext = _load()
            return ext.gemv(x.reshape(-1), self.q, self.scale,
                            self.out_features, self.in_features)
        return torch.nn.functional.linear(x, self._dequant())


def deploy_int8g(model, verbose=True, include_lm_head=True):
    from ixrun.linear import iter_quantizable_linears, _set_parent_child

    n = 0
    for name, mod in list(iter_quantizable_linears(model)):
        W = mod.weight.data.cuda()
        pk = int8g_pack(W)
        del W
        _set_parent_child(model, name, Int8GLinear(pk))
        n += 1
    lm = getattr(model, 'lm_head', None)
    if include_lm_head and type(lm).__name__ == 'Linear' \
            and lm.weight.numel() > 1_000_000:
        W = lm.weight.data.cuda()
        pk = int8g_pack(W)
        del W
        _set_parent_child(model, 'lm_head', Int8GLinear(pk))
        n += 1
    torch.cuda.empty_cache()
    if verbose:
        print(f'[int8g] {n} linears deployed (9bpw)', flush=True)
    return n
