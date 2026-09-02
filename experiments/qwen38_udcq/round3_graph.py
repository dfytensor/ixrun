# -*- coding: utf-8 -*-
"""round3 v2: clean flow — capture on fresh cache, hard-reset, prefill,
eager-validate, hard-reset, prefill, graph decode. No snapshot/restore."""
import sys, time, gc
sys.setrecursionlimit(20000)
sys.path.insert(0, r'E:\IXRUN')
import pandas
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from transformers.cache_utils import StaticCache
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5Attention
from transformers.models.qwen3_5.modeling_qwen3_5 import apply_rotary_pos_emb

sys.argv = ['x']
import importlib.util
spec = importlib.util.spec_from_file_location(
    'slim', r'E:\IXRUN\experiments\qwen38_udcq\qwen38_slim_resident.py')
slim = importlib.util.module_from_spec(spec)
spec.loader.exec_module(slim)

from ixrun.fla_patch import apply_fla_kernels
apply_fla_kernels()

MAX_CTX = 256


def _attn_forward_static(
    self, hidden_states, position_embeddings, attention_mask=None,
    past_key_values=None, **kwargs,
):
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)

    query_states, gate = torch.chunk(
        self.q_proj(hidden_states).view(*input_shape, -1, self.head_dim * 2),
        2, dim=-1)
    gate = gate.reshape(*input_shape, -1)
    query_states = self.q_norm(query_states.view(hidden_shape)).transpose(1, 2)
    key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    k_full, v_full = past_key_values.update(key_states, value_states, self.layer_idx)

    cum = past_key_values.layers[self.layer_idx].cumulative_length
    MAX = k_full.shape[-2]
    ar = torch.arange(MAX, device=k_full.device)
    keep = ar < cum
    mask = torch.where(keep, 0.0, float('-inf')).to(query_states.dtype)
    mask = mask.view(1, 1, 1, MAX)

    n_rep = query_states.shape[1] // k_full.shape[1]
    if n_rep > 1:
        k_full = k_full.repeat_interleave(n_rep, dim=1)
        v_full = v_full.repeat_interleave(n_rep, dim=1)
    attn_output = F.scaled_dot_product_attention(
        query_states, k_full, v_full, attn_mask=mask, scale=self.scaling)

    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    attn_output = attn_output * torch.sigmoid(gate)
    return self.o_proj(attn_output), None


Qwen3_5Attention.forward = _attn_forward_static


@torch.no_grad()
def main():
    tok = AutoTokenizer.from_pretrained(r'E:\models\Qwen3.8-27B')
    print('deploying slim...', flush=True)
    m = slim.load_model()
    slim.deploy_slim(m, verbose=True)
    m.eval()

    tm = m.model
    if hasattr(tm, 'language_model'):
        tm = tm.language_model
    layers = tm.layers
    final_norm = tm.norm
    rotary = tm.rotary_emb
    H = tm.config.hidden_size
    dev = 'cuda'

    pos_all = torch.arange(MAX_CTX, device=dev).unsqueeze(0)
    with torch.no_grad():
        dummy = torch.zeros(1, MAX_CTX, H, dtype=torch.bfloat16, device=dev)
        cos_all, sin_all = rotary(dummy, pos_all)
    if cos_all.dim() == 4:
        cos_all = cos_all[:, :, 0]
    if cos_all.dim() == 3:
        cos_all = cos_all[0]
        sin_all = sin_all[0]

    cache = StaticCache(config=m.config, max_cache_len=MAX_CTX)

    emb_buf = torch.zeros(1, 1, H, dtype=torch.bfloat16, device=dev)
    cos_buf = torch.zeros(1, 1, cos_all.shape[-1], dtype=cos_all.dtype, device=dev)
    sin_buf = torch.zeros_like(cos_buf)
    pos_buf = torch.zeros(1, dtype=torch.long, device=dev)

    @torch.no_grad()
    def step():
        h = emb_buf
        for layer in layers:
            h = layer(h, position_embeddings=(cos_buf, sin_buf),
                      attention_mask=None, position_ids=pos_buf.view(1, 1),
                      past_key_values=cache)
            if isinstance(h, tuple):
                h = h[0]
        h = final_norm(h)
        return m.lm_head(h)

    emb_w = None
    def slim_emb_weight():
        nonlocal emb_w
        if emb_w is None:
            for name, mod in m.named_modules():
                if type(mod).__name__ == 'CpuEmbed':
                    emb_w = mod.weight_cpu
                    break
        return emb_w

    def set_inputs(token, t):
        emb_buf.copy_(slim_emb_weight()[token].view(1, 1, H).to(dev, torch.bfloat16))
        cos_buf.copy_(cos_all[t].view(1, 1, -1))
        sin_buf.copy_(sin_all[t].view(1, 1, -1))
        pos_buf.fill_(t)

    def hard_reset():
        for lay in cache.layers:
            cum = getattr(lay, 'cumulative_length', None)
            if cum is not None:
                cum.zero_()
            if hasattr(lay, 'has_previous_state'):
                hps = lay.has_previous_state
                if isinstance(hps, dict):
                    for k in hps:
                        hps[k] = False
                else:
                    lay.has_previous_state = [False] * len(hps)
            for cs in getattr(lay, 'conv_states', []) or []:
                if isinstance(cs, torch.Tensor):
                    cs.zero_()
            for rs in getattr(lay, 'recurrent_states', []) or []:
                if isinstance(rs, torch.Tensor):
                    rs.zero_()

    def prefill(ids):
        last = None
        for i, tid in enumerate(ids):
            set_inputs(tid, i)
            last = step()
        return last

    # ================= capture on a FRESH cache =================
    print('capturing graph on fresh cache...', flush=True)
    emb_buf.zero_(); cos_buf.zero_(); sin_buf.zero_(); pos_buf.fill_(0)
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            step()
    torch.cuda.current_stream().wait_stream(s)

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        logits_static = step()
    hard_reset()
    print('captured + reset.', flush=True)

    prompt = 'The theory of relativity states that'
    ids = tok(prompt, return_tensors='pt')['input_ids'][0].tolist()

    # ================= pass 1: eager reference =================
    logits = prefill(ids)
    nxt = logits[:, -1].argmax(-1).item()
    gen_eager = [nxt]
    t0 = time.time()
    N = 30
    for t in range(len(ids), len(ids) + N):
        set_inputs(nxt, t)
        logits = step()
        nxt = logits[:, -1].argmax(-1).item()
        gen_eager.append(nxt)
    torch.cuda.synchronize()
    dt = (time.time() - t0) / N
    print(f'eager decode: {dt*1000:.0f} ms/tok ({1/dt:.2f} tok/s)', flush=True)
    print('  ->', repr(tok.decode(gen_eager[:24])), flush=True)

    # ================= pass 2: graph decode =================
    hard_reset()
    prefill(ids)
    nxt = ids[-1]
    gen_graph = []
    t0 = time.time()
    for t in range(len(ids), len(ids) + N):
        set_inputs(nxt, t)
        g.replay()
        nxt = logits_static[:, -1].argmax(-1).item()
        gen_graph.append(nxt)
    torch.cuda.synchronize()
    dt = (time.time() - t0) / N
    print(f'graph decode: {dt*1000:.1f} ms/tok ({1/dt:.2f} tok/s)', flush=True)
    print('  ->', repr(tok.decode(gen_graph[:24])), flush=True)
    print(f'agreement (first 10): {gen_eager[:10] == gen_graph[:10]}', flush=True)

    # ---- graph self-consistency: rerun from scratch, must match exactly ----
    hard_reset()
    prefill(ids)
    nxt = ids[-1]
    gen_graph2 = []
    for t in range(len(ids), len(ids) + N):
        set_inputs(nxt, t)
        g.replay()
        nxt = logits_static[:, -1].argmax(-1).item()
        gen_graph2.append(nxt)
    print(f'graph self-consistency: {gen_graph == gen_graph2}', flush=True)

    # ---- more prompts, content sanity ----
    for prompt in ['def quick_sort(arr):', '北京最值得游览的三个景点是']:
        ids2 = tok(prompt, return_tensors='pt')['input_ids'][0].tolist()
        hard_reset()
        prefill(ids2)
        nxt = ids2[-1]
        gg = []
        t0 = time.time()
        for t in range(len(ids2), len(ids2) + 40):
            set_inputs(nxt, t)
            g.replay()
            nxt = logits_static[:, -1].argmax(-1).item()
            gg.append(nxt)
        torch.cuda.synchronize()
        dt = (time.time() - t0) / 40
        print(f'[{1/dt:.2f} tok/s] {prompt!r}\n  -> {tok.decode(gg)[:120]!r}',
              flush=True)
    print(f'peak GPU = {torch.cuda.max_memory_allocated()/1e9:.2f}GB', flush=True)


if __name__ == '__main__':
    main()
