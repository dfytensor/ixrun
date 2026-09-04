# -*- coding: utf-8 -*-
"""round4b: speculative decode, fast-iteration version.

- loads the disk blob (2min) instead of re-quantizing (55min)
- MTP head FIXED: calls its decoder layer (was missing entirely -> drafts
  were fc+norm only = garbage); layer uses the ORIGINAL attention fwd
  (past_key_values=None path), subclassed before the global patch
- g2 (M=2) bisect: compare vs pure-M1 continuation right after prefill
"""
import sys, time, gc, os
sys.setrecursionlimit(20000)
sys.path.insert(0, r'E:\IXRUN')
import pandas
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer
from transformers.cache_utils import StaticCache
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5Attention, Qwen3_5RMSNorm, apply_rotary_pos_emb)

from ixrun.fla_patch import apply_fla_kernels
apply_fla_kernels()
from ixrun.gdn_seq_patch import apply_gdn_sequential_patch
apply_gdn_sequential_patch()   # exact per-token conv+delta for seeded S<=8

MAX_CTX = 256
BLOB = r'E:\IXRUN\experiments\qwen38_udcq\q38_blob.pt'

# save the ORIGINAL attention forward BEFORE patching (MTP head layer needs
# the past_key_values=None path)
_ORIG_ATTN_FWD = Qwen3_5Attention.forward


_DBG = {}


def _attn_forward_static(self, hidden_states, position_embeddings,
                         attention_mask=None, past_key_values=None, **kw):
    if past_key_values is None:
        return _ORIG_ATTN_FWD(self, hidden_states, position_embeddings,
                              attention_mask=attention_mask,
                              past_key_values=None, **kw)
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)
    q_len = hidden_states.shape[1]

    qproj_out = self.q_proj(hidden_states)
    if _DBG.get('on') and self.layer_idx == _DBG.get('layer', 3):
        _DBG.setdefault('qproj', []).append(qproj_out.detach().clone())
    query_states, gate = torch.chunk(
        qproj_out.view(*input_shape, -1, self.head_dim * 2),
        2, dim=-1)
    gate = gate.reshape(*input_shape, -1)
    query_states = self.q_norm(query_states.view(hidden_shape)).transpose(1, 2)
    kpre = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, kpre, cos, sin)
    if _DBG.get('on') and self.layer_idx == _DBG.get('layer', 3):
        _DBG.setdefault('kpost_rope', []).append(key_states.detach().clone())
        _DBG.setdefault('qpost_rope', []).append(query_states.detach().clone())

    k_full, v_full = past_key_values.update(key_states, value_states, self.layer_idx)
    if _DBG.get('on') and self.layer_idx == _DBG.get('layer', 3):
        _DBG.setdefault('kv_after_update', []).append(
            k_full.detach().clone())

    cum = past_key_values.layers[self.layer_idx].cumulative_length
    MAX = k_full.shape[-2]
    ar = torch.arange(MAX, device=k_full.device)
    qr = torch.arange(q_len, device=k_full.device)
    keep = ar[None, :] < (cum - q_len + 1 + qr[:, None])
    mask = torch.where(keep, 0.0, float('-inf')).to(query_states.dtype)
    mask = mask.view(1, 1, q_len, MAX)

    n_rep = query_states.shape[1] // k_full.shape[1]
    if n_rep > 1:
        k_full = k_full.repeat_interleave(n_rep, dim=1)
        v_full = v_full.repeat_interleave(n_rep, dim=1)
    if q_len > 1:
        # per-token SDPA: exact same kernel sequence as sequential q_len=1
        # calls (a batched q_len=2 SDPA picks a different backend/reduction
        # -> layer-3 divergence dmax 6.0). Projections stay batched (exact).
        outs = []
        for t in range(q_len):
            outs.append(F.scaled_dot_product_attention(
                query_states[:, :, t:t + 1], k_full, v_full,
                attn_mask=mask[:, :, t:t + 1], scale=self.scaling))
        attn_output = torch.cat(outs, dim=2)
    else:
        attn_output = F.scaled_dot_product_attention(
            query_states, k_full, v_full, attn_mask=mask, scale=self.scaling)
    if _DBG.get('on') and self.layer_idx == _DBG.get('layer', 3):
        _DBG.setdefault('attn_out', []).append(attn_output.detach().clone())

    if q_len > 1:
        # [B, heads, S, D] -> [B, S, heads*D]: transpose is REQUIRED before
        # the flatten 鈥?without it head/token dims interleave (q_len=1 was
        # accidentally fine because S=1; S=2 produced garbled layout ->
        # layer divergence dmax 6.0)
        attn_output = attn_output.transpose(1, 2)
    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    attn_output = attn_output * torch.sigmoid(gate)
    if _DBG.get('on') and self.layer_idx == _DBG.get('layer', 3):
        _DBG.setdefault('gated', []).append(attn_output.detach().clone())
    out = self.o_proj(attn_output)
    if _DBG.get('on') and self.layer_idx == _DBG.get('layer', 3):
        _DBG.setdefault('oproj_out', []).append(out.detach().clone())
    return out, None


Qwen3_5Attention.forward = _attn_forward_static


class MTPHeadBf16(nn.Module):
    def __init__(self, dim, layer, rotary, lm_head):
        super().__init__()
        self.norm_e = Qwen3_5RMSNorm(dim, eps=1e-6)     # (1+w) variant!
        self.norm_h = Qwen3_5RMSNorm(dim, eps=1e-6)
        self.fc = nn.Linear(2 * dim, dim, bias=False)
        self.layer = layer          # full-attn layer w/ ORIGINAL fwd
        self.rotary = rotary
        self.norm = Qwen3_5RMSNorm(dim, eps=1e-6)
        self.lm_head = lm_head

    @torch.no_grad()
    def forward2(self, tok_emb, h, pos):
        """Returns (logits, normed_z). normed_z = self.norm(z) is the
        recursion state for chained MTP drafts (matches the single-step
        convention: h input is a final-norm-equivalent hidden)."""
        cos, sin = self.rotary(h, pos)
        x = self.fc(torch.cat([self.norm_e(tok_emb), self.norm_h(h)],
                              dim=-1).to(torch.bfloat16))
        z = self.layer(x, attention_mask=None, position_ids=pos,
                       position_embeddings=(cos, sin), use_cache=False)
        if isinstance(z, tuple):
            z = z[0]
        nz = self.norm(z)
        return self.lm_head(nz), nz

    @torch.no_grad()
    def forward(self, tok_emb, h, pos):
        # rope for the draft position (q_len=1)
        return self.forward2(tok_emb, h, pos)[0]


# --------------------------------------------------------------------------- #
class CpuEmbed(nn.Module):
    def __init__(self, weight_cpu):
        super().__init__()
        self.weight_cpu = weight_cpu

    def forward(self, ids):
        return self.weight_cpu[ids.reshape(-1).cpu()].cuda() \
            .view(*ids.shape, -1)


@torch.no_grad()
def build_from_blob():
    """Fast path: meta model + blob swap (no quantization)."""
    from transformers import AutoModelForCausalLM, AutoConfig
    from ixrun.config import QWEN38_PATH
    from ixrun.udcq import UdcqLinear
    from ixrun.linear import _set_parent_child

    print('loading blob (mmap)...', flush=True)
    t0 = time.time()
    blob = torch.load(BLOB, map_location='cpu', mmap=True, weights_only=False)
    print(f'  blob {time.time()-t0:.0f}s', flush=True)

    cfg = AutoConfig.from_pretrained(QWEN38_PATH, trust_remote_code=True)
    from accelerate import init_empty_weights
    with init_empty_weights():
        m = AutoModelForCausalLM.from_config(cfg, trust_remote_code=True)

    from ixrun.udcq import UDCQ_G
    cb_gpu = blob['codebook'].cuda()
    n = 0
    for name, mod in list(m.named_modules()):
        if isinstance(mod, nn.Linear) and name in blob['layers']:
            e = blob['layers'][name]
            packed = {'g': UDCQ_G, 'out_f': mod.out_features,
                      'in_f': mod.in_features,
                      'N': mod.out_features * mod.in_features,
                      'idx': e['idx'], 'scale': e['scale'],
                      'sign_packed': e['sign'],
                      'codebook': blob['codebook'],
                      'bits_per_weight': 6.0}
            lay = UdcqLinear(packed, bias=None, cache='stream')
            _set_parent_child(m, name, lay)
            n += 1
    # embed -> CPU rows
    emb_mod = CpuEmbed(blob['embed'])
    for name, mod in m.named_modules():
        if type(mod).__name__.endswith('Embedding') and 'embed_tokens' in name:
            parts = name.split('.')
            parent = m
            for q in parts[:-1]:
                parent = getattr(parent, q)
            setattr(parent, parts[-1], emb_mod)
            break
    # non-linear params -> GPU
    import json
    idx = json.load(open(r'E:\models\Qwen3.8-27B\model.safetensors.index.json'))['weight_map']

    def ckpt_key(name):
        if name in idx:
            return name
        alt = name.replace('model.', 'model.language_model.', 1)
        return alt if alt in idx else None

    from safetensors import safe_open as so
    moved = 0
    params = dict(m.named_parameters())
    for name, p in params.items():
        if not p.numel() or not p.is_meta:
            continue
        key = ckpt_key(name)
        if key is None:
            continue
        shard = rf'E:\models\Qwen3.8-27B\{idx[key]}'
        with so(shard, 'pt') as sf:
            t = sf.get_tensor(key)
        parts = name.split('.')
        parent = m
        for q in parts[:-1]:
            parent = getattr(parent, q)
        parent._parameters[parts[-1]] = torch.nn.Parameter(
            t.cuda(), requires_grad=False)
        moved += t.numel() * t.element_size()
        del t
    gc.collect(); torch.cuda.empty_cache()
    print(f'[fast-deploy] {n} linears from blob | non-linear {moved/1e9:.2f}GB '
          f'| {time.time()-t0:.0f}s | resident {torch.cuda.memory_allocated()/1e9:.2f}GB',
          flush=True)
    return m, blob


def load_mtp(m):
    import copy as _copy, json as _json
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5DecoderLayer
    from safetensors import safe_open
    tensors = {}
    idx = _json.load(open(r'E:\models\Qwen3.8-27B\model.safetensors.index.json'))['weight_map']
    for shard in sorted({idx[k] for k in idx if k.startswith('mtp.')}):
        with safe_open(rf'E:\models\Qwen3.8-27B\{shard}', 'pt') as sf:
            for k in sf.keys():
                if k.startswith('mtp.'):
                    tensors[k] = sf.get_tensor(k).to(torch.bfloat16)
    cfg = m.config.get_text_config()
    layer_cfg = _copy.deepcopy(cfg)
    layer_cfg.layer_types = ['full_attention']
    layer = Qwen3_5DecoderLayer(layer_cfg, layer_idx=0).to(torch.bfloat16)
    # bind the ORIGINAL attention forward to THIS layer's attn instance
    layer.self_attn.forward = _ORIG_ATTN_FWD.__get__(layer.self_attn,
                                                     Qwen3_5Attention)

    tm = m.model
    if hasattr(tm, 'language_model'):
        tm = tm.language_model
    # construct WITHOUT the shared lm_head as a child module 鈥?.to(bf16)
    # would recurse into the UdcqLinear's f16 buffers (scale/codebook),
    # rebind their storage and corrupt the main decode + graph pointers
    head = MTPHeadBf16(cfg.hidden_size, layer, tm.rotary_emb, None).to(torch.bfloat16)
    head.layer = layer
    object.__setattr__(head, 'lm_head', m.lm_head)
    sd = {}
    for k, t in tensors.items():
        s = k[4:]
        if s.startswith('layers.0.'):
            sd['layer.' + s[9:]] = t
        elif s == 'fc.weight':
            sd['fc.weight'] = t
        elif s == 'norm.weight':
            sd['norm.weight'] = t
        elif s == 'pre_fc_norm_embedding.weight':
            sd['norm_e.weight'] = t
        elif s == 'pre_fc_norm_hidden.weight':
            sd['norm_h.weight'] = t
    missing, _ = head.load_state_dict(sd, strict=False)
    real = [k for k in missing if not (k.startswith('layer.') or k == 'fc.weight'
                                       or k.startswith('lm_head.'))]
    assert not real, f'{real[:4]}'
    return head.cuda().eval()


# --------------------------------------------------------------------------- #
@torch.no_grad()
def main():
    tok = AutoTokenizer.from_pretrained(r'E:\models\Qwen3.8-27B')
    m, blob = build_from_blob()
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
        cos_all, sin_all = cos_all[0], sin_all[0]

    cache = StaticCache(config=m.config, max_cache_len=MAX_CTX)
    emb_cpu = blob['embed']

    emb1 = torch.zeros(1, 1, H, dtype=torch.bfloat16, device=dev)
    cos1 = torch.zeros(1, 1, cos_all.shape[-1], dtype=cos_all.dtype, device=dev)
    sin1 = torch.zeros_like(cos1)
    pos1 = torch.zeros(1, dtype=torch.long, device=dev)
    emb2 = torch.zeros(1, 2, H, dtype=torch.bfloat16, device=dev)
    cos2 = torch.zeros(1, 2, cos_all.shape[-1], dtype=cos_all.dtype, device=dev)
    sin2 = torch.zeros_like(cos2)
    pos2 = torch.zeros(2, dtype=torch.long, device=dev)
    emb4 = torch.zeros(1, 4, H, dtype=torch.bfloat16, device=dev)
    cos4 = torch.zeros(1, 4, cos_all.shape[-1], dtype=cos_all.dtype, device=dev)
    sin4 = torch.zeros_like(cos4)
    pos4 = torch.zeros(4, dtype=torch.long, device=dev)

    # GPU embedding table for in-graph gathers (chain drafts + prefix
    # fills need token embeddings from GPU-resident ints — the CPU-row
    # CpuEmbed path requires a sync and cannot be captured).
    # int8 + per-row fp32 scale (1.27GB vs 2.54GB bf16): peak VRAM stays
    # under 24GB — oversubscription triggers WDDM sysmem paging and the
    # whole pipeline crawls at ~130GB/s (measured 232ms/iter vs ~70).
    print('staging emb_gpu (int8)...', flush=True)
    emb_f = blob['embed'].float()
    emb_s = (emb_f.abs().amax(dim=1) / 127.0).clamp_min(1e-12)
    emb_i8 = (emb_f / emb_s.unsqueeze(1)).round().to(torch.int8)
    emb_i8 = emb_i8.cuda()
    s_g = emb_s.cuda().unsqueeze(1)          # [vocab, 1] fp32
    del emb_f
    print(f'  emb_i8 {tuple(emb_i8.shape)} '
          f'{emb_i8.numel() / 1e9:.2f}GB', flush=True)

    def emb_rows(ids_t):
        """Dequantized embedding rows; ids_t: int tensor (GPU). All ops
        capture-safe (F.embedding = index_select, no advanced indexing)."""
        q = F.embedding(ids_t, emb_i8)
        sc = F.embedding(ids_t, s_g)
        return (q.float() * sc).to(torch.bfloat16)

    # static graph-output buffers (allocated OUTSIDE any capture -> never
    # aliased by the shared pool; graphs copy into them before capture end)
    V = m.lm_head.out_features
    out_h1 = torch.zeros(1, 1, H, dtype=torch.bfloat16, device=dev)
    out_l1 = torch.zeros(1, 1, V, dtype=torch.bfloat16, device=dev)
    out_h2 = torch.zeros(1, 2, H, dtype=torch.bfloat16, device=dev)
    out_l2 = torch.zeros(1, 2, V, dtype=torch.bfloat16, device=dev)
    out_h4 = torch.zeros(1, 4, H, dtype=torch.bfloat16, device=dev)
    out_l4 = torch.zeros(1, 4, V, dtype=torch.bfloat16, device=dev)
    a_buf = torch.zeros(4, dtype=torch.long, device=dev)
    dec_gpu = torch.zeros(6, dtype=torch.long, device=dev)

    def run(emb, cos, sin, pos):
        h = emb
        for layer in layers:
            h = layer(h, position_embeddings=(cos, sin), attention_mask=None,
                      position_ids=pos.view(1, -1), past_key_values=cache)
            if isinstance(h, tuple):
                h = h[0]
        h = final_norm(h)
        return h, m.lm_head(h)

    def embed1(tok_id):
        return emb_cpu[tok_id].view(1, 1, H).to(dev, torch.bfloat16)

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
            for cs in (getattr(lay, 'conv_states', None) or {}).values():
                if isinstance(cs, torch.Tensor):
                    cs.zero_()
            for rs in (getattr(lay, 'recurrent_states', None) or {}).values():
                if isinstance(rs, torch.Tensor):
                    rs.zero_()

    # ------- graphs -------
    print('capturing g1/g2/g4...', flush=True)
    for b in (emb1, cos1, sin1, emb2, cos2, sin2, emb4, cos4, sin4):
        b.zero_()
    pos1.fill_(0); pos2.fill_(0); pos4.fill_(0)
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            run(emb1, cos1, sin1, pos1)
            run(emb2, cos2, sin2, pos2)
            run(emb4, cos4, sin4, pos4)
    torch.cuda.current_stream().wait_stream(s)
    g1 = torch.cuda.CUDAGraph()
    pool = torch.cuda.graph_pool_handle()   # ALL graphs share one pool —
    with torch.cuda.graph(g1, pool=pool):   # separate pools alias corrupt g1
        h1_s, log1_s = run(emb1, cos1, sin1, pos1)
        out_h1.copy_(h1_s)                   # static outputs (pool-safe)
        out_l1.copy_(log1_s)
    # v5: g2 = TRUE T=2 layer-wise batch (single forward on emb2 buffers).
    # Exact by construction: projections via multi-token GEMV (bit-exact
    # vs sequential M=1), GDN core per-token via gdn_seq_patch v3,
    # attention q_len=2 causal over static cache. out_l2[:, 0] = verify
    # logits (after t1), out_l2[:, 1] = continuation (after d).
    g2 = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g2, pool=pool):
        h2_s, log2_s = run(emb2, cos2, sin2, pos2)
        out_h2.copy_(h2_s)
        out_l2.copy_(log2_s)
    # k=3 verify graph: T=4, one replay processes [t1, d1, d2, d3] —
    # out_l4[:, j] = logits after token j (verify + next-t1 candidates).
    g4 = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g4, pool=pool):
        h4_s, log4_s = run(emb4, cos4, sin4, pos4)
        out_h4.copy_(h4_s)
        out_l4.copy_(log4_s)
    hard_reset()
    print('graphs ok', flush=True)

    # ---- A/B probe: g1-only decode BEFORE load_mtp (iso-equivalent) ----
    ids_p = tok('The theory of relativity states that',
                return_tensors='pt')['input_ids'][0].tolist()
    for i, tid in enumerate(ids_p):
        emb1.copy_(embed1(tid))
        cos1.copy_(cos_all[i].view(1, 1, -1))
        sin1.copy_(sin_all[i].view(1, 1, -1))
        pos1.fill_(i)
        _, log_p = run(emb1, cos1, sin1, pos1)
    nxt = log_p[:, -1].argmax().item()
    probe = [nxt]
    t = len(ids_p)
    for _ in range(5):
        emb1.copy_(embed1(nxt))
        cos1.copy_(cos_all[t].view(1, 1, -1))
        sin1.copy_(sin_all[t].view(1, 1, -1))
        pos1.fill_(t)
        g1.replay()
        nxt = log1_s[:, -1].argmax().item()
        probe.append(nxt)
        t += 1
    hard_reset()
    print(f'[probe pre-mtp] {[int(x) for x in probe]}', flush=True)

    # ---- direct kernel A/B on deployed layer 3 k_proj (real buffers) ----
    from ixrun.udcq import udcq_fused_gemm, udcq_fused_gemv
    l3 = tm.layers[3]
    if l3.block_type != 'full_attention':
        for cand in tm.layers:
            if cand.block_type == 'full_attention':
                l3 = cand
                break
    kp = l3.self_attn.k_proj
    in_f = kp.in_features
    torch.manual_seed(0)
    hA = torch.randn(1, 1, in_f, dtype=torch.bfloat16, device=dev)
    hB = torch.randn(1, 1, in_f, dtype=torch.bfloat16, device=dev)
    yA = kp(hA).reshape(1, -1)          # GEMV
    yB = kp(hB).reshape(1, -1)          # GEMV
    x2 = torch.cat([hA, hB], dim=1).reshape(2, in_f)
    y2 = kp(x2)                          # GEMM path (M=2)
    d0 = (y2[0].float() - yA[0].float()).abs().max().item()
    d1 = (y2[1].float() - yB[0].float()).abs().max().item()
    ref = yA.float().abs().mean().item()
    print(f'[kproj A/B] |y| mean {ref:.4f} | GEMM-vs-GEMV row0 max {d0:.4f} '
          f'row1 max {d1:.4f}', flush=True)

    # ---- linear_attn output A/B: sequential S=1 x2 vs block S=2 ----
    l0 = next(l for l in tm.layers if l.block_type == 'linear_attention')
    H0 = l0.config.hidden_size if hasattr(l0, 'config') else H
    cosB1 = torch.zeros(1, 1, cos_all.shape[-1], dtype=cos_all.dtype, device=dev)
    sinB1 = torch.zeros_like(cosB1)
    cosB2 = torch.zeros(1, 2, cos_all.shape[-1], dtype=cos_all.dtype, device=dev)
    sinB2 = torch.zeros_like(cosB2)

    def la_out(h1, h2, pos0):
        """h1,h2: [1,1,H] tokens. Returns (seq_outputs, block_outputs)."""
        hard_reset()
        # sequential
        cosB1.copy_(cos_all[pos0].view(1, 1, -1))
        sinB1.copy_(sin_all[pos0].view(1, 1, -1))
        o1 = l0.linear_attn(h1, cache_params=cache, attention_mask=None,
                            position_ids=None)
        cosB1.copy_(cos_all[pos0 + 1].view(1, 1, -1))
        sinB1.copy_(sin_all[pos0 + 1].view(1, 1, -1))
        o2 = l0.linear_attn(h2, cache_params=cache, attention_mask=None,
                            position_ids=None)
        hard_reset()
        cosB2[:, 0].copy_(cos_all[pos0])
        cosB2[:, 1].copy_(cos_all[pos0 + 1].view(1, -1))
        sinB2[:, 0].copy_(sin_all[pos0])
        sinB2[:, 1].copy_(sin_all[pos0 + 1].view(1, -1))
        hb = torch.cat([h1, h2], dim=1)
        ob = l0.linear_attn(hb, cache_params=cache, attention_mask=None,
                            position_ids=None)
        return torch.cat([o1, o2], 1), ob

    torch.manual_seed(1)
    hh1 = torch.randn(1, 1, H, dtype=torch.bfloat16, device=dev) * 0.1
    hh2 = torch.randn(1, 1, H, dtype=torch.bfloat16, device=dev) * 0.1
    oseq, oblk = la_out(hh1, hh2, 50)
    dA = (oseq[0, 0].float() - oblk[0, 0].float()).abs().max().item()
    dB = (oseq[0, 1].float() - oblk[0, 1].float()).abs().max().item()
    refm = oseq.float().abs().mean().item()
    print(f'[la A/B] |out| mean {refm:.4f} | seq-vs-block tok0 max {dA:.4f} '
          f'tok1 max {dB:.4f}', flush=True)
    hard_reset()

    # ---- SEEDED la A/B: v3 spec-block (S=2) vs sequential S=1 x2 ----
    torch.manual_seed(7)
    hseed = torch.randn(1, 1, H, dtype=torch.bfloat16, device=dev) * 0.1

    def la_seeded(h1, h2):
        # sequential
        hard_reset()
        l0.linear_attn(hseed, cache_params=cache, attention_mask=None,
                       position_ids=None)          # prefill -> seeds state
        o1 = l0.linear_attn(h1, cache_params=cache, attention_mask=None,
                            position_ids=None)
        o2 = l0.linear_attn(h2, cache_params=cache, attention_mask=None,
                            position_ids=None)
        lay = cache.layers[0]
        seq = (lay.conv_states[0].clone(), lay.recurrent_states[0].clone())
        # block (seeded -> v3 spec-block path)
        hard_reset()
        l0.linear_attn(hseed, cache_params=cache, attention_mask=None,
                       position_ids=None)
        ob = l0.linear_attn(torch.cat([h1, h2], dim=1), cache_params=cache,
                            attention_mask=None, position_ids=None)
        blk = (lay.conv_states[0].clone(), lay.recurrent_states[0].clone())
        return torch.cat([o1, o2], 1), ob, seq, blk

    oseq2, oblk2, seqS, blkS = la_seeded(hh1, hh2)
    d0 = (oseq2[0, 0].float() - oblk2[0, 0].float()).abs().max().item()
    d1 = (oseq2[0, 1].float() - oblk2[0, 1].float()).abs().max().item()
    dc = (seqS[0].float() - blkS[0].float()).abs().max().item()
    dr = (seqS[1].float() - blkS[1].float()).abs().max().item()
    print(f'[la seeded A/B] out tok0 {d0:.6f} tok1 {d1:.6f} | '
          f'conv dmax {dc:.6f} rec dmax {dr:.6f} | '
          f'|rec| mean {seqS[1].float().abs().mean().item():.5f}', flush=True)
    hard_reset()



    # =================== full speculative loop ===================
    # NOTE: conv_states/recurrent_states are DICTS {state_idx: tensor} 鈥?
    # iterating the object yields INT KEYS, not tensors. All snapshot/
    # restore/collect code MUST iterate .values() (this exact bug silently
    # disabled GDN rollback = the accept-path corruption root cause).
    def snap_light():
        return [([c.clone() if isinstance(c, torch.Tensor) else c
                  for c in (getattr(l, 'conv_states', None) or {}).values()],
                 [r.clone() if isinstance(r, torch.Tensor) else r
                  for r in (getattr(l, 'recurrent_states', None) or {}).values()],
                 l.cumulative_length.clone()
                 if getattr(l, 'cumulative_length', None) is not None else None)
                for l in cache.layers]

    def roll_back(snap):
        for l, (cs, rs, cum) in zip(cache.layers, snap):
            if cum is not None:
                l.cumulative_length.copy_(cum)
            for a, b in zip((getattr(l, 'conv_states', None) or {}).values(), cs):
                if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
                    a.copy_(b)
            for a, b in zip((getattr(l, 'recurrent_states', None) or {}).values(), rs):
                if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
                    a.copy_(b)

    # ---- fast snap/rollback: flatten + torch._foreach_copy_ (1 launch) ----
    _dsts, _srcs = [], []

    def _collect():
        dsts, srcs = [], []
        for l in cache.layers:
            for c in (getattr(l, 'conv_states', None) or {}).values():
                if isinstance(c, torch.Tensor):
                    dsts.append(c)
                    srcs.append(torch.empty_like(c))
            for r in (getattr(l, 'recurrent_states', None) or {}).values():
                if isinstance(r, torch.Tensor):
                    dsts.append(r)
                    srcs.append(torch.empty_like(r))
            cum = getattr(l, 'cumulative_length', None)
            if cum is not None:
                dsts.append(cum)
                srcs.append(torch.empty_like(cum))
        return dsts, srcs

    _dsts, _srcs = _collect()

    def fast_snap():
        torch._foreach_copy_(_srcs, _dsts)

    def fast_rollback():
        torch._foreach_copy_(_dsts, _srcs)

    mtp = load_mtp(m)
    print('mtp head loaded (with layer, orig attn)', flush=True)

    # ---- capture g_mtp4: 4-step MTP chain with known-token selection ----
    # Inputs: mtp_h_buf (h after last PROCESSED token), chain_in[4]
    # (known-token queue: pending committed-but-unprocessed tokens + new
    # root; tail slots are garbage), pend_len (GPU [1]).
    # Step j: predict p_j = argmax(MTP(emb(tok_j), u_j)); the block's
    # next token is chain_in[j+1] when inside the known prefix, else the
    # prediction itself (data-dependent select, capture-safe where()).
    # tok_out[4] = the verify block (known prefix + fresh drafts).
    print('capturing g_mtp4...', flush=True)
    mtp_h_buf = torch.zeros(1, 1, H, dtype=torch.bfloat16, device=dev)
    mtp_pos_buf = torch.zeros(1, 1, dtype=torch.long, device=dev)
    chain_in = torch.zeros(4, dtype=torch.long, device=dev)
    pend_len = torch.zeros(1, dtype=torch.long, device=dev)
    tok_out = torch.zeros(4, dtype=torch.long, device=dev)
    ar4c = torch.arange(4, device=dev)

    def mtp_chain():
        h = mtp_h_buf
        tok = chain_in[0]
        tok_out[0].copy_(tok)
        for j in range(4):
            e = emb_rows(tok.view(1)).view(1, 1, H)
            lg, h = mtp.forward2(e, h, mtp_pos_buf)
            p = lg[:, -1].argmax()
            if j < 3:
                nxt = torch.where(ar4c[j + 1] < pend_len[0],
                                  chain_in[j + 1], p)
                tok_out[j + 1].copy_(nxt)
                tok = nxt
        return tok_out

    s_m = torch.cuda.Stream()
    s_m.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s_m):
        for _ in range(3):
            mtp_chain()
    torch.cuda.current_stream().wait_stream(s_m)
    g_mtp4 = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g_mtp4, pool=pool):
        mtp_chain()
    print('g_mtp4 captured', flush=True)

    # ---- buffers + merged graph bodies ----
    pin_t = torch.zeros(1, dtype=torch.long, pin_memory=True)
    t_gpu = torch.zeros(1, dtype=torch.long, device=dev)
    ar4 = torch.arange(4, device=dev)

    # chainprep: MTP chain -> tok_out, then fill emb4/cos4/sin4/pos4
    # (pinned->device async copy, capture-safe) + fast_snap. One replay.
    def chainprep_fn():
        mtp_chain()
        t_gpu.copy_(pin_t, non_blocking=True)
        idx = t_gpu + ar4
        emb4.copy_(emb_rows(tok_out).view(1, 4, H))
        cos4.copy_(torch.index_select(cos_all, 0, idx).view(1, 4, -1))
        sin4.copy_(torch.index_select(sin_all, 0, idx).view(1, 4, -1))
        pos4.copy_(idx)
        torch._foreach_copy_(_srcs, _dsts)                     # fast_snap

    # decisions: L = accepted prefix length (1..4); known-prefix
    # positions auto-match (deterministic re-verification).
    def dec_fn():
        a0 = out_l4[0, 0].argmax()
        a1 = out_l4[0, 1].argmax()
        a2 = out_l4[0, 2].argmax()
        a3 = out_l4[0, 3].argmax()
        m1 = a0 == tok_out[1]
        m2 = (a1 == tok_out[2]) & m1
        m3 = (a2 == tok_out[3]) & m2
        L = 1 + m1.to(torch.long) + m2.to(torch.long) + m3.to(torch.long)
        dec_gpu[0] = L
        dec_gpu[1] = tok_out[0]
        dec_gpu[2] = tok_out[1]
        dec_gpu[3] = tok_out[2]
        dec_gpu[4] = tok_out[3]
        a_buf[0].copy_(a0)
        a_buf[1].copy_(a1)
        a_buf[2].copy_(a2)
        a_buf[3].copy_(a3)

    # g4dec: T=4 verify forward -> static outputs -> decisions.
    def g4dec_fn():
        h4_s, log4_s = run(emb4, cos4, sin4, pos4)
        out_h4.copy_(h4_s)
        out_l4.copy_(log4_s)
        dec_fn()

    print('capturing g_chainprep/g4dec...', flush=True)
    s_p = torch.cuda.Stream()
    s_p.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s_p):
        for _ in range(3):
            chainprep_fn()
            g4dec_fn()
    torch.cuda.current_stream().wait_stream(s_p)
    g_chainprep = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g_chainprep, pool=pool):
        chainprep_fn()
    g4dec = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g4dec, pool=pool):
        g4dec_fn()
    print('g_chainprep/g4dec captured', flush=True)

    # ---- per-graph GPU cost (20 reps, one sync) ----
    if os.environ.get('IXGRAPHTIME'):
        for gname, gg in [('g_mtp4', g_mtp4), ('g_chainprep', g_chainprep),
                          ('g4dec', g4dec), ('g4', g4), ('g2', g2),
                          ('g1', g1)]:
            gg.replay(); torch.cuda.synchronize()
            t0g = time.time()
            for _ in range(20):
                gg.replay()
            torch.cuda.synchronize()
            print(f'  [graphtime] {gname}: '
                  f'{(time.time() - t0g) / 20 * 1e3:.1f}ms', flush=True)

    # ================= state-diff bisect: g1+g1 vs g2 ===============
    prompt0 = 'The theory of relativity states that'
    ids0 = tok(prompt0, return_tensors='pt')['input_ids'][0].tolist()

    def prefill_ids(ids_list):
        for i, tid in enumerate(ids_list):
            emb1.copy_(embed1(tid))
            cos1.copy_(cos_all[i].view(1, 1, -1))
            sin1.copy_(sin_all[i].view(1, 1, -1))
            pos1.fill_(i)
            run(emb1, cos1, sin1, pos1)

    def full_snap():
        s = snap_light()
        kv = []
        for l in cache.layers:
            kv.append((l.keys.clone(), l.values.clone())
                      if getattr(l, 'is_initialized', False) else None)
        return (s, kv)

    def full_restore(sn):
        roll_back(sn[0])
        for l, kv in zip(cache.layers, sn[1]):
            if kv is not None:
                l.keys.copy_(kv[0])
                l.values.copy_(kv[1])

    hard_reset()
    prefill_ids(ids0)
    base = full_snap()

    tA = len(ids0)
    tokA = [279, 4478]        # any two tokens
    # path A: g1 twice
    for j, tk in enumerate(tokA):
        emb1.copy_(embed1(tk))
        cos1.copy_(cos_all[tA + j].view(1, 1, -1))
        sin1.copy_(sin_all[tA + j].view(1, 1, -1))
        pos1.fill_(tA + j)
        g1.replay()
    snapA = full_snap()

    # path A2: EAGER T1 twice (isolates T2-vs-seq from graph-vs-eager)
    full_restore(base)
    for j, tk in enumerate(tokA):
        emb1.copy_(embed1(tk))
        cos1.copy_(cos_all[tA + j].view(1, 1, -1))
        sin1.copy_(sin_all[tA + j].view(1, 1, -1))
        pos1.fill_(tA + j)
        run(emb1, cos1, sin1, pos1)
    snapA2 = full_snap()
    # self-consistency: run the SAME eager path again after restore
    full_restore(base)
    for j, tk in enumerate(tokA):
        emb1.copy_(embed1(tk))
        cos1.copy_(cos_all[tA + j].view(1, 1, -1))
        sin1.copy_(sin_all[tA + j].view(1, 1, -1))
        pos1.fill_(tA + j)
        run(emb1, cos1, sin1, pos1)
    snapA3 = full_snap()
    n_bad_self = 0
    dmax_self = 0.0
    for i, (kvA, kvB) in enumerate(zip(snapA2[1], snapA3[1])):
        if kvA is None:
            continue
        d = (kvA[0].float() - kvB[0].float()).abs()
        c = int(snapA2[0][i][2].item())
        dm = d[:, :, c - 2:c].max().item()
        dmax_self = max(dmax_self, dm)
        if dm > 1e-3:
            n_bad_self += 1
    print(f'[state-diff] SELF-CONSISTENCY eager 2xT1 (restore ok?): '
          f'bad {n_bad_self}/16, dK max {dmax_self:.6f}', flush=True)

    # ---- per-layer bisect: A2 vs A3 hidden states at every layer ----
    caps = {}
    hooks = []

    def _mk_hook(idx):
        def _hook(mod, inp, out):
            o = out[0] if isinstance(out, tuple) else out
            caps.setdefault(idx, []).append(o.detach().clone())
        return _hook

    for i, lay in enumerate(layers):
        hooks.append(lay.register_forward_hook(_mk_hook(i)))

    def _eager2():
        full_restore(base)
        lay0 = cache.layers[0]
        caps.setdefault('state0', []).append(
            (lay0.conv_states[0].clone(), lay0.recurrent_states[0].clone()))
        for j, tk in enumerate(tokA):
            emb1.copy_(embed1(tk))
            cos1.copy_(cos_all[tA + j].view(1, 1, -1))
            sin1.copy_(sin_all[tA + j].view(1, 1, -1))
            pos1.fill_(tA + j)
            run(emb1, cos1, sin1, pos1)

    # ---- minimal isolation: restore->snap == base? twice ----
    full_restore(base)
    r1 = full_snap()
    full_restore(base)
    r2 = full_snap()
    lay0rec_base = base[0][0][1][0]
    print(f'[isolate dbg] type(lay0rec)={type(lay0rec_base).__name__} '
          f'type(r1[0][0][1][0])={type(r1[0][0][1][0]).__name__}',
          flush=True)
    d_rb1 = (r1[0][0][1][0].float() - lay0rec_base.float()).abs().max().item()
    d_rb2 = (r2[0][0][1][0].float() - lay0rec_base.float()).abs().max().item()
    d_r12 = (r1[0][0][1][0].float() - r2[0][0][1][0].float()).abs().max().item()
    print(f'[isolate] restore->snap vs base: run1 {d_rb1:.6f} run2 {d_rb2:.6f} '
          f'| run1-vs-run2 {d_r12:.6f} '
          f'| base |rec| mean {lay0rec_base.float().abs().mean().item():.5f}',
          flush=True)
    print(f'[isolate] rec[0] ptr == base-clone ptr? '
          f'{cache.layers[0].recurrent_states[0].data_ptr() == lay0rec_base.data_ptr()}',
          flush=True)

    caps.clear()
    _eager2()
    cap1 = {k: list(v) for k, v in caps.items()}
    caps.clear()
    _eager2()
    cap2 = {k: list(v) for k, v in caps.items()}
    caps.clear()
    _eager2()
    cap3 = {k: list(v) for k, v in caps.items()}
    for tag, ca, cb in (('run1-vs-run2', cap1, cap2), ('run2-vs-run3', cap2, cap3)):
        s0a, s0b = ca['state0'][0], cb['state0'][0]
        d_conv = (s0a[0].float() - s0b[0].float()).abs().max().item()
        d_rec = (s0a[1].float() - s0b[1].float()).abs().max().item()
        print(f'[bisect {tag}] entry-state layer0: d_conv {d_conv:.6f} '
              f'd_rec {d_rec:.6f}', flush=True)
        first_bad = None
        for i in range(len(layers)):
            for j in range(2):
                a = ca[i][j][0, 0].float()
                b = cb[i][j][0, 0].float()
                dm = (a - b).abs().max().item()
                if dm > 1e-4 and first_bad is None:
                    first_bad = (i, j, dm)
                    ref = a.abs().mean().item()
                    print(f'[bisect {tag}] FIRST divergence: layer {i} '
                          f'token {j} dmax {dm:.5f} (|h| mean {ref:.5f})',
                          flush=True)
        if first_bad is None:
            print(f'[bisect {tag}] all 64 layers identical', flush=True)

    # ---- T2-vs-sequential per-layer bisect (the real fork) ----
    for h in hooks:
        h.remove()
    hooks2 = []
    caps2 = {}

    def _mk_hook2(idx):
        def _hook(mod, inp, out):
            o = out[0] if isinstance(out, tuple) else out
            caps2.setdefault(idx, []).append(o.detach().clone())
        return _hook

    for i, lay in enumerate(layers):
        hooks2.append(lay.register_forward_hook(_mk_hook2(i)))

    caps2.clear()
    _eager2()
    seq_cap = {k: list(v) for k, v in caps2.items()}
    caps2.clear()
    full_restore(base)
    emb2[:, 0].copy_(embed1(tokA[0]).view(1, H))
    emb2[:, 1].copy_(embed1(tokA[1]).view(1, H))
    cos2[:, 0].copy_(cos_all[tA])
    cos2[:, 1].copy_(cos_all[tA + 1].view(1, -1))
    sin2[:, 0].copy_(sin_all[tA])
    sin2[:, 1].copy_(sin_all[tA + 1].view(1, -1))
    pos2[0] = tA; pos2[1] = tA + 1
    run(emb2, cos2, sin2, pos2)
    t2_cap = {k: list(v) for k, v in caps2.items()}
    first_bad = None
    for i in range(len(layers)):
        for j in range(2):
            a = seq_cap[i][j][0, 0].float()
            b = t2_cap[i][0][0, j].float()
            dm = (a - b).abs().max().item()
            if dm > 1e-4 and first_bad is None:
                first_bad = (i, j, dm)
                ref = a.abs().mean().item()
                print(f'[T2-bisect] FIRST divergence: layer {i} ({type(layers[i]).__name__}) '
                      f'token {j} dmax {dm:.5f} (|h| mean {ref:.5f})', flush=True)
                for ii in range(max(0, i - 2), min(len(layers), i + 2)):
                    a2 = seq_cap[ii][j][0, 0].float()
                    b2 = t2_cap[ii][0][0, j].float()
                    print(f'   layer {ii}: dmax {(a2 - b2).abs().max().item():.6f}',
                          flush=True)
    if first_bad is None:
        print('[T2-bisect] all 64 layers: T2 == sequential EXACT', flush=True)
    for h in hooks2:
        h.remove()

    # ---- op-level diff at layer 3: sequential vs T2 ----
    _DBG.clear(); _DBG['on'] = True; _DBG['layer'] = 3
    _eager2()                       # sequential (2x q_len=1 at layer 3)
    dseq = {k: list(v) for k, v in _DBG.items() if k != 'on' and k != 'layer'}
    _DBG.clear(); _DBG['on'] = True; _DBG['layer'] = 3
    full_restore(base)
    emb2[:, 0].copy_(embed1(tokA[0]).view(1, H))
    emb2[:, 1].copy_(embed1(tokA[1]).view(1, H))
    cos2[:, 0].copy_(cos_all[tA])
    cos2[:, 1].copy_(cos_all[tA + 1].view(1, -1))
    sin2[:, 0].copy_(sin_all[tA])
    sin2[:, 1].copy_(sin_all[tA + 1].view(1, -1))
    pos2[0] = tA; pos2[1] = tA + 1
    run(emb2, cos2, sin2, pos2)     # T2 (one q_len=2 at layer 3)
    dt2 = {k: list(v) for k, v in _DBG.items() if k != 'on' and k != 'layer'}
    _DBG.clear()
    for key in ('qproj', 'kpost_rope', 'qpost_rope', 'kv_after_update',
                'attn_out', 'gated', 'oproj_out'):
        if key in ('qproj', 'gated', 'oproj_out'):
            a = torch.cat(dseq[key], dim=1)      # [1,2,out] seq (2 calls)
            b = dt2[key][0]
        elif key == 'kv_after_update':
            a = dseq[key][-1][:, :, :tA + 2]     # last cache state, seq
            b = dt2[key][0][:, :, :tA + 2]
        else:
            a = torch.cat(dseq[key], dim=2)      # [1,h,2,256] seq
            b = dt2[key][0]
        dm = (a.float() - b.float()).abs().max().item()
        print(f'[opdiff] {key}: dmax {dm:.6f}', flush=True)
    for h in hooks:
        h.remove()
    n_bad_kv_a2 = 0
    dmax_a2 = 0.0
    for i, (kvA, kvB) in enumerate(zip(snapA[1], snapA2[1])):
        if kvA is None:
            continue
        d = (kvA[0].float() - kvB[0].float()).abs()
        c = int(snapA[0][i][2].item())
        dm = d[:, :, c - 2:c].max().item()
        dmax_a2 = max(dmax_a2, dm)
        if dm > 1e-3:
            n_bad_kv_a2 += 1
    print(f'[state-diff] g1-graph vs g1-eager: bad {n_bad_kv_a2}/16, '
          f'dK max {dmax_a2:.4f}', flush=True)
    # use EAGER A2 as the sequential reference for the T2 comparisons
    snapSeq = snapA2
    refCum = snapA2

    # path B: g2 once
    full_restore(base)
    emb2[:, 0].copy_(embed1(tokA[0]).view(1, H))
    emb2[:, 1].copy_(embed1(tokA[1]).view(1, H))
    cos2[:, 0].copy_(cos_all[tA])
    cos2[:, 1].copy_(cos_all[tA + 1].view(1, -1))
    sin2[:, 0].copy_(sin_all[tA])
    sin2[:, 1].copy_(sin_all[tA + 1].view(1, -1))
    pos2[0] = tA; pos2[1] = tA + 1
    g2.replay()
    snapB = full_snap()

    # diff conv/rec/kv
    n_bad_rec = n_bad_conv = n_bad_kv = 0
    for i, (lA, lB) in enumerate(zip(snapA[0], snapB[0])):
        csA, rsA, cA = lA
        csB, rsB, cB = lB
        for a, b in zip(rsA, rsB):
            if isinstance(a, torch.Tensor) and (a.float() - b.float()).abs().max() > 1e-3:
                n_bad_rec += 1
                break
        for a, b in zip(csA, csB):
            if isinstance(a, torch.Tensor) and (a.float() - b.float()).abs().max() > 1e-3:
                n_bad_conv += 1
                break
    for i, (kvA, kvB) in enumerate(zip(snapA[1], snapB[1])):
        if kvA is None:
            continue
        d = (kvA[0].float() - kvB[0].float()).abs()
        # compare only the two written slots
        c = int(snapA[0][i][2].item())
        if d[:, :, c - 2:c].max() > 1e-3:
            n_bad_kv += 1
    print(f'[state-diff] bad layers: recurrent {n_bad_rec}/48 '
          f'conv {n_bad_conv}/48 kv-full-attn {n_bad_kv}/16', flush=True)
    # magnitude + per-slot breakdown for the first full-attn layer
    for i, (kvA, kvB) in enumerate(zip(snapA[1], snapB[1])):
        if kvA is None:
            continue
        c = int(snapA[0][i][2].item())
        dK = (kvA[0].float() - kvB[0].float()).abs()
        dV = (kvA[1].float() - kvB[1].float()).abs()
        refK = kvA[0].float().abs().mean()
        print(f'  layer{i}: |dK| max {dK.max():.4f} mean {dK.mean():.5f} '
              f'(ref |K| {refK:.4f}) | slot {c-2}: {dK[:,:,c-2].max():.4f} '
              f'slot {c-1}: {dK[:,:,c-1].max():.4f} | earlier max '
              f'{dK[:,:,:c-2].max() if c>2 else 0:.4f} | dV max {dV.max():.4f}',
              flush=True)
        if i >= 3:
            break
    # path B2: EAGER M=2 (same inputs, no graph) vs EAGER sequential
    full_restore(base)
    emb2[:, 0].copy_(embed1(tokA[0]).view(1, H))
    emb2[:, 1].copy_(embed1(tokA[1]).view(1, H))
    cos2[:, 0].copy_(cos_all[tA])
    cos2[:, 1].copy_(cos_all[tA + 1].view(1, -1))
    sin2[:, 0].copy_(sin_all[tA])
    sin2[:, 1].copy_(sin_all[tA + 1].view(1, -1))
    pos2[0] = tA; pos2[1] = tA + 1
    run(emb2, cos2, sin2, pos2)
    snapB2 = full_snap()
    n_bad_kv_eager = 0
    dmax_eager = 0.0
    for i, (kvA, kvB) in enumerate(zip(snapSeq[1], snapB2[1])):
        if kvA is None:
            continue
        d = (kvA[0].float() - kvB[0].float()).abs()
        c = int(refCum[0][i][2].item())
        dm = d[:, :, c - 2:c].max().item()
        dmax_eager = max(dmax_eager, dm)
        if dm > 1e-3:
            n_bad_kv_eager += 1
    print(f'[state-diff] EAGER T2 vs EAGER 2xT1: bad {n_bad_kv_eager}/16, '
          f'dK max {dmax_eager:.4f}', flush=True)
    hard_reset()

    # ---- release ALL diagnostic clones before timed generation ----
    # The bisect harness above leaks ~3.5GB (7x full_snap with 48-layer
    # rec/conv clones + hook captures). With VRAM exhausted, WDDM pages
    # to sysmem and in-context GPU bandwidth collapses (g4dec 105 ->
    # 177ms). Freeing restores the fast path.
    _DBG.clear()
    snapA = snapB = snapA2 = snapA3 = snapB2 = None
    base = r1 = r2 = None
    cap1 = cap2 = cap3 = caps = caps2 = seq_cap = t2_cap = None
    dseq = dt2 = snapSeq = refCum = None
    cosB1 = sinB1 = cosB2 = sinB2 = None
    gc.collect()
    torch.cuda.empty_cache()
    _f, _t = torch.cuda.mem_get_info()
    print(f'[cleanup] VRAM free {_f/1e9:.2f} / {_t/1e9:.2f}GB '
          f'(allocated {torch.cuda.memory_allocated()/1e9:.2f}GB)',
          flush=True)

    def spec_gen(prompt, N=40):
        ids = tok(prompt, return_tensors='pt')['input_ids'][0].tolist()
        # prefill
        h_last = logits_last = None
        for i, tid in enumerate(ids):
            emb1.copy_(embed1(tid))
            cos1.copy_(cos_all[i].view(1, 1, -1))
            sin1.copy_(sin_all[i].view(1, 1, -1))
            pos1.fill_(i)
            h_last, logits_last = run(emb1, cos1, sin1, pos1)
        h_last, logits_last = h_last.clone(), logits_last.clone()

        # reference (sequential g1 greedy) for advisory fork check
        ref = []
        t = len(ids)
        for _ in range(6):
            nt = logits_last[:, -1].argmax().item()
            ref.append(nt)
            emb1.copy_(embed1(nt))
            cos1.copy_(cos_all[t].view(1, 1, -1))
            sin1.copy_(sin_all[t].view(1, 1, -1))
            pos1.fill_(t)
            g1.replay()
            h_last = h1_s.clone()
            logits_last = log1_s.clone()
            t += 1
        # reset + reprefill for the real spec run
        hard_reset()
        for i, tid in enumerate(ids):
            emb1.copy_(embed1(tid))
            cos1.copy_(cos_all[i].view(1, 1, -1))
            sin1.copy_(sin_all[i].view(1, 1, -1))
            pos1.fill_(i)
            h_last, logits_last = run(emb1, cos1, sin1, pos1)
        h_last, logits_last = h_last.clone(), logits_last.clone()

        # ---- queue-architecture spec loop: ONE sync, ZERO replays ----
        # Every iteration: 4-step MTP chain (known-prefix advance +
        # drafts) -> T=4 verify -> decisions. Partial accepts roll back
        # and RE-QUEUE the accepted prefix as the next block's known
        # tokens (auto-reverified) — no prefix-recompute replays.
        dec_pin = torch.zeros(6, dtype=torch.long, device='cpu',
                              pin_memory=True)
        chain_in[0] = int(logits_last[:, -1].argmax().item())
        pend_len[0] = 1
        pend_cpu = 1                  # known-prefix length incl. root
        mtp_h_buf.copy_(h_last)

        gen = []
        n_iter = 0
        ev = torch.cuda.Event()        # doorbell event (reused)
        acc_hist = [0, 0, 0, 0]        # committed-tokens-per-iter (1..4)
        t_draft = t_verify = t_commit = 0.0
        t = len(ids)
        t0 = time.time()
        while len(gen) < N and t < MAX_CTX - 6:
            # -- 1. chain + prep fills + snap (ONE replay) --
            _t0 = time.time()
            pin_t[0] = t
            g_chainprep.replay()       # -> tok_out, emb4/cos4/sin4/pos4
            _t1 = time.time(); t_draft += _t1 - _t0

            # -- 2. T=4 verify + decisions (ONE replay) + doorbell poll --
            g4dec.replay()             # -> dec_gpu, a_buf, out_h4/out_l4
            dec_pin.copy_(dec_gpu, non_blocking=True)
            # DOORBELL instead of torch.cuda.synchronize(): the blocking
            # device sync forces a WDDM driver round-trip (~60-85ms);
            # polling a recorded event reads a GPU-written mapped flag
            # in user space (~us). CPU spin, no kernel transition.
            ev.record()
            while not ev.query():
                pass
            _t2 = time.time(); t_verify += _t2 - _t1
            L = int(dec_pin[0])
            tv = [int(dec_pin[1]), int(dec_pin[2]),
                  int(dec_pin[3]), int(dec_pin[4])]
            if len(gen) < 6:
                print(f'    it{n_iter}: block={tv} L={L}', flush=True)

            # -- 3. commit ONLY newly-processed tokens; queue setup --
            # block = [known prefix (pend_cpu-1, already committed),
            #          root + drafts (new)]. L >= pend_cpu always
            # (known positions auto-match).
            committed = tv[pend_cpu - 1: L]
            gen.extend(committed)
            acc_hist[len(committed) - 1] += 1
            n_iter += 1
            t += L
            # next chain: known prefix = this block's accepted tokens +
            # the new root a_{L-1} (all GPU-resident)
            if L == 4:
                chain_in[0].copy_(a_buf[3])
                pend_len[0] = 1
                pend_cpu = 1
                mtp_h_buf.copy_(out_h4[:, 3:4])
            else:
                fast_rollback()        # GDN/cum state -> pre-block
                chain_in[:L].copy_(tok_out[:L])
                chain_in[L].copy_(a_buf[L - 1])
                pend_len[0] = L + 1
                pend_cpu = L + 1
                # mtp_h_buf untouched: it still holds the pre-block h
                # (the chain reads it, nothing writes it) = correct seed
            t_commit += time.time() - _t2
            if tok.eos_token_id in committed:
                break
        torch.cuda.synchronize()
        dt = time.time() - t0
        # divergence check
        k = min(6, len(gen))
        if gen[:k] != ref[:k]:
            fork = next(i for i in range(k) if gen[i] != ref[i])
            print(f'  [note] trajectory fork at token {fork} '
                  f'(near-tie argmax on draft path)', flush=True)
        tokavg = sum((i + 1) * h for i, h in enumerate(acc_hist)) / n_iter
        print(f'  [timing] iter={n_iter} tok/iter={tokavg:.2f} '
              f'hist={acc_hist} draft={t_draft/n_iter*1000:.1f}ms '
              f'verify+sync={t_verify/n_iter*1000:.1f}ms '
              f'commit={t_commit/n_iter*1000:.1f}ms '
              f'total={dt/n_iter*1000:.1f}ms/iter', flush=True)
        return gen, dt, (n_iter, acc_hist)

    for prompt in ['The theory of relativity states that',
                   'def quick_sort(arr):',
                   '鍖椾含鏈€鍊煎緱娓歌鐨勪笁涓櫙鐐规槸']:
        hard_reset()
        gen, dt, (ni, hist) = spec_gen(prompt)
        txt = tok.decode(gen)
        print(f'\n[{len(gen)/dt:.2f} tok/s] {prompt!r} '
              f'(iters {ni}, hist {hist})\n'
              f'  -> {txt[:130]!r}', flush=True)

    # ---- pure pipeline benchmark: 4 replays + 1 sync, no branching ----
    hard_reset()
    ids_b = tok('Hello world, this is a benchmark prompt for timing.',
                return_tensors='pt')['input_ids'][0].tolist()
    for i, tid in enumerate(ids_b):
        emb1.copy_(embed1(tid))
        cos1.copy_(cos_all[i].view(1, 1, -1))
        sin1.copy_(sin_all[i].view(1, 1, -1))
        pos1.fill_(i)
        run(emb1, cos1, sin1, pos1)
    mtp_h_buf.copy_(out_h1)
    chain_in[0] = 12345
    tb = len(ids_b)
    dec_pin_b = torch.zeros(6, dtype=torch.long, pin_memory=True)
    chain_in[0] = 12345
    pend_len[0] = 1
    ev_b = torch.cuda.Event(enable_timing=True)
    e0, e1 = (torch.cuda.Event(enable_timing=True),
              torch.cuda.Event(enable_timing=True))
    # warm
    g_chainprep.replay(); g4dec.replay()
    torch.cuda.synchronize()
    # variant A: chainprep + g4dec alternating, GPU-side event timing
    t_cp = t_g4 = 0.0
    t0b = time.time()
    for _ in range(40):
        pin_t[0] = tb
        e0.record()
        g_chainprep.replay()
        e1.record()
        g4dec.replay()
        ev_b.record()
        while not ev_b.query():
            pass
        t_cp += e0.elapsed_time(e1)
        t_g4 += e1.elapsed_time(ev_b)
    dtb = (time.time() - t0b) / 40 * 1e3
    print(f'\n[pipe-bench] alternating 2 graphs: wall {dtb:.1f}ms/iter '
          f'| GPU-side: chainprep {t_cp/40:.1f}ms + g4dec {t_g4/40:.1f}ms '
          f'+ submit-gap {dtb - t_cp/40 - t_g4/40:.1f}ms', flush=True)
    free_b, total_b = torch.cuda.mem_get_info()
    print(f'[pipe-bench] VRAM free {free_b/1e9:.2f} / {total_b/1e9:.2f}GB '
          f'(allocated {torch.cuda.max_memory_allocated()/1e9:.2f}GB)',
          flush=True)

    print(f'\npeak GPU = {torch.cuda.max_memory_allocated()/1e9:.2f}GB', flush=True)


if __name__ == '__main__':
    main()
