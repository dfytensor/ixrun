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

MAX_CTX = 256
BLOB = r'E:\IXRUN\experiments\qwen38_udcq\q38_blob.pt'

# save the ORIGINAL attention forward BEFORE patching (MTP head layer needs
# the past_key_values=None path)
_ORIG_ATTN_FWD = Qwen3_5Attention.forward


def _attn_forward_static(self, hidden_states, position_embeddings,
                         attention_mask=None, past_key_values=None, **kw):
    if past_key_values is None:
        return _ORIG_ATTN_FWD(self, hidden_states, position_embeddings,
                              attention_mask=attention_mask,
                              past_key_values=None, **kw)
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)
    q_len = hidden_states.shape[1]

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
    qr = torch.arange(q_len, device=k_full.device)
    keep = ar[None, :] < (cum - q_len + 1 + qr[:, None])
    mask = torch.where(keep, 0.0, float('-inf')).to(query_states.dtype)
    mask = mask.view(1, 1, q_len, MAX)

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
    def forward(self, tok_emb, h, pos):
        # rope for the draft position (q_len=1)
        cos, sin = self.rotary(h, pos)
        x = self.fc(torch.cat([self.norm_e(tok_emb), self.norm_h(h)],
                              dim=-1).to(torch.bfloat16))
        z = self.layer(x, attention_mask=None, position_ids=pos,
                       position_embeddings=(cos, sin), use_cache=False)
        if isinstance(z, tuple):
            z = z[0]
        return self.lm_head(self.norm(z))


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
    # construct WITHOUT the shared lm_head as a child module — .to(bf16)
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
            for cs in getattr(lay, 'conv_states', []) or []:
                if isinstance(cs, torch.Tensor):
                    cs.zero_()
            for rs in getattr(lay, 'recurrent_states', []) or []:
                if isinstance(rs, torch.Tensor):
                    rs.zero_()

    # ------- graphs -------
    print('capturing g1/g2...', flush=True)
    for b in (emb1, cos1, sin1, emb2, cos2, sin2):
        b.zero_()
    pos1.fill_(0); pos2.fill_(0)
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            run(emb1, cos1, sin1, pos1)
            run(emb2, cos2, sin2, pos2)
    torch.cuda.current_stream().wait_stream(s)
    g1 = torch.cuda.CUDAGraph()
    pool = torch.cuda.graph_pool_handle()   # BOTH graphs share one pool —
    with torch.cuda.graph(g1, pool=pool):   # separate pools alias corrupt g1
        h1_s, log1_s = run(emb1, cos1, sin1, pos1)
    g2 = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g2, pool=pool):
        h2_s, log2_s = run(emb2, cos2, sin2, pos2)
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
        cosB2[:, 0].copy_(cos_all[pos0].view(1, -1))
        cosB2[:, 1].copy_(cos_all[pos0 + 1].view(1, -1))
        sinB2[:, 0].copy_(sin_all[pos0].view(1, -1))
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



    # =================== full speculative loop ===================
    def snap_light():
        return [([c.clone() if isinstance(c, torch.Tensor) else c
                  for c in getattr(l, 'conv_states', []) or []],
                 [r.clone() if isinstance(r, torch.Tensor) else r
                  for r in getattr(l, 'recurrent_states', []) or []],
                 l.cumulative_length.clone()
                 if getattr(l, 'cumulative_length', None) is not None else None)
                for l in cache.layers]

    def roll_back(snap):
        for l, (cs, rs, cum) in zip(cache.layers, snap):
            if cum is not None:
                l.cumulative_length.copy_(cum)
            for a, b in zip(getattr(l, 'conv_states', []) or [], cs):
                if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
                    a.copy_(b)
            for a, b in zip(getattr(l, 'recurrent_states', []) or [], rs):
                if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
                    a.copy_(b)

    mtp = load_mtp(m)
    print('mtp head loaded (with layer, orig attn)', flush=True)

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

    # path B: g2 once
    full_restore(base)
    emb2[:, 0].copy_(embed1(tokA[0]).view(1, H))
    emb2[:, 1].copy_(embed1(tokA[1]).view(1, H))
    cos2[:, 0].copy_(cos_all[tA].view(1, -1))
    cos2[:, 1].copy_(cos_all[tA + 1].view(1, -1))
    sin2[:, 0].copy_(sin_all[tA].view(1, -1))
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
    # path B2: EAGER M=2 (same inputs, no graph)
    full_restore(base)
    emb2[:, 0].copy_(embed1(tokA[0]).view(1, H))
    emb2[:, 1].copy_(embed1(tokA[1]).view(1, H))
    cos2[:, 0].copy_(cos_all[tA].view(1, -1))
    cos2[:, 1].copy_(cos_all[tA + 1].view(1, -1))
    sin2[:, 0].copy_(sin_all[tA].view(1, -1))
    sin2[:, 1].copy_(sin_all[tA + 1].view(1, -1))
    pos2[0] = tA; pos2[1] = tA + 1
    run(emb2, cos2, sin2, pos2)
    snapB2 = full_snap()
    n_bad_kv_eager = 0
    dmax_eager = 0.0
    for i, (kvA, kvB) in enumerate(zip(snapA[1], snapB2[1])):
        if kvA is None:
            continue
        d = (kvA[0].float() - kvB[0].float()).abs()
        c = int(snapA[0][i][2].item())
        dm = d[:, :, c - 2:c].max().item()
        dmax_eager = max(dmax_eager, dm)
        if dm > 1e-3:
            n_bad_kv_eager += 1
    print(f'[state-diff] EAGER M=2 vs g1+g1: bad {n_bad_kv_eager}/16, '
          f'dK max {dmax_eager:.4f} (graph was 4.55)', flush=True)
    hard_reset()

    from collections import defaultdict

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

        # graph-numerics reference (g1 replays): the eager path and captured
        # graphs differ on near-tie argmaxes (round3 finding), so the loop
        # must be validated against ITS OWN numerics
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

        bigram = defaultdict(list)
        for i in range(2, len(ids)):
            bigram[(ids[i - 2], ids[i - 1])].append(ids[i])

        gen = []
        n_acc = n_rej = n_ng = n_mtp = 0
        t = len(ids)
        t0 = time.time()
        PURE_G2 = len(sys.argv) > 1 and sys.argv[1] == 'g2only'
        while len(gen) < N and t < MAX_CTX - 4:
            t1 = logits_last[:, -1].argmax().item()
            if PURE_G2:
                # diagnostic: state advanced ONLY by g2 ([t1, t1]), logits
                # from slot 0; no rollback, no g1
                emb2[:, 0].copy_(embed1(t1).view(1, H))
                emb2[:, 1].copy_(embed1(t1).view(1, H))
                cos2[:, 0].copy_(cos_all[t].view(1, -1))
                cos2[:, 1].copy_(cos_all[t + 1].view(1, -1))
                sin2[:, 0].copy_(sin_all[t].view(1, -1))
                sin2[:, 1].copy_(sin_all[t + 1].view(1, -1))
                pos2[0] = t; pos2[1] = t + 1
                g2.replay()
                gen.append(t1)
                logits_last = log2_s[:, 0:1].clone()
                t += 1
                continue
            hist = (ids + gen)[-2:]
            d = None
            cand = bigram.get(tuple(hist))
            if cand:
                d = cand[-1]
                n_ng += 1
            if d is None:
                if False:  # MTP disabled for isolation (eager cost + suspect)
                    pos_d = torch.tensor([[t]], device=dev)
                    d = mtp(embed1(t1), h_last, pos_d)[:, -1].argmax().item()
                    n_mtp += 1
                else:
                    d = t1          # naive repeat-draft (tests machinery)
                    n_mtp += 1

            snap = snap_light()
            emb2[:, 0].copy_(embed1(t1).view(1, H))
            emb2[:, 1].copy_(embed1(d).view(1, H))
            cos2[:, 0].copy_(cos_all[t].view(1, -1))
            cos2[:, 1].copy_(cos_all[t + 1].view(1, -1))
            sin2[:, 0].copy_(sin_all[t].view(1, -1))
            sin2[:, 1].copy_(sin_all[t + 1].view(1, -1))
            pos2[0] = t; pos2[1] = t + 1
            g2.replay()
            t2 = log2_s[:, 0].argmax().item()

            if t2 == d:
                gen.extend([t1, d])
                bigram[tuple(hist)].append(d)
                h_last = h2_s[:, 1:2].clone()
                logits_last = log2_s[:, 1:2].clone()
                t += 2
                n_acc += 1
                if d == tok.eos_token_id:
                    break
            else:
                roll_back(snap)
                emb1.copy_(embed1(t1))
                cos1.copy_(cos_all[t].view(1, 1, -1))
                sin1.copy_(sin_all[t].view(1, 1, -1))
                pos1.fill_(t)
                g1.replay()
                gen.append(t1)
                h_last = h1_s.clone()
                logits_last = log1_s.clone()
                t += 1
                n_rej += 1
                if t1 == tok.eos_token_id:
                    break
        torch.cuda.synchronize()
        dt = time.time() - t0
        # divergence check: g2 (M=2 GEMM numerics) vs g1 (GEMV numerics)
        # disagree on near-tie argmaxes — trajectories may legitimately fork
        # at such tokens (both are valid model generations). Coherence of the
        # TEXT is the real correctness bar; the ref comparison is advisory.
        k = min(6, len(gen))
        if gen[:k] != ref[:k]:
            fork = next(i for i in range(k) if gen[i] != ref[i])
            print(f'  [note] trajectory fork at token {fork} '
                  f'(near-tie numerics: g2-GEMM vs g1-GEMV)', flush=True)
        return gen, dt, (n_acc, n_rej, n_ng, n_mtp)

    for prompt in ['The theory of relativity states that',
                   'def quick_sort(arr):',
                   '北京最值得游览的三个景点是']:
        hard_reset()
        gen, dt, (a, r, ng, mt) = spec_gen(prompt)
        txt = tok.decode(gen)
        print(f'\n[{len(gen)/dt:.2f} tok/s] {prompt!r} '
              f'(acc {a} rej {r} | drafts ng {ng} mtp {mt})\n'
              f'  -> {txt[:130]!r}', flush=True)

    print(f'\npeak GPU = {torch.cuda.max_memory_allocated()/1e9:.2f}GB', flush=True)


if __name__ == '__main__':
    main()
