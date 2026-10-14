"""Batched SELF-FEEDING decode graph v3 — the RIGHT-PAD variant.

The v2 lesson: the LEFT-pad requires the pad-blocking attention masks
everywhere (the prefill 4-D, the decode growing) — and the explicit
masks diverge from is_causal in this stack (the layer-0 explosion).
The v3: the RIGHT-pad makes the PREFILL mask-free (the causal hides
the pads naturally: the query i attends [0, i], the pads at [> i] are
beyond = invisible under is_causal), and the per-row decode slots via
the cache_position [B] tensor (the 5.15 StaticCache supports the
per-row advanced indexing).  The decode mask = the per-row growing
open-window via the in-graph scatter (the capture-safe).

The audit: gen_buf [B, cap], n_gen [B], eos_hit [B], nxt [B] — the
same as v2.
"""
import threading

import torch

__all__ = ["BatchedGraphEngineV3", "CohortRequest"]


class CohortRequest:
    __slots__ = ("ids", "max_new", "temp", "top_p", "top_k",
                 "rep", "pres", "freq", "out_q", "done")

    def __init__(self, ids, max_new, temp, top_p, top_k,
                 rep, pres, freq):
        self.ids = ids
        self.max_new = max_new
        self.temp = temp
        self.top_p = top_p
        self.top_k = top_k
        self.rep = rep
        self.pres = pres
        self.freq = freq
        self.out_q = _queue.Queue()
        self.done = threading.Event()


import queue as _queue  # noqa: E402


class BatchedGraphEngineV3:
    def __init__(self, model_raw, cfg, tokenizer, B: int = 8,
                 max_new_cap: int = 512, max_ctx: int = 8192,
                 verbose: bool = True):
        self.m = model_raw
        self.cfg = cfg
        self.tok = tokenizer
        self.B = B
        self.max_new_cap = max_new_cap
        self.max_ctx = max_ctx
        self.verbose = verbose
        dev = 'cuda'
        V = getattr(cfg, 'vocab_size', 128256)
        self.V = V

        self.in_ids = torch.zeros(B, 1, dtype=torch.long, device=dev)
        self.nxt = torch.zeros(B, dtype=torch.long, device=dev)
        self.gen_buf = torch.zeros(B, max_new_cap, dtype=torch.long,
                                   device=dev)
        self.temp_g = torch.ones(B, dtype=torch.float32, device=dev)
        self.top_p_g = torch.ones(B, dtype=torch.float32, device=dev)
        self.top_k_g = torch.zeros(B, dtype=torch.long, device=dev)
        self.rep_g = torch.ones(B, dtype=torch.float32, device=dev)
        self.pres_g = torch.zeros(B, dtype=torch.float32, device=dev)
        self.freq_g = torch.zeros(B, dtype=torch.float32, device=dev)
        self.n_gen = torch.zeros(B, dtype=torch.long, device=dev)
        self.eos_hit = torch.zeros(B, dtype=torch.float32, device=dev)
        self.max_new_g = torch.full((B,), max_new_cap, dtype=torch.long,
                                    device=dev)
        self.eos_mask = torch.zeros(V, dtype=torch.float32, device=dev)
        self.row_idx = torch.arange(B, device=dev)
        self.sub_ar = torch.arange(1024, device=dev)
        self.pos_buf = torch.zeros(B, 1, dtype=torch.long, device=dev)
        # the per-row decode mask [B, 1, 1, max_ctx]: 0 = attend,
        # -1e4 = blocked; the row's column n_gen[r] opens per step
        self.mbuf = torch.full((B, 1, 1, max_ctx), -1e4,
                               dtype=torch.bfloat16, device=dev)
        # the per-row cache_position [B]
        self.cpos_buf = torch.zeros(B, dtype=torch.long, device=dev)

        from transformers.cache_utils import StaticCache
        self.bcache = StaticCache(config=cfg, max_batch_size=B,
                                  max_cache_len=max_ctx)

        self.sgraph = None
        self.busy = threading.Lock()

    # ------------------------------------------------------------------
    def set_eos(self, eos_ids):
        self.eos_mask.zero_()
        for e in eos_ids:
            if 0 <= e < self.eos_mask.numel():
                self.eos_mask[e] = 1.0

    # ------------------------------------------------------------------
    @torch.no_grad()
    def _samp_body(self):
        # the row's mask column n_gen[r] opens (the newest own slot)
        z = torch.zeros(self.B, 1, 1, 1, dtype=torch.bfloat16,
                        device=self.in_ids.device)
        idx = (self.row_idx * self.mbuf.shape[-1]
               + self.cpos_buf).view(-1)
        self.mbuf.view(-1).scatter_(0, idx, z.view(-1))
        lg = self.m(self.in_ids, position_ids=self.pos_buf,
                    cache_position=self.cpos_buf,
                    past_key_values=self.bcache, use_cache=True,
                    attention_mask=self.mbuf).logits[:, -1].float()
        # the per-row penalties
        cnt = torch.zeros(self.B, self.V, device=lg.device)
        cnt.scatter_add_(1, self.gen_buf,
                         (self.sub_ar[:self.gen_buf.shape[1]]
                          .unsqueeze(0) < self.n_gen.unsqueeze(1))
                         .float())
        act = cnt > 0
        lg = torch.where(act, torch.where(
            lg > 0, lg / self.rep_g.clamp(min=1e-6).unsqueeze(1),
            lg * self.rep_g.unsqueeze(1)), lg)
        lg = torch.where(act, lg - self.pres_g.unsqueeze(1), lg)
        lg = torch.where(act, lg - self.freq_g.unsqueeze(1) * cnt, lg)
        vk, ik = torch.topk(lg, 1024, dim=-1)
        v = vk / torch.clamp(self.temp_g, min=1e-6).unsqueeze(1)
        q_all = torch.softmax(v, dim=-1)
        NB = 4096
        rr = self.row_idx.unsqueeze(1)
        bins = torch.zeros(self.B, NB, device=lg.device)
        bb = (q_all * NB).long().clamp_(0, NB - 1) + rr * NB
        bins.view(-1).scatter_add_(0, bb.reshape(-1),
                                   q_all.reshape(-1))
        suf = bins.flip(1).cumsum(1).flip(1)
        bidx = (suf >= self.top_p_g.unsqueeze(1)).float().argmax(dim=1)
        mask = bb >= bidx.unsqueeze(1)
        kmask = (self.sub_ar.unsqueeze(0) < self.top_k_g.unsqueeze(1)) \
            | (self.top_k_g.unsqueeze(1) <= 0)
        mask = mask & kmask
        v = v.masked_fill(~mask, float('-inf'))
        gumbel = -torch.log(-torch.log(torch.rand_like(v) + 1e-20)
                            + 1e-20)
        pick = (v + gumbel).argmax(dim=-1)
        self.nxt = ik.gather(1, pick.unsqueeze(1)).squeeze(1)
        # the feed + the audit (scatter_ only)
        self.in_ids.copy_(self.nxt.unsqueeze(1))
        flat = (self.row_idx.unsqueeze(1) * self.gen_buf.shape[1]
                + self.n_gen.unsqueeze(1))
        self.gen_buf.view(-1).scatter_(0, flat.reshape(-1), self.nxt)
        self.pos_buf += 1
        self.cpos_buf += 1
        self.n_gen += 1
        self.eos_hit += self.eos_mask.gather(0, self.nxt)

    # ------------------------------------------------------------------
    @torch.no_grad()
    def capture(self, verbose: bool = True):
        for _ in range(3):
            self._samp_body()
        torch.cuda.synchronize()
        sg = torch.cuda.CUDAGraph()
        with torch.cuda.graph(sg):
            self._samp_body()
        self.sgraph = sg
        self.hard_reset()
        if verbose:
            print(f'[bgraph-v3] captured B={self.B} (right-pad, '
                  f'per-row cache_position)', flush=True)

    # ------------------------------------------------------------------
    @torch.no_grad()
    def hard_reset(self):
        self.n_gen.zero_()
        self.eos_hit.zero_()
        self.temp_g.fill_(1.0)
        self.top_p_g.fill_(1.0)
        self.top_k_g.zero_()
        self.rep_g.fill_(1.0)
        self.pres_g.zero_()
        self.freq_g.zero_()
        self.gen_buf.zero_()
        self.mbuf.fill_(-1e4)
        self.cpos_buf.zero_()
        for layer in self.bcache.layers:
            for attr in ('keys', 'values'):
                t = getattr(layer, attr, None)
                if isinstance(t, torch.Tensor):
                    t.zero_()

    # ------------------------------------------------------------------
    @torch.no_grad()
    def run_cohort(self, reqs, period: int = 8):
        """reqs: list[CohortRequest] (len <= B); the RIGHT-pad to the
        full B.  The prefill = the RIGHT-padded batched [B, L] EAGER
        with cache_position=arange(L) + the mask=None + is_causal (the
        pads land AFTER the reals = invisible under the causal).  The
        per-row state: cpos_buf[r] = li (the row's prefill len), the
        mbuf row opens [0, li).  The decode graph: the per-row
        cache_position + the per-row mask column scatter."""
        B = self.B
        tok = self.tok
        dummies = B - len(reqs)
        pad_id = tok.pad_token_id or 0
        all_reqs = list(reqs) + [
            CohortRequest(torch.tensor([[pad_id]], dtype=torch.long,
                                       device='cuda'), 1, 0.0, 1.0, 0,
                          1.0, 0.0, 0.0) for _ in range(dummies)]
        L = max(r.ids.shape[1] for r in all_reqs)
        inp = torch.full((B, L), pad_id, dtype=torch.long,
                         device='cuda')
        for i, r in enumerate(all_reqs):
            inp[i, :r.ids.shape[1]] = r.ids[0]
        cpos = torch.arange(L, device='cuda')
        # the RIGHT-padded batched prefill: the mask=None -> the
        # is_causal path (the pads at [> li] invisible under the
        # causal); the cache_position = arange(L) writes the KV at
        # [0, L) for all the rows (the pads' KV = the garbage but
        # the per-row decode mask blocks [li, ...) for the row)
        out = self.m(input_ids=inp, cache_position=cpos,
                     past_key_values=self.bcache, use_cache=True)
        # the per-row state: the last REAL token index + the next pos
        for i, r in enumerate(all_reqs):
            li = r.ids.shape[1]
            t0 = int(out.logits[i, li - 1].float().argmax())
            self.nxt[i] = t0
            self.gen_buf[i, 0] = t0
            self.in_ids[i, 0] = t0
            self.cpos_buf[i] = li
            self.pos_buf[i, 0] = li
            # the decode mask row: the [0, li) = 0 (the real + the
            # first generated), [li,) = -1e4 (the graph opens the
            # row's column per step)
            self.mbuf[i, :, :, :li] = 0
            self.mbuf[i, :, :, li:] = -1e4
            self.temp_g[i] = max(r.temp, 1e-6)
            self.top_p_g[i] = r.top_p
            self.top_k_g[i] = r.top_k
            self.rep_g[i] = r.rep
            self.pres_g[i] = r.pres
            self.freq_g[i] = r.freq
            self.max_new_g[i] = r.max_new
        self.n_gen.fill_(1)
        self.eos_hit.zero_()
        # the decode loop
        n_steps = min(max((r.max_new for r in all_reqs), default=1),
                      self.max_ctx - L - 2)
        it = 0
        while it < n_steps:
            self.sgraph.replay()
            it += 1
        # the final: the per-row emission
        gen = self.gen_buf.cpu()
        ng = self.n_gen.cpu()
        for i, r in enumerate(reqs):
            n = int(ng[i])
            toks = gen[i, :n].tolist()
            out_ids = []
            for t_ in toks:
                if self.eos_mask[t_].item() > 0:
                    break
                out_ids.append(t_)
            text = tok.decode(out_ids, skip_special_tokens=True)
            if text:
                r.out_q.put(text)
            r.done.set()
