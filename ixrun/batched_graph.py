"""Batched SELF-FEEDING decode graph: B rows, in-graph per-row
sampling, in-graph bookkeeping — the single-request sampling graph
generalized to a fixed cohort.

Model of operation (FIXED COHORT, no mid-batch joins):
  1. the server coalesces up to B requests (a left-pad makes every
     row's newest token the last position -> the rows decode aligned)
  2. the cohort prefill runs EAGERLY into a batched StaticCache
  3. the decode graph replays: forward[B,1] -> the per-row penalties
     -> the top-1024 subset -> the temp -> the histogram nucleus ->
     the Gumbel-max pick -> the feed/audit, ALL in-graph
  4. the host polls the pinned audit counters every `period` steps
     and trims the per-row outputs at their first eos

The capture rules are the proven ones: scatter_/gather only (the
device-indexed tensor[idx] = x writes and fill_(0-dim CUDA tensor)
break the capture), no torch.sort (the histogram nucleus instead),
the FULL body warmed up before the capture, no data-dependent host
branches (the per-row params live in device buffers).
"""
import os
import threading
import time
import queue as _queue

import torch

__all__ = ["BatchedGraphEngine", "CohortRequest"]


class CohortRequest:
    __slots__ = ("ids", "n_in", "max_new", "temp", "top_p", "top_k",
                 "rep", "pres", "freq", "out_q", "done", "error")

    def __init__(self, ids, max_new, temp, top_p, top_k,
                 rep, pres, freq):
        self.ids = ids                  # [1, L] on GPU
        self.max_new = max_new
        self.temp = temp
        self.top_p = top_p
        self.top_k = top_k
        self.rep = rep
        self.pres = pres
        self.freq = freq
        self.out_q = _queue.Queue()
        self.done = threading.Event()
        self.error = None


class BatchedGraphEngine:
    """Fixed-cohort batched decode on top of a StepGraphEngine's model.

    B slots; the cohort = up to B requests launched together; every
    row decodes in lock-step (the left-pad alignment); the early-finish
    rows keep computing (masked) until the whole cohort ends — the
    simple and graph-shape-static continuous batching.
    """

    def __init__(self, engine, B: int = 8, max_new_cap: int = 512,
                 max_ctx: int = 8192, verbose: bool = True):
        self.eng = engine
        self.tok = engine.tokenizer
        self.B = B
        self.max_new_cap = max_new_cap
        self.max_ctx = max_ctx
        self.verbose = verbose
        dev = 'cuda'
        V = engine.logits.shape[-1]
        self.V = V

        # the per-cohort static buffers (the host fills at the launch)
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
        self.active = torch.ones(B, dtype=torch.float32, device=dev)
        self.n_gen = torch.zeros(B, dtype=torch.long, device=dev)
        self.eos_hit = torch.zeros(B, dtype=torch.float32, device=dev)
        self.max_new_g = torch.full((B,), max_new_cap, dtype=torch.long,
                                    device=dev)
        self.eos_mask = torch.zeros(V, dtype=torch.float32, device=dev)
        self.row_idx = torch.arange(B, device=dev)
        self.sub_ar = torch.arange(1024, device=dev)
        # the batched-decode graph buffers
        self.pos_buf = torch.zeros(B, 1, dtype=torch.long, device=dev)
        self.abuf = torch.full((B, 1, 1, self.max_ctx), float('-inf'),
                               dtype=torch.bfloat16, device=dev)
        self.n_dev = torch.zeros(1, dtype=torch.long, device=dev)
        self.aid_len = torch.zeros(1, dtype=torch.long, device=dev)
        self.ABW = self.max_ctx + 8192
        self.aid_buf = torch.zeros(self.ABW, dtype=torch.long,
                                   device=dev)
        self.aid_ar = torch.arange(self.ABW, device=dev)
        self.aid_ones = torch.ones(self.ABW, dtype=torch.float32,
                                   device=dev)

        self.sgraph = None
        self.cohort_lock = threading.Lock()

    # ------------------------------------------------------------------
    def set_eos(self, eos_ids):
        self.eos_mask.zero_()
        for e in eos_ids:
            if 0 <= e < self.eos_mask.numel():
                self.eos_mask[e] = 1.0

    # ------------------------------------------------------------------
    @torch.no_grad()
    def _samp_body(self):
        """The in-graph batched sampling: forward -> the per-row
        penalties -> the per-row top-1024 + the histogram nucleus +
        the Gumbel-max -> the feed/audit.  Every op capture-safe."""
        m = self.eng.model
        # the attention mask column for THIS step (the position =
        # aid_len + n_dev, 0 = attend): the prefill attended [0, L),
        # the decode step t attends [0, L+t]
        col = self.aid_len + self.n_dev
        z = torch.zeros(self.B, 1, 1, 1, dtype=self.abuf.dtype,
                        device=self.in_ids.device)
        self.abuf.scatter_(3, col.view(1, 1, 1, 1).expand(
            self.B, 1, 1, 1), z)
        lg = m(self.in_ids, position_ids=self.pos_buf,
               past_key_values=self.eng.bcache, use_cache=True,
               attention_mask=self.abuf).logits[:, -1].float()
        self.pos_buf += 1
        # the per-row penalties via the per-row counts over the generated
        cnt = torch.zeros(self.B, self.V, device=lg.device)
        cnt.scatter_add_(1, self.gen_buf,
                         (self.sub_ar[:self.gen_buf.shape[1]]
                          .unsqueeze(0) < self.n_gen.unsqueeze(1))
                         .float())
        act = cnt > 0
        lg = torch.where(
            act, torch.where(lg > 0, lg / self.rep_g.clamp(min=1e-6).unsqueeze(1),
                             lg * self.rep_g.unsqueeze(1)), lg)
        lg = torch.where(act, lg - self.pres_g.unsqueeze(1), lg)
        lg = torch.where(act, lg - self.freq_g.unsqueeze(1) * cnt, lg)
        # the per-row penalties degrade to the identity when the row is
        # neutral (rep=1/pres=0/freq=0) via the params themselves
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
        self.gen_buf.view(-1).scatter_(0, flat.reshape(-1),
                                      self.nxt)
        self.n_gen += 1
        self.eos_hit += self.eos_mask.gather(0, self.nxt).sum()
        self.active = self.active * (
            1.0 - (self.eos_hit > 0).float()) * (
            (self.n_gen < self.max_new_g).float())

    # ------------------------------------------------------------------
    @torch.no_grad()
    def capture(self, verbose: bool = True):
        """The warmup + the double capture (the prefill stays eager; the
        decode graph = the forward + the sampling, the shared pool with
        the single-request graph)."""
        eng = self.eng
        self.pos_buf = torch.zeros(self.B, 1, dtype=torch.long,
                                   device='cuda')
        # the attention mask buffer [B, 1, 1, max_ctx]: 0 = attend,
        # -inf = blocked; the graph opens the decode column per step
        # via scatter_ (the index writes break the capture)
        self.abuf = torch.full((self.B, 1, 1, self.max_ctx),
                               float('-inf'),
                               dtype=torch.bfloat16, device='cuda')
        # a batched StaticCache for the decode graph
        from transformers.cache_utils import StaticCache
        eng.bcache = StaticCache(config=eng._cfg, max_batch_size=self.B,
                                 max_cache_len=self.max_ctx)
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                self._samp_body()
        torch.cuda.current_stream().wait_stream(s)
        sg = torch.cuda.CUDAGraph()
        with torch.cuda.graph(sg):
            self._samp_body()
        self.sgraph = sg
        self.hard_reset()
        if verbose:
            print(f'[batch-graph] captured B={self.B} '
                  f'in-graph sampling', flush=True)

    # ------------------------------------------------------------------
    @torch.no_grad()
    def hard_reset(self):
        self.n_gen.zero_()
        self.eos_hit.zero_()
        self.active.fill_(1.0)
        self.temp_g.fill_(1.0)
        self.top_p_g.fill_(1.0)
        self.top_k_g.zero_()
        self.rep_g.fill_(1.0)
        self.pres_g.zero_()
        self.freq_g.zero_()
        self.gen_buf.zero_()
        for layer in self.eng.bcache.layers:
            for attr in ('keys', 'values'):
                t = getattr(layer, attr, None)
                if isinstance(t, torch.Tensor):
                    t.zero_()

    # ------------------------------------------------------------------
    @torch.no_grad()
    def run_cohort(self, reqs, period: int = 8):
        """reqs: list[CohortRequest] (len <= B).  The cohort runs to
        completion (the dummy rows pad to the FULL B: the StaticCache
        update requires the exact batch).  The per-row outputs land in
        req.out_q at the end (the batched streaming = the deferred
        whole-row emission for v1)."""
        B = self.B
        eng = self.eng
        tok = self.tok
        # the dummy rows pad to the FULL B (the StaticCache update
        # requires the exact batch; the dummies decode garbage that is
        # never read)
        dummies = B - len(reqs)
        pad_id = tok.pad_token_id or 0
        all_reqs = list(reqs) + [
            CohortRequest(torch.tensor([[pad_id]], dtype=torch.long,
                                       device='cuda'), 1, 0.0, 1.0, 0,
                          1.0, 0.0, 0.0) for _ in range(dummies)]
        # the left-pad the prompts to the common length
        L = max(r.ids.shape[1] for r in all_reqs)
        inp = torch.full((B, L), pad_id, dtype=torch.long,
                         device='cuda')
        attn = torch.zeros((B, L), dtype=torch.long, device='cuda')
        for i, r in enumerate(all_reqs):
            li = r.ids.shape[1]
            inp[i, L - li:] = r.ids[0]
            attn[i, L - li:] = 1
        pos = attn.cumsum(-1) - 1
        pos.clamp_(min=0)
        # the EXPLICIT 4-D mask (the causal + the pads): the
        # pos-attention install stubs create_causal_mask globally, so
        # the batched prefill cannot rely on the HF mask machinery
        mask4 = torch.zeros((B, 1, L, L), dtype=torch.bfloat16,
                            device='cuda')
        mask4.masked_fill_(attn[:, None, None, :] == 0, float('-inf'))
        pm = pos[:, None, :] > pos[:, :, None]     # the key after the query
        mask4.masked_fill_(pm[:, None, :, :], float('-inf'))
        # the eager batched prefill into the batched cache
        out = eng.model(input_ids=inp, attention_mask=mask4,
                        position_ids=pos, past_key_values=eng.bcache,
                        use_cache=True)
        lg_last = out.logits[:, -1].float()
        # the seed: the prefill argmax per row (the eager sampled first
        # token would need the per-row gumbel — the graph handles the
        # rest of every row)
        for i, r in enumerate(all_reqs):
            t0 = int(lg_last[i].argmax())
            self.nxt[i] = t0
            self.gen_buf[i, 0] = t0
            self.in_ids[i, 0] = t0
        # the per-row params + the per-row rope positions (the real
        # position of each row's FIRST generated token = its prompt
        # length, since the left-pad aligns the last prompt token at
        # L-1)
        for i, r in enumerate(all_reqs):
            self.temp_g[i] = max(r.temp, 1e-6)
            self.top_p_g[i] = r.top_p
            self.top_k_g[i] = r.top_k
            self.rep_g[i] = r.rep
            self.pres_g[i] = r.pres
            self.freq_g[i] = r.freq
            self.max_new_g[i] = r.max_new
            self.pos_buf[i, 0] = int(pos[i, -1].item()) + 1
        # the mask: the decode region opened per step in-graph; the
        # per-row PAD key regions stay blocked
        self.abuf.zero_()
        self.abuf[:, :, :, L:] = float('-inf')
        for i, r in enumerate(all_reqs):
            pl = L - r.ids.shape[1]
            if pl > 0:
                self.abuf[i, :, :, :pl] = float('-inf')
        self.active.fill_(1.0)
        self.eos_hit.zero_()
        self.n_gen.fill_(1)
        # the decode loop
        n_steps = min(max((r.max_new for r in all_reqs), default=1),
                      self.max_ctx - L - 2)
        it = 0
        while it < n_steps:
            self.sgraph.replay()
            it += 1
        # the final: the per-row trim + the emission
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
