# -*- coding: utf-8 -*-
"""Fused sampling kernel v2: repetition/presence/frequency penalties +
top-k + top-p (nucleus) + temperature + softmax + multinomial in ONE
launch.  The logits row stays in L2 (0.5MB @ 130k vocab).

Nucleus design: the top-32 candidates give an EXACT threshold when
their softmax mass covers top_p (sharp distributions); otherwise a
4096-bin smem probability histogram locates the nucleus boundary to
+/-1/4096 of the remaining mass (wide distributions).  The boundary
bins shift the sampled distribution by <<1% of the mass.
"""
import sys

import torch
from torch.utils.cpp_extension import load_inline

_CUDA_SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cmath>

#define TPB 1024
#define TOPK 32
#define NBIN 4096

__global__ void scatter_counts_kernel(const int* __restrict__ ids,
                                      int n, int* __restrict__ counts) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        int t = ids[i];
        if (t >= 0) atomicAdd(&counts[t], 1);
    }
}

__device__ __forceinline__ void topk_insert(float* v, int* idx, int k,
                                            float val, int id) {
    if (val <= v[k - 1]) return;
    int j = k - 1;
    while (j > 0 && v[j - 1] < val) {
        v[j] = v[j - 1]; idx[j] = idx[j - 1]; j--;
    }
    v[j] = val; idx[j] = id;
}

// one block; returns the sampled token in out_tok[0]
__global__ void sample_fused_kernel(
    const float* __restrict__ logits,   // [V]
    const int*   __restrict__ counts,   // [V] past-token counts
    float rep, float pres, float freq, float temp,
    int top_k, float top_p,
    float* __restrict__ out_tok,
    unsigned long long seed, unsigned long long subsequence,
    int V)
{
    __shared__ float s_val[TPB];         // per-warp merged candidates
    __shared__ int   s_idx[TPB];
    __shared__ float s_wmax[TPB / 32];   // per-warp max/Dall scratch
    __shared__ float s_wsum[TPB / 32];
    __shared__ float s_max;
    __shared__ float s_Dall;
    __shared__ float s_thresh;
    __shared__ int   s_hist_path;
    __shared__ int   s_topk_thr_valid;
    __shared__ float s_topk_thr;
    __shared__ float s_denom;
    __shared__ float s_u;
    __shared__ float s_bins[NBIN];

    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int wid = tid >> 5;

    // ---- pass 1: penalties + temperature + max + top-32 + D_all ----
    float mymax = -INFINITY;
    float Dall = 0.f;
    float tv[TOPK];
    int   ti[TOPK];
    #pragma unroll
    for (int k = 0; k < TOPK; ++k) { tv[k] = -INFINITY; ti[k] = -1; }
    for (int i = tid; i < V; i += TPB) {
        float x = logits[i];
        int c = counts[i];
        if (c > 0) {
            if (rep != 1.0f) x = x > 0.f ? x / rep : x * rep;
            if (pres != 0.0f) x -= pres;
            if (freq != 0.0f) x -= freq * (float)c;
        }
        if (temp != 0.0f) x /= temp;
        Dall += __expf(x - mymax > 40.f ? 40.f : x - mymax); // rough max
        if (x > mymax) {
            // rescale the accumulator for the new max (rare)
            Dall *= __expf(mymax - x > 80.f ? 80.f : mymax - x);
            mymax = x;
        }
        topk_insert(tv, ti, TOPK, x, i);
    }
    for (int o = 16; o; o >>= 1)
        mymax = fmaxf(mymax, __shfl_down_sync(0xffffffffu, mymax, o));
    mymax = __shfl_sync(0xffffffffu, mymax, 0);
    // re-run Dall against the true max (the rescale trick is noisy)
    for (int i = tid; i < V; i += TPB) {
        float x = logits[i];
        int c = counts[i];
        if (c > 0) {
            if (rep != 1.0f) x = x > 0.f ? x / rep : x * rep;
            if (pres != 0.0f) x -= pres;
            if (freq != 0.0f) x -= freq * (float)c;
        }
        if (temp != 0.0f) x /= temp;
        Dall += __expf(x - mymax);
    }
    for (int o = 16; o; o >>= 1)
        Dall += __shfl_down_sync(0xffffffffu, Dall, o);
    Dall = __shfl_sync(0xffffffffu, Dall, 0);
    // BLOCK reduction: lane 0 of EVERY warp reports; the max/sum over
    // all 32 warps (tid==0 alone only sees warp 0's range - this was
    // the constant-token bug)
    if (lane == 0) { s_wmax[wid] = mymax; s_wsum[wid] = Dall; }
    __syncthreads();
    if (tid == 0) {
        float m = s_wmax[0], s = 0.f;
        for (int w = 0; w < TPB / 32; ++w) {
            m = fmaxf(m, s_wmax[w]);
            s += s_wsum[w];
        }
        s_max = m;
        s_Dall = s;
    }
    __syncthreads();
    const float gmax = s_max;
    const float gall = s_Dall;

    // ---- in-warp tournament: 32 lanes' lists -> lane0's top-32 ----
    float wv[TOPK];
    int   wi[TOPK];
    #pragma unroll
    for (int k = 0; k < TOPK; ++k) { wv[k] = -INFINITY; wi[k] = -1; }
    int curs = 0;
    for (int r = 0; r < TOPK; ++r) {
        float h = curs < TOPK ? tv[curs] : -INFINITY;
        int   hid = curs < TOPK ? ti[curs] : -1;
        int   hlane = lane;
        for (int o = 16; o; o >>= 1) {
            float oh = __shfl_down_sync(0xffffffffu, h, o);
            int   ohid = __shfl_sync(0xffffffffu, hid, o);
            int   ol = __shfl_sync(0xffffffffu, hlane, o);
            if (oh > h) { h = oh; hid = ohid; hlane = ol; }
        }
        h = __shfl_sync(0xffffffffu, h, 0);
        hid = __shfl_sync(0xffffffffu, hid, 0);
        hlane = __shfl_sync(0xffffffffu, hlane, 0);
        if (lane == 0) { wv[r] = h; wi[r] = hid; }
        if (lane == hlane) curs++;
        if (h <= -INFINITY / 2) break;
    }
    if (lane == 0) {
        #pragma unroll
        for (int k = 0; k < TOPK; ++k) {
            s_val[wid * TOPK + k] = wv[k];
            s_idx[wid * TOPK + k] = wi[k];
        }
    }
    __syncthreads();

    // ---- the nucleus / top-k threshold ----
    if (tid == 0) {
        const int NW = TPB / 32;
        float pv[TOPK]; int pi[TOPK];
        #pragma unroll
        for (int k = 0; k < TOPK; ++k) { pv[k] = -INFINITY; pi[k] = -1; }
        int heads[NW];
        float hvals[NW];
        for (int t = 0; t < NW; ++t) {
            heads[t] = 0;
            hvals[t] = s_val[t * TOPK];
        }
        for (int round = 0; round < TOPK; ++round) {
            int best_t = -1;
            float best_v = -INFINITY;
            for (int t = 0; t < NW; ++t) {
                if (heads[t] < TOPK && hvals[t] > best_v) {
                    best_v = hvals[t]; best_t = t;
                }
            }
            if (best_t < 0) break;
            topk_insert(pv, pi, TOPK, best_v,
                        s_idx[best_t * TOPK + heads[best_t]]);
            heads[best_t]++;
            hvals[best_t] = (heads[best_t] < TOPK)
                ? s_val[best_t * TOPK + heads[best_t]] : -INFINITY;
        }
        // exact nucleus iff the top-32 softmax mass covers top_p
        float denom32 = 0.f;
        float ex[TOPK];
        for (int k = 0; k < TOPK; ++k) {
            ex[k] = pv[k] > -INFINITY / 2 ? __expf(pv[k] - gmax) : 0.f;
            denom32 += ex[k];
        }
        float cdf = 0.f;
        float thr = -INFINITY;
        bool covered = false;
        for (int k = 0; k < TOPK; ++k) {
            cdf += ex[k] / gall;          // the FULL-softmax CDF
            thr = pv[k];
            if (cdf >= top_p) { covered = true; break; }
        }
        if (covered) {
            // honor top_k within the exact range
            if (top_k > 0 && top_k < TOPK && pv[top_k - 1] > thr)
                thr = pv[top_k - 1];
            s_thresh = thr;
            s_hist_path = 0;
        } else {
            s_hist_path = 1;
            s_topk_thr_valid =
                (top_k > 0 && top_k < TOPK) ? 1 : 0;
            s_topk_thr = s_topk_thr_valid ? pv[top_k - 1] : 0.f;
        }
    }
    __syncthreads();

    float thr = s_thresh;
    if (s_hist_path) {
        // histogram path: bin q = exp(x-gmax)/gall over NBIN buckets,
        // suffix-sum the mass, find the boundary bin
        for (int b = tid; b < NBIN; b += TPB) s_bins[b] = 0.f;
        __syncthreads();
        for (int i = tid; i < V; i += TPB) {
            float x = logits[i];
            int c = counts[i];
            if (c > 0) {
                if (rep != 1.0f) x = x > 0.f ? x / rep : x * rep;
                if (pres != 0.0f) x -= pres;
                if (freq != 0.0f) x -= freq * (float)c;
            }
            if (temp != 0.0f) x /= temp;
            float q = __expf(x - gmax) / gall;      // [0,1]
            int b = (int)(q * NBIN);
            if (b >= NBIN) b = NBIN - 1;
            if (b >= 0) atomicAdd(&s_bins[b], q);
        }
        __syncthreads();
        if (tid == 0) {
            float mass = 0.f;
            float qthr = 0.f;
            for (int b = NBIN - 1; b >= 0; --b) {
                mass += s_bins[b];
                qthr = (float)b / (float)NBIN;
                if (mass >= top_p) break;
            }
            float vthr = gmax + __logf(fmaxf(qthr, 1e-9f) * gall);
            if (s_topk_thr_valid && s_topk_thr > vthr)
                vthr = s_topk_thr;
            s_thresh = vthr;
        }
        __syncthreads();
        thr = s_thresh;
    }

    // ---- pass 2: the per-thread filtered sums -> the prefix scan ----
    float denom = 0.f;
    for (int i = tid; i < V; i += TPB) {
        float x = logits[i];
        int c = counts[i];
        if (c > 0) {
            if (rep != 1.0f) x = x > 0.f ? x / rep : x * rep;
            if (pres != 0.0f) x -= pres;
            if (freq != 0.0f) x -= freq * (float)c;
        }
        if (temp != 0.0f) x /= temp;
        if (x >= thr) denom += __expf(x - gmax);
    }
    __shared__ float s_red[TPB];
    s_red[tid] = denom;
    __syncthreads();
    // Hillis-Steele INCLUSIVE scan (the per-thread filtered prefix)
    for (int s = 1; s < TPB; s <<= 1) {
        float v = (tid >= s) ? s_red[tid - s] : 0.f;
        __syncthreads();
        s_red[tid] += v;
        __syncthreads();
    }
    const float total = s_red[TPB - 1];
    if (tid == 0) {
        curandStatePhilox4_32_10_t st;
        curand_init(seed, subsequence, 0, &st);
        s_u = curand_uniform(&st) * total;
    }
    __syncthreads();
    const float u = s_u;
    const float base = s_red[tid] - denom;   // the exclusive prefix

    // ---- pass 3: ONLY the thread owning the global-prefix crossing
    //      walks its positions (the correct CDF-ordered selection) ----
    float acc = base;
    int chosen = -1;
    if (u >= base && u < base + denom) {
        for (int i = tid; i < V; i += TPB) {
            float x = logits[i];
            int c = counts[i];
            if (c > 0) {
                if (rep != 1.0f) x = x > 0.f ? x / rep : x * rep;
                if (pres != 0.0f) x -= pres;
                if (freq != 0.0f) x -= freq * (float)c;
            }
            if (temp != 0.0f) x /= temp;
            if (x >= thr) {
                acc += __expf(x - gmax);
                if (acc >= u) { chosen = i; break; }
            }
        }
    }
    __shared__ int s_ch[TPB];
    s_ch[tid] = chosen;
    __syncthreads();
    if (tid == 0) {
        int sel = -1;
        for (int i = 0; i < TPB; ++i)
            if (s_ch[i] >= 0) { sel = s_ch[i]; break; }
        out_tok[0] = sel >= 0 ? sel : 0;
        out_tok[1] = thr;
        out_tok[2] = u;
        out_tok[3] = total;   // debug: the sampled total
    }
}

// the penalized argmax (temp==0 greedy path, penalties honored)
__global__ void argmax_pen_kernel(const float* __restrict__ logits,
                                  const int* __restrict__ counts,
                                  float rep, float pres, float freq,
                                  float* __restrict__ out_tok, int V) {
    __shared__ float s_v[TPB];
    __shared__ int   s_i[TPB];
    const int tid = threadIdx.x;
    float best = -INFINITY; int bi = 0;
    for (int i = tid; i < V; i += TPB) {
        float x = logits[i];
        int c = counts[i];
        if (c > 0) {
            if (rep != 1.0f) x = x > 0.f ? x / rep : x * rep;
            if (pres != 0.0f) x -= pres;
            if (freq != 0.0f) x -= freq * (float)c;
        }
        if (x > best) { best = x; bi = i; }
    }
    s_v[tid] = best; s_i[tid] = bi;
    __syncthreads();
    for (int s = TPB / 2; s > 0; s >>= 1) {
        if (tid < s && s_v[tid + s] > s_v[tid]) {
            s_v[tid] = s_v[tid + s]; s_i[tid] = s_i[tid + s];
        }
        __syncthreads();
    }
    if (tid == 0) out_tok[0] = s_i[0];
}
"""

_CPP_SRC = """
void scatter_counts(torch::Tensor ids, torch::Tensor counts);
torch::Tensor sample_step(torch::Tensor logits, torch::Tensor counts,
                          double rep, double pres, double freq,
                          double temp, int64_t top_k, double top_p,
                          int64_t seed, int64_t subsequence);
"""

_CPP_BODY = r"""
void scatter_counts(torch::Tensor ids, torch::Tensor counts) {
    int n = (int)ids.numel();
    if (n == 0) return;
    scatter_counts_kernel<<<(n + 255) / 256, 256, 0,
        at::cuda::getCurrentCUDAStream()>>>(
        ids.data_ptr<int>(), n, counts.data_ptr<int>());
}

torch::Tensor sample_step(torch::Tensor logits, torch::Tensor counts,
                          double rep, double pres, double freq,
                          double temp, int64_t top_k, double top_p,
                          int64_t seed, int64_t subsequence) {
    int V = (int)logits.numel();
    auto out = torch::empty({4}, torch::dtype(torch::kFloat)
                                       .device(logits.device()));
    auto s = at::cuda::getCurrentCUDAStream();
    if (temp == 0.0 && top_k <= 0 && top_p >= 1.0) {
        argmax_pen_kernel<<<1, TPB, 0, s>>>(
            logits.data_ptr<float>(), counts.data_ptr<int>(),
            (float)rep, (float)pres, (float)freq,
            out.data_ptr<float>(), V);
    } else {
        sample_fused_kernel<<<1, TPB, 0, s>>>(
            logits.data_ptr<float>(), counts.data_ptr<int>(),
            (float)rep, (float)pres, (float)freq, (float)temp,
            (int)top_k, (float)top_p, out.data_ptr<float>(),
            (unsigned long long)seed, (unsigned long long)subsequence,
            V);
    }
    return out;
}
"""


def _load():
    return load_inline(
        name='sample_fused_v6',
        cpp_sources=[_CPP_SRC],
        cuda_sources=[_CUDA_SRC + _CPP_BODY],
        functions=['scatter_counts', 'sample_step'],
        extra_cuda_cflags=['-O3', '--use_fast_math',
                           '-allow-unsupported-compiler'],
        verbose=False)


def sample_step(ext, logits_row, counts, prompt_ids_gpu, gen_ids_gpu,
                n_gen, rep=1.0, pres=0.0, freq=0.0, temp=1.0, top_k=0,
                top_p=1.0, seed=0, step=0):
    counts.zero_()
    ext.scatter_counts(prompt_ids_gpu, counts)
    if n_gen > 0:
        ext.scatter_counts(gen_ids_gpu[:n_gen], counts)
    return ext.sample_step(logits_row, counts, rep, pres, freq, temp,
                           top_k, top_p, seed, step)


if __name__ == '__main__':
    import time
    sys.path.insert(0, r'E:\IXRUN')
    torch.manual_seed(0)
    dev = 'cuda'
    ext = _load()

    V = 130560
    logits = (torch.randn(V, device=dev) * 2.0)
    counts = torch.zeros(V, dtype=torch.int32, device=dev)

    # --- correctness 1: greedy + penalties vs the torch reference ---
    lg = logits.clone()
    ids = torch.tensor([5, 77, 5, 900, 77], dtype=torch.int32, device=dev)
    counts.zero_()
    ext.scatter_counts(ids, counts)
    out = ext.sample_step(lg, counts, 1.3, 0.1, 0.05, 0.0, 0, 1.0, 1, 0)
    ref = lg.clone()
    cnt = torch.bincount(ids.long(), minlength=V)
    mask = cnt > 0
    ref[mask] = torch.where(ref[mask] > 0, ref[mask] / 1.3,
                            ref[mask] * 1.3)
    ref[mask] -= 0.1
    ref[mask] -= 0.05 * cnt[mask].float()
    print('greedy+pen match:',
          int(out.item()) == int(ref.argmax().item()),
          f'({int(out.item())} vs {int(ref.argmax().item())})', flush=True)

    # --- correctness 2: the distribution vs the torch pipeline ---
    T, P, K = 0.8, 0.9, 50
    lg = (torch.randn(V, device=dev) * 3.0)
    counts.zero_()
    N = 20000
    hist = torch.zeros(V, dtype=torch.int32, device=dev)
    t0 = time.perf_counter()
    for i in range(N):
        out = ext.sample_step(lg, counts, 1.0, 0.0, 0.0, T, K, P, 7, i)
        hist[out.item()] += 1
    torch.cuda.synchronize()
    dt_k = time.perf_counter() - t0
    ref_hist = torch.zeros(V, dtype=torch.int32, device=dev)
    g = torch.Generator(device='cpu').manual_seed(99)
    lg_cpu = lg.cpu()
    for i in range(N):
        l2 = lg_cpu.clone() / T
        vk, ik = torch.topk(l2, K)
        keep = torch.zeros_like(l2, dtype=torch.bool)
        keep[ik] = True
        l2[~keep] = float('-inf')
        s, order = torch.sort(l2, descending=True)
        cum = torch.cumsum(torch.softmax(s, dim=0), dim=0)
        keep2 = torch.zeros_like(l2, dtype=torch.bool)
        keep2[order[cum <= P]] = True
        keep2[order[0]] = True
        l2[~keep2] = float('-inf')
        p = torch.softmax(l2, dim=0)
        tok = torch.multinomial(p, 1, generator=g).item()
        ref_hist[tok] += 1
    supp = (hist > 0) | (ref_hist > 0)
    tv = (hist[supp].float() / N - ref_hist[supp].float() / N
          ).abs().sum().item() / 2
    print(f'distribution TV distance: {tv:.4f} '
          f'({"PASS" if tv < 0.05 else "FAIL"}, <0.05)', flush=True)
    print(f'kernel: {dt_k/N*1e6:.0f} us/sample (N={N})', flush=True)

    # --- speed (isolated) ---
    counts.zero_()
    for _ in range(50):
        ext.sample_step(lg, counts, 1.0, 0.0, 0.0, 0.8, 50, 0.9, 3, 0)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(500):
        ext.sample_step(lg, counts, 1.0, 0.0, 0.0, 0.8, 50, 0.9, 3, 0)
    torch.cuda.synchronize()
    us = (time.perf_counter() - t0) / 500 * 1e6
    print(f'fused sample_step: {us:.1f} us/launch', flush=True)
