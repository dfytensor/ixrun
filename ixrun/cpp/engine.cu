// IXRUN C++ engine — C1: rmsnorm + GSQ GEMV + SiLU-MLP chain in one
// host call (zero python per-op). Attention/rope/KV arrive in C2.
#include <torch/extension.h>
#include <cuda_bf16.h>
#include <cstdint>
#include <ATen/cuda/CUDAContext.h>
#include <cmath>

__global__ void rope_qk_kernel(float* q, float* k, int pos,
    int n_heads, int n_kv_heads, int head_dim, float theta_base);
__global__ void gqa_attn_kernel(const float* q,
    const __nv_bfloat16* kv, float* out, int pos, int n_heads,
    int n_kv_heads, int head_dim, long long ctx, long long v_off);
__global__ void cache_write_kernel(const float* src,
    __nv_bfloat16* dst, int n);
__global__ void add_kernel(const float* a, const float* b,
    float* out, int n);

// ---------------------------------------------------------------- //
// GSQ GEMV (ported from experiments/gsq_gemv_cuda, bit-compatible):
// K32 scalar codebook + per-16 log-i8 scale, 10B codes per group.
__device__ __forceinline__ uint32_t ld_u32(const uint8_t* p) {
    const uintptr_t a = (uintptr_t) p;
    const uint32_t* pa = (const uint32_t*)(a & ~(uintptr_t)3);
    int sh = (int)(a & 3) * 8;
    return sh == 0 ? pa[0] : __funnelshift_r(pa[0], pa[1], sh);
}

__global__ void gsq_gemv_kernel(
    const __nv_bfloat16* __restrict__ x,
    const uint8_t* __restrict__ codes,   // [nR*nGr, 10]
    const float* __restrict__ cb,        // [32]
    const uint8_t* __restrict__ s_i8,    // [nR*nGr]
    float s_base, float s_step,
    float* __restrict__ yf,
    int n_gr)
{
    __shared__ float cb_sm[32];
    for (int i = threadIdx.x; i < 32; i += blockDim.x) {
        cb_sm[i] = cb[i];
    }
    __syncthreads();
    int lane = threadIdx.x & 31;
    int r = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    float acc = 0.f;
    for (int jb = lane; jb < n_gr; jb += 32) {
        long long gidx = (long long)r * n_gr + jb;
        const uint8_t* p = codes + gidx * 10;
        uint32_t d0 = (uint32_t)p[0] | ((uint32_t)p[1] << 8)
            | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
        uint32_t d1 = (uint32_t)p[4] | ((uint32_t)p[5] << 8)
            | ((uint32_t)p[6] << 16) | ((uint32_t)p[7] << 24);
        uint16_t d2 = (uint16_t)(p[8] | (p[9] << 8));
        unsigned long long lo = (unsigned long long)d0
            | ((unsigned long long)d1 << 32);
        float s = exp2f(s_base + (float)s_i8[gidx] * s_step);
        const __nv_bfloat16* x16 = x + jb * 16;
        float inner = 0.f;
        #pragma unroll
        for (int i = 0; i < 12; ++i) {
            int c = (int)((lo >> (5 * i)) & 0x1F);
            inner += cb_sm[c] * __bfloat162float(x16[i]);
        }
        int c12 = (int)(((lo >> 60)
            | ((unsigned long long)d2 << 4)) & 0x1F);
        inner += cb_sm[c12] * __bfloat162float(x16[12]);
        #pragma unroll
        for (int i = 0; i < 3; ++i) {
            int c = (int)((d2 >> (5 * i + 1)) & 0x1F);
            inner += cb_sm[c] * __bfloat162float(x16[13 + i]);
        }
        acc += inner * s;
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        acc += __shfl_down_sync(0xffffffff, acc, off);
    }
    if (lane == 0) {
        yf[r] = acc;
    }
}

static void gsq_gemv_run(const __nv_bfloat16* x,
                         const torch::Tensor& pk_codes,
                         const torch::Tensor& pk_cb,
                         const torch::Tensor& pk_s,
                         double s_base, double s_step,
                         float* yf, int out_f, int in_f) {
    int n_gr = in_f / 16;
    int wpb = 8;
    unsigned gx = (unsigned)(out_f / wpb);
    auto s0 = at::cuda::getCurrentCUDAStream();
    gsq_gemv_kernel<<<gx, wpb * 32, 0, s0>>>(
        x, pk_codes.data_ptr<uint8_t>(), pk_cb.data_ptr<float>(),
        pk_s.data_ptr<uint8_t>(), (float)s_base, (float)s_step,
        yf, n_gr);
}

__global__ void rmsnorm_kernel(const __nv_bfloat16* __restrict__ x,
                               const __nv_bfloat16* __restrict__ w,
                               __nv_bfloat16* __restrict__ out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    // single-CTA reduction done on host-side helper kernel below
    out[i] = __float2bfloat16(__bfloat162float(x[i])
        * __bfloat162float(w[i]));
}

// rmsnorm with fp32 mean-square computed by block 0 (n <= 3072 fits)
__global__ void rmsnorm_full(const __nv_bfloat16* __restrict__ x,
                             const __nv_bfloat16* __restrict__ w,
                             __nv_bfloat16* __restrict__ out, int n) {
    extern __shared__ float red[];
    float sq = 0.f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        float v = __bfloat162float(x[i]);
        sq += v * v;
    }
    if (threadIdx.x < 32) red[threadIdx.x] = 0.f;
    __syncthreads();
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        sq += __shfl_down_sync(0xffffffff, sq, off);
    }
    if ((threadIdx.x & 31) == 0) red[(threadIdx.x >> 5) & 31] = sq;
    __syncthreads();
    if (threadIdx.x == 0) {
        float t = 0.f;
        for (int k = 0; k < (int)(blockDim.x >> 5) && k < 32; ++k) {
            t += red[k];
        }
        red[0] = t / (float)n;
    }
    __syncthreads();
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        float inv = rsqrtf(red[0] + 1e-5f);
        out[i] = __float2bfloat16(
            __bfloat162float(x[i]) * inv * __bfloat162float(w[i]));
    }
}

__global__ void silu_mul_kernel(const float* __restrict__ a,
                                const float* __restrict__ b,
                                float* __restrict__ out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v = a[i];
    out[i] = v / (1.f + expf(-v)) * b[i];
}

__global__ void argmax_kernel(const float* __restrict__ logits, int n,
                              int* __restrict__ out) {
    float best = logits[0];
    int bi = 0;
    for (int i = 1; i < n; ++i) {
        if (logits[i] > best) { best = logits[i]; bi = i; }
    }
    *out = bi;
}

// packs for one linear, flattened as struct-of-arrays entries
struct GsqP {
    torch::Tensor codes, cb, s;
    double s_base, s_step;
    int64_t out_f, in_f;
};

static void lin(const torch::Tensor& xb,
                const torch::Tensor& codes, const torch::Tensor& cb,
                const torch::Tensor& s8, double s_base, double s_step,
                int64_t out_f, int64_t in_f, float* yf) {
    gsq_gemv_run(reinterpret_cast<const __nv_bfloat16*>(
                     xb.data_ptr()),
                 codes, cb, s8, s_base, s_step,
                 yf, (int)out_f, (int)in_f);
}

// full MLP: y = down( silu(gate(norm_x)) * up(norm_x) )
// tensors: bf16 x [in], per-linear GSQ planes, norms.
torch::Tensor mlp_forward(torch::Tensor x, torch::Tensor norm_w,
                 torch::Tensor gc, torch::Tensor gcb, torch::Tensor gs,
                 double gb, double gst, int64_t go, int64_t gi,
                 torch::Tensor uc, torch::Tensor ucb, torch::Tensor us,
                 double ub, double ust, int64_t uo, int64_t ui,
                 torch::Tensor dc, torch::Tensor dcb, torch::Tensor ds,
                 double dbase, double dstep, int64_t dof, int64_t dif) {
    int in_f = (int)x.numel();
    auto xg = torch::empty({in_f},
        torch::dtype(torch::kBFloat16).device(x.device()));
    auto s0 = at::cuda::getCurrentCUDAStream();
    rmsnorm_full<<<1, 256, 32 * sizeof(float)>>>(
        reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(norm_w.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(xg.data_ptr()), in_f);
    int h = (int)go;
    auto gf = torch::empty({h}, torch::dtype(torch::kFloat32)
                                  .device(x.device()));
    auto uf = torch::empty({h}, torch::dtype(torch::kFloat32)
                                  .device(x.device()));
    lin(xg, gc, gcb, gs, gb, gst, go, gi, gf.data_ptr<float>());
    lin(xg, uc, ucb, us, ub, ust, uo, ui, uf.data_ptr<float>());
    auto act = torch::empty({h}, torch::dtype(torch::kFloat32)
                                   .device(x.device()));
    silu_mul_kernel<<<(h + 255) / 256, 256, 0, s0>>>(
        gf.data_ptr<float>(), uf.data_ptr<float>(),
        act.data_ptr<float>(), h);
    auto df32 = torch::empty({in_f}, torch::dtype(torch::kFloat32)
                                         .device(x.device()));
    lin(act, dc, dcb, ds, dbase, dstep, dof, dif,
        df32.data_ptr<float>());
    return df32.to(torch::kBFloat16);
}

__global__ void noop_kernel() {}

// stage probes for C1 numerical isolation
torch::Tensor rmsnorm_out(torch::Tensor x, torch::Tensor w) {
    int n = (int)x.numel();
    auto out = torch::empty({n}, torch::dtype(torch::kBFloat16)
                                     .device(x.device()));
    rmsnorm_full<<<1, 256, 32 * sizeof(float)>>>(
        reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(w.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), n);
    return out;
}

void rope_probe(torch::Tensor q, torch::Tensor k, int64_t pos,
                int64_t n_heads, int64_t n_kv_heads, int64_t head_dim,
                double theta_base) {
    auto s0 = at::cuda::getCurrentCUDAStream();
    rope_qk_kernel<<<(unsigned)n_heads, (unsigned)(head_dim / 2), 0,
                     s0>>>(
        q.data_ptr<float>(), k.data_ptr<float>(), (int)pos,
        (int)n_heads, (int)n_kv_heads, (int)head_dim,
        (float)theta_base);
}

torch::Tensor attn_probe(torch::Tensor q, torch::Tensor kv,
                         int64_t pos, int64_t n_heads,
                         int64_t n_kv_heads, int64_t head_dim,
                         int64_t ctx, int64_t v_off) {
    auto out = torch::empty({n_heads * head_dim},
        torch::dtype(torch::kFloat32).device(q.device()));
    auto s0 = at::cuda::getCurrentCUDAStream();
    gqa_attn_kernel<<<(unsigned)n_heads, (unsigned)head_dim, 0, s0>>>(
        q.data_ptr<float>(),
        reinterpret_cast<const __nv_bfloat16*>(kv.data_ptr()),
        out.data_ptr<float>(), (int)pos, (int)n_heads,
        (int)n_kv_heads, (int)head_dim, ctx, v_off);
    return out;
}

torch::Tensor gsq_gemv_out(torch::Tensor x,
                           torch::Tensor codes, torch::Tensor cb,
                           torch::Tensor s8, double s_base, double s_step,
                           int64_t out_f, int64_t in_f) {
    auto yf = torch::zeros({out_f}, torch::dtype(torch::kFloat32)
                                        .device(x.device()));
    auto xf = x.to(torch::kFloat32);
    gsq_gemv_run(xf.data_ptr<float>(),
                 codes, cb, s8, s_base, s_step,
                 yf.data_ptr<float>(), (int)out_f, (int)in_f);
    return yf;
}

// NOTE: no PYBIND11_MODULE here - load_inline generates the binding
// from cpp_sources declarations + functions=[...] (double definition
// = LNK2005 PyInit / LNK1169).

// ---------------------------------------------------------------- //
// C2: RoPE + GQA S=1 attention + per-layer decode step

__global__ void rope_qk_kernel(float* q, float* k,
                               int pos, int n_heads, int n_kv_heads,
                               int head_dim, float theta_base) {
    int h = blockIdx.x;
    int i = threadIdx.x;                 // pairs: 2*i, 2*i+1
    if (i >= head_dim / 2) return;
    float theta = pos / powf(theta_base,
                             2.0f * i / (float)head_dim);
    float c = cosf(theta), s = sinf(theta);
    int off = h * head_dim + 2 * i;
    float q0 = q[off];
    float q1 = q[off + 1];
    q[off] = q0 * c - q1 * s;
    q[off + 1] = q0 * s + q1 * c;
    if (h < n_kv_heads) {                // k only has n_kv_heads rows!
        int koff = h * head_dim + 2 * i;
        float k0 = k[koff];
        float k1 = k[koff + 1];
        k[koff] = k0 * c - k1 * s;
        k[koff + 1] = k0 * s + k1 * c;
    }
}

// S=1 attention, one block per q-head. kv cache: bf16 [n_kv_heads][ctx][hd]
__global__ void gqa_attn_kernel(
    const float* __restrict__ q,             // [n_heads, hd]
    const __nv_bfloat16* __restrict__ kv,    // [2*n_kv_heads, ctx, hd]
    float* __restrict__ out,                 // [n_heads, hd]
    int pos, int n_heads, int n_kv_heads,
    int head_dim, long long ctx, long long v_off) {
    int h = blockIdx.x;
    int kvh = n_kv_heads == n_heads ? h : h / (n_heads / n_kv_heads);
    int d0 = threadIdx.x;
    if (d0 >= head_dim) return;
    const __nv_bfloat16* ksec = kv + kvh * ctx * head_dim;
    const __nv_bfloat16* vsec = kv + v_off + kvh * ctx * head_dim;
    float inv_sqrt_hd = rsqrtf((float)head_dim);
    float maxs = -1e30f;
    for (int t = 0; t <= pos; ++t) {
        float s = 0.f;
        for (int d = 0; d < head_dim; ++d) {
            s += q[h * head_dim + d]
               * __bfloat162float(ksec[t * head_dim + d]);
        }
        s *= inv_sqrt_hd;
        if (s > maxs) maxs = s;
    }
    float denom = 0.f;
    float sum = 0.f;
    for (int t = 0; t <= pos; ++t) {
        float s = 0.f;
        for (int d = 0; d < head_dim; ++d) {
            s += q[h * head_dim + d]
               * __bfloat162float(ksec[t * head_dim + d]);
        }
        s = expf(s * inv_sqrt_hd - maxs);
        denom += s;
        sum += s * __bfloat162float(vsec[t * head_dim + d0]);
    }
    out[h * head_dim + d0] = sum / denom;
}
// ---------------------------------------------------------------- //
// C2: full decoder layer in one host call

static torch::Tensor rmsn(torch::Tensor x, torch::Tensor w) {
    int n = (int)x.numel();
    auto out = torch::empty({n}, torch::dtype(torch::kBFloat16)
                                     .device(x.device()));
    rmsnorm_full<<<1, 256, 32 * sizeof(float)>>>(
        reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(w.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), n);
    return out;
}

static torch::Tensor gsv(torch::Tensor x,
                         torch::Tensor codes, torch::Tensor cb,
                         torch::Tensor s8, double s_base, double s_step,
                         int64_t out_f, int64_t in_f) {
    auto yf = torch::zeros({out_f}, torch::dtype(torch::kFloat32)
                                        .device(x.device()));
    auto xb = x.to(torch::kBFloat16);
    gsq_gemv_run(reinterpret_cast<const __nv_bfloat16*>(xb.data_ptr()),
                 codes, cb, s8, s_base, s_step,
                 yf.data_ptr<float>(), (int)out_f, (int)in_f);
    return yf;
}

__global__ void add_kernel(const __nv_bfloat16* __restrict__ a,
                           const __nv_bfloat16* __restrict__ b,
                           __nv_bfloat16* __restrict__ out, int n);

__global__ void cache_write_kernel(const __nv_bfloat16* __restrict__ src,
                                   __nv_bfloat16* __restrict__ dst,
                                   int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = src[i];
}

torch::Tensor layer_forward(
    torch::Tensor h, torch::Tensor in_nw, torch::Tensor post_nw,
    torch::Tensor qc, torch::Tensor qcb, torch::Tensor qs,
    double qb, double qst, int64_t qo, int64_t qi,
    torch::Tensor kc, torch::Tensor kcb, torch::Tensor ks,
    double kb, double kst, int64_t ko, int64_t ki,
    torch::Tensor vc, torch::Tensor vcb, torch::Tensor vs,
    double vb, double vst, int64_t vo, int64_t vi,
    torch::Tensor oc, torch::Tensor ocb, torch::Tensor os8,
    double ob, double ost, int64_t oo, int64_t oi,
    torch::Tensor gc, torch::Tensor gcb, torch::Tensor gs8,
    double gb, double gst, int64_t go, int64_t gi,
    torch::Tensor uc, torch::Tensor ucb, torch::Tensor us8,
    double ub, double ust, int64_t uo, int64_t ui,
    torch::Tensor dc, torch::Tensor dcb, torch::Tensor ds8,
    double db, double dst, int64_t dof, int64_t dif,
    torch::Tensor kv_cache, int64_t pos, int64_t n_heads,
    int64_t n_kv_heads, int64_t head_dim, int64_t ctx,
    double theta_base)
{
    auto s0 = at::cuda::getCurrentCUDAStream();
    auto xn = rmsn(h, in_nw);
    auto q = gsv(xn, qc, qcb, qs, qb, qst, qo, qi);
    auto k = gsv(xn, kc, kcb, ks, kb, kst, ko, ki);
    auto v = gsv(xn, vc, vcb, vs, vb, vst, vo, vi);
    int nh = (int)n_heads, hd = (int)head_dim;
    rope_qk_kernel<<<nh, hd / 2, 0, s0>>>(
        q.data_ptr<float>(), k.data_ptr<float>(),
        (int)pos, nh, (int)n_kv_heads, hd, (float)theta_base);
    // write k/v into cache[kvh][pos] (cache stays bf16)
    int kvn = (int)n_kv_heads * hd;
    auto k16 = k.to(torch::kBFloat16);
    auto v16 = v.to(torch::kBFloat16);
    for (int kvh = 0; kvh < (int)n_kv_heads; ++kvh) {
        auto kdst = kv_cache.narrow(0, kvh, 1).narrow(1, pos, 1);
        cache_write_kernel<<<(kvn + 255) / 256, 256, 0, s0>>>(
            reinterpret_cast<const __nv_bfloat16*>(k16.data_ptr())
                + kvh * hd,
            reinterpret_cast<__nv_bfloat16*>(kdst.data_ptr()), hd);
        cache_write_kernel<<<(kvn + 255) / 256, 256, 0, s0>>>(
            reinterpret_cast<const __nv_bfloat16*>(v16.data_ptr())
                + kvh * hd,
            reinterpret_cast<__nv_bfloat16*>(
                kv_cache.narrow(0, n_kv_heads + kvh, 1)
                    .narrow(1, pos, 1).data_ptr()), hd);
    }
    auto attn = torch::empty({nh * hd},
        torch::dtype(torch::kFloat32).device(h.device()));
    gqa_attn_kernel<<<nh, hd, 0, s0>>>(
        q.data_ptr<float>(),
        reinterpret_cast<const __nv_bfloat16*>(kv_cache.data_ptr()),
        attn.data_ptr<float>(),
        (int)pos, nh, (int)n_kv_heads, hd, ctx,
        (long long)n_kv_heads * ctx * hd);
    auto o = gsv(attn, oc, ocb, os8, ob, ost, oo, oi);
    auto hf = h.to(torch::kFloat32);
    auto h1 = hf + o;
    auto xn2 = rmsn((h1).to(torch::kBFloat16), post_nw);
    auto mg = gsv(xn2, gc, gcb, gs8, gb, gst, go, gi);
    auto mu = gsv(xn2, uc, ucb, us8, ub, ust, uo, ui);
    auto act = torch::empty({mg.numel()},
        torch::dtype(torch::kFloat32).device(h.device()));
    silu_mul_kernel<<<((int)mg.numel() + 255) / 256, 256, 0, s0>>>(
        mg.data_ptr<float>(), mu.data_ptr<float>(),
        act.data_ptr<float>(), (int)mg.numel());
    auto md = gsv(act, dc, dcb, ds8, db, dst, dof, dif);
    auto h2 = ((h1 + md).to(torch::kBFloat16)).to(torch::kFloat32);
    return h2.to(torch::kBFloat16);
}

__global__ void add_kernel(const __nv_bfloat16* __restrict__ a,
                           const __nv_bfloat16* __restrict__ b,
                           __nv_bfloat16* __restrict__ out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = __float2bfloat16(__bfloat162float(a[i])
                                  + __bfloat162float(b[i]));
    }
}
