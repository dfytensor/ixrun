// IXRUN C++ engine v5 — clean rewrite, all-fp32 internal chain.
// GSQ 5.5bpw weights, bf16 model boundaries, fp32 computation.
// Lessons applied: single dtype per kernel, no mixed reinterpret,
// all kernels on current stream, no split-K (keep it simple first).
#include <torch/extension.h>
#include <cuda_bf16.h>
#include <cstdint>
#include <ATen/cuda/CUDAContext.h>
#include <cmath>

// ---------------- GSQ GEMV (fp32 x, fp32 y) ---------------- //
__global__ void gsq_gemv_kernel(
    const float* __restrict__ x,
    const uint8_t* __restrict__ codes,
    const float* __restrict__ cb,
    const uint8_t* __restrict__ s_i8,
    float s_base, float s_step,
    float* __restrict__ yf,
    int n_gr)
{
    __shared__ float cb_sm[32];
    for (int i = threadIdx.x; i < 32; i += blockDim.x) cb_sm[i] = cb[i];
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
        const float* x16 = x + jb * 16;
        float inner = 0.f;
        #pragma unroll
        for (int i = 0; i < 12; ++i)
            inner += cb_sm[(int)((lo >> (5*i)) & 0x1F)] * x16[i];
        inner += cb_sm[(int)(((lo >> 60)
            | ((unsigned long long)d2 << 4)) & 0x1F)] * x16[12];
        #pragma unroll
        for (int i = 0; i < 3; ++i)
            inner += cb_sm[(int)((d2 >> (5*i+1)) & 0x1F)] * x16[13+i];
        acc += inner * s;
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        acc += __shfl_down_sync(0xffffffff, acc, off);
    if (lane == 0) yf[r] = acc;
}

static torch::Tensor gsq_gemv(torch::Tensor x_f32,
                              torch::Tensor codes,
                              torch::Tensor cb,
                              torch::Tensor s_i8,
                              double s_base, double s_step,
                              int64_t out_f, int64_t in_f) {
    auto y = torch::zeros({out_f}, torch::dtype(torch::kFloat32)
                                        .device(x_f32.device()));
    int n_gr = (int)(in_f / 16);
    int wpb = 8;
    auto st = at::cuda::getCurrentCUDAStream();
    gsq_gemv_kernel<<<(unsigned)(out_f / wpb), wpb * 32, 0, st>>>(
        x_f32.data_ptr<float>(),
        codes.data_ptr<uint8_t>(), cb.data_ptr<float>(),
        s_i8.data_ptr<uint8_t>(), (float)s_base, (float)s_step,
        y.data_ptr<float>(), n_gr);
    return y;
}

// ---------------- rmsnorm (bf16 in, fp32 out) ---------------- //
__global__ void rmsnorm_kernel(
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ w,
    float* __restrict__ out, int n)
{
    __shared__ float red[32];
    float sq = 0.f;
    for (int i = threadIdx.x; i < n; i += blockDim.x)
        sq += __bfloat162float(x[i]) * __bfloat162float(x[i]);
    if (threadIdx.x < 32) red[threadIdx.x] = 0.f;
    __syncthreads();
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        sq += __shfl_down_sync(0xffffffff, sq, off);
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = sq;
    __syncthreads();
    if (threadIdx.x == 0) {
        float t = 0.f;
        for (int k = 0; k < (int)(blockDim.x >> 5); ++k) t += red[k];
        red[0] = rsqrtf(t / (float)n + 1e-5f);
    }
    __syncthreads();
    for (int i = threadIdx.x; i < n; i += blockDim.x)
        out[i] = __bfloat162float(x[i]) * red[0]
                * __bfloat162float(w[i]);
}

static torch::Tensor rmsn(torch::Tensor x_bf16, torch::Tensor w_bf16) {
    int n = (int)x_bf16.numel();
    auto out = torch::empty({n}, torch::dtype(torch::kFloat32)
                                     .device(x_bf16.device()));
    rmsnorm_kernel<<<1, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const __nv_bfloat16*>(x_bf16.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(w_bf16.data_ptr()),
        out.data_ptr<float>(), n);
    return out;
}

// ---------------- rope (fp32 in-place, rotate_half convention) ------ //
// Llama/HF: rotate pairs (i, i+hd/2), NOT interleaved (2i, 2i+1)
__global__ void rope_kernel(float* q, float* k,
                            int pos, int n_heads, int n_kv_heads,
                            int head_dim, float theta_base) {
    int h = blockIdx.x;
    int i = threadIdx.x;  // i < hd/2
    if (i >= head_dim / 2) return;
    float th = pos / powf(theta_base, 2.0f*i / (float)head_dim);
    float c = cosf(th), s = sinf(th);
    int base = h * head_dim;
    // rotate_half: (base+i, base+i+hd/2)
    int lo = base + i;
    int hi = base + i + head_dim / 2;
    float q0=q[lo], q1=q[hi];
    q[lo] = q0*c - q1*s;  q[hi] = q0*s + q1*c;
    if (h < n_kv_heads) {
        float k0=k[lo], k1=k[hi];
        k[lo] = k0*c - k1*s;  k[hi] = k0*s + k1*c;
    }
}

// ---------------- attention (S=1, GQA) ---------------- //
// kv_cache: bf16 [2*n_kv_heads, ctx, head_dim]
__global__ void attn_kernel(
    const float* __restrict__ q,          // [n_heads, hd]
    const __nv_bfloat16* __restrict__ kv, // [2*nkv, ctx, hd]
    float* __restrict__ out,              // [n_heads, hd]
    int pos, int n_heads, int n_kv_heads,
    int head_dim, int ctx)
{
    int h = blockIdx.x;
    int d = threadIdx.x;
    if (d >= head_dim) return;
    int kvh = h / (n_heads / n_kv_heads);
    const __nv_bfloat16* ks = kv + kvh * ctx * head_dim;
    const __nv_bfloat16* vs = kv + (long long)n_kv_heads*ctx*head_dim
                              + kvh * ctx * head_dim;
    float inv_hd = rsqrtf((float)head_dim);
    // scores
    float maxs = -1e30f;
    for (int t = 0; t <= pos; ++t) {
        float sc = 0.f;
        for (int j = 0; j < head_dim; ++j)
            sc += q[h*head_dim + j]
                * __bfloat162float(ks[t*head_dim + j]);
        sc *= inv_hd;
        if (sc > maxs) maxs = sc;
    }
    // weighted sum
    float denom = 0.f, sum = 0.f;
    for (int t = 0; t <= pos; ++t) {
        float sc = 0.f;
        for (int j = 0; j < head_dim; ++j)
            sc += q[h*head_dim + j]
                * __bfloat162float(ks[t*head_dim + j]);
        sc = expf(sc * inv_hd - maxs);
        denom += sc;
        sum += sc * __bfloat162float(vs[t*head_dim + d]);
    }
    out[h * head_dim + d] = sum / denom;
}

// ---------------- silu-mul (fp32) ---------------- //
__global__ void silu_kernel(const float* a, const float* b,
                             float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float v = a[i];
        out[i] = v / (1.f + expf(-v)) * b[i];
    }
}

// ---------------- cache write (fp32 -> bf16) ---------------- //
__global__ void cache_write(const float* src,
                            __nv_bfloat16* dst, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = __float2bfloat16(src[i]);
}

// ---------------- layer_forward (one call per layer) ---------------- //
torch::Tensor layer_forward(
    torch::Tensor h,           // bf16 [hidden] — previous layer output
    torch::Tensor in_nw,       // bf16 [hidden] — input_layernorm weight
    torch::Tensor post_nw,     // bf16 [hidden]
    // q_proj pack
    torch::Tensor qc, torch::Tensor qcb, torch::Tensor qs,
    double qb, double qst, int64_t qo, int64_t qi,
    // k_proj pack
    torch::Tensor kc, torch::Tensor kcb, torch::Tensor ks,
    double kb, double kst, int64_t ko, int64_t ki,
    // v_proj pack
    torch::Tensor vc, torch::Tensor vcb, torch::Tensor vs,
    double vb, double vst, int64_t vo, int64_t vi,
    // o_proj pack
    torch::Tensor oc, torch::Tensor ocb, torch::Tensor os8,
    double ob, double ost, int64_t oo, int64_t oi,
    // gate pack
    torch::Tensor gc, torch::Tensor gcb, torch::Tensor gs8,
    double gb, double gst, int64_t go, int64_t gi,
    // up pack
    torch::Tensor uc, torch::Tensor ucb, torch::Tensor us8,
    double ub, double ust, int64_t uo, int64_t ui,
    // down pack
    torch::Tensor dc, torch::Tensor dcb, torch::Tensor ds8,
    double db, double dst, int64_t dof, int64_t dif,
    // cache & dims
    torch::Tensor kv_cache, int64_t pos,
    int64_t n_heads, int64_t n_kv_heads,
    int64_t head_dim, int64_t ctx, double theta)
{
    auto st = at::cuda::getCurrentCUDAStream();
    int hd = (int)head_dim;

    // 1. input norm (bf16 h → fp32 xn)
    auto xn = rmsn(h, in_nw);

    // 2. q/k/v GEMVs (fp32 out)
    auto q = gsq_gemv(xn, qc, qcb, qs, qb, qst, qo, qi);
    auto k = gsq_gemv(xn, kc, kcb, ks, kb, kst, ko, ki);
    auto v = gsq_gemv(xn, vc, vcb, vs, vb, vst, vo, vi);

    // 3. rope in-place on fp32 q/k
    rope_kernel<<<(unsigned)n_heads, (unsigned)(hd/2), 0, st>>>(
        q.data_ptr<float>(), k.data_ptr<float>(),
        (int)pos, (int)n_heads, (int)n_kv_heads, hd, (float)theta);

    // 4. cache write (fp32 k/v → bf16 cache)
    for (int kvh = 0; kvh < (int)n_kv_heads; ++kvh) {
        auto kd = kv_cache.narrow(0, kvh, 1).narrow(1, pos, 1);
        cache_write<<<1, hd, 0, st>>>(
            k.data_ptr<float>() + kvh * hd,
            reinterpret_cast<__nv_bfloat16*>(kd.data_ptr()), hd);
        auto vd = kv_cache.narrow(0, n_kv_heads + kvh, 1)
                     .narrow(1, pos, 1);
        cache_write<<<1, hd, 0, st>>>(
            v.data_ptr<float>() + kvh * hd,
            reinterpret_cast<__nv_bfloat16*>(vd.data_ptr()), hd);
    }

    // 5. attention (fp32 q, bf16 cache → fp32 attn_out)
    auto attn = torch::empty({n_heads * head_dim},
        torch::dtype(torch::kFloat32).device(h.device()));
    attn_kernel<<<(unsigned)n_heads, (unsigned)head_dim, 0, st>>>(
        q.data_ptr<float>(),
        reinterpret_cast<const __nv_bfloat16*>(kv_cache.data_ptr()),
        attn.data_ptr<float>(), (int)pos,
        (int)n_heads, (int)n_kv_heads, hd, (int)ctx);

    // 6. o_proj (fp32 → fp32)
    auto o = gsq_gemv(attn, oc, ocb, os8, ob, ost, oo, oi);

    // 7. residual: h1 = h + o (both cast to fp32)
    auto h1 = h.to(torch::kFloat32) + o;

    // 8. post norm (fp32 h1 → cast bf16 → fp32 xn2)
    auto xn2 = rmsn(h1.to(torch::kBFloat16), post_nw);

    // 9. gate/up GEMVs
    auto mg = gsq_gemv(xn2, gc, gcb, gs8, gb, gst, go, gi);
    auto mu = gsq_gemv(xn2, uc, ucb, us8, ub, ust, uo, ui);

    // 10. silu * up
    auto act = torch::empty({go},
        torch::dtype(torch::kFloat32).device(h.device()));
    silu_kernel<<<(unsigned)((go+255)/256), 256, 0, st>>>(
        mg.data_ptr<float>(), mu.data_ptr<float>(),
        act.data_ptr<float>(), (int)go);

    // 11. down GEMV
    auto md = gsq_gemv(act, dc, dcb, ds8, db, dst, dof, dif);

    // 12. residual: h2 = h1 + md → bf16
    return (h1 + md).to(torch::kBFloat16);
}

// probe: rmsnorm exposed for final norm + lm_head chain
torch::Tensor rmsnorm_out(torch::Tensor x, torch::Tensor w) {
    return rmsn(x, w);
}

// probe: raw gemv for lm_head
torch::Tensor gemv_out(torch::Tensor x,
                       torch::Tensor codes, torch::Tensor cb,
                       torch::Tensor s_i8,
                       double s_base, double s_step,
                       int64_t out_f, int64_t in_f) {
    return gsq_gemv(x, codes, cb, s_i8, s_base, s_step,
                    out_f, in_f);
}

// ---------------- C3: all-24-layers in one C++ call ------------------ //
// Eliminates 24 Python→C++ round trips + pre-allocates intermediates.
// kv_caches: [n_layers][2*nkv, ctx, hd] list of per-layer caches.
torch::Tensor decode_24(
    torch::Tensor h_bf16,
    std::vector<torch::Tensor> kv_caches,
    std::vector<torch::Tensor> in_norms,
    std::vector<torch::Tensor> post_norms,
    std::vector<torch::Tensor> codes,
    std::vector<torch::Tensor> cbs,
    std::vector<torch::Tensor> s_i8s,
    std::vector<double> bases,
    std::vector<double> steps,
    std::vector<int64_t> out_fs,
    std::vector<int64_t> in_fs,
    int64_t pos, int64_t n_heads, int64_t n_kv_heads,
    int64_t head_dim, int64_t ctx, double theta)
{
    int nl = (int)kv_caches.size();
    auto h = h_bf16;
    for (int l = 0; l < nl; ++l) {
        int b = l * 7;
        h = layer_forward(
            h, in_norms[l], post_norms[l],
            codes[b],   cbs[b],   s_i8s[b],
            bases[b],   steps[b],   out_fs[b],   in_fs[b],
            codes[b+1], cbs[b+1], s_i8s[b+1],
            bases[b+1], steps[b+1], out_fs[b+1], in_fs[b+1],
            codes[b+2], cbs[b+2], s_i8s[b+2],
            bases[b+2], steps[b+2], out_fs[b+2], in_fs[b+2],
            codes[b+3], cbs[b+3], s_i8s[b+3],
            bases[b+3], steps[b+3], out_fs[b+3], in_fs[b+3],
            codes[b+4], cbs[b+4], s_i8s[b+4],
            bases[b+4], steps[b+4], out_fs[b+4], in_fs[b+4],
            codes[b+5], cbs[b+5], s_i8s[b+5],
            bases[b+5], steps[b+5], out_fs[b+5], in_fs[b+5],
            codes[b+6], cbs[b+6], s_i8s[b+6],
            bases[b+6], steps[b+6], out_fs[b+6], in_fs[b+6],
            kv_caches[l], pos, n_heads, n_kv_heads,
            head_dim, ctx, theta);
    }
    return h;
}
