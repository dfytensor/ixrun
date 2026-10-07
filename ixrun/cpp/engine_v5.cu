// IXRUN C++ engine v5 — clean rewrite, all-fp32 internal chain.
// GSQ 5.5bpw weights, bf16 model boundaries, fp32 computation.
// Lessons applied: single dtype per kernel, no mixed reinterpret,
// all kernels on current stream, no split-K (keep it simple first).
#include <torch/extension.h>
#include <cuda_bf16.h>
#include <cstdint>
#include <ATen/cuda/CUDAContext.h>
#include <cmath>

// ---------------- device-side position (CUDA-graph parameterizable) --- //
__device__ int d_pos = 0;

__global__ void set_pos_kernel(const int* pos_gpu) {
    d_pos = *pos_gpu;
}

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

// ---------------- rope (fp32 in-place, rotate_half, reads d_pos) ---- //
__global__ void rope_kernel(float* q, float* k,
                            int n_heads, int n_kv_heads,
                            int head_dim, float theta_base) {
    int h = blockIdx.x;
    int i = threadIdx.x;
    if (i >= head_dim / 2) return;
    int pos = d_pos;
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
    const float* __restrict__ q,
    const __nv_bfloat16* __restrict__ kv,
    float* __restrict__ out,
    int n_heads, int n_kv_heads,
    int head_dim, int ctx)
{
    int pos = d_pos;
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
// cache_write: reads d_pos for destination offset (graph-safe)
__global__ void cache_write(const float* src,
                            __nv_bfloat16* dst_base, int hd) {
    int i = threadIdx.x;
    if (i < hd)
        dst_base[d_pos * hd + i] = __float2bfloat16(src[i]);
}

// ---------------- helper kernels (graph-safe, no ATen) --------------- //
__global__ void cast_bf16_f32(const __nv_bfloat16* src,
                              float* dst, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = __bfloat162float(src[i]);
}
__global__ void cast_f32_bf16(const float* src,
                              __nv_bfloat16* dst, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = __float2bfloat16(src[i]);
}
__global__ void add_f32(const float* a, const float* b,
                        float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = a[i] + b[i];
}

// ---------------- layer_forward (graph-safe, all raw kernels) -------- //
torch::Tensor layer_forward(
    torch::Tensor h,           // bf16 [hidden] — previous layer output
    torch::Tensor in_nw,       // bf16 [hidden] — input_layernorm weight
    torch::Tensor post_nw,     // bf16 [hidden]
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
    torch::Tensor kv_cache,
    int64_t n_heads, int64_t n_kv_heads,
    int64_t head_dim, int64_t ctx, double theta)
{
    auto st = at::cuda::getCurrentCUDAStream();
    int hd = (int)head_dim;
    auto dev = h.device();

    // ---- static buffer pool (allocated once, reused across calls) ----
    static torch::Tensor b_xn, b_q, b_k, b_v, b_attn, b_o,
                         b_h1f, b_xn2, b_mg, b_mu, b_act, b_md,
                         b_h1bf, b_hf, b_out;
    static int pool_init = 0;
    if (!pool_init) {
        int H = (int)h.numel();
        auto opts_f = torch::TensorOptions()
                        .dtype(torch::kFloat32).device(dev);
        auto opts_b = torch::TensorOptions()
                        .dtype(torch::kBFloat16).device(dev);
        b_xn  = torch::empty({H},   opts_f);
        b_q   = torch::empty({(long)qo},  opts_f);
        b_k   = torch::empty({(long)ko},  opts_f);
        b_v   = torch::empty({(long)vo},  opts_f);
        b_attn= torch::empty({(long)qo},  opts_f);
        b_o   = torch::empty({(long)oo},  opts_f);
        b_h1f = torch::empty({H},   opts_f);
        b_hf  = torch::empty({H},   opts_f);
        b_h1bf= torch::empty({H},   opts_b);
        b_xn2 = torch::empty({H},   opts_f);
        b_mg  = torch::empty({(long)go},  opts_f);
        b_mu  = torch::empty({(long)uo},  opts_f);
        b_act = torch::empty({(long)go},  opts_f);
        b_md  = torch::empty({(long)dof}, opts_f);
        b_out = torch::empty({H},   opts_b);
        pool_init = 1;
    }
    int H_n = (int)h.numel();
    int nblk = (H_n + 255) / 256;

    // 1. input norm
    rmsnorm_kernel<<<1, 256, 0, st>>>(
        reinterpret_cast<const __nv_bfloat16*>(h.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(in_nw.data_ptr()),
        b_xn.data_ptr<float>(), H_n);

    // 2. q/k/v GEMVs
    gsq_gemv_kernel<<<(unsigned)(qo / 8), 256, 0, st>>>(
        b_xn.data_ptr<float>(), qc.data_ptr<uint8_t>(),
        qcb.data_ptr<float>(), qs.data_ptr<uint8_t>(),
        (float)qb, (float)qst, b_q.data_ptr<float>(),
        (int)(qi / 16));
    gsq_gemv_kernel<<<(unsigned)(ko / 8), 256, 0, st>>>(
        b_xn.data_ptr<float>(), kc.data_ptr<uint8_t>(),
        kcb.data_ptr<float>(), ks.data_ptr<uint8_t>(),
        (float)kb, (float)kst, b_k.data_ptr<float>(),
        (int)(ki / 16));
    gsq_gemv_kernel<<<(unsigned)(vo / 8), 256, 0, st>>>(
        b_xn.data_ptr<float>(), vc.data_ptr<uint8_t>(),
        vcb.data_ptr<float>(), vs.data_ptr<uint8_t>(),
        (float)vb, (float)vst, b_v.data_ptr<float>(),
        (int)(vi / 16));

    // 3. rope (reads d_pos)
    rope_kernel<<<(unsigned)n_heads, (unsigned)(hd/2), 0, st>>>(
        b_q.data_ptr<float>(), b_k.data_ptr<float>(),
        (int)n_heads, (int)n_kv_heads, hd, (float)theta);

    // 4. cache write (device-side pos via d_pos)
    auto kv_base = reinterpret_cast<__nv_bfloat16*>(kv_cache.data_ptr());
    for (int kvh = 0; kvh < (int)n_kv_heads; ++kvh) {
        cache_write<<<1, hd, 0, st>>>(
            b_k.data_ptr<float>() + kvh * hd,
            kv_base + kvh * (int)ctx * hd, hd);
        cache_write<<<1, hd, 0, st>>>(
            b_v.data_ptr<float>() + kvh * hd,
            kv_base + ((int)n_kv_heads + kvh) * (int)ctx * hd, hd);
    }

    // 5. attention (reads d_pos)
    attn_kernel<<<(unsigned)n_heads, (unsigned)hd, 0, st>>>(
        b_q.data_ptr<float>(),
        reinterpret_cast<const __nv_bfloat16*>(kv_cache.data_ptr()),
        b_attn.data_ptr<float>(),
        (int)n_heads, (int)n_kv_heads, hd, (int)ctx);

    // 6. o_proj
    gsq_gemv_kernel<<<(unsigned)(oo / 8), 256, 0, st>>>(
        b_attn.data_ptr<float>(), oc.data_ptr<uint8_t>(),
        ocb.data_ptr<float>(), os8.data_ptr<uint8_t>(),
        (float)ob, (float)ost, b_o.data_ptr<float>(),
        (int)(oi / 16));

    // 7. residual: b_h1f = h_f32 + b_o (raw kernels)
    cast_bf16_f32<<<nblk, 256, 0, st>>>(
        reinterpret_cast<const __nv_bfloat16*>(h.data_ptr()),
        b_hf.data_ptr<float>(), H_n);
    add_f32<<<nblk, 256, 0, st>>>(
        b_hf.data_ptr<float>(), b_o.data_ptr<float>(),
        b_h1f.data_ptr<float>(), H_n);

    // 8. post norm: bf16 copy of b_h1f → rmsnorm
    cast_f32_bf16<<<nblk, 256, 0, st>>>(
        b_h1f.data_ptr<float>(),
        reinterpret_cast<__nv_bfloat16*>(b_h1bf.data_ptr()),
        H_n);
    rmsnorm_kernel<<<1, 256, 0, st>>>(
        reinterpret_cast<const __nv_bfloat16*>(b_h1bf.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(post_nw.data_ptr()),
        b_xn2.data_ptr<float>(), H_n);

    // 9-10. gate/up + silu
    gsq_gemv_kernel<<<(unsigned)(go / 8), 256, 0, st>>>(
        b_xn2.data_ptr<float>(), gc.data_ptr<uint8_t>(),
        gcb.data_ptr<float>(), gs8.data_ptr<uint8_t>(),
        (float)gb, (float)gst, b_mg.data_ptr<float>(),
        (int)(gi / 16));
    gsq_gemv_kernel<<<(unsigned)(uo / 8), 256, 0, st>>>(
        b_xn2.data_ptr<float>(), uc.data_ptr<uint8_t>(),
        ucb.data_ptr<float>(), us8.data_ptr<uint8_t>(),
        (float)ub, (float)ust, b_mu.data_ptr<float>(),
        (int)(ui / 16));
    silu_kernel<<<(unsigned)((go + 255) / 256), 256, 0, st>>>(
        b_mg.data_ptr<float>(), b_mu.data_ptr<float>(),
        b_act.data_ptr<float>(), (int)go);

    // 11. down GEMV
    gsq_gemv_kernel<<<(unsigned)(dof / 8), 256, 0, st>>>(
        b_act.data_ptr<float>(), dc.data_ptr<uint8_t>(),
        dcb.data_ptr<float>(), ds8.data_ptr<uint8_t>(),
        (float)db, (float)dst, b_md.data_ptr<float>(),
        (int)(dif / 16));

    // 12. residual: b_out = bf16(b_h1f + b_md) (in-place on b_h1f)
    add_f32<<<nblk, 256, 0, st>>>(
        b_h1f.data_ptr<float>(), b_md.data_ptr<float>(),
        b_h1f.data_ptr<float>(), H_n);
    cast_f32_bf16<<<nblk, 256, 0, st>>>(
        b_h1f.data_ptr<float>(),
        reinterpret_cast<__nv_bfloat16*>(b_out.data_ptr()), H_n);

    return b_out;
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
    torch::Tensor pos_gpu,       // [1] int32 GPU tensor
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
    int64_t n_heads, int64_t n_kv_heads,
    int64_t head_dim, int64_t ctx, double theta)
{
    int nl = (int)kv_caches.size();
    auto h = h_bf16;
    auto st = at::cuda::getCurrentCUDAStream().stream();

    // Update device-side position (graph-safe: pos_gpu is a fixed buffer)
    set_pos_kernel<<<1, 1, 0, st>>>(
        reinterpret_cast<const int*>(pos_gpu.data_ptr()));

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
            kv_caches[l], n_heads, n_kv_heads,
            head_dim, ctx, theta);
    }
    return h;
}

// device-side argmax (single-block, grid-stride over full vocab)
__global__ void argmax_f32(const float* logits, int n,
                           int64_t* out) {
    __shared__ float s_val[256];
    __shared__ int s_idx[256];
    int i = threadIdx.x;
    float bv = -1e30f; int bi = 0;
    for (int j = i; j < n; j += 256) {
        float v = logits[j];
        if (v > bv) { bv = v; bi = j; }
    }
    s_val[i] = bv; s_idx[i] = bi;
    __syncthreads();
    for (int off = 128; off > 0; off >>= 1) {
        if (i < off) {
            if (s_val[i + off] > s_val[i]) {
                s_val[i] = s_val[i + off];
                s_idx[i] = s_idx[i + off];
            }
        }
        __syncthreads();
    }
    if (i == 0) *out = (int64_t)s_idx[0];
}

// ---------------- full C++ generation step (zero Python per token) ---- //
// Does: embedding lookup → 24 layers → final norm → lm_head → argmax
// Returns next token_id. Python calls this once per token.
int64_t generate_step(
    int64_t token_id,
    torch::Tensor embed_table,     // [vocab, hidden] bf16
    torch::Tensor pos_gpu,         // [1] int32
    std::vector<torch::Tensor> kv_caches,
    std::vector<torch::Tensor> in_norms,
    std::vector<torch::Tensor> post_norms,
    std::vector<torch::Tensor> fn_w,   // [hidden] bf16 final norm
    std::vector<torch::Tensor> codes,
    std::vector<torch::Tensor> cbs,
    std::vector<torch::Tensor> s_i8s,
    std::vector<double> bases,
    std::vector<double> steps,
    std::vector<int64_t> out_fs,
    std::vector<int64_t> in_fs,
    // lm_head pack
    torch::Tensor lh_codes, torch::Tensor lh_cb,
    torch::Tensor lh_s, double lh_base, double lh_step,
    int64_t lh_out_f, int64_t lh_in_f,
    int64_t n_heads, int64_t n_kv_heads,
    int64_t head_dim, int64_t ctx, double theta,
    int64_t hidden)
{
    auto st = at::cuda::getCurrentCUDAStream();

    // 1. embedding lookup: copy row token_id from embed_table
    static torch::Tensor emb_buf;
    if (!emb_buf.defined()) {
        emb_buf = torch::empty({hidden},
            torch::TensorOptions().dtype(torch::kBFloat16)
                .device(embed_table.device()));
    }
    cudaMemcpyAsync(emb_buf.data_ptr(),
        reinterpret_cast<const char*>(embed_table.data_ptr())
            + token_id * hidden * 2,
        hidden * 2, cudaMemcpyDeviceToDevice, st);

    // 2. decode 24 layers
    auto h = decode_24(emb_buf, pos_gpu, kv_caches, in_norms,
                       post_norms, codes, cbs, s_i8s, bases,
                       steps, out_fs, in_fs,
                       n_heads, n_kv_heads, head_dim, ctx, theta);

    // 3. final norm (bf16 in → fp32 out)
    static torch::Tensor fn_buf;
    if (!fn_buf.defined()) {
        fn_buf = torch::empty({hidden},
            torch::TensorOptions().dtype(torch::kFloat32)
                .device(h.device()));
    }
    rmsnorm_kernel<<<1, 256, 0, st>>>(
        reinterpret_cast<const __nv_bfloat16*>(h.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(fn_w[0].data_ptr()),
        fn_buf.data_ptr<float>(), (int)hidden);

    // 4. lm_head GEMV
    static torch::Tensor lg_buf;
    if (!lg_buf.defined()) {
        lg_buf = torch::empty({lh_out_f},
            torch::TensorOptions().dtype(torch::kFloat32)
                .device(h.device()));
    }
    gsq_gemv_kernel<<<(unsigned)(lh_out_f / 8), 256, 0, st>>>(
        fn_buf.data_ptr<float>(), lh_codes.data_ptr<uint8_t>(),
        lh_cb.data_ptr<float>(), lh_s.data_ptr<uint8_t>(),
        (float)lh_base, (float)lh_step,
        lg_buf.data_ptr<float>(), (int)(lh_in_f / 16));

    // 5. argmax (device-side, single block reduction)
    static torch::Tensor tok_buf;
    if (!tok_buf.defined()) {
        tok_buf = torch::empty({1},
            torch::TensorOptions().dtype(torch::kInt64)
                .device(h.device()));
    }
    argmax_f32<<<1, 256, 0, st>>>(
        lg_buf.data_ptr<float>(), (int)lh_out_f,
        tok_buf.data_ptr<int64_t>());

    // 6. read result (one sync per token — unavoidable)
    return tok_buf.item<int64_t>();
}

// ---------------- static-init generation (llama.cpp architecture) ------ //
static bool g_init = false;
static torch::Tensor g_embed, g_pos, g_fnw,
                     g_lh_codes, g_lh_cb, g_lh_s;
static double g_lh_base, g_lh_step, g_theta;
static int64_t g_lh_out_f, g_lh_in_f, g_nheads, g_nkv, g_hd,
               g_ctx, g_hidden;
static std::vector<torch::Tensor> g_kcs, g_inws, g_postws,
                                  g_codes, g_cbs, g_s;
static std::vector<double> g_bases, g_steps;
static std::vector<int64_t> g_outfs, g_infs;

void init_model(
    torch::Tensor embed_table, torch::Tensor pos_gpu,
    std::vector<torch::Tensor> kv_caches,
    std::vector<torch::Tensor> in_norms,
    std::vector<torch::Tensor> post_norms,
    torch::Tensor final_norm_w,
    std::vector<torch::Tensor> codes,
    std::vector<torch::Tensor> cbs,
    std::vector<torch::Tensor> s_i8s,
    std::vector<double> bases, std::vector<double> steps,
    std::vector<int64_t> out_fs, std::vector<int64_t> in_fs,
    torch::Tensor lh_codes, torch::Tensor lh_cb,
    torch::Tensor lh_s, double lh_base, double lh_step,
    int64_t lh_out_f, int64_t lh_in_f,
    int64_t n_heads, int64_t n_kv_heads,
    int64_t head_dim, int64_t ctx, double theta)
{
    g_embed = embed_table; g_pos = pos_gpu;
    g_kcs = kv_caches; g_inws = in_norms; g_postws = post_norms;
    g_fnw = final_norm_w;
    g_codes = codes; g_cbs = cbs; g_s = s_i8s;
    g_bases = bases; g_steps = steps;
    g_outfs = out_fs; g_infs = in_fs;
    g_lh_codes = lh_codes; g_lh_cb = lh_cb; g_lh_s = lh_s;
    g_lh_base = lh_base; g_lh_step = lh_step;
    g_lh_out_f = lh_out_f; g_lh_in_f = lh_in_f;
    g_nheads = n_heads; g_nkv = n_kv_heads;
    g_hd = head_dim; g_ctx = ctx; g_theta = theta;
    g_hidden = in_norms[0].numel();
    g_init = true;
}

// shared output buffers (used by step, step_graph, sf_step_graph
// so eager and graph paths are bit-comparable)
static torch::Tensor g_emb_buf, g_fn_buf, g_lg_buf, g_tok_gpu,
                     g_tok_hist;
static bool g_bufs_init = false;

static void ensure_g_bufs() {
    if (g_bufs_init) return;
    auto dev = g_embed.device();
    g_emb_buf = torch::empty({g_hidden},
        torch::TensorOptions().dtype(torch::kBFloat16)
            .device(dev));
    g_fn_buf = torch::empty({g_hidden},
        torch::TensorOptions().dtype(torch::kFloat32)
            .device(dev));
    g_lg_buf = torch::empty({g_lh_out_f},
        torch::TensorOptions().dtype(torch::kFloat32)
            .device(dev));
    g_tok_gpu = torch::zeros({1},
        torch::TensorOptions().dtype(torch::kInt64)
            .device(dev));
    g_tok_hist = torch::zeros({g_ctx},
        torch::TensorOptions().dtype(torch::kInt64)
            .device(dev));
    g_bufs_init = true;
}

int64_t step(int64_t token_id) {
    if (!g_init) throw std::runtime_error("init_model not called");
    auto st = at::cuda::getCurrentCUDAStream();
    ensure_g_bufs();

    // embedding lookup
    cudaMemcpyAsync(g_emb_buf.data_ptr(),
        reinterpret_cast<const char*>(g_embed.data_ptr())
            + token_id * g_hidden * 2,
        g_hidden * 2, cudaMemcpyDeviceToDevice, st);

    // set position
    set_pos_kernel<<<1, 1, 0, st>>>(
        reinterpret_cast<const int*>(g_pos.data_ptr()));

    // 24 layers
    auto h = g_emb_buf;
    for (int l = 0; l < (int)g_kcs.size(); ++l) {
        int b = l * 7;
        h = layer_forward(
            h, g_inws[l], g_postws[l],
            g_codes[b], g_cbs[b], g_s[b],
            g_bases[b], g_steps[b], g_outfs[b], g_infs[b],
            g_codes[b+1], g_cbs[b+1], g_s[b+1],
            g_bases[b+1], g_steps[b+1], g_outfs[b+1], g_infs[b+1],
            g_codes[b+2], g_cbs[b+2], g_s[b+2],
            g_bases[b+2], g_steps[b+2], g_outfs[b+2], g_infs[b+2],
            g_codes[b+3], g_cbs[b+3], g_s[b+3],
            g_bases[b+3], g_steps[b+3], g_outfs[b+3], g_infs[b+3],
            g_codes[b+4], g_cbs[b+4], g_s[b+4],
            g_bases[b+4], g_steps[b+4], g_outfs[b+4], g_infs[b+4],
            g_codes[b+5], g_cbs[b+5], g_s[b+5],
            g_bases[b+5], g_steps[b+5], g_outfs[b+5], g_infs[b+5],
            g_codes[b+6], g_cbs[b+6], g_s[b+6],
            g_bases[b+6], g_steps[b+6], g_outfs[b+6], g_infs[b+6],
            g_kcs[l], g_nheads, g_nkv, g_hd, g_ctx, g_theta);
    }

    // final norm + lm_head + argmax
    rmsnorm_kernel<<<1, 256, 0, st>>>(
        reinterpret_cast<const __nv_bfloat16*>(h.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(g_fnw.data_ptr()),
        g_fn_buf.data_ptr<float>(), (int)g_hidden);
    gsq_gemv_kernel<<<(unsigned)(g_lh_out_f / 8), 256, 0, st>>>(
        g_fn_buf.data_ptr<float>(),
        g_lh_codes.data_ptr<uint8_t>(),
        g_lh_cb.data_ptr<float>(),
        g_lh_s.data_ptr<uint8_t>(),
        (float)g_lh_base, (float)g_lh_step,
        g_lg_buf.data_ptr<float>(), (int)(g_lh_in_f / 16));
    argmax_f32<<<1, 256, 0, st>>>(
        g_lg_buf.data_ptr<float>(), (int)g_lh_out_f,
        g_tok_gpu.data_ptr<int64_t>());

    return g_tok_gpu.item<int64_t>();
}

// GPU-resident embedding lookup (no CPU sync for token_id)
__global__ void pos_incr(int* pos) { (*pos)++; }

__global__ void embed_lookup(const __nv_bfloat16* table,
                             const int64_t* token,
                             __nv_bfloat16* out, int hidden) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < hidden)
        out[i] = table[token[0] * hidden + i];
}

// record token into history at current d_pos (for batch reading)
__global__ void tok_record_kernel(const int64_t* tok,
                                  int64_t* hist) {
    hist[d_pos] = tok[0];
}

// Batch generation: token stays on GPU, zero CPU sync per token
std::vector<int64_t> generate_batch_fast(int64_t start_token,
                                         int64_t n_tokens) {
    if (!g_init) throw std::runtime_error("init_model not called");
    auto st = at::cuda::getCurrentCUDAStream();
    int nl = (int)g_kcs.size();

    static torch::Tensor emb_buf, fn_buf, lg_buf,
                           tok_gpu;
    if (!emb_buf.defined()) {
        auto dev = g_embed.device();
        emb_buf = torch::empty({g_hidden},
            torch::TensorOptions().dtype(torch::kBFloat16)
                .device(dev));
        fn_buf = torch::empty({g_hidden},
            torch::TensorOptions().dtype(torch::kFloat32)
                .device(dev));
        lg_buf = torch::empty({g_lh_out_f},
            torch::TensorOptions().dtype(torch::kFloat32)
                .device(dev));
        tok_gpu = torch::zeros({1},
            torch::TensorOptions().dtype(torch::kInt64)
                .device(dev));
    }
    tok_gpu.fill_(start_token);

    for (int i = 0; i < (int)n_tokens; ++i) {
        // increment position on GPU
        pos_incr<<<1, 1, 0, st>>>(
            reinterpret_cast<int*>(g_pos.data_ptr()));
        // set device pos
        set_pos_kernel<<<1, 1, 0, st>>>(
            reinterpret_cast<const int*>(g_pos.data_ptr()));
        // GPU-resident embedding lookup
        embed_lookup<<<(unsigned)((g_hidden + 255) / 256), 256, 0, st>>>(
            reinterpret_cast<const __nv_bfloat16*>(
                g_embed.data_ptr()),
            tok_gpu.data_ptr<int64_t>(),
            reinterpret_cast<__nv_bfloat16*>(emb_buf.data_ptr()),
            (int)g_hidden);
        // 24 layers
        auto h = emb_buf;
        for (int l = 0; l < nl; ++l) {
            int b = l * 7;
            h = layer_forward(
                h, g_inws[l], g_postws[l],
                g_codes[b], g_cbs[b], g_s[b],
                g_bases[b], g_steps[b], g_outfs[b], g_infs[b],
                g_codes[b+1], g_cbs[b+1], g_s[b+1],
                g_bases[b+1], g_steps[b+1],
                g_outfs[b+1], g_infs[b+1],
                g_codes[b+2], g_cbs[b+2], g_s[b+2],
                g_bases[b+2], g_steps[b+2],
                g_outfs[b+2], g_infs[b+2],
                g_codes[b+3], g_cbs[b+3], g_s[b+3],
                g_bases[b+3], g_steps[b+3],
                g_outfs[b+3], g_infs[b+3],
                g_codes[b+4], g_cbs[b+4], g_s[b+4],
                g_bases[b+4], g_steps[b+4],
                g_outfs[b+4], g_infs[b+4],
                g_codes[b+5], g_cbs[b+5], g_s[b+5],
                g_bases[b+5], g_steps[b+5],
                g_outfs[b+5], g_infs[b+5],
                g_codes[b+6], g_cbs[b+6], g_s[b+6],
                g_bases[b+6], g_steps[b+6],
                g_outfs[b+6], g_infs[b+6],
                g_kcs[l], g_nheads, g_nkv, g_hd, g_ctx, g_theta);
        }
        // final norm + lm_head + argmax (all GPU)
        rmsnorm_kernel<<<1, 256, 0, st>>>(
            reinterpret_cast<const __nv_bfloat16*>(h.data_ptr()),
            reinterpret_cast<const __nv_bfloat16*>(
                g_fnw.data_ptr()),
            fn_buf.data_ptr<float>(), (int)g_hidden);
        gsq_gemv_kernel<<<(unsigned)(g_lh_out_f / 8), 256, 0, st>>>(
            fn_buf.data_ptr<float>(),
            g_lh_codes.data_ptr<uint8_t>(),
            g_lh_cb.data_ptr<float>(),
            g_lh_s.data_ptr<uint8_t>(),
            (float)g_lh_base, (float)g_lh_step,
            lg_buf.data_ptr<float>(), (int)(g_lh_in_f / 16));
        argmax_f32<<<1, 256, 0, st>>>(
            lg_buf.data_ptr<float>(), (int)g_lh_out_f,
            tok_gpu.data_ptr<int64_t>());
    }
    // SINGLE sync at the end: read final token
    return {tok_gpu.item<int64_t>()};
}

std::vector<int64_t> generate_batch(int64_t start_token,
                                    int64_t n_tokens) {
    std::vector<int64_t> out;
    int64_t tok = start_token;
    for (int i = 0; i < (int)n_tokens; ++i) {
        tok = step(tok);
        out.push_back(tok);
    }
    return out;
}

// Single-step, ALL-GPU, no .item() — PyTorch CUDA graph compatible.
// BUGFIX: set_pos_kernel was MISSING here — replays ran at the
// stale d_pos left by the last eager step() (position corruption).
void step_graph() {
    if (!g_init) throw std::runtime_error("init_model not called");
    auto st = at::cuda::getCurrentCUDAStream();
    ensure_g_bufs();
    int nl = (int)g_kcs.size();

    // sync device position from g_pos (graph-safe: fixed buffer)
    set_pos_kernel<<<1, 1, 0, st>>>(
        reinterpret_cast<const int*>(g_pos.data_ptr()));

    // NO feedback loop: input is g_emb_buf (set from Python),
    // output is g_tok_gpu (read from Python after replay).
    // The 24 layers chain through layer_forward's static buffers.
    auto h = g_emb_buf;
    for (int l = 0; l < nl; ++l) {
        int b = l * 7;
        h = layer_forward(
            h, g_inws[l], g_postws[l],
            g_codes[b], g_cbs[b], g_s[b],
            g_bases[b], g_steps[b], g_outfs[b], g_infs[b],
            g_codes[b+1], g_cbs[b+1], g_s[b+1],
            g_bases[b+1], g_steps[b+1], g_outfs[b+1], g_infs[b+1],
            g_codes[b+2], g_cbs[b+2], g_s[b+2],
            g_bases[b+2], g_steps[b+2], g_outfs[b+2], g_infs[b+2],
            g_codes[b+3], g_cbs[b+3], g_s[b+3],
            g_bases[b+3], g_steps[b+3], g_outfs[b+3], g_infs[b+3],
            g_codes[b+4], g_cbs[b+4], g_s[b+4],
            g_bases[b+4], g_steps[b+4], g_outfs[b+4], g_infs[b+4],
            g_codes[b+5], g_cbs[b+5], g_s[b+5],
            g_bases[b+5], g_steps[b+5], g_outfs[b+5], g_infs[b+5],
            g_codes[b+6], g_cbs[b+6], g_s[b+6],
            g_bases[b+6], g_steps[b+6], g_outfs[b+6], g_infs[b+6],
            g_kcs[l], g_nheads, g_nkv, g_hd, g_ctx, g_theta);
    }
    rmsnorm_kernel<<<1, 256, 0, st>>>(
        reinterpret_cast<const __nv_bfloat16*>(h.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(g_fnw.data_ptr()),
        g_fn_buf.data_ptr<float>(), (int)g_hidden);
    gsq_gemv_kernel<<<(unsigned)(g_lh_out_f / 8), 256, 0, st>>>(
        g_fn_buf.data_ptr<float>(),
        g_lh_codes.data_ptr<uint8_t>(),
        g_lh_cb.data_ptr<float>(),
        g_lh_s.data_ptr<uint8_t>(),
        (float)g_lh_base, (float)g_lh_step,
        g_lg_buf.data_ptr<float>(), (int)(g_lh_in_f / 16));
    argmax_f32<<<1, 256, 0, st>>>(
        g_lg_buf.data_ptr<float>(), (int)g_lh_out_f,
        g_tok_gpu.data_ptr<int64_t>());
}


void set_start_token(int64_t token) {
    ensure_g_bufs();  // pure alloc, no kernels
    g_tok_gpu.fill_(token);
}

void set_input_embedding(torch::Tensor embedding) {
    ensure_g_bufs();  // pure alloc, no kernels
    g_emb_buf.copy_(embedding);
}

int64_t get_last_token() {
    if (!g_bufs_init) throw std::runtime_error("no tokens generated");
    return g_tok_gpu.item<int64_t>();
}

// ---------------- SELF-FEEDING graph (zero Python per token) ---------- //
// The graph closes the loop ON GPU through static buffers:
//   set_pos(g_pos) -> embed_lookup(g_tok_gpu -> g_emb_buf)
//   -> 24 layers -> final norm -> lm_head -> argmax -> g_tok_gpu
//   -> tok_record(hist[d_pos]) -> pos_incr(g_pos)
// Each replay = 1 full token. Python: seed once, replay N, read hist.
void sf_step_graph() {
    if (!g_init) throw std::runtime_error("init_model not called");
    auto st = at::cuda::getCurrentCUDAStream();
    ensure_g_bufs();
    int nl = (int)g_kcs.size();

    // 1. d_pos = *g_pos (graph-safe: fixed buffer address)
    set_pos_kernel<<<1, 1, 0, st>>>(
        reinterpret_cast<const int*>(g_pos.data_ptr()));

    // 2. embedding lookup from GPU-resident token
    embed_lookup<<<(unsigned)((g_hidden + 255) / 256), 256, 0, st>>>(
        reinterpret_cast<const __nv_bfloat16*>(g_embed.data_ptr()),
        g_tok_gpu.data_ptr<int64_t>(),
        reinterpret_cast<__nv_bfloat16*>(g_emb_buf.data_ptr()),
        (int)g_hidden);

    // 3. 24 layers
    auto h = g_emb_buf;
    for (int l = 0; l < nl; ++l) {
        int b = l * 7;
        h = layer_forward(
            h, g_inws[l], g_postws[l],
            g_codes[b], g_cbs[b], g_s[b],
            g_bases[b], g_steps[b], g_outfs[b], g_infs[b],
            g_codes[b+1], g_cbs[b+1], g_s[b+1],
            g_bases[b+1], g_steps[b+1], g_outfs[b+1], g_infs[b+1],
            g_codes[b+2], g_cbs[b+2], g_s[b+2],
            g_bases[b+2], g_steps[b+2], g_outfs[b+2], g_infs[b+2],
            g_codes[b+3], g_cbs[b+3], g_s[b+3],
            g_bases[b+3], g_steps[b+3], g_outfs[b+3], g_infs[b+3],
            g_codes[b+4], g_cbs[b+4], g_s[b+4],
            g_bases[b+4], g_steps[b+4], g_outfs[b+4], g_infs[b+4],
            g_codes[b+5], g_cbs[b+5], g_s[b+5],
            g_bases[b+5], g_steps[b+5], g_outfs[b+5], g_infs[b+5],
            g_codes[b+6], g_cbs[b+6], g_s[b+6],
            g_bases[b+6], g_steps[b+6], g_outfs[b+6], g_infs[b+6],
            g_kcs[l], g_nheads, g_nkv, g_hd, g_ctx, g_theta);
    }

    // 4. final norm
    rmsnorm_kernel<<<1, 256, 0, st>>>(
        reinterpret_cast<const __nv_bfloat16*>(h.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(g_fnw.data_ptr()),
        g_fn_buf.data_ptr<float>(), (int)g_hidden);

    // 5. lm_head GEMV
    gsq_gemv_kernel<<<(unsigned)(g_lh_out_f / 8), 256, 0, st>>>(
        g_fn_buf.data_ptr<float>(),
        g_lh_codes.data_ptr<uint8_t>(),
        g_lh_cb.data_ptr<float>(),
        g_lh_s.data_ptr<uint8_t>(),
        (float)g_lh_base, (float)g_lh_step,
        g_lg_buf.data_ptr<float>(), (int)(g_lh_in_f / 16));

    // 6. argmax -> g_tok_gpu (feeds NEXT replay's embed_lookup)
    argmax_f32<<<1, 256, 0, st>>>(
        g_lg_buf.data_ptr<float>(), (int)g_lh_out_f,
        g_tok_gpu.data_ptr<int64_t>());

    // 7. record token history (batch read later, zero per-token sync)
    tok_record_kernel<<<1, 1, 0, st>>>(
        g_tok_gpu.data_ptr<int64_t>(),
        g_tok_hist.data_ptr<int64_t>());

    // 8. advance position for next replay
    pos_incr<<<1, 1, 0, st>>>(
        reinterpret_cast<int*>(g_pos.data_ptr()));
}

void sf_seed(int64_t token, int64_t pos) {
    // PURE init: allocate buffers WITHOUT running any kernels.
    // (The old lazy sf_step_graph() call ran at the STALE g_pos,
    //  clobbering the last real cache slot with token-0 garbage.)
    ensure_g_bufs();
    g_tok_gpu.fill_(token);
    g_pos.fill_((int32_t)pos);
}

torch::Tensor sf_get_hist(int64_t from, int64_t n) {
    if (!g_bufs_init) throw std::runtime_error("no history");
    return g_tok_hist.narrow(0, from, n).cpu();
}

torch::Tensor sf_get_tok() {
    return g_tok_gpu.cpu();
}

torch::Tensor sf_get_emb() {
    return g_emb_buf.cpu();
}

torch::Tensor sf_get_lg() {
    return g_lg_buf.cpu();
}

torch::Tensor sf_get_fn() {
    return g_fn_buf.cpu();
}

// ---------------- CUDA Graph with raw cudaMalloc input (WDDM-safe) ---- //
static cudaGraphExec_t raw_exec = nullptr;
static cudaGraph_t raw_graph = nullptr;
static bool raw_captured = false;
static float* raw_input = nullptr;
static float* raw_output = nullptr;
static int raw_n = 0;

__global__ void copy_kernel(const float* src, float* dst, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = src[i] * 2.0f;
}

void raw_graph_init(int64_t n) {
    raw_n = (int)n;
    if (raw_input) cudaFree(raw_input);
    if (raw_output) cudaFree(raw_output);
    cudaMalloc(&raw_input, raw_n * sizeof(float));
    cudaMalloc(&raw_output, raw_n * sizeof(float));
    cudaMemset(raw_input, 0, raw_n * sizeof(float));
    cudaMemset(raw_output, 0, raw_n * sizeof(float));
}

void raw_graph_set_input(torch::Tensor data) {
    cudaMemcpy(raw_input, data.data_ptr<float>(),
               raw_n * sizeof(float),
               cudaMemcpyDeviceToDevice);
}

torch::Tensor raw_graph_get_output() {
    return torch::from_blob(raw_output, {raw_n},
        torch::TensorOptions().dtype(torch::kFloat32)
            .device(torch::kCUDA)).clone();
}

void raw_graph_capture() {
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    copy_kernel<<<1, 256, 0, stream>>>(
        raw_input, raw_output, raw_n);
    cudaStreamSynchronize(stream);
    cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
    copy_kernel<<<1, 256, 0, stream>>>(
        raw_input, raw_output, raw_n);
    cudaStreamEndCapture(stream, &raw_graph);
    cudaGraphInstantiate(&raw_exec, raw_graph, NULL, NULL, 0);
    raw_captured = true;
}

void raw_graph_replay() {
    if (!raw_captured) throw std::runtime_error("not captured");
    cudaGraphLaunch(raw_exec,
                    at::cuda::getCurrentCUDAStream().stream());
}

static cudaGraphExec_t g_exec = nullptr;
static cudaGraph_t g_graph = nullptr;
static bool g_captured = false;
static torch::Tensor g_result;
static torch::Tensor g_input;   // static input buffer (address baked in graph)

void graph_capture(
    torch::Tensor h_bf16,
    torch::Tensor pos_gpu,
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
    int64_t n_heads, int64_t n_kv_heads,
    int64_t head_dim, int64_t ctx, double theta)
{
    auto stream = at::cuda::getCurrentCUDAStream().stream();

    // Store REFERENCE (not clone) — Python owns the tensor and
    // updates it in-place between replays. Address never changes.
    g_input = h_bf16;

    // Warmup (initializes static buffers)
    g_result = decode_24(g_input, pos_gpu, kv_caches, in_norms,
                         post_norms, codes, cbs, s_i8s, bases,
                         steps, out_fs, in_fs,
                         n_heads, n_kv_heads, head_dim, ctx, theta);
    cudaStreamSynchronize(stream);

    // Capture (same input address, same buffers)
    cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
    g_result = decode_24(g_input, pos_gpu, kv_caches, in_norms,
                         post_norms, codes, cbs, s_i8s, bases,
                         steps, out_fs, in_fs,
                         n_heads, n_kv_heads, head_dim, ctx, theta);
    cudaStreamEndCapture(stream, &g_graph);
    cudaGraphInstantiate(&g_exec, g_graph, NULL, NULL, 0);
    g_captured = true;
}

void graph_set_input(torch::Tensor embedding) {
    if (!g_captured || !g_input.defined()) {
        throw std::runtime_error("graph not captured");
    }
    g_input.copy_(embedding);
}

torch::Tensor graph_replay() {
    if (!g_captured) {
        throw std::runtime_error(
            "graph not captured - call graph_capture first");
    }
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    cudaGraphLaunch(g_exec, stream);
    return g_result;
}
