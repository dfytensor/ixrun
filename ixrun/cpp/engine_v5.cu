// IXRUN C++ engine v5 鈥?clean rewrite, all-fp32 internal chain.
// GSQ 5.5bpw weights, bf16 model boundaries, fp32 computation.
// Lessons applied: single dtype per kernel, no mixed reinterpret,
// all kernels on current stream, no split-K (keep it simple first).
#include <torch/extension.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <ATen/cuda/CUDAContext.h>
#include <cmath>

// ---------------- device-side position (CUDA-graph parameterizable) --- //
__device__ int d_pos = 0;
extern int s27_probe_l;         // defined in engine_27b.cu
extern torch::Tensor s27_h_out; // defined in engine_27b.cu
static torch::Tensor s27d_gated; // probe: gated norm out
static torch::Tensor s27d_o;     // probe: pre-gated o

__global__ void set_pos_kernel(const int* pos_gpu) {
    d_pos = *pos_gpu;
}

// host-value variant: kernel args are captured at ENQUEUE time, so a
// host loop can enqueue all iterations without a pinned-buffer race.
__global__ void write_pos_kernel(int* pos_gpu, int val) {
    *pos_gpu = val;
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

// ---------------- GSQ GEMV v2 (llama.cpp-style: smem x + coop) ---- //
// Same math order as v1 (bit-exact goal): warp-per-row, per-lane
// group-stride, group dot in i=0..15 order. Adds: x staged in smem
// once per block (all warps share), cb in smem (as v1).
__global__ void gsq_gemv_v2_kernel(
    const float* __restrict__ x,
    const uint8_t* __restrict__ codes,
    const float* __restrict__ cb,
    const uint8_t* __restrict__ s_i8,
    float s_base, float s_step,
    float* __restrict__ yf,
    int n_gr, int in_f)
{
    __shared__ float cb_sm[32];
    extern __shared__ float x_sm[];   // in_f floats, dynamic smem
    for (int i = threadIdx.x; i < 32; i += blockDim.x)
        cb_sm[i] = cb[i];
    for (int i = threadIdx.x; i < in_f; i += blockDim.x)
        x_sm[i] = x[i];
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
        const float* x16 = x_sm + jb * 16;
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

// launcher: v2 with dynamic smem for x staging
static torch::Tensor gsq_gemv_v2(torch::Tensor x_f32,
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
    gsq_gemv_v2_kernel<<<(unsigned)(out_f / wpb), wpb * 32,
                         (size_t)in_f * sizeof(float), st>>>(
        x_f32.data_ptr<float>(),
        codes.data_ptr<uint8_t>(), cb.data_ptr<float>(),
        s_i8.data_ptr<uint8_t>(), (float)s_base, (float)s_step,
        y.data_ptr<float>(), n_gr, (int)in_f);
    return y;
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
        red[0] = __frsqrt_rn(t / (float)n + 1e-5f);
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

// ---------------- attention v2 (smem weights, bit-exact) ------------- //
// v1 recomputed the full score scan in EVERY thread (128x) and again
// in pass 2. v2: threads split the t-range for scores (each t computed
// once, same j-order), block-max reduce (exact, order-free), then the
// weighted-sum pass reads smem weights -> identical arithmetic, ~2x
// faster and 128x less score work.
__global__ void attn_kernel_v2(
    const float* __restrict__ q,
    const __nv_bfloat16* __restrict__ kv,
    float* __restrict__ out,
    int n_heads, int n_kv_heads,
    int head_dim, int ctx)
{
    int pos = d_pos;
    int h = blockIdx.x;
    int d = threadIdx.x;
    int kvh = h / (n_heads / n_kv_heads);
    const __nv_bfloat16* ks = kv + kvh * ctx * head_dim;
    const __nv_bfloat16* vs = kv + (long long)n_kv_heads*ctx*head_dim
                              + kvh * ctx * head_dim;
    float inv_hd = rsqrtf((float)head_dim);
    extern __shared__ float w_sm[];          // ctx floats
    __shared__ float red[32];
    // 1. scores for t = d, d+blockDim, ... (same j-order as v1)
    float maxs = -1e30f;
    for (int t = d; t <= pos; t += blockDim.x) {
        float sc = 0.f;
        for (int j = 0; j < head_dim; ++j)
            sc += q[h*head_dim + j]
                * __bfloat162float(ks[t*head_dim + j]);
        sc *= inv_hd;
        w_sm[t] = sc;
        if (sc > maxs) maxs = sc;
    }
    // 2. block max (exact 鈥?no rounding, order-free)
    if (threadIdx.x < 32) red[threadIdx.x] = -1e30f;
    __syncthreads();
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        maxs = fmaxf(maxs, __shfl_down_sync(0xffffffff, maxs, off));
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = maxs;
    __syncthreads();
    if (threadIdx.x == 0) {
        float m = -1e30f;
        for (int k = 0; k < (int)(blockDim.x >> 5); ++k)
            m = fmaxf(m, red[k]);
        red[0] = m;
    }
    __syncthreads();
    maxs = red[0];
    // 3. weighted sum (same t-order, same exp arg as v1)
    float denom = 0.f, sum = 0.f;
    for (int t = 0; t <= pos; ++t) {
        float sc = expf(w_sm[t] - maxs);
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
    torch::Tensor h,           // bf16 [hidden] 鈥?previous layer output
    torch::Tensor in_nw,       // bf16 [hidden] 鈥?input_layernorm weight
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
    attn_kernel_v2<<<(unsigned)n_heads, (unsigned)hd,
                     (size_t)ctx * sizeof(float), st>>>(
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

    // 8. post norm: bf16 copy of b_h1f 鈫?rmsnorm
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

// probe: raw gemv v2 (smem-x variant) for bit-exact + speed gate
torch::Tensor gemv_v2_out(torch::Tensor x,
                          torch::Tensor codes, torch::Tensor cb,
                          torch::Tensor s_i8,
                          double s_base, double s_step,
                          int64_t out_f, int64_t in_f) {
    return gsq_gemv_v2(x, codes, cb, s_i8, s_base, s_step,
                       out_f, in_f);
}

// ---------------- UDCQ GEMV (27B format port, stage 1) ---------------- //
// w = sign * scale_g * CB[idx]; byte-aligned nibbles, pure LUT walk.
// Gate: rel-err vs fp64 decode reference (fp32 reorder noise tier).
//
// ULS variant (log2 scale): scale stream = uint8[out, 8+n_gr] where each row
// head is (float base, float step) and scale(g) = exp2(base + step*i8[g]).
// 5.5bpw vs 6.0; toggled process-wide via udcq_set_uls(1).
static int g_uls = 0;

void udcq_set_uls(int64_t on) { g_uls = (on != 0) ? 1 : 0; }

__device__ __forceinline__ float uls_scale_row(const uint8_t* srow, int g) {
    float base = *reinterpret_cast<const float*>(srow);
    float step = *reinterpret_cast<const float*>(srow + 4);
    return exp2f(base + step * (float)srow[8 + g]);
}

template <bool ULS>
__global__ void udcq_gemv_kernel(
    const float* __restrict__ x,
    const uint8_t* __restrict__ idx,
    const uint32_t* __restrict__ sign,
    const __half* __restrict__ scale,          // f16-resident
    const float* __restrict__ cb,              // [16] fp32
    float* __restrict__ y,
    int in_f, int GROUP)
{
    __shared__ float red[32];
    int r = blockIdx.x;
    long long base = (long long)r * in_f;
    const uint8_t* urow = reinterpret_cast<const uint8_t*>(scale)
                          + (size_t)r * (size_t)(in_f / GROUP + 8);
    float acc = 0.f;
    for (int j = threadIdx.x; j < in_f; j += blockDim.x) {
        long long o = base + j;
        uint8_t b = idx[o >> 1];
        int nib = (o & 1) ? ((b >> 4) & 0x0F) : (b & 0x0F);
        float sc = ULS ? uls_scale_row(urow, j / GROUP)
                       : __half2float(scale[o / GROUP]);
        uint32_t sw = sign[o >> 5];
        float sgn = ((sw >> (o & 31)) & 1u) ? 1.f : -1.f;
        acc += cb[nib] * sc * sgn * x[j];
    }
    if (threadIdx.x < 32) red[threadIdx.x] = 0.f;
    __syncthreads();
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        acc += __shfl_down_sync(0xffffffff, acc, off);
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = acc;
    __syncthreads();
    if (threadIdx.x == 0) {
        float t = 0.f;
        for (int k = 0; k < (int)(blockDim.x >> 5); ++k) t += red[k];
        y[r] = t;
    }
}

static void udcq_gemv_launch(
    const float* x, const uint8_t* idx, const uint32_t* sign,
    const __half* scale, const float* cb, float* y,
    int out_f, int in_f, int group, cudaStream_t st);

torch::Tensor udcq_gemv_out(torch::Tensor x,
                            torch::Tensor idx, torch::Tensor sign,
                            torch::Tensor scale, torch::Tensor cb,
                            int64_t out_f, int64_t in_f,
                            int64_t group) {
    auto y = torch::zeros({out_f}, torch::dtype(torch::kFloat32)
                                         .device(x.device()));
    udcq_gemv_launch(
        x.data_ptr<float>(), idx.data_ptr<uint8_t>(),
        reinterpret_cast<const uint32_t*>(sign.data_ptr()),
        reinterpret_cast<const __half*>(scale.data_ptr()),
        cb.data_ptr<float>(), y.data_ptr<float>(),
        (int)out_f, (int)in_f, (int)group,
        at::cuda::getCurrentCUDAStream());
    return y;
}

// UDCQ GEMV v2: warp-per-row + x staged in smem once per block (8 rows).
// v1 re-read the whole x per row => ~5x L2 traffic; v2 streams only the
// weight bytes. Per 16-elem group: 8B idx (uint2) + 2B scale + 4B sign
// word (shared by 2 groups) + x from smem. Not bit-identical to v1
// (different reduction order); gate vs fp64 at the 1e-7 tier.
template <bool ULS>
__global__ void udcq_gemv_v2_kernel(
    const float* __restrict__ x,
    const uint8_t* __restrict__ idx,
    const uint32_t* __restrict__ sign,
    const __half* __restrict__ scale,
    const float* __restrict__ cb,
    float* __restrict__ y,
    int in_f, int out_f, int GROUP)
{
    extern __shared__ float smem[];
    float* x_sm = smem;              // in_f floats
    float* cb_sm = smem + in_f;      // 32 floats: [0..15]=cb, [16..31]=-cb
    for (int i = threadIdx.x; i < 16; i += blockDim.x) {
        cb_sm[i] = -cb[i];           // sign bit 0 -> negative
        cb_sm[i + 16] = cb[i];       // sign bit 1 -> positive
    }
    for (int i = threadIdx.x; i < in_f; i += blockDim.x) x_sm[i] = x[i];
    __syncthreads();
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    int r = blockIdx.x * (blockDim.x >> 5) + warp;
    if (r >= out_f) return;
    int n_gr = in_f / GROUP;
    const uint8_t* irow = idx + (size_t)r * (in_f / 2);
    const __half* srow = scale + (size_t)r * n_gr;
    const uint8_t* urow = reinterpret_cast<const uint8_t*>(scale)
                          + (size_t)r * (size_t)(n_gr + 8);
    float sbase = 0.f, sstep = 0.f;
    if (ULS) {
        sbase = *reinterpret_cast<const float*>(urow);
        sstep = *reinterpret_cast<const float*>(urow + 4);
    }
    const uint32_t* wrow = sign + (size_t)r * (in_f / 32);
    float acc = 0.f;
    for (int g = lane; g < n_gr; g += 32) {
        uint2 b2 = *reinterpret_cast<const uint2*>(irow + (size_t)g * 8);
        uint32_t sw = wrow[g >> 1] >> (16 * (g & 1));
        float sc = ULS ? exp2f(sbase + sstep * (float)urow[8 + g])
                       : __half2float(srow[g]);
        const float* xs = x_sm + (size_t)g * GROUP;
        float xr[16];
        #pragma unroll
        for (int q = 0; q < 4; ++q)
            *reinterpret_cast<float4*>(&xr[q * 4]) =
                *reinterpret_cast<const float4*>(xs + q * 4);
        float inner = 0.f;
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            int b = (int)((b2.x >> (8 * i)) & 0xFF);
            int s0 = (int)((sw >> (2 * i)) & 1u) << 4;
            int s1 = (int)((sw >> (2 * i + 1)) & 1u) << 4;
            inner = fmaf(cb_sm[(b & 0xF) | s0], xr[2 * i], inner);
            inner = fmaf(cb_sm[(b >> 4) | s1], xr[2 * i + 1], inner);
        }
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            int b = (int)((b2.y >> (8 * i)) & 0xFF);
            int s0 = (int)((sw >> (8 + 2 * i)) & 1u) << 4;
            int s1 = (int)((sw >> (8 + 2 * i + 1)) & 1u) << 4;
            inner = fmaf(cb_sm[(b & 0xF) | s0], xr[8 + 2 * i], inner);
            inner = fmaf(cb_sm[(b >> 4) | s1], xr[8 + 2 * i + 1], inner);
        }
        acc += inner * sc;
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        acc += __shfl_down_sync(0xffffffff, acc, off);
    if (lane == 0) y[r] = acc;
}

// dispatcher: v2 for the supported format, v1 fallback otherwise
static void udcq_gemv_launch(
    const float* x, const uint8_t* idx, const uint32_t* sign,
    const __half* scale, const float* cb, float* y,
    int out_f, int in_f, int group, cudaStream_t st)
{
    if (group == 16 && in_f % 32 == 0 && in_f > 0
        && (long long)out_f * in_f >= 8LL * 1024 * 1024) {
        static int max_sm = -1;
        if (max_sm < 0) {
            int dev = 0;
            cudaGetDevice(&dev);
            cudaDeviceGetAttribute(&max_sm,
                cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
            if (max_sm > 0) {
                cudaFuncSetAttribute(udcq_gemv_v2_kernel<false>,
                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                    max_sm);
                cudaFuncSetAttribute(udcq_gemv_v2_kernel<true>,
                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                    max_sm);
            }
        }
        size_t sm = (size_t)in_f * sizeof(float) + 16 * sizeof(float);
        if ((int)sm <= max_sm) {
            int wpb = (in_f > 12288) ? 16 : 8;   // big smem -> 1 blk/SM: use all warps
            if (g_uls)
                udcq_gemv_v2_kernel<true><<<(out_f + wpb - 1) / wpb, wpb * 32, sm, st>>>(
                    x, idx, sign, scale, cb, y, in_f, out_f, group);
            else
                udcq_gemv_v2_kernel<false><<<(out_f + wpb - 1) / wpb, wpb * 32, sm, st>>>(
                    x, idx, sign, scale, cb, y, in_f, out_f, group);
            return;
        }
    }
    if (g_uls)
        udcq_gemv_kernel<true><<<(unsigned)out_f, 256, 0, st>>>(
            x, idx, sign, scale, cb, y, in_f, group);
    else
        udcq_gemv_kernel<false><<<(unsigned)out_f, 256, 0, st>>>(
            x, idx, sign, scale, cb, y, in_f, group);
}

// UDCQ GEMV multi-token T=4: weights decoded ONCE per 4 tokens.
// Same warp-per-row walk + same per-group fmaf sequence as v2 for each
// token => bit-exact vs 4 separate v2 calls. x read directly as float4
// from global (L1/L2-hot). Measured 1.65x vs 4xT=1 (smem-chunked variant
// is SLOWER: 1 blk/SM occupancy loss > L2 savings). Keep for any future
// speculative path; NOT deployed (T=1 kernel is already near-BW).
template <bool ULS>
__global__ void udcq_gemv_mt4_kernel(
    const float* __restrict__ x,      // [4, in_f]
    const uint8_t* __restrict__ idx,
    const uint32_t* __restrict__ sign,
    const __half* __restrict__ scale,
    const float* __restrict__ cb,
    float* __restrict__ y,            // [4, out_f]
    int in_f, int out_f, int GROUP)
{
    __shared__ float cb_sm[32];
    for (int i = threadIdx.x; i < 16; i += blockDim.x) {
        cb_sm[i] = -cb[i];
        cb_sm[i + 16] = cb[i];
    }
    __syncthreads();
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    int r = blockIdx.x * (blockDim.x >> 5) + warp;
    int n_gr = in_f / GROUP;
    const uint8_t* irow = idx + (size_t)r * (in_f / 2);
    const __half* srow = scale + (size_t)r * n_gr;
    const uint8_t* urow = reinterpret_cast<const uint8_t*>(scale)
                          + (size_t)r * (size_t)(n_gr + 8);
    float sbase = 0.f, sstep = 0.f;
    if (ULS) {
        sbase = *reinterpret_cast<const float*>(urow);
        sstep = *reinterpret_cast<const float*>(urow + 4);
    }
    const uint32_t* wrow = sign + (size_t)r * (in_f / 32);
    const float* xp0 = x;
    const float* xp1 = x + in_f;
    const float* xp2 = x + 2 * (size_t)in_f;
    const float* xp3 = x + 3 * (size_t)in_f;
    float acc0 = 0.f, acc1 = 0.f, acc2 = 0.f, acc3 = 0.f;
    for (int g = lane; g < n_gr; g += 32) {
        uint2 b2 = *reinterpret_cast<const uint2*>(irow + (size_t)g * 8);
        uint32_t sw = wrow[g >> 1] >> (16 * (g & 1));
        float sc = ULS ? exp2f(sbase + sstep * (float)urow[8 + g])
                       : __half2float(srow[g]);
        const size_t xo = (size_t)g * GROUP;
        float in0 = 0.f, in1 = 0.f, in2 = 0.f, in3 = 0.f;
        #pragma unroll
        for (int q = 0; q < 4; ++q) {
            float4 v0 = *reinterpret_cast<const float4*>(xp0 + xo + 4 * q);
            float4 v1 = *reinterpret_cast<const float4*>(xp1 + xo + 4 * q);
            float4 v2 = *reinterpret_cast<const float4*>(xp2 + xo + 4 * q);
            float4 v3 = *reinterpret_cast<const float4*>(xp3 + xo + 4 * q);
            uint32_t w = (q < 2) ? b2.x : b2.y;
            int bo = (q & 1) * 16;
            int so = q * 4;
            int b0 = (int)((w >> bo) & 0xFF);
            int b1 = (int)((w >> (bo + 8)) & 0xFF);
            float c00 = cb_sm[(b0 & 0xF) | ((int)((sw >> so) & 1u) << 4)];
            float c01 = cb_sm[(b0 >> 4) | ((int)((sw >> (so + 1)) & 1u) << 4)];
            float c10 = cb_sm[(b1 & 0xF) | ((int)((sw >> (so + 2)) & 1u) << 4)];
            float c11 = cb_sm[(b1 >> 4) | ((int)((sw >> (so + 3)) & 1u) << 4)];
            in0 = fmaf(c00, v0.x, in0); in0 = fmaf(c01, v0.y, in0);
            in0 = fmaf(c10, v0.z, in0); in0 = fmaf(c11, v0.w, in0);
            in1 = fmaf(c00, v1.x, in1); in1 = fmaf(c01, v1.y, in1);
            in1 = fmaf(c10, v1.z, in1); in1 = fmaf(c11, v1.w, in1);
            in2 = fmaf(c00, v2.x, in2); in2 = fmaf(c01, v2.y, in2);
            in2 = fmaf(c10, v2.z, in2); in2 = fmaf(c11, v2.w, in2);
            in3 = fmaf(c00, v3.x, in3); in3 = fmaf(c01, v3.y, in3);
            in3 = fmaf(c10, v3.z, in3); in3 = fmaf(c11, v3.w, in3);
        }
        acc0 += in0 * sc;
        acc1 += in1 * sc;
        acc2 += in2 * sc;
        acc3 += in3 * sc;
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        acc0 += __shfl_down_sync(0xffffffff, acc0, off);
        acc1 += __shfl_down_sync(0xffffffff, acc1, off);
        acc2 += __shfl_down_sync(0xffffffff, acc2, off);
        acc3 += __shfl_down_sync(0xffffffff, acc3, off);
    }
    if (lane == 0) {
        y[r] = acc0;
        y[(size_t)out_f + r] = acc1;
        y[(size_t)2 * out_f + r] = acc2;
        y[(size_t)3 * out_f + r] = acc3;
    }
}

static void udcq_gemv_mt4_launch(
    const float* x, const uint8_t* idx, const uint32_t* sign,
    const __half* scale, const float* cb, float* y,
    int out_f, int in_f, int group, cudaStream_t st)
{
    if (g_uls)
        udcq_gemv_mt4_kernel<true><<<(unsigned)((out_f + 7) / 8), 256, 0, st>>>(
            x, idx, sign, scale, cb, y, in_f, out_f, group);
    else
        udcq_gemv_mt4_kernel<false><<<(unsigned)((out_f + 7) / 8), 256, 0, st>>>(
            x, idx, sign, scale, cb, y, in_f, out_f, group);
}

torch::Tensor udcq_gemv_mt4_out(torch::Tensor x, torch::Tensor idx,
                                torch::Tensor sign, torch::Tensor scale,
                                torch::Tensor cb,
                                int64_t out_f, int64_t in_f,
                                int64_t group) {
    auto y = torch::empty({4, out_f}, torch::dtype(torch::kFloat32)
                                          .device(x.device()));
    udcq_gemv_mt4_launch(
        x.data_ptr<float>(), idx.data_ptr<uint8_t>(),
        reinterpret_cast<const uint32_t*>(sign.data_ptr()),
        reinterpret_cast<const __half*>(scale.data_ptr()),
        cb.data_ptr<float>(), y.data_ptr<float>(),
        (int)out_f, (int)in_f, (int)group,
        at::cuda::getCurrentCUDAStream());
    return y;
}

// dual small-shape GEMV: two packs sharing one x, one launch.
// grid = 2 * ceil(out_f/8); first half -> pack0/y0, second -> pack1/y1.
// Same warp-per-row walk as v2 (x staged in smem).
template <bool ULS>
__global__ void udcq_gemv_dual_kernel(
    const float* __restrict__ x,
    const uint8_t* __restrict__ i0,
    const uint32_t* __restrict__ s0,
    const __half* __restrict__ sc0,
    const uint8_t* __restrict__ i1,
    const uint32_t* __restrict__ s1,
    const __half* __restrict__ sc1,
    const float* __restrict__ cb,
    float* __restrict__ y0, float* __restrict__ y1,
    int in_f, int out_f, int GROUP)
{
    extern __shared__ float smem[];
    float* x_sm = smem;
    float* cb_sm = smem + in_f;
    for (int i = threadIdx.x; i < 16; i += blockDim.x) {
        cb_sm[i] = -cb[i];
        cb_sm[i + 16] = cb[i];
    }
    for (int i = threadIdx.x; i < in_f; i += blockDim.x) x_sm[i] = x[i];
    __syncthreads();
    int half = gridDim.x >> 1;
    bool second = (int)blockIdx.x >= half;
    int blk = second ? (int)blockIdx.x - half : (int)blockIdx.x;
    const uint8_t* idx = second ? i1 : i0;
    const uint32_t* sign = second ? s1 : s0;
    const __half* scale = second ? sc1 : sc0;
    float* y = second ? y1 : y0;
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    int r = blk * (blockDim.x >> 5) + warp;
    if (r >= out_f) return;
    int n_gr = in_f / GROUP;
    const uint8_t* irow = idx + (size_t)r * (in_f / 2);
    const __half* srow = scale + (size_t)r * n_gr;
    const uint8_t* urow = reinterpret_cast<const uint8_t*>(scale)
                          + (size_t)r * (size_t)(n_gr + 8);
    float sbase = 0.f, sstep = 0.f;
    if (ULS) {
        sbase = *reinterpret_cast<const float*>(urow);
        sstep = *reinterpret_cast<const float*>(urow + 4);
    }
    const uint32_t* wrow = sign + (size_t)r * (in_f / 32);
    float acc = 0.f;
    for (int g = lane; g < n_gr; g += 32) {
        uint2 b2 = *reinterpret_cast<const uint2*>(irow + (size_t)g * 8);
        uint32_t sw = wrow[g >> 1] >> (16 * (g & 1));
        float sc = ULS ? exp2f(sbase + sstep * (float)urow[8 + g])
                       : __half2float(srow[g]);
        const float* xs = x_sm + (size_t)g * GROUP;
        float xr[16];
        #pragma unroll
        for (int q = 0; q < 4; ++q)
            *reinterpret_cast<float4*>(&xr[q * 4]) =
                *reinterpret_cast<const float4*>(xs + q * 4);
        float inner = 0.f;
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            int b = (int)((b2.x >> (8 * i)) & 0xFF);
            int s0_ = (int)((sw >> (2 * i)) & 1u) << 4;
            int s1_ = (int)((sw >> (2 * i + 1)) & 1u) << 4;
            inner = fmaf(cb_sm[(b & 0xF) | s0_], xr[2 * i], inner);
            inner = fmaf(cb_sm[(b >> 4) | s1_], xr[2 * i + 1], inner);
        }
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            int b = (int)((b2.y >> (8 * i)) & 0xFF);
            int s0_ = (int)((sw >> (8 + 2 * i)) & 1u) << 4;
            int s1_ = (int)((sw >> (8 + 2 * i + 1)) & 1u) << 4;
            inner = fmaf(cb_sm[(b & 0xF) | s0_], xr[8 + 2 * i], inner);
            inner = fmaf(cb_sm[(b >> 4) | s1_], xr[8 + 2 * i + 1], inner);
        }
        acc += inner * sc;
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        acc += __shfl_down_sync(0xffffffff, acc, off);
    if (lane == 0) y[r] = acc;
}

static void udcq_gemv_dual_launch(
    const float* x, const uint8_t* i0, const uint32_t* s0,
    const __half* sc0, const uint8_t* i1, const uint32_t* s1,
    const __half* sc1, const float* cb, float* y0, float* y1,
    int out_f, int in_f, int group, cudaStream_t st)
{
    size_t sm = (size_t)in_f * sizeof(float) + 32 * sizeof(float);
    unsigned nb = (unsigned)((out_f + 7) / 8);
    if (g_uls)
        udcq_gemv_dual_kernel<true><<<2 * nb, 256, sm, st>>>(
            x, i0, s0, sc0, i1, s1, sc1, cb, y0, y1, in_f, out_f, group);
    else
        udcq_gemv_dual_kernel<false><<<2 * nb, 256, sm, st>>>(
            x, i0, s0, sc0, i1, s1, sc1, cb, y0, y1, in_f, out_f, group);
}

// multi-pack GEMV: ONE launch covers several packs that share x (qkv+z+b+a
// in GDN layers; q+k+v and gate+up in attn layers). grid.x covers
// ceil(sum(out_f)/8) blocks; each warp locates its pack with a short linear
// scan over out_tab (npack <= 8). The per-pack walk is IDENTICAL to
// udcq_gemv_v2_kernel, so every output is bit-equal to the separate calls
// it replaces (the launch/dual-call overhead is what gets saved).
// tables layout per slot (288B): 8x ptr64 idx @0, 8x ptr64 sign @64,
// 8x ptr64 sc @128, 8x ptr64 y @192, 8x i32 out_f @256.
template <bool ULS>
__global__ void udcq_gemv_multi_kernel(
    const float* __restrict__ x,
    const uint8_t* const* __restrict__ idx_tab,
    const uint32_t* const* __restrict__ sign_tab,
    const __half* const* __restrict__ sc_tab,
    float* const* __restrict__ y_tab,
    const int* __restrict__ out_tab,
    const float* __restrict__ cb,
    int npack, int in_f, int GROUP)
{
    extern __shared__ float smem[];
    float* x_sm = smem;              // in_f floats
    float* cb_sm = smem + in_f;      // 32 floats: [0..15]=cb, [16..31]=-cb
    for (int i = threadIdx.x; i < 16; i += blockDim.x) {
        cb_sm[i] = -cb[i];
        cb_sm[i + 16] = cb[i];
    }
    for (int i = threadIdx.x; i < in_f; i += blockDim.x) x_sm[i] = x[i];
    __syncthreads();
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    int r_g = blockIdx.x * (blockDim.x >> 5) + warp;
    int total = 0;
    for (int p = 0; p < npack; ++p) total += out_tab[p];
    if (r_g >= total) return;
    int p = 0, rem = r_g;
    while (rem >= out_tab[p]) { rem -= out_tab[p]; ++p; }
    int n_gr = in_f / GROUP;
    const uint8_t* irow = idx_tab[p] + (size_t)rem * (in_f / 2);
    const uint32_t* wrow = sign_tab[p] + (size_t)rem * (in_f / 32);
    const __half* srow = sc_tab[p] + (size_t)rem * n_gr;
    const uint8_t* urow = reinterpret_cast<const uint8_t*>(sc_tab[p])
                          + (size_t)rem * (size_t)(n_gr + 8);
    float sbase = 0.f, sstep = 0.f;
    if (ULS) {
        sbase = *reinterpret_cast<const float*>(urow);
        sstep = *reinterpret_cast<const float*>(urow + 4);
    }
    float acc = 0.f;
    for (int g = lane; g < n_gr; g += 32) {
        uint2 b2 = *reinterpret_cast<const uint2*>(irow + (size_t)g * 8);
        uint32_t sw = wrow[g >> 1] >> (16 * (g & 1));
        float sc = ULS ? exp2f(sbase + sstep * (float)urow[8 + g])
                       : __half2float(srow[g]);
        const float* xs = x_sm + (size_t)g * GROUP;
        float xr[16];
        #pragma unroll
        for (int q = 0; q < 4; ++q)
            *reinterpret_cast<float4*>(&xr[q * 4]) =
                *reinterpret_cast<const float4*>(xs + q * 4);
        float inner = 0.f;
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            int b = (int)((b2.x >> (8 * i)) & 0xFF);
            int s0 = (int)((sw >> (2 * i)) & 1u) << 4;
            int s1 = (int)((sw >> (2 * i + 1)) & 1u) << 4;
            inner = fmaf(cb_sm[(b & 0xF) | s0], xr[2 * i], inner);
            inner = fmaf(cb_sm[(b >> 4) | s1], xr[2 * i + 1], inner);
        }
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            int b = (int)((b2.y >> (8 * i)) & 0xFF);
            int s0 = (int)((sw >> (8 + 2 * i)) & 1u) << 4;
            int s1 = (int)((sw >> (8 + 2 * i + 1)) & 1u) << 4;
            inner = fmaf(cb_sm[(b & 0xF) | s0], xr[8 + 2 * i], inner);
            inner = fmaf(cb_sm[(b >> 4) | s1], xr[8 + 2 * i + 1], inner);
        }
        acc += inner * sc;
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        acc += __shfl_down_sync(0xffffffff, acc, off);
    if (lane == 0) y_tab[p][rem] = acc;
}

// per-layer pack tables (built lazily on first call, before graph capture).
// Slot map: l*2+0 = input-projection group, l*2+1 = gate/up group.
static torch::Tensor g_mt_tables;   // [128, 288] uint8
static bool g_mt_ready[128] = {false};
static const int MT_SLOT = 288;

static void mt_build_table(int slot, cudaStream_t st,
                           const void* idx[], const void* sg[],
                           const void* sc[], const void* y[],
                           const int out[], int npack) {
    if (!g_mt_tables.defined())
        g_mt_tables = torch::zeros({128, MT_SLOT}, torch::TensorOptions()
            .dtype(torch::kUInt8).device(torch::kCUDA));
    uint8_t host[MT_SLOT] = {0};
    for (int i = 0; i < npack; ++i) {
        reinterpret_cast<const void**>(host)[i] = idx[i];
        reinterpret_cast<const void**>(host + 64)[i] = sg[i];
        reinterpret_cast<const void**>(host + 128)[i] = sc[i];
        reinterpret_cast<const void**>(host + 192)[i] = y[i];
        reinterpret_cast<int*>(host + 256)[i] = out[i];
    }
    // SYNC copy: host[] is stack memory; an async H2D would read it after
    // the frame is gone (pinned-buffer race class).
    cudaMemcpy(g_mt_tables.data_ptr<uint8_t>() + (size_t)slot * MT_SLOT,
               host, MT_SLOT, cudaMemcpyHostToDevice);
    g_mt_ready[slot] = true;
}

static void mt_gemv_launch(const float* x, const float* cb, int l,
                           int total, int npack, int in_f, int group,
                           cudaStream_t st) {
    static int max_sm = -1;
    if (max_sm < 0) {
        int dev = 0;
        cudaGetDevice(&dev);
        cudaDeviceGetAttribute(&max_sm,
            cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
        if (max_sm > 0) {
            cudaFuncSetAttribute(udcq_gemv_multi_kernel<false>,
                cudaFuncAttributeMaxDynamicSharedMemorySize, max_sm);
            cudaFuncSetAttribute(udcq_gemv_multi_kernel<true>,
                cudaFuncAttributeMaxDynamicSharedMemorySize, max_sm);
        }
    }
    const uint8_t* base = g_mt_tables.data_ptr<uint8_t>() + (size_t)l * MT_SLOT;
    const uint8_t* const* idx_tab =
        reinterpret_cast<const uint8_t* const*>(base);
    const uint32_t* const* sign_tab =
        reinterpret_cast<const uint32_t* const*>(base + 64);
    const __half* const* sc_tab =
        reinterpret_cast<const __half* const*>(base + 128);
    float* const* y_tab = reinterpret_cast<float* const*>(base + 192);
    const int* out_tab = reinterpret_cast<const int*>(base + 256);
    size_t sm = (size_t)in_f * sizeof(float) + 32 * sizeof(float);
    int wpb = (in_f > 12288) ? 16 : 8;
    int grid = (total + wpb - 1) / wpb;
    if (g_uls)
        udcq_gemv_multi_kernel<true><<<grid, wpb * 32, sm, st>>>(
            x, idx_tab, sign_tab, sc_tab, y_tab, out_tab, cb,
            npack, in_f, group);
    else
        udcq_gemv_multi_kernel<false><<<grid, wpb * 32, sm, st>>>(
            x, idx_tab, sign_tab, sc_tab, y_tab, out_tab, cb,
            npack, in_f, group);
}

// UDCQ batched GEMM: [T,in_f] @ W^T -> [T,out_f], same walk as
// udcq_gemv_kernel per row (bit-exact by construction), grid.y = t.
template <bool ULS>
__global__ void udcq_gemm_kernel(
    const float* __restrict__ x,
    const uint8_t* __restrict__ idx,
    const uint32_t* __restrict__ sign,
    const __half* __restrict__ scale,
    const float* __restrict__ cb,
    float* __restrict__ y,
    int in_f, int out_f, int GROUP)
{
    __shared__ float red[32];
    int r = blockIdx.x;
    int t = blockIdx.y;
    long long base = (long long)r * in_f;
    const float* xt = x + (long long)t * in_f;
    const uint8_t* urow = reinterpret_cast<const uint8_t*>(scale)
                          + (size_t)r * (size_t)(in_f / GROUP + 8);
    float acc = 0.f;
    for (int j = threadIdx.x; j < in_f; j += blockDim.x) {
        long long o = base + j;
        uint8_t b = idx[o >> 1];
        int nib = (o & 1) ? ((b >> 4) & 0x0F) : (b & 0x0F);
        float sc = ULS ? uls_scale_row(urow, j / GROUP)
                       : __half2float(scale[o / GROUP]);
        uint32_t sw = sign[o >> 5];
        float sgn = ((sw >> (o & 31)) & 1u) ? 1.f : -1.f;
        acc += cb[nib] * sc * sgn * xt[j];
    }
    if (threadIdx.x < 32) red[threadIdx.x] = 0.f;
    __syncthreads();
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        acc += __shfl_down_sync(0xffffffff, acc, off);
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = acc;
    __syncthreads();
    if (threadIdx.x == 0) {
        float s = 0.f;
        for (int k = 0; k < (int)(blockDim.x >> 5); ++k) s += red[k];
        y[(long long)t * out_f + r] = s;
    }
}

torch::Tensor udcq_gemm_out(torch::Tensor x2d,
                            torch::Tensor idx, torch::Tensor sign,
                            torch::Tensor scale, torch::Tensor cb,
                            int64_t out_f, int64_t in_f,
                            int64_t group) {
    int T = (int)x2d.size(0);
    auto y = torch::zeros({T, out_f},
        torch::dtype(torch::kFloat32).device(x2d.device()));
    dim3 grid((unsigned)out_f, (unsigned)T);
    if (g_uls)
        udcq_gemm_kernel<true><<<grid, 256, 0,
                           at::cuda::getCurrentCUDAStream()>>>(
            x2d.data_ptr<float>(), idx.data_ptr<uint8_t>(),
            reinterpret_cast<const uint32_t*>(sign.data_ptr()),
            reinterpret_cast<const __half*>(scale.data_ptr()), cb.data_ptr<float>(),
            y.data_ptr<float>(), (int)in_f, (int)out_f, (int)group);
    else
        udcq_gemm_kernel<false><<<grid, 256, 0,
                           at::cuda::getCurrentCUDAStream()>>>(
            x2d.data_ptr<float>(), idx.data_ptr<uint8_t>(),
            reinterpret_cast<const uint32_t*>(sign.data_ptr()),
            reinterpret_cast<const __half*>(scale.data_ptr()), cb.data_ptr<float>(),
            y.data_ptr<float>(), (int)in_f, (int)out_f, (int)group);
    return y;
}

// l2norm (fla-aligned): y = x * rsqrt(sum(x^2) + 1e-6), one row/block
__global__ void l2norm_kernel(const float* __restrict__ x,
                              float* __restrict__ y, int d) {
    __shared__ float red[32];
    const float* xr = x + (long long)blockIdx.x * d;
    float* yr = y + (long long)blockIdx.x * d;
    float sq = 0.f;
    for (int i = threadIdx.x; i < d; i += blockDim.x)
        sq += xr[i] * xr[i];
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
        red[0] = __frsqrt_rn(t + 1e-6f);
    }
    __syncthreads();
    for (int i = threadIdx.x; i < d; i += blockDim.x)
        yr[i] = xr[i] * red[0];
}

// causal_conv1d_update (S=1): depthwise K-tap over [state, x],
// state shifts left (drop oldest, append x), optional silu.
__global__ void conv1d_update_kernel(
    const float* __restrict__ x,       // [C]
    float* __restrict__ conv_state,    // [C, state_len] in/out
    const float* __restrict__ w,       // [C, K]
    const float* __restrict__ bias,    // [C]
    float* __restrict__ out,           // [C]
    int state_len, int K, int use_act)
{
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    const float* st_in = conv_state + (long long)c * state_len;
    float* st = conv_state + (long long)c * state_len;
    float acc = 0.f;
    for (int k = 0; k < state_len; ++k)
        acc += w[c * K + k] * st_in[k];
    acc += w[c * K + state_len] * x[c];   // last tap = new token
    acc += bias[c];
    for (int j = 0; j < state_len - 1; ++j)
        st[j] = st[j + 1];
    st[state_len - 1] = x[c];
    out[c] = use_act ? (acc / (1.f + expf(-acc))) : acc;
}

torch::Tensor l2norm_out(torch::Tensor x2d) {
    int R = (int)x2d.size(0), d = (int)x2d.size(1);
    auto y = torch::empty_like(x2d);
    l2norm_kernel<<<R, 256, 0,
                    at::cuda::getCurrentCUDAStream()>>>(
        x2d.data_ptr<float>(), y.data_ptr<float>(), d);
    return y;
}

torch::Tensor conv1d_update_out(torch::Tensor x,
                                torch::Tensor conv_state,
                                torch::Tensor w, torch::Tensor bias,
                                int64_t use_act) {
    int C = (int)x.size(0);
    int state_len = (int)conv_state.size(1);
    int K = (int)w.size(1);
    auto out = torch::empty_like(x);
    conv1d_update_kernel<<<(C + 255) / 256, 256, 0,
                           at::cuda::getCurrentCUDAStream()>>>(
        x.data_ptr<float>(), conv_state.data_ptr<float>(),
        w.data_ptr<float>(), bias.data_ptr<float>(),
        out.data_ptr<float>(), state_len, K, (int)use_act);
    return out;
}

// gated rmsnorm (Qwen3_5RMSNormGated): out[i] = (w[i]*o[i]*inv) * silu(z[i])
// inv = rsqrt(mean(o^2)+eps) per row; one block per row.
__global__ void gated_rmsnorm_kernel(const float* __restrict__ o,
                                     const float* __restrict__ z,
                                     const float* __restrict__ w,
                                     float* __restrict__ out,
                                     int d, float eps) {
    __shared__ float red[32];
    const float* orow = o + (long long)blockIdx.x * d;
    const float* zrow = z + (long long)blockIdx.x * d;
    float* orow_out = out + (long long)blockIdx.x * d;
    float sq = 0.f;
    for (int i = threadIdx.x; i < d; i += blockDim.x)
        sq += orow[i] * orow[i];
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
        red[0] = __frsqrt_rn(t / (float)d + eps);
    }
    __syncthreads();
    for (int i = threadIdx.x; i < d; i += blockDim.x) {
        float g = zrow[i];
        float s = g / (1.f + expf(-g));
        orow_out[i] = (w[i] * orow[i] * red[0]) * s;
    }
}

torch::Tensor gated_rmsnorm_out(torch::Tensor o, torch::Tensor z,
                                torch::Tensor w, double eps) {
    int R = (int)o.size(0), d = (int)o.size(1);
    auto out = torch::empty_like(o);
    gated_rmsnorm_kernel<<<R, 256, 0,
                           at::cuda::getCurrentCUDAStream()>>>(
        o.data_ptr<float>(), z.data_ptr<float>(),
        w.data_ptr<float>(), out.data_ptr<float>(),
        d, (float)eps);
    return out;
}

// 27B mRoPE (partial rotary 64/256, rotate_half over (i, i+32), theta 1e7)
// in-place on q [nh*256] / k [nkv*256]; pos as kernel arg.
__global__ void rope27_kernel(float* q, float* k,
                              int n_heads, int n_kv_heads,
                              int head_dim, float theta,
                              const int* dpos) {
    int h = blockIdx.x;
    int i = threadIdx.x;
    if (i >= 32) return;   // rotary_dim/2 = 32 pairs
    int pos = *dpos;
    float th = pos * powf(theta, -2.0f * i / 64.0f);
    float c = cosf(th), s = sinf(th);
    int base = h * head_dim;
    int lo = base + i;
    int hi = base + i + 32;
    float q0 = q[lo], q1 = q[hi];
    q[lo] = q0 * c - q1 * s;  q[hi] = q0 * s + q1 * c;
    if (h < n_kv_heads) {
        float k0 = k[lo], k1 = k[hi];
        k[lo] = k0 * c - k1 * s;  k[hi] = k0 * s + k1 * c;
    }
    // dims 64..255 pass through untouched
}

torch::Tensor rope27_out(torch::Tensor q, torch::Tensor k,
                         int64_t n_heads, int64_t n_kv_heads,
                         int64_t head_dim, double theta,
                         int64_t pos) {
    auto dp = torch::tensor({(int)pos}, torch::TensorOptions()
        .dtype(torch::kInt32).device(q.device()));
    rope27_kernel<<<(unsigned)n_heads, 32, 0,
                    at::cuda::getCurrentCUDAStream()>>>(
        q.data_ptr<float>(), k.data_ptr<float>(),
        (int)n_heads, (int)n_kv_heads, (int)head_dim,
        (float)theta, dp.data_ptr<int>());
    return q;
}

// fp32 rmsnorm with weight (Qwen3_5RMSNorm on fp32 q/k rows)
__global__ void rmsnorm_fw_kernel(const float* __restrict__ x,
                                  const float* __restrict__ w,
                                  float* __restrict__ y,
                                  int d, float eps) {
    __shared__ float red[32];
    const float* xr = x + (long long)blockIdx.x * d;
    float* yr = y + (long long)blockIdx.x * d;
    float sq = 0.f;
    for (int i = threadIdx.x; i < d; i += blockDim.x)
        sq += xr[i] * xr[i];
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
        red[0] = __frsqrt_rn(t / (float)d + eps);
    }
    __syncthreads();
    for (int i = threadIdx.x; i < d; i += blockDim.x)
        yr[i] = w[i] * xr[i] * red[0];
}

torch::Tensor rmsnorm_fw_out(torch::Tensor x2d, torch::Tensor w,
                             double eps) {
    int R = (int)x2d.size(0), d = (int)x2d.size(1);
    auto y = torch::empty_like(x2d);
    rmsnorm_fw_kernel<<<R, 256, 0,
                        at::cuda::getCurrentCUDAStream()>>>(
        x2d.data_ptr<float>(), w.data_ptr<float>(),
        y.data_ptr<float>(), d, (float)eps);
    return y;
}

// fused residual add + rms-norm in ONE launch:
//   y_add = a + b;  y_norm = w * y_add * rsqrt(mean(y_add^2) + eps)
// Same per-thread accumulation order as add_f32 + rmsnorm_fw_kernel (thread i
// walks i, i+256, ...), so the pair is bit-identical to the two-kernel
// sequence. Replaces 2 launches/layer x 2 with 1 (norm calls were 4.2% of
// step time at 7.4us/call, pure launch overhead for d=5120).
__global__ void add_norm_f32(const float* __restrict__ a,
                             const float* __restrict__ b,
                             const float* __restrict__ w,
                             float* __restrict__ y_add,
                             float* __restrict__ y_norm,
                             int d, float eps) {
    __shared__ float red[32];
    float sq = 0.f;
    for (int i = threadIdx.x; i < d; i += blockDim.x) {
        float v = a[i] + b[i];
        y_add[i] = v;
        sq += v * v;
    }
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
        red[0] = __frsqrt_rn(t / (float)d + eps);
    }
    __syncthreads();
    for (int i = threadIdx.x; i < d; i += blockDim.x)
        y_norm[i] = w[i] * y_add[i] * red[0];
}

// ---------------- Stage 4 step 3a: GDN layer step --------------------- //
__global__ void gdn_recurrent_kernel(
    const float* __restrict__ q, const float* __restrict__ k,
    const float* __restrict__ v, const float* __restrict__ g,
    const float* __restrict__ beta, float* __restrict__ S,
    float* __restrict__ out, int dk, int dv);

__global__ void gdn_recurrent_v2_kernel(
    const float* __restrict__ q, const float* __restrict__ k,
    const float* __restrict__ v, const float* __restrict__ g,
    const float* __restrict__ beta, float* __restrict__ S,
    float* __restrict__ out, int dk, int dv);

__global__ void repeat_heads_kernel(const float* src, float* dst,
                                    int dk, int rep) {
    int j = blockIdx.x;      // output head (nv); HF repeat_interleave:
    int i = threadIdx.x;     // out head j uses src head j/rep
    if (i < dk)
        dst[(long long)j * dk + i]
            = src[(long long)(j / rep) * dk + i];
}
__global__ void sigmoid_kernel(const float* x, float* y, int n) {
    int i = threadIdx.x;
    if (i < n) y[i] = 1.f / (1.f + expf(-x[i]));
}
__global__ void ggate_kernel(const float* a, const float* A_log,
                             const float* dt_bias, float* g,
                             int n) {
    int i = threadIdx.x;
    if (i < n) {
        float x = a[i] + dt_bias[i];
        float sp = (x > 20.f) ? x : (float)log1p(exp((double)x));
        g[i] = -expf(A_log[i]) * sp;
    }
}
__global__ void scale_kernel(float* x, float c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] *= c;
}

torch::Tensor gdn_layer_step(
    torch::Tensor h, torch::Tensor cb,
    torch::Tensor qkv_i, torch::Tensor qkv_s, torch::Tensor qkv_sc,
    torch::Tensor z_i,   torch::Tensor z_s,   torch::Tensor z_sc,
    torch::Tensor b_i,   torch::Tensor b_s,   torch::Tensor b_sc,
    torch::Tensor a_i,   torch::Tensor a_s,   torch::Tensor a_sc,
    torch::Tensor o_i,   torch::Tensor o_s,   torch::Tensor o_sc,
    torch::Tensor conv_w, torch::Tensor conv_b,
    torch::Tensor A_log, torch::Tensor dt_bias,
    torch::Tensor norm_w,
    torch::Tensor conv_state, torch::Tensor S,
    int64_t nv, int64_t nk, int64_t dk, int64_t dv, int64_t l)
{
    auto st = at::cuda::getCurrentCUDAStream();
    int hidden = (int)h.numel();
    int key_dim = (int)(nk * dk);
    int value_dim = (int)(nv * dv);
    int conv_dim = key_dim * 2 + value_dim;
    int GROUP = 16;
    auto f32 = torch::TensorOptions()
        .dtype(torch::kFloat32).device(h.device());

    static torch::Tensor qkv, z, bo, ao, mq, qi, ki, vr, qh, kh,
                          beta, gvec, o, zr, out;
    static bool init = false;
    if (!init) {
        qkv  = torch::empty({conv_dim}, f32);
        z    = torch::empty({value_dim}, f32);
        bo   = torch::empty({nv}, f32);
        ao   = torch::empty({nv}, f32);
        mq   = torch::empty({conv_dim}, f32);
        qi   = torch::empty({nv, dk}, f32);
        ki   = torch::empty({nv, dk}, f32);
        vr   = torch::empty({nv, dv}, f32);
        qh   = torch::empty({nv, dk}, f32);
        kh   = torch::empty({nv, dk}, f32);
        beta = torch::empty({nv}, f32);
        gvec = torch::empty({nv}, f32);
        o    = torch::empty({nv, dv}, f32);
        zr   = torch::empty({nv, dv}, f32);
        out  = torch::empty({hidden}, f32);
        init = true;
    }

    // input projections share h: qkv+z+b+a in ONE multi-pack launch
    int mslot = (int)l * 2;
    if (!g_mt_ready[mslot]) {
        const void* ii[4] = {qkv_i.data_ptr(), z_i.data_ptr(),
                             b_i.data_ptr(), a_i.data_ptr()};
        const void* ss[4] = {qkv_s.data_ptr(), z_s.data_ptr(),
                             b_s.data_ptr(), a_s.data_ptr()};
        const void* cc[4] = {qkv_sc.data_ptr(), z_sc.data_ptr(),
                             b_sc.data_ptr(), a_sc.data_ptr()};
        const void* yy[4] = {qkv.data_ptr(), z.data_ptr(),
                             bo.data_ptr(), ao.data_ptr()};
        int oo[4] = {conv_dim, value_dim, (int)nv, (int)nv};
        mt_build_table(mslot, st, ii, ss, cc, yy, oo, 4);
    }
    mt_gemv_launch(h.data_ptr<float>(), cb.data_ptr<float>(), mslot,
                   conv_dim + value_dim + 2 * (int)nv, 4, hidden, GROUP, st);

    conv1d_update_kernel<<<(conv_dim + 255) / 256, 256, 0, st>>>(
        qkv.data_ptr<float>(), conv_state.data_ptr<float>(),
        conv_w.data_ptr<float>(), conv_b.data_ptr<float>(),
        mq.data_ptr<float>(), 3, 4, 1);

    // z -> [nv,dv] rows BEFORE gated norm (pitfall 1)
    cudaMemcpyAsync(zr.data_ptr<float>(), z.data_ptr<float>(),
                    value_dim * sizeof(float),
                    cudaMemcpyDeviceToDevice, st);
    repeat_heads_kernel<<<(unsigned)nv, dk, 0, st>>>(
        mq.data_ptr<float>(), qi.data_ptr<float>(), dk,
        (int)(nv / nk));
    repeat_heads_kernel<<<(unsigned)nv, dk, 0, st>>>(
        mq.data_ptr<float>() + key_dim, ki.data_ptr<float>(), dk,
        (int)(nv / nk));
    cudaMemcpyAsync(vr.data_ptr<float>(),
        mq.data_ptr<float>() + 2 * key_dim,
        value_dim * sizeof(float), cudaMemcpyDeviceToDevice, st);
    l2norm_kernel<<<nv, 256, 0, st>>>(
        qi.data_ptr<float>(), qh.data_ptr<float>(), dk);
    l2norm_kernel<<<nv, 256, 0, st>>>(
        ki.data_ptr<float>(), kh.data_ptr<float>(), dk);
    scale_kernel<<<(nv * dk + 255) / 256, 256, 0, st>>>(
        qh.data_ptr<float>(), 1.0f / std::sqrt((float)dk), nv * dk);
    sigmoid_kernel<<<1, 256, 0, st>>>(
        bo.data_ptr<float>(), beta.data_ptr<float>(), nv);
    ggate_kernel<<<1, 256, 0, st>>>(
        ao.data_ptr<float>(), A_log.data_ptr<float>(),
        dt_bias.data_ptr<float>(), gvec.data_ptr<float>(), nv);

    gdn_recurrent_v2_kernel<<<dim3((unsigned)nv, (unsigned)(dv / 32)), 128,
                              0, st>>>(
        qh.data_ptr<float>(), kh.data_ptr<float>(),
        vr.data_ptr<float>(), gvec.data_ptr<float>(),
        beta.data_ptr<float>(), S.data_ptr<float>(),
        o.data_ptr<float>(), dk, dv);
    // separate out buffer 鈥?gated norm must NOT alias (pitfall 2)
    static torch::Tensor on;
    if (!init) {}
    if (!on.defined()) on = torch::empty({nv, dv}, f32);
    gated_rmsnorm_kernel<<<nv, 256, 0, st>>>(
        o.data_ptr<float>(), zr.data_ptr<float>(),
        norm_w.data_ptr<float>(), on.data_ptr<float>(), dv, 1e-6f);
    if ((int)l == s27_probe_l) {
        s27d_o = o.clone();
        s27d_gated = on.clone();
    }
    udcq_gemv_launch(
        on.view(-1).data_ptr<float>(), o_i.data_ptr<uint8_t>(),
        reinterpret_cast<const uint32_t*>(o_s.data_ptr()),
        reinterpret_cast<const __half*>(o_sc.data_ptr()), cb.data_ptr<float>(),
        out.data_ptr<float>(), hidden, value_dim, GROUP, st);
    return out;
}

// ---------------- GDN recurrent (27B stage 3a) ----------------------- //
// torch_recurrent_gated_delta_rule for S=1 decode step, per head:
//   S = S*exp(g); kv_mem[j] = sum_i S[i][j]*k[i];
//   delta[j] = (v[j]-kv_mem[j])*beta; S[i][j] += k[i]*delta[j];
//   out[j] = sum_i S[i][j]*q[i]
// q,k must be L2-normalized and q pre-scaled by 1/sqrt(dk) on host
// (matches use_qk_l2norm_in_kernel path). One block per head;
// thread per dv column (requires dv <= blockDim), sequential dk walk.
__global__ void gdn_recurrent_kernel(
    const float* __restrict__ q,   // [nh, dk]
    const float* __restrict__ k,   // [nh, dk]
    const float* __restrict__ v,   // [nh, dv]
    const float* __restrict__ g,   // [nh] (log decay)
    const float* __restrict__ beta,// [nh]
    float* __restrict__ S,         // [nh, dk, dv] in/out
    float* __restrict__ out,       // [nh, dv]
    int dk, int dv)
{
    int h = blockIdx.x;
    int j = threadIdx.x;
    if (j >= dv) return;
    float gt = (float)exp((double)g[h]);
    float bt = beta[h];
    const float* kh = k + (long long)h * dk;
    const float* qh = q + (long long)h * dk;
    float* Sh = S + (long long)h * dk * dv;
    const float* vh = v + (long long)h * dv;
    float* oh = out + (long long)h * dv;

    float kv_mem = 0.f;
    for (int i = 0; i < dk; ++i) {
        float* s = Sh + (long long)i * dv + j;
        *s *= gt;
        kv_mem += *s * kh[i];
    }
    float delta = (vh[j] - kv_mem) * bt;
    for (int i = 0; i < dk; ++i)
        Sh[(long long)i * dv + j] += kh[i] * delta;
    float o = 0.f;
    for (int i = 0; i < dk; ++i)
        o += Sh[(long long)i * dv + j] * qh[i];
    oh[j] = o;
}

// v2: same math, parallel over BOTH i and j. grid = (nv, dv/32),
// 128 threads = 4 i-chunks x 32 j. Reduction order differs from v1
// (deterministic pairwise) -> not bit-exact; gate tier ~1e-6.
__global__ void gdn_recurrent_v2_kernel(
    const float* __restrict__ q,
    const float* __restrict__ k,
    const float* __restrict__ v,
    const float* __restrict__ g,
    const float* __restrict__ beta,
    float* __restrict__ S,
    float* __restrict__ out,
    int dk, int dv)
{
    int h = blockIdx.x;
    int j0 = blockIdx.y * 32;
    int s = threadIdx.x >> 5;          // i-chunk 0..3
    int jl = threadIdx.x & 31;
    int j = j0 + jl;
    float gt = (float)exp((double)g[h]);
    float bt = beta[h];
    const float* kh = k + (long long)h * dk;
    const float* qh = q + (long long)h * dk;
    float* Sh = S + (long long)h * dk * dv;
    __shared__ float red[4][32];
    __shared__ float kvs[32], dvs[32];
    int i0 = s * (dk / 4);
    int i1 = i0 + dk / 4;
    float part = 0.f;
    for (int i = i0; i < i1; ++i) {
        float* sp = Sh + (long long)i * dv + j;
        float sv = *sp * gt;
        *sp = sv;
        part += sv * kh[i];
    }
    red[s][jl] = part;
    __syncthreads();
    if (s == 0) {
        float kv = (red[0][jl] + red[1][jl]) + (red[2][jl] + red[3][jl]);
        kvs[jl] = kv;
        dvs[jl] = (v[(long long)h * dv + j] - kv) * bt;
    }
    __syncthreads();
    float delta = dvs[jl];
    float po = 0.f;
    for (int i = i0; i < i1; ++i) {
        float* sp = Sh + (long long)i * dv + j;
        float sv = *sp + kh[i] * delta;
        *sp = sv;
        po += sv * qh[i];
    }
    red[s][jl] = po;
    __syncthreads();
    if (s == 0)
        out[(long long)h * dv + j] =
            (red[0][jl] + red[1][jl]) + (red[2][jl] + red[3][jl]);
}

torch::Tensor gdn_recurrent_out(
    torch::Tensor q, torch::Tensor k, torch::Tensor v,
    torch::Tensor g, torch::Tensor beta,
    torch::Tensor S) {
    int nh = (int)q.size(0);
    int dk = (int)q.size(1);
    int dv = (int)v.size(1);
    auto out = torch::zeros({nh, dv},
        torch::dtype(torch::kFloat32).device(q.device()));
    gdn_recurrent_kernel<<<nh, 256, 0,
                           at::cuda::getCurrentCUDAStream()>>>(
        q.data_ptr<float>(), k.data_ptr<float>(),
        v.data_ptr<float>(), g.data_ptr<float>(),
        beta.data_ptr<float>(), S.data_ptr<float>(),
        out.data_ptr<float>(), dk, dv);
    return out;
}

// split-K GEMV: each row split across S warps (contiguous chunks),
// partials summed in warp order (deterministic). NOTE: reduction
// order differs from v2/v3 -> NOT bit-exact vs them; gate = rel-err
// vs fp64 reference accumulation + engine E2E coherence.
__global__ void gsq_gemv_sk_kernel(
    const float* __restrict__ x,
    const uint8_t* __restrict__ codes,
    const float* __restrict__ cb,
    const uint8_t* __restrict__ s_i8,
    float s_base, float s_step,
    float* __restrict__ yf,
    int n_gr, int in_f, int S)
{
    __shared__ float cb_sm[32];
    __shared__ float part_sm[8];
    extern __shared__ float x_sm[];
    for (int i = threadIdx.x; i < 32; i += blockDim.x)
        cb_sm[i] = cb[i];
    for (int i = threadIdx.x; i < in_f; i += blockDim.x)
        x_sm[i] = x[i];
    __syncthreads();
    int lane = threadIdx.x & 31;
    int w = threadIdx.x >> 5;          // warp in block
    int nw = blockDim.x >> 5;          // warps per block (8)
    int rows_pb = nw / S;              // rows per block
    int r = blockIdx.x * rows_pb + (w / S);
    int c = w % S;                     // chunk id
    int csize = (n_gr + S - 1) / S;
    int jb0 = c * csize;
    int jb1 = min(n_gr, jb0 + csize);
    float acc = 0.f;
    for (int jb = jb0 + lane; jb < jb1; jb += 32) {
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
        const float* x16 = x_sm + jb * 16;
        float inner = 0.f;
        #pragma unroll
        for (int i = 0; i < 12; ++i)
            inner += cb_sm[(int)((lo >> (5*i)) & 0x1F)] * x16[i];
        inner += cb_sm[(int)(((lo >> 60)
            | ((unsigned long long)d2 << 4)) & 0x1F)] * x16[12];
        #pragma unroll
        for (int i = 0; i < 3; ++i)
            inner += cb_sm[(int)((d2 >> (5*i+1)) & 0x1F)]
                   * x16[13+i];
        acc += inner * s;
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        acc += __shfl_down_sync(0xffffffff, acc, off);
    if (lane == 0) part_sm[w] = acc;
    __syncthreads();
    if (w < rows_pb && lane == 0) {
        float t = 0.f;
        for (int k = 0; k < S; ++k)
            t += part_sm[w * S + k];
        yf[r] = t;
    }
}

torch::Tensor gemv_sk_out(torch::Tensor x,
                          torch::Tensor codes, torch::Tensor cb,
                          torch::Tensor s_i8,
                          double s_base, double s_step,
                          int64_t out_f, int64_t in_f,
                          int64_t S) {
    auto y = torch::zeros({out_f}, torch::dtype(torch::kFloat32)
                                         .device(x.device()));
    int n_gr = (int)(in_f / 16);
    int S_ = (int)S;
    int rows_pb = 8 / S_;
    unsigned grid = (unsigned)(out_f / rows_pb);
    gsq_gemv_sk_kernel<<<grid, 256, (size_t)in_f * sizeof(float),
                         at::cuda::getCurrentCUDAStream()>>>(
        x.data_ptr<float>(),
        codes.data_ptr<uint8_t>(), cb.data_ptr<float>(),
        s_i8.data_ptr<uint8_t>(), (float)s_base, (float)s_step,
        y.data_ptr<float>(), n_gr, (int)in_f, S_);
    return y;
}

// v3: software-pipelined v2. Order-preserving (bit-exact by
// construction): each lane still sums its groups jb, jb+32,... in
// sequence; only the NEXT group's codes are prefetched during the
// current group's compute to hide load latency.
__global__ void gsq_gemv_v3_kernel(
    const float* __restrict__ x,
    const uint8_t* __restrict__ codes,
    const float* __restrict__ cb,
    const uint8_t* __restrict__ s_i8,
    float s_base, float s_step,
    float* __restrict__ yf,
    int n_gr, int in_f)
{
    __shared__ float cb_sm[32];
    extern __shared__ float x_sm[];
    for (int i = threadIdx.x; i < 32; i += blockDim.x)
        cb_sm[i] = cb[i];
    for (int i = threadIdx.x; i < in_f; i += blockDim.x)
        x_sm[i] = x[i];
    __syncthreads();
    int lane = threadIdx.x & 31;
    int r = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    float acc = 0.f;
    int jb = lane;
    if (jb < n_gr) {
        long long gidx = (long long)r * n_gr + jb;
        const uint8_t* p = codes + gidx * 10;
        uint32_t d0 = (uint32_t)p[0] | ((uint32_t)p[1] << 8)
            | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
        uint32_t d1 = (uint32_t)p[4] | ((uint32_t)p[5] << 8)
            | ((uint32_t)p[6] << 16) | ((uint32_t)p[7] << 24);
        uint16_t d2 = (uint16_t)(p[8] | (p[9] << 8));
        uint8_t s8 = s_i8[gidx];
        for (; jb + 32 < n_gr; ) {
            int jn = jb + 32;
            long long gN = (long long)r * n_gr + jn;
            const uint8_t* pn = codes + gN * 10;
            uint32_t n0 = (uint32_t)pn[0] | ((uint32_t)pn[1] << 8)
                | ((uint32_t)pn[2] << 16) | ((uint32_t)pn[3] << 24);
            uint32_t n1 = (uint32_t)pn[4] | ((uint32_t)pn[5] << 8)
                | ((uint32_t)pn[6] << 16) | ((uint32_t)pn[7] << 24);
            uint16_t n2 = (uint16_t)(pn[8] | (pn[9] << 8));
            uint8_t n8 = s_i8[gN];
            // compute current (identical order to v2)
            unsigned long long lo = (unsigned long long)d0
                | ((unsigned long long)d1 << 32);
            float s = exp2f(s_base + (float)s8 * s_step);
            const float* x16 = x_sm + jb * 16;
            float inner = 0.f;
            #pragma unroll
            for (int i = 0; i < 12; ++i)
                inner += cb_sm[(int)((lo >> (5*i)) & 0x1F)] * x16[i];
            inner += cb_sm[(int)(((lo >> 60)
                | ((unsigned long long)d2 << 4)) & 0x1F)] * x16[12];
            #pragma unroll
            for (int i = 0; i < 3; ++i)
                inner += cb_sm[(int)((d2 >> (5*i+1)) & 0x1F)]
                       * x16[13+i];
            acc += inner * s;
            // shift next -> cur
            d0 = n0; d1 = n1; d2 = n2; s8 = n8; jb = jn;
        }
        // tail (same compute)
        unsigned long long lo = (unsigned long long)d0
            | ((unsigned long long)d1 << 32);
        float s = exp2f(s_base + (float)s8 * s_step);
        const float* x16 = x_sm + jb * 16;
        float inner = 0.f;
        #pragma unroll
        for (int i = 0; i < 12; ++i)
            inner += cb_sm[(int)((lo >> (5*i)) & 0x1F)] * x16[i];
        inner += cb_sm[(int)(((lo >> 60)
            | ((unsigned long long)d2 << 4)) & 0x1F)] * x16[12];
        #pragma unroll
        for (int i = 0; i < 3; ++i)
            inner += cb_sm[(int)((d2 >> (5*i+1)) & 0x1F)]
                   * x16[13+i];
        acc += inner * s;
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        acc += __shfl_down_sync(0xffffffff, acc, off);
    if (lane == 0) yf[r] = acc;
}

// launcher for v3 probe
torch::Tensor gemv_v3_out(torch::Tensor x,
                          torch::Tensor codes, torch::Tensor cb,
                          torch::Tensor s_i8,
                          double s_base, double s_step,
                          int64_t out_f, int64_t in_f) {
    auto y = torch::zeros({out_f}, torch::dtype(torch::kFloat32)
                                         .device(x.device()));
    int n_gr = (int)(in_f / 16);
    int wpb = 8;
    gsq_gemv_v3_kernel<<<(unsigned)(out_f / wpb), wpb * 32,
                         (size_t)in_f * sizeof(float),
                         at::cuda::getCurrentCUDAStream()>>>(
        x.data_ptr<float>(),
        codes.data_ptr<uint8_t>(), cb.data_ptr<float>(),
        s_i8.data_ptr<uint8_t>(), (float)s_base, (float)s_step,
        y.data_ptr<float>(), n_gr, (int)in_f);
    return y;
}

// probe: batched GSQ GEMM (stage 1 of batched prefill).
// Same warp-walk as v2 (bit-exact goal per token); blockIdx.y = token.
__global__ void gsq_gemm_kernel(
    const float* __restrict__ x,      // [T, in_f]
    const uint8_t* __restrict__ codes,
    const float* __restrict__ cb,
    const uint8_t* __restrict__ s_i8,
    float s_base, float s_step,
    float* __restrict__ y,            // [T, out_f]
    int n_gr, int in_f, int out_f)
{
    __shared__ float cb_sm[32];
    extern __shared__ float x_sm[];
    for (int i = threadIdx.x; i < 32; i += blockDim.x)
        cb_sm[i] = cb[i];
    int t = blockIdx.y;
    for (int i = threadIdx.x; i < in_f; i += blockDim.x)
        x_sm[i] = x[(long long)t * in_f + i];
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
        const float* x16 = x_sm + jb * 16;
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
    if (lane == 0)
        y[(long long)t * out_f + r] = acc;
}

torch::Tensor gemm_out(torch::Tensor x2d,   // [T, in_f] fp32
                       torch::Tensor codes, torch::Tensor cb,
                       torch::Tensor s_i8,
                       double s_base, double s_step,
                       int64_t out_f, int64_t in_f) {
    int T = (int)x2d.size(0);
    auto y = torch::zeros({T, out_f},
        torch::dtype(torch::kFloat32).device(x2d.device()));
    int n_gr = (int)(in_f / 16);
    int wpb = 8;
    dim3 grid((unsigned)(out_f / wpb), (unsigned)T);
    gsq_gemm_kernel<<<grid, wpb * 32,
                      (size_t)in_f * sizeof(float),
                      at::cuda::getCurrentCUDAStream()>>>(
        x2d.data_ptr<float>(),
        codes.data_ptr<uint8_t>(), cb.data_ptr<float>(),
        s_i8.data_ptr<uint8_t>(), (float)s_base, (float)s_step,
        y.data_ptr<float>(), n_gr, (int)in_f, (int)out_f);
    return y;
}

torch::Tensor gemv_out(torch::Tensor x,
                       torch::Tensor codes, torch::Tensor cb,
                       torch::Tensor s_i8,
                       double s_base, double s_step,
                       int64_t out_f, int64_t in_f) {
    return gsq_gemv(x, codes, cb, s_i8, s_base, s_step,
                    out_f, in_f);
}

// ---------------- C3: all-24-layers in one C++ call ------------------ //
// Eliminates 24 Python鈫扖++ round trips + pre-allocates intermediates.
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
// Does: embedding lookup 鈫?24 layers 鈫?final norm 鈫?lm_head 鈫?argmax
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

    // 3. final norm (bf16 in 鈫?fp32 out)
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

    // 6. read result (one sync per token 鈥?unavoidable)
    return tok_buf.item<int64_t>();
}

// shared model/buffer statics (used by batch + step paths)
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

// ---------------- batched prefill (stage 2) --------------------------- //
// Whole [T,H] chain with batched kernels. Every kernel keeps the
// per-token arithmetic ORDER (bit-exact goal vs prefill_tokens).
// Positions passed as kernel args (start_pos + t), never d_pos.
__global__ void rmsnorm_b_kernel(const __nv_bfloat16* x,
                                 const __nv_bfloat16* w,
                                 float* out, int n, int H) {
    int t = blockIdx.x;
    const __nv_bfloat16* xr = x + (long long)t * H;
    float* orow = out + (long long)t * H;
    __shared__ float red[32];
    float sq = 0.f;
    for (int i = threadIdx.x; i < n; i += blockDim.x)
        sq += __bfloat162float(xr[i]) * __bfloat162float(xr[i]);
    if (threadIdx.x < 32) red[threadIdx.x] = 0.f;
    __syncthreads();
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        sq += __shfl_down_sync(0xffffffff, sq, off);
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = sq;
    __syncthreads();
    if (threadIdx.x == 0) {
        float s = 0.f;
        for (int k = 0; k < (int)(blockDim.x >> 5); ++k) s += red[k];
        red[0] = __frsqrt_rn(s / (float)n + 1e-5f);
    }
    __syncthreads();
    for (int i = threadIdx.x; i < n; i += blockDim.x)
        orow[i] = __bfloat162float(xr[i]) * red[0]
                * __bfloat162float(w[i]);
}

__global__ void rope_b_kernel(float* q, float* k,
                              int n_heads, int n_kv_heads,
                              int head_dim, float theta_base,
                              int start_pos) {
    int th = blockIdx.x;
    int t = th / n_heads;
    int h = th % n_heads;
    int i = threadIdx.x;
    if (i >= head_dim / 2) return;
    int pos = start_pos + t;
    float th_ = pos / powf(theta_base, 2.0f*i / (float)head_dim);
    float c = cosf(th_), s = sinf(th_);
    long long qbase = (long long)t * n_heads * head_dim + h * head_dim;
    long long kbase = (long long)t * n_kv_heads * head_dim
                    + h * head_dim;
    int lo = (int)(qbase + i);
    int hi = (int)(qbase + i + head_dim / 2);
    float q0=q[lo], q1=q[hi];
    q[lo] = q0*c - q1*s;  q[hi] = q0*s + q1*c;
    if (h < n_kv_heads) {
        int klo = (int)(kbase + i);
        int khi = (int)(kbase + i + head_dim / 2);
        float k0=k[klo], k1=k[khi];
        k[klo] = k0*c - k1*s;  k[khi] = k0*s + k1*c;
    }
}

__global__ void cache_write_b_kernel(
    const float* k2d, const float* v2d,
    __nv_bfloat16* kv_base, int hd, int ctx,
    int n_kv_heads, const int* dpos) {
    int tv = blockIdx.x;
    int t = tv / (2 * n_kv_heads);
    int kvi = tv % (2 * n_kv_heads);
    int i = threadIdx.x;
    if (i >= hd) return;
    int start_pos = *dpos;
    const float* src = (kvi < n_kv_heads) ? k2d : v2d;
    int row = t * n_kv_heads + (kvi % n_kv_heads);
    kv_base[((long long)kvi * ctx + start_pos + t) * hd + i]
        = __float2bfloat16(src[(long long)row * hd + i]);
}

__global__ void attn_b_kernel(const float* q,
                              const __nv_bfloat16* kv,
                              float* out,
                              int n_heads, int n_kv_heads,
                              int head_dim, int ctx,
                              const int* dpos) {
    int th = blockIdx.x;
    int t = th / n_heads;
    int h = th % n_heads;
    int pos = *dpos + t;
    int d = threadIdx.x;
    if (d >= head_dim) return;
    int kvh = h / (n_heads / n_kv_heads);
    const __nv_bfloat16* ks = kv + kvh * ctx * head_dim;
    const __nv_bfloat16* vs = kv + (long long)n_kv_heads*ctx*head_dim
                              + kvh * ctx * head_dim;
    float inv_hd = rsqrtf((float)head_dim);
    extern __shared__ float w_sm[];
    __shared__ float red[32];
    const float* qt = q + (long long)t * n_heads * head_dim
                    + h * head_dim;
    float maxs = -1e30f;
    for (int tt = d; tt <= pos; tt += blockDim.x) {
        float sc = 0.f;
        for (int j = 0; j < head_dim; ++j)
            sc += qt[j] * __bfloat162float(ks[tt*head_dim + j]);
        sc *= inv_hd;
        w_sm[tt] = sc;
        if (sc > maxs) maxs = sc;
    }
    if (threadIdx.x < 32) red[threadIdx.x] = -1e30f;
    __syncthreads();
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        maxs = fmaxf(maxs, __shfl_down_sync(0xffffffff, maxs, off));
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = maxs;
    __syncthreads();
    if (threadIdx.x == 0) {
        float m = -1e30f;
        for (int k2 = 0; k2 < (int)(blockDim.x >> 5); ++k2)
            m = fmaxf(m, red[k2]);
        red[0] = m;
    }
    __syncthreads();
    maxs = red[0];
    float denom = 0.f, sum = 0.f;
    for (int tt = 0; tt <= pos; ++tt) {
        float sc = expf(w_sm[tt] - maxs);
        denom += sc;
        sum += sc * __bfloat162float(vs[tt*head_dim + d]);
    }
    out[(long long)t * n_heads * head_dim + h * head_dim + d]
        = sum / denom;
}

// split-K attention decode (v3): ATTN_V3_C chunk-blocks per head with an
// online-softmax merge. The old attn_b_kernel sweeps t serially per thread
// with only nh blocks in flight; it stalls the in-order pipeline on strided
// V loads for ~pos iterations (measured 554us/call at pos 3800 in-graph vs
// 15.6us at pos 50). Chunks beyond pos early-exit with m=-inf/D=0.
#define ATTN_V3_C 8

__global__ void attn_b_v3_kernel(const float* __restrict__ q,
                                 const __nv_bfloat16* __restrict__ kv,
                                 float* __restrict__ part,  // [nh*C*2] m,D
                                 float* __restrict__ psum,  // [nh*C, hd]
                                 int n_heads, int n_kv_heads, int head_dim,
                                 int ctx, const int* dpos) {
    int h = blockIdx.x;
    int c = blockIdx.y;
    int pos = *dpos;
    int chunk = (ctx + ATTN_V3_C - 1) / ATTN_V3_C;
    int t0 = c * chunk;
    int t1 = min(t0 + chunk - 1, pos);
    int kvh = h / (n_heads / n_kv_heads);
    const __nv_bfloat16* ks = kv + (size_t)kvh * ctx * head_dim;
    const __nv_bfloat16* vs = kv + (size_t)n_kv_heads * ctx * head_dim
                              + (size_t)kvh * ctx * head_dim;
    extern __shared__ float smem[];        // chunk floats: scores -> exps
    __shared__ float red[32];
    float* mD = part + (size_t)h * ATTN_V3_C * 2 + c * 2;
    float* ps = psum + ((size_t)h * ATTN_V3_C + c) * head_dim;
    if (t0 > pos) {
        for (int d = threadIdx.x; d < head_dim; d += blockDim.x) ps[d] = 0.f;
        if (threadIdx.x == 0) { mD[0] = -1e30f; mD[1] = 0.f; }
        return;
    }
    const float* qt = q + (size_t)h * head_dim;
    float inv_hd = rsqrtf((float)head_dim);
    float maxs = -1e30f;
    for (int tt = t0 + threadIdx.x; tt <= t1; tt += blockDim.x) {
        float sc = 0.f;
        const __nv_bfloat16* kr = ks + (size_t)tt * head_dim;
        #pragma unroll 8
        for (int j = 0; j < head_dim; ++j)
            sc = fmaf(qt[j], __bfloat162float(kr[j]), sc);
        sc *= inv_hd;
        smem[tt - t0] = sc;
        maxs = fmaxf(maxs, sc);
    }
    if (threadIdx.x < 32) red[threadIdx.x] = -1e30f;
    __syncthreads();
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        maxs = fmaxf(maxs, __shfl_down_sync(0xffffffff, maxs, off));
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = maxs;
    __syncthreads();
    if (threadIdx.x == 0) {
        float m = -1e30f;
        for (int k2 = 0; k2 < (int)(blockDim.x >> 5); ++k2)
            m = fmaxf(m, red[k2]);
        red[0] = m;
    }
    __syncthreads();
    maxs = red[0];
    int nchunk = t1 - t0 + 1;
    for (int i = threadIdx.x; i < nchunk; i += blockDim.x)
        smem[i] = expf(smem[i] - maxs);
    __syncthreads();
    int d = threadIdx.x;
    float den_i = 0.f, sum_i = 0.f;
    #pragma unroll 4
    for (int i = 0; i < nchunk; ++i) {
        float e = smem[i];
        den_i += e;
        sum_i = fmaf(e, __bfloat162float(vs[(size_t)(t0 + i) * head_dim + d]),
                     sum_i);
    }
    // every thread walks ALL i, so den_i is already the full chunk sum
    // (do NOT block-reduce it: that would count it 256x)
    if (threadIdx.x == 0) {
        mD[0] = maxs;
        mD[1] = den_i;
    }
    if (d < head_dim) ps[d] = sum_i;
}

__global__ void attn_merge_kernel(const float* __restrict__ part,
                                  const float* __restrict__ psum,
                                  float* __restrict__ out,
                                  int n_heads, int head_dim) {
    int h = blockIdx.x;
    int d = threadIdx.x;
    const float* mD = part + (size_t)h * ATTN_V3_C * 2;
    float M = -1e30f;
    for (int c = 0; c < ATTN_V3_C; ++c) M = fmaxf(M, mD[c * 2]);
    float den = 0.f, num = 0.f;
    for (int c = 0; c < ATTN_V3_C; ++c) {
        float w = expf(mD[c * 2] - M);
        den += w * mD[c * 2 + 1];
        num += w * psum[((size_t)h * ATTN_V3_C + c) * head_dim + d];
    }
    out[(size_t)h * head_dim + d] = num / den;
}

torch::Tensor attn_v3_test(torch::Tensor q, torch::Tensor kv,
                           int64_t nh, int64_t nkv, int64_t hd,
                           int64_t ctx, int64_t pos) {
    auto st = at::cuda::getCurrentCUDAStream();
    auto f32 = torch::TensorOptions().dtype(torch::kFloat32).device(q.device());
    auto part = torch::zeros({nh * ATTN_V3_C * 2}, f32);
    auto psum = torch::zeros({nh * ATTN_V3_C * hd}, f32);
    auto out = torch::zeros({nh * hd}, f32);
    auto dpos = torch::tensor({(int)pos}, torch::TensorOptions()
        .dtype(torch::kInt32).device(q.device()));
    size_t sm = (size_t)((ctx + ATTN_V3_C - 1) / ATTN_V3_C) * sizeof(float);
    attn_b_v3_kernel<<<dim3((unsigned)nh, ATTN_V3_C), 256, sm, st>>>(
        q.data_ptr<float>(),
        reinterpret_cast<const __nv_bfloat16*>(kv.data_ptr()),
        part.data_ptr<float>(), psum.data_ptr<float>(),
        (int)nh, (int)nkv, (int)hd, (int)ctx, dpos.data_ptr<int>());
    attn_merge_kernel<<<(unsigned)nh, 256, 0, st>>>(
        part.data_ptr<float>(), psum.data_ptr<float>(),
        out.data_ptr<float>(), (int)nh, (int)hd);
    return out;
}

__global__ void embed_rows_kernel(const __nv_bfloat16* table,
                                  const int64_t* toks,
                                  __nv_bfloat16* out,
                                  int T, int H) {
    int r = blockIdx.x;
    int i = blockIdx.y * blockDim.x + threadIdx.x;
    if (r < T && i < H)
        out[(long long)r * H + i]
            = table[toks[r] * H + i];
}

torch::Tensor prefill_batch(torch::Tensor ids_gpu,   // int64 [T] cuda
                            int64_t start_pos) {
    if (!g_init) throw std::runtime_error("init_model not called");
    auto st = at::cuda::getCurrentCUDAStream();
    ensure_g_bufs();
    int T = (int)ids_gpu.size(0);
    int H = (int)g_hidden;
    // layer-0 dims (identical across layers)
    int qo = (int)g_outfs[0], ko = (int)g_outfs[1],
        vo = (int)g_outfs[2], oo = (int)g_outfs[3],
        go = (int)g_outfs[4], uo = (int)g_outfs[5],
        dof = (int)g_outfs[6];
    auto opts_f = torch::TensorOptions()
        .dtype(torch::kFloat32).device(ids_gpu.device());
    auto opts_b = torch::TensorOptions()
        .dtype(torch::kBFloat16).device(ids_gpu.device());

    static torch::Tensor hb, h1f, h1bf, xn, q2, k2, v2, at,
                          o2, xn2, mg, mu, act, md, outb;
    static int cap = 0;
    if (cap < T) {
        cap = g_ctx;
        hb   = torch::empty({cap, H}, opts_b);
        h1f  = torch::empty({cap, H}, opts_f);
        h1bf = torch::empty({cap, H}, opts_b);
        xn   = torch::empty({cap, H}, opts_f);
        q2   = torch::empty({cap, qo}, opts_f);
        k2   = torch::empty({cap, ko}, opts_f);
        v2   = torch::empty({cap, vo}, opts_f);
        at   = torch::empty({cap, qo}, opts_f);
        o2   = torch::empty({cap, oo}, opts_f);
        xn2  = torch::empty({cap, H}, opts_f);
        mg   = torch::empty({cap, go}, opts_f);
        mu   = torch::empty({cap, uo}, opts_f);
        act  = torch::empty({cap, go}, opts_f);
        md   = torch::empty({cap, dof}, opts_f);
        outb = torch::empty({cap, H}, opts_b);
    }
    long long TH = (long long)T * H;
    int nblkH = (int)((TH + 255) / 256);
    auto dp = torch::tensor({(int)start_pos}, torch::TensorOptions()
        .dtype(torch::kInt32).device(ids_gpu.device()));

    embed_rows_kernel<<<dim3(T, (H + 255) / 256), 256, 0, st>>>(
        reinterpret_cast<const __nv_bfloat16*>(g_embed.data_ptr()),
        ids_gpu.data_ptr<int64_t>(),
        reinterpret_cast<__nv_bfloat16*>(hb.data_ptr()),
        T, H);

    auto hcur = hb;
    for (int l = 0; l < (int)g_kcs.size(); ++l) {
        int b = l * 7;
        rmsnorm_b_kernel<<<T, 256, 0, st>>>(
            reinterpret_cast<const __nv_bfloat16*>(hcur.data_ptr()),
            reinterpret_cast<const __nv_bfloat16*>(g_inws[l].data_ptr()),
            xn.data_ptr<float>(), H, H);
        dim3 gq((unsigned)(qo / 8), T), gk((unsigned)(ko / 8), T),
             gv((unsigned)(vo / 8), T);
        gsq_gemm_kernel<<<gq, 256, (size_t)H*4, st>>>(
            xn.data_ptr<float>(), g_codes[b].data_ptr<uint8_t>(),
            g_cbs[b].data_ptr<float>(), g_s[b].data_ptr<uint8_t>(),
            (float)g_bases[b], (float)g_steps[b],
            q2.data_ptr<float>(), H/16, H, qo);
        gsq_gemm_kernel<<<gk, 256, (size_t)H*4, st>>>(
            xn.data_ptr<float>(), g_codes[b+1].data_ptr<uint8_t>(),
            g_cbs[b+1].data_ptr<float>(), g_s[b+1].data_ptr<uint8_t>(),
            (float)g_bases[b+1], (float)g_steps[b+1],
            k2.data_ptr<float>(), H/16, H, ko);
        gsq_gemm_kernel<<<gv, 256, (size_t)H*4, st>>>(
            xn.data_ptr<float>(), g_codes[b+2].data_ptr<uint8_t>(),
            g_cbs[b+2].data_ptr<float>(), g_s[b+2].data_ptr<uint8_t>(),
            (float)g_bases[b+2], (float)g_steps[b+2],
            v2.data_ptr<float>(), H/16, H, vo);
        rope_b_kernel<<<T * (unsigned)g_nheads, (unsigned)(g_hd/2),
                        0, st>>>(
            q2.data_ptr<float>(), k2.data_ptr<float>(),
            (int)g_nheads, (int)g_nkv, (int)g_hd, (float)g_theta,
            (int)start_pos);
        cache_write_b_kernel<<<T * 2 * (unsigned)g_nkv, (unsigned)g_hd,
                               0, st>>>(
            k2.data_ptr<float>(), v2.data_ptr<float>(),
            reinterpret_cast<__nv_bfloat16*>(g_kcs[l].data_ptr()),
            (int)g_hd, (int)g_ctx, (int)g_nkv, dp.data_ptr<int>());
        attn_b_kernel<<<T * (unsigned)g_nheads, (unsigned)g_hd,
                        (size_t)g_ctx * 4, st>>>(
            q2.data_ptr<float>(),
            reinterpret_cast<const __nv_bfloat16*>(g_kcs[l].data_ptr()),
            at.data_ptr<float>(),
            (int)g_nheads, (int)g_nkv, (int)g_hd, (int)g_ctx,
            dp.data_ptr<int>());
        dim3 go_((unsigned)(oo / 8), T);
        gsq_gemm_kernel<<<go_, 256, (size_t)qo*4, st>>>(
            at.data_ptr<float>(), g_codes[b+3].data_ptr<uint8_t>(),
            g_cbs[b+3].data_ptr<float>(), g_s[b+3].data_ptr<uint8_t>(),
            (float)g_bases[b+3], (float)g_steps[b+3],
            o2.data_ptr<float>(), qo/16, qo, oo);
        cast_bf16_f32<<<nblkH, 256, 0, st>>>(
            reinterpret_cast<const __nv_bfloat16*>(hcur.data_ptr()),
            h1f.data_ptr<float>(), (int)TH);
        add_f32<<<nblkH, 256, 0, st>>>(
            h1f.data_ptr<float>(), o2.data_ptr<float>(),
            h1f.data_ptr<float>(), (int)TH);
        cast_f32_bf16<<<nblkH, 256, 0, st>>>(
            h1f.data_ptr<float>(),
            reinterpret_cast<__nv_bfloat16*>(h1bf.data_ptr()),
            (int)TH);
        rmsnorm_b_kernel<<<T, 256, 0, st>>>(
            reinterpret_cast<const __nv_bfloat16*>(h1bf.data_ptr()),
            reinterpret_cast<const __nv_bfloat16*>(g_postws[l].data_ptr()),
            xn2.data_ptr<float>(), H, H);
        dim3 gg((unsigned)(go / 8), T), gu((unsigned)(uo / 8), T);
        gsq_gemm_kernel<<<gg, 256, (size_t)H*4, st>>>(
            xn2.data_ptr<float>(), g_codes[b+4].data_ptr<uint8_t>(),
            g_cbs[b+4].data_ptr<float>(), g_s[b+4].data_ptr<uint8_t>(),
            (float)g_bases[b+4], (float)g_steps[b+4],
            mg.data_ptr<float>(), H/16, H, go);
        gsq_gemm_kernel<<<gu, 256, (size_t)H*4, st>>>(
            xn2.data_ptr<float>(), g_codes[b+5].data_ptr<uint8_t>(),
            g_cbs[b+5].data_ptr<float>(), g_s[b+5].data_ptr<uint8_t>(),
            (float)g_bases[b+5], (float)g_steps[b+5],
            mu.data_ptr<float>(), H/16, H, uo);
        silu_kernel<<<(int)((go * (long long)T + 255) / 256), 256,
                      0, st>>>(
            mg.data_ptr<float>(), mu.data_ptr<float>(),
            act.data_ptr<float>(), (int)((long long)go * T));
        dim3 gd((unsigned)(dof / 8), T);
        gsq_gemm_kernel<<<gd, 256, (size_t)go*4, st>>>(
            act.data_ptr<float>(), g_codes[b+6].data_ptr<uint8_t>(),
            g_cbs[b+6].data_ptr<float>(), g_s[b+6].data_ptr<uint8_t>(),
            (float)g_bases[b+6], (float)g_steps[b+6],
            md.data_ptr<float>(), go/16, go, dof);
        add_f32<<<nblkH, 256, 0, st>>>(
            h1f.data_ptr<float>(), md.data_ptr<float>(),
            h1f.data_ptr<float>(), (int)TH);
        cast_f32_bf16<<<nblkH, 256, 0, st>>>(
            h1f.data_ptr<float>(),
            reinterpret_cast<__nv_bfloat16*>(outb.data_ptr()),
            (int)TH);
        hcur = outb;
    }
    return hcur;
}
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

// Batch prefill: whole prompt in ONE C++ call. Layer-only (no lm_head
// / argmax / .item() sync per token 鈥?those were ~90% of prefill time
// on WDDM). Caller runs step(last_token) afterwards for the first
// generated token. Semantics identical to per-token step().
void prefill_tokens(std::vector<int64_t> toks, int64_t start_pos) {
    if (!g_init) throw std::runtime_error("init_model not called");
    auto st = at::cuda::getCurrentCUDAStream();
    ensure_g_bufs();
    int nl = (int)g_kcs.size();
    for (size_t i = 0; i < toks.size(); ++i) {
        int p = (int)(start_pos + i);
        write_pos_kernel<<<1, 1, 0, st>>>(
            reinterpret_cast<int*>(g_pos.data_ptr()), p);
        set_pos_kernel<<<1, 1, 0, st>>>(
            reinterpret_cast<const int*>(g_pos.data_ptr()));
        cudaMemcpyAsync(g_emb_buf.data_ptr(),
            reinterpret_cast<const char*>(g_embed.data_ptr())
                + toks[i] * g_hidden * 2,
            g_hidden * 2, cudaMemcpyDeviceToDevice, st);
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
    }
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
    gsq_gemv_v2_kernel<<<(unsigned)(g_lh_out_f / 8), 256, (size_t)g_lh_in_f * sizeof(float), st>>>(
        g_fn_buf.data_ptr<float>(),
        g_lh_codes.data_ptr<uint8_t>(),
        g_lh_cb.data_ptr<float>(),
        g_lh_s.data_ptr<uint8_t>(),
        (float)g_lh_base, (float)g_lh_step,
        g_lg_buf.data_ptr<float>(), (int)(g_lh_in_f / 16),
        (int)g_lh_in_f);
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
        gsq_gemv_v2_kernel<<<(unsigned)(g_lh_out_f / 8), 256, (size_t)g_lh_in_f * sizeof(float), st>>>(
            fn_buf.data_ptr<float>(),
            g_lh_codes.data_ptr<uint8_t>(),
            g_lh_cb.data_ptr<float>(),
            g_lh_s.data_ptr<uint8_t>(),
            (float)g_lh_base, (float)g_lh_step,
            lg_buf.data_ptr<float>(), (int)(g_lh_in_f / 16),
            (int)g_lh_in_f);
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

// Single-step, ALL-GPU, no .item() 鈥?PyTorch CUDA graph compatible.
// BUGFIX: set_pos_kernel was MISSING here 鈥?replays ran at the
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
    gsq_gemv_v2_kernel<<<(unsigned)(g_lh_out_f / 8), 256, (size_t)g_lh_in_f * sizeof(float), st>>>(
        g_fn_buf.data_ptr<float>(),
        g_lh_codes.data_ptr<uint8_t>(),
        g_lh_cb.data_ptr<float>(),
        g_lh_s.data_ptr<uint8_t>(),
        (float)g_lh_base, (float)g_lh_step,
        g_lg_buf.data_ptr<float>(), (int)(g_lh_in_f / 16),
        (int)g_lh_in_f);
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
    gsq_gemv_v2_kernel<<<(unsigned)(g_lh_out_f / 8), 256, (size_t)g_lh_in_f * sizeof(float), st>>>(
        g_fn_buf.data_ptr<float>(),
        g_lh_codes.data_ptr<uint8_t>(),
        g_lh_cb.data_ptr<float>(),
        g_lh_s.data_ptr<uint8_t>(),
        (float)g_lh_base, (float)g_lh_step,
        g_lg_buf.data_ptr<float>(), (int)(g_lh_in_f / 16),
        (int)g_lh_in_f);

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

static cudaGraphExec_t g_exec = nullptr;
static cudaGraph_t g_graph = nullptr;
static bool g_captured = false;
static torch::Tensor g_result;
static torch::Tensor g_input;   // static input buffer (address baked in graph)



// ---------------- Stage 4 step 4: full GDN decoder layer -------------- //
// Decoder semantics with fused add+norm (ONE add_norm_f32 launch each):
//   core = gdn_core(xn);  h1,xn2 = add_norm(h, core, post_w)
//   out,xn_next = add_norm(h1, mlp(xn2), in_w_next)
// xn is the ALREADY-NORMED input (produced by the previous layer's final
// add_norm, or primed by step27 for layer 0). conv_state/S update in place.
// packs: 24 tensors = qkv(3),z(3),b(3),a(3),o(3),g(3),u(3),d(3)
static torch::Tensor s27d_xn;   // diagnostics probe (file scope)
static torch::Tensor s27d_xn2;  // probe: MLP input (post-norm)
static torch::Tensor s27d_core; // probe: gdn core output
static torch::Tensor s27d_h1;   // probe: residual h + core
int s27_probe_l = -1;           // probe (defined here)
torch::Tensor s27_h_out;
std::vector<torch::Tensor> gdn_decoder_step(
    torch::Tensor h, torch::Tensor xn, torch::Tensor cb,
    std::vector<torch::Tensor> PK,
    torch::Tensor post_w, torch::Tensor in_w_next,
    torch::Tensor conv_w, torch::Tensor conv_b,
    torch::Tensor A_log, torch::Tensor dt_bias,
    torch::Tensor gnorm_w,
    torch::Tensor conv_state, torch::Tensor S,
    int64_t nv, int64_t nk, int64_t dk, int64_t dv,
    int64_t inter, int64_t l)
{
    auto st = at::cuda::getCurrentCUDAStream();
    torch::Tensor qkv_i = PK[0], qkv_s = PK[1], qkv_sc = PK[2];
    torch::Tensor z_i = PK[3], z_s = PK[4], z_sc = PK[5];
    torch::Tensor b_i = PK[6], b_s = PK[7], b_sc = PK[8];
    torch::Tensor a_i = PK[9], a_s = PK[10], a_sc = PK[11];
    torch::Tensor o_i = PK[12], o_s = PK[13], o_sc = PK[14];
    torch::Tensor gg_i = PK[15], gg_s = PK[16], gg_sc = PK[17];
    torch::Tensor uu_i = PK[18], uu_s = PK[19], uu_sc = PK[20];
    torch::Tensor dd_i = PK[21], dd_s = PK[22], dd_sc = PK[23];
    int hidden = (int)h.numel();
    auto f32 = torch::TensorOptions()
        .dtype(torch::kFloat32).device(h.device());
    static torch::Tensor h1, xn2, out, xn_out, mgs, mus, acts;
    static bool init = false;
    if (!init) {
        h1  = torch::empty({hidden}, f32);
        xn2 = torch::empty({hidden}, f32);
        out = torch::empty({hidden}, f32);
        xn_out = torch::empty({hidden}, f32);
        mgs = torch::empty({inter}, f32);
        mus = torch::empty({inter}, f32);
        acts = torch::empty({inter}, f32);
        init = true;
    }
    if ((int)l == s27_probe_l) s27d_xn = xn.clone();   // probe
    torch::Tensor core = gdn_layer_step(
        xn.view({-1}), cb,
        qkv_i, qkv_s, qkv_sc, z_i, z_s, z_sc,
        b_i, b_s, b_sc, a_i, a_s, a_sc,
        o_i, o_s, o_sc,
        conv_w, conv_b, A_log, dt_bias, gnorm_w,
        conv_state, S, nv, nk, dk, dv, l);
    if ((int)l == s27_probe_l) s27d_core = core.clone();   // probe
    add_norm_f32<<<1, 1024, 0, st>>>(
        h.data_ptr<float>(), core.data_ptr<float>(),
        post_w.data_ptr<float>(), h1.data_ptr<float>(),
        xn2.data_ptr<float>(), hidden, 1e-6f);
    if ((int)l == s27_probe_l) s27d_h1 = h1.clone();       // probe
    if ((int)l == s27_probe_l) s27d_xn2 = xn2.clone(); // probe
    // mlp: gate+up share xn2 in ONE multi-pack launch
    int gslot = (int)l * 2 + 1;
    if (!g_mt_ready[gslot]) {
        const void* ii[2] = {gg_i.data_ptr(), uu_i.data_ptr()};
        const void* ss[2] = {gg_s.data_ptr(), uu_s.data_ptr()};
        const void* cc[2] = {gg_sc.data_ptr(), uu_sc.data_ptr()};
        const void* yy[2] = {mgs.data_ptr(), mus.data_ptr()};
        int oo[2] = {(int)inter, (int)inter};
        mt_build_table(gslot, st, ii, ss, cc, yy, oo, 2);
    }
    mt_gemv_launch(xn2.data_ptr<float>(), cb.data_ptr<float>(), gslot,
                   2 * (int)inter, 2, hidden, 16, st);
    silu_kernel<<<(unsigned)((inter + 255) / 256), 256, 0, st>>>(
        mgs.data_ptr<float>(), mus.data_ptr<float>(),
        acts.data_ptr<float>(), (int)inter);
    torch::Tensor md = udcq_gemv_out(
        acts, dd_i, dd_s, dd_sc, cb, hidden, 17408, 16);
    add_norm_f32<<<1, 1024, 0, st>>>(
        h1.data_ptr<float>(), md.data_ptr<float>(),
        in_w_next.data_ptr<float>(), out.data_ptr<float>(),
        xn_out.data_ptr<float>(), hidden, 1e-6f);
    return {out, xn_out};
}

torch::Tensor s27d_get_xn() { return s27d_xn.cpu(); }
torch::Tensor s27d_get_xn2() { return s27d_xn2.cpu(); }
torch::Tensor s27d_get_core() { return s27d_core.cpu(); }
torch::Tensor s27d_get_h1() { return s27d_h1.cpu(); }
torch::Tensor s27d_get_gated() { return s27d_gated.cpu(); }
torch::Tensor s27d_get_o() { return s27d_o.cpu(); }

// q_proj fused gate split: src [nh, 512] -> q [nh,256], gate [nh,256]
__global__ void chunk_qgate_kernel(const float* src, float* q,
                                   float* gate, int hd) {
    int h = blockIdx.x;
    int i = threadIdx.x;
    if (i < hd) {
        q[(long long)h * hd + i]
            = src[(long long)h * hd * 2 + i];
        gate[(long long)h * hd + i]
            = src[(long long)h * hd * 2 + hd + i];
    }
}
// y[i] *= sigmoid(g[i])
__global__ void sigmoid_mul_kernel(float* y, const float* g,
                                   int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] *= 1.f / (1.f + expf(-g[i]));
}

// ---------------- Stage 4 step 3b: full-attn layer step ---------------- //
// Fused add+norm decoder semantics (xn input pre-normed by the previous
// layer's final add_norm; this layer's final add_norm emits xn_next).
// packs: 21 tensors = q(3),k(3),v(3),o(3),g(3),u(3),d(3) idx/sign/scale
std::vector<torch::Tensor> attn_layer_step(
    torch::Tensor h,                    // [5120] fp32 residual stream
    torch::Tensor xn,                   // [5120] fp32 normed layer input
    torch::Tensor cb,
    std::vector<torch::Tensor> PK,
    torch::Tensor post_w, torch::Tensor in_w_next,
    torch::Tensor q_norm_w, torch::Tensor k_norm_w,
    torch::Tensor kv_cache,             // [8, ctx, 256] bf16
    double theta, const int* dpos,
    int64_t nh, int64_t nkv, int64_t hd,
    int64_t hidden, int64_t inter, int64_t ctx, int64_t l)
{
    auto st = at::cuda::getCurrentCUDAStream();
    torch::Tensor q_i = PK[0], q_s = PK[1], q_sc = PK[2];
    torch::Tensor k_i = PK[3], k_s = PK[4], k_sc = PK[5];
    torch::Tensor v_i = PK[6], v_s = PK[7], v_sc = PK[8];
    torch::Tensor o_i = PK[9], o_s = PK[10], o_sc = PK[11];
    torch::Tensor g_i = PK[12], g_s = PK[13], g_sc = PK[14];
    torch::Tensor u_i = PK[15], u_s = PK[16], u_sc = PK[17];
    torch::Tensor d_i = PK[18], d_s = PK[19], d_sc = PK[20];
    int GROUP = 16;
    auto f32 = torch::TensorOptions()
        .dtype(torch::kFloat32).device(h.device());
    static torch::Tensor q2, gt, k2, v2, qh, kh, o2, att, o,
                        h1, xn2, mg, mu, act, md, out, xn_out;
    static bool init = false;
    if (!init) {
        q2  = torch::empty({nh * hd * 2}, f32);
        gt  = torch::empty({nh * hd}, f32);
        k2  = torch::empty({1, nkv * hd}, f32);
        v2  = torch::empty({1, nkv * hd}, f32);
        qh  = torch::empty({nh, hd}, f32);
        kh  = torch::empty({nkv, hd}, f32);
        o2  = torch::empty({1, nh * hd}, f32);
        att = torch::empty({1, nh * hd}, f32);
        o   = torch::empty({hidden}, f32);
        h1  = torch::empty({hidden}, f32);
        xn2 = torch::empty({hidden}, f32);
        mg  = torch::empty({inter}, f32);
        mu  = torch::empty({inter}, f32);
        act = torch::empty({inter}, f32);
        md  = torch::empty({hidden}, f32);
        out = torch::empty({hidden}, f32);
        xn_out = torch::empty({hidden}, f32);
        init = true;
    }
    // q+k+v share xn in ONE multi-pack launch (slot l*2)
    int qslot = (int)l * 2;
    if (!g_mt_ready[qslot]) {
        const void* ii[3] = {q_i.data_ptr(), k_i.data_ptr(), v_i.data_ptr()};
        const void* ss[3] = {q_s.data_ptr(), k_s.data_ptr(), v_s.data_ptr()};
        const void* cc[3] = {q_sc.data_ptr(), k_sc.data_ptr(), v_sc.data_ptr()};
        const void* yy[3] = {q2.data_ptr(), k2.data_ptr(), v2.data_ptr()};
        int oo[3] = {(int)(nh * hd * 2), (int)(nkv * hd), (int)(nkv * hd)};
        mt_build_table(qslot, st, ii, ss, cc, yy, oo, 3);
    }
    mt_gemv_launch(xn.data_ptr<float>(), cb.data_ptr<float>(), qslot,
                   (int)(nh * hd * 2 + 2 * nkv * hd), 3, hidden, GROUP, st);
    // fused per-head gate: q2 [nh,512] -> q [nh,256] + gate [nh,256]
    chunk_qgate_kernel<<<(unsigned)nh, (unsigned)hd, 0, st>>>(
        q2.data_ptr<float>(), qh.data_ptr<float>(),
        gt.data_ptr<float>(), (int)hd);
    rmsnorm_fw_kernel<<<nh, 256, 0, st>>>(
        qh.data_ptr<float>(), q_norm_w.data_ptr<float>(),
        qh.data_ptr<float>(), hd, 1e-6f);
    k2 = k2.view({1, nkv * hd});
    rmsnorm_fw_kernel<<<nkv, 256, 0, st>>>(
        k2.data_ptr<float>(), k_norm_w.data_ptr<float>(),
        kh.data_ptr<float>(), hd, 1e-6f);
    rope27_kernel<<<(unsigned)nh, 32, 0, st>>>(
        qh.data_ptr<float>(), kh.data_ptr<float>(),
        (int)nh, (int)nkv, (int)hd, (float)theta, dpos);
    cache_write_b_kernel<<<(unsigned)(2 * nkv), (unsigned)hd, 0,
                           st>>>(
        k2.data_ptr<float>(), v2.data_ptr<float>(),
        reinterpret_cast<__nv_bfloat16*>(kv_cache.data_ptr()),
        (int)hd, (int)ctx, (int)nkv, dpos);
    static torch::Tensor attn_part, attn_psum;
    if (!attn_part.defined()) {
        attn_part = torch::zeros({nh * ATTN_V3_C * 2}, f32);
        attn_psum = torch::zeros({nh * ATTN_V3_C * (int)hd}, f32);
    }
    size_t asm3 = (size_t)(((int)ctx + ATTN_V3_C - 1) / ATTN_V3_C)
                  * sizeof(float);
    attn_b_v3_kernel<<<dim3((unsigned)nh, ATTN_V3_C), 256, asm3, st>>>(
        qh.data_ptr<float>(),
        reinterpret_cast<const __nv_bfloat16*>(kv_cache.data_ptr()),
        attn_part.data_ptr<float>(), attn_psum.data_ptr<float>(),
        (int)nh, (int)nkv, (int)hd, (int)ctx, dpos);
    attn_merge_kernel<<<(unsigned)nh, (unsigned)hd, 0, st>>>(
        attn_part.data_ptr<float>(), attn_psum.data_ptr<float>(),
        att.data_ptr<float>(), (int)nh, (int)hd);
    // per-head sigmoid gate before o_proj
    sigmoid_mul_kernel<<<(unsigned)((nh * hd + 255) / 256), 256, 0,
                         st>>>(
        att.data_ptr<float>(), gt.data_ptr<float>(), nh * hd);
    udcq_gemv_launch(
        att.data_ptr<float>(), o_i.data_ptr<uint8_t>(),
        reinterpret_cast<const uint32_t*>(o_s.data_ptr()),
        reinterpret_cast<const __half*>(o_sc.data_ptr()), cb.data_ptr<float>(),
        o.data_ptr<float>(), hidden, nh * hd, GROUP, st);
    add_norm_f32<<<1, 1024, 0, st>>>(
        h.data_ptr<float>(), o.data_ptr<float>(),
        post_w.data_ptr<float>(), h1.data_ptr<float>(),
        xn2.data_ptr<float>(), hidden, 1e-6f);
    // mlp: gate+up share xn2 in ONE multi-pack launch (slot l*2+1)
    if (!g_mt_ready[(int)l * 2 + 1]) {
        const void* ii[2] = {g_i.data_ptr(), u_i.data_ptr()};
        const void* ss[2] = {g_s.data_ptr(), u_s.data_ptr()};
        const void* cc[2] = {g_sc.data_ptr(), u_sc.data_ptr()};
        const void* yy[2] = {mg.data_ptr(), mu.data_ptr()};
        int oo[2] = {(int)inter, (int)inter};
        mt_build_table((int)l * 2 + 1, st, ii, ss, cc, yy, oo, 2);
    }
    mt_gemv_launch(xn2.data_ptr<float>(), cb.data_ptr<float>(),
                   (int)l * 2 + 1, 2 * (int)inter, 2, hidden, 16, st);
    silu_kernel<<<(unsigned)((inter + 255) / 256), 256, 0, st>>>(
        mg.data_ptr<float>(), mu.data_ptr<float>(),
        act.data_ptr<float>(), (int)inter);
    udcq_gemv_launch(
        act.data_ptr<float>(), d_i.data_ptr<uint8_t>(),
        reinterpret_cast<const uint32_t*>(d_s.data_ptr()),
        reinterpret_cast<const __half*>(d_sc.data_ptr()), cb.data_ptr<float>(),
        md.data_ptr<float>(), hidden, inter, GROUP, st);
    add_norm_f32<<<1, 1024, 0, st>>>(
        h1.data_ptr<float>(), md.data_ptr<float>(),
        in_w_next.data_ptr<float>(), out.data_ptr<float>(),
        xn_out.data_ptr<float>(), hidden, 1e-6f);
    return {out, xn_out};
}
