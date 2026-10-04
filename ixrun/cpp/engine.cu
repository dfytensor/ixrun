// IXRUN C++ engine — C1: rmsnorm + GSQ GEMV + SiLU-MLP chain in one
// host call (zero python per-op). Attention/rope/KV arrive in C2.
#include <torch/extension.h>
#include <cuda_bf16.h>
#include <cstdint>
#include <ATen/cuda/CUDAContext.h>
#include <cmath>

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
    int n_gr, int n_sp)
{
    int lane = threadIdx.x & 31;
    int r = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    int per = (n_gr + n_sp - 1) / n_sp;
    int jb0 = blockIdx.y * per;
    int jb1 = min(jb0 + per, n_gr);
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
        const __nv_bfloat16* x16 = x + jb * 16;
        float inner = 0.f;
        #pragma unroll
        for (int i = 0; i < 12; ++i) {
            int c = (int)((lo >> (5 * i)) & 0x1F);
            inner += cb[c] * __bfloat162float(x16[i]);
        }
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            int c = (int)((d2 >> (5 * i)) & 0x1F);
            inner += cb[c] * __bfloat162float(x16[12 + i]);
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
    int n_sp = n_gr / 96; if (n_sp < 1) n_sp = 1;
    if (n_sp > 6) n_sp = 6;
    int wpb = 8;
    unsigned gx = (unsigned)(out_f / wpb);
    dim3 grid(gx, (unsigned)n_sp);
    auto s0 = at::cuda::getCurrentCUDAStream();
    gsq_gemv_kernel<<<grid, wpb * 32, 0, s0>>>(
        x, pk_codes.data_ptr<uint8_t>(), pk_cb.data_ptr<float>(),
        pk_s.data_ptr<uint8_t>(), (float)s_base, (float)s_step,
        yf, n_gr, n_sp);
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
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float v = i < n ? __bfloat162float(x[i]) : 0.f;
    float sq = v * v;
    if (threadIdx.x < 32) red[threadIdx.x] = 0.f;
    __syncthreads();
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
    if (i < n) {
        float inv = rsqrtf(red[0] + 1e-5f);
        out[i] = __float2bfloat16(
            __bfloat162float(x[i]) * inv * __bfloat162float(w[i]));
    }
}

__global__ void silu_mul_kernel(const float* __restrict__ a,
                                const float* __restrict__ b,
                                __nv_bfloat16* __restrict__ out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v = a[i];
    out[i] = __float2bfloat16(v / (1.f + expf(-v)) * b[i]);
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
    auto act = torch::empty({h}, torch::dtype(torch::kBFloat16)
                                   .device(x.device()));
    silu_mul_kernel<<<(h + 255) / 256, 256, 0, s0>>>(
        gf.data_ptr<float>(), uf.data_ptr<float>(),
        reinterpret_cast<__nv_bfloat16*>(act.data_ptr()), h);
    auto df32 = torch::empty({in_f}, torch::dtype(torch::kFloat32)
                                         .device(x.device()));
    lin(act, dc, dcb, ds, dbase, dstep, dof, dif,
        df32.data_ptr<float>());
    return df32.to(torch::kBFloat16);
}

__global__ void noop_kernel() {}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("mlp_forward", &mlp_forward);
    m.def("noop", []() { noop_kernel<<<1, 1>>>(); });
}
