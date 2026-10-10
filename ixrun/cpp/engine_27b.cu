// ---------------- Stage 4 step 4b: 27B model scheduler ---------------- //
// Single-model statics (mirrors the 1B init_model pattern). Layer
// packs normalized to 8 slots x 3 tensors per layer:
//   GDN  slots: qkv,z,b,a,out,gate,up,down
//   ATTN slots: q,k,v,o,gate,up,down,(unused)
static std::vector<torch::Tensor> s27_packs;     // [64*8*3]
static std::vector<torch::Tensor> s27_nw1, s27_nw2;
static std::vector<torch::Tensor> s27_gex;       // [48*4]
static std::vector<torch::Tensor> s27_gnorm;     // [48]
static std::vector<torch::Tensor> s27_aex;       // [16*2]
static std::vector<torch::Tensor> s27_convst, s27_S, s27_kv;
static torch::Tensor s27_cb, s27_fnw, s27_lh_i, s27_lh_s,
    s27_lh_sc;
static std::vector<int64_t> s27_attn_layers;
static int64_t s27_hidden, s27_inter, s27_ctx;
static bool s27_init = false;
static torch::Tensor s27_lg_out;
static torch::Tensor s27_tok;          // argmax output (device int64[1])
static std::vector<float> s27_hnorm;   // per-layer probe
static std::vector<torch::Tensor> s27_layer_h;
extern int s27_probe_l;
extern torch::Tensor s27_h_out;

void init27(
    torch::Tensor cb,
    std::vector<torch::Tensor> packs,
    std::vector<torch::Tensor> nw1,
    std::vector<torch::Tensor> nw2,
    std::vector<torch::Tensor> gex,
    std::vector<torch::Tensor> gnorm,
    std::vector<torch::Tensor> aex,
    torch::Tensor fnw,
    torch::Tensor lh_i, torch::Tensor lh_s, torch::Tensor lh_sc,
    std::vector<int64_t> attn_layers,
    int64_t hidden, int64_t inter, int64_t ctx)
{
    s27_cb = cb;
    s27_packs = packs;
    s27_nw1 = nw1; s27_nw2 = nw2;
    s27_gex = gex; s27_gnorm = gnorm; s27_aex = aex;
    s27_fnw = fnw;
    s27_lh_i = lh_i; s27_lh_s = lh_s; s27_lh_sc = lh_sc;
    s27_attn_layers = attn_layers;
    s27_hidden = hidden; s27_inter = inter; s27_ctx = ctx;
    auto dev = cb.device();
    auto f32 = torch::TensorOptions()
        .dtype(torch::kFloat32).device(dev);
    auto bf = torch::TensorOptions()
        .dtype(torch::kBFloat16).device(dev);
    int nl = (int)nw1.size();
    int ng = nl - (int)attn_layers.size();
    for (int i = 0; i < ng; ++i) {
        s27_convst.push_back(torch::zeros({10240, 3}, f32));
        s27_S.push_back(torch::zeros({48, 128, 128}, f32));
    }
    for (int i = 0; i < (int)attn_layers.size(); ++i)
        s27_kv.push_back(torch::zeros({8, ctx, 256}, bf));
    s27_init = true;
}

// graph-capture-safe core: no .item(), pos comes from a device scalar
static void step27_impl(torch::Tensor h, const int* dpos, double theta) {
    auto st = at::cuda::getCurrentCUDAStream();
    int hidden = (int)s27_hidden;
    int inter = (int)s27_inter;
    int nl = (int)s27_nw1.size();
    int GROUP = 16;
    auto f32 = torch::TensorOptions()
        .dtype(torch::kFloat32).device(h.device());
    static torch::Tensor fn, lg, xn0;
    static bool init = false;
    if (!init) {
        fn  = torch::empty({hidden}, f32);
        xn0 = torch::empty({hidden}, f32);
        int64_t vocab = s27_lh_i.numel() * 2 / hidden;
        lg  = torch::empty({vocab}, f32);
        s27_tok = torch::zeros({1}, torch::TensorOptions()
            .dtype(torch::kInt64).device(h.device()));
        init = true;
    }
    auto T3 = [&](int l, int s) -> std::vector<torch::Tensor> {
        int k = (l * 8 + s) * 3;
        return {s27_packs[k], s27_packs[k + 1], s27_packs[k + 2]};
    };
    // prime layer 0's normed input; every layer's final add_norm emits the
    // next layer's xn (last layer uses fnw) -- no separate in_norm calls.
    rmsnorm_fw_kernel<<<1, 1024, 0, st>>>(
        h.data_ptr<float>(), s27_nw1[0].data_ptr<float>(),
        xn0.data_ptr<float>(), hidden, 1e-6f);
    torch::Tensor xn = xn0;
    int ig = 0, ia = 0;
    for (int l = 0; l < nl; ++l) {
        bool is_attn = false;
        for (int64_t a : s27_attn_layers)
            if (a == l) { is_attn = true; break; }
        torch::Tensor next_w = (l + 1 < nl) ? s27_nw1[l + 1] : s27_fnw;
        if (is_attn) {
            std::vector<torch::Tensor> P;
            for (int s = 0; s < 7; ++s) {
                auto t3 = T3(l, s);
                P.push_back(t3[0]); P.push_back(t3[1]);
                P.push_back(t3[2]);
            }
            auto res = attn_layer_step(
                h, xn, s27_cb, P,
                s27_nw2[l], next_w,
                s27_aex[ia * 2], s27_aex[ia * 2 + 1],
                s27_kv[ia], theta, dpos,
                24, 4, 256, hidden, inter, s27_ctx, l);
            h = res[0]; xn = res[1];
            ia++;
        } else {
            std::vector<torch::Tensor> P;
            for (int s = 0; s < 8; ++s) {
                auto t3 = T3(l, s);
                P.push_back(t3[0]); P.push_back(t3[1]);
                P.push_back(t3[2]);
            }
            auto res = gdn_decoder_step(
                h, xn, s27_cb, P,
                s27_nw2[l], next_w,
                s27_gex[ig * 4], s27_gex[ig * 4 + 1],
                s27_gex[ig * 4 + 2], s27_gex[ig * 4 + 3],
                s27_gnorm[ig], s27_convst[ig], s27_S[ig],
                48, 16, 128, 128, inter, l);
            h = res[0]; xn = res[1];
            ig++;
        }
        if (s27_probe_l >= 0) {   // diagnostics only (no syncs in hot path)
            torch::Tensor hn = h.norm();
            s27_hnorm.push_back(hn.item<float>());
            if ((int)l == s27_probe_l) s27_h_out = h.clone();
        }
    }
    // xn now holds norm(h_final, fnw) -- written by the last layer's
    // add_norm (in_w_next = fnw); feed it straight to lm_head.
    int64_t vocab = s27_lh_i.numel() * 2 / hidden;
    udcq_gemv_launch(
        xn.data_ptr<float>(), s27_lh_i.data_ptr<uint8_t>(),
        reinterpret_cast<const uint32_t*>(s27_lh_s.data_ptr()),
        reinterpret_cast<const __half*>(s27_lh_sc.data_ptr()),
        s27_cb.data_ptr<float>(),
        lg.data_ptr<float>(), (int)vocab, hidden, GROUP, st);
    argmax_f32<<<1, 256, 0, st>>>(
        lg.data_ptr<float>(), (int)vocab,
        s27_tok.data_ptr<int64_t>());
    s27_lg_out = lg;   // probe (diagnostics)
}

int64_t step27(torch::Tensor h, int64_t pos, double theta) {
    if (!s27_init) throw std::runtime_error("init27 not called");
    s27_hnorm.clear();
    s27_layer_h.clear();
    static torch::Tensor dpos_dev;
    if (!dpos_dev.defined())
        dpos_dev = torch::zeros({1}, torch::TensorOptions()
            .dtype(torch::kInt32).device(h.device()));
    dpos_dev.fill_(pos);
    step27_impl(h, dpos_dev.data_ptr<int>(), theta);
    return s27_tok.item<int64_t>();
}

// graph path: capture-safe (no sync inside); pos read from device at run time
void step27_g(torch::Tensor h, torch::Tensor dpos, double theta) {
    if (!s27_init) throw std::runtime_error("init27 not called");
    step27_impl(h, dpos.data_ptr<int>(), theta);
}

// ---------------- GEMM prefill (T <= 256, layer-major segment) --------- //
// Per layer: weights dequantized to bf16 ONCE (udcq_dequant_bf16) and reused
// across the whole segment via cuBLAS matmuls -> the weight read amortizes
// over T tokens instead of the mt-family's 8. The per-token cores (GDN
// conv/recurrent/norms, attention) are unchanged. All shapes are static for
// a given T => graph-capturable. lm_head runs on the last row every call
// (cheap; the caller ignores s27_tok for non-final segments).
#define GEMM_TMAX 256

void step27_prefill_gemm(torch::Tensor hT, torch::Tensor dposT, double theta) {
    if (!s27_init) throw std::runtime_error("init27 not called");
    auto st = at::cuda::getCurrentCUDAStream();
    int hidden = (int)s27_hidden;
    int inter = (int)s27_inter;
    int nl = (int)s27_nw1.size();
    const int NV = 48, NK = 16, DK = 128, DV = 128;
    const int NH = 24, NKV = 4, HD = 256;
    int value_dim = NV * DV;
    int conv_dim = 2 * NK * DK + value_dim;
    int T = (int)hT.size(0);
    if (T < 1 || T > GEMM_TMAX) throw std::runtime_error("T out of range");
    auto f32 = torch::TensorOptions()
        .dtype(torch::kFloat32).device(hT.device());
    auto bf = torch::TensorOptions()
        .dtype(torch::kBFloat16).device(hT.device());
    static torch::Tensor hA, hB, xnA, xnB, h1T, xn2T, oT, q2T, k2T, v2T,
        attT, qkvT, zT, boT, aoT, onT, mgT, muT, actT, mdT;
    static torch::Tensor xnb, xn2b, onb, actb, attb, q2b, k2b, v2b, ob,
        qkvb, zb, bob, aob, mgb, mub, mdb;
    static torch::Tensor wq, wk, wv, wo, wg, wu, wd, wqkv, wz, wb, wa, lg;
    static bool init = false;
    if (!init) {
        int M = GEMM_TMAX;
        hA = torch::empty({M, hidden}, f32);
        hB = torch::empty({M, hidden}, f32);
        xnA = torch::empty({M, hidden}, f32);
        xnB = torch::empty({M, hidden}, f32);
        h1T = torch::empty({M, hidden}, f32);
        xn2T = torch::empty({M, hidden}, f32);
        oT = torch::empty({M, hidden}, f32);
        q2T = torch::empty({M, NH * HD * 2}, f32);
        k2T = torch::empty({M, NKV * HD}, f32);
        v2T = torch::empty({M, NKV * HD}, f32);
        attT = torch::empty({M, NH * HD}, f32);
        qkvT = torch::empty({M, conv_dim}, f32);
        zT = torch::empty({M, value_dim}, f32);
        boT = torch::empty({M, NV}, f32);
        aoT = torch::empty({M, NV}, f32);
        onT = torch::empty({M, value_dim}, f32);
        mgT = torch::empty({M, inter}, f32);
        muT = torch::empty({M, inter}, f32);
        actT = torch::empty({M, inter}, f32);
        mdT = torch::empty({M, hidden}, f32);
        xnb = torch::empty({M, hidden}, bf);
        xn2b = torch::empty({M, hidden}, bf);
        onb = torch::empty({M, value_dim}, bf);
        actb = torch::empty({M, inter}, bf);
        attb = torch::empty({M, NH * HD}, bf);
        q2b = torch::empty({M, NH * HD * 2}, bf);
        k2b = torch::empty({M, NKV * HD}, bf);
        v2b = torch::empty({M, NKV * HD}, bf);
        ob = torch::empty({M, hidden}, bf);
        qkvb = torch::empty({M, conv_dim}, bf);
        zb = torch::empty({M, value_dim}, bf);
        bob = torch::empty({M, NV}, bf);
        aob = torch::empty({M, NV}, bf);
        mgb = torch::empty({M, inter}, bf);
        mub = torch::empty({M, inter}, bf);
        mdb = torch::empty({M, hidden}, bf);
        wq = torch::empty({NH * HD * 2, hidden}, bf);
        wk = torch::empty({NKV * HD, hidden}, bf);
        wv = torch::empty({NKV * HD, hidden}, bf);
        wo = torch::empty({hidden, NH * HD}, bf);
        wg = torch::empty({inter, hidden}, bf);
        wu = torch::empty({inter, hidden}, bf);
        wd = torch::empty({hidden, inter}, bf);
        wqkv = torch::empty({conv_dim, hidden}, bf);
        wz = torch::empty({value_dim, hidden}, bf);
        wb = torch::empty({NV, hidden}, bf);
        wa = torch::empty({NV, hidden}, bf);
        int64_t vocab = s27_lh_i.numel() * 2 / hidden;
        lg = torch::empty({vocab}, f32);
        init = true;
    }
    if (!s27_tok.defined())
        s27_tok = torch::zeros({1}, torch::TensorOptions()
            .dtype(torch::kInt64).device(hT.device()));
    const int* dpos = dposT.data_ptr<int>();
    auto T3 = [&](int l, int s) -> std::vector<torch::Tensor> {
        int k = (l * 8 + s) * 3;
        return {s27_packs[k], s27_packs[k + 1], s27_packs[k + 2]};
    };
    auto deq = [&](torch::Tensor& w, const std::vector<torch::Tensor>& P,
                   int of, int inf) {
        udcq_dequant_bf16_launch(
            P[0].data_ptr<uint8_t>(),
            reinterpret_cast<const uint32_t*>(P[1].data_ptr()),
            reinterpret_cast<const __half*>(P[2].data_ptr()),
            s27_cb.data_ptr<float>(),
            reinterpret_cast<__nv_bfloat16*>(w.data_ptr()),
            of, inf, 16, st);
    };
    auto castb = [&](const float* src, torch::Tensor& dst, int n) {
        int total = T * n;
        f32_to_bf16_kernel<<<(unsigned)((total + 255) / 256), 256, 0, st>>>(
            src, reinterpret_cast<__nv_bfloat16*>(dst.data_ptr()), total);
    };
    auto castf = [&](const torch::Tensor& src, float* dst, int n) {
        int total = T * n;
        bf16_to_f32_kernel<<<(unsigned)((total + 255) / 256), 256, 0, st>>>(
            reinterpret_cast<const __nv_bfloat16*>(src.data_ptr()), dst,
            total);
    };
    auto mm = [&](torch::Tensor& outb, torch::Tensor& xb,
                  const torch::Tensor& w) {
        torch::Tensor o = outb.narrow(0, 0, T);
        torch::Tensor x = xb.narrow(0, 0, T);
        at::matmul_out(o, x, w.t());
    };
    // copy input into hA (ping-pong staging)
    cudaMemcpyAsync(hA.data_ptr<float>(), hT.data_ptr<float>(),
                    (size_t)T * hidden * sizeof(float),
                    cudaMemcpyDeviceToDevice, st);
    torch::Tensor h_cur = hA, h_nxt = hB;
    torch::Tensor xn_cur = xnA, xn_nxt = xnB;
    rmsnorm_fw_kernel<<<(unsigned)T, 1024, 0, st>>>(
        h_cur.data_ptr<float>(), s27_nw1[0].data_ptr<float>(),
        xn_cur.data_ptr<float>(), hidden, 1e-6f);
    int ig = 0, ia = 0;
    for (int l = 0; l < nl; ++l) {
        bool is_attn = false;
        for (int64_t a : s27_attn_layers)
            if (a == l) { is_attn = true; break; }
        torch::Tensor next_w = (l + 1 < nl) ? s27_nw1[l + 1] : s27_fnw;
        if (!is_attn) {
            deq(wqkv, T3(l, 0), conv_dim, hidden);
            deq(wz, T3(l, 1), value_dim, hidden);
            deq(wb, T3(l, 2), NV, hidden);
            deq(wa, T3(l, 3), NV, hidden);
        } else {
            deq(wq, T3(l, 0), NH * HD * 2, hidden);
            deq(wk, T3(l, 1), NKV * HD, hidden);
            deq(wv, T3(l, 2), NKV * HD, hidden);
        }
        deq(wo, T3(l, is_attn ? 3 : 4), hidden, NH * HD);
        deq(wg, T3(l, is_attn ? 4 : 5), inter, hidden);
        deq(wu, T3(l, is_attn ? 5 : 6), inter, hidden);
        deq(wd, T3(l, is_attn ? 6 : 7), hidden, inter);
        castb(xn_cur.data_ptr<float>(), xnb, hidden);
        if (!is_attn) {
            mm(qkvb, xnb, wqkv);
            castf(qkvb, qkvT.data_ptr<float>(), conv_dim);
            mm(zb, xnb, wz);
            castf(zb, zT.data_ptr<float>(), value_dim);
            mm(bob, xnb, wb);
            castf(bob, boT.data_ptr<float>(), NV);
            mm(aob, xnb, wa);
            castf(aob, aoT.data_ptr<float>(), NV);
            if (getenv("IXRUN_BATCH_GDN") == nullptr ||
                std::string(getenv("IXRUN_BATCH_GDN")) != "0") {
            gdn_core_batch(qkvT.data_ptr<float>(), zT.data_ptr<float>(),
                           boT.data_ptr<float>(), aoT.data_ptr<float>(),
                           s27_gex[ig * 4], s27_gex[ig * 4 + 1],
                           s27_gex[ig * 4 + 2], s27_gex[ig * 4 + 3],
                           s27_gnorm[ig], s27_convst[ig], s27_S[ig],
                           onT.data_ptr<float>(), T, NV, NK, DK, DV, l);
            } else {
            for (int i = 0; i < T; ++i)
                gdn_core_from_proj(
                    qkvT.data_ptr<float>() + (size_t)i * conv_dim,
                    zT.data_ptr<float>() + (size_t)i * value_dim,
                    boT.data_ptr<float>() + (size_t)i * NV,
                    aoT.data_ptr<float>() + (size_t)i * NV,
                    s27_gex[ig * 4], s27_gex[ig * 4 + 1],
                    s27_gex[ig * 4 + 2], s27_gex[ig * 4 + 3],
                    s27_gnorm[ig], s27_convst[ig], s27_S[ig],
                    onT.data_ptr<float>() + (size_t)i * value_dim,
                    NV, NK, DK, DV, l);
            }
            castb(onT.data_ptr<float>(), onb, value_dim);
            mm(ob, onb, wo);
            castf(ob, oT.data_ptr<float>(), hidden);
            ig++;
        } else {
            mm(q2b, xnb, wq);
            castf(q2b, q2T.data_ptr<float>(), NH * HD * 2);
            mm(k2b, xnb, wk);
            castf(k2b, k2T.data_ptr<float>(), NKV * HD);
            mm(v2b, xnb, wv);
            castf(v2b, v2T.data_ptr<float>(), NKV * HD);
            attn_batch_proj(q2T, k2T, v2T, attT, dpos,
                            s27_aex[ia * 2], s27_aex[ia * 2 + 1], s27_kv[ia],
                            theta, T, NH, NKV, HD, (int)s27_ctx);
            castb(attT.data_ptr<float>(), attb, NH * HD);
            mm(ob, attb, wo);
            castf(ob, oT.data_ptr<float>(), hidden);
            ia++;
        }
        add_norm_f32<<<(unsigned)T, 1024, 0, st>>>(
            h_cur.data_ptr<float>(), oT.data_ptr<float>(),
            s27_nw2[l].data_ptr<float>(), h1T.data_ptr<float>(),
            xn2T.data_ptr<float>(), hidden, 1e-6f);
        castb(xn2T.data_ptr<float>(), xn2b, hidden);
        mm(mgb, xn2b, wg);
        castf(mgb, mgT.data_ptr<float>(), inter);
        mm(mub, xn2b, wu);
        castf(mub, muT.data_ptr<float>(), inter);
        silu_kernel<<<(unsigned)((T * inter + 255) / 256), 256, 0, st>>>(
            mgT.data_ptr<float>(), muT.data_ptr<float>(),
            actT.data_ptr<float>(), T * inter);
        castb(actT.data_ptr<float>(), actb, inter);
        mm(mdb, actb, wd);
        castf(mdb, mdT.data_ptr<float>(), hidden);
        add_norm_f32<<<(unsigned)T, 1024, 0, st>>>(
            h1T.data_ptr<float>(), mdT.data_ptr<float>(),
            next_w.data_ptr<float>(), h_nxt.data_ptr<float>(),
            xn_nxt.data_ptr<float>(), hidden, 1e-6f);
        torch::Tensor ht = h_cur; h_cur = h_nxt; h_nxt = ht;
        torch::Tensor xt = xn_cur; xn_cur = xn_nxt; xn_nxt = xt;
    }
    int64_t vocab = s27_lh_i.numel() * 2 / hidden;
    udcq_gemv_launch(
        xn_cur.data_ptr<float>() + (size_t)(T - 1) * hidden,
        s27_lh_i.data_ptr<uint8_t>(),
        reinterpret_cast<const uint32_t*>(s27_lh_s.data_ptr()),
        reinterpret_cast<const __half*>(s27_lh_sc.data_ptr()),
        s27_cb.data_ptr<float>(),
        lg.data_ptr<float>(), (int)vocab, hidden, 16, st);
    argmax_f32<<<1, 256, 0, st>>>(
        lg.data_ptr<float>(), (int)vocab, s27_tok.data_ptr<int64_t>());
}


// Processes 8 tokens (rows of h8) at the positions in dpos8 (device int32[8],
// written by the caller between replays -- capture-safe: no host syncs inside).
// final_block=1 runs row 7 -> lm_head -> argmax (s27_tok).
void step27_prefill(torch::Tensor h8, torch::Tensor dpos8, double theta,
                    int64_t final_block) {
    if (!s27_init) throw std::runtime_error("init27 not called");
    auto st = at::cuda::getCurrentCUDAStream();
    int hidden = (int)s27_hidden;
    int inter = (int)s27_inter;
    int nl = (int)s27_nw1.size();
    int GROUP = 16;
    const int NV = 48, NK = 16, DK = 128, DV = 128;
    const int NH = 24, NKV = 4, HD = 256;
    int value_dim = NV * DV;
    int conv_dim = 2 * NK * DK + value_dim;
    auto f32 = torch::TensorOptions()
        .dtype(torch::kFloat32).device(h8.device());
    static torch::Tensor xn8, h1_8, xn2_8, out8, xn_next8,
        qkv8, z8, bo8, ao8, on8, o8, mg8, mu8, act8, md8,
        q2_8, k2_8, v2_8, att8, lg;
    static torch::Tensor xnb, xn2b, onb, actb, attb;
    static bool init = false;
    if (!init) {
        xn8   = torch::empty({8, hidden}, f32);
        h1_8  = torch::empty({8, hidden}, f32);
        xn2_8 = torch::empty({8, hidden}, f32);
        out8  = torch::empty({8, hidden}, f32);
        xn_next8 = torch::empty({8, hidden}, f32);
        qkv8  = torch::empty({8, conv_dim}, f32);
        z8    = torch::empty({8, value_dim}, f32);
        bo8   = torch::empty({8, NV}, f32);
        ao8   = torch::empty({8, NV}, f32);
        on8   = torch::empty({8, value_dim}, f32);
        o8    = torch::empty({8, hidden}, f32);
        mg8   = torch::empty({8, inter}, f32);
        mu8   = torch::empty({8, inter}, f32);
        act8  = torch::empty({8, inter}, f32);
        md8   = torch::empty({8, hidden}, f32);
        q2_8  = torch::empty({8, NH * HD * 2}, f32);
        k2_8  = torch::empty({8, NKV * HD}, f32);
        v2_8  = torch::empty({8, NKV * HD}, f32);
        att8  = torch::empty({8, NH * HD}, f32);
        int64_t vocab = s27_lh_i.numel() * 2 / hidden;
        lg    = torch::empty({vocab}, f32);
        auto bf = torch::TensorOptions()
            .dtype(torch::kBFloat16).device(h8.device());
        xnb   = torch::empty({8, hidden}, bf);
        xn2b  = torch::empty({8, hidden}, bf);
        onb   = torch::empty({8, value_dim}, bf);
        actb  = torch::empty({8, inter}, bf);
        attb  = torch::empty({8, NH * HD}, bf);
        init = true;
    }
    if (!s27_tok.defined())
        s27_tok = torch::zeros({1}, torch::TensorOptions()
            .dtype(torch::kInt64).device(h8.device()));
    const int* dpos = dpos8.data_ptr<int>();
    rmsnorm_fw_kernel<<<8, 1024, 0, st>>>(
        h8.data_ptr<float>(), s27_nw1[0].data_ptr<float>(),
        xn8.data_ptr<float>(), hidden, 1e-6f);
    auto T3 = [&](int l, int s) -> std::vector<torch::Tensor> {
        int k = (l * 8 + s) * 3;
        return {s27_packs[k], s27_packs[k + 1], s27_packs[k + 2]};
    };
    auto castb = [&](const float* src, torch::Tensor& dst, int n) {
        int total = 8 * n;
        f32_to_bf16_kernel<<<(unsigned)((total + 255) / 256), 256, 0, st>>>(
            src, reinterpret_cast<__nv_bfloat16*>(dst.data_ptr()), total);
    };
    auto mt8 = [&](const __nv_bfloat16* xp,
                   const std::vector<torch::Tensor>& P,
                   float* y, int of, int inf) {
        udcq_gemv_mt8b_launch(
            xp, P[0].data_ptr<uint8_t>(),
            reinterpret_cast<const uint32_t*>(P[1].data_ptr()),
            reinterpret_cast<const __half*>(P[2].data_ptr()),
            s27_cb.data_ptr<float>(), y, of, inf, GROUP, st);
    };
    const float* h_p = h8.data_ptr<float>();
    float* h_out_p = out8.data_ptr<float>();
    const float* xn_p = xn8.data_ptr<float>();
    float* xn_out_p = xn_next8.data_ptr<float>();
    int ig = 0, ia = 0;
    for (int l = 0; l < nl; ++l) {
        bool is_attn = false;
        for (int64_t a : s27_attn_layers)
            if (a == l) { is_attn = true; break; }
        torch::Tensor next_w = (l + 1 < nl) ? s27_nw1[l + 1] : s27_fnw;
        if (!is_attn) {
            castb(xn_p, xnb, hidden);
            const __nv_bfloat16* xb =
                reinterpret_cast<const __nv_bfloat16*>(xnb.data_ptr());
            mt8(xb, T3(l, 0), qkv8.data_ptr<float>(), conv_dim, hidden);
            mt8(xb, T3(l, 1), z8.data_ptr<float>(), value_dim, hidden);
            mt8(xb, T3(l, 2), bo8.data_ptr<float>(), NV, hidden);
            mt8(xb, T3(l, 3), ao8.data_ptr<float>(), NV, hidden);
            for (int i = 0; i < 8; ++i)
                gdn_core_from_proj(
                    qkv8.data_ptr<float>() + (size_t)i * conv_dim,
                    z8.data_ptr<float>() + (size_t)i * value_dim,
                    bo8.data_ptr<float>() + (size_t)i * NV,
                    ao8.data_ptr<float>() + (size_t)i * NV,
                    s27_gex[ig * 4], s27_gex[ig * 4 + 1],
                    s27_gex[ig * 4 + 2], s27_gex[ig * 4 + 3],
                    s27_gnorm[ig], s27_convst[ig], s27_S[ig],
                    on8.data_ptr<float>() + (size_t)i * value_dim,
                    NV, NK, DK, DV, l);
            castb(on8.data_ptr<float>(), onb, value_dim);
            mt8(reinterpret_cast<const __nv_bfloat16*>(onb.data_ptr()),
                T3(l, 4), o8.data_ptr<float>(), hidden, value_dim);
            ig++;
        } else {
            castb(xn_p, xnb, hidden);
            const __nv_bfloat16* xb =
                reinterpret_cast<const __nv_bfloat16*>(xnb.data_ptr());
            mt8(xb, T3(l, 0), q2_8.data_ptr<float>(), NH * HD * 2, hidden);
            mt8(xb, T3(l, 1), k2_8.data_ptr<float>(), NKV * HD, hidden);
            mt8(xb, T3(l, 2), v2_8.data_ptr<float>(), NKV * HD, hidden);
            for (int i = 0; i < 8; ++i)
                attn_from_proj(
                    q2_8.data_ptr<float>() + (size_t)i * NH * HD * 2,
                    k2_8.data_ptr<float>() + (size_t)i * NKV * HD,
                    v2_8.data_ptr<float>() + (size_t)i * NKV * HD,
                    s27_aex[ia * 2], s27_aex[ia * 2 + 1], s27_kv[ia],
                    theta, dpos8.data_ptr<int>() + i,
                    att8.data_ptr<float>() + (size_t)i * NH * HD,
                    NH, NKV, HD, (int)s27_ctx);
            castb(att8.data_ptr<float>(), attb, NH * HD);
            mt8(reinterpret_cast<const __nv_bfloat16*>(attb.data_ptr()),
                T3(l, 3), o8.data_ptr<float>(), hidden, NH * HD);
            ia++;
        }
        add_norm_f32<<<8, 1024, 0, st>>>(
            h_p, o8.data_ptr<float>(), s27_nw2[l].data_ptr<float>(),
            h1_8.data_ptr<float>(), xn2_8.data_ptr<float>(), hidden, 1e-6f);
        castb(xn2_8.data_ptr<float>(), xn2b, hidden);
        const __nv_bfloat16* x2b =
            reinterpret_cast<const __nv_bfloat16*>(xn2b.data_ptr());
        mt8(x2b, T3(l, is_attn ? 4 : 5),
            mg8.data_ptr<float>(), inter, hidden);
        mt8(x2b, T3(l, is_attn ? 5 : 6),
            mu8.data_ptr<float>(), inter, hidden);
        silu_kernel<<<(unsigned)((8 * inter + 255) / 256), 256, 0, st>>>(
            mg8.data_ptr<float>(), mu8.data_ptr<float>(),
            act8.data_ptr<float>(), 8 * inter);
        castb(act8.data_ptr<float>(), actb, inter);
        mt8(reinterpret_cast<const __nv_bfloat16*>(actb.data_ptr()),
            T3(l, is_attn ? 6 : 7),
            md8.data_ptr<float>(), hidden, inter);
        add_norm_f32<<<8, 1024, 0, st>>>(
            h1_8.data_ptr<float>(), md8.data_ptr<float>(),
            next_w.data_ptr<float>(), h_out_p, xn_out_p, hidden, 1e-6f);
        const float* ht = h_p; h_p = h_out_p; h_out_p = (float*)ht;
        const float* xt = xn_p; xn_p = xn_out_p; xn_out_p = (float*)xt;
    }
    if (final_block) {
        int64_t vocab = s27_lh_i.numel() * 2 / hidden;
        udcq_gemv_launch(
            xn_p + (size_t)7 * hidden, s27_lh_i.data_ptr<uint8_t>(),
            reinterpret_cast<const uint32_t*>(s27_lh_s.data_ptr()),
            reinterpret_cast<const __half*>(s27_lh_sc.data_ptr()),
            s27_cb.data_ptr<float>(),
            lg.data_ptr<float>(), (int)vocab, hidden, GROUP, st);
        argmax_f32<<<1, 256, 0, st>>>(
            lg.data_ptr<float>(), (int)vocab, s27_tok.data_ptr<int64_t>());
    }
}

void s27_reset() {
    auto st = at::cuda::getCurrentCUDAStream();
    for (auto& t : s27_convst)
        cudaMemsetAsync(t.data_ptr(), 0, t.numel() * 4, st);
    for (auto& t : s27_S)
        cudaMemsetAsync(t.data_ptr(), 0, t.numel() * 4, st);
    for (auto& t : s27_kv)
        cudaMemsetAsync(t.data_ptr(), 0, t.numel() * 2, st);
}

torch::Tensor s27_get_tok() { return s27_tok; }

torch::Tensor s27_get_lg() { return s27_lg_out.cpu(); }

torch::Tensor s27_get_hnorm() {
    return torch::from_blob(s27_hnorm.data(),
        {(long long)s27_hnorm.size()},
        torch::TensorOptions().dtype(torch::kFloat32)).clone();
}

torch::Tensor s27_get_layer_h() {
    return torch::zeros({1});  // AV probe off
    return torch::stack(s27_layer_h).cpu();
}

void s27_set_probe(int64_t l) { s27_probe_l = (int)l; }
torch::Tensor s27_get_probe_h() { return s27_h_out.cpu(); }


