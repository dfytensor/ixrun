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
static std::vector<float> s27_hnorm;   // per-layer probe
static std::vector<torch::Tensor> s27_layer_h;

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

int64_t step27(torch::Tensor h, int64_t pos, double theta) {
    if (!s27_init) throw std::runtime_error("init27 not called");
    s27_hnorm.clear();
    s27_layer_h.clear();
    auto st = at::cuda::getCurrentCUDAStream();
    int hidden = (int)s27_hidden;
    int inter = (int)s27_inter;
    int nl = (int)s27_nw1.size();
    int GROUP = 16;
    auto f32 = torch::TensorOptions()
        .dtype(torch::kFloat32).device(h.device());
    static torch::Tensor fn, lg, tok;
    static bool init = false;
    if (!init) {
        fn  = torch::empty({hidden}, f32);
        int64_t vocab = s27_lh_i.numel() * 2 / hidden;
        lg  = torch::empty({vocab}, f32);
        tok = torch::zeros({1}, torch::TensorOptions()
            .dtype(torch::kInt64).device(h.device()));
        init = true;
    }
    auto T3 = [&](int l, int s) -> std::vector<torch::Tensor> {
        int k = (l * 8 + s) * 3;
        return {s27_packs[k], s27_packs[k + 1], s27_packs[k + 2]};
    };
    int ig = 0, ia = 0;
    for (int l = 0; l < nl; ++l) {
        bool is_attn = false;
        for (int64_t a : s27_attn_layers)
            if (a == l) { is_attn = true; break; }
        torch::Tensor hcur;
        if (is_attn) {
            std::vector<torch::Tensor> P;
            for (int s = 0; s < 7; ++s) {
                auto t3 = T3(l, s);
                P.push_back(t3[0]); P.push_back(t3[1]);
                P.push_back(t3[2]);
            }
            hcur = attn_layer_step(
                h, s27_cb, P,
                s27_nw1[l], s27_nw2[l],
                s27_aex[ia * 2], s27_aex[ia * 2 + 1],
                s27_kv[ia], theta, pos,
                24, 4, 256, hidden, inter, s27_ctx);
            ia++;
        } else {
            std::vector<torch::Tensor> P;
            for (int s = 0; s < 8; ++s) {
                auto t3 = T3(l, s);
                P.push_back(t3[0]); P.push_back(t3[1]);
                P.push_back(t3[2]);
            }
            hcur = gdn_decoder_step(
                h, s27_cb, P,
                s27_nw1[l], s27_nw2[l],
                s27_gex[ig * 4], s27_gex[ig * 4 + 1],
                s27_gex[ig * 4 + 2], s27_gex[ig * 4 + 3],
                s27_gnorm[ig], s27_convst[ig], s27_S[ig],
                48, 16, 128, 128, inter);
            ig++;
        }
        h = hcur;
        torch::Tensor hn = hcur.norm();
        s27_hnorm.push_back(hn.item<float>());
        // s27_layer_h.push_back(hcur.clone());  // AV probe off
    }
    rmsnorm_fw_kernel<<<1, 256, 0, st>>>(
        h.data_ptr<float>(), s27_fnw.data_ptr<float>(),
        fn.data_ptr<float>(), hidden, 1e-6f);
    int64_t vocab = s27_lh_i.numel() * 2 / hidden;
    udcq_gemv_kernel<<<(unsigned)vocab, 256, 0, st>>>(
        fn.data_ptr<float>(), s27_lh_i.data_ptr<uint8_t>(),
        reinterpret_cast<const uint32_t*>(s27_lh_s.data_ptr()),
        reinterpret_cast<const __half*>(s27_lh_sc.data_ptr()),
        s27_cb.data_ptr<float>(),
        lg.data_ptr<float>(), hidden, GROUP);
    argmax_f32<<<1, 256, 0, st>>>(
        lg.data_ptr<float>(), (int)vocab,
        tok.data_ptr<int64_t>());
    s27_lg_out = lg;   // probe (diagnostics)
    return tok.item<int64_t>();
}

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


