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
    double theta_base);

torch::Tensor rmsnorm_out(torch::Tensor x, torch::Tensor w);
void rope_probe(torch::Tensor q, torch::Tensor k, int64_t pos,
                int64_t n_heads, int64_t n_kv_heads, int64_t head_dim,
                double theta_base);
torch::Tensor attn_probe(torch::Tensor q, torch::Tensor kv,
                         int64_t pos, int64_t n_heads,
                         int64_t n_kv_heads, int64_t head_dim,
                         int64_t ctx, int64_t v_off);
torch::Tensor gsq_gemv_out(torch::Tensor x,
                   torch::Tensor codes, torch::Tensor cb,
                   torch::Tensor s8, double s_base, double s_step,
                   int64_t out_f, int64_t in_f);