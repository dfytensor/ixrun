# GatedDeltaNet 层规格（27B Stage 3b 执行 spec）

来源：transformers/models/qwen3_5/modeling_qwen3_5.py:387-543
（S=1 decode 路径，use_qk_l2norm_in_kernel=True）

## 解码一步全流程（已有件标注）

```
1. qkv = in_proj_qkv(h)              # [key_dim*2+value_dim]   (GEMV: UDCQ)
2. z   = in_proj_z(h)                # [value_dim]             (GEMV: UDCQ)
3. b   = in_proj_b(h)                # [num_v_heads]           (GEMV: UDCQ)
   a   = in_proj_a(h)                # [num_v_heads]           (GEMV: UDCQ)
4. qkv = conv1d_update(qkv, conv_state, conv1d_w, conv1d_b, silu)   ✅ kernel
5. split q,k (key_dim=nk*dk), v (value_dim=nv*dv)，reshape 成 head
   若 nv//nk > 1: q,k repeat_interleave 到 nv
6. beta = sigmoid(b)
   g = -exp(A_log.float()) * softplus(a.float() + dt_bias)
7. q = l2norm(q); k = l2norm(k)                                     ✅ kernel
   q *= 1/sqrt(head_k_dim)          # 在 recurrent fn 内部做 scale
8. core_out, S = gdn_recurrent(q,k,v,g,beta,S)                      ✅ kernel
9. GATED NORM（Qwen3_5RMSNormGated）:
   o32 = core_out.float()
   o32 = o32 * rsqrt(mean(o32^2) + eps)      # 先 norm
   # gate 在 norm 之后（activation=silu），需再读 forward 尾部确认
   #   是 `o32 * silu(z)` 还是 `o32 * (1+silu(z))` —— 执行时读
   #   modeling_qwen3_5.py RMSNormGated.forward 尾部 3 行定案
10. out = out_proj(gated)            # [value_dim -> hidden]   (GEMV: UDCQ)
```

## 状态与缓存（per layer）
- conv_state: [conv_dim, K-1] fp32     ✅ kernel 原地移位
- recurrent_state S: [nv, head_k_dim, head_v_dim] fp32  ✅ kernel 原地更新

## 27B 维度（从 config 读，勿硬编码）
hidden_size / linear_num_value_heads / linear_num_key_heads /
linear_key_head_dim / linear_value_head_dim / linear_conv_kernel_dim /
hidden_act（silu）

## Stage 3b gate 方案
1. 从 E:\models\Qwen3.8-27B safetensors CPU 读 layer N 的
   linear_attn.* 权重（不整体加载模型）
2. HF 侧：torch_recurrent_gated_delta_rule + 上述流程组装 ref
3. C++ 侧：现有 kernel 拼装
4. gate：out rel-err <= 1e-5 tier（纯 fp32 数学，无量化）
5. 注意 in_proj 是 UDCQ 量化权重 vs HF bf16 —— gate 用同一
   bf16 权重喂两侧（量化层另测），隔离数学误差

## 已有件清单
gdn_recurrent_kernel ✅ / l2norm_kernel ✅ / conv1d_update_kernel ✅ /
silu_kernel ✅ / rmsnorm 变体（gated 版需新增 ~20 行，norm 后接
gate 的组合）/ UDCQ GEMV ✅（含批量 T 版待写，仿 gsq_gemm）
