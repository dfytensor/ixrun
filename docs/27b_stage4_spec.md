# 27B Stage 4 装配规格（从 E:\models\Qwen3.8-27B config 实测）

## 真实维度（text_config 嵌套，勿读顶层！）
```
hidden_size            5120
num_hidden_layers      64    (48 linear_attention + 16 full_attention)
num_attention_heads    24    num_key_value_heads 4    head_dim 256
linear: nv=48 nk=16 dk=128 dv=128 conv_k=4
intermediate_size      17408
vocab_size             248320   (lm_head GEMV ~24.8万行)
rms_norm_eps           1e-06
rope_theta             None → 走 rope_scaling（默认 1e6? 执行时读
                       text_config.rope_scaling.rope_theta，1B 的坑复刻）
```

## mRoPE（full-attn 层，从 modeling_qwen3_5.py:84-164,554-585 提取）
```
rotary_dim = 64 (partial_rotary_factor 0.25 × head_dim 256)
inv_freq[j] = 1e7^(-2j/64), j = 0..31（theta=1e7）
纯文本：三轴位置全等于 text pos → interleave 分节无影响
emb = cat(freq, freq) → cos/sin [64]
q_rot = q[..., :64]；q[..., 64:] 原样直通（partial！）
q_rot = q_rot*cos + rotate_half_64(q_rot)*sin   # 对 (i, i+32) 对
```
与 1B rope 的三大差异：仅前 64 维旋转 / 配对跨 32 / theta 1e7

## per-layer 权重集
GDN 层 (48): in_proj_qkv [2*key_dim+value_dim=4096+6144=10240, 5120]
  in_proj_z [6144,5120] in_proj_b/a [48,5120] conv1d [10240,1,4]
  dt_bias/A_log [48] norm.w [128] out_proj [5120,6144]
  + input_layernorm/post_attention_layernorm [5120]
full-attn 层 (16): q/k/v/o proj + gate/up/down + 2 norms + q/k norm?
  (qwen3 系 q/k head-norm 执行时从 modeling 确认)

## 量化
UDCQ 6bpw（现有打包管线 experiments/qwen38_udcq 已产出 16.91GB
blob —— Stage 4 直接复用 blob 的 per-layer packs，不重新量化；
blob tensor 命名清单从 experiments/qwen38_udcq/pack_to_disk.py 读）

## 装配顺序（每步一个 gate）
1. single-layer real-weight gate：blob 里 layer 0 (GDN) packs →
   C++ 链 vs HF qwen3_5 层（bf16 权重喂两侧隔离量化差）→ 然后
   量化 tier 复测
2. decode_64 host 循环（48 GDN + 16 full-attn 调度，g_pos 参数化）
3. PyTorch CUDA graph 捕获（自喂架构照搬 1B：argmax→embed→64层
   →argmax + hist + pos_incr）
4. prefill：UDCQ GEMM (T 摊薄) 批量链，逐 token 版先行为对照
5. MTP：round4b_bisect.py 队列语义 + 1B 验证过的 prefill_batch
   验证链；MTP 头权重从 blob 取

## VRAM 预算（24GB 卡）
blob packed 16.91GB + KV: GDN 层 S=128 [nv48,dk,dv] fp32 = 1.5MB
×48 = 73MB；full-attn 层 KV bf16 [2*4, ctx, 256] ctx=512 →
2MB/层 ×16 = 32MB；激活/工作集 ~1GB → 总 ~18.1GB ✓
（桌面程序占用需 <6GB，超出即 WDDM 分页悬崖）
