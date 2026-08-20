# ixgs — INT8-X Group-Scale (v3)

> 在 INT8-X (3,5,8) 无损编码之上引入 **per-group scale**，
> 解决 per-tensor scale 在深层模型上误差相干累积的问题。
> 方案验证于 MiniMax-H3 视频 DiT（50 层 × 10 步去噪），实测出片无方块。

## 为什么需要 group scale

INT8-X 原版量化层用 per-tensor scale（`max_abs/127`）。编码层本身无损，
但**量化误差是系统性相关的**：同一层内所有误差方向一致，跨层相干叠加。

- 文本 LLM（~24 层，单次前向）：误差累积 ×24，ppl 55.9→62，可接受 ✅
- 视频 DiT（50 层 × 10 去噪步）：误差累积 ~×500，最终 latents 偏差 4×，输出方块 ❌

per-group scale 让每个 64 元素组有独立步长：
1. **误差去相关** — 组间误差方向随机，互相抵消，按 √N 而非 N 增长
2. **小组步长更细** — 小幅值组的 scale 远小于全局 max/127，组内相对误差骤降

## 实测数据（MiniMax-H3 DiT `blocks.*.attn.qkv_proj`）

| 方案 | 量化 scale | bpw | 逐层 SNR | H3 10 步视频 |
|---|---|---|---|---|
| INT8-X (3,5,8) per-tensor | `max/127` | 3.29 | 20.1 dB | 方块 |
| NF4 (bitsandbytes) | per-64 非线性 | 4.0 | 20.5 dB | 好 |
| **ixgs (3,5,8) per-group** | **`group_max/15`** | **4.16** | **25.4 dB** | **好** |

关键结论：
- **逐层 SNR 相同的两个量化，输出可以天差地别** — 决定因素是误差的相关结构
- ixgs 逐层 SNR **超过** per-tensor INT8-X 和 NF4，bpw 仅 4.16（仍低于原版 5.46）
- 最终 latents：NF4 参考 mean/std = 0.141/1.109；per-tensor 版 0.03/0.77（偏 4 倍）；
  ixgs 0.143/1.33 ✅

### 适用边界（重要）

group-scale 的优势**只在重尾权重分布**上成立。合成测试实测：

| 权重分布 | ixgs SNR | per-tensor SNR | 胜者 |
|---|---|---|---|
| 高斯 (无离群值) | — | **39.6 dB** | per-tensor |
| 重尾 (0.05% 离群, LLM 实测形态) | **22.5 dB** | 7.7~11.9 dB | **ixgs** |

纯高斯数据下全局 max 不膨胀，`/127` 步长天然更细，per-tensor 反而更准。
真实 LLM/DiT 权重都是重尾的（少数大通道值撑爆全局 max），所以 ixgs 是
为真实模型设计的方案。判断准则：`max_abs / std > ~10` 就该用 group-scale。

## 设计细节

```
bf16 权重 (16 bit/w)
  ↓  按 64 元素分组, scale_g = group_max_abs / 15   (max |int| = 15 → 落在 L1/L2)
int 值 (每组内统一步长)
  ↓  按值域分三级 (无损, 不截断):
     L1: v ∈ [-3, 4]    → 3 bit (codes 0-7)
     L2: v ∈ [-15, 16]  → 5 bit (codes 0-31)
     L3: |v| > 15       → 8 bit (raw)
  + 嵌套位图 b1/b2 (~1 bit/w)
  + group_scales (fp16, 16/64 = 0.25 bit/w)
  ≈ 4.16 bit/w, 编码层 100% 无损 (int match = 100.0000%)
```

**为什么除以 15 而不是 127**：`/15` 使组内最大 int 值 = 15，
全部落在 L1+L2（3/5-bit），L3 恒为空 → 平均 bit 最低。
`/4` 则只有 8 级（SNR 掉到 14dB），`/127` 则 int 值膨胀全进 L3（体积爆炸）。

### 实现要点（踩过的坑）

1. **分层必须按值域，不能按百分位** — 百分位截断边缘值，破坏无损性
2. **L3 空流必须补 1 个哑元** — 否则 Triton `tl.load(l3_ptr)` 非法访存 (CUDA crash)
3. **l1/l2 流末尾补 1 个零 word** — 跨字位提取会读 `w1i+1`，最后一个值可能越界
4. **group_scales 存 fp16，量化时必须先做 fp16 舍入再用** — 否则 encode 用 fp32
   scale、decode 用 fp16 scale，重建不闭合（实测差 3.7%）
5. **±0.0 符号位差异是无害的** — `round(-0.4) = -0.0` 浮点保留符号，int 编码后
   解回 +0.0；数值恒等，无损判定用数值相等而非位相等
6. **scatter_add 打包依赖"每位恰好属于一个值"** — sum == OR，int64 累加不溢出

## 使用

```python
from ixgs import int8gs_quantize, decode_weight, Int8GSLinear, deploy_model_gs

# 单层量化
packed = int8gs_quantize(weight, group_size=64)   # weight: bf16 2-D
w_rec = decode_weight(packed)                      # Triton 或 scatter 回退

# 整模型部署 (替换 nn.Linear)
stats = deploy_model_gs(model, cache="full")       # 解码一次, F.linear 直跑
stats = deploy_model_gs(model, cache="none")       # 每次前向实时解码 (省显存)

# 文本生成兼容 ixrun 引擎模式
```

## 测试

```powershell
& 'F:\rwkv\.venv\Scripts\python.exe' -m ixgs.test_gs
```

覆盖：编码无损 (int match 100%)、Triton == scatter、SNR 优于 per-tensor、前向等价。

## 与 h3_engine 的关系

H3 生产实现（流式推理、共享 decode buffer、TE/DiT 分离）在：
- `E:\h3_engine\ix358_triton.py` — kernel + 流式 Linear（本包的 kernel 即由此提炼）
- `E:\h3_engine\encode_dit_v3.py` — GPU 编码器（23.8GB DiT, ~10 分钟）
- `E:\h3_engine\dit_standalone.py` — 独立去噪推理（10 步 787s @ 480p）

## 追记（2026-08-17）：H3 深挖后的方法论教训

在 MiniMax-H3 DiT 上的后续深挖发现：早期"自造量化必糊"的结论被**两个基准 bug 污染**——
(1) standalone NF4 解码数学错误（nibble 顺序 + nested offset 遗漏）；(2) scheduler
shift 用错（7.0 应为 12.0）。修复后重新测得真排名：NF4 pipeline lap=296，
NF4-replica（本组逆向的 bit-exact 复刻）standalone lap=240，TPAB 各变体 15-30。

**最终结论不变（TPAB/DG4 在 H3 画质不足），但方法论教训永久有效：**
- 任何"方案A vs 方案B"的结论，先验证 B 的实现正确性（逆向到 bit-exact 是金标准）
- 嵌套位图/分组量化的解码细节（nibble 顺序、双重 absmax 的 offset）极易错且错得静默
- 逐层 SNR 是必要不充分指标：误差相关结构（相干 vs 去相关）决定深层累积行为

## 终章（2026-08-17 深夜）：TPAB 在 H3 上的盖棺实验

修复全部三个基准 bug（NF4 解码 nibble/offset、scheduler shift、TE embeds 污染）后，
TPAB strip+NU 在干净环境重测：26dB 目标 lap=55，提高到 30dB 目标（bpw 6.0->6.9，
体积 28.5GB，逐层 SDR 30.7dB 反超 NF4 十 dB）画质纹丝不动（lap=48）。
**加 bit 无效 = 结构性死因**：变字宽条带间独立 scale 产生误差不连续，50 层混沌放大
后高频细节死亡。对照 NF4（4.1bpw, lap=711）：统一字宽 + 分位数码表 + 双重 fp8 scale
是系统性优势。结论：变字宽/tile 混合方案不适合深层视频 DiT；TPAB 定位回归文本 LLM。

## 终局定论（2026-08-18）：PEAK-Q 在 H3 上成功

量化考古的最终章：TPAB 全变体覆灭后，PEAK-Q（本库 peakq.py）在 MiniMax-H3 视频
DiT 上盲测胜出 NF4（用户验收"非常好" vs "好"）。成功要素与 TPAB 的失败要素
严格互补：
- 组峰值 BIT-EXACT（TPAB 的 outlier 摘除破坏峰值；int8 缩放破坏峰值）
- 误差锚定组局部相对误差（TPAB 误差锚定条带决策边界）
- 组粒度 16（比 NF4 的 64 更细，误差更局部）
- 46% 权重完全无损 + 54dB SNR

生产配方：NF4-TE + PEAK-Q-DiT(Ref2VA 专权重) + PEAK-Q-VAE + TEACache，
480p/124f/20步 实测 14.5min，画质超 NF4 全套。文档: F:\B站视频计划2\生产管线.md
