# IXRUN — 单卡 24GB LLM 压缩推理引擎

> 四代权重编码 + Triton/手写 CUDA kernel + CUDA-Graph + MTP 投机解码 + OpenAI 兼容服务。
> 目标硬件：RTX 4090 24GB / WDDM。旗舰结果：**Qwen3.8-27B 单卡投机解码 60 tok/s**。

## 能力速览

| 能力 | 说明 |
|---|---|
| **压缩格式** | INT8-X 5.5bpw（int8 无损）/ PEAK-Q 10.6bpw（54dB 近无损）/ UDCQ 6bpw（4-bit 码本，ppl≈±0）/ **GMM 5-6bpw（贝叶斯高斯混合 + 残差补偿）** / TPAB / ixgs |
| **解码 kernel** | Triton fused decode+GEMV · 多 token GEMV（bit-exact，T=4 成本≈单 token）· **手写 CUDA GEMV**（~700GB/s，2.2× Triton）· CUDA mma 批量 |
| **图执行** | CUDA-Graph 全步捕获（MiniCPM5 ~134 tok/s）；27B 验证图/链图多图管线（共享池、静态输出） |
| **投机解码** | 队列架构 k=3 + 真-h MTP seed + 概率接受（温度兼容）；贪心 E=2.8 |
| **采样** | temperature / top_p / top_k / presence / frequency / repetition penalty（HF 语义）+ Qwen3.8 官方参数组合 |
| **推理引擎** | 5 个统一接口引擎（tokenizer/generate/stream），全部可挂 CLI + OpenAI serve |
| **模型规模** | MiniCPM5-1B（整步图）→ Qwen3.8-27B（UDCQ blob 19.4GB，9 秒部署）|

## 一、编码格式（四代）

| 格式 | bpw | 精度 | 用途 |
|---|---|---|---|
| **INT8-X** | 5.5 | 对 int8 无损（ppl 差 = int8 量化本身）| 引擎默认（cached/streaming/graph）|
| **PEAK-Q** | 10.6 | 54dB SNR，69% 元素 bit-exact | 近无损档 |
| **UDCQ** | 6.0 | 4-bit 码本（分布自适应），ppl≈±0 | **27B 单卡主力** |
| **GMM**（贝叶斯高斯混合）| 5.0-6.0 | K=32 6bpw ppl 57.92 ≈ UDCQ；+残差 8.5bpw == bf16 | 最省 bpw 档；流式 5.03bpw |
| **ixgs** | 4.2 | per-group scale，25.4dB | 重尾权重/视频 DiT 方向 |
| TPAB | 2-6 | tile 定长 | 原型 |

GMM 要点：EM + Dirichlet 先验拟合分量（自动剪枝），**分量中心带符号 → 无 sign 流**，编码成
UDCQ 兼容布局（sign 流全 1）即可零 kernel 改动复用全部 fused decode+GEMV 基础设施；
残差补偿是精度关键（4.25bpw 70.4 → +4bit 残差 8.5bpw 55.99 == bf16）。

关键设计：UDCQ 的 byte-aligned nibble → **纯 LUT 解码**（无位图/rank/跨字提取），解码便宜是比位宽更值钱的资产；多 token GEMV 与逐位一致因此投机验证免费。

## 二、快速开始

```powershell
# 环境
$env:HF_HUB_OFFLINE='1'; $env:TRANSFORMERS_OFFLINE='1'; $env:PYTHONUTF8='1'
$env:UDCQ_CUDA_GEMV='1'          # 手写 CUDA kernel（贪心 15→34 tok/s 的关键）

# MiniCPM5-1B 整步图（~134 tok/s）
python -m ixrun.cli chat --mode step-graph --codec udcq
# GMM 流式整步图（141 tok/s @ 1.40GB —— 当前最快最省）
python -m ixrun.cli chat --mode step-graph --codec gmm-stream

# Qwen3.8-27B 投机解码（最快）
python -m ixrun.cli chat --mode udcq-spec `
  --model E:\models\Qwen3.8-27B `
  --cache E:\IXRUN\experiments\qwen38_udcq\q38_blob.pt

# 作为 OpenAI 兼容服务
python -m ixrun.cli serve --mode udcq-spec `
  --model E:\models\Qwen3.8-27B --cache ...\q38_blob.pt --port 8000
```

## 三、27B 性能阶梯

| 档位 | 速度 | 条件 |
|---|---|---|
| **投机全速（贪心）** | **~53-60 tok/s**（E=2.8）| `--no-sample` |
| **投机 + 温度**（概率接受）| ~43 tok/s（E=2.08）| 仅 `--temperature`（如 0.9）|
| 完整采样（含 penalty）| ~33 tok/s | top_p/top_k/presence/frequency/rep |
| CUDA 贪心图（无投机）| 33.7 tok/s | `--mode udcq-graph` |
| Triton 贪心（无 CUDA env）| 15.4 tok/s | 不设 `UDCQ_CUDA_GEMV` |

里程碑（本仓库演进）：Triton 贪心 15.4 → CUDA 贪心 33.7 → 投机贪心 53-60 tok/s。
三大提速支柱：**手写 CUDA GEMV**（320→700GB/s，warp-per-row + smem 码本 + 128-bit 装载 + 4 独立累加链）、**队列投机架构**（零前缀重放、单同步）、**真-h MTP seed**（草稿从 `out_h4` 真主模型隐状态起拟，E 2.2→2.8）。

### 投机 × 采样路由规则

- 纯贪心 → argmax 投机（全速；CLI 自动启用 `Q38_GREEDY_ONLY=1`，免 g1 图捕获）
- 仅 temperature → **概率接受**：草稿同温度采样 + `min(1, q(d)/p(d))` 逐槽接受（温度不关投机）
- top_p / top_k / presence / frequency / repetition → 完整采样档（贪心图 + CPU 采样，HF 语义）
- 统计事实：投机收益 ∝ 草稿在 target 分布下的概率质量——温度摊平分布必然降接受率，故采样档位速度低于贪心

### Prefill（首字延迟 TTFT）

27B 曾是逐 token eager prefill（~24ms/token → 2k prompt 近百秒首字）。
现役为**块化 prefill**：`Q38_MAX_BLOCK`（graph 引擎，默认 512）/
`Q38_PREFILL_BLOCK`（spec 引擎，默认 64），全注意力层每层一次 batched SDPA。

| 引擎 @2048 tok prompt | 旧 | 新 | 提速 |
|---|---|---|---|
| graph（S=512）| 97.0s | **7.7s** | 12.6× |
| spec（S=64，@512tok 实测外推）| 100.6s | **4.4s** | 22.8× |

正确性：块化 prefill 与逐 token 路径**生成 token 全同**（16/16）；
prefill top-8 仅 bf16 级名次互换。块大小阶梯实测：256→10.3s、
512→7.7s（甜点）、2048→10.8s（SDPA math 后端物化 S×ctx 分数矩阵反而退化）。
spec 引擎图捕获需 ≥2GB 空闲显存（`Q38_MIN_FREE_GB` 可调；1.5GB 空闲时
1.2 覆盖值实证安全，0.4GB 曾出过静默损坏）。

## 四、Qwen3.8 官方参数组合（一键复刻）

```bash
# 思考模式：temp 1.0, top_p 0.95, top_k 20
python -m ixrun.cli serve --mode udcq-spec --cache q38_blob.pt `
  --model E:\models\Qwen3.8-27B --temperature 1.0 --top-p 0.95 --top-k 20

# 非思考模式：temp 0.7, top_p 0.80, top_k 20, presence_penalty 1.5
python -m ixrun.cli serve --mode udcq-spec --cache q38_blob.pt `
  --model E:\models\Qwen3.8-27B --temperature 0.7 --top-p 0.8 `
  --top-k 20 --presence-penalty 1.5
```

serve 启动参数 = **服务端默认**，API 请求字段可覆盖；OpenAI 请求的
`temperature/top_p/top_k/presence_penalty/frequency_penalty` 字段全部直通。

## 五、引擎矩阵（统一 tokenizer/generate/stream 接口）

| 引擎 | 库 | CLI mode | serve | 说明 |
|---|---|---|---|---|
| Int8XEngine | `ixrun/engine.py` | 默认 / `--codec int8x` | ✅ | INT8-X cached/streaming/graph，HF generate |
| PeakQEngine | `ixrun/peakq_engine.py` | `--codec peakq` | ✅ | PEAK-Q 近无损 |
| StepGraphEngine | `ixrun/step_graph.py` | `--mode step-graph` | ✅ | 整步 CUDA-Graph（Llama-arch，MiniCPM5 134 tok/s）|
| Q38GraphEngine | `ixrun/q38_graph.py` | `--mode udcq-graph` | ✅ | 27B 贪心图解码（33.7 tok/s）|
| **Q38SpecEngine** | `ixrun/q38_spec.py` | `--mode udcq-spec` | ✅ | **27B 投机解码（60/43/33 三档）** |

通用参数：`--max-ctx`（KV 窗口，默认 256，上限 16384）· `--codec` · `--cache`
（blob/打包权重）· 全部采样参数。server 另支持 `<think>` 隐藏、SSE 流式、
请求级采样、连续批处理。

## 六、服务 API

- `GET /v1/models` · `POST /v1/chat/completions`（stream SSE / 非流式）· `GET /health`
- OpenAI 语义：model / messages / max_tokens / stream / temperature / top_p /
  top_k / presence_penalty / frequency_penalty / repetition_penalty
- `reasoning_content` 字段承载思考内容；多轮会话由调用方维护历史（每轮重 prefill）

## 七、架构原理（投机解码）

1. **队列投机**：T=4 单图验证 [已知前缀 + root + 3 草稿]；部分接受 → 回滚 →
   接受前缀作为下块已知 token **自动再验证**（位置簿记精确：槽真实位置 =
   状态前沿 - pending 数）→ 零前缀重放、每迭代一次门铃同步。
2. **真-h seed**：回滚不清 `out_h4` —— 草稿从验证刚算出的**真主模型隐状态**
   seed（按 pending 长度选 4 张链图之一），消除 MTP 递归漂移。
3. **概率接受**（仅温度档）：草稿链同温度采样并存草稿概率 p(d)；验证算
   q(d) = target 同温分布概率；`u < min(1, q/p)` 逐槽接受。
4. **手写 CUDA GEMV**（`experiments/udcq_gemv_cuda/`，env `UDCQ_CUDA_GEMV=1`）：
   每 warp 一行、每线程 8 元素连续块、128-bit 装载、smem 码本 gather、
   4 路独立 fp32 累加；M=1 与 M=4（单遍解码服务 4 token）均可用。

## 八、基准数据

### MiniCPM5-1B（整步图 + 延迟同步）

| 模式 | tok/s | VRAM |
|---|---|---|
| eager cached | 25 | 2.2GB |
| StepGraphEngine bf16 | 105 | 2.28GB |
| StepGraphEngine UDCQ | 134 | 2.28GB |
| **StepGraphEngine GMM 流式（`--codec gmm-stream`）** | **141** | **1.40GB** |
| GMM 流式 eager（对照）| 25.7 | 1.35GB |

GMM 流式整步图 = 当前全局最优：**比 resident 整步图更快（141 vs 134）且显存 -39%**——
fused decode+GEMV 无 decode buffer 往返，整步图消灭 Python/launch，两者叠加 5.5×。

### 27B 负载画像（profiler 结论）

单 token GPU 时间 = 121 个 GEMV（合计 2.2ms，18μs/个）+ attention@KV +
elementwise；真正的税是 CPU 同步与 Python —— 整步图 + 门铃轮询 + 延迟批量
读回逐一消除（WDDM 上无同步税：wall == GPU 时间，前提显存不换页）。

### 编码格式 A/B（wikitext ppl delta vs bf16）

| 格式 | bpw | Δppl |
|---|---|---|
| MXINT8（OCP 对照）| 8.25 | -0.21 |
| UDCQ | 6.0 | +2.04 |
| MXFP6（OCP 对照）| 6.25 | +3.93 |
| INT8-X | 5.46 | +6.13 |

### HPQ 研究线（块级乘积量化，外部报告复现与超越）

对一份外部 HPQ 报告做了完整复现、戳伪与超越（`benchmarks/hpq_*`）：

1. **m（子空间数）才是 PQ 的第一参数**——报告最优 `bs=4 m=8 k=32`，最初只调
   k 的评测方向性错误：m4→m8 让 ppl 44,968 → 374.6 →（k=64）68.5。
2. **报告的 "k=64 + 4bit codes = 10x 压缩" 是存储虚标**：`code_bits` 只进
   存储公式，编码器从未使用；实测 64 中心全活跃、top-16 仅覆盖 40% 块，
   4bit 截断必毁——真实压缩 ~5x。
3. **HPQ × per-block scale（我们补的拼片）**：块内 fp16 scale 恢复被维度
   绑定抹掉的量值信息，每档误差减半：

| 方案 | bpw | ppl |
|---|---|---|
| bf16 | 16 | 56.02 |
| GMM K32 | 6.0 | 57.92 |
| UDCQ | 6.0 | 58.06 |
| HPQ×scale m8k32 | 6.0 | 60.17 |
| **HPQ×scale 混合 down+o**（UDCQ 其余）| **6.36** | **57.22** |
| HPQ×scale m8k64 全模型 | 7.0 | **56.93** |

4. **Triton 运行时 kernel**（`benchmarks/hpqs_runtime.py`）：GEMV 位精确
   （gmax=0.0000），但 0.02-0.04× bf16——gather 延迟链 + 4 行串行游走，
   追平需手写 CUDA。**裁决：部署保持 UDCQ/GMM，HPQ×scale 为研究线**
   （编码器映射不变量与 kernel 蓝图已入 AGENTS.md）。
   附注：kmeans 必须固定 seed——未播种 ppl 抖动 ±0.5。

## 九、调试中钉死的七个暗坑

1. transformers 5.15 缓存 `conv_states/recurrent_states` 是 **dict**——迭代得
   int key，快照/回滚静默失效（曾伪装成"文本退化"与"图 vs eager 不同"）。
2. 注意力输出 `[B,H,S,D]` reshape 前必须 transpose——q_len≥2 布局乱码。
3. CUDA-Graph 共享池混叠——图输出必须拷入静态 buffer。
4. "同步税"是假象——真凶是显存超卖触发 WDDM sysmem 换页；计时前查
   `mem_get_info()`。
5. WDDM 图捕获顺序：S=1 图先于 T=4 图捕获（反序进程静默挂死）；快照流在
   首次前向后再收集（conv/rec dict 初值 None）。
6. **torchvision 导入死锁**（Windows loader-lock）：transformers 导入多模态
   模型（qwen3_5）会拉 torchvision，其 pyd 在 torch CUDA 初始化后加载必永久
   挂死（712MB RSS、0% CPU）。`ixrun/__init__.py` 顶部提前导入 torchvision
   根治——MiniCPM5（纯文本）不触发，故只在 27B 路径暴露。
7. **env 开关 "0" 是真值**：`os.environ.get(X)` 对 "UDCQ_CUDA_GEMV=0" 返回
   "0"（truthy），开关关不掉。一律 `not in ("", "0")`。Triton 的 `None`
   索引同理——静默丢维，必须 `tl.expand_dims`。

## 十、项目结构

```
ixrun/            config.py · quantize.py(INT8-X) · peakq.py · udcq.py
                  triton_kernels.py · fused.py · tpab*.py
                  linear.py · engine.py(Int8XEngine) · peakq_engine.py
                  step_graph.py · q38_graph.py · q38_spec.py(投机)
                  fla_patch.py · gdn_seq_patch.py · sampling.py
                  generate.py · chat.py · cli.py · server.py
ixgs/             Group-Scale 方向
experiments/      qwen38_udcq/(blob/round4b 投机原型) · udcq_gemv_cuda/(手写 kernel)
                  udcq_mma/ · mx 格式 A/B
benchmarks/       bench_minicpm5 · bench_q38_ab · bench_formats_minicpm5 · bench_q38_cfg
tests/            test_core（无损验证）
```

## 环境与铁律

- Python `F:\rwkv\.venv\Scripts\python.exe`；torch 2.13+cu126、triton 3.7.1、
  transformers 5.15；RTX 4090 24GB / WDDM；nvcc 13.1 + VS2022（扩展编译）
- `import pandas` 先于 `import torch`；离线 env 三件套；中文加 `PYTHONUTF8=1`
- kernel 配置必须用部署模型实测校验（孤立计时不可信，warps=4 在模型中慢 15-60%）
- 27B 显存预算：blob 19.4GB + int8 emb 1.27GB + 快照/图 ≈ 21.9GB —— 严禁超 24GB
- 投机引擎构建需先跑一次 CUDA smoke（进程强杀后 WDDM 可能让图捕获挂死）
