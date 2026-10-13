# IXRUN — 单卡 24GB LLM 压缩推理引擎

> 四代权重编码 + Triton/手写 CUDA kernel（含 tensor-core mma）+ CUDA-Graph + MTP 投机解码 + OpenAI 兼容服务。
> 目标硬件：RTX 4090 24GB / WDDM。旗舰结果：**Qwen3.8-27B 单卡 ULS 5.5bpw（C++ 引擎）：解码 37.4 tok/s / 预填充 603 tok/s（901 token 1.53s）**。

## 能力速览

| 能力 | 说明 |
|---|---|
| **压缩格式** | 六档编解码器全家桶：**UDCQ+ULS 5.5bpw（27B 主力：4-bit 码本 + i8 对数组 scale）** / **GSQ 5.5bpw（学习码本×组 scale，1B 带宽王）** / UDCQ 6bpw / **GMM 5-6bpw（贝叶斯高斯混合）** / **HPQ×scale 6.36bpw（块码本×组尺度质量强化）** / PEAK-Q 10.6bpw（54dB）/ **bf16xl 11.5bpw（真无损，逐 bit == bf16）** / INT8-X / ixgs |
| **解码 kernel** | Triton fused decode+GEMV · 多 token GEMV（bit-exact，T=4 成本≈单 token）· **手写 CUDA GEMV**（多包合并发射，787-869GB/s 实测）· CUDA mma 批量 |
| **图执行** | CUDA-Graph 全步捕获（MiniCPM5-1B + torch.compile ~463 tok/s）；27B 验证图/链图多图管线（共享池、静态输出）|
| **投机解码** | 队列架构 + 真-h MTP seed + 概率接受（温度兼容）；**实测 E=2.09 → 36.5 tok/s (en) / 28.5 (zh)**（PY 轨道）|
| **采样** | temperature / top_p / top_k / presence / frequency / repetition penalty（HF 语义）+ Qwen3.8 官方参数组合 |
| **推理引擎** | PY 统一接口引擎 ×5 + **C++ 引擎 ×2**（27B ULS / 1B GSQ），全部可挂 CLI + OpenAI serve |
| **模型规模** | MiniCPM5-1B（整步图 + torch.compile ~463 tok/s）→ Qwen3.8-27B（**ULS blob 17.9GB**，秒级部署）|

## 一、编码格式（多代）

| 格式 | bpw | 精度 | 用途 |
|---|---|---|---|
| **INT8-X** | 5.5 | 对 int8 无损（ppl 差 = int8 量化本身）| 引擎默认（cached/streaming/graph）|
| **PEAK-Q** | 10.6 | 54dB SNR，69% 元素 bit-exact | 近无损档 |
| **UDCQ+ULS** | **5.5** | 4-bit 码本 + i8 对数组 scale（逐行 max 误差 1.29%），ppl ≈ UDCQ | **27B 单卡主力（C++ 引擎）** |
| **UDCQ** | 6.0 | 4-bit 码本（分布自适应），ppl≈±0 | 27B 经典档 |
| **UDCQ-ULS**（对数尺度）| **5.5** | i8 log2 尺度流（每行 [base,step] 头）；尺度误差 ≤1.3%| **1B ~361 tok/s；27B 生产格式**
| **GMM**（贝叶斯高斯混合）| 5.0-6.0 | K=32 6bpw ppl 57.92 ≈ UDCQ；+残差 8.5bpw == bf16 | 最省 bpw 档；流式 5.03bpw |
| **GSQ**（组尺度量化）| **5.50** | K32 学习码本 + 对数 i8 组 scale，**ppl 57.9** | **1B StepGraph ~463 tok/s（compile 后）** |
| **HPQ×scale**（块码本×组尺度）| 6.36 | 混合部署 ppl 57.22；CUDA kernel 0.84-0.90× bf16 | 质量强化层（27B 支持点状注入）|
| **bf16xl**（无损压缩 bf16）| 11.5 | **逐 bit == bf16（零误差）**；~384 tok/s = llama.cpp BF16X 等价 | 无损档：质量敏感部署 |
| **int8g**（int8 + 组尺度，Q8_0 级）| 8.50 | 1B StepGraph ~487 tok/s（= llama.cpp Q8_0 的 96%，deferred-sync 即超越）| 新增：速度优先档 |
| **ixgs** | 4.2 | per-group scale，25.4dB | 重尾权重/视频 DiT 方向 |
| TPAB | 2-6 | tile 定长 | 原型 |

### 量化研究体系（全部可复现，`benchmarks/`）

同比特对标实验定界了设计空间：**学习码本（GSQ-kmeans）在 5.5bpw 同管线下
比解析式级别（高斯分位数/NF4/分段+开方）好 7 ppl 以上**；bf16x 共享 emax
证明指数元数据可无损压缩至 10.85bpw 但截断式进不了 sub-10bpw 档；
int8 逐行在 8bpw 仍无敌（40.6dB）。负结果全部入库（segroot 165、
NF4 65.4、bf16x-mb2 65.15）。

GMM 要点：EM + Dirichlet 先验拟合分量（自动剪枝），**分量中心带符号 → 无 sign 流**，编码成
UDCQ 兼容布局（sign 流全 1）即可零 kernel 改动复用全部 fused decode+GEMV 基础设施；
残差补偿是精度关键（4.25bpw 70.4 → +4bit 残差 8.5bpw 55.99 == bf16）。

关键设计：UDCQ 的 byte-aligned nibble → **纯 LUT 解码**（无位图/rank/跨字提取），解码便宜是比位宽更值钱的资产；多 token GEMV 与逐位一致因此投机验证免费。

## 〇、安装与获取（从零到跑通）

### 0.1 获取源码 + 环境

```powershell
git clone https://github.com/dfytensor/ixrun.git
cd ixrun
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
# Windows Triton（Linux 用官方 triton 包）:
pip install triton-windows
```

要求：Python 3.12、NVIDIA GPU（≥16GB 建议）+ CUDA 12.6/13.1 运行时。
手写 CUDA GEMV 扩展在**首次运行时 JIT 编译**（torch load_inline），
需要 MSVC 2022 Build Tools + CUDA Toolkit，并在编译会话里先执行
`vcvars64.bat`（验证过的构建配方见 AGENTS.md）：

```powershell
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
set PATH=%CD%\.venv\Scripts;%PATH%
```

### 0.2 获取模型（零配置路径：下载到仓库内 models\ 即可）

`ixrun/config.py` 会自动发现 `<仓库>\models\` 下的模型目录——
下载到默认位置后**无需任何环境变量**：

```powershell
huggingface-cli download openbmb/MiniCPM5-1B --local-dir models\MiniCPM5-1B
huggingface-cli download Qwen/Qwen3.8-27B --local-dir models\Qwen3.8-27B
```

模型放别处时用环境变量（**PowerShell 语法**）：

```powershell
$env:IXRUN_MODEL_PATH = "D:\models\MiniCPM5-1B"
$env:IXRUN_QWEN38_PATH = "D:\models\Qwen3.8-27B"
$env:IXRUN_DATASET_CACHE = "D:\hf_datasets"
```

（CMD 用 `set IXRUN_MODEL_PATH=D:\models\MiniCPM5-1B`；Linux 用 `export ...`。）

### 0.3 运行

```powershell
# ① MiniCPM5-1B：零构建，直接跑（整步图 + torch.compile，~463 tok/s）
python -m ixrun.cli chat --mode step-graph --codec gmm-stream
# ② Qwen3.8-27B：先一次性构建 UDCQ blob（之后秒级部署）；
#    再重打包为 ULS 5.5bpw（-8% 字节，tests/repack_uls.py）
python -m experiments.qwen38_udcq.pack_q38_blob
python -m ixrun.cli chat --mode udcq-spec --cache experiments\qwen38_udcq\q38_blob.pt
# ③ Qwen3.8-27B C++ 引擎（最快：解码 37.4 / 预填充 603 tok/s）—— Python API
python -c "from ixrun.cpp_engine_27b import CppQwen27bEngine; eng = CppQwen27bEngine.from_blob(r'F:\models\qwen38_uls_blob.pt', r'E:\models\Qwen3.8-27B', ctx=4096); print(eng.generate('你好', 64))"
# ④ OpenAI 兼容服务
python -m ixrun.cli serve --mode step-graph --codec gsq --port 8000
```

（27B C++ 引擎目前经 Python API 驱动，CLI/serve 接入在路线图上；
构建配方见 `tests/build_27b_ext.cmd` 与 AGENTS.md。）

无容器镜像（本仓库目标是单卡 Windows/WDDM 深度调优，容器化不在路线图）。
评测复现脚本全部在 `benchmarks/`（wikitext ppl、格式 A/B、R-D 扫描）。

## 二、快速开始

```powershell
# 环境
$env:HF_HUB_OFFLINE='1'; $env:TRANSFORMERS_OFFLINE='1'; $env:PYTHONUTF8='1'
$env:UDCQ_CUDA_GEMV='1'          # 手写 CUDA kernel（贪心 15→34 tok/s 的关键）

# MiniCPM5-1B 整步图 + torch.compile（~463 tok/s）
python -m ixrun.cli chat --mode step-graph --codec udcq
# GMM 流式整步图（省显存档，1.40GB）
python -m ixrun.cli chat --mode step-graph --codec gmm-stream
# GSQ（5.5bpw，compile 后 ~463 tok/s）
python -m ixrun.cli chat --mode step-graph --codec gsq
# bf16xl（14.12bpw 真无损，逐 bit == bf16）

# Qwen3.8-27B：最快 = C++ 引擎（见 〇.3 ③）；PY 投机次之
python -m ixrun.cli chat --mode udcq-spec `
  --model E:\models\Qwen3.8-27B `
  --cache E:\IXRUN\experiments\qwen38_udcq\q38_blob.pt

# 作为 OpenAI 兼容服务
python -m ixrun.cli serve --mode udcq-spec `
  --model E:\models\Qwen3.8-27B --cache ...\q38_blob.pt --port 8000
```

## 三、27B 性能阶梯（全部为实测值）

| 档位 | 速度 | 条件 |
|---|---|---|
| **C++ 引擎贪心图解码** | **37.4 tok/s**（27ms/步）| ULS 5.5bpw + CUDA 图（逐位门通过）|
| C++ 引擎 @4K ctx | 35.3 tok/s | 长上下文衰减已消除（attn v3）|
| PY 投机（贪心）| 36.5 tok/s (en) / 28.5 (zh)（E=2.09 实测（接受比较在图内））| mt4 内核 + 队列架构 |
| PY 贪心图 | 31.5 tok/s | udcq-graph |
| 历史档：PY 投机早期宣称 53-60（E=2.8）| 未能在受控测量中复现 | 保留为特定内容/早期口径 |

 decode 天花板账本：权重流 17.6GB @900-940GB/s = 18.7-19.5ms → **实用极限
51-53 tok/s**；当前 37 = 极限的 72%、2-3 小时可及最大值的 ~95%
（每条字节削减路线均已实测关闭：per-32 误差 16.5%、int8 4.0%、融合 mma
解码 ALU+寄存器墙、spec E 上限——见 AGENTS.md 与 tests/ 的裁决脚本）。

里程碑（本仓库演进）：Triton 贪心 15.4 → CUDA 贪心 33.7 → C++ ULS 引擎
37.4 tok/s；预填充逐 token 24ms/tok → 块化 → **反量化+cuBLAS+批量核
603 tok/s（16x）**。
三大提速支柱：**手写 CUDA GEMV**（warp-per-row + smem 码本折叠 + 128-bit
装载 + 多包合并发射）、**层内批量化**（conv/l2norm/注意力/递归全 T-row 化 +
状态常驻 smem）、**反量化 + cuBLAS 分段 GEMM**（权重读摊销到整段）。

### 投机 × 采样路由规则

- 纯贪心 → argmax 投机（全速；CLI 自动启用 `Q38_GREEDY_ONLY=1`，免 g1 图捕获）
- 仅 temperature → **概率接受**：草稿同温度采样 + `min(1, q(d)/p(d))` 逐槽接受（温度不关投机）
- top_p / top_k / presence / frequency / repetition → 完整采样档（贪心图 + CPU 采样，HF 语义）
- 统计事实：投机收益 ∝ 草稿在 target 分布下的概率质量——温度摊平分布必然降接受率，故采样档位速度低于贪心

### Prefill（首字延迟 TTFT）

27B 曾是逐 token eager prefill（~24ms/token）。演进四代：

| 引擎 @2048 tok prompt | 旧 | 新 | 提速 |
|---|---|---|---|
| graph（S=512）| 97.0s | **7.7s** | 12.6× |
| spec（S=64，@512tok 实测外推）| 100.6s | **4.4s** | 22.8× |
| **C++ 引擎（GEMM 段 + 批量核，901 tok 实测）**| ~24.5s 等效 | **1.5s（~600 tok/s）** | **16×** |

C++ GEMM prefill 管线：**反量化到 bf16 + cuBLAS 分段 GEMM**（权重读摊销到
整段）+ 层内批量化（T-row conv/l2norm/注意力 + 状态常驻 smem 的 T-token
递归）+ 逐 token 精确核。段预算（T=256）：反量化 83ms（带宽地板）+
cuBLAS 99ms（~132 TFLOPS）+ 批量核 ~70ms。已实测关闭的替代路线：
int8（链误差 4.0%）、融合 mma 解码 GEMM（解码 ALU + 寄存器墙，v3 1.14ms
vs 对 0.68ms）、双流重叠（段带宽饱和，中性）。

正确性：块化 prefill 与逐 token 路径**生成 token 全同**（16/16）；
C++ GEMM/mt8b 路径与逐 token 文本逐字一致（A/B 门）。
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
| **Qwen27bCppEngine（C++）** | `ixrun/cpp_engine_27b.py` | Python API + CLI + serve（--mode cpp-q38）| — | **27B ULS 5.5bpw：解码 37.4 / 预填充 603 tok/s**；反量化+cuBLAS GEMM 段 + 批量核 + 状态常驻 smem |
| CppGsqEngine（C++）| `ixrun/cpp_engine.py` | `--mode cpp-gsq` | ✅ | 1B GSQ C++ 引擎 |
| Int8XEngine | `ixrun/engine.py` | 默认 / `--codec int8x` | ✅ | INT8-X cached/streaming/graph，HF generate |
| PeakQEngine | `ixrun/peakq_engine.py` | `--codec peakq` | ✅ | PEAK-Q 近无损 |
| StepGraphEngine | `ixrun/step_graph.py` | `--mode step-graph` | ✅ | 整步 CUDA-Graph + torch.compile（MiniCPM5 ~463 tok/s）|
| Q38GraphEngine | `ixrun/q38_graph.py` | `--mode udcq-graph` | ✅ | 27B 贪心图解码（31.5 tok/s）|
| **Q38SpecEngine** | `ixrun/q38_spec.py` | `--mode udcq-spec` | ✅ | **27B 投机解码（实测 36.5 tok/s (en) / 28.5 (zh)，E=2.09）** |

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

### MiniCPM5-1B（整步图 + torch.compile + 延迟同步）

| 模式 | tok/s | VRAM |
|---|---|---|
| eager cached | 25 | 2.2GB |
| StepGraphEngine bf16 | 105 | 2.28GB |
| StepGraphEngine UDCQ（无 compile，历史）| 134 | 2.28GB |
| GMM 流式（无 compile，历史）| 141 | 1.40GB |
| **int8g（8.5bpw，Q8_0 级）** | **~487** | ~1.1GB |
| **GSQ + compile（5.5bpw，现默认）** | **~463** | ~2.3GB |
| bf16xl（无损 11.5bpw）| ~384 | ~2.6GB |
| UDCQ-ULS（5.5bpw）| ~361 | ~1.9GB |
| CppGsqEngine（C++）| 264.9 | — |

llama.cpp 对照（4090，官方数）：F16 371 / BF16X 389 / Q8_0 508 → bf16xl = BF16X 等价；gsq 超 BF16X 19%；int8g = Q8_0 的 96%（deferred-sync 即超越）。协议：48-token 解码、逐 token 同步（保守）；deferred-sync generate 更快。

torch.compile 对 1B 整步图 +8-17%（默认开启，`STEP_GRAPH_COMPILE=0`
关闭）；compile 对 27B GDN 引擎有损（状态函数化 → 退化），保持关闭。

### Qwen3.8-27B（C++ 引擎，ULS 5.5bpw）

| 指标 | 值 |
|---|---|
| decode（短 ctx，CUDA 图）| **37.4 tok/s**（27ms/步，逐位门通过）|
| decode @4K ctx | 35.3 tok/s（长上下文衰减已消除）|
| prefill | **~600 tok/s**（901 tok 1.49s；3061 tok 5.7s）|
| 段预算（T=256）| 反量化 83ms（带宽地板）+ cuBLAS 99ms（~132 TFLOPS）+ 批量核 ~70ms |
| decode GEMV 逐形状 | qkv 787 / z 720 / q 832 / gate 825 / down 793 / o 677 / lm_head 869 GB/s |
| 天花板 | 权重流 17.6GB @940GB/s = 18.7ms → 实用极限 ~51 tok/s；当前为极限的 72% |

已实测关闭的替代路线（数据在 `tests/` 与 AGENTS.md）：per-32 scale
（组误差 16.5%）、int8 prefill（链误差 4.0%）、融合 mma 解码 GEMM
（解码 ALU ≥ mma + 寄存器墙，v3 1.14ms vs 对 0.68ms）、双流重叠
（段带宽饱和，中性）。

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

4. **手写 CUDA kernel 毕业为第三编解码器**（`experiments/hpqs_gemv_cuda/`）：
   warp-per-4-row-group + split-K fp32 原子累加 + uint4 装码 + smem 码本，
   0.84-0.90× bf16 matmul、位精确（gmax 0.0039 归约序噪声级）。6-bit codes
   打包（pack6/unpack6，12B/块）后存储 **6.36 bpw** 兑现。
5. **双路径部署验证**：MiniCPM5 StepGraph（`--codec hpqs-mixed`，48 层
   HpqsLinear + 120 层 UDCQ-stream，同 harness 速度 ≥ UDCQ-stream）与
   27B spec 引擎注入（`Q38_HPQS_INJECT`/`Q38_HPQS_PACK`，1.60GB free
   捕获 + 连贯生成）全绿。
6. **裁决（更新）**：HPQ×scale 混合 = 57.22 ppl @ 6.36bpw——同比特质量
   优于 UDCQ/GMM，速度持平，三编解码器（UDCQ / GMM / HPQ×scale）全部
   可部署。方法论遗产见 AGENTS.md：H2D 计时假象、legacy-stream 图捕获
   逃逸、smem 层级 8+s 索引、CPU 指针进 kernel。

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
- **僵尸进程会污染测量**：被中断的测试留下占卡进程时，后续测出的是竞争
  速度（曾把 27ms/步测成 177ms）——先 `Get-Process python | Stop-Process`
  清场再相信任何回归
- 27B 显存预算：**ULS blob 17.9GB**（legacy UDCQ blob 19.4GB）+ int8 emb
  1.27GB + 快照/图 ≈ 20.4GB —— 严禁超 24GB
- 投机引擎构建需先跑一次 CUDA smoke（进程强杀后 WDDM 可能让图捕获挂死）

---

## 许可证

MIT License（见仓库根目录 `LICENSE`）。

## 开发与贡献

- **环境**：Python 3.12 + CUDA 12.6/13.1；`pip install -r requirements.txt`（Windows Triton 用 `triton-windows`）。
- **单元测试**：`python -m tests.test_core`（INT8-X 无损性）；`python -m ixgs.test_gs`（组尺度包）。
- **量化基准**：`benchmarks/` 下全部可复现（wikitext ppl、格式 A/B、R-D 扫描、kernel 位精确门禁）。
- **kernel 纪律**：任何新 kernel 变体必须先过位精确单测再进部署（AGENTS.md 全量铁律）。
- **提交规范**：`TYPE(scope): 一行结论 + 证据/数字`；负结果同样入库并写明根因。
