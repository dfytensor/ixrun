# IXRUN — INT8-X 推理引擎

> Block-INT8 + nested-bitmap (3,5,8) 无损权重编码 + Triton GPU 解码 + 流式推理，
> 面向 LLM 文本生成。整理自 `F:\dg_minicpm5` 的 int8x 研究代码，重构成完整推理引擎。

## 核心能力

| 模块 | 功能 |
|---|---|
| **bfloat16_to_int8x** (`quantize.py`) | bf16 权重 → int8 → (3,5,8) 嵌套位图打包，~5.5 bit/w，2.9× 压缩 |
| **分析搜索** (`search.py`) | 穷举 2~5 级位图组合，按 bit/w 排序找最优方案 |
| **Triton 解码** (`triton_kernels.py`) | 融合单 kernel：cumsum + 位图查询 + 跨字位提取，并行解码 |
| **流推理** (`engine.py`) | GPU常驻packed + 共享decode buf + Triton实时解码 (3模式: cached/streaming/graph) |
| **资源调度** (`ResourceScheduler`) | 估算 bf16/packed/peak 显存，按 GPU 预算选 cached/streaming 模式 |
| **文本生成** (`generate.py`) | greedy/sampling + token 流式输出 |

## (3,5,8) 原理

```
bf16 权重 (16 bit/w)
  ↓  per-tensor scale = max_abs / 127
int8 (8 bit/w, 有微小量化误差)
  ↓  按幅值分三级，嵌套位图无损重编码
  L1: |v| ≤ 3   (~55%) → 3 bit
  L2: 3 < |v| ≤ 15 (~40%) → 5 bit
  L3: |v| > 15  (~4%)  → 8 bit
  + 嵌套位图开销 ~1.49 bit/w
  ≈ 5.5 bit/w → 2.9× vs bf16
```

## 快速使用

```python
from ixrun import Int8XEngine
from ixrun.config import MODEL_PATH

# 加载 + 量化 + 部署 (cached 模式：解码一次，极速推理)
eng = Int8XEngine.from_pretrained(MODEL_PATH, mode="cached")

# 文本生成
print(eng.generate("The theory of relativity states that", max_new_tokens=64))

# 流式输出
for chunk in eng.stream("Once upon a time", max_new_tokens=64):
    print(chunk, end="", flush=True)
```

### 搜索最优编码方案

```python
from transformers import AutoModelForCausalLM
from ixrun.search import search_optimal_levels

model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, dtype="bfloat16")
for r in search_optimal_levels(model, topk=10):
    print(r["level_bits"], f"bpw={r['bpw']:.2f}", f"comp={r['compression']:.2f}x")
```

### 流式推理 (省显存)

```python
eng = Int8XEngine.from_pretrained(MODEL_PATH, mode="streaming")
# packed 数据在 pinned host RAM，GPU 仅需 ~22MB 共享 decode buffer
```

## 命令行

```bash
# 分析最优位图组合
python -m ixrun.cli search --topk 12

# 生成文本
python -m ixrun.cli generate "Hello, my name is" --stream --max-new-tokens 64

# 基准测试 (bf16 vs INT8-X)
python -m ixrun.cli bench
python -m ixrun.cli bench --mode streaming
```

## MiniCPM5-1B 实测结果

| 模式 | 前向 | GPU 显存 | ppl | 存储 | 压缩比 | 说明 |
|---|---|---|---|---|---|---|
| bf16 基线 | 38ms | 2.2GB | 55.90 | 2161MB | 1.0× | 原始精度 |
| 纯 int8 | — | — | 62.15 | 1080MB | 2.0× | 参照 |
| INT8-X cached | 38ms | 2.2GB | 62.15 | 463MB | 4.66× | 解码一次,极速 |
| INT8-X streaming | 46ms | 1.3GB | — | 463MB* | 4.66× | GPU常驻packed+共享buf,实时解码 |
| INT8-X graph | 41ms | 4.3GB | — | 463MB* | 4.66× | CUDA Graph融合168层解码 |

### 精度等价性验证 (tests/test_int8_equivalence.py)

INT8-X 是 int8 的**无损编解码器**，实测与纯 int8 完全等价：

```
层级验证:  10 层 bit-exact (max_diff = 0.0), SNR 相同 (30.25dB)
模型验证:  ppl 纯int8=62.1496  INT8-X=62.1496 (完全相同)
          logits max|diff| = 0.000e+00 (逐位相等)
          greedy next-token 一致率 = 100.00%
```

ppl 55.90→62.15 的差距**全部来自 bf16→int8 量化本身**（任何 int8 量化器相同），
(3,5,8) 位图重编码零额外损失。

\* streaming: packed数据GPU常驻(463MB)+共享decode buf(14MB),总GPU权重≈1.3GB
\* graph: packed+per-layer decode buf,CUDA Graph replay消除168次kernel launch

搜索最优方案（实测）：`(3,4,5,6,8)` bpw=5.33 comp=3.00×，默认 `(3,5,8)` bpw=5.46 comp=2.93×。

## Qwen3.8-27B 适配（多模态 + 混合线性/全注意力）

ixrun 的量化/解码对架构完全透明（任何 `nn.Linear` 都适用）。Qwen3.8-27B
（64 层 = 48 GatedDeltaNet 线性注意力 + 16 全注意力 + 27 层 ViT 视觉塔）：

```
[deploy] 606 layers | packed=16.91GB GPU | shared decode buf=178.3MB
[vram]   allocated=22.45GB (含 bf16 embeddings ~5GB) — 单张 24GB 卡
[verify] 5 层 bit-exact vs 纯 int8 (INT8-X == plain int8 同样成立)
[gen]    "The capital of France is" → "Paris" ✓ / 中文相对论问题 ✓ (310 ms/tok)
```

27B bf16 (51.75GB) 无法放进 24GB GPU；INT8-X streaming 把权重压到 16.91GB
实现单卡推理。大模型路径：CPU 懒加载 → 逐层量化 → packed 逐层上 GPU →
bf16 原权重即时释放（64GB RAM 峰值安全，分块打包限制临时内存 <100MB/层）。

```bash
# 运行 (benchmarks/bench_qwen38.py)
$env:HF_HUB_OFFLINE='1'; & 'F:\rwkv\.venv\Scripts\python.exe' -m benchmarks.bench_qwen38
```

环境要求：transformers>=5.8 (qwen3_5 架构)，模型路径见 `ixrun/config.py:QWEN38_PATH`。

## 项目结构

```
ixrun/
├── config.py            # 路径、常量、默认 (3,5,8) 方案
├── bitpack.py           # 比特流打包/解包
├── quantize.py          # bf16→int8→嵌套位图量化 (bfloat16_to_int8x)
├── search.py            # 穷举搜索最优编码组合 (分析工具)
├── triton_kernels.py    # Triton 融合解码 kernel + PyTorch 后备
├── linear.py            # Int8XLinear 部署层 (cached/live 解码)
├── engine.py            # 流式推理引擎 + ResourceScheduler
├── generate.py          # 文本生成 (greedy/sampling/streaming)
├── eval_utils.py        # 前向测速 + 困惑度评估
└── cli.py               # 命令行入口
ixgs/                    # ★ v3 方案: Group-Scale INT8-X (见 ixgs/README.md)
├── quantize.py          #   per-group scale (max/15) + (3,5,8) 无损编码
├── kernels.py           #   Triton kernel + per-group scale 查表
├── linear.py            #   Int8GSLinear + deploy_model_gs
└── test_gs.py           #   无损/SNR/kernel 等价性测试
benchmarks/
└── bench_minicpm5.py    # MiniCPM5-1B 全流程基准
tests/
├── test_core.py               # 核心正确性测试 (无损验证)
└── test_int8_equivalence.py   # INT8-X == 纯int8 精度等价验证
```

## 未来方向：Group-Scale INT8-X (`ixgs/`)

**per-tensor scale 的量化误差是系统性相关的**，在深层模型上相干累积导致输出崩坏。
在 MiniMax-H3 视频 DiT（50 层 × 10 步去噪）上实测：

| 方案 | scale | bpw | 逐层 SNR | H3 视频 |
|---|---|---|---|---|
| INT8-X per-tensor | `max/127` | 3.29 | 20.1 dB | 方块伪影 |
| NF4 | per-64 非线性 | 4.0 | 20.5 dB | 好 |
| **ixgs (3,5,8) per-group** | **`group_max/15`** | **4.16** | **25.4 dB** | **好** |

ixgs 保留 (3,5,8) 无损编码层，把量化层升级为 per-group scale（group_size=64），
SNR 同时超过 per-tensor 和 NF4。编码层 100% 无损（数值相等），
Triton 与 scatter 解码 bit-exact。适用于重尾权重分布（真实 LLM/DiT 均是）；
纯高斯合成数据上 per-tensor 反而更优（无离群值时全局步长更细）。

详见 [`ixgs/README.md`](ixgs/README.md)，测试：`python -m ixgs.test_gs`

## 环境

- Python: `F:\rwkv\.venv\Scripts\python.exe` (3.12)
- torch 2.13+cu126, triton 3.7.1, transformers 4.57.6, RTX 4090 D
- 离线运行：`HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1`
