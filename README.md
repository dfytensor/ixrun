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
| INT8-X cached | 38ms | 2.2GB | 62.15 | 463MB | 4.66× | 解码一次,极速 |
| INT8-X streaming | 46ms | 1.3GB | — | 463MB* | 4.66× | GPU常驻packed+共享buf,实时解码 |
| INT8-X graph | 41ms | 4.3GB | — | 463MB* | 4.66× | CUDA Graph融合168层解码 |

\* streaming: packed数据GPU常驻(463MB)+共享decode buf(14MB),总GPU权重≈1.3GB
\* graph: packed+per-layer decode buf,CUDA Graph replay消除168次kernel launch

搜索最优方案（实测）：`(3,4,5,6,8)` bpw=5.33 comp=3.00×，默认 `(3,5,8)` bpw=5.46 comp=2.93×。

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
benchmarks/
└── bench_minicpm5.py    # MiniCPM5-1B 全流程基准
tests/
└── test_core.py         # 核心正确性测试 (无损验证)
```

## 环境

- Python: `F:\rwkv\.venv\Scripts\python.exe` (3.12)
- torch 2.13+cu126, triton 3.7.1, transformers 4.57.6, RTX 4090 D
- 离线运行：`HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1`
