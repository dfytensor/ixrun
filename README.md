# IXRUN — 单卡 24GB LLM 压缩推理引擎

> 三代权重编码（INT8-X / PEAK-Q / UDCQ）+ Triton 融合 kernel + CUDA Graph 解码 +
> MTP 投机解码。目标：RTX 4090 24GB 上跑 Qwen3.8-27B（51.75GB bf16 → 20GB 压缩）。

## 核心能力

| 模块 | 文件 | 说明 |
|---|---|---|
| **INT8-X** 无损位图编码 | `quantize.py` | bf16→int8→(3,5,8) 嵌套位图，5.5 bpw，对 int8 逐位无损 |
| **PEAK-Q** 近无损指数分组 | `peakq.py` | 10.6 bpw，54dB SNR，ppl +0.14；v2 rows 布局免前缀表 |
| **UDCQ** 4-bit 码本量化 | `udcq.py` | 6.0 bpw，ppl ±0；nibble idx + 融合 GEMV/GEMM |
| **多 token GEMV** | `udcq.py` | `udcq_fused_gemv_mt` T∈{2,4,8}：同走带+同累加式 ⇒ 与 M=1 **逐位一致**，带宽受限 ⇒ T=4 成本≈单 token（投机解码免费验证） |
| **融合 kernel** | `fused.py` `tpab_gemv*.py` | decode+GEMV 单 kernel，bf16 权重永不落地（3× 流量→1×） |
| **CUDA mma kernel** | `experiments/udcq_mma/` | m16n8k16 手写 mma.sync + cp.async 双缓冲，M=256 时 1.06-2.26× vs cublas |
| **27B 投机解码** | `experiments/qwen38_udcq/round4b_bisect.py` | 队列架构 k=3：MTP 链式草稿 + T=4 单图验证 + 单同步 + **零前缀重放** |
| **流式引擎** | `engine.py` | cached / streaming / graph 三模式 + ResourceScheduler |
| **OpenAI 兼容 server** | `cli.py serve` | fastapi，SSE 流式，`<think>` 自动隐藏 |

## 三代编码格式

```
INT8-X  5.5 bpw  对 int8 无损（ppl 差全部来自 int8 量化本身）— 基础设施
PEAK-Q  10.6 bpw 54dB SNR，69% 元素 bit-exact — 近无损档
UDCQ    6.0 bpw  4-bit 码本 + per-group scale，ppl ±0 — 27B 单卡落地的主力
ixgs    4.2 bpw  per-group scale（group=64），25.4dB — 重尾权重/视频 DiT 方向
TPAB    2-6 bit  64×64 tile 定长布局 — 小模型 8.9×，27B 不敌 UDCQ
```

## 快速使用

```python
from ixrun import Int8XEngine
from ixrun.config import MODEL_PATH

eng = Int8XEngine.from_pretrained(MODEL_PATH, mode="cached")
print(eng.generate("The theory of relativity states that", max_new_tokens=64))
for chunk in eng.stream("Once upon a time", max_new_tokens=64):
    print(chunk, end="", flush=True)
```

```bash
python -m ixrun.cli search                  # 穷举最优位图组合
python -m ixrun.cli generate "Hello" --stream
python -m ixrun.cli bench                   # bf16 vs INT8-X

# 27B UDCQ 6bpw + CUDA Graph (~15 tok/s, blob 9s 快载)
python -m ixrun.cli chat --model E:\models\Qwen3.8-27B --mode udcq-graph \
    --cache experiments/qwen38_udcq/q38_blob.pt
python -m ixrun.cli serve --model E:\models\Qwen3.8-27B --mode udcq-graph \
    --cache experiments/qwen38_udcq/q38_blob.pt --port 8000   # OpenAI 兼容
```

## MiniCPM5-1B 基准（今日实测）

| 模式 | 前向 | GPU 显存 | ppl | 存储 | 压缩比 |
|---|---|---|---|---|---|
| bf16 基线 | 37ms | 2.2GB | 56.02 | 2161MB | 1.0× |
| INT8-X cached | 38ms | 2.2GB | 62.15 | 463MB | 4.66× |
| INT8-X streaming | 49ms | 1.3GB | — | 463MB* | 4.66× |
| INT8-X graph | 44ms | 4.3GB | — | 463MB* | 4.66× |
| UDCQ (4-bit) | ≡bf16 | — | ±0 | ~390MB | ~5.5× |

无损等价性：INT8-X == 纯 int8 逐位相等（ppl 62.1496 完全相同，10 层 bit-exact，
greedy 一致率 100%）。多 token GEMV bit-exact 测试：
`python -X utf8 experiments/qwen38_udcq/test_mt_gemv.py`（全部形状 × T∈{2,4,8} 零差异）。

## Qwen3.8-27B 单卡推理（UDCQ 6bpw）

64 层混合架构（48 GatedDeltaNet 线性注意力 + 16 全注意力），27B bf16 51.75GB
无法进 24GB；UDCQ 压到 ~20GB（blob 快速部署 `q38_blob.pt` 21.76GB，18s 加载
vs 55min 重量化）。

### 性能阶梯（全部文本连贯）

| 阶段 | 速度 | 说明 |
|---|---|---|
| INT8-X streaming 基线 | 310→138 ms/tok | cumsum 削减 + fused GEMV + fla 绑定 + split-K |
| **MiniCPM5 整步图解码** | **~50-130 tok/s** | `StepGraphEngine`（`--mode step-graph`），延迟同步后 134 tok/s |
| UDCQ 静态 KV + CUDA Graph 贪心 | **15.4→33.7 tok/s** | 手写 CUDA GEMV（`UDCQ_CUDA_GEMV=1`，~700GB/s）|
| **队列投机 k=3 + CUDA + 真-h seed** | **53.4/45.4/32.3 tok/s**（英/码/中）| 草稿从 `out_h4` 真主模型 h 起拟（递归漂移消除），43-50ms/迭代 |
| 队列投机（递归 seed 对照）| 39.9/35.8/25.2 tok/s | 同架构，seed 为 MTP 递归近似 |

投机解码当前受限于两件事：① 权重读取地板（20GB@220GB/s≈106ms/前向，T=4 已摊薄
到 26ms/token）；② MTP 链式草稿质量衰减（d1 用真 h ~75% 接受，d2/d3 用递归近似
递减 → E=2.22；中文最弱 1.25）。加深到 k=7（T=8）预估 22-25 tok/s。

### 调试中钉死的四个暗坑（详见 AGENTS.md）

1. **transformers 5.15 缓存 `conv_states`/`recurrent_states` 是 dict** — 迭代对象
   得到 int key，`isinstance(Tensor)` 守卫静默失效 → 快照/回滚从未生效（曾伪装成
   "10-30 token 后文本退化"+"图 vs eager 数值不同"两个玄学）。
2. **注意力输出 `[B,H,S,D]` reshape 前必须 transpose(1,2)** — q_len=1 碰巧无害，
   q_len≥2 head/token 布局乱码（dmax 6.0）。
3. **CUDA Graph 共享池混叠** — 捕获返回的张量在后续图回放中被覆盖，输出必须在
   图内拷入静态 buffer（症状：token id 变 float 位模式）。
4. **"同步税"不存在** — event 门铃轮询证明 wall==GPU 时间；真凶是诊断克隆泄漏
   3.5GB 显存 → WDDM sysmem 换页（GPU 时间翻倍）。计时前必须释放诊断 + 查
   `mem_get_info`。

| 格式/能力 | 库级 API | CLI（chat/generate）| serve（OpenAI 兼容）|
|---|---|---|---|
| **INT8-X**（cached/streaming/graph）| ✅ `Int8XEngine` | ✅ 默认 | ✅ |
| **PEAK-Q** 10.6bpw 54dB | ✅ `deploy_peakq` | ✅ `--codec peakq` | ✅ 同左 |
| **UDCQ 27B 图解码**（blob + 15 tok/s）| ✅ `Q38GraphEngine.from_blob` | ✅ `--mode udcq-graph` | ✅ 同左 |

`Q38GraphEngine`（`ixrun/q38_graph.py`）与 `Int8XEngine` / `PeakQEngine`（`ixrun/peakq_engine.py`）
/ `StepGraphEngine`（`ixrun/step_graph.py`，整步图解码 ~50 tok/s）同接口
（tokenizer/generate/stream），chat/serve 无缝继承（`<think>` 隐藏、SSE 流式均可用）：

```bash
# INT8-X（默认）          python -m ixrun.cli chat
# PEAK-Q 近无损           python -m ixrun.cli chat --codec peakq
# 整步图解码（快）        python -m ixrun.cli chat --mode step-graph --codec udcq
# 27B UDCQ 图解码         python -m ixrun.cli chat --mode udcq-graph --cache q38_blob.pt
# 27B 投机解码（最快）    python -m ixrun.cli chat --mode udcq-spec --cache q38_blob.pt
# 同上作为 OpenAI 服务    python -m ixrun.cli serve --mode udcq-spec --cache q38_blob.pt --port 8000
```

## API Server（OpenAI 兼容）

```powershell
$env:HF_HUB_OFFLINE='1'; $env:PYTHONPATH='E:\IXRUN'
& 'F:\rwkv\.venv\Scripts\python.exe' -m ixrun.cli serve --model E:\models\Qwen3.8-27B --cache E:\models\qwen38_packed.pt --port 8000 --model-id qwen3.8-27b
```

端点：`GET /v1/models` · `POST /v1/chat/completions`（`stream=true` SSE）· `GET /health`。
opencode 接入：provider baseURL `http://127.0.0.1:8000/v1`，模型 `qwen3.8-27b`。

## 项目结构

```
ixrun/
├── config.py            # 路径与常量
├── quantize.py          # INT8-X: bf16→int8→(3,5,8) 位图
├── peakq.py             # PEAK-Q v2: 指数分组近无损 + 融合 GEMV
├── udcq.py              # UDCQ: 4-bit 码本 + 融合 GEMV/GEMM/多token GEMV
├── triton_kernels.py    # INT8-X 解码 kernel
├── fused.py             # INT8-X 融合 decode+GEMV
├── tpab*.py hybrid.py   # TPAB tile 编码与混合后端
├── linear.py            # 部署层（Int8XLinear / UdcqLinear / PeakQLinear）
├── engine.py            # 三模式引擎 + ResourceScheduler
├── fla_patch.py         # fla Triton kernel 绑定（否则 HF 回退 120ms/tok Python 循环）
├── gdn_seq_patch.py     # GDN 种子块逐 token 精确路径 (S≤8，投机验证用)
├── generate.py eval_utils.py cli.py mtp.py
ixgs/                    # Group-Scale v3（视频 DiT 方向，见 ixgs/README.md）
experiments/
├── qwen38_udcq/         # 27B 全链路：blob 部署/静态KV图解码/投机解码/test_mt_gemv
├── udcq_mma/            # CUDA mma.sync 批量 kernel
benchmarks/              # bench_minicpm5 / bench_qwen38 / bench_peakq
tests/                   # test_core（无损验证）
```

## 环境与铁律

- Python `F:\rwkv\.venv\Scripts\python.exe` (3.12)，torch 2.13+cu126，triton 3.7.1，
  transformers 5.15（qwen3_5 需 ≥5.8），RTX 4090 24GB / WDDM
- 离线：`HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1`；中文/特殊字符加 `PYTHONUTF8=1`
- **`import pandas` 必须在 `import torch` 之前**（反序会堆损坏 0xC0000374）
- kernel 配置**必须用部署模型实测**校验（孤立计时在此 WDDM 机器不可靠，
  warps=4 孤立更快、在模型中慢 15-60%）
- 计时基准前：释放诊断张量 + `torch.cuda.empty_cache()` + 检查
  `torch.cuda.mem_get_info()`（free=0 ⇒ 已进换页区，数字全废）
