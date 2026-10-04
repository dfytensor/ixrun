# cpp_engine_plan.md — IXRUN-C++ 解码引擎路线图

目标：去除 Python 宿主开销（当前 ~3× 差距的引擎侧根因），对齐
llama.cpp tg 循环（365 tok/s @ 11.5bpw MiniCPM5-1B）。当前 Python
StepGraph 稳态 ~137 tok/s @ 11.12bpw。

## 架构（单扩展，零 Python per-token）

```
ixrun/cpp/engine.cu          全部内核 + 宿主循环（一个扩展）
ixrun/cpp/pack.py            权重打包到 .bin（GSQ / bf16xl 双格式）
ixrun/cpp/engine.py          薄封装：加载 bin，暴露 generate()
```

宿主循环（C++ 内）：
```
for each token:
  embed_lookup(tok)                       # bf16 行拷贝
  for layer in 24:
    rmsnorm(x, w_in)                      # fp32 累加
    qkv = gsq_gemv(x)  x3                 # 复用 GSQ kernel（改宿主绑定）
    rope + gqa_attention(kv_cache)        # S=1 路径，flash-decoding 式
    o_proj = gsq_gemv(attn)
    rmsnorm; mlp = gate/up gsq_gemv + silu; down gsq_gemv
    x += residual
  rmsnorm; logits = lm_head_gemv(x)       # 151936 行, split-K
  argmax → next_tok（设备侧，宿主零回读）
```

宿主每 token 只做：一次引擎调用 + 每 N token 批量读回（延迟同步，
同 Python StepGraph 的成熟方案）。

## 里程碑

- **C1 内核整合**：把 GSQ GEMV / bf16xl GEMV / rmsnorm / rope /
  attention / silu-mlp / argmax 合入单 engine.cu，权重结构体
  `LayerWeights{gsq_q,gsq_k,gsq_v,gsq_o,gsq_gate,gsq_up,gsq_down,norms}`。
  验收：单层前向与 Python 侧 bit-exact 对拍。
- **C2 KV + 循环**：KV cache（bf16, [24][kv_heads][ctx][hd]）、RoPE、
  S=1 attention、24 层循环、argmax 留在设备。验收：128 token 生成
  与 Python GSQ StepGraph 贪心一致（token 级）。
- **C3 性能**：批量读回（每 32 token）、kernel 融合
  （rmsnorm+gemv）、GSQ 行分块占用率重构（对齐 MMVQ ny-tiling）。
  验收：tg128 ≥ 250 tok/s @ 5.5bpw（llama.cpp BF16X 365 的 70%）。
- **C4 格式双持**：bf16xl 无损档同路径（11.12bpw）；可选 Q8_1 激活
  量化（llama.cpp 680 上限路径）。

## 已知风险

- MiniCPM5 vocab 151936 → lm_head GEMV 占每 token ~30% 流量，
  需要 split-K + 可选 W8A16。
- GQA head 布局需从 modeling_config 精确对齐（num_key_value_heads）。
- Windows nvcc 13.1 + VS2022 链已验证可编译（本项目三扩展先例）。

## 不做

- prefill 的 C++ 化（Python 块化 prefill 已 48ms，够用）
- 多模型泛化（先钉死 MiniCPM5-1B llama-arch）
