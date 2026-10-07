# engine_v5.cu 增量卫生计划（2030 行 → 目标 <1200 行）

## 原则
- 每步之后必须全绿：`tests/test_cpp_engine_e2e.py`（64/64）+
  `test_gemv_v2.py` + `test_prefill_batch.py` + `test_udcq_gemv.py`
  + `test_gdn_recurrent.py` + `test_gdn_primitives.py`
- 每步一个 commit；禁止大爆炸重构
- 任何 kernel 语义变更 → bump 扩展名（铁律）

## Step 1：删除死代码（零语义变更，无 bump）
- `attn_kernel`（v1）：确认 `attn_probe` 引用面后删或保留 probe
- `generate_batch` / `generate_batch_fast`：旧路径（embed_lookup
  历史 bug 所在），确认 2/4 处外部引用为测试后删测试+代码
- `set_start_token`：2 处引用，确认后删
- 保留：sk/v3（研究 gate 活体）、raw_graph（WDDM 负结果证明，
  AGENTS.md 引用）
预期：-250 行

## Step 2：device 函数去重（语义零变更 → 需 bump）
- `__device__ __forceinline__ float gsq_group_dot(...)`
  收敛 5 处复制的 10B 组解码块（v1/v2/sk/v3/gemm）
- `__device__ float block_reduce_sum(float)` 收敛 6 处 red[32] 模式
- bump 所有测试扩展名；全套 gate 重跑
预期：-300 行；消除"修一处忘四处"类 bug 的结构根源

## Step 3：编排层拆分（可选，收益最大风险最高）
- statics g_* → `struct EngineCtx` + handle 参数（多实例不再需要
  模块后缀，内存省 N×0.7GB）
- launcher 与 kernel 分文件（load_inline 拼接编译，源码层分组织）
- 仅在新功能（Stage 3b/4）需要多上下文时做

## 拒绝项
- 不重写 kernel 数学顺序（bit-exact 资产不可再生）
- 不引入模板/抽象层（调试成本 > 收益）
