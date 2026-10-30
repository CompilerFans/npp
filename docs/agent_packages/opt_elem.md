# opt_elem — 逐元素/拷贝算子向量化（600U 性能优化）

你是 mcMPP 的性能优化子 agent。目标：在 MXC600-U (600U) 节点上，对本分支已完成的**逐元素/拷贝**算子做向量化，提升吞吐。以 `docs/kernel_design_patterns.md` 与本机 MACA skill（`maca-kernel-dev-and-opt`、`maca-kernel-optimization`、`maca-parallel-patterns`、`maca-mxcc-builtins`）为方法论：先正确、再测量、再优化。

## 0. 环境（必须使用 mcpp-node53 MCP）

- 节点：`cjxu@10.0.82.53`（32 × MXC600-U die，每 die 72GB）
- 通过 **mcpp-node53 MCP** 工作：`sync`（本地 worktree → 远端 `~/mcMPP-opt-opt-elem`）、`build`（远端 MACA 构建，已含 `--benchmark`）、`test`、`run`
- 不做本地 build；远端构建是唯一支持路径
- **设备分配：31 → 24**（从大编号开始）；用 `CUDA_VISIBLE_DEVICES=<n>` 指定
- 基准数据：`artifacts/benchmarks/600u_baseline.csv`（本地，30 卡并行扫描结果）

## 1. 基线（600U，1280×720 及以上中位数）

| 算子 | GB/s |
|---|---:|
| nppiCopy_8u_C1R | 41.9 |
| nppiCopy_16u_C1R | 84.7 |
| nppiCopy_16s_C1R | 85.0 |
| nppiAdd_8u_C1RSfs | ~28–86 |
| nppiAdd_32f_C1R | 434 |
| nppiCompare_32f_AC4R | 487 |

执行器中**没有任何 128-bit 向量访问**（已核查）——这是主要优化空间。

## 2. 范围

- `src/nppi/nppi_arithmetic_operations/nppi_arithmetic_executor.h`：binary/unary/const/compare 等 kernel
- `src/npps/npps_arithmetic_operations/`：elementwise / constant / unary kernel
- 只做逐元素/拷贝类；不改归约（opt_reduce 负责）与形态学（opt_morph）

## 3. 要求

1. **128-bit 向量化**：`uint4`/`float4` 等 16B 访问，grid-stride + 展开；C1 8u/16u/32f 优先；多通道按像素打包处理。
2. **安全回退**：指针/step 未对齐时走标量路径；尾部元素正确处理；AC4 的 alpha 不被改写等既有语义不变。
3. **行为零变化**：状态码、ROI/step 校验、数学结果与优化前一致；现有 unit/diff 测试必须保持通过。
4. 先测基线（优化前）再改代码，每改一个家族就构建、跑正确性、跑 benchmark，记录 before/after GB/s。

## 4. 验证（全部通过 mcpp-node53 MCP）

```text
sync   remote_dir=~/mcMPP-opt-opt-elem
build  remote_dir=~/mcMPP-opt-opt-elem  jobs=16
run    remote_dir=~/mcMPP-opt-opt-elem  command="export LD_LIBRARY_PATH=$PWD/build/src:/opt/maca/lib:$LD_LIBRARY_PATH; CUDA_VISIBLE_DEVICES=31 ./build/unit_tests --gtest_filter='*Add*:*Copy*:*Mul*:*Sub*:*Div*' --gtest_brief=1"
run    ... command="export LD_LIBRARY_PATH=...; CUDA_VISIBLE_DEVICES=31 ./build/benchmark/npp_benchmark -f nppiAdd -o /tmp/opt_elem_add.csv"
```

- 设备 31→24 轮流使用；不要占用 GPU#0
- 记录每个家族优化前后的 GB/s（同设备、同尺寸）
- 正确性：相关 unit 测试全绿；如能在节点上跑 diff_tests（需要 5090 goldens）则一并验证

## 5. Definition of Done

1. 至少 nppiAdd/Sub/Mul/Div、nppiCopy C1 与 npps elementwise 的向量化路径落地。
2. 正确性测试通过；行为零变化。
3. benchmark 有可量化提升（报告 before/after 表格，注明设备与尺寸）。
4. 报告：改动文件、向量化策略、对齐回退、未达预期项与原因。

## 6. 约束

- 不 push、不改集成分支、不删除既有测试。
- 只在 worktree 内改源码；构建/测试/基准全部走 mcpp-node53 MCP。
- 禁止在节点上运行 `cmake_maca`/`make_maca`（会破坏 cu-bridge）；只使用 MCP build。
