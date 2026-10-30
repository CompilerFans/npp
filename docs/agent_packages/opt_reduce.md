# opt_reduce — 统计归约算子向量化（600U 性能优化）

你是 mcMPP 的性能优化子 agent。目标：在 MXC600-U 节点上，对本分支已完成的**统计归约**算子做向量化与归约结构优化。方法论同 `maca-kernel-dev-and-opt` / `maca-kernel-optimization` / `maca-parallel-patterns`。

## 0. 环境（必须使用 mcpp-node53 MCP）

- 节点 `cjxu@10.0.82.53`，32 × MXC600-U die
- 通过 **mcpp-node53 MCP**：`sync` → 远端 `~/mcMPP-opt-opt-reduce`，`build`（已含 `--benchmark`）、`test`、`run`
- 不做本地 build
- **设备分配：23 → 16**（从大编号开始）；`CUDA_VISIBLE_DEVICES=<n>`
- 基准数据：`artifacts/benchmarks/600u_baseline.csv`

## 1. 基线（600U 中位数）

| 算子 | GB/s |
|---|---:|
| nppiSum_8u_C1R | 16.8 |
| nppiMean_8u_C1R | 14.4 |
| nppiMean_StdDev_8u_C1R | 8.4 |
| nppiMin/Max/MinMax_8u | 13–17 |
| nppiNorm_L1_8u_C1R | 15.6 |
| nppiMaxIndx/MinIndx_8u | 15–17 |

归约 kernel 目前逐元素标量读，block reduce 前的加载带宽很低。

## 2. 范围

- `src/nppi/nppi_statistics_functions/nppi_statistics_executor.{h,cu}`：Sum/Mean/MeanStdDev/Min/Max/MinMax/Indx/Norm_NormDiff/NormRel/DotProd/AverageError/MaximumError 及 buffer-size helper
- `src/include/npp_reduction.cuh`：两段归约核心
- 不改逐元素算术（opt_elem 负责）

## 3. 要求

1. **向量化加载**：每个线程一次读取 16B（`uint4`/`float4`），减少加载指令与事务；对齐/尾部回退。
2. **归约结构**：保持 double 累加与既有两段归约语义；可调整 block/grid、减少中间写回。
3. **语义零变化**：零分母、空 ROI、AC4、masked/COI 等既有约定不变；状态码/校验顺序不变。
4. 先测基线再改；每个家族改完即构建 + 正确性 + benchmark，记录 before/after。

## 4. 验证（mcpp-node53 MCP）

```text
sync   remote_dir=~/mcMPP-opt-opt-reduce
build  remote_dir=~/mcMPP-opt-opt-reduce  jobs=16
run    ... command="export LD_LIBRARY_PATH=$PWD/build/src:/opt/maca/lib:$LD_LIBRARY_PATH; CUDA_VISIBLE_DEVICES=23 ./build/unit_tests --gtest_filter='*Sum*:*Mean*:*Min*:*Max*:*Norm*' --gtest_brief=1"
run    ... command="export LD_LIBRARY_PATH=...; CUDA_VISIBLE_DEVICES=23 ./build/benchmark/npp_benchmark -f nppiSum -o /tmp/opt_reduce_sum.csv"
```

- 设备 23→16 轮流使用；不占 GPU#0
- 记录同设备同尺寸的 before/after GB/s

## 5. Definition of Done

1. Sum/Mean/MeanStdDev/Min/Max/MinMax/Indx/Norm 等主要家族的向量化加载落地。
2. 正确性测试通过；语义零变化。
3. benchmark 可量化提升（before/after 表格）。
4. 报告：改动、归约策略、回退、未达预期项。

## 6. 约束

- 不 push、不改集成分支、不删除既有测试。
- 只在 worktree 内改源码；构建/测试/基准全部走 mcpp-node53 MCP。
- 禁止节点上运行 `cmake_maca`/`make_maca`。
