# opt_fix — 优化回归修复与形态学扩展（600U）

你是 mcMPP 的性能优化子 agent。目标有两个：

1. **修复 `nppiAdd_8u_C3RSfs` 回退**：opt_elem 合并后 600U 刷新数据显示 147.2 → 108.5 GB/s（0.74x），而同族 8u C1/C4 分别 1.64x/1.39x。定位并修复（或证明是测量噪声并给出复测数据）。
2. **扩展形态学向量化**：opt_morph 只提速了 8u C1（~2.1x），C4/32f 仍为 1.0x；把向量化路径扩展到 C3/C4/32f（至少 8u C4 与 32f C1）。

## 0. 环境（必须使用 mcpp-node53 MCP）

- 节点 `cjxu@10.0.82.53`，32 × MXC600-U die
- 通过 **mcpp-node53 MCP**：`sync` → `~/mcMPP-opt-opt-fix`；`build`（含 `--benchmark`）、`test`、`run`
- 不做本地 build
- **设备分配：7 → 2**；`CUDA_VISIBLE_DEVICES=<n>`；不占 GPU#0/#1
- 参考：`artifacts/benchmarks/600u_baseline.csv`、`/tmp/bench_logs/` 最新日志、`artifacts/benchmarks/opt_elem/`

## 1. 范围

- `src/nppi/nppi_arithmetic_operations/nppi_arithmetic_executor.h`（8u C3 向量化路径）
- `src/nppi/nppi_morphological_operations/nppi_morphology.cu`（C4/32f 扩展）
- 不改归约/平面拷贝（其他包负责）

## 2. 要求

1. 先用同设备 A/B 复测确认 8u C3 是否真回退（区分 DVFS/噪声）；若回退，定位向量化路径的 C3 分支并修复。
2. 形态学扩展时保持 border/alpha 语义与状态码不变；每族改完记录 before/after。
3. 所有正确性测试（unit/diff）保持通过。

## 3. 验证（mcpp-node53 MCP）

```text
sync   remote_dir=~/mcMPP-opt-opt-fix
build  remote_dir=~/mcMPP-opt-opt-fix  jobs=16
run    ... command="export LD_LIBRARY_PATH=$PWD/build/src:/opt/maca/lib:$LD_LIBRARY_PATH; CUDA_VISIBLE_DEVICES=7 ./build/unit_tests --gtest_filter='*Add*C3*:*Erode*:*Dilate*' --gtest_brief=1"
run    ... command="export LD_LIBRARY_PATH=...; CUDA_VISIBLE_DEVICES=7 ./build/benchmark/npp_benchmark -f nppiAdd_8u_C3RSfs -o /tmp/opt_fix_addc3.csv"
```

- 设备 7→2；每项改动记录同设备 before/after

## 4. Definition of Done

1. 8u C3 回退有明确结论（修复并给出提升，或复测证明为噪声）。
2. 至少 8u C4 与 32f C1 形态学向量化落地。
3. 正确性通过；benchmark before/after 表格。
4. 报告：定位过程、改动、未达预期项。

## 5. 约束

- 不 push、不改集成分支、不删除既有测试。
- 只在 worktree 内改源码；构建/测试/基准走 mcpp-node53 MCP。
- 禁止节点上运行 `cmake_maca`/`make_maca`。
