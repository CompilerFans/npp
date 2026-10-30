# opt_morph — 形态学算子向量化（600U 性能优化）

你是 mcMPP 的性能优化子 agent。目标：在 MXC600-U 节点上对已实现的**形态学**算子做向量化/共享内存优化。方法论同 `maca-kernel-dev-and-opt` / `maca-kernel-optimization` / `maca-parallel-patterns`。

## 0. 环境（必须使用 mcpp-node53 MCP）

- 节点 `cjxu@10.0.82.53`，32 × MXC600-U die
- 通过 **mcpp-node53 MCP**：`sync` → 远端 `~/mcMPP-opt-opt-morph`，`build`（已含 `--benchmark`）、`test`、`run`
- 不做本地 build
- **设备分配：15 → 8**（从大编号开始）；`CUDA_VISIBLE_DEVICES=<n>`
- 基线：`artifacts/benchmarks/600u_baseline.csv`（Erode/Dilate 8u_C1R 约 13 GB/s）

## 1. 范围

- `src/nppi/nppi_morphological_operations/nppi_morphology.{cpp,cu}`
- 家族：Dilate/Erode（普通、3x3、Border、3x3Border）、MorphGradient/Open/Close/TopHat/BlackHat（Border）、GrayDilate/GrayErodeBorder
- 不改算术/归约（opt_elem/opt_reduce 负责）

## 2. 要求

1. **向量化访问**：128-bit 加载相邻像素/结构元素；矩形 mask 可先用行缓存/共享内存。
2. **边界与 alpha**：Border 语义、AC4 alpha 不写、ROI/step 校验与优化前一致。
3. 先测基线再改；每家族改完构建 + 正确性 + benchmark，记录 before/after GB/s。

## 3. 验证（mcpp-node53 MCP）

```text
sync   remote_dir=~/mcMPP-opt-opt-morph
build  remote_dir=~/mcMPP-opt-opt-morph  jobs=16
run    ... command="export LD_LIBRARY_PATH=$PWD/build/src:/opt/maca/lib:$LD_LIBRARY_PATH; CUDA_VISIBLE_DEVICES=15 ./build/unit_tests --gtest_filter='*Erode*:*Dilate*:*Morph*' --gtest_brief=1"
run    ... command="export LD_LIBRARY_PATH=...; CUDA_VISIBLE_DEVICES=15 ./build/benchmark/npp_benchmark -f nppiErode -o /tmp/opt_morph_erode.csv"
```

- 设备 15→8 轮流使用；不占 GPU#0；不要使用设备 7→2（保留给持续负载）
- 记录同设备同尺寸 before/after

## 4. Definition of Done

1. 至少 Erode/Dilate（普通+3x3+Border）向量化路径落地。
2. 正确性通过、语义零变化。
3. benchmark 可量化提升（before/after 表格）。
4. 报告：改动、策略、回退、未达预期项。

## 5. 约束

- 不 push、不改集成分支、不删除既有测试。
- 只在 worktree 内改源码；构建/测试/基准走 mcpp-node53 MCP。
- 禁止节点上运行 `cmake_maca`/`make_maca`。
