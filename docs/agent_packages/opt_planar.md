# opt_planar — 平面/指针数组拷贝向量化（600U）

你是 mcMPP 的性能优化子 agent。目标：提升**平面（planar）/指针数组**拷贝的吞吐。当前 600U 基线只有 2.9–7.9 GB/s（如 `nppiCopy_16s_C3P3R` 7.7、`nppiCopy_32s_P3C3R` 2.9），远低于打包拷贝（opt_elem 已做到 200–470 GB/s）。

## 0. 环境（必须使用 mcpp-node53 MCP）

- 节点 `cjxu@10.0.82.53`，32 × MXC600-U die
- 通过 **mcpp-node53 MCP**：`sync` → 远端 `~/mcMPP-opt-opt-planar`；`build`（已含 `--benchmark`）、`test`、`run`
- 不做本地 build
- **设备分配：15 → 8**（大编号优先）；`CUDA_VISIBLE_DEVICES=<n>`；不占 GPU#0/#1
- 基线：`artifacts/benchmarks/600u_baseline.csv`；最新刷新见 `/tmp/bench_logs/`

## 1. 范围

- `src/nppi/nppi_data_exchange_and_initialization/nppi_copy.cu`（P3C3/C3P3/P4C4/C4P4 等平面/指针数组路径）
- 只做拷贝族；不改颜色转换/几何

## 2. 要求

1. **逐平面 128-bit 拷贝**：每个平面内连续，按 16B 向量化 + grid-stride；正确处理每个平面的 step。
2. 指针数组（`const T * const pSrc[N]`）的对齐/尾部回退；奇数宽度等边界。
3. 语义零变化：step/ROI 校验、状态码、空 ROI 行为不变。
4. 先测基线再改；每族改完构建 + 正确性 + benchmark，记录 before/after GB/s（同设备同尺寸）。

## 3. 验证（mcpp-node53 MCP）

```text
sync   remote_dir=~/mcMPP-opt-opt-planar
build  remote_dir=~/mcMPP-opt-opt-planar  jobs=16
run    ... command="export LD_LIBRARY_PATH=$PWD/build/src:/opt/maca/lib:$LD_LIBRARY_PATH; CUDA_VISIBLE_DEVICES=15 ./build/unit_tests --gtest_filter='*Copy*P3*:*Copy*C3P3*:*Copy*P4*' --gtest_brief=1"
run    ... command="export LD_LIBRARY_PATH=...; CUDA_VISIBLE_DEVICES=15 ./build/benchmark/npp_benchmark -f nppiCopy_8u_C3P3R -o /tmp/opt_planar_c3p3.csv"
```

- 设备 15→8；记录 before/after

## 4. Definition of Done

1. P3C3/C3P3/P4C4/C4P4 代表性 8u/16u/32f 用例向量化。
2. 正确性测试通过、语义零变化。
3. benchmark 有可量化提升（表格）。
4. 报告：改动、策略、回退、未达预期项。

## 5. 约束

- 不 push、不改集成分支、不删除既有测试。
- 只在 worktree 内改源码；构建/测试/基准走 mcpp-node53 MCP。
- 禁止节点上运行 `cmake_maca`/`make_maca`。
