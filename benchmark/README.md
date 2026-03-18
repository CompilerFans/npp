# NPP Benchmark Quick Start

`benchmark/` 提供当前仓库的 benchmark 入口，固定支持两种运行方式：

- 使用仓库内编译出的 MPP：`build/benchmark/npp_benchmark`
- 使用 NVIDIA NPP：`build-nvidia/benchmark/npp_benchmark`

结果文件默认输出到 `benchmark/results/`。

补充说明：

- `nppi/`、`npps/`、`nppcore/` 下是正式 benchmark 源文件
- 框架当前只保留手写 benchmark 和生成器，不再维护覆盖型自动注册层

## 1. 快速开始

```bash
# 1) 构建 MPP benchmark
./benchmark/scripts/build_benchmark.sh

# 2) 运行 MPP benchmark
./benchmark/scripts/run_benchmark.sh -f Resize

# 3) 构建 NVIDIA benchmark
./benchmark/scripts/build_benchmark.sh --use-nvidia-npp

# 4) 运行 NVIDIA benchmark
./benchmark/scripts/run_benchmark.sh --use-nvidia-npp -f Resize
```

如果要显式指定输出文件：

```bash
./benchmark/scripts/run_benchmark.sh -f Resize -o benchmark/results/resize_mpp
./benchmark/scripts/run_benchmark.sh --use-nvidia-npp -f Resize -o benchmark/results/resize_nvidia
```

## 2. 稳定 CLI

### `scripts/build_benchmark.sh`

```bash
./benchmark/scripts/build_benchmark.sh [--use-nvidia-npp] [--debug] [--clean] [--rebuild] [--arch 89] [-j 8]
```

- 默认构建目录：`build/`
- `--use-nvidia-npp` 时构建目录：`build-nvidia/`
- benchmark 可执行文件固定输出到：`<build_dir>/benchmark/npp_benchmark`

### `scripts/run_benchmark.sh`

```bash
./benchmark/scripts/run_benchmark.sh [--use-nvidia-npp] [-f Resize] [-o out_prefix] [-l]
```

- warmup 和 timed-batch 次数使用程序内固定常量，定义在 `benchmark_base.h`
- 随机初始化使用程序内固定随机种子，定义在 `benchmark_base.h`
- 不带 `-o` 时，默认输出到 `benchmark/results/benchmark_results_YYYYMMDD_HHMMSS.csv`
- `-o foo` 会生成 `foo.csv`
- `--json -o foo` 会生成 `foo.json`
- `-l` 只列出 benchmark，不生成结果文件

benchmark CSV 固定最小字段集：

- `function_name`
- `size`
- `variant_tags`
- `data_type`
- `channels`
- `avg_time_ms`
- `throughput_gbps`
- `success`
- `error`

### `scripts/compare_csv.py`

```bash
python3 benchmark/scripts/compare_csv.py \
  benchmark/results/resize_mpp.csv \
  benchmark/results/resize_nvidia.csv \
  -o benchmark/results/resize_compare.csv \
  --html benchmark/results/resize_compare.html \
  --labels MPP "NVIDIA NPP"
```

比较口径：

- 主指标是 `avg_time_ms`
- 若同一场景出现多行成功结果，先对 `avg_time_ms` 取 `mean`
- 场景主键是 `function_name + size + variant_tags + data_type + channels`
- compare 列 `mpp_time_pct_of_nvidia = (mpp_avg_time_ms / nvidia_avg_time_ms) * 100`
- `< 100%` 表示 MPP 比 NVIDIA 快
- `= 100%` 表示打平
- `> 100%` 表示 MPP 比 NVIDIA 慢

HTML 报告包含：

- 按 `op family` 聚合的 Top Catch-Up / Top Behind 图表
- 单 API 的 size 曲线图
- 明细表

### `scripts/progress_report.py`

```bash
python3 benchmark/scripts/progress_report.py \
  benchmark/results/resize_before.csv \
  benchmark/results/resize_after.csv \
  benchmark/results/resize_nvidia.csv \
  -o benchmark/results/resize_progress.csv \
  --html benchmark/results/resize_progress.html
```

进展口径：

- 若同一场景出现多行成功结果，先对 `avg_time_ms` 取 `mean`
- 场景主键是 `function_name + size + variant_tags + data_type + channels`
- `Improve%`：MPP 当前版本相对上一版本的提升幅度
- `Gap% (After/NVIDIA)`：当前 MPP 耗时是 NVIDIA 的百分之多少
- HTML 报告包含按 `op family` 聚合的 Top Improved / Top Catch-Up 图表

## 3. 新增 Benchmark

```bash
./benchmark/scripts/new_benchmark.sh nppi addc nppi_arithmetic_and_logical_operations.h \
  --functions nppiAddC_8u_C1RSfs,nppiAddC_32f_C1R \
  --variant-tags baseline,stress
```

生成规则：

- 输出文件：`benchmark/<type>/bench_<category>.cpp`
- 生成后文件可以直接编译
- 每个函数会自动注册成 benchmark 条目
- 生成的初始版本默认返回 `success=false`
- `variant_tags` 会写入独立结果列，不再混入 `size`
- 生成文件默认提供 `benchmarkCases()`，需要特殊尺寸时可直接覆写
- 若 benchmark 输入需要随机初始化，必须复用 `benchmark_base.h` 中的公共 helper，保持固定随机种子和结果可复现

这是“可编译骨架”，不是“已完成功能 benchmark”。后续需要把生成的 TODO stub 替换成真实 API 调用和计时逻辑。

补充建议：

- 类似仓库里临时编写的单 API benchmark 原型程序，可以用来借鉴真实 API 调用方式、pitch/ROI/buffer 组织和吞吐量计算
- 但不要直接按这类独立程序继续扩框架；应把可复用逻辑迁入 `benchmark/<type>/bench_*.cpp`
- 原因是独立原型通常直接 `exit(1)`、没有统一 CSV 字段，也不接入 compare / progress
- 可直接复用的经验通常包括：
  - packed 输入的 pitch 按 `width * channels * sizeof(T)` 计算
  - planar 输入的每个 plane pitch 按 `width * sizeof(T)` 计算
  - 图像类 API 的 ROI 统一整理成 `NppiSize{width, height}`
  - planar / packed / scratch buffer 的分配方式先按 API 真实参数布局组织，再迁入 `DeviceBuffer`
  - 吞吐量按 `processed_bytes / elapsed_seconds / 1e9` 计算；若是一次调用同时读写多块内存，`processed_bytes` 应按实际搬运字节数估算

## 4. 设计取向

- benchmark 的目标不是一次性写完所有 API，而是先稳定框架、固定结果格式、降低新增单个 API benchmark 的成本
- 推荐按功能实现节奏逐步补 benchmark，优先补高价值或正在优化的 API
- 当前版本不再提供 `--auto` / `--catalog` 这类批量覆盖入口，避免为了“全量”引入额外维护复杂度
- 对需要随机输入的 benchmark，框架默认要求固定随机种子；前后对比时不得临时改成非确定性初始化

## 5. 文档分工

- 本文档只保留 quick start、稳定 CLI、结果口径
- 架构细节、目录结构、扩展方式见 [FRAMEWORK.md](/local/m01019/mcmpp_perf/npp/benchmark/FRAMEWORK.md)
