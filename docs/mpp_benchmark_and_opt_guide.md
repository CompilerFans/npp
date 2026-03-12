# NPP 工程 Benchmark 与性能优化指南

本文档用于说明当前 NPP 仓库中 benchmark 构建、运行、结果对比与性能优化迭代的实际流程。文中的脚本、目录、输出文件和对比规则均对齐当前仓库 `benchmark_base/` 下的真实实现。

## 1. 关键约束

### 1.1 严禁事项

- **禁止在正确性未闭环时做性能优化**：benchmark 和调优只能在 `build` / `build-nvidia` 行为已经可信之后进行。
- **禁止前后 benchmark 场景不一致还直接比较**：filter、warmup、iterations、输入规模、构建模式必须一致。
- **禁止一次混入多个无关热点**：否则无法判断收益来自哪里。
- **禁止只看单次最快样本**：当前框架核心比较指标是 `avg_time_ms`，不是单次最小值。
- **禁止忽略 benchmark 执行目录差异**：`build-bench*` 和 `build*` 都可能产出 `npp_benchmark`，比较前必须确认用的是哪一套产物。
- **禁止性能提升后不做回归验证**：任何 benchmark 收益都不能替代功能正确性。

### 1.2 允许并推荐的做法

- **允许使用独立 benchmark 构建链路**：优先使用 `benchmark_base/build_benchmark.sh` 生成 `build-bench*` 目录下的 benchmark。
- **允许复用主工程构建产物**：若已通过 `./build.sh --benchmark` 生成 `build/benchmark/npp_benchmark` 或 `build-nvidia/benchmark/npp_benchmark`，`run_benchmark.sh` 会自动回退使用。
- **允许同时对比 MPP、NVIDIA NPP、MACA NPP**：当前脚本已支持三套运行模式。
- **允许使用 CSV/HTML 报告做阶段总结**：当前仓库已提供 `compare_csv.py` 与 `progress_report.py`。
- **允许使用 `--auto` 做 coverage 驱动 benchmark**：当前框架已支持从 `api_analysis/coverage.csv` 自动注册 benchmark 入口。

## 2. 工作区结构

围绕 benchmark 和性能优化，重点关注以下目录与产物：

```text
.
├── benchmark_base/                    # benchmark 框架与全部脚本工具
│   ├── benchmark_main.cpp            # CLI 入口
│   ├── benchmark_runner.cpp          # 执行与输出逻辑
│   ├── benchmark_base.h              # 基类、计时、注册、工具
│   ├── build_benchmark.sh            # 独立构建 benchmark
│   ├── run_benchmark.sh              # 运行 benchmark
│   ├── compare_csv.py                # MPP vs NVIDIA 或 before vs after 对比
│   ├── progress_report.py            # before/after/target 进展报告
│   ├── auto/bench_from_coverage.cpp  # coverage 驱动的自动 benchmark
│   ├── nppi/                         # 图像类 benchmark
│   └── npps/                         # 信号类 benchmark
├── build-bench/                      # 独立 MPP benchmark 构建目录
├── build-bench-nvidia/               # 独立 NVIDIA benchmark 构建目录
├── build-bench-maca/                 # 独立 MACA benchmark 构建目录
├── build/benchmark/                  # 主工程 MPP benchmark 产物（若用 build.sh --benchmark）
├── build-nvidia/benchmark/           # 主工程 NVIDIA benchmark 产物
├── build-maca/benchmark/             # 主工程 MACA benchmark 产物
├── src/                              # 被优化的真实实现
├── test/unit/                        # 回归验证测试
└── api_analysis/                     # coverage.csv、模块热度、实现状态分析
```

### 目录作用说明

- **`benchmark_base/`**：当前 benchmark 能力的核心目录，除了脚本，还包含 benchmark 主程序、模板和自动注册代码。
- **`build-bench*`**：由 `benchmark_base/build_benchmark.sh` 生成的独立 benchmark 构建目录，通常是最稳定的 benchmark 入口。
- **`build*/benchmark/`**：由主工程 `./build.sh --benchmark` 生成的 benchmark 可执行文件，`run_benchmark.sh` 会优先选择 `build-bench*`，找不到时再回退到这里。
- **`src/`**：性能优化真正落地的代码目录；benchmark 只负责暴露热点，不应直接替代源码分析。
- **`test/unit/`**：优化后必须回归的测试目录。
- **`api_analysis/`**：当需要做 coverage 驱动 benchmark 或结合模块热度排序热点时，这里是上游输入。

## 3. 统一工作流

### 第 1 步：确认 benchmark 目标

先明确本轮工作属于哪类：

- 建立 MPP 基线
- 对比 NVIDIA NPP
- 做单 API 热点分析
- 做模块级热点排序
- 验证某次优化前后收益

目标必须具体，不能写成“整体看看哪里慢”这种不可执行范围。

### 第 2 步：固定 benchmark 场景

在采集基线前，先固定以下信息：

- benchmark 模式：MPP / NVIDIA NPP
- filter 名称
- warmup 次数
- iteration 次数
- 输出格式：CSV / JSON
- benchmark 可执行文件来源：`build-bench*` 还是 `build*`

当前 `run_benchmark.sh` 默认 warmup 为 `5`、iterations 为 `100`。若你改了这些值，前后对比时必须保持一致。

### 第 3 步：构建 benchmark

当前仓库有两条实际可用的构建路径：

1. 独立 benchmark 构建：

```bash
./benchmark_base/build_benchmark.sh
./benchmark_base/build_benchmark.sh --use-nvidia-npp
```

2. 主工程联动构建：

```bash
./build.sh --benchmark
./build.sh --benchmark --use-nvidia-npp
```

通常优先使用 `build_benchmark.sh`，因为它会把 benchmark 单独构建到 `build-bench*`，更便于控制输入条件。

### 第 4 步：采集基线数据

推荐命令：

```bash
./benchmark_base/run_benchmark.sh  -f Resize -i 100 -w 5 -o resize
./benchmark_base/run_benchmark.sh --use-nvidia-npp -f Resize -i 100 -w 5 -o resize_ref
```

当前脚本特性需要注意：

- 若未指定 `-o` 且不是 `--json` / `-l`，脚本会自动生成时间戳 CSV。
- `--no-build` 会强制使用现有 benchmark 可执行文件；若路径不存在会直接报错。
- `--build` 会强制重构 benchmark 可执行文件后再运行。

### 第 5 步：对比结果

当前仓库建议用手动双跑 + `compare_csv.py` 的方式对比：

```bash
./benchmark_base/run_benchmark.sh -f Resize -o resize_mpp
./benchmark_base/run_benchmark.sh --use-nvidia-npp -f Resize -o resize_nvidia
python3 benchmark_base/compare_csv.py \
  resize_mpp.csv \
  resize_nvidia.csv \
  -o resize_compare.csv \
  --html resize_compare.html \
  --print-table \
  --filter Resize
```

当前对比脚本中的速度比定义是：

- `speedup_nvidia_over_mpp = nvidia_avg_time_ms / mpp_avg_time_ms`
- 大于 `1` 表示 **MPP 更快**
- 小于 `1` 表示 **NVIDIA NPP 更快**

比较时应以这个规则为准。

### 第 6 步：形成优化假设

在改代码之前，先给出一个明确假设。常见来源包括：

- 临时 buffer 或 scratch buffer 过多
- 小尺寸场景 launch overhead 过高
- pitch / step 处理带来额外分支
- 边界模式处理成本高
- 类型转换或 dispatch 层级过深
- kernel 访问模式不理想

如果 benchmark 数据不足以支撑假设，应先补测，而不是直接改实现。

### 第 7 步：小步修改并回归

每轮迭代只做一组最小必要修改，用来验证当前假设。修改后至少执行：

1. 目标 API 功能测试
2. 相邻模块回归测试
3. 同一组 benchmark

如果优化可能改变行为语义，还应补做 `build-nvidia` 下的对标测试。

### 第 8 步：记录优化前后进展

当前仓库已提供 `progress_report.py`，适合对比：

- MPP(before) vs MPP(after)
- MPP(after) vs NVIDIA(reference)

推荐用法：

```bash
python3 benchmark_base/progress_report.py \
  mpp_before.csv \
  mpp_after.csv \
  nvidia_ref.csv \
  -o mpp_progress.csv \
  --html mpp_progress.html
```

适合在模块级优化迭代中做阶段总结。

## 4. 工具命令参考

### 4.1 独立构建 benchmark

```bash
./benchmark_base/build_benchmark.sh
./benchmark_base/build_benchmark.sh --use-nvidia-npp
```

常用选项包括：

- `--debug`
- `--release`
- `--clean`
- `--rebuild`
- `--arch N`
- `-j N`

### 4.2 运行 benchmark

```bash
./benchmark_base/run_benchmark.sh
./benchmark_base/run_benchmark.sh --use-nvidia-npp
```

常用选项包括：

- `-f, --filter`
- `-i, --iterations`
- `-w, --warmup`
- `-o, --output`
- `--build`
- `--no-build`
- `--auto`
- `--json`
- `--verbose`

### 4.3 手动生成对比报告

```bash
./benchmark_base/run_benchmark.sh -f Resize -o resize_mpp
./benchmark_base/run_benchmark.sh --use-nvidia-npp -f Resize -o resize_nvidia
python3 benchmark_base/compare_csv.py \
  resize_mpp.csv \
  resize_nvidia.csv \
  --labels MPP "NVIDIA NPP" \
  --html resize_compare.html
```

### 4.4 生成优化进展报告

```bash
python3 benchmark_base/progress_report.py \
  mpp_before.csv \
  mpp_after.csv \
  nvidia_ref.csv \
  -o progress.csv \
  --html progress.html
```

## 5. 优化检查清单

### benchmark 执行检查

- [ ] 已确认 benchmark 可执行文件来自哪一个 build 目录
- [ ] 已固定 filter、warmup、iterations
- [ ] 已确认输出文件命名
- [ ] 已确认比较双方使用的是同一组场景
- [ ] 已确认速度比解释规则

### 性能优化检查

- [ ] 功能正确性已稳定
- [ ] 已收集 MPP 基线
- [ ] 已收集 NVIDIA 或 MACA 参考数据
- [ ] 已明确热点排序依据
- [ ] 已提出可验证的优化假设
- [ ] 已完成回归测试
- [ ] 已记录本轮收益、风险与下一步

## 6. 常见问题与处理方式

### benchmark 跑出来但结果不可比

优先检查：

- warmup / iterations 是否一致
- filter 是否一致
- 输入文件是否来自同一轮
- benchmark 可执行文件是否来自不同 build 目录

### `run_benchmark.sh` 找不到 benchmark 可执行文件

优先检查：

- 是否先执行了 `build_benchmark.sh` 或 `./build.sh --benchmark`
- 当前模式是 MPP、NVIDIA 还是 MACA
- 是否误用了 `--no-build`

### compare 结果方向看反了

当前仓库使用的是：

- `speedup_nvidia_over_mpp = nvidia_avg_time_ms / mpp_avg_time_ms`

因此：

- `> 1` 表示 MPP 更快
- `< 1` 表示 NVIDIA NPP 更快

### 优化后 benchmark 变快但结果不可信

优先检查：

- 是否漏跑功能回归
- 是否改变了 benchmark 场景
- 是否只比较了少量样本
- 是否在不同 benchmark 可执行文件之间混比

## 7. 完成标准

一次完整的 benchmark / 性能优化任务完成，至少应满足：

- benchmark 场景可复现
- 基线与参考数据可追溯
- 对比规则清晰
- 优化前后有量化结果
- 功能回归结果明确
- 已留下 CSV、HTML 或阶段性总结

## 8. 关键提醒

1. benchmark 只是暴露性能问题的入口，不是替代源码分析的工具。
2. 当前仓库已经有比较完整的 benchmark 脚本链路，优先复用，不要临时发明新命令流程。
3. `run_benchmark.sh -o` 默认对应单次运行输出文件，跨实现对比请手动分别生成 CSV。
4. `speedup_nvidia_over_mpp` 的解释方向必须记清，否则结论会反。
5. 优化工作必须和功能回归绑定在一起，不能只留下“更快了”的结论。
