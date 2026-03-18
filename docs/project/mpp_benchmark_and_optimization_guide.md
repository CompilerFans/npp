# MPP Benchmark 搭建与性能优化指南

本文档参考 `/local/m01019/mcmpp_perf/SKILL.md` 的组织方式编写，但内容严格对齐当前 `npp` 仓库中的真实目录、构建入口、benchmark 脚本与结果处理工具。目标不是单独介绍某个脚本，而是形成一套可复用、可回归、可扩展的 benchmark 与性能优化闭环。

本文档主要回答五个问题：

1. 当前仓库里 benchmark 应该怎么搭建和运行。
2. benchmark 数据怎样进入性能优化流程，而不是停留在“跑出一个 CSV”。
3. 一轮优化结束后怎样形成可复现的结果闭环。
4. Codex 如何基于当前框架和脚本直接生成可用 benchmark 程序。
5. 未来如果扩展 benchmark 能力，应该优先改哪里，尽量不要动哪里。

## 1. 关键约束

### 1.1 严禁事项

- **禁止在正确性未闭环时直接做性能调优**：必须先保证 `build-nvidia` 对标测试和 `build` 下本地实现测试可信。
- **禁止前后 benchmark 场景不一致还直接比较**：filter、固定运行常量、输入规模、构建模式、二进制来源必须一致。
- **禁止把 benchmark 脚本当成唯一事实来源**：性能结论必须回到 `src/`、`test/`、`api_analysis/coverage.csv` 一起分析。
- **禁止一次混入多类优化**：一次迭代只验证一组假设，否则无法归因。
- **禁止 benchmark 收益替代功能回归**：更快不代表正确。
- **禁止为了 benchmark 方便而改动主工程已有配置**：优先复用当前 `build.sh`、根 `CMakeLists.txt`、的现有开关和目录约定。

### 1.2 允许并推荐的做法

- **优先复用当前 benchmark 构建链路**：优先使用 `benchmark/scripts/build_benchmark.sh` 产出 `build/benchmark` 或 `build-nvidia/benchmark` 下的 benchmark。
- **允许直接复用主工程 benchmark 产物**：若已执行 `./build.sh --benchmark`，会复用同一套 `build*/benchmark/npp_benchmark` 产物目录。
- **允许分别采集 MPP / NVIDIA NPP 数据**：当前 benchmark 只维护这两种模式。
- **允许使用 `benchmark/scripts/compare_csv.py` 与 `benchmark/scripts/progress_report.py` 形成阶段报告**：性能数据对比与汇总。
- **允许按功能实现节奏逐步补 benchmark**：优先补正在开发、正在优化、或需要对标汇报的 API，不追求一次性全量落地。
- **允许扩展 benchmark 能力，但默认限制改动范围在 `benchmark/` 和 `docs/` 内**：只有现有接口无法表达需求时，才考虑修改主工程配置。

## 2. 工作区结构

围绕 benchmark 搭建与性能优化，当前仓库中最关键的目录和产物如下：

```text
.
├── benchmark/                    # benchmark 框架、脚本和对比工具
│   ├── benchmark_base.h              # BenchmarkBase / Registry / Result / Timer / DeviceBuffer / DynamicSymbol
│   ├── benchmark_main.cpp            # CLI 入口
│   ├── benchmark_runner.cpp          # 运行与输出
│   ├── benchmark_utils.cu            # 设备与数据初始化工具
│   ├── scripts/                      # benchmark 脚本入口
│   ├── nppcore/                      # nppcore benchmark
│   ├── nppi/                         # nppi benchmark
│   └── npps/                         # npps benchmark
├── build/benchmark/                  # 主工程 MPP benchmark 产物
├── build-nvidia/benchmark/           # 主工程 NVIDIA benchmark 产物
├── src/                              # 实际待优化实现
├── test/unit/                        # 功能与精度回归测试
├── api_analysis/                     # coverage.csv、模块分析、开发排序输入
└── docs/project/                     # 方法文档与流程约定
```

### 目录职责说明

- **`benchmark/`**：负责“构建、运行、输出、对比、扩展入口”，是 benchmark 工程层的事实来源。
- **`build*/benchmark/`**：适合和主工程联动验证，便于复用已有 `build.sh` 配置。
- **`src/`**：性能问题最终仍要落在源码实现上，不应停留在 benchmark 层。
- **`test/unit/`**：承担功能正确性与精度正确性回归。
- **`api_analysis/`**：用于决定“先 benchmark 哪些 API”“优先优化哪些热点”。

## 3. 统一闭环工作流

当前仓库中，benchmark 搭建和性能优化应该按一个固定闭环执行，而不是把 build、run、compare、优化拆成互相独立的零散动作。

### 3.1 闭环总览

```text
确定目标
  -> 固定场景
  -> 构建 benchmark
  -> 采集基线/参考
  -> 对比定位热点
  -> 形成优化假设
  -> 修改 src/ + 回归测试
  -> 再次 benchmark
  -> 生成对比/进展报告
  -> 记录结论并进入下一轮
```

这条链路必须形成闭环。缺少任何一环，benchmark 都只能算“采样”，不能算工程化性能迭代。

### 3.2 第 1 步：确定目标与范围

先明确本轮任务属于哪一类：

- 新搭建一批 benchmark
- 建立某个模块的 MPP 基线
- 与 NVIDIA NPP 做对比
- 验证某次优化前后收益
- 按功能实现或优化节奏逐步补齐 benchmark

每轮任务必须写清楚：

- 目标 API / 模块
- 对比对象
- 输入规模
- 期望产物

不推荐“全量先跑一遍再说”这种没有范围约束的做法。

### 3.3 第 2 步：固定 benchmark 场景

在采样前，需要固定以下条件：

- benchmark 模式：MPP / NVIDIA NPP
- benchmark 来源：`build*/benchmark`
- filter
- 固定运行常量
- 固定随机种子
- 输出格式：CSV 或 JSON

当前默认值来自 `benchmark/benchmark_base.h`：

- warmup 默认 `5`
- timed batch iterations 默认 `100`
- random seed 默认 `kBenchmarkRandomSeed`

只要前后比较，就必须使用同一套条件。

### 3.4 第 3 步：构建 benchmark

当前仓库有两条真实可用的路径。

#### 路径 A：独立 benchmark 构建

```bash
./benchmark/scripts/build_benchmark.sh
./benchmark/scripts/build_benchmark.sh --use-nvidia-npp
```

适用场景：

- 希望显式触发 benchmark 构建
- 希望明确使用 `build/benchmark` 或 `build-nvidia/benchmark` 产物
- 希望更稳定复现实验

#### 路径 B：主工程联动构建

```bash
./build.sh --benchmark
./build.sh --benchmark --use-nvidia-npp
```

适用场景：

- 已经在主工程构建链路中工作
- 希望复用已有 build 目录
- 需要和主工程测试流程一起跑

推荐顺序：默认优先路径 A；只有在需要联动主工程时才回退路径 B。

### 3.5 第 4 步：运行 benchmark 并生成原始数据

推荐先分别采集单边结果，再做对比：

```bash
./benchmark/scripts/run_benchmark.sh -f Resize -o resize_mpp
./benchmark/scripts/run_benchmark.sh --use-nvidia-npp -f Resize -o resize_nvidia
```

脚本职责必须记清：

- `scripts/build_benchmark.sh`：只负责构建
- `scripts/run_benchmark.sh`：只负责运行并生成原始 CSV / JSON
- `benchmark/scripts/compare_csv.py`：只负责对比
- `benchmark/scripts/progress_report.py`：只负责阶段进展汇总

### 3.6 第 5 步：对比结果并定位热点

当前推荐的对比方式是：

```bash
python3 benchmark/scripts/compare_csv.py \
  resize_mpp.csv \
  resize_nvidia.csv \
  -o resize_compare.csv \
  --html resize_compare.html \
  --print-table \
  --filter Resize
```

当前仓库里的关键比较规则是：

- `mpp_time_pct_of_nvidia = (mpp_avg_time_ms / nvidia_avg_time_ms) * 100`
- `< 100%` 表示 **MPP 更快**
- `> 100%` 表示 **NVIDIA NPP 更快**

对比阶段要回答三个问题：

1. 哪些 API / 场景最慢。
2. 哪些 API 与 NVIDIA 的差距最大。
3. 当前热点是单个 API 问题，还是某个模块的共性问题。

### 3.7 第 6 步：形成优化假设

benchmark 数据只负责暴露热点，不负责直接给出答案。优化前必须形成明确假设，例如：

- dispatch 层过深
- 内存访问模式不理想
- pitch / step 处理过重
- scratch buffer 申请与释放过多
- 小尺寸场景 launch overhead 偏高
- border / interpolation 参数路径分支过多

假设必须可验证。没有假设就直接改代码，通常会把性能工作变成无效试错。

### 3.8 第 7 步：修改实现并执行回归

当前仓库的开发顺序需要继续遵守：

1. 先在 `build-nvidia` 下验证测试逻辑或对标行为
2. 修改 `src/` 中目标实现
3. 在 `build` 下运行本地实现测试
4. 更新 coverage

benchmark 优化后，至少要回归：

- 目标 API 功能测试
- 目标 API 精度测试
- 相邻模块 smoke 测试
- 必要时 `build-nvidia` 下的对标测试

### 3.9 第 8 步：再次 benchmark 并生成阶段报告

优化完成后，至少要重新生成三类数据：

- `mpp_before.csv`
- `mpp_after.csv`
- `nvidia_ref.csv`

然后使用：

```bash
python3 benchmark/scripts/progress_report.py \
  mpp_before.csv \
  mpp_after.csv \
  nvidia_ref.csv \
  -o mpp_progress.csv \
  --html mpp_progress.html
```

这一步的目标不是“看一眼变快了没有”，而是回答：

- 本轮优化是否真的有效
- 哪些 API 已经接近 NVIDIA
- 哪些 API 仍需要下一轮迭代

### 3.10 第 9 步：归档产物并进入下一轮

一轮闭环结束后，建议至少保留：

- benchmark 命令
- 生成的 CSV / HTML
- 对应 build 模式
- 本轮优化假设
- 已改动源码位置
- 回归测试结果
- 下一轮计划

只有这样，后续 AI 或开发者才能在相同上下文上继续迭代，而不是重新猜测前一轮做了什么。

## 4. 生成 benchmark 的执行规范

在当前仓库里新增 benchmark 程序时应遵守的默认流程。

### 4.1 目标定义

目标不是输出一段建议文字，而是直接在 `benchmark/` 目录下生成并补完一个可用 benchmark 程序。
“可用”至少满足：

- 程序代码写到 `benchmark/nppi/`、`benchmark/npps/` 或 `benchmark/nppcore/`
- 能成功执行 `./benchmark/scripts/build_benchmark.sh`
- `./benchmark/scripts/run_benchmark.sh -f <API>` 能产出真实结果，而不是只返回 `success=false`
- 结果字段符合固定 schema：
  - `function_name`
  - `size`
  - `variant_tags`
  - `data_type`
  - `channels`
  - `avg_time_ms`
  - `throughput_gbps`
  - `success`
  - `error`
- 若符号缺失或当前后端不支持，不崩溃，返回 `success=false`

### 4.2 必须执行的步骤

新增 benchmark 时，默认按下面顺序执行：

1. 找到目标 API 所属 header、模块目录和最近邻 benchmark 文件。
2. 使用 `./benchmark/scripts/new_benchmark.sh` 生成骨架文件，而不是手工新建空文件。
3. 在生成文件内依据api定义补充真实 benchmark body：
   - 参数组织
   - pitch / step
   - ROI / 长度
   - buffer 分配
   - 真实 API 调用
   - 吞吐量计算
   - 失败兜底
4. 如目标 API 不适合默认尺寸，直接覆写生成文件内的 `benchmarkCases()`。
5. 运行 `./benchmark/scripts/build_benchmark.sh` 和 `./benchmark/scripts/run_benchmark.sh -f <filter>` 验证。
6. 如需对标，再运行 `--use-nvidia-npp` 路径验证结果格式与运行闭环。

确保编译运行benchmark程序能获得api真实性能数据

### 4.3 骨架生成命令

标准入口是：

```bash
./benchmark/scripts/new_benchmark.sh <type> <category> <header> \
  --functions f1,f2[,f3...] \
  [--variant-tags tag1,tag2]
```

示例：

```bash
./benchmark/scripts/new_benchmark.sh nppi addc nppi_arithmetic_and_logical_operations.h \
  --functions nppiAddC_8u_C1RSfs,nppiAddC_32f_C1R \
  --variant-tags baseline
```

生成后，依据api定义和功能把默认 `success=false` stub 替换成真实 benchmark 实现。

### 4.4 真实 body 的补完要求

在补完生成文件时，默认优先复用：

- 同模块已有 benchmark 文件
- 已有独立原型里验证过的 API 调用方式
- `benchmark_base.h` 中现成的计时、buffer、结果结构

至少需要补这些内容：

- `DynamicSymbol` 成功后，调用真实 API，而不是停留在占位逻辑
- 输入/输出 buffer 的分配与释放
- `step` / `pitch` 的正确计算
- `NppiSize roi` 或 signal length 的正确组织
- `runBatchedMeasurement(...)` 的真实调用闭环
- `processed_bytes / elapsed_seconds / 1e9` 的吞吐量计算
- `BenchmarkResult` 中各字段的完整赋值

### 4.5 尺寸与变体处理

新增 benchmark 时默认遵循：

- 默认先使用生成器给出的 `benchmarkCases()`
- 若 API 需要特殊尺寸、mask、插值或其它变体，直接在该 benchmark 文件内局部覆写
- 不为了单个 API 去改主框架脚本

变体统一放在 `variant_tags`，不要再混入 `size`。

### 4.6 完成标准

只有在以下条件都满足时，才应认为“benchmark 程序已生成完成”：

- `./benchmark/scripts/build_benchmark.sh` 通过
- 目标 API 在 `./benchmark/scripts/run_benchmark.sh -l` 或 `-f <API>` 下可见
- 运行后输出真实 CSV 行
- CSV 字段符合固定 schema
- 若需要对标，`--use-nvidia-npp` 路径也已验证

如果无法补完真实调用，应明确说明阻塞点，而不是把 stub 当成完成品。

## 5. 输入输出约定

为了让 benchmark 流程真正可复用，建议每一轮都按相同的输入输出约定执行。

| 阶段 | 主要输入 | 主要输出 |
|---|---|---|
| 目标定义 | API / 模块 / 对标对象 | benchmark 范围 |
| 场景固定 | filter / 固定运行常量 / build mode | 可复现实验条件 |
| 构建 | `scripts/build_benchmark.sh` 或 `build.sh --benchmark` | `npp_benchmark` 可执行文件 |
| 运行 | `scripts/run_benchmark.sh` | 原始 CSV / JSON |
| 对比 | `benchmark/scripts/compare_csv.py` | compare CSV / HTML |
| 优化 | `src/` 修改 | 新实现 |
| 回归 | `unit_tests`、必要时 `build-nvidia` | 测试结果 |
| 汇总 | `benchmark/scripts/progress_report.py` | before/after/reference 报告 |

## 6. 工具命令参考

### 5.1 构建

```bash
./benchmark/scripts/build_benchmark.sh
./benchmark/scripts/build_benchmark.sh --use-nvidia-npp
./build.sh --benchmark
./build.sh --benchmark --use-nvidia-npp
```

常用选项：

- `--debug`
- `--release`
- `--clean`
- `--rebuild`
- `--arch N`
- `-j N`

### 5.2 运行

```bash
./benchmark/scripts/run_benchmark.sh
./benchmark/scripts/run_benchmark.sh -f Resize -o resize_mpp
./benchmark/scripts/run_benchmark.sh --use-nvidia-npp -f Resize -o resize_nvidia
./benchmark/scripts/run_benchmark.sh -l
```

常用选项：

- `--use-nvidia-npp`
- `-f, --filter`
- `-o, --output`
- `--build`
- `--no-build`
- `--json`
- `--verbose`

### 5.3 对比与进展报告

```bash
python3 benchmark/scripts/compare_csv.py \
  resize_mpp.csv \
  resize_nvidia.csv \
  --labels "MPP" "NVIDIA NPP" \
  --html resize_compare.html

python3 benchmark/scripts/progress_report.py \
  mpp_before.csv \
  mpp_after.csv \
  nvidia_ref.csv \
  -o progress.csv \
  --html progress.html
```

## 7. 可扩展性设计

当前 benchmark 流程已经能用，但要形成长期工程能力，还需要把扩展入口定义清楚。

### 6.1 扩展一类新的 benchmark

优先路径：

1. 先判断是高价值稳定 API，还是仅用于覆盖率扩展。
2. 高价值稳定 API，优先放入 `benchmark/nppi/`、`benchmark/npps/` 或 `benchmark/nppcore/`，实现手写 benchmark。
3. 新增 benchmark 先使用 `scripts/new_benchmark.sh` 生成骨架，再在对应模块下补真实调用。

这样可以把复杂度控制在新增 API 本身，尽可能降低新增benchmark程序时的成本。

### 6.2 扩展新的 benchmark 场景

当前很多 benchmark 场景仍硬编码在各 benchmark 文件中。建议未来扩展遵守两条原则：

- 先抽到 `benchmark_base.h` 或公共 helper 中，形成共享场景集
- 再决定是否引入 JSON / YAML / CLI 驱动场景配置

优先统一“场景生成函数”，再引入“场景配置文件”，这样改动风险更低。

### 6.3 扩展新的对比维度

当前主指标是：

- `avg_time_ms`
- `throughput_gbps`
- `success`

后续若要扩展：

- 构建元数据
- toolchain 信息
- git commit
- benchmark binary 路径
- seed
- 场景标签

推荐先扩展 JSON 输出和 CSV 元数据列，不要先改 shell 脚本接口。

### 6.4 扩展新的后端

当前仓库已支持：

- in-tree MPP
- NVIDIA NPP

如果未来接入新后端，优先复用以下边界：

- `scripts/build_benchmark.sh` 中的构建目录与 toolchain 选择
- `benchmark/CMakeLists.txt` 中的 compile definition 和 link library 分支
- `scripts/run_benchmark.sh` 中的可执行文件发现逻辑
- `benchmark/scripts/compare_csv.py` / `benchmark/scripts/progress_report.py` 中的数据消费逻辑

不建议先改：

- 根 `build.sh`
- 根 `CMakeLists.txt`
- 非 benchmark 目录的构建配置

只有当新后端无法通过 benchmark 层现有开关表达时，才考虑扩大改动面。

### 6.5 扩展自动化流程

后续如果希望让 AI 自动跑一轮完整性能闭环，推荐的最小自动化粒度是：

1. build
2. run baseline
3. run reference
4. compare
5. 生成 progress report
6. 收集测试结果

不要一开始就把“源码修改 + benchmark + 回归 + 汇报”全部塞进一个超大脚本。应优先保持各阶段输入输出清晰。

## 8. 检查清单

### 7.1 benchmark 搭建检查

- [ ] 已确认使用独立 benchmark 构建还是主工程联动构建
- [ ] 已确认 benchmark 二进制来自哪个目录
- [ ] 已固定 filter 与程序内固定运行常量
- [ ] 已固定 benchmark 模式与输出格式
- [ ] 已明确 compare 与 progress 使用的输入文件

### 7.2 性能优化检查

- [ ] 已完成正确性与精度基线
- [ ] 已收集 MPP 基线
- [ ] 已收集 NVIDIA 参考数据
- [ ] 已明确热点排序依据
- [ ] 已写出可验证的优化假设
- [ ] 已执行目标 API 回归测试
- [ ] 已生成 before / after / reference 报告
- [ ] 已留下下一轮迭代输入

## 9. 常见问题与处理方式

### benchmark 结果不可比

优先检查：

- 是否使用了不同 build 目录下的可执行文件
- 固定运行常量是否一致
- filter 是否一致
- 是否混用了不同版本的 benchmark 可执行文件或不同输出口径

### `scripts/run_benchmark.sh` 找不到可执行文件

优先检查：

- 是否先执行了 `scripts/build_benchmark.sh` 或 `./build.sh --benchmark`
- 当前是 MPP 还是 NVIDIA 模式
- 是否误用了 `--no-build`

### compare 方向看反

当前仓库中：

- `mpp_time_pct_of_nvidia = (mpp_avg_time_ms / nvidia_avg_time_ms) * 100`

因此：

- `< 100%` 表示 MPP 更快
- `> 100%` 表示 NVIDIA 更快

### 优化后 benchmark 变快，但结果不可信

优先检查：

- 是否漏掉功能或精度回归
- 是否改变了 benchmark 场景
- 是否把不同来源的 benchmark 二进制混在一起比较
- 是否只观察了局部少量样本

## 9. 完成标准

一次完整的 benchmark 搭建与性能优化任务，至少应满足：

- benchmark 构建与运行路径明确
- benchmark 场景可复现
- 基线与参考数据可追溯
- 优化假设明确且已验证
- 功能回归结果明确
- 已生成 CSV、HTML 或阶段报告
- 下一轮扩展或优化入口清晰

## 10. 关键提醒

1. benchmark 的价值不在“跑出一个数字”，而在于把性能工作纳入固定闭环。
2. 当前仓库已经有稳定的 build/run/compare/report 分层，优先复用，不要临时发明新流程。
3. 对比与阶段汇总优先留在 Python 层，避免让 `scripts/run_benchmark.sh` 承担越来越多非运行职责。
4. 未来若扩 benchmark，优先改 `benchmark/` 与 `docs/`，尽量不要波及主工程配置。
5. benchmark 只是入口，真正的性能收益必须和 `src/` 修改、测试回归、阶段报告一起成立。
