# NPP 开发与优化总流程指南

本文档参考 `/local/m01019/mcmpp_perf/SKILL.md` 的组织方式编写，用于说明当前 NPP 工程中从覆盖率规划、测试验证、功能实现，到 MACA 适配和性能优化的一整套执行框架。文中的目录、构建路径、benchmark 脚本和分析工具均对齐当前仓库现状。

## 1. 关键约束

### 1.1 严禁事项

- **禁止脱离仓库工作流随意开发**：必须按规划、测试、实现、回归、覆盖率、移植、优化的顺序执行。
- **禁止先实现后写测试**：所有功能开发默认测试优先。
- **禁止跳过 `build-nvidia` 验证**：需要对齐 NVIDIA NPP 的行为，必须先在 `build-nvidia` 验证测试语义。
- **禁止在正确性闭环前做性能优化**：性能工作永远是第二阶段。
- **禁止把多个无关热点混在同一轮调优**：否则无法判断收益来源。
- **禁止不更新覆盖率状态就宣称任务完成**：只要实现或测试状态变化，就应刷新。

### 1.2 允许并推荐的做法

- **允许从 `coverage.csv` 出发规划任务**：推荐按模块和批次组织工作。
- **允许优先处理可复用的一批 API**：相邻 API 能共享测试或 helper 时，应批量推进。
- **允许先做 baseline，再做 MACA porting**：baseline 稳定是移植的前提。
- **允许在功能稳定后再进入 benchmark 闭环**：benchmark 和优化应建立在稳定实现上。
- **允许优先复用仓库脚本链路**：优先使用 `build.sh`、`update_coverage.sh`、`benchmark_base/*.sh` 与分析脚本，而不是临时拼接新流程。

## 2. 工作区结构

围绕开发与优化总流程，重点关注以下目录：

```text
.
├── API/ API-11.4/ API-12.2/ API-12.8/  # 多版本 API 头文件参考
├── api_analysis/                       # coverage.csv、coverage_summary、热度分析
├── src/nppcore/                        # nppcore 实现
├── src/nppi/                           # nppi 模块实现
├── src/npps/                           # npps 模块实现
├── test/unit/framework/                # 测试公共基础设施
├── test/unit/nppcore/                  # nppcore 测试
├── test/unit/nppi/                     # nppi 模块测试
├── test/unit/npps/                     # npps 模块测试
├── build/                              # 默认 MPP 构建目录
├── build-nvidia/                       # NVIDIA NPP 参考验证目录
├── build-maca/                         # MACA NPP 验证目录（按需生成）
├── benchmark_base/                     # benchmark 框架、脚本、对比工具
└── docs/ ref_code/                     # 项目文档、专项分析、CPU 参考实现
```

### 目录作用说明

- **`api_analysis/`**：当前规划工作的核心入口，包含 `coverage.csv`、`coverage_summary.md`、`nppi_color_conversion_usage_heat.md` 等文件。
- **`src/nppi/` 与 `src/npps/`**：实现已经按功能头文件拆成多个子模块，例如 `nppi_filtering_functions`、`nppi_geometry_transforms`、`npps_statistics_functions`。
- **`test/unit/framework/`**：公共测试基座。新增测试前先看这里，通常比复制相邻测试文件更稳妥。
- **`build/`**：`./build.sh` 默认输出目录，对应本仓库 MPP 实现。
- **`build-nvidia/`**：`./build.sh --use-nvidia-npp` 输出目录，用来验证测试语义和行为基线。
- **`build-maca/`**：`./build.sh --use-maca-npp` 或相关 MACA 构建路径的输出目录。
- **`benchmark_base/`**：不只是脚本目录，还包含 benchmark 主程序、模板、自动 benchmark 文件和对比工具。

## 3. 统一工作流

### 第 1 步：确定任务范围

先明确本轮工作属于哪类：

- 缺失实现
- 缺失测试
- 行为修复
- 模块规划
- MACA 适配
- 性能优化

然后收集必要上下文：

- `api_analysis/coverage.csv`
- `api_analysis/coverage_summary.md`
- 相关头文件
- 相关 `src/`
- 相关 `test/unit/`
- 相关 `docs/`

若目标模块是 `nppi_color_conversion.h` 一类已有热度分析的模块，还应额外收集 `api_analysis/nppi_color_conversion_usage_heat.md`。

### 第 2 步：按模块规划

如果任务来自覆盖率或缺口扫描，应优先按模块推进，而不是随机挑单个 API。

建议先区分：

- `implemented=no`
- `implemented=yes,tested=no`
- `implemented=yes,tested=yes`

排序时优先考虑：

1. 使用热度
2. 复用价值
3. 模块连续开发价值
4. NVIDIA 对齐与移植风险

当前仓库已经提供 `skills/npp-development-optimization/scripts/summarize_coverage.py`，建议优先用脚本生成候选清单，而不是手动 grep 整个 `coverage.csv`。

### 第 3 步：测试先行

进入具体开发前，先写或修测试。测试应覆盖：

- 正常输入
- 多尺寸
- 多数据类型
- 多通道布局
- 非法参数
- 精度敏感场景
- 边界条件

优先参数化 gtest，尽量复用已有 fixture 和 `test/unit/framework/`。

### 第 4 步：在 `build-nvidia` 验证测试

推荐命令：

```bash
./build.sh --use-nvidia-npp
cd build-nvidia
ctest --output-on-failure
./test/unit/unit_tests --gtest_filter="*ApiOrModule*"
```

这里的目标不是验证本地实现，而是验证测试是否可信。

`build.sh` 当前支持 `--use-nvidia-npp`、`--use-maca-npp`、`--benchmark`、`--lib-only`、`--no-werror` 等选项。标准开发流程里最常用的是默认构建与 `--use-nvidia-npp` 两个入口。

### 第 5 步：修改 `src/` 实现

测试被验证后，再改本地实现。建议顺序：

1. 参数校验
2. 类型 / 通道 / 布局分发
3. 核心实现
4. helper 抽取
5. 错误码与边界行为修正

### 第 6 步：在 `build` 做回归

推荐命令：

```bash
./build.sh
cd build
ctest --output-on-failure
./test/unit/unit_tests --gtest_filter="*ApiOrModule*"
```

目标是确认新测试通过，且相邻模块没有被带坏。

如需先快速确认库是否能编过，可用 `./build.sh --lib-only`；但只要本次任务涉及测试或行为修复，最终仍应回到标准测试路径。

### 第 7 步：刷新覆盖率

只要实现状态或测试状态变化，就执行：

```bash
./update_coverage.sh
```

并确认覆盖率结果变化正确。

当前 `update_coverage.sh` 还会为算术模块生成 `api_analysis/untested_arithmetic.txt` 与 `api_analysis/untested_arithmetic_summary.md`。因此 arithmetic 模块变更后，建议同步检查这两个文件。

### 第 8 步：做 MACA 适配

只有 baseline 稳定后，才进入 MACA 阶段。重点关注：

- 工具链差异
- 条件编译差异
- 编译错误
- 运行时行为差异

仓库中既有 `build.sh --use-maca-npp`，也有单独的 `build_maca.sh`。除非任务明确指定其他方式，优先沿用主构建脚本以减少配置漂移。

### 第 9 步：做 benchmark 和性能优化

性能路径必须建立在正确性稳定之上。推荐闭环：

1. 固定 benchmark 场景
2. 采集 MPP 基线
3. 采集 NVIDIA NPP 对标数据
4. 排序热点
5. 提出优化假设
6. 小步修改
7. 重新跑功能测试
8. 重新跑 benchmark
9. 记录结果并进入下一轮

当前仓库里的 benchmark 能力主要集中在 `benchmark_base/`，其中已经有：

- `build_benchmark.sh`：独立构建 benchmark
- `run_benchmark.sh`：运行 benchmark、过滤 API、做 compare
- `compare_csv.py`：生成对比 CSV 与 HTML 报告
- `progress_report.py`：跟踪 MPP before/after 与 NVIDIA 目标之间的进展

## 4. 工具命令参考

### 4.1 覆盖率缺口扫描

```bash
python3 skills/npp-development-optimization/scripts/summarize_coverage.py \
  --module nppi_color_conversion.h \
  --tested no \
  --heat api_analysis/nppi_color_conversion_usage_heat.md \
  --limit 30
```

如果想直接看到实现文件和测试文件，可追加 `--show-files`。

### 4.2 NVIDIA 路径测试

```bash
./build.sh --use-nvidia-npp
cd build-nvidia
ctest --output-on-failure
./test/unit/unit_tests --gtest_filter="*ApiOrModule*"
```

### 4.3 本地实现测试

```bash
./build.sh
cd build
ctest --output-on-failure
./test/unit/unit_tests --gtest_filter="*ApiOrModule*"
```

### 4.4 覆盖率刷新

```bash
./update_coverage.sh
```

### 4.5 benchmark 构建与运行

```bash
./benchmark_base/build_benchmark.sh --use-mpp
./benchmark_base/build_benchmark.sh --use-nvidia-npp

./benchmark_base/run_benchmark.sh --use-mpp -f Resize
./benchmark_base/run_benchmark.sh --use-nvidia-npp -f Resize
```

### 4.6 benchmark 对比与报告

```bash
python3 benchmark_base/compare_csv.py \
  benchmark_compare_mpp.csv \
  benchmark_compare_nvidia.csv \
  --labels MPP "NVIDIA NPP" \
  --html benchmark_compare_report.html
```

如果是同一个 API 的 MPP 优化前后对比，也可以把两份 MPP CSV 直接喂给 `compare_csv.py`，只是输出列名仍保持 `mpp_*` / `nvidia_*` 的兼容格式。

## 5. 工作检查清单

### 功能开发检查

- [ ] 已明确任务属于哪一类
- [ ] 已完成模块或 API 范围界定
- [ ] 已确认参考的头文件版本和模块目录
- [ ] 已先写或修测试
- [ ] 已检查 `test/unit/framework/` 是否已有可复用基础设施
- [ ] 已在 `build-nvidia` 验证测试可信
- [ ] 已在 `build` 验证本地实现
- [ ] 已刷新覆盖率状态

### 性能优化检查

- [ ] 正确性闭环已稳定
- [ ] benchmark 场景已固定
- [ ] 已采集 MPP 和 NVIDIA NPP 数据
- [ ] 已明确 benchmark 使用的是 `build/`、`build-nvidia/` 还是 `build-bench/` 路径产物
- [ ] 已明确优化假设
- [ ] 已做回归验证
- [ ] 已记录本轮收益与风险

## 6. 常见问题与处理方式

### 任务开始就不知道先做什么

先看 `coverage.csv`、目标头文件、附近 `src/`、附近 `test/unit/`，先确定模块范围，再开始。

更稳妥的做法是先跑一遍 `summarize_coverage.py`，再决定本轮是做“实现缺口”还是“测试缺口”。

### 测试在 `build-nvidia` 就失败

优先怀疑测试本身，不要立即改 `src/`。先判断测试逻辑、预期理解和 fixture。

如果失败只发生在 `build-nvidia`，通常优先排查测试和输入构造；如果 `build-nvidia` 通过而 `build` 失败，再回头看本地实现。

### 本地实现通过单测但覆盖率没变化

优先检查：

- 是否忘了运行 `./update_coverage.sh`
- 覆盖率条目是否对应错模块
- 当前任务是否实际上只补了局部逻辑，没有改变统计口径
- 当前模块是否属于 `update_coverage.sh` 额外摘要覆盖范围之外

### 性能优化没有明显收益

优先检查：

- benchmark 场景是否固定
- 热点是否选对
- 修改是否过于分散
- 是否遗漏了功能回归导致结果不可信
- 是否误用了不同 build 目录下的 benchmark 可执行文件

## 7. 完成标准

一个完整的开发或优化任务完成，至少应满足：

- 测试在 `build-nvidia` 可验证
- 本地实现在 `build` 通过回归
- 覆盖率状态已刷新
- 若涉及移植，MACA 状态明确
- 若涉及性能优化，benchmark 收益和回归结果明确
- 最终结果说明清楚本轮完成范围和剩余范围

## 8. 关键提醒

1. 覆盖率、测试、实现、移植、优化不是并列关系，而是有先后依赖的链路。
2. `build-nvidia` 的作用是验证测试，不是替代本地回归。
3. baseline 稳定前不要急着做 MACA 和性能优化。
4. benchmark 必须建立在同一场景上，否则结论不成立。
5. 每轮工作都应该有可追溯结果，而不是只留下口头结论。
6. 当前仓库已经有比较完整的脚本链路，优先复用已有脚本和目录结构，不要临时发明新流程。
