# NPP API 实现与修复指南

本文档用于指导当前 NPP 工程中的单个 API 实现、修复、测试对齐与覆盖率更新工作。

## 1. 关键约束

### 1.1 严禁事项

- **禁止先写实现后补测试**：测试必须先于实现存在，或者至少先被修到可信。
- **禁止跳过 `build-nvidia` 验证**：只要目标行为需要对齐 NVIDIA NPP，就必须先在 `build-nvidia` 验证测试本身成立。
- **禁止只测正常路径**：必须覆盖错误路径、ROI 边界、step / pitch 等关键行为。
- **禁止把 `_Ctx` 与非 `_Ctx` 路径随意混写**：除非当前模块已有成熟统一模式。
- **禁止在正确性未稳定时做性能优化**：单个 API 的第一目标是功能对齐，不是速度。

### 1.2 允许并推荐的做法

- **允许复用近邻实现**：优先复用 `src/` 中相邻 API 的 helper、dispatch、边界处理模式。
- **允许复用近邻测试**：优先复用 `test/unit/` 中现有 fixture、参数化模板和辅助函数。
- **允许以 NVIDIA NPP 为功能基线**：除非仓库文档已明确声明故意偏离。
- **允许先修测试再修实现**：当已有测试错误或不完整时，应先修测试。
- **允许先用覆盖率文件定位落点**：单个 API 任务开始前，可先借助 `api_analysis/coverage.csv` 和 `tools/check_coverage.py` 找到实现文件与测试文件。
- **允许用户提及不在几个API定义中的api时不进行实现**：用户提及不在几个API目录中定义的api时，告知用户nvidia npp库中不存在该api，不用进行实现。

## 2. 工作区结构

围绕单个 API 工作时，重点关注以下目录：

```text
.
├── API/ API-11.4/ API-12.2/ API-12.8/   # 多版本 NPP 头文件参考
├── api_analysis/                        # coverage.csv、api_functions_12.2.csv 等分析产物
├── src/nppcore/                         # nppcore 实现
├── src/nppi/                            # nppi 各子模块实现
├── src/npps/                            # npps 各子模块实现
├── test/unit/framework/                 # 公共测试基础设施
├── test/unit/nppcore/                   # nppcore 测试
├── test/unit/nppi/                      # nppi 各子模块测试
├── test/unit/npps/                      # npps 各子模块测试
├── build/                               # 默认 MPP 构建目录
├── build-nvidia/                        # NVIDIA NPP 参考验证目录
└── docs/ ref_code/                      # 专项分析、CPU 参考实现、行为说明
```

### 目录作用说明

- **`API/`、`API-12.2/`、`API-12.8/`**：查看目标 API 的声明、参数含义和不同版本函数签名。当前仓库很多分析脚本默认围绕 12.2 数据展开，因此 `api_analysis/api_functions_12.2.csv` 常作为检索入口。
- **`src/nppi/` 与 `src/npps/`**：实现按模块拆分，例如 `src/nppi/nppi_color_conversion_operations/`、`src/nppi/nppi_geometry_transforms/`、`src/npps/npps_arithmetic_operations/`。
- **`test/unit/framework/`**：公共 fixture 和测试基础设施所在目录。新增测试前应先检查这里是否已有通用基类和辅助工具。
- **`test/unit/nppi/` 与 `test/unit/npps/`**：测试目录与 `src/` 模块基本一一对应，新增测试应尽量放到现有子目录。
- **`api_analysis/coverage.csv`**：确认当前 API 是未实现、已实现未测试，还是已实现已测试。
- **`docs/` 与 `ref_code/`**：当边界模式、CPU 参考实现或 NVIDIA 行为分析已有专项文档时，应先消费这些资料，再动手写测试。当实现新模块api时，功能分析总结到`docs/`下存档。

## 3. 统一工作流

### 第 1 步：确认目标 API

先确定以下信息：

- 精确函数签名
- 数据类型
- 通道布局
- ROI 规则
- step / pitch 规则
- 错误码语义

优先查看：

- `API/`、`API-12.2/`、`API-12.8/`
- `api_analysis/api_functions_12.2.csv`
- NVIDIA NPP 头文件声明

如果不同版本头文件存在差异，应在任务记录中明确本次采用的参考来源，避免测试和实现混用不同版本的签名。

### 第 2 步：找到最近邻实现和测试

在改代码前，先找：

- `src/nppi/...` 或 `src/npps/...` 中最相近的实现
- `test/unit/nppi/...` 或 `test/unit/npps/...` 中最相近的测试

目标是复用，而不是重写一套新结构。当前工程里常见的匹配关系例如：

- 颜色转换：`src/nppi/nppi_color_conversion_operations/` 对应 `test/unit/nppi/nppi_color_conversion_operations/`
- 几何变换：`src/nppi/nppi_geometry_transforms/` 对应 `test/unit/nppi/nppi_geometry_transforms/`
- 阈值与比较：`src/nppi/nppi_threshold_and_compare_operations/` 对应 `test/unit/nppi/nppi_threshold_and_compare_operations/`

### 第 3 步：先写测试或修测试

测试至少覆盖：

- 正常输入
- 非法指针
- 非法 step / pitch
- ROI 边界行为
- 精度敏感或饱和行为
- 同一函数族的通道 / 布局变体

优先使用参数化 gtest，并尽量复用 `test/unit/framework/` 和同模块已有 fixture。

### 第 4 步：在 `build-nvidia` 验证测试

先证明测试程序是对的，可以达到测试功能的目的，再信任测试失败。运行单个API测试`TestAPI`执行命令：

```bash
./build.sh --use-nvidia-npp
cd build-nvidia
ctest --output-on-failure
./unit_tests --gtest_filter="*TestAPI*"
```

`build.sh --use-nvidia-npp` 会把构建目录切到 `build-nvidia/`，其意义是使用同一套测试代码去链接 NVIDIA NPP 作为参考实现。

若失败，优先排查：

1. 测试是否写错
2. 预期是否理解错
3. NVIDIA NPP 是否与当前假设不同
4. 测试框架或 fixture 是否有问题

### 第 5 步：修改本地实现

只有测试通过 NVIDIA 路径验证后，才修改 `src/`。推荐实现顺序：

1. 参数校验
2. 类型 / 通道 / 布局分发
3. 核心计算逻辑
4. 公共 helper 抽取
5. 错误码与边界行为修正

当前仓库目录已经按模块拆得较细，新增文件应优先落到现有子目录，而不是再创建平行命名体系, 命名沿用仓库中已有的命名方式。

### 第 6 步：在 `build` 做本地回归

实现完成后，运行同一组聚焦测试：

```bash
./build.sh
cd build
ctest --output-on-failure
./unit_tests --gtest_filter="*TestAPI*"
```

目标是确认本地实现是否与 NVIDIA NPP 行为对齐。

如需缩小编译范围，可先用 `./build.sh --lib-only` 快速验证库构建；但只要本次任务涉及测试变更，最终仍应回到标准测试路径确认。

### 第 7 步：刷新覆盖率

如果实现状态或测试状态发生变化，执行：

```bash
./update_coverage.sh
```

并确认对应 API 的覆盖率条目变化是否符合预期。

当前 `update_coverage.sh` 除了刷新 `api_analysis/coverage.csv` 和 `api_analysis/coverage_summary.md`，还会为算术模块生成 `api_analysis/untested_arithmetic.txt` 与 `api_analysis/untested_arithmetic_summary.md`。若本次工作位于 arithmetic 模块，则顺带检查这两个文件。

## 4. 工具命令参考

### 4.1 NVIDIA 基线路径

```bash
./build.sh --use-nvidia-npp
cd build-nvidia
ctest --output-on-failure
./unit_tests --gtest_filter="*TestAPI*"
```

### 4.2 本地实现路径

```bash
./build.sh
cd build
ctest --output-on-failure
./unit_tests --gtest_filter="*TestAPI*"
```

### 4.3 覆盖率刷新

```bash
./update_coverage.sh
```

### 4.4 模块状态快速扫描

```bash
python3 tools/check_coverage.py
```

这个脚本会直接输出 `api_analysis/coverage.csv` 和 `api_analysis/coverage_summary.md`，用于确定模块中api的覆盖情况。

## 5. 实现检查清单

### 功能检查

- [ ] 已确认目标 API 的精确签名
- [ ] 已确认使用的是哪一套头文件版本作为参考
- [ ] 已确认 ROI / step / 错误码语义
- [ ] 已找到最近邻实现
- [ ] 已找到最近邻测试
- [ ] 已检查 `test/unit/framework/` 中是否已有可复用基础设施
- [ ] 已在 `build-nvidia` 下验证测试可信
- [ ] 已在 `build` 下验证本地实现通过
- [ ] 已确认是否需要更新覆盖率

### 行为对齐检查

重点比对以下维度：

- [ ] `NppStatus`
- [ ] 输出值
- [ ] alpha 通道保留规则
- [ ] 是否支持原地操作
- [ ] 饱和 / 舍入 / 截断规则
- [ ] ROI 尺寸是否合法
- [ ] buffer-size helper 行为

如果目标 API 所属模块在仓库内已有 CPU 参考实现、专项行为分析或边界模式文档，也应在这里补做人工核对。

## 6. 常见问题与处理方式

### 测试在 `build-nvidia` 失败

优先判断：

- 测试本身错误
- 预期理解错误
- NVIDIA NPP 行为与假设不一致
- fixture 或输入构造不合理

### 本地实现在 `build` 失败

优先判断：

- 参数校验遗漏
- 通道 / 类型 dispatch 错误
- ROI 或 step 处理错误
- 边界行为未与 NVIDIA NPP 对齐
- 模块内已有 helper 复用方式不一致

### 测试通过但行为仍可疑

优先补充：

- 非法参数测试
- 边界 ROI 测试
- 饱和 / 精度敏感测试
- 函数族变体测试

## 7. 完成标准

一个 API 实现或修复完成，至少应满足：

- 测试已在 `build-nvidia` 通过
- 本地实现已在 `build` 通过同一组测试
- 覆盖率状态已刷新（若本次任务影响覆盖率）
- 结果中明确写出对齐了哪些 NVIDIA NPP 行为
- 明确写出当前仍未覆盖或暂未处理的范围

## 8. 关键提醒

1. 单个 API 的任务首先是“行为对齐”，不是“快速写出一版实现”。
2. 越接近已有实现族，越应该复用现有模式。
3. 只看正常路径的测试结论没有意义。
4. 编译`build-nvidia` 进行测试验证是判断测试可信度的关键步骤。
5. 只要状态变化就应更新覆盖率。
6. 当前仓库模块目录和测试目录已经较完整，新增文件优先贴合现有框架，而不是另起体系。
