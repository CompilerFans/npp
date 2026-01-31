# Repository Guidelines

### 执行规则
- 不向用户提问；信息不足时使用最合理假设继续，并在结果中写明假设。
- 除非会造成不可逆 / 高风险 / 高成本，否则不允许停下来确认。
- 先给出 3-6 步简短计划，然后直接执行并给出最终产物（代码 / 命令 / 文件清单）。
- 若有多方案，默认选择成功率最高的方案，并附备选方案。


## 项目结构与模块组织
- `src/`: 核心库实现，按 NPP 模块组织（`nppcore/`、`nppi/`、`npps/`）。
- `test/`: 基于 GoogleTest 的单元测试，模块级测试见 `test/unit/`。
- `examples/`: 示例程序与用法演示。
- `docs/`、`status/`、`api_analysis/`: 文档与分析记录。
- `third_party/`: 外部依赖（如 GoogleTest）。

## 构建、测试与开发命令
- `./build.sh`: 主要构建入口（Ninja + CMake），默认 Release，包含测试与示例。
- `./build.sh -d`: Debug 构建。
- `./build.sh --no-tests` / `--no-examples` / `--lib-only`: 精简构建目标。
- `./build.sh --use-nvidia-npp`: 测试时链接 NVIDIA NPP 而非 MPP。
- `ctest --output-on-failure`: 在已配置的构建目录中运行测试。
- `./test/unit/unit_tests --gtest_filter="AddTest.*"`: 运行指定测试子集。

## 代码风格与命名规范
- 格式化：LLVM 风格，2 空格缩进，120 列限制；运行 `./format.sh`（使用 `clang-format`）。
- C/C++/CUDA 代码位于 `src/` 和 `test/`。
- 测试文件：`test_<function_name>.cpp`（详见 `test/README.md`）。
- 测试类：`<function_name>Test`；测试用例：`<function_name>_<data_type>_<scenario>`。

## 测试指南
- 框架：GoogleTest（见 `test/README.md`）。
- 测试关注功能正确性，默认不做性能基准。
- 新测试放在对应模块目录（`test/unit/nppcore/`、`test/unit/nppi/`）。

## 提交与 PR 规范
- 提交信息简短、动词开头、可描述内容（例如 “Optimize nppiCopy_32f_C3P3R performance”）。
- PR 应包含：简要说明、影响模块、测试结果（命令 + 结果）。
- 性能相关改动需注明性能影响或测试说明。

## 配置说明
- CUDA 架构在 `build.sh` 中通过 `-DCMAKE_CUDA_ARCHITECTURES="89"` 指定；如需适配其他 GPU 请调整。

## 开发流程与执行规则
### 开发流程（按顺序执行）
1. 先写测试用例，再写实现代码。
2. 先在 `build-nvidia` 目录开发并运行测试，确保测试通过；测试需包含功能测试与精度测试，优先使用 gtest 参数化覆盖不同数据量、数据类型。需要执行build-nvidia下的unit_tests，使用nvidia的npp库，验证测试代码正确性
2.1 如果遇到执行nvidia下的unit_tests错误，需要优先修正，并且不能因为错误影响其他测试
3. 再开发 `src/` 函数实现；实现需考虑复用性，抽象可复用逻辑。
4. 在 `build` 目录运行测试，确保测试通过。
5. 执行 `./update_coverage.sh` 更新覆盖率数据。
6. 完成后提交 `git commit` 并 `push` 到远程仓库。

## 开发计划
- 扫描并提取npp/api_analysis/coverage.csv中，nppi_color_conversion.h缺失的api，总结至npp/api_analysis目录，再逐步实现
- 基于nppi_color_conversion.h中在实际中的实际使用热度，制定开发计划
- 开发要考虑测试和源码的复用性