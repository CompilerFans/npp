# <包编号> — <模块 / 家族>

你是 mcMPP 的 NPP API 实现子 agent。请在当前 worktree 内完成本包，全部工作以 NVIDIA NPP 13.4（5090）为功能基线。

## 0. 必读

1. `docs/npp_api_expansion_plan.md`
2. `docs/mpp_api_implementation_guide.md`
3. `mpp_scripts/families/README.md`
4. `api_analysis/packages/<包编号>.csv`
5. 最近邻实现（至少 2 个同模块已实现家族）与 `test/unit/framework/npp_diff_harness.h`

## 1. 范围

| 项 | 值 |
|---|---|
| 模块 | `<header>` |
| 家族 | `...` |
| 待实现 | `<N>`（见 package CSV） |

## 2. 交付物

### 2.1 Executor / kernel

- 新建或复用 `src/<模块目录>/<family>_executor.h`。
- 模板/策略复用现有 `nppi_arithmetic_executor.h` 风格。
- 参数校验、ROI/长度、step、状态码与 NPP 13.4 对齐（以差分结果为准）。
- launch 前清空陈旧 CUDA error；使用 `nppStreamCtx.hStream`。

### 2.2 生成 wrapper

```bash
python3 mpp_scripts/gen_family.py --spec mpp_scripts/families/<spec>.json --list
python3 mpp_scripts/gen_family.py --spec mpp_scripts/families/<spec>.json --emit
```

生成文件入库，不手改。

### 2.3 差分测试

- `test/unit/diff/test_<family>_generated.cpp`。
- 覆盖所有代表性签名形状；每个函数至少被一个 case 覆盖。
- 覆盖正常、边界、错误路径、原地、步长/ROI。
- 5090 record 通过 → 本地 verify 通过。

### 2.4 覆盖率

```bash
./update_coverage.sh
python3 tools/make_package_list.py --package <包编号>
```

## 3. 验证

```bash
./build.sh
NPP_VERIFY_DIR=artifacts/5090/npp_records LD_LIBRARY_PATH=build/src \
  ./build/diff_tests --gtest_filter='<Suite>*'
tools/remote5090.sh build
tools/remote5090.sh record '<Suite>*'
tools/remote5090.sh pull
LD_LIBRARY_PATH=build/src ./build/unit_tests
```

## 4. Definition of Done

1. package CSV 中所有待实现函数变为 `real`。
2. 5090 record + 本地 verify 全绿。
3. 本地全量基线（当前 4,803）无新增失败。
4. 覆盖率已刷新。
5. 已提交到当前包分支；报告包含实现/跳过清单、验证结果、未覆盖项。

## 5. 约束

- 不 push、不 force push、不改集成分支、不删除既有测试。
- 正确性优先，不做性能优化。
- 只改本包文件；公共组件改动需在报告中说明。
- 生成文件不手改。

## 6. 建议顺序

1. 确认签名/语义/状态码。
2. executor + 一个最小 dtype 的差分 case。
3. 扩全 dtype/变体 + spec 生成。
4. 测试矩阵 record/verify。
5. 全量回归 + 覆盖率。
