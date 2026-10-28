# NPP API 全量实现进度账本

> 最后更新：2026-10-28
> 集成分支：`feat/npp-api-expansion`（基于 `dev` @ `3a81f00`）
> 公共契约：`API/`（API-unified）
> 参考实现：NVIDIA NPP 13.4 / RTX 5090 / CUDA 13.4

## 1. 基线

| 指标 | 数值 |
|---|---:|
| 公共函数 | 12,537 |
| 真实实现 | 3,336 |
| 显式 stub | 166 |
| 缺失 | 9,035 |
| 待实现（stub+missing） | 9,201 |
| 待实现 `_Ctx` | 4,655 |
| 本地回归基线 | 4,803 tests, 0 failures |

## 2. 基础设施（已完成）

| 提交 | 内容 |
|---|---|
| `5fcabf4` | `tools/npp_api_catalog.py`：按 `API/` 生成 catalog/backlog，区分 real/stub/missing |
| `a841455` | `npp_diff_harness.h` + `diff_tests` + 5090 record/verify runner；nppsSum pilot 通过 |
| `40b96ec` | `mpp_scripts/gen_family.py` + family spec 机制；mixed dtype 解析修复 |
| `88ff4ef` | 工作包清单 + 子 agent runner（`tools/agents/run_package.sh`） |

验证矩阵：

- 本地：`source ~/.config/mcpp-build.env`（全新 worktree 需要 `CUDAToolkit_ROOT`）→ `./build.sh` → `LD_LIBRARY_PATH=build/src` → `./build/unit_tests` / `./build/diff_tests`
- 5090：`tools/remote5090.sh build|record|pull`，`metax@10.2.120.6`
- 差分：5090 record（NVIDIA NPP 13.4）→ 本地 verify（MACA）

## 3. 工作包状态

| 包 | 内容 | 待实现 | 状态 | 分支/提交 |
|---|---|---:|---|---|
| P1.1a | NPPS 二元算术 Add/Sub/Mul/Div/AddProduct/Div_Round | 220 | 运行中 | `pkg/p1.1a` |
| P1.1b | NPPS 常量/归一化 AddC/SubC/.../Normalize | 256 | 待启动 | — |
| P1.1c | NPPS 一元/位运算/移位 | 258 | 待启动 | — |
| P1.2a | NPPS 初始化 Set/Copy/Zero | 60 | 待启动 | — |
| P1.2b | NPPS 转换/阈值 Convert/Threshold | 174 | 待启动 | — |
| P1.3a | NPPI/NPPS 统计归约（Sum/Min/Max/Mean） | 待拆 | 规划中 | — |
| P1.3b | Norm/NormDiff/NormRel/DotProd | 待拆 | 规划中 | — |
| P1.3c | 误差度量 Average/Maximum(+Relative) | 待拆 | 规划中 | — |
| P1.4 | NPPI 数据交换 Convert/Copy/Set/SwapChannels/Scale | 861 | 规划中 | — |
| P1.5 | NPPI 阈值 | 366 | 规划中 | — |
| P1.6 | NPPI 形态学 | 268 | 规划中 | — |
| P1.7 | NPPI 几何缺口 | 317 | 规划中 | — |
| P1.8 | NPPI 统计 CrossCorr/长尾 | 待拆 | 规划中 | — |
| P1.9 | NPPI 滤波 | 1,802 | 规划中 | — |
| P1.10 | NPPI 颜色（待 238733 合入后） | 646 | 阻塞 | — |

## 4. 每个包的合并检查

1. `git diff --stat feat/npp-api-expansion..pkg/<id>` 只包含本包文件。
2. 本地 `./build.sh` + `LD_LIBRARY_PATH=build/src ./build/unit_tests` 全绿（基线 4,803）。
3. `build/diff_tests` 在 `NPP_VERIFY_DIR=artifacts/5090/npp_records` 下全绿。
4. `python3 tools/npp_api_catalog.py --update` 后包内 remaining=0。
5. `git merge --no-ff pkg/<id>` 到集成分支；更新本账本。

## 5. 决策记录

- 基线使用 `dev`，不包含待合并的 238733；P1.10（颜色）必须等 238733 合入或显式 rebase 后启动。
- 参考基线是 5090 上实际安装的 NPP 13.4，不照搬文档；状态按名称比较，buffer-size 精确值不作为契约。
- 非 `_Ctx` 入口由生成器统一 `nppGetStreamContext` 转发；NPP 13 构建只验证 `_Ctx`。
- 生成器只负责 wrapper；kernel/executor/测试手写，spec 入库、生成物入库。
- 不 push Gerrit；集成分支与包分支都只在本地，待用户要求再拆分提交。

## 6. 已知风险

- `nppsDiv_Round` / 整数缩放的舍入语义需以 5090 差分结果为准。
- MACA 运行时会在极小 `cudaMalloc` 后残留 `invalid argument`；新 executor 必须在 launch 前清空错误。
- `nppi_not_support.cpp` 的 166 个 stub 需要在对应 nppi 包中删除。
- 复杂算法家族（CrossCorr、Canny、DistanceTransform）需要单独评估工期。
