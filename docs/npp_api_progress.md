# NPP API 全量实现进度账本

> 最后更新：2026-10-28
> 集成分支：`feat/npp-api-expansion`（基于 `dev` @ `3a81f00`）
> 公共契约：`API/`（API-unified）
> 参考实现：NVIDIA NPP 13.4 / RTX 5090 / CUDA 13.4

## 1. 基线

| 指标 | 数值 |
|---|---:|
| 公共函数 | 12,537 |
| 真实实现 | 4,304 |
| 显式 stub | 166 |
| 缺失 | 8,067 |
| 待实现（stub+missing） | 8,233 |
| 待实现 `_Ctx` | 4,171 |
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
| P1.1a | NPPS 二元算术 Add/Sub/Mul/Div/AddProduct/Div_Round | 220 | ✅ 已合并 | `pkg/p1.1a` → `388785f` |
| P1.1b | NPPS 常量/归一化 AddC/SubC/.../Normalize | 256 | ✅ 已合并 | `pkg/p1.1b` → `c140859` |
| P1.1c | NPPS 一元/位运算/移位 | 258 | ✅ 已合并 | `pkg/p1.1c` → `a9757d5` |
| P1.2a | NPPS 初始化 Set/Copy/Zero | 60 | ✅ 已合并 | `pkg/p1.2a` → merge |
| P1.2b | NPPS 转换/阈值 Convert/Threshold | 174 | ✅ 已合并 | `pkg/p1.2b` → merge |
| P2.1a | NPPI Convert 数据类型转换 | 200 | ✅ 已实现 | `pkg/p2.1a` → `ed8a5bd` |
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

- **P1.1a 结果**：220 个函数全部实现；136 个差分 case（68 组 Ctx/非 Ctx）在 5090 record、本地 verify 全绿；本地 4,803 全量回归通过；package remaining=0。
- **P1.1c 结果**：258 个函数全部实现；529 个差分 case（binary+constant+unary 全矩阵）在 5090 record、本地 verify 全绿；本地 4,803 全量回归通过；package remaining=0。三个差分生成器的种子改为 SHA-256（原先 Python `hash()` 每次进程随机，导致重生成后 golden 全部失效）。`nppsLn_64f32f` 为有意偏离：NPP 13.4 返回 round(ln)，我们按 API 契约返回自然对数，并用 CPU 参考测试。`nppsExp_32s_Sfs` 因 expf 跨平台末位差异使用 1e-6/2 容差。
- **P1.1b 结果**：256 个函数全部实现；174 个差分 case 在 5090 record、本地 verify 全绿；本地 4,803 全量回归通过；package remaining=0。`nppsMulC_64f64s_ISfs` 的原地缓冲区实际按 int64 读取，舍入为“先 round-half-away 到 int64，再按符号加 ±1 后算术右移”，已用 5090 直连探针验证 204 个点。
- **Agent 运行方式**：全新 worktree 无法用 CMake 4.4 配置 cu-bridge（`FindCUDAToolkit` 需要 `nvvm/libdevice`，cucc 的 implicit-link 输出也不兼容），因此工作包 agent 改为在主 checkout 上 `--in-place` 串行运行，复用已配置的 `build/` 缓存。**禁止运行 `cmake_maca`/`make_maca`**（它们会 `rm -rf ~/cu-bridge`，曾导致工具链损坏，已用 cu-bridge 的 `make_maca` 重建）。
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
