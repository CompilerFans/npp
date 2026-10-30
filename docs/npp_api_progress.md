# NPP API 全量实现进度账本

> 最后更新：2026-10-29
> 集成分支：`feat/npp-api-expansion`（基于 `dev` @ `3a81f00`）
> 公共契约：`API/`（API-unified）
> 参考实现：NVIDIA NPP 13.4 / RTX 5090 / CUDA 13.4

## 1. 基线

| 指标 | 数值 |
|---|---:|
| 公共函数 | 12,537 |
| 真实实现 | 6,688 |
| 显式 stub | 166 |
| 缺失 | 5,683 |
| 待实现（stub+missing） | 5,849 |
| 待实现 `_Ctx` | 2,978 |
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
| P2.1a | NPPI Convert 数据类型转换 | 200 | ✅ 已合并 | `pkg/p2.1a` → `ed8a5bd` |
| P2.1b | NPPI Copy（packed/planar/planar↔packed） | 160 | ✅ 已合并 | `pkg/p2.1b` → merge |
| P2.1c | NPPI Set / Dup | 174 | ✅ 已合并 | `pkg/p2.1c` → merge |
| P2.2a | NPPI SwapChannels / CopySubpix | 110 | ✅ 已合并 | `pkg/p2.2a` → `5db538e` |
| P2.2b | NPPI Scale / 边界复制 | 182 | ✅ 已合并 | `pkg/p2.2b` → `66bf923` |
| P2.3 | NPPI 阈值 / 比较 | 366 | ✅ 已合并 | `pkg/p2.3` → `f61153d` |
| P3.1a1 | NPPI Sum/Mean/MeanStdDev + buffer sizes | 212 | ✅ 已合并 | `pkg/p3.1a1` → merge |
| P3.1b1 | NPPI Norm/NormDiff（L1/L2/Inf）+ buffer sizes | 580 | ✅ 已合并 | `pkg/p3.1b1` → `ff1085e` |
| P3.1b2 | NPPI NormRel/DotProd + NormLevel buffer helpers | 544 | ✅ 已合并 | `pkg/p3.1b2` → `a38f9c4` |
| P1.3b | Norm/NormDiff/NormRel/DotProd | 待拆 | ✅ 已完成（P3.1b1 + P3.1b2） | — |
| P1.3c | NPPI AverageError/AverageRelativeError + GBHS | 352 | ✅ 已合并 | `pkg/p1.3c` → `07af3da` |
| P0.1 | 纯重构：公共 npp_kernel_utils.h + npp_reduction.cuh（无新 API） | 0 | ✅ 已完成（待合并） | `pkg/p0.1` → `b52d81c` |
| P1.3d | NPPI MaximumError/MaximumRelativeError + GBHS | 352 | ✅ 已合并 | `pkg/p1.3d` → `44ce788` |
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

- **P1.3d 结果**：MaximumError/MaximumRelativeError + GBHS 共 352 个函数全部 real，package remaining=0；catalog real 7,580→7,932、missing 4,791→4,439。验证：5090 record 148 个 golden，本地 `NppiMaximumErrorDiff.*` 13/13 通过，全量 unit 4,803、全量 diff 1,638 通过。关键语义（5090 实测）：**输出对所有布局 C1–C4 都是单个 `Npp64f` 标量**（NPP 文档“C3/C4 每通道数组”的说法是错的）；零分母样本贡献 0（全零图 → 0.0）；无反号 2.0 截断（16s 极值为 1.999969482421875）；全零 MaximumError 写出 −0.0；空 ROI → NO_ERROR 且不写；校验顺序 null src → 负 ROI → 零 ROI → 每源 step（u32、NOT_EVEN 规则 `sizeof(D)·2^v2(Ch)`）→ null out → null scratch；C4 × 8 字节 dtype 需 step 为 32 的倍数。遗留：p1.3c 的 C4 64f/32fc/32sc 17×13 golden 存在 NOT_EVEN 盲区，后续补记。
- **P0.1 结果**：纯重构包（p0.1.csv 为空，无新 API），产出两个公共头。`src/include/npp_kernel_utils.h`（namespace `npp_kernel`，宿主安全，被生成的 .cpp 包含）：`BlockConfig` 改为 `template <int BlockX=16, int BlockY=16>`；`getThreadCoords`（引用 blockIdx 等 builtin，以 `__CUDACC__` 保护，宿主编译器实测不接受裸 blockIdx 函数体）；`getSrcRow`/`getDstRow`、`checkCudaKernelError`、`validateShiftConstants`/`validateBinaryParameters`/`validateUnaryParameters`（自 nppi_arithmetic_executor.h 原样搬移）；新增变参 `validateMaskedParameters`（全指针非空 + ROI 严格为正，无 step 检查），7 个 masked 风格 executor（MaskedBinary/MaskedUnary/MixedUnary/Weighted/MaskedWeighted/AlphaPremul/AlphaPremulAC4）迁移到该模板。`src/include/npp_reduction.cuh`（namespace `npp_reduce`，仅设备端，.cuh 惯例允许未保护的 `__global__`）：两级归约 25 个定义（kBlock/blockReduce/numBlocks + Sum/Mean/MeanStdDev/masked 的 partial+finalize + Norm/NormRel/DotProd 内核）自 nppi_statistics_executor.cu 匿名命名空间原样搬移；Norm 内核模板参数改为中性 `npp_reduce::NormKind{L1,L2,Inf}`（共享头不依赖家族头），statistics .cu 以 constexpr `normKind(NormType)` 在 18 个 launch 点内联转换，宿主面 `nppi_statistics_executor.h` 零改动、NormType 原位保留。npps 侧删除 6 份完全等价的 `finish_launch`（NPP_SUCCESS==NPP_NO_ERROR==0），32 个调用点改用 using 声明再导出的 `checkCudaKernelError`。验证：全量 unit 4,803/4,803、全量 diff 1,625 通过（2 个既有 skip，与基线逐条一致）、构建告警数与基线一致（68 条既有 CMake policy 提示，零编译告警）。未收敛点：`nppi_convert_executor.cu` 的 `finish_launch`（nppi 模块，超出本包命名范围）；npps 各 .cu 本地 `kBlock=256`（1D 启动几何，无共享等价物，npp_reduce::kBlock 属归约头）；Compare/Ternary/MixedTernary/Quaternary 的逐类型 step 校验与 MultiConst16f 校验（非重复实现）；statistics 的 extreme/indx/avgErr/zeroDevice 内核留在 .cu（非重复）；其余 nppi 家族的硬编码 `dim3 blockSize(16,16)` 亦不在本包范围。
- **P1.3c 结果**：AverageError/AverageRelativeError + GetBufferHostSize 共 352 个函数全部 real，package remaining=0；catalog real 7,232→7,580、missing 5,849→4,791。验证：5090 record 10 个测试/157 golden 通过，本地 `NPP_VERIFY_DIR` 12/12 通过，全量 unit 4,803 通过，全量 diff 通过。关键语义（5090 实测）：AverageError = Σ|a−b|/(W·H·Channels)，复数按模；AverageRelativeError 逐像素 |a−b|/max(|a|,|b|)，分母为 0 的像素项记 0 但**仍计入归一化**（全零图 → 0.0）；step 按 **u32** 比较，`(u32)step < w·ch·sizeof(D)` → STEP_ERROR，`(u32)step % (sizeof(D)·2^v2(Channels)) ≠ 0` → NOT_EVEN_STEP_ERROR（step=−1 可通过）；校验顺序 null src → 负 ROI → 零 ROI（no-op）→ src1 step → src2 step → null dst → null scratch（**scratch 为空返回 NULL_POINTER_ERROR**，与 Sum/Mean/DotProd 的 no-op 不同）；GBHS 零 ROI 返回 NO_ERROR 且不写。AC4 在 p1.3c.csv 中为 0。
- **P3.1b2 结果**：544 个函数全部实现并标记 real（`nppi_statistics_functions.h` 的 NormRel L1/L2/Inf + GetBufferHostSize、DotProd + GetBufferHostSize、Full/Same/Valid NormLevel GetBufferHostSize）；package remaining=0，catalog real 6,688→7,232、remaining 5,849→5,305。验证：5090 record 13 个测试（75 个 golden）通过，本地 `NPP_VERIFY_DIR` 回放 15/15 通过（含 2 个非 Ctx），全量 `diff_tests` 1,613 通过，全量 `unit_tests` 4,803 通过、零回归。关键语义（5090 实测）：NormRel 分母为 0 且分子 > 0 → 返回**源类型的正最大值转 double**（8u→255、8s→127、16u→65535、16s→32767、32f→FLT_MAX），不是 inf；分子分母同时为 0 → 0.0（不是 NaN）；有限比值**不做钳制**（8u L1 比值 56354.0 原样通过）。NormLevel buffer helper 只承诺 status，精确大小待 P3.4 CrossCorr 需要时再探针；非 Ctx 变体只能靠 golden 回放覆盖。
- **P1.1a 结果**：220 个函数全部实现；136 个差分 case（68 组 Ctx/非 Ctx）在 5090 record、本地 verify 全绿；本地 4,803 全量回归通过；package remaining=0。
- **P2.3 结果**：418 个函数全部实现（366 待实现 + 52 已有；含 2 个运行时派发 `nppiFusedAbsDiff_Threshold_GTVal[_I]`）；`nppi_threshold_and_compare_operations.h` coverage 由 `366 missing` 降为 `0`。差分 12 case（阈值 7 族 + 比较 4 族 + Contract，另有 Fused 6 case）在 5090 record、本地 verify 全绿；本地 4,803 全量回归通过；package remaining=0。关键契约（以 5090 golden 为准）：阈值 AC4/AC4IR 的 alpha 通道**完全不写**（dst alpha 保持缓冲区原有值，而非拷贝 src）；比较 AC4 只比较 RGB（忽略 alpha 通道），而 C4R 比较全部 4 通道；比较类入口对非法 `eComparisonOperation` **不校验**，直接短路为 `NPP_NO_ERROR`（在指针/ROI 检查之前）；`nppiThreshold`（base/Val）对非法 op 返回 `NPP_NOT_SUPPORTED_MODE_ERROR`；负 ROI 宽高 → `NPP_SIZE_ERROR`、空 ROI → `NPP_NO_ERROR`。FusedAbsDiff 的 CH_A4 也符合“alpha 不写”约定。`nppiFusedAbsDiff_Threshold_GTVal[_I]` 走 `NppDataType`/`NppiChannels` 运行时派发（支持 8U/16U/16S/32F × CH_1/CH_3/CH_A4），阈值/替换数组按值传入 kernel。
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

## 6. 性能优化（600U，2026-10-30）

- **600U（cjxu@10.0.82.53）仅每日 00:00–07:00 可用**；其余时间 SSH 认证后即被关闭（节点不可用，属预定窗口）。
- 基准：`benchmark/` 已有 506 项；30 die 并行扫描，基线 `artifacts/benchmarks/600u_baseline.csv`；节点本地 30 分钟刷新循环（`/tmp/bench_refresh.sh`）。
- 已合并优化：
  - `opt_reduce`（`c1cce5a`）：归约 16B 向量化，geomean **3.36x**，峰值 463–536 GB/s。
  - `opt_elem`（`ea894fe`）：逐元素/拷贝 128-bit；Copy **2–11x**、Add 1.2–1.85x；`nppiAdd_8u_C3RSfs` 回退 0.74x 待查。
  - `opt_morph`（`8a1a6a6`）：Erode/Dilate 8u C1 **~2.1x**；C4/32f 未提速。
- 待办（因节点窗口暂停，00:00 后重启）：`opt_planar`（平面拷贝 2.9–7.9 GB/s）、`opt_fix`（8u_C3 回退 + 形态学 C4/32f 扩展）。
