# NPP API 全量实现进度账本

> 最后更新：2026-10-31
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
| P2.4 | NPPS 统计（Error/DotProd/Mean/StdDev/MinMax/Norm/Sum/Every/Count/ZC） | 766 | ✅ 已合并 | `pkg/p2.4` + `pkg/p2.4b` → `817f293` |
| P2.4b | NPPS 统计差分测试（239 例，无新 API） | 0 | ✅ 已合并 | `pkg/p2.4b` → `817f293` + `c419d00` + `c61ca5f` |
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
| P1.8 | NPPI CrossCorr/SqrDistance Full/Same/Valid + Norm/NormLevel/Advanced | 466 | ✅ 已合并 | `pkg/p1.8` + `pkg/p1.8b` + `pkg/p1.8c` → `e016ecd` |
| P1.8b | CrossCorr 数值算术修复（rss/f32/NormLevel 整数形式） | 0 | ✅ 已合并 | `pkg/p1.8b` → `e016ecd` |
| P1.8c | CrossCorr status/契约/scratch 对齐 | 0 | ✅ 已合并 | `pkg/p1.8c` → `e016ecd` |
| P1.9 | NPPI 滤波 | 1,802 | 规划中 | — |
| P1.10 | NPPI 颜色（待 238733 合入后） | 646 | 阻塞 | — |
| G3 | NPPI Mirror / MirrorBatch 全变体 | 48 | ✅ 已合并 | `pkg/g3` → `e9d52ff` |
| G1 | NPPI WarpAffine 家族（实现+测试已写，cubic 边界核待对齐） | 156 | ⏸ 暂停 | `pkg/g1` → `84ee401` |
| F1 | NPPI 基础滤波（Filter/FilterBorder/32f 行/列 + border） | 262 | ✅ 已合并 | `pkg/f1` → `6390cd6` |
| F4 | NPPI 导数滤波（Sobel/Scharr/Prewitt/Roberts 及 mask/second/border 变体） | 348 | ✅ 已合并 | `pkg/f4` → `4ecdea2` |
| F12 | NPPI 滤波长尾（nppiFilter*/Row/Column + SumWindow 全变体） | 240 | ✅ 已验证 | `pkg/f12` + `f12-verify` → `c866fe3` |
| F11 | NPPI 固定核滤波（HighPass/LowPass/Sharpen/Laplace + border） | 296 | ⏳ 部分合并（248/296，Unsharp 待做） | `pkg/f11` → `691a899` |
| G5 | NPPI 透视/旋转/几何辅助（8 个 host get* 已合入；perspective/rotate 为 WIP） | 116 | ⏳ 部分合并（8/116） | `pkg/g5` → `fabfb09` |
| opt_fix | 600U：形态学 word-per-pixel 向量化 + 8u C3 回退定位 | 0 | ✅ 已合并 | `pkg/opt_fix` → `409d5ef` |
| opt_planar | 600U：平面↔打包拷贝 128-bit 逐平面向量化 | 0 | ✅ 已合并 | `pkg/opt_planar` → `409d5ef` |

## 4. 每个包的合并检查

1. `git diff --stat feat/npp-api-expansion..pkg/<id>` 只包含本包文件。
2. 本地 `./build.sh` + `LD_LIBRARY_PATH=build/src ./build/unit_tests` 全绿（基线 4,803）。
3. `build/diff_tests` 在 `NPP_VERIFY_DIR=artifacts/5090/npp_records` 下全绿。
4. `python3 tools/npp_api_catalog.py --update` 后包内 remaining=0。
5. `git merge --no-ff pkg/<id>` 到集成分支；更新本账本。

## 5. 决策记录

- **G5 结果（部分）**：8 个 host 几何辅助（GetAffineTransform/Quad/Bound、GetPerspectiveTransform/Quad/Bound、GetRotateQuad/Bound）合入（`pkg/g5` + `g5-verify` → `fabfb09`），catalog real 10,302→10,310、tested 8,236→8,244。验证：`NppiGetTransformDiff.*` 12/12、整树 unit 4,823、diff 2,029 用例 0 failed。`pkg/g5` 上保留了 perspective/rotate 的 4 个差分 suite（64 个 case，当前全部打在 legacy 实现上失败）作为后续 108 个函数的规格，其 453 个 golden 已放入共享 `artifacts/5090/npp_records`；`docs/agent_packages/g5_notes.md` 记录了实测的透视区域/采样模型（区域裁剪、采样点判定、边界不写入）。这些 WIP 测试不并入集成分支，避免 64 个失败用例。
- **F11 结果（部分）**：HighPass/LowPass/Sharpen/Laplace 及其 Border 变体共 248 个函数合入（`pkg/f11` + `f11-verify` → `691a899`），catalog real 10,054→10,302、missing 2,365→2,117、remaining 2,483→2,235；剩余 48 个为 `nppiFilterUnsharpBorder`（32）与 `nppiFilterUnsharpGetBufferSize`（16），agent 的 probe_j 已钉住部分 unsharp 语义。验证：`NppiFilterFixedDiff.*` 8/8（f11 agent 在 5090 重录 796 个 golden 后由 lead 在 C500 本地验证）、整树 unit 4,823、diff 2,017 用例 0 failed。关键语义/修复：NPP 13.4 的 `nppiFilterSharpenBorder_16u_C1R` 是唯一不校验 srcSize/offset/border 的固定核 border 变体（这些情况全部返回 NO_ERROR；null、负 ROI、dst step 仍正常校验），已按 golden 在 `filterFixedBorder` 中做实例级例外；其余变体的校验顺序为 null → 负 ROI → srcSize → offset → border → 零 ROI（no-op，先于 step）→ dst step（含 NOT_EVEN）。f11 的 796 个 golden（`nppiFilterFixed/`）已并入共享 `artifacts/5090/npp_records`。
- **F12 结果**：nppiFilter(*)/Row/Column + SumWindow 共 240 个函数全部 real（`api_analysis/packages/f12.csv` 240/240），catalog real 9,824→10,054、missing 2,595→2,365、remaining 2,713→2,483。验证：f12 四个差分 suite 35/35，整树 unit 4,823/4,823、diff 2,011 用例 0 failed、全程无设备 fault。关键语义（5090 直接探针 + golden）：kernel/mask 参数是 **host 指针**，所有入口改用 `cudaMallocAsync` + `cudaMemcpyDefault` staging（同时兼容差分 harness 传的设备指针；此前内核直接把 host 指针当设备指针读，导致 MXC ATU fault，差分 harness 传设备指针掩盖了这个问题）；空/零/负核与 mask 的零填充范围是每轴 `clamp(dim-1+k, 0, dim)`（k=0 保留最后一条线，负值收缩），AC4 只填 RGB 保留 alpha，64f 2D 填满整个 ROI；64f 行/列不校验 anchor，mask≤0 是带符号 anchor 上的单 tap `k[0]`，正 mask 时 anchor 钳到 `[0, mask-1]`；8u C4R/AC4R 行/列（含 border）用无符号 32 位环加 + 无符号除法（负和回绕后饱和到 255），2D 8u 仍为有符号；整数行/列 anchor 上界是 mask（含），且在 step/divisor 之前校验；32f C1R 对奇数方形核≥3（2D）和奇数 mask≥3（行/列）把 tap 钳进 ROI，其他形状/layout 读原始内存；sum-window 窗口为 `[p-a, p-a+mask-1]`，行核把 tap 坐标钳到 `W+mask-1`（列核不钳，脉冲探针拟合），`anchor < -mask` → OUT_OFF_RANGE。差分 status 探针的源缓冲加 margin、负 step 目标提前分配（MXC 读到缓冲之前会 ATU fault；case 内 `cudaMalloc` 会把 invalid argument 留在 sticky error 里）；单测 sum-window 改用零 padding 源以匹配零边界 CPU 参考。f12 的 1,765 个 golden（agent 的 `rec-f12b/`）已拉回共享 `artifacts/5090/npp_records`。
- **600U 优化结果（opt_fix + opt_planar，2026-10-31）**：两包合入 `409d5ef`。opt_fix：`nppiMorphWordPix_kernel`（8u C4R uchar4 SWAR、32f C1R float compare-select，保持 NaN/±0 与 clamped 边界，6 个接入点，不对齐自动回退）实测 8u C4R 3x3 1.19–1.49×、5x5/7x7 2.59–3.78×、32f C1R 2.68–3.98×；此前报告的 `nppiAdd_8u_C3RSfs` 147→108 GB/s “回退”经同设备 A/B/C 三臂复测确认是刷新 harness 的测量噪声（同二进制波动 2.8×），向量路径实际 1.34–1.65×。opt_planar：逐平面 128-bit 拷贝，C3P3 656–800 GB/s、P3C3 307–318 GB/s、C4P4/P4C4 188–698 GB/s，相对旧实现 1.4×–136×；P3C3 方向受读分散限制约 310 GB/s。验证：集成树 unit 4,823、diff 1,974；600U 上全量 unit 4,820/4,820（健康设备 4）。环境记录：node53 MCP 未挂载到子会话（当时 MCP 配置只含 5090），两个 opt agent 改用直连 SSH；opt_planar 报告设备 15/14/12/10/8 出现 per-device 驱动楔死（非阻塞流创建 ioctl 超时），疑似被 kill 的 GPU 进程泄漏队列资源，换设备 4 后全过。
- **F4 结果**：Sobel/Scharr/Prewitt/Roberts 及其 mask、second-derivative、border 变体共 22 个 family、348 个函数合入，catalog real 9,476→9,824、missing 2,943→2,595、remaining 3,061→2,713。验证：`NppiFilterDerivDiff.*` 22/22、整树 unit 4,803、diff 1,974 全绿。f4 的 758 个 golden 已从其 worktree 并入共享 `artifacts/5090/npp_records/nppiFilterDeriv`。注意：第一次跑全量 unit 时 C500 发生一次设备复位（`ResetType=1`，之后所有分配返回 10002），全新进程重跑全量无失败，属一次性事件而非 f4 回归。
- **F1 结果**：nppiFilter32f / nppiFilterBorder32f / nppiFilterBorder（整数 divisor）/ nppiFilterRow32f / nppiFilterColumn32f / 行/列 border 32f 共 262 个函数合入，catalog real 9,214→9,476、missing 3,205→2,943、remaining 3,323→3,061。验证：三个 filter 差分 suite 23/23、整树 unit 4,803、diff 1,952 全绿。5090 实测补齐的边界语义：空核/空 mask 会零填充但**保留最后一条线**（2D 保留最后一行+最后一列，行滤波保留最后一列，列滤波保留最后一行）；负 mask 在 border 行/列族是 NO_ERROR no-op，在非 border 族走 anchor 校验返回 OUT_OFF_RANGE；负 dst step 通过无符号 size 检查后 NPP 返回 NO_ERROR（我们跳过 launch）；整数 nppiFilterBorder 的 srcOffset 越界检查先于 divisor==0；16f 变体在 5090 上是真实可用的（golden 为 NO_ERROR），已改为 half→float→half 的真实实现（原实现错误地返回 NOT_SUFFICIENT_COMPUTE_CAPABILITY）。f1 的 254 个 golden 已从 agent 的远端目录拉回共享 `artifacts/5090/npp_records`（路径按 case id 扁平化，如 `nppiFilter32f_.../`）。
- **G3 结果**：nppiMirror / nppiMirrorBatch 全变体（16s/16u/32s/32f/8u × C1/C3/C4/AC4 × R/IR + batch）合入，catalog real 9,166→9,214、stub 166→118、remaining 3,371→3,323。验证：`NppiMirrorDiff.*` 27/27、整树 unit 4,803、diff 1,929 全绿。关键修复：legacy Mirror 实现对空 ROI 会以 0 高度网格启动并返回 CUDA 错误，现按 NPP 语义改为 NO_ERROR no-op；MirrorBatch 差分测试的 input-only Case 无 status 会被 verifier 拒绝，已把输入哈希并入 output Case。g3 的 73 个 golden（nppiMirror*）已并入共享 `artifacts/5090/npp_records`。
- **G1 进展（暂停）**：WarpAffine/WarpAffineBack/Quad/Batch 实现与 5 个差分测试文件已完成（`pkg/g1` → `84ee401`），`NppiWarpAffineDiff.*` 已修到 9/12。已验证的语义：仿射映射用 float32 系数与非融合乘加、双线性用权重式、采样点落在（裁剪后的）src ROI 外时目标像素不写入、step 校验（size 先于 `sizeof·2^v2(Channels)` 对齐、src 先于 dst）、负 src ROI → SIZE、负 dst 尺寸 → SIZE、零 dst 尺寸 → NO_ERROR、src 图像需 ≥2x2。剩余：cubic 边界处理（探针确认权重是标准 Catmull-Rom，但越界 tap 的处理既非零填充也非边缘复制）、32s/C3 个别线性 ±1、Batch 家族仍用 double 坐标且缺 ROI/step 语义。未合入，不影响集成树。
- **P1.8c 结果**：p1.8 的 466 个 CrossCorr/SqrDistance 函数全部 real（`api_analysis/packages/p1.8.csv` 466/466），p1.8b 修复数值算术，p1.8c 把剩余 11 个 status/contract/scratch 用例清零（`NppiCrossCorrDiff.*` 15/15）。catalog real 8,700→9,166、missing 3,671→3,205、remaining 3,837→3,371。验证：合并后 unit 4,803、diff 1,902 全绿。关键语义（5090 直接探针）：src step 在 pTpl/pDst 判空之前校验；even-step 模数为 `sizeof(element) * 2^v2(Channels)`（8u C3 接受奇数 step；C4/AC4 要求 4×元素大小，32f C4 dst 为 16）；Valid 模式下负的 dst 维度 → SIZE_ERROR、零 → 空 no-op；NormLevel 的 pDeviceBuffer 判空先于 Full/Same 的 tpl>src 几何错误；GetAdvancedScratchBufferSize 的 null 指针最先返回 NULL，Full 要求 dst 扩展为正（负 tpl 不直接报错），Same/Valid 拒绝零/负 src 且完全忽略 tpl；NormLevel GetBufferHostSize = `8 * channels * (height + 5)`（与宽度/元素类型无关），生成器改用 `{chan}` 占位符按布局后缀传通道数。
- **P2.4b 结果**：p2.4 的 766 个 NPPS 统计函数全部 real（`api_analysis/packages/p2.4.csv` 766/766），p2.4b 共补 239 个 `NppsStatisticsDiff` 差分 case（5090 NPP 13.4 录制，0 failed / 0 crash），覆盖 DotProd、四类误差、Sum/Mean/StdDev/MeanStdDev、Min/Max/Indx/Abs/MinMax（含 tie/NaN 探针）、Norm/NormDiff（Inf/L1/L2/L2Sqr，plain+Sfs，含复数）、MinEvery/MaxEvery、CountInRange、ZeroCrossing（ZCR/ZCXor/ZCC）。catalog real 7,942→8,700、missing 4,429→3,671、remaining 4,595→3,837。验证：合并后 unit 4,803、diff 1,887 全绿。关键修复/语义（5090 直接探针，非 golden）：p2.4 executor 原先编译不过（header 内 `__global__` 需 `static`、`ExtSlot` 命名空间、unsigned complex traits、Norm_Inf 误用双输入 max 核）；Maximum/AverageRelativeError 实际是 `|a-b| / max(a,b)` 的**有符号** max（文档的 `max(|a|,|b|)` 对实型是错的），分母 ≤0 记 0，Average\* = max/n；Sfs StdDev/MeanStdDev 先把 `sqrt(var)` 截断为整数再按 `2^sf` 做 round-half-even（探针：sqrt 9.899→9；574/4 = 143.5→144）；CountInRange 是**严格开区间** `lo < x < hi`，`lo > hi` 记 0（原实现用无序乘积判断，反转区间会误计）；Norm/NormDiff/Every/ZC 的现有实现在 63 个新 golden 上直接通过。
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
