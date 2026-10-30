# REF_MISC — 收尾剩余参考测试类别（全量测试最后一批）

你是 mcMPP 的**参考测试收尾**子 agent。目标：在 `ref/planar` 已覆盖 planar/pointer-array/batch（549 条）之后，把剩余可测试的 skip 类别全部自动化，使 **NPP 13.4 上所有可调用的 ctx=1 函数都有参考测试并 record 全绿**。

## 0. 必读

1. `docs/agent_packages/REFERENCE_TESTS.md`（record/verify 协议、harness、override 格式）
2. `docs/agent_packages/REGEN_ALL.md`、`docs/agent_packages/REF_PLANAR.md`
3. `tools/gen_ref_tests.py`、`test/unit/framework/npp_diff_harness.h`
4. 当前 skip 清单（`test/unit/ref/skipped_*.json`）与 `tools/gen_ref_tests.py --all --dry-run`

## 1. 剩余类别（以当前 dry-run 为准，约 611 个）

| 类别 | 数量 | 代表 |
|---|---:|---|
| 未识别的输出标量/数组指针 | ~250 | `pNormDiff` 36、`pNormRel` 36、`pDp` 30、`pMinValue` 16、`pMaxAbs`/`pMinAbs` 8、`pIndx` 8、`pIndexX/pIndexY` 8、`pMSE`/`pPSNR`/`pSSIM` 各 2、`pCounts` 3、`pMSSSIM` 3、`pSqr` 6、`pValZC` 2、`pNewNumber` 1、`pD2FVal` 1、`pBuffer64` 1 |
| 未识别的 host 输出 | ~57 | `nBufferSize` 32、`nLevels` 21、`nColorKeyConst` 3、`hpLocations` 8（NppiPoint） |
| 未识别的 scratch | ~61 | `pAdvancedScratchBuffer` 45、`pAntialiasingDeviceBuffer` 16 |
| 未支持的枚举/结构体标量 | ~80 | `NppiNorm` 68、`NppiBayerGridPosition` 6、`NppDataType` 2、`NppPointPolar` 2、`NppiPoint32f` 2 |
| 未支持的指针类型 | ~10 | `Npp64sc` 8（另 1 个标量）、`NppiCompressedMarkerLabelsInfo` 1 |
| 复杂 API（可能无法自动化） | ~5 | `pContoursImageDev` 2、`pMarkerLabels` 2、`pCompressedMarkerLabels` 1、`pMarkerLabelsInfoListDev` 1 |

不在本包范围：non-ctx ABI（6,146，NPP 13 已移除，永久 skip）、已有 override（18，NPP 崩溃先例）。

## 2. 生成器扩展要求

只改 `tools/gen_ref_tests.py`（必要时扩展 harness）的通用规则：

1. **输出角色**：把上述输出标量/数组加入 `OUTPUT_SCALAR`/`OUTPUT_PREFIXES` 或新增 `OUTPUT_ARRAY` 角色，按元素类型分配 device 缓冲并 `ref_capture_bytes` 捕获。
2. **host 输出**：`nBufferSize`/`nLevels`/`nColorKeyConst`/`hpLocations` 按 host 内存分配，调用后按类型捕获（`NppiPoint` 数组按 `{x,y}` 序列化）。
3. **scratch**：`pAdvancedScratchBuffer`/`pAntialiasingDeviceBuffer` 归入 scratch 角色（按 `1024*1024` 字节分配）。
4. **标量/枚举**：`NppiNorm`（如 `NPP_NORM_L1`）、`NppiBayerGridPosition`、`NppDataType`、`NppPointPolar`、`NppiPoint32f` 给安全默认值；`Npp64sc` 加入 `SCALAR`（CF64，16B）。
5. **复杂 API**：确实无法安全构造的（压缩 marker labels、contours 等），在 skip manifest 中用**精确原因**记录，不要强行生成非法调用。

## 3. 验证

1. 扩展后全量重生成；**已入库的 5,980 个测试（5,431 + 549）对应文件必须零 diff**；如有变化必须重新 record 并说明。
2. 5090 隔离目录（`MCPP_5090_DIR=/home/metax/mcMPP-ref-misc`）sync + build（`USE_NVIDIA_NPP=ON`）。
3. 先 smoke record，再全量 `RefSuite.*`，迭代到 **0 failed / 0 crash**。
4. 拉回 golden 到 `artifacts/5090/ref_records/`。
5. 提交到 `ref/misc`（分批）。
6. 报告：新增测试数、每个剩余 skip 的原因、生成器修复摘要、全量 record 结果、复现性证明。

## 4. Definition of Done

1. NPP 13.4 上所有可调用的 ctx=1 函数要么有参考测试并通过，要么在 skip manifest 中有精确的、可复核的不可测原因。
2. 全量 `RefSuite.*`：0 failed / 0 crash。
3. 既有测试零回归；重跑生成器 `git status` 为空。
4. golden 拉回；提交在 `ref/misc`。

## 5. 约束

- 不 push、不改集成分支、不删除既有测试、禁止 `cmake_maca`/`make_maca`。
- 生成文件不手改；工具输出保持小；分批提交。
- 不为凑数生成参数非法或会崩溃的调用；宁 skip 并说明。

## 6. 执行纪律（2026-10-30 补充，超时教训）

上一轮 agent 6 小时未产出任何生成器/测试提交，只留下探针。本轮必须遵守：

1. **45 分钟内必须产生第一个 commit**（先做输出角色，跑通一个模块的生成+record 即提交）。
2. **按类别分阶段提交**，每类完成（生成 + 5090 record 通过）立即 commit：
   1) 输出标量/数组（~250）；2) host 输出（~57）；3) scratch（~61）；4) 枚举/结构体/Npp64sc（~90）；5) 复杂 API 的精确 skip 原因（~5）。
3. **单类探针预算 ≤10 分钟**：优先复用现有 `pointer_role`/`host_array_init`/`SCALAR` 模式；不要为单个函数长时间手工推演。
4. 每个类别先跑受影响模块的 smoke record，再跑全量 `RefSuite.*`；不要等全部做完才第一次 record。
5. 若某类别确实无法在预算内完成，提交已完成部分并在报告中列明剩余项，不要空转。
