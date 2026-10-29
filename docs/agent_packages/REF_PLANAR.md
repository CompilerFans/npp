# REF_PLANAR — 平面/指针数组与 batch 结构体参考测试

你是 mcMPP 的**参考测试扩展**子 agent。统一生成器目前跳过 254 个 planar 指针数组、186 个 array-of-pointers、109 个 batch 结构体函数（见 REGEN_ALL 报告的 skip 分类）。本包扩展生成器，使这些布局能自动生成可 record 的参考测试。

## 0. 必读

1. `docs/agent_packages/REFERENCE_TESTS.md`（record/verify 协议、harness、override 格式）
2. `docs/agent_packages/REGEN_ALL.md`（统一生成器与上一轮结果）
3. `tools/gen_ref_tests.py`、`test/unit/framework/npp_diff_harness.h`
4. 相关头文件中的布局约定：`API/nppi_data_exchange_and_initialization.h`、`API/nppi_geometry_transforms.h`、`API/nppi_filtering_functions.h` 等

## 1. 范围（只做这三类，不扩大）

1. **planar 指针数组**：`pSrc[3]` / `pDst[3]` / `pSrcDst[4]` 等，逐平面 buffer。
2. **array-of-pointers 参数**：如 `const Npp8u * pSrc[N]`，按平面数量分配并传数组。
3. **batch 结构体**：`NppiWarpAffineBatchCXR`、`NppiWarpPerspectiveBatchCXR`、`NppiMirrorBatchCXR`、`NppiResizeBatchCXR`、`NppiImageDescriptor` + `pBatchList` / `nBatchSize`，构造 host 端描述符数组指向 device buffer。

当前 skip 数量（以 `--all --dry-run` 为准）：planar 254、array-of-pointers 186、batch 109。

## 2. 生成器扩展要求

- 只改 `tools/gen_ref_tests.py` 的通用逻辑：参数分类（planar/ptr-array/batch）、逐平面/逐元素 buffer 分配与 step、golden 捕获命名（平面/元素后缀）、batch 描述符构造。
- `test/unit/framework/npp_diff_harness.h` 如需多平面捕获，做**通用**扩展（如 `ref_capture_planes`），不要为单模块特判。
- 不改变已支持函数的生成结果：扩展后运行 `python3 tools/gen_ref_tests.py --all --chunk-size 200`，**已入库的 5,431 个测试对应的文件必须零 diff**（若确实需要变化，必须重新 record 并在报告中说明）。
- 生成文件入库，不手改。

## 3. 验证

1. 生成扩展后先 dry-run 统计新增测试数；再全量重生成并确认既有文件无 diff。
2. 在隔离目录（`MCPP_5090_DIR=/home/metax/mcMPP-ref-planar`）sync + build（`USE_NVIDIA_NPP=ON`）。
3. 先 smoke record 新增用例，再全量：
   ```bash
   NPP_RECORD_DIR=<remote>/build-nvidia/ref_records ./ref_tests --gtest_filter='RefSuite.*'
   ```
4. 全量必须 **0 failed / 0 crash**；NPP 崩溃/头库不一致的用 overrides 记录（附最小复现与证据）。生成器 bug 做通用修复。
5. 拉回 golden 到 `artifacts/5090/ref_records/`。
6. 提交到 `ref/planar`（分批提交）。

## 4. Definition of Done

1. 三类布局中 NPP 13.4 实际支持且可安全调用的函数已生成测试并 record 全绿；确实无法支持的（如 NPP 崩溃、遗留 ABI）在 skip manifest/override 中记录原因。
2. 既有 5,431 个测试零回归（文件零 diff 或已重新 record 且全绿）。
3. 全量 `RefSuite.*` record：0 failed / 0 crash。
4. 重跑生成器后 `git status` 为空（复现性）。
5. 报告：新增测试数、三类各自的覆盖/剩余、overrides、生成器改动摘要。

## 5. 约束

- 不 push、不改集成分支、不删除既有测试、禁止 `cmake_maca`/`make_maca`。
- 生成文件不手改；工具输出保持小；分批提交。
- 不要为了凑数生成参数非法的调用；宁可 skip 并说明。
