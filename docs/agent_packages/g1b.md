# g1b — NPPI WarpAffine 收尾（cubic 边界规则 + Batch 语义）

你是 mcMPP 的实现子 agent。分支 `pkg/g1` 已有完整实现与 5 个差分测试文件
（提交 `84ee401`），`NppiWarpAffineDiff.*` 当前 **9/12**。本包目标：清零剩余失败。

## 0. 必读

1. `docs/npp_semantics_reverse_engineering.md`：**golden > NPP 库 SASS/PTX > 文档**
2. `src/nppi/nppi_geometry_transforms/nppi_warp_affine_executor.{h,cu}`（已知语义）
3. `test/unit/diff/test_nppi_warp_affine*_generated.cpp` 与 `npp_warp_affine_diff_helpers.h`

## 1. 已知状态（lead 已验证）

- 仿射映射用 float32 系数 + 非融合乘加；双线性用权重式 `(1-d)*v0 + d*v1`（FMA 契约）——NN/linear 已对齐
- 采样点落在（裁剪后的）src ROI 外时目标像素不写入；step/ROI 校验顺序已按 5090 钉死
- 剩余问题：
  1. **cubic 越界 tap 规则**：权重是标准 Catmull-Rom，但越界邻点的处理既非零填充也非边缘复制。
     用 5090 上 `cuobjdump -symbols libnppig.so | grep -i 'warpaffine.*cubic'`
     （符号里有 `SampleCubicImage` / `SampleCubicSplineImage`）再 `-sass -fun` 反汇编确认。
     也可用脉冲图探针（单个非零像素 + 分数平移）直接读出权重与边界行为。
  2. 32s/C3 线性个别 ±1 像素（比对同一输入的 f32/f64 路径定位舍入链）
  3. Batch 家族仍用 double 坐标、缺 srcROI 跳过与 step/ROI 校验；对齐单图路径的语义
- 线性/NN 其余用例已通过，改动不要回退它们。

## 2. 要求

1. **45 分钟内产生第一个 commit**；每修一类（cubic / batch / rounding）立即 commit。
2. 用 `NppiWarpAffineDiff.*` 全绿 + 5090 record 作为完成标准；不要删除或放宽测试。
3. 若 cubic 仍无法钉死，提交 probe 数据与结论（哪些模型被排除），列剩余项。

## 3. 环境

- 本 worktree 不做本地 MPP build；使用 **mcpp-5090 MCP**：
  `sync`（repo_path=当前 worktree，remote_dir=`/home/metax/mcMPP-g1`）、`build`（jobs=4）、
  `record`、`pull`。golden 已在 `artifacts/5090/npp_records/nppiWarpAffine*`。
- 机器共享；输出重定向后 grep/tail。禁止 `cmake_maca`/`make_maca`。

## 4. Definition of Done

1. `NppiWarpAffineDiff.*` 15/15（或 Listing 明确剩余项与证据）。
2. 5090 record 0 failed / 0 crash；提交到 `pkg/g1`（未 push）。
