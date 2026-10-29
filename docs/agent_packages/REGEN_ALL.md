# REGEN_ALL — 全量重生成并重新录制参考测试

你是 mcMPP 的**参考测试重生成**子 agent。`tools/gen_ref_tests.py` 已经合并了各分支的生成器修复（data 分支的多维参数解析/逐指针通道与 step/host 常量数组/WMAX-HMAX 缓冲，color 的 `is_ptr_array`，morpharith 的 AC1=2 值/像素、定点复数宽度、`NppiAlphaOp`、`pConstant`）。但已合入的生成文件早于这个统一版本，其中定点复数（`Npp16sc`=4B、`Npp8uc`=2B、`Npp8sc`=2B、`Npp16uc`=4B）此前按 8B 计算 step，存在越界隐患。

本包的任务：**用统一后的生成器重生成全部参考测试，在 5090 上一次性重新 record，消除生成器与 golden 之间的漂移。**

## 0. 必读

1. `docs/agent_packages/REFERENCE_TESTS.md`（record/verify 协议、harness、override 格式、已知 5090 限制）
2. `tools/gen_ref_tests.py`（统一后的生成器）
3. `test/unit/framework/npp_diff_harness.h`
4. `test/unit/ref/overrides/` 下已有的 skip 先例

## 1. 范围

全部 `API/npps_*.h`、`API/nppi_*.h` 模块中 `ctx=1` 的函数（即 `tools/gen_ref_tests.py --all` 覆盖的范围）。当前 dry-run 结果为 **5,431 个测试 / 7,106 个 skip**；以实际生成为准。

预期会变化的文件（属正常）：
- 所有定点复数/AC1/混合类型函数：step 与缓冲尺寸变化；
- 之前被旧生成器跳过的函数现在可能生成新测试（例如 host 常量数组、border 系列）；
- planar、非 Ctx、batch 结构体仍按现有规则 skip，不要扩大范围。

## 2. 步骤

1. 在 worktree 根目录全量重生成（不要手改生成文件）：
   ```bash
   python3 tools/gen_ref_tests.py --all --chunk-size 200
   git status --short test/unit/ref
   ```
   若生成器崩溃或产出明显非法代码，做**通用**修复后重新生成，不要针对单个函数打补丁。
2. 在隔离的 5090 目录（`MCPP_5090_DIR`，即 `/home/metax/mcMPP-ref-regen`）同步并构建 `ref_tests`：
   - 优先使用 `mcp__mcpp-5090__sync/build/test/record/pull`；MCP 不可用时用 `tools/remote5090.sh`。
   - 构建必须 `USE_NVIDIA_NPP=ON`（ref_tests 只链接 NVIDIA NPP）。
3. 先做小规模 smoke record（每模块 1–2 个函数），确认 harness 正常，再执行全量：
   ```bash
   NPP_RECORD_DIR=<remote>/build-nvidia/ref_records ./ref_tests --gtest_filter='RefSuite.*'
   ```
4. 对失败分类处理：
   - **NPP 崩溃 / 头文件声明但库未导出**：核对参数合法后，用 `test/unit/ref/overrides/<module>.json` 的 `skip_exact` 记录，报告中给出最小复现与证据（如 cuda-gdb 回溯、`nm -D`）。
   - **生成器错误**：修生成器（通用规则）→ 重新生成 → 重新 record；禁止手改生成文件。
   - 不允许为了过 record 而删除既有测试或放宽语义。
5. 迭代直到全量 `RefSuite.*`：**0 failed / 0 crash**（跳过项必须在 skip manifest 或 overrides 中可见）。
6. 拉回 golden 到本地 `artifacts/5090/ref_records/`（gitignored，不入库）。
7. 在 `ref/regen` 上提交：重生成的测试、skip manifest、overrides、生成器通用修复。**分批提交**，避免单次请求过大。
8. 最终报告：各模块测试数变化、新增/删除的测试、step/缓冲变化摘要、新增 overrides 及原因、仍无法自动化的 follow-up（planar 指针数组、batch 结构体等）、以及「重跑生成器后 `git status` 为空」的复现性证明。

## 3. Definition of Done

1. 统一生成器重生成后，全量 record 在 5090 上 **0 failed / 0 crash**。
2. 已提交全部重生成文件与必要 overrides；工作区干净。
3. 再次运行 `tools/gen_ref_tests.py --all --chunk-size 200` 不产生任何 diff（生成器与入库文件一一对应）。
4. golden 已拉回 `artifacts/5090/ref_records/`。
5. 报告完成。

## 4. 约束

- 不 push、不改集成分支、不 force push、不删除既有测试。
- 禁止运行 `cmake_maca`/`make_maca`（会删除 `~/cu-bridge`）。
- 只改本任务相关文件；生成文件不手改。
- 工具输出保持小（重定向到文件后 grep/tail），先提交再继续，防止网关请求大小限制丢工作。
