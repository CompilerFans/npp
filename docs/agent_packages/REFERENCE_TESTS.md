# 参考测试子 agent — 官方 NPP 全量单元测试

你是 mcMPP 的**参考测试开发**子 agent。目标：为分配的 NPP 模块生成可编译、可在 5090 官方 NPP 13.4 上 **record 全绿**的参考单元测试。参考测试是后续实现对齐的验收基线；本包**不要求** MPP 实现通过（mismatch 汇总报告即可）。

## 0. 必读

1. `docs/mcp_5090.md`（5090 MCP 工具）
2. `tools/gen_ref_tests.py`（测试生成器）
3. `test/unit/ref/ref_test_utils.h`（确定性 buffer / 输出捕获）
4. `test/unit/ref/test_npps_arithmetic_and_logical_operations_*_generated.cpp`（试点样例）
5. `docs/npp_api_expansion_plan.md`、`docs/mpp_api_implementation_guide.md`

## 1. 你的环境

- 独立 worktree：`~/worktrees/mcMPP-ref-<id>`，分支 `ref/<id>`。
- 独立远端目录：环境变量 `MCPP_5090_DIR` 已指向 `/home/metax/mcMPP-ref-<id>`；**只用这个目录**，不要碰共享的 `~/mcMPP`。
- MCP 工具：`mcp__mcpp-5090__sync/build/test/record/pull/status/run`；默认 remote_dir 即你的独立目录。
- 5090 构建请用 `jobs=4`，不要在本地 `./build.sh`（本地 CMake 4.4 无法配置 cu-bridge）。

## 2. 工作流

1. 对每个分配模块生成测试：

   ```bash
   python3 tools/gen_ref_tests.py --module <header.h> --chunk-size 200
   ```

   产物：`test/unit/ref/test_<module>_<n>_generated.cpp`、`skipped_<module>.json`。
   生成器跳过的不支持签名会写入 skip 清单；不要手改生成文件。

2. 同步并构建远端参考测试：

   ```
   mcp__mcpp-5090__sync   {}
   mcp__mcpp-5090__build  {"jobs":4, "timeout":1800}
   ```

3. 录制：

   ```
   mcp__mcpp-5090__record {"filter":"RefSuite.*",
                           "out_dir":"<MCPP_5090_DIR>/build-nvidia/ref_records",
                           "timeout":1800}
   ```

   `out_dir` 用绝对路径（`/home/metax/mcMPP-ref-<id>/build-nvidia/ref_records`）。

4. 若编译失败/录制崩溃：
   - 编译错误优先用**模块 override** 修复：`test/unit/ref/overrides/<module_stem>.json`
     ```json
     {
       "skip_exact": ["nppiFoo_8u_C1R_Ctx"],
       "skip_contains": ["Batch"],
       "arg_overrides": {"nppiBar_8u_C1R_Ctx": {"nBorder": "NPP_BORDER_REPLICATE"}}
     }
     ```
     再重新生成、构建。
   - 录制崩溃：用 `mcp__mcpp-5090__test` 或 `record` 的 `filter` 二分定位到具体 `RefSuite.<fn>`，把该函数加入 `skip_exact`（并在报告中注明原因）。
   - 只有**通用**生成器缺陷才改 `tools/gen_ref_tests.py`；保持改动小而通用，并在报告里列明（其他 agent 也在用这个文件）。

5. 验收：5090 上 `ref_tests` record 运行 **EXIT=0、0 failed、0 crash**。允许合理 skip，但要把 skip 清单和原因报告清楚。

6. 把生成文件、skip 清单、overrides 提交到 `ref/<id>`。不要提交 `artifacts/`，不要 push。

## 3. 可选：MPP 差异

`record` 完成后可拉回 golden 并本地 verify（可选，不阻塞）：

```
mcp__mcpp-5090__pull {"src":"/home/metax/mcMPP-ref-<id>/build-nvidia/ref_records",
                      "dest":"<worktree>/artifacts/5090/ref_records"}
NPP_VERIFY_DIR=artifacts/5090/ref_records LD_LIBRARY_PATH=build/src ./build/ref_tests --gtest_filter='RefSuite.*'
```

把 mismatch 的函数清单汇总到报告，供后续实现包处理。

## 4. 约束

- 只改本包模块的生成文件/override；不改集成分支、不 push、不运行 `cmake_maca`/`make_maca`。
- 不删除试点生成文件；不修改 `test/unit/ref/ref_test_utils.h` 的语义（需要改先报告）。
- 生成文件必须能被 `tools/gen_ref_tests.py` 从模块 + override 重现；override 文件入库。

## 5. 报告

- 模块、生成测试数、skip 数（按原因分类）、record 结果（pass/fail/crash）。
- 崩溃/跳过的具体函数与原因。
- 对生成器的通用修复（如有）。
- 可选 MPP verify 的 mismatch 清单。
