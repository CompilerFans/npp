# NPP 实现层设计模式与复用规范

本文记录 mcMPP NPP 实现层（wrapper / executor / kernel / 测试）已经采用的模式、
当前与目标设计的落差，以及后续重构路线。新包的 subagent 应先读本文再动手，避免
另起平行框架。

## 1. 分层架构

```
API Surface（生成）        nppiXxx_8u_C1R[_Ctx]，extern "C"，C ABI
        ↓
Family Facade / Executor   namespace nppi::<module>，参数校验→workspace→launch→错误检查
        ↓
Algorithm Strategies       dtype / channel/layout / border / interpolation / reduction
        ↓
Kernels（模板）            1D/2D 逐像素、归约、滤波、几何
        ↓
Runtime Services           stream context、scratch、error mapping、buffer-size helper
```

完整目标见 `docs/npp_api_expansion_plan.md` §3。

## 2. 已落地的模式

| 模式 | 落点 | 说明 |
|---|---|---|
| Data-Driven Generation | `mpp_scripts/gen_family.py` + `mpp_scripts/families/*.json`（38 个 spec） | 从 catalog 生成全部 dtype/channel 变体 wrapper；生成文件带 `DO NOT EDIT` 头 |
| Adapter | 生成的 non-Ctx 入口 → `_Ctx` | 统一 `nppGetStreamContext(&ctx)` 后转发，约 4,500 个入口零手写 |
| Policy-Based Templates | `nppi_arithmetic_executor.h` 的 `binaryKernel<T,Channels,BinaryOp>`、`BinaryOperationExecutor<T,Channels,OpType>` 等 15+ 执行器 | op 以 functor 传入，同一 kernel 复用于 Add/Sub/Mul/Div/… |
| Template Method | `Executor::run()` 固定“校验→workspace→launch→错误检查”骨架 | 各家族状态码/ROI 语义一致 |
| Compile-time Strategy | `NormType {L1,L2,Inf}`、`AlphaPreserve`、`Channels` 模板参数 | 避免运行时 switch |
| 共享归约框架 | `nppi_statistics_executor.cu` 的 `reducePartialKernel` + `finalize*Kernel` | Sum/Mean/StdDev/Norm/NormDiff/NormRel/DotProd 共用两段归约、double 累加 |
| Host/Device 分离 | 模板声明在 `.h`，定义 + 显式实例化在 `.cu` | 普通 .cpp 不编译设备代码 |
| Differential Golden Testing | `test/unit/framework/npp_diff_harness.h` + `tools/gen_ref_tests.py` | 5090 record / 本地 verify；容差 kind 统一 |
| Object Mother | 生成测试的确定性输入（SHA-256 种子、ROI/pitch 变体） | 可复现、可重新 record |

## 3. 当前落差（截至 2026-10-29）

1. **没有公共工具层**：`getThreadCoords`、行指针、参数校验、`checkCudaKernelError`
   定义在 `nppi_arithmetic_executor.h` 内；statistics/npps 各自重复。
   `src/include/` 目前只有版本兼容头。
2. **文档承诺的具名 Policy 未落地**：`DTypePolicy<T>` / `ChannelPolicy<C>` /
   `BorderPolicy<B>` 仍是裸模板参数，无 concepts 约束。
3. **显式实例化手写**：`nppi_statistics_executor.cu:1695+` 有数百行
   `template NppStatus runX<...>`，与 wrapper 必须手工同步，漏项表现为链接错误。
4. **RAII 未实现**：文档中的 `DeviceBuffer`/`ScratchWorkspace` 不存在。API 边界的
   scratch 由调用方提供，但内部临时缓冲仍可用 RAII。
5. **Strategy+Registry 只做了一半**：buffer-size 是逐家族函数，没有注册表；
   `NppDataType`/`NppiChannels` 运行时派发（如 FusedAbsDiff）仍是每家族 switch。
6. **统计 executor 单文件 2,101 行**，聚合 Sum/Mean/StdDev/Min/Max/Indx/Norm/
   NormRel/DotProd；归约核心与家族代码混在同一个匿名 namespace。
7. **wrapper 生成器与参考测试生成器各自独立**，只共享 API catalog。

## 4. 重构路线

### P0.1 — 公共 kernel 工具 + 归约核心

- 新建 `src/include/npp_kernel_utils.h`：`BlockConfig`、`getThreadCoords`、
  `getSrcRow/getDstRow`、`checkCudaKernelError`、step/ROI 校验模板。
- 新建 `src/include/npp_reduction.cuh`：把 `nppi_statistics_executor.cu` 的两段归约
  核心（partial/finalize、double 累加）抽成可复用头，statistics 首先迁入。
- 迁移 arithmetic 的本地工具到头文件；**行为零变化**。
- 验收：4,803 全量 unit + 1,613 全量 diff 全绿；无新增告警。

### P0.2 — 生成实例化 + 具名 Policy + 运行时派发

- `gen_family.py --emit-instantiations`：从 spec 生成 `.cu` 显式实例化清单（`.inc`）。
- 具名 policy 类型（可先 alias）：`DTypePolicy`、`ChannelPolicy`、`LayoutPolicy`；
  如工具链支持再引入 C++20 concepts。
- 运行时派发用小型注册表/函数指针表替换家族 switch。
- 验收同上，且生成物可复现。

### 后续

- p1.3c/p1.3d（Average/Maximum 误差）必须复用 P0.1 的归约核心；
- P1.9（滤波）落地时引入 `BorderPolicy` / `InterpolationStrategy`；
- 参考测试生成器与 wrapper 生成器共享 API 解析层（远期）。

## 5. 新包开发规范

1. 先找同模块 2 个最近邻实现，复用其 executor/校验/测试模式；**不新建平行框架**。
2. 公共组件只放 `src/include/`；不要复制一份到模块目录。
3. 生成文件不手改；改 spec 后重新生成并入库。
4. 行为以 5090 NPP 13.4 差分基线为准；文档语义与实测冲突时以实测为准，并在报告里记录。
5. 新增归约/误差度量家族：复用两段归约；零分母、通道归一化等语义必须用探针钉死。
6. 工具输出保持小，分批提交；不 push、不改集成分支、禁止 `cmake_maca`/`make_maca`。
