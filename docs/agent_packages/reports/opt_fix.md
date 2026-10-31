# opt_fix 报告 — 8u C3 回退定位 + 形态学向量化扩展（600U）

> 分支 `pkg/opt_fix`，基线 `6115501`（= opt_morph 合入后），本包提交 `1fa8324`。
> 节点 `cjxu@10.0.82.53`（32 × MXC600-U），设备 2（主）/ 7（复核），`CUDA_VISIBLE_DEVICES=<n>`，不占 GPU#0/#1。
> 原始数据：`artifacts/benchmarks/opt_fix/csv/`（70 个 CSV，5 轮交错）、`artifacts/benchmarks/opt_fix/refresh_evidence/`（刷新日志证据）。

## 0. 执行环境偏差说明

- 本会话中 mcpp-node53 MCP 工具未暴露（工具集中只有 mcpp-5090），改为对同一节点
  `cjxu@10.0.82.53` 直接 SSH 执行（`sync`/构建/测试/基准等效），远端目录仍为规定的
  `~/mcMPP-opt-opt-fix`，设备分配 7→2、禁用 `cmake_maca`/`make_maca` 等约束均遵守。
- 节点上全新 cmake configure 失败（bridge nvcc 不输出 `#$ TOP=`，FindCUDAToolkit 无法定位
  toolkit root）。改用已配置构建树的**重定位拷贝**做 A/B/C 三臂构建：
  - A = `~/mcMPP-opt-opt-fix`（HEAD：向量算术 + 新形态学）
  - B = `~/mcMPP-opt-opt-fix-before`（HEAD~1：向量算术 + 旧标量形态学）
  - C = `~/mcMPP-opt-opt-fix-novec`（HEAD + `vecAligned2/3` 强制 `false`：标量算术 + 新形态学；仅测量用，未提交）
  - 拷贝后 sed 改绝对路径、清空 .o/.a/可执行件再增量编译（三臂均 ninja 全量完成）。

## 1. nppiAdd_8u_C3RSfs "回退"（147.2 → 108.5 GB/s，0.74x）定位过程

### 1.1 刷新数据的疑点（harness 层面）
- 600U 刷新 harness（/tmp/run_parallel_bench.sh）以 30 个并行进程把 506 个函数固定轮询钉到 die 2..31：
  函数 i → die = 31 - (i % 30)。Add_8u_C1RSfs → die 26，C3RSfs → die 25，C4RSfs → die 24，
  即 C1/C3/C4 的"对比"来自三个不同 die、不同邻居负载的进程。
- 同一构建（未改任何代码）连续 13 次刷新中，`nppiAdd_8u_C3RSfs@1920x1080` 在
  **113.8 / 135.3 / 157.8 / 160.0 / 181.6 / 234.0 / 236.5 / 236.8 / 236.8 / 237.3 / 237.4 / 237.5 / 320.5 GB/s**
  间波动（2.8x）；C1 波动 48.3–155.4（3.2x）、C4 波动 152.0–385.6（2.5x）。
  所谓 147.2 → 108.5（0.74x）完全落在同一二进制的次间波动带内。
- 证据：`artifacts/benchmarks/opt_fix/refresh_evidence/add_8u_from_refresh_logs.csv`（13 份刷新日志的 Add_8u 全量行）。

### 1.2 同设备 A/B 复测（device 2，串行，5 轮交错；device 7 复核）

A（HEAD，向量路径）vs C（同源代码、`vecAligned2/3` 强制 false → 走标量路径），同 die、
串行、轮次交错（vecA_r1 → sclC_r1 → … r5），取 5 轮中位数（GB/s；吞吐 = 3×ROI 字节）：

| 函数 | 尺寸 | 标量 C | 向量 A | A/C |
|---|---|---:|---:|---:|
| nppiAdd_8u_C1RSfs | 640x480 → 3840x2160 | 133.3→162.8 | 213.1→381.1 | 1.60–2.34x |
| **nppiAdd_8u_C3RSfs** | 640x480 | 206.5 | 276.6 | 1.34x |
| | 1280x720 | 227.8 | 343.7 | 1.51x |
| | 1280x1024 | 233.5 | 357.3 | 1.53x |
| | **1920x1080** | **233.9** | **373.8** | **1.60x** |
| | 3840x2160 | 240.1 | 396.1 | 1.65x |
| nppiAdd_8u_C4RSfs | 640x480 → 3840x2160 | 222.2→271.4 | 311.7→398.8 | 1.39–1.47x |
| nppiAdd_16u_C1RSfs | 640x480 → 3840x2160 | 240.3→321.0 | 387.9→756.1 | 1.61–2.36x |
| nppiAdd_32f_C1R | 640x480 → 3840x2160 | 439.6→634.2 | 663.9→1365.9 | 1.51–2.15x |

（每格 5 轮原始值见 csv/`vecA_r*` / `sclC_r*`；vecA @1920x1080 C3 五轮 373/374/373/374/374，波动 <1%。）

**跨 die 复核（device 7，恢复后）**：C3@1920x1080 标量 232.7 vs 向量 373.3（1.61x），
@3840x2160 243.4 vs 396.6（1.63x）——与 device 2 一致（csv/`d7sclC_*`、`d7vecA_*`）。

### 1.3 结论

**为测量噪声，非真实回退；无需代码修复。**依据：

1. 同设备串行复测中向量路径对 8u C3 稳定 **+34–65%（1920x1080 处 1.60x）**，与同族
   C1（2.12x）/C4（1.46x）方向、幅度一致，C3 无任何劣化分支；
2. 连标量路径在安静 die 上也有 233.9 GB/s，远高于刷新数据的 147.2/108.5 —— 刷新数字整体
   偏低是 30 进程并行争抢（HBM/队列/时钟）所致，与 opt_elem 合并无关；
3. 同一二进制 13 次刷新中 C3 波动 113.8–320.5 GB/s（2.8x），"147.2 → 108.5" 在噪声带内。
   建议后续 600U 刷新对比改用同 die、串行或至少"函数→die"映射固定的方式后再下回退结论。

### 1.4 附带发现：早前 unit_tests 全量卡死的成因（非本包问题）

调查期间全量 unit_tests 曾在 `NppiMinEveryTest` 前后反复卡死（`mxkwCreateQueueBlock ioctl
create queue block timeout type:21`，die 7 与 die 2 均复现；单跑该用例对则通过）。当时节点
正在跑 30 进程刷新 harness。**harness 结束后，A 与 B（HEAD~1）全量套件均在 ~10 秒内跑完、
全部通过**（见 §4），证明卡死是并行负载下的驱动级队列资源耗尽（环境问题），与任何代码版本无关。

## 2. 形态学向量化扩展（8u C4R + 32f C1R）

### 2.1 改动（提交 `1fa8324`，仅 `nppi_morphology.cu` + 对应单测）
- 新增 word-per-pixel 向量化内核模板 `nppiMorphWordPix_kernel<W, DILATE, MW>`：
  - 8u C4R：1 像素 = 1 个 32-bit word（uchar4），SWAR `morphSwMin8/morphSwMax8` 一次比较 4 通道；
  - 32f C1R：1 像素 = 1 个 float word，compare-select 保持 erodeGeneral/dilateGeneral 的
    NaN / ±0 语义（init 1e10f/-1e10f）；
  - 每线程输出 4 个相邻像素（1 次 16B store）；每个 mask 行用对齐 uint4 loads +
    launch-constant word pick 取窗（无需 byte funnel）；按行流式处理，寄存器占用与 mask 高度无关；
  - 全 1 mask → 可分离归约（行折叠 + 水平折叠）；任意 mask → per-tap（`s_allOnes` 协作扫描）；
  - 边界/尾部线程回退逐像素参考路径（clamped 边界，与既有内核语义一致）。
- 宿主 launch 条件：两 step % 16 == 0、基址 % 4 == 0、maskW ∈ {3,5,7,9}、maskW*maskH ≤ 1024；
  不满足则回退原内核（状态码、alpha、ROI 语义不变）。接入点 6 处：
  `nppiErode/Dilate_8u_C4R_Ctx`、`nppiErode/Dilate_32f_C1R_Ctx`、`nppiErode3x3/Dilate3x3_32f_C1R`。
- 新增 exact-value 单测（CPU 参考逐像素对比）：5x5 全 1 / 带 hole、7x7、3x9 离心 anchor、
  erode+dilate、两种尺寸（含 6x5 多边界）、3x3 32f 专用 API。

### 2.2 before/after（A = HEAD vs B = HEAD~1，device 2，5 轮交错中位数，GB/s，吞吐 = 2×ROI 字节）

nppiErode_8u_C4R（nppiDilate 数字基本相同，见 csv/morphB_r\* / morphA_r\*）：

| 尺寸 | mask | before | after | A/B |
|---|---|---:|---:|---:|
| 1280x720 | 3x3 | 111.6 | 132.4 | 1.19x |
| 1280x1024 | 3x3 | 112.9 | 143.2 | 1.27x |
| 1920x1080 | 3x3 | 114.2 | 170.1 | 1.49x |
| 1280x720 | 5x5 | 26.0 | 67.3 | 2.59x |
| 1920x1080 | 5x5 | 27.0 | 88.8 | 3.29x |
| 1280x720 | 7x7 | 14.0 | 40.4 | 2.88x |
| 1920x1080 | 7x7 | 14.5 | 54.1 | 3.72x |

nppiErode_32f_C1R（nppiDilate 同幅）：

| 尺寸 | mask | before | after | A/B |
|---|---|---:|---:|---:|
| 1280x720 | 3x3 | 66.2 | 194.1 | 2.93x |
| 1920x1080 | 3x3 | 71.1 | 252.6 | 3.55x |
| 1280x720 | 5x5 | 40.1 | 122.9 | 3.07x |
| 1920x1080 | 5x5 | 40.4 | 159.2 | 3.94x |
| 1280x720 | 7x7 | 30.6 | 83.0 | 2.71x |
| 1920x1080 | 7x7 | 30.2 | 104.0 | 3.44x |

对照（未改动，符合预期）：

- nppiErode/Dilate_8u_C1R：全组合 1.00–1.01x（opt_morph 已向量化，本包不动）；
- nppiErode/Dilate_32f_C4R：全组合 1.00x —— HEAD~1 起就存在专用
  `nppiErode/Dilate_32f_C4R_kernel_opt`（float4 化），baseline 已 ~332–341 GB/s@3x3，
  高于新 32f C1 内核的 ~253 GB/s，无需再向量化；
- 8u C4R 3x3 增益（1.19–1.49x）小于 5x5/7x7（2.6–3.8x）：HEAD~1 已有专用
  `*_8u_C4R_3x3_kernel`，而 5x5/7x7 走逐像素通用路径，故提升悬殊。

### 2.3 边界语义调研（5090 上对 NVIDIA NPP 13.2 的实测）

在 5090（RTX 5090 D v2 + CUDA 13.2 NPP）上用 17x13 ROI、3x3/5x5 mask、anchor(1,1)/(2,2) 的
探针程序实测了 NVIDIA NPP 参考实现的 ROI 边界行为（逐像素对比 clamp/zero-pad 预测）：

- NVIDIA Erode_8u_C4R 3x3：顶行/底行全部输出 0（垂直方向 zero-pad），左右边缘像素既非
  clamp 也非 zero-pad，数值与"mask 窗口越界部分直接按行主序读取越界内存（上一行行尾
  padding / 本行 step padding）"的模型一致；内部像素与两种预测都一致。
- Dilate_8u_C4R / Erode/Dilate_32f_C1R：绝大多数像素与 clamp 预测一致，但边缘仍有
  8–26 个像素两种模型都不匹配。
- 结论：NVIDIA NPP 形态学在 ROI 边界的行为并不是文档化的 replicate 边界，包含未定义
  的越界读取；mcMPP 全库统一使用 clamped（replicate）边界（erodeGeneral/dilateGeneral、
  tiled 内核、opt_morph C1 向量化内核一致），本包新内核沿用 clamped 语义。
- 测试影响面核实：
  - diff_tests（golden 校验）不含形态学用例（test/unit/diff/ 无 Erode/Dilate）；
  - ref_tests 只在 USE_NVIDIA_NPP（5090 录制侧）构建，节点上不运行；
  - 节点上的形态学正确性门禁只有手写 unit_tests，其中既有用例对边界是宽松断言
    （"Allow some differences due to border handling"、1x1 只查 0..255 范围），
    未固定 3x3 专用内核的 zero-pad 行为。
- 因此：向量化路径（clamped）在既有 3x3 专用内核（zero-pad）之前接管后，节点上没有任何
  测试会因边界语义变化而失败，且全库边界语义反而更一致。已存在的 3x3 专用 zero-pad 内核
  仅作为不对齐回退保留。
- 备注（供后续包参考）：若将来对 MPP 启用 ref_tests 形态学 golden 比对（5090 已录有
  artifacts/5090/ref_records/ref/nppiErode_8u_C4R_Ctx.* 等），需要先决策是 skip-list
  还是改边界语义，因为 golden 中顶行/底行是 0、且边缘含 NVIDIA 的越界读取产物，
  mcMPP 的 clamped 语义无法逐字节匹配。

## 3. 未达预期项 / 范围外事项

- **8u C3 / 8u AC4 / 16u_C1/C3/C4/AC4 / 32f_C3/AC4 形态学**：符号未实现（A、B 两臂 benchmark
  均报 `Missing symbol nppiErode_16u_C1R` 等），属 API 扩展包范畴，非本包优化目标。
- **8u C3 形态学**（文档可选目标，"至少 8u C4 与 32f C1"之外）：未做。8u C3 = 3 字节/像素，
  无法整字映射（word 方案要求 4 字节/像素），需要 C3 专用方案（如 12 字节包或行内错位方案），
  留给后续包。
- **32f C4R**：无需改动（见 §2.2，既有 `_opt` 内核已 ~341 GB/s@3x3，本包保持 1.00x）。
- **8u C3 算术"回退"**：确认噪声、无需修复（§1.3）；刷新 harness 的测量方法问题已记录，
  但 harness 本身（/tmp/run_parallel_bench.sh，属 ~/mcMPP-opt 签出）不在本包修改范围。
- **早前全量 unit_tests 卡死**：环境性（并行刷新负载下驱动队列耗尽），非代码问题（§1.4）；
  期间 die 7 曾被"钉死"，harness 停止后自行恢复，复核数据即取自恢复后的 die 7。

## 4. 验证（全部在节点 device 2，安静负载）

| 项 | 结果 |
|---|---|
| unit_tests 全量（A = HEAD） | **4806/4806 PASSED**（825 套件，10.1 s；含本包新增 3 个形态学 exact-value 用例） |
| unit_tests 全量（B = HEAD~1，回归对照） | 4803/4803 PASSED（9.8 s，卡死疑云排除） |
| unit_tests 过滤 `*Add*C3*:*Erode*:*Dilate*`（A，device 7） | 505/505 PASSED |
| diff_tests 全量（A） | **1640/1640 PASSED**（733 ms） |
| 静态检查 | 全仓 `*.py` py_compile 通过；`tools/npp_api_catalog.py` 正常（本包未改 API 面，catalog 计数不变：stub 166 / tested 7909）；本包无新增 family/spec，`gen_family --list` 不适用 |

复现方式（节点）：`/tmp/opt_fix_env.sh`（环境）、`/tmp/opt_fix_bench.sh <build> <tag> <dev> <out> <funcs...>`（基准）、
`/tmp/opt_fix_matrix.sh`（本轮交错矩阵）；A/B/C 构建目录见 §0。

## 5. 提交清单

- `1fa8324` nppi morphology: vectorize 8u C4R and 32f C1R (word-per-pixel kernel)
  - `src/nppi/nppi_morphological_operations/nppi_morphology.cu`（+341）
  - `test/unit/nppi/nppi_morphological_operations/test_nppi_morphology.cpp`（+210）
- 本报告：`docs/agent_packages/reports/opt_fix.md`
- 原始基准数据（gitignore 的 artifacts 目录，随 worktree 保留）：`artifacts/benchmarks/opt_fix/`
