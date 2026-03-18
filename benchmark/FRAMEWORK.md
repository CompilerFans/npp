# Benchmark Framework

本文档只说明 `benchmark/` 的实现逻辑、目录分层和扩展方式，不再承担 quick start 或进度汇报职责。

## 1. 目标

`benchmark/` 解决的是同一批 benchmark 程序在两种后端上的统一运行问题：

- in-tree MPP
- NVIDIA NPP

核心目标：

- 一套 benchmark 源码，切换不同链接后端
- 固定的输出格式：console / CSV / JSON
- 未实现或未导出的 API 不崩溃，输出 `success=false`
- benchmark 结果可以继续进入 compare / progress 两层分析

## 2. 目录结构

```text
benchmark/
├── benchmark_base.h          # 框架核心：结果结构、计时、DeviceBuffer、注册表、dlsym
├── benchmark_main.cpp        # CLI 入口
├── benchmark_runner.cpp      # 执行与 CSV/JSON 输出
├── benchmark_utils.cu        # 通用 CUDA 初始化与设备信息
├── CMakeLists.txt            # benchmark 子工程配置
├── scripts/
│   ├── build_benchmark.sh    # 构建入口，对齐主工程 build / build-nvidia
│   ├── run_benchmark.sh      # 运行入口，默认结果落到 benchmark/results/
│   ├── compare_csv.py        # 两份 benchmark CSV 的对比层
│   ├── progress_report.py    # before / after / nvidia 的进展层
│   └── new_benchmark.sh      # 生成可编译 benchmark stub
├── nppcore/                  # nppcore benchmark
├── nppi/                     # nppi benchmark
└── npps/                     # npps benchmark
```

## 3. 构建与运行链路

### 3.1 构建

`scripts/build_benchmark.sh` 现在只保留两种构建模式：

- 默认：配置并构建 `build/benchmark/npp_benchmark`
- `--use-nvidia-npp`：配置并构建 `build-nvidia/benchmark/npp_benchmark`

这与主工程的目录口径保持一致，不再维护旧的独立构建目录约定或第三方分支变体。

### 3.2 运行

`scripts/run_benchmark.sh` 的职责只有三件事：

1. 确认对应的 benchmark 二进制是否存在
2. 必要时触发 `scripts/build_benchmark.sh`
3. 把过滤条件、输出路径和运行模式转发给 `npp_benchmark`

默认结果目录固定为：

```text
benchmark/results/
```

这样可以避免 benchmark 结果散落到仓库根目录。

### 3.3 benchmark 源码分层

`benchmark/` 里的 benchmark 源码只保留两层：

1. `nppi/`、`npps/`、`nppcore/` 下的手写 benchmark
2. `scripts/new_benchmark.sh` 生成的可编译骨架

其中：

- 手写 benchmark 负责“正式、定制化”的 benchmark 逻辑
- 生成器负责降低新增单个 API benchmark 的样板成本

当前版本明确不再追求“一次性批量接入全部 API”。框架目标是稳定主干、降低增量接入成本、固定结果格式。

### 3.4 默认运行参数

`benchmark` 的默认 warmup / measure iterations 在框架内部统一定义，脚本默认不再重复写死同一组值。

这意味着：

- 程序内默认值只有一处来源
- `scripts/run_benchmark.sh` 不再暴露 warmup / iterations 配置选项
- CSV 不再导出 `iterations` 一类运行配置字段

随机初始化也使用框架内部统一常量：

- `kBenchmarkRandomSeed` 定义在 `benchmark_base.h`
- host 侧随机输入应通过公共 deterministic helper 生成
- device 侧随机初始化 kernel 也使用固定种子

这样可以保证同一 benchmark 场景在不同时间重复运行时，输入分布保持一致，结果更容易复现和对比。

## 4. 框架核心

### 4.1 `BenchmarkBase`

所有 benchmark 类都继承 `BenchmarkBase`，统一实现：

```cpp
virtual std::vector<BenchmarkResult> run(int warmupIterations, int measureIterations) = 0;
```

一个 benchmark 类通常对应一个 API 或一组高度相关的 API 变体。

### 4.2 `BenchmarkResult`

统一输出字段包括：

- `functionName`
- `size`
- `variantTags`
- `dataType`
- `channels`
- `avgTimeMs`
- `throughputGBps`
- `success`
- `errorMessage`

其中：

- benchmark 主比较指标始终是 `avg_time_ms`
- `throughputGBps` 只是辅助指标
- `variantTags` 是独立场景维度，不再混入 `size`
- `success=false` 是框架级失败兜底，不应该导致整个进程崩溃

### 4.3 `BenchmarkRegistry`

通过 `REGISTER_BENCHMARK(...)` 静态注册 benchmark。

这意味着新增 benchmark 时：

- 不需要维护手工列表
- 只要新源文件被 CMake 收集到，注册类就会自动进入运行集合

### 4.4 `CudaTimer` 与 `runBatchedMeasurement`

计时方式是：

1. warmup 若干次
2. 用一对 CUDA event 包住一批 `measureIterations` 次调用
3. 再折算成单次 `avg_time_ms`

这样做的目的不是追求统计学意义上的方差分析，而是尽量减少极短 API 的单次调用噪声。

### 4.5 `DeviceBuffer`

`DeviceBuffer<T>` 负责：

- `cudaMalloc/cudaFree` 生命周期
- 分配失败检查
- fatal CUDA error 后的 best-effort reset

这让 benchmark 文件不用重复写大量样板内存管理逻辑。

对于需要随机数据的场景，`DeviceBuffer::fillRandom()` 现在也默认使用固定随机种子生成确定性输入，而不是进程级 `rand()`。

### 4.6 `DynamicSymbol`

`DynamicSymbol<Fn>` 使用 `dlsym` 在运行时查找函数符号。

用途是：

- benchmark 可以覆盖“已声明但当前后端未导出”的 API
- 对找不到符号的场景输出 `success=false` + `Missing symbol`
- 不因单个 API 缺失而影响整批 benchmark 执行

## 5. compare / progress 的数据口径

### 5.1 重复场景聚合

当输入 CSV 中同一场景出现多行成功结果时，`scripts/compare_csv.py` 和 `scripts/progress_report.py` 现在统一先做：

```text
mean(avg_time_ms)
```

不再取“最小一行”作为代表值。

场景键统一为：

```text
function_name + size + variant_tags + data_type + channels
```

这意味着输出更偏向稳定比较，而不是 best-case。

### 5.2 对标 NVIDIA 的主口径

compare 层统一使用：

```text
mpp_time_pct_of_nvidia = (mpp_avg_time_ms / nvidia_avg_time_ms) * 100
```

解释：

- `< 100%`：MPP 更快
- `= 100%`：打平
- `> 100%`：MPP 更慢

progress 层的 `Gap% (After/NVIDIA)` 与上述含义一致，只是它比较的是“优化后 MPP”相对 NVIDIA 的位置。

## 6. HTML 报告的扩展方式

### 6.1 compare 报告

当前 compare HTML 聚焦两层视角：

- `op family` 聚合后的 Top Catch-Up / Top Behind
- 单 API 的按 size 曲线

`op family` 的提取方式是：

- 去掉 `nppi` / `npps` 前缀
- 截取第一个 `_` 前的操作名

例如：

- `nppiResize_8u_C1R` -> `Resize`
- `nppiFilterBoxBorder_8u_C1R` -> `FilterBoxBorder`
- `nppsAdd_32f` -> `Add`

### 6.2 progress 报告

当前 progress HTML 聚焦：

- Top Improved Families
- Top Catch-Up Families

它回答的是两个不同问题：

- 哪些 op family 相比之前的 MPP 版本提升最大
- 哪些 op family 已经最接近 NVIDIA

## 7. 生成器逻辑

### 7.1 输入

`scripts/new_benchmark.sh` 现在至少接收：

- `type`
- `category`
- `header`
- `--functions`
- 可选 `--variant-tags`

### 7.2 输出

生成文件具备以下性质：

- 可直接编译
- 自动注册 benchmark
- 能在 `-l` 中出现
- 能在运行时输出结果
- 未实现的 body 默认返回 `success=false`
- 默认带 `benchmarkCases()`，特殊 API 只需在文件内覆盖 case 集合

### 7.3 为什么先生成 stub

因为只给 `header + function list + variant tags` 时，脚本无法可靠推断每个 API 的真实参数语义、scratch buffer、mask、插值模式、ROI 规则。

所以生成器的职责是：

- 先生成“结构正确”的 benchmark 骨架
- 再由开发者把 TODO stub 替换成真实 API 调用
- 若 body 需要随机输入，必须继续复用 `benchmark_base.h` 中的公共 deterministic 初始化 helper，不要在 benchmark 文件里单独写 `rand()`

这比旧版只生成 `YourFunction` 模板更有价值，因为它已经完成了：

- benchmark 名称落地
- 注册落地
- 结果字段落地
- `variant_tags` 独立字段落地

## 8. 扩展建议

当需要继续扩 benchmark 时，建议遵循以下顺序：

1. 先决定是“手写 benchmark”还是“生成 stub 后补全”
2. 明确该 API 的场景维度：size / dtype / channels / variant_tags / scratch
3. 先保证缺失实现时输出 `success=false`
4. 再补真实 API 调用和吞吐估算
5. 最后进入 compare / progress 做对标分析

### 8.1 如何借鉴独立 benchmark 原型

仓库里临时编写的单文件 benchmark 原型，适合拿来做“实现参考”，不适合拿来做“框架参考”。

适合借鉴的内容：

- 某个具体 API 的真实调用姿势
- packed / planar / pitch / ROI / scratch 这类参数组织方式
- 单个 API 的吞吐量估算公式
- 为了构造输入输出而需要的最小 buffer 布局

可以直接落成 benchmark 实现规范的细节：

- 若 API 是 packed 输入，例如 `C3` / `C4` 图像，pitch 通常按：

```text
srcStep = width * channels * sizeof(T)
```

- 若 API 是 planar 输入，例如 `P3` / `P4` 图像，每个 plane 的 pitch 通常按：

```text
planeStep = width * sizeof(T)
```

- 图像 ROI 建议统一先整理成：

```cpp
NppiSize roi{width, height};
```

再传给真实 NPP/MPP API，而不是把 `width` / `height` 零散地直接塞进调用点。

- packed -> planar / planar -> packed 这类 API，输入输出 buffer 组织建议先按“参数语义”建模：
  - packed 缓冲区：一块连续内存
  - planar 缓冲区：`T* planes[N]` 或 `const T* planes[N]`
  - scratch / temp buffer：单独建一个 buffer，不和输入输出混在一起

- 若前期只是验证 API 调用是否合理，可以先手工 `cudaMalloc` 摆正内存布局；正式迁入 `benchmark` 时，再统一替换成 `DeviceBuffer<T>`。

- 吞吐量建议按“本次 API 实际处理的总字节数”估算：

```text
throughput_gbps = processed_bytes / elapsed_seconds / 1e9
```

其中 `processed_bytes` 不要机械只算输入，也不要一律乘 2；应按 API 真实读写行为估算。  
例如 packed / planar copy 这类搬运型 API，通常至少要把输入读和输出写两部分都考虑清楚。

- 计时包围方式可以借鉴“批量调用后再折算单次耗时”的思路，但正式 benchmark 不建议继续使用原型程序里那种：
  - `cudaDeviceSynchronize()`
  - `std::chrono`
  - 失败即 `exit(1)`

在 `benchmark` 中应改成：

  - 用统一 warmup
  - 用 `CudaTimer` / batched measurement 计时
  - 失败写入 `success=false` 和 `error`
  - 结果回到统一 CSV schema

不建议直接照搬的内容：

- `CHECK_* + exit(1)` 这种失败即退出的控制流
- `main()` 中硬编码尺寸、iterations、输出方式
- 不接入统一结果结构、CSV schema、compare / progress 的独立打印逻辑

更合理的做法是：

1. 先从独立原型里提取 API 调用和数据组织方式
2. 再把这些逻辑写进 `benchmark/<type>/bench_*.cpp`
3. 让结果继续走统一 `BenchmarkResult` 和统一 CSV 字段

这样既能复用前期试验代码的价值，又不会把零散原型继续扩成第二套 benchmark 体系。

## 9. 文档边界

- Quick start 和稳定 CLI：见 [README.md](/local/m01019/mcmpp_perf/npp/benchmark/README.md)
- 架构与扩展：本文档
