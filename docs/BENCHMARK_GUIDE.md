# NPP 性能测试快速开始指南

## 📝 概述

本指南帮助你快速上手 NPP 项目的性能测试框架。

## 🎯 性能测试的目标

1. **测量性能指标**：延迟、吞吐量、带宽利用率
2. **对比基准**：与 NVIDIA NPP 官方实现对比
3. **性能回归检测**：确保优化不会降低性能
4. **优化指导**：识别性能瓶颈

## 🚀 第一次使用

### 步骤 1：安装 Google Benchmark

有三种方式：

#### 方式 A：自动下载（最简单，推荐）

```bash
cd /Users/jiaozihan/Desktop/MPP/npp
mkdir -p build && cd build
cmake .. -DBUILD_BENCHMARKS=ON
make -j$(nproc)
```

CMake 会自动下载并编译 Google Benchmark。

#### 方式 B：手动克隆到项目

```bash
cd /Users/jiaozihan/Desktop/MPP/npp
mkdir -p third_party
cd third_party
git clone https://github.com/google/benchmark.git
cd benchmark
git checkout v1.8.3

# 回到项目根目录构建
cd ../..
mkdir -p build && cd build
cmake .. -DBUILD_BENCHMARKS=ON
make -j$(nproc)
```

#### 方式 C：系统安装

```bash
# macOS
brew install google-benchmark

# Ubuntu/Debian
sudo apt-get install libbenchmark-dev
```

### 步骤 2：运行第一个性能测试

```bash
cd /Users/jiaozihan/Desktop/MPP/npp/build/benchmark

# 运行算术运算性能测试
./nppi_arithmetic_benchmark

# 预期输出：
# Running ./nppi_arithmetic_benchmark
# Run on (16 X 3000 MHz CPU s)
# CPU Caches:
#   L1 Data 32 KiB (x8)
#   L1 Instruction 32 KiB (x8)
#   L2 Unified 256 KiB (x8)
#   L3 Unified 16384 KiB (x1)
# -------------------------------------------------------------------
# Benchmark                         Time             CPU   Iterations
# -------------------------------------------------------------------
# BM_nppiAdd_8u_C1RSfs_Fixed    0.234 ms        0.234 ms         2987
# BM_nppiAdd_32f_C1R/1920/1080  0.512 ms        0.512 ms         1367
# ...
```

### 步骤 3：对比 MPP 和 NVIDIA NPP

```bash
cd /Users/jiaozihan/Desktop/MPP/npp/test/benchmark

# 给脚本执行权限
chmod +x run_comparison.sh

# 运行对比测试（需要 15-30 分钟）
./run_comparison.sh

# 输出示例：
# === NPP Performance Benchmark Comparison ===
# 
# Step 1: Building and benchmarking MPP implementation...
# [Build output...]
# ✓ MPP benchmarks completed
# 
# Step 2: Building and benchmarking NVIDIA NPP implementation...
# [Build output...]
# ✓ NVIDIA NPP benchmarks completed
# 
# === Quick Summary ===
# Performance comparison (sample):
#   nppiAdd_8u_C1RSfs (1920x1080):
#     MPP:        0.245 ms
#     NVIDIA NPP: 0.238 ms
#     Speedup:    0.97x
```

## 📊 理解输出结果

### 基本输出格式

```
BM_nppiAdd_8u_C1RSfs_Fixed    0.234 ms    0.234 ms    2987    53.7M/s
│                             │           │           │       │
│                             │           │           │       └─ 数据吞吐量
│                             │           │           └───────── 迭代次数
│                             │           └───────────────────── CPU 时间
│                             └───────────────────────────────── 实际时间
└─────────────────────────────────────────────────────────────── 测试名称
```

### 性能指标解读

| 指标 | 说明 | 如何优化 |
|------|------|----------|
| **Time (ms)** | 实际运行时间 | • 减少全局内存访问<br/>• 使用共享内存<br/>• 优化线程配置 |
| **Bytes/s** | 数据吞吐量 | • 合并内存访问<br/>• 减少bank冲突<br/>• 提高占用率 |
| **Bandwidth %** | 带宽利用率 | 应该 > 60% |

### 性能评估标准

```
相对于 NVIDIA NPP：

🟢 优秀:     > 95%  性能
🟡 良好:     80-95% 性能
🟠 可接受:   60-80% 性能
🔴 需优化:   < 60%  性能
```

## 🎓 常用命令

### 1. 运行特定测试

```bash
# 只测试 Add 操作
./nppi_arithmetic_benchmark --benchmark_filter=Add

# 只测试 8u 类型
./nppi_arithmetic_benchmark --benchmark_filter=8u

# 只测试 Full HD 尺寸
./nppi_arithmetic_benchmark --benchmark_filter=1920x1080
```

### 2. 导出结果

```bash
# 导出 JSON 格式
./nppi_arithmetic_benchmark \
    --benchmark_out=results.json \
    --benchmark_out_format=json

# 导出 CSV 格式（方便 Excel 分析）
./nppi_arithmetic_benchmark \
    --benchmark_out=results.csv \
    --benchmark_out_format=csv
```

### 3. 提高测试准确性

```bash
# 每个测试重复 10 次
./nppi_arithmetic_benchmark --benchmark_repetitions=10

# 只显示统计结果（均值、中位数、标准差）
./nppi_arithmetic_benchmark \
    --benchmark_repetitions=10 \
    --benchmark_report_aggregates_only=true
```

### 4. 快速测试（CI 环境）

```bash
# 最小运行时间 0.1 秒（默认是自适应）
./nppi_arithmetic_benchmark --benchmark_min_time=0.1
```

## 📝 编写自己的性能测试

### 模板：简单测试

创建文件：`test/benchmark/nppi/my_module/benchmark_my_func.cpp`

```cpp
#include "benchmark_base.h"
#include <nppi_xxx.h>  // 你的头文件

using namespace npp_benchmark;

static void BM_MyFunc_8u_C1R(benchmark::State& state) {
    // 1. 准备测试数据
    int width = 1920, height = 1080;
    ImageBenchmarkBase<Npp8u> base;
    base.SetupImageMemory(width, height);
    
    NppiSize roi = {width, height};
    
    // 2. 性能测试主循环
    for (auto _ : state) {
        // 调用你的函数
        nppiMyFunc_8u_C1R(
            base.d_src1_, base.step_,
            base.d_dst_, base.step_,
            roi);
        
        // 同步 GPU
        base.SyncAndCheckError();
    }
    
    // 3. 报告性能指标
    size_t bytes = base.ComputeImageBytes(1, 1);  // 1 输入, 1 输出
    REPORT_THROUGHPUT(state, bytes);
    
    // 4. 清理
    base.TeardownImageMemory();
}

// 注册测试
BENCHMARK(BM_MyFunc_8u_C1R)->UseRealTime();
```

### 模板：参数化测试

```cpp
static void BM_MyFunc_Sizes(benchmark::State& state) {
    // 从参数获取尺寸
    int width = state.range(0);
    int height = state.range(1);
    
    ImageBenchmarkBase<Npp8u> base;
    base.SetupImageMemory(width, height);
    
    NppiSize roi = {width, height};
    
    for (auto _ : state) {
        nppiMyFunc_8u_C1R(base.d_src1_, base.step_,
                          base.d_dst_, base.step_, roi);
        base.SyncAndCheckError();
    }
    
    size_t bytes = base.ComputeImageBytes(1, 1);
    REPORT_THROUGHPUT(state, bytes);
    REPORT_CUSTOM_METRIC(state, "Megapixels", (width * height) / 1e6);
    
    base.TeardownImageMemory();
}

// 注册多个尺寸
BENCHMARK(BM_MyFunc_Sizes)
    ->Args({640, 480})        // VGA
    ->Args({1280, 720})       // HD
    ->Args({1920, 1080})      // Full HD
    ->Args({3840, 2160})      // 4K
    ->Unit(benchmark::kMillisecond)
    ->UseRealTime();
```

### 添加到构建系统

编辑 `test/benchmark/CMakeLists.txt`：

```cmake
# 你的新测试
set(MY_MODULE_BENCHMARK_SOURCES
    nppi/my_module/benchmark_my_func.cpp
)

npp_create_benchmark_target(
    my_module_benchmark
    "${MY_MODULE_BENCHMARK_SOURCES}"
    npp_nppi_lib  # 或你的库目标
)

target_include_directories(my_module_benchmark
    PRIVATE
    ${BENCHMARK_INCLUDE_DIRS}
)
```

## 🐛 常见问题

### Q1: 编译错误 "benchmark.h not found"

```bash
# 确保启用了 BENCHMARK
cmake .. -DBUILD_BENCHMARKS=ON

# 或清理重新构建
rm -rf build
mkdir build && cd build
cmake .. -DBUILD_BENCHMARKS=ON
make -j$(nproc)
```

### Q2: 运行时 CUDA 错误

```bash
# 检查 GPU 状态
nvidia-smi

# 确保没有其他程序占用 GPU
# 设置使用特定 GPU
export CUDA_VISIBLE_DEVICES=0
```

### Q3: 性能结果波动很大

```bash
# 增加重复次数
./benchmark --benchmark_repetitions=20

# 禁用 CPU 频率缩放（需要 sudo）
echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor

# 关闭其他应用程序
# 使用专用的性能测试机器
```

### Q4: 想要更详细的 GPU 性能分析

```bash
# 使用 NVIDIA Nsight Compute 分析
ncu --set full -o profile ./nppi_arithmetic_benchmark --benchmark_filter=Add

# 使用 NVIDIA Nsight Systems 查看时间线
nsys profile -o timeline ./nppi_arithmetic_benchmark
```

## 📈 性能优化工作流

```
1. 编写功能测试 → 确保正确性
   ↓
2. 编写性能测试 → 建立基准
   ↓
3. 分析性能瓶颈 → Nsight Compute/Systems
   ↓
4. 优化代码 → 改进 kernel
   ↓
5. 运行性能测试 → 验证改进
   ↓
6. 对比 NVIDIA NPP → 评估差距
   ↓
7. 迭代优化
```

## 📚 进阶主题

### 性能剖析

```bash
# 使用 Nsight Compute
ncu --set full \
    --section SpeedOfLight \
    --section MemoryWorkloadAnalysis \
    -o my_profile \
    ./nppi_arithmetic_benchmark --benchmark_filter=Add

# 分析报告
ncu-ui my_profile.ncu-rep
```

### CI 集成

```yaml
# .github/workflows/benchmark.yml
name: Performance Regression
on: [push, pull_request]

jobs:
  benchmark:
    runs-on: ubuntu-latest-gpu
    steps:
      - uses: actions/checkout@v2
      - name: Build benchmarks
        run: |
          mkdir build && cd build
          cmake .. -DBUILD_BENCHMARKS=ON
          make -j$(nproc)
      - name: Run benchmarks
        run: |
          cd build/benchmark
          ./nppi_arithmetic_benchmark \
            --benchmark_out=results.json \
            --benchmark_out_format=json
      - name: Compare with baseline
        run: |
          python3 tools/compare_benchmark.py \
            baseline.json results.json
```

## ✅ 检查清单

完成以下任务：

- [ ] 成功编译性能测试
- [ ] 运行第一个性能测试
- [ ] 理解输出结果
- [ ] 与 NVIDIA NPP 对比
- [ ] 编写自己的性能测试
- [ ] 识别性能瓶颈
- [ ] 优化并验证改进

## 🤝 需要帮助？

- 查看 `test/benchmark/README.md` 了解更多细节
- 参考 `test/benchmark/nppi/arithmetic/benchmark_nppi_add.cpp` 示例
- 阅读 [Google Benchmark 文档](https://github.com/google/benchmark)

## 📊 性能目标参考

| GPU | Add_8u (Full HD) | FilterBox_8u (3x3) | Resize (bilinear) |
|-----|------------------|---------------------|-------------------|
| **RTX 3090** | < 0.3 ms | < 2.0 ms | < 5.0 ms |
| **A100** | < 0.2 ms | < 1.5 ms | < 3.5 ms |
| **V100** | < 0.4 ms | < 2.5 ms | < 6.0 ms |

目标：达到 NVIDIA NPP 性能的 90% 以上。
