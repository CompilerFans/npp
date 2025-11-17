# NPP 性能测试框架

本目录包含 NPP 库的性能基准测试（Benchmarks），用于测量和对比性能指标。

## 📋 目录结构

```
benchmark/
├── framework/              # 性能测试基础框架
│   ├── benchmark_base.h   # 基础类和工具函数
│   ├── benchmark_config.h  # 配置选项
│   └── performance_metrics.h  # 性能指标计算
├── nppi/                   # NPPI 模块性能测试
│   ├── arithmetic/         # 算术运算
│   ├── filtering/          # 滤波操作
│   └── geometry/           # 几何变换
├── comparison/             # MPP vs NVIDIA NPP 对比测试
├── CMakeLists.txt          # 构建配置
├── run_comparison.sh       # 自动对比脚本
└── README.md               # 本文档
```

## 🚀 快速开始

### 1. 安装依赖

#### 方案 A：使用 FetchContent 自动下载（推荐）
```bash
# 无需手动安装，CMake 会自动下载 Google Benchmark
cd /path/to/npp
mkdir -p build && cd build
cmake .. -DBUILD_BENCHMARKS=ON
```

#### 方案 B：手动安装 Google Benchmark
```bash
# Ubuntu/Debian
sudo apt-get install libbenchmark-dev

# macOS
brew install google-benchmark

# 从源码安装
git clone https://github.com/google/benchmark.git
cd benchmark
cmake -E make_directory "build"
cmake -E chdir "build" cmake -DBENCHMARK_DOWNLOAD_DEPENDENCIES=on -DCMAKE_BUILD_TYPE=Release ../
cmake --build "build" --config Release
sudo cmake --build "build" --config Release --target install
```

### 2. 编译性能测试

```bash
cd /path/to/npp

# 编译 MPP 版本的性能测试
./build.sh --release
cd build/benchmark

# 运行性能测试
./nppi_arithmetic_benchmark
```

### 3. 运行对比测试

自动对比 MPP 和 NVIDIA NPP 的性能：

```bash
cd /path/to/npp/test/benchmark
chmod +x run_comparison.sh
./run_comparison.sh
```

## 📊 性能测试使用说明

### 基本用法

```bash
# 运行所有测试
./nppi_arithmetic_benchmark

# 运行特定测试（使用过滤器）
./nppi_arithmetic_benchmark --benchmark_filter=Add_8u

# 指定重复次数（提高统计准确性）
./nppi_arithmetic_benchmark --benchmark_repetitions=10

# 输出 JSON 格式结果
./nppi_arithmetic_benchmark \
    --benchmark_out=results.json \
    --benchmark_out_format=json

# 输出 CSV 格式结果
./nppi_arithmetic_benchmark \
    --benchmark_out=results.csv \
    --benchmark_out_format=csv

# 只运行包含 "1920x1080" 的测试
./nppi_arithmetic_benchmark --benchmark_filter=1920x1080

# 显示详细输出
./nppi_arithmetic_benchmark --benchmark_enable_random_interleaving=true
```

### 高级选项

```bash
# 最小运行时间（秒）
./nppi_arithmetic_benchmark --benchmark_min_time=5.0

# 报告聚合统计信息（均值、中位数、标准差）
./nppi_arithmetic_benchmark \
    --benchmark_repetitions=10 \
    --benchmark_report_aggregates_only=true

# 设置 CPU 亲和性（避免线程迁移）
./nppi_arithmetic_benchmark --benchmark_enable_random_interleaving=false

# 显示计数器信息
./nppi_arithmetic_benchmark --benchmark_counters_tabular=true
```

## 📈 性能指标说明

### 输出指标解读

```
---------------------------------------------------------------------------
Benchmark                                 Time      CPU   Iterations  Bytes/s
---------------------------------------------------------------------------
BM_nppiAdd_8u_C1RSfs_Fixed         0.234 ms  0.234 ms         2987   53.7M/s
```

- **Time**: 实际运行时间（wall time）
- **CPU**: CPU 时间（对于 GPU 操作，通常与 Time 接近）
- **Iterations**: 运行次数（自动调整以获得稳定测量）
- **Bytes/s**: 数据吞吐量（内存带宽利用率）

### 自定义指标

```
BM_nppiAdd_8u_C1RSfs_Sizes/1920/1080   0.245 ms   Megapixels=2.07  Width=1920  Height=1080
```

- **Megapixels**: 图像大小（百万像素）
- **Width/Height**: 图像尺寸
- **ScaleFactor**: 缩放因子（如适用）

## 🔍 编写新的性能测试

### 示例：简单性能测试

```cpp
#include "benchmark_base.h"
#include <nppi_arithmetic_and_logical_operations.h>

using namespace npp_benchmark;

static void BM_MyFunction(benchmark::State& state) {
    // 准备数据
    int width = 1920, height = 1080;
    ImageBenchmarkBase<Npp8u> base;
    base.SetupImageMemory(width, height);
    
    NppiSize roi = {width, height};
    
    // 性能测试主循环
    for (auto _ : state) {
        // 调用被测函数
        nppiMyFunction_8u_C1R(
            base.d_src1_, base.step_,
            base.d_dst_, base.step_,
            roi);
        
        // 确保 GPU 完成
        base.SyncAndCheckError();
    }
    
    // 报告性能指标
    size_t bytes = base.ComputeImageBytes(1, 1);  // 1 input, 1 output
    REPORT_THROUGHPUT(state, bytes);
    
    base.TeardownImageMemory();
}

// 注册测试
BENCHMARK(BM_MyFunction)->UseRealTime();
```

### 示例：参数化性能测试

```cpp
static void BM_MyFunction_Sizes(benchmark::State& state) {
    int width = state.range(0);
    int height = state.range(1);
    
    ImageBenchmarkBase<Npp8u> base;
    base.SetupImageMemory(width, height);
    
    NppiSize roi = {width, height};
    
    for (auto _ : state) {
        nppiMyFunction_8u_C1R(base.d_src1_, base.step_,
                              base.d_dst_, base.step_, roi);
        base.SyncAndCheckError();
    }
    
    size_t bytes = base.ComputeImageBytes(1, 1);
    REPORT_THROUGHPUT(state, bytes);
    REPORT_CUSTOM_METRIC(state, "Megapixels", (width * height) / 1e6);
    
    base.TeardownImageMemory();
}

// 注册不同尺寸的测试
BENCHMARK(BM_MyFunction_Sizes)
    ->Args({640, 480})        // VGA
    ->Args({1920, 1080})      // Full HD
    ->Args({3840, 2160})      // 4K
    ->Unit(benchmark::kMillisecond)
    ->UseRealTime();
```

## 🎯 性能优化目标

### 参考指标（NVIDIA A100）

| 操作 | 图像尺寸 | 目标性能 | 带宽利用率 |
|------|---------|---------|-----------|
| Add_8u_C1R | 1920x1080 | < 0.3 ms | > 70% |
| Add_32f_C1R | 1920x1080 | < 1.0 ms | > 60% |
| FilterBox_8u_C1R | 1920x1080 (3x3) | < 2.0 ms | > 50% |
| Resize_8u_C1R | 1920x1080 → 3840x2160 | < 5.0 ms | > 40% |

### 带宽计算

```
理论带宽 = GPU Memory Bandwidth (GB/s)
实际带宽 = (Data Read + Data Write) / Time

利用率 = 实际带宽 / 理论带宽 * 100%
```

## 📊 对比测试结果解读

### 性能对比示例

```
Operation: nppiAdd_8u_C1RSfs (1920x1080)
├─ MPP:        0.245 ms  (41.2 GB/s)  ← 你的实现
├─ NVIDIA NPP: 0.238 ms  (42.4 GB/s)  ← NVIDIA 官方
└─ Speedup:    0.97x     (97% of NVIDIA performance)
```

### 性能目标

- **优秀**: > 95% NVIDIA NPP 性能
- **良好**: 80-95% NVIDIA NPP 性能
- **可接受**: 60-80% NVIDIA NPP 性能
- **需优化**: < 60% NVIDIA NPP 性能

## 🐛 故障排查

### 常见问题

#### 1. 编译错误：找不到 benchmark.h

```bash
# 确保启用了性能测试
cmake .. -DBUILD_BENCHMARKS=ON

# 或者手动安装 Google Benchmark
```

#### 2. 运行时 CUDA 错误

```bash
# 检查 CUDA 设备
nvidia-smi

# 设置正确的 CUDA 设备
export CUDA_VISIBLE_DEVICES=0
```

#### 3. 性能结果不稳定

```bash
# 增加重复次数
./benchmark --benchmark_repetitions=20

# 设置最小运行时间
./benchmark --benchmark_min_time=3.0

# 禁用 CPU 频率缩放（需要 root）
sudo cpupower frequency-set --governor performance
```

## 📚 参考资源

- [Google Benchmark 文档](https://github.com/google/benchmark)
- [NVIDIA NPP 文档](https://docs.nvidia.com/cuda/npp/index.html)
- [CUDA 性能优化指南](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html)

## 🤝 贡献指南

添加新的性能测试时：

1. 在相应模块目录创建 `benchmark_nppi_xxx.cpp`
2. 继承 `ImageBenchmarkBase` 或 `NppBenchmarkBase`
3. 使用 `BENCHMARK()` 宏注册测试
4. 报告合适的性能指标（吞吐量、延迟等）
5. 更新本 README 文档

## 📝 注意事项

- ⚠️ 性能测试会占用 GPU 资源，运行时请关闭其他 GPU 应用
- ⚠️ 测试结果受硬件配置影响，不同 GPU 结果会有差异
- ⚠️ 使用 `--benchmark_repetitions` 获得更稳定的统计结果
- ⚠️ 性能测试不验证功能正确性，需配合单元测试使用
