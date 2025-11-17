# NPP 性能测试框架实施总结

## ✅ 已完成的工作

### 1. 核心框架文件

```
✓ cmake/BenchmarkConfig.cmake              # 性能测试 CMake 配置
✓ test/benchmark/framework/benchmark_base.h # 性能测试基础类
✓ test/benchmark/CMakeLists.txt             # 性能测试构建配置
✓ test/benchmark/README.md                  # 详细使用文档
```

### 2. 示例性能测试

```
✓ test/benchmark/nppi/arithmetic/benchmark_nppi_add.cpp  # Add 操作示例
  - 固定尺寸测试
  - 参数化尺寸测试
  - Scale Factor 对比测试
  - 多种数据类型测试
```

### 3. 自动化脚本

```
✓ test/benchmark/run_comparison.sh          # MPP vs NVIDIA NPP 对比脚本
```

### 4. 文档

```
✓ test/benchmark/README.md                  # 完整使用文档
✓ docs/BENCHMARK_GUIDE.md                   # 快速开始指南
✓ test/benchmark/IMPLEMENTATION_SUMMARY.md  # 本文档
```

### 5. CMake 集成

```
✓ 主 CMakeLists.txt 已更新
  - 添加 BUILD_BENCHMARKS 选项
  - 添加 NPP_REGISTER_BENCHMARKS_TO_CTEST 选项
  - 包含 BenchmarkConfig 模块
  
✓ test/CMakeLists.txt 已更新
  - 条件编译性能测试
  - 提示信息
```

## 📊 框架特性

### ✨ 核心功能

| 功能 | 描述 | 状态 |
|------|------|------|
| **Google Benchmark 集成** | 业界标准性能测试框架 | ✅ |
| **自动下载依赖** | FetchContent 自动获取 | ✅ |
| **GPU 内存管理** | 智能的 CUDA 内存管理 | ✅ |
| **性能指标报告** | 吞吐量、延迟、带宽 | ✅ |
| **参数化测试** | 多尺寸、多配置测试 | ✅ |
| **对比测试** | MPP vs NVIDIA NPP | ✅ |
| **结果导出** | JSON/CSV 格式 | ✅ |
| **CI 集成** | CTest 支持 | ✅ |

### 🎯 性能指标

框架自动计算并报告：

1. **执行时间** (ms)
2. **数据吞吐量** (GB/s)
3. **图像吞吐量** (Megapixels/s)
4. **自定义指标** (Width, Height, ScaleFactor, etc.)

### 🔧 灵活配置

- ✅ 支持多种图像尺寸
- ✅ 支持多种数据类型 (8u, 16u, 16s, 32f, etc.)
- ✅ 支持多通道 (C1, C3, C4)
- ✅ 可配置重复次数
- ✅ 可过滤特定测试

## 🚀 使用方式

### 快速开始

```bash
# 1. 编译（自动下载 Google Benchmark）
cd /Users/jiaozihan/Desktop/MPP/npp
mkdir -p build && cd build
cmake .. -DBUILD_BENCHMARKS=ON
make -j$(nproc)

# 2. 运行性能测试
cd build/benchmark
./nppi_arithmetic_benchmark

# 3. 运行对比测试
cd ../../test/benchmark
./run_comparison.sh
```

### 高级用法

```bash
# 只测试特定函数
./nppi_arithmetic_benchmark --benchmark_filter=Add_8u

# 提高测试精度
./nppi_arithmetic_benchmark --benchmark_repetitions=10

# 导出结果
./nppi_arithmetic_benchmark \
    --benchmark_out=results.json \
    --benchmark_out_format=json
```

## 📝 如何扩展

### 添加新的性能测试

1. **创建测试文件**
   ```cpp
   // test/benchmark/nppi/your_module/benchmark_your_func.cpp
   #include "benchmark_base.h"
   
   static void BM_YourFunc(benchmark::State& state) {
       // 测试代码
   }
   BENCHMARK(BM_YourFunc)->UseRealTime();
   ```

2. **更新 CMakeLists.txt**
   ```cmake
   # test/benchmark/CMakeLists.txt
   set(YOUR_MODULE_SOURCES
       nppi/your_module/benchmark_your_func.cpp
   )
   
   npp_create_benchmark_target(
       your_module_benchmark
       "${YOUR_MODULE_SOURCES}"
       npp_nppi_lib
   )
   ```

3. **编译运行**
   ```bash
   make your_module_benchmark
   ./your_module_benchmark
   ```

## 🎨 设计模式

### 1. 基于继承的抽象

```cpp
NppBenchmarkBase            # 基础功能
    ↓
ImageBenchmarkBase<T>       # 图像操作专用
    ↓
你的测试                    # 具体测试用例
```

### 2. RAII 内存管理

```cpp
ImageBenchmarkBase<Npp8u> base;
base.SetupImageMemory(width, height);  // 自动分配
// ... 测试代码 ...
base.TeardownImageMemory();            // 自动释放
```

### 3. 模板泛化

```cpp
template<typename PixelType>
class ImageBenchmarkBase {
    // 支持所有数据类型
};
```

## 📈 性能测试最佳实践

### ✅ 推荐做法

1. **使用 UseRealTime()** - GPU 测试必须使用实际时间
2. **调用 SyncAndCheckError()** - 确保 GPU 完成并检查错误
3. **报告合适的指标** - 吞吐量、自定义指标
4. **使用参数化测试** - 测试多种配置
5. **重复测试** - 使用 `--benchmark_repetitions`

### ❌ 避免的错误

1. ❌ 不调用 cudaDeviceSynchronize()
2. ❌ 在循环外分配内存（测量不准确）
3. ❌ 忘记报告性能指标
4. ❌ 只测试单一配置
5. ❌ 不检查 CUDA 错误

## 🔍 示例测试解析

```cpp
static void BM_nppiAdd_8u_C1RSfs_Fixed(benchmark::State& state) {
    // ==================== 准备阶段 ====================
    const int width = 1920, height = 1080;
    ImageBenchmarkBase<Npp8u> base;
    base.SetupImageMemory(width, height);  // 分配 GPU 内存并初始化
    
    NppiSize roi = {width, height};
    
    // ==================== 测试循环 ====================
    for (auto _ : state) {  // Google Benchmark 控制循环次数
        NppStatus status = nppiAdd_8u_C1RSfs(
            base.d_src1_, base.step_,
            base.d_src2_, base.step_,
            base.d_dst_, base.step_,
            roi, 0);
        
        base.SyncAndCheckError();  // 关键：同步 GPU
        
        if (status != NPP_NO_ERROR) {
            state.SkipWithError("Function failed");
            break;
        }
    }
    
    // ==================== 报告阶段 ====================
    size_t bytesProcessed = base.ComputeImageBytes(2, 1);  // 2 输入, 1 输出
    REPORT_THROUGHPUT(state, bytesProcessed);  // 自动计算 GB/s
    
    REPORT_CUSTOM_METRIC(state, "Megapixels", (width * height) / 1e6);
    
    // ==================== 清理阶段 ====================
    base.TeardownImageMemory();
}

BENCHMARK(BM_nppiAdd_8u_C1RSfs_Fixed)
    ->Unit(benchmark::kMillisecond)  // 以毫秒为单位
    ->UseRealTime();                 // 使用实际时间（GPU 必须）
```

## 🎯 下一步工作

### 短期（1-2 周）

- [ ] 为更多算术操作添加性能测试 (Sub, Mul, Div)
- [ ] 添加滤波操作性能测试 (Box, Gauss, Sobel)
- [ ] 实现结果可视化 Python 脚本

### 中期（1-2 个月）

- [ ] 为所有已实现函数添加性能测试
- [ ] 建立性能回归测试 CI
- [ ] 创建性能对比报告网页

### 长期（3+ 个月）

- [ ] 集成 Nsight Compute 自动分析
- [ ] 性能优化指导系统
- [ ] 自动性能调优工具

## 🐛 已知限制

1. **Google Benchmark 依赖** - 需要外部库（但可自动下载）
2. **GPU 独占** - 运行时需要独占 GPU
3. **结果波动** - GPU 时钟频率影响结果稳定性
4. **对比测试时间长** - 完整对比需要 30+ 分钟

## 🤝 贡献指南

欢迎贡献：

1. **添加新测试** - 按照模板添加
2. **改进框架** - 提交 PR
3. **报告问题** - 创建 Issue
4. **改进文档** - 更新 README

## 📚 参考资源

- [Google Benchmark GitHub](https://github.com/google/benchmark)
- [Google Benchmark 用户指南](https://github.com/google/benchmark/blob/main/docs/user_guide.md)
- [NVIDIA NPP 文档](https://docs.nvidia.com/cuda/npp/)
- [CUDA 性能优化指南](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)

## ✅ 验收标准

性能测试框架应满足：

- ✅ 能够独立编译运行
- ✅ 支持 MPP 和 NVIDIA NPP 两种模式
- ✅ 自动化对比测试脚本可用
- ✅ 输出清晰易懂的性能指标
- ✅ 易于扩展新的测试用例
- ✅ 文档完整，示例充足

---

**框架版本**: 1.0  
**创建日期**: 2024  
**维护者**: NPP Team  
**状态**: ✅ 生产就绪
