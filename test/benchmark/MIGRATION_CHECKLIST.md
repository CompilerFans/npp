# 测试框架迁移检查清单

将当前测试框架迁移到优化版 MPP 库的完整指南。

## ✅ 前置条件检查

### 1. 优化版 MPP 库结构要求

```
optimized-mpp/
├── src/                     ✅ 必需：源代码目录
│   ├── nppi/               ✅ 必需：NPP 实现
│   │   ├── nppi_arithmetic_operations/
│   │   ├── nppi_filtering_functions/
│   │   └── ...
│   └── CMakeLists.txt      ✅ 必需：必须创建 "npp" target
├── include/                ⚠️  推荐：公共头文件
│   └── nppi_*.h
└── (测试框架将复制到这里)
```

### 2. CMakeLists.txt 兼容性要求

优化版 MPP 的 `src/CMakeLists.txt` **必须**包含：

```cmake
# ✅ 必须：创建名为 "npp" 的 target
add_library(npp ${YOUR_SOURCES})

# ✅ 必须：设置正确的包含目录
target_include_directories(npp PUBLIC
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)

# ⚠️  可选：输出名称可以不同
set_target_properties(npp PROPERTIES
    OUTPUT_NAME "mpp_optimized"  # 实际生成 libmpp_optimized.so
)
```

## 📋 迁移步骤

### 步骤 1: 备份当前测试框架

```bash
# 在当前 MPP 目录
cd /path/to/current-mpp

# 打包测试框架
tar -czf test-framework-$(date +%Y%m%d).tar.gz \
    test/benchmark/ \
    cmake/ \
    CMakeLists.txt

# 或使用 git
git archive --format=tar.gz --output=test-framework.tar.gz HEAD \
    test/benchmark cmake CMakeLists.txt
```

### 步骤 2: 复制到优化版 MPP

```bash
# 进入优化版 MPP 目录
cd /path/to/optimized-mpp

# 创建必要的目录
mkdir -p test/benchmark cmake

# 复制测试框架
cp -r /path/to/current-mpp/test/benchmark/* test/benchmark/
cp -r /path/to/current-mpp/cmake/* cmake/

# 复制或合并 CMakeLists.txt（如果还没有）
cp /path/to/current-mpp/CMakeLists.txt ./

# 或合并配置（如果已有 CMakeLists.txt）
# 手动合并以下部分：
# - CMAKE_MODULE_PATH 设置
# - include(BenchmarkConfig) 等
# - BUILD_BENCHMARKS 选项
# - add_subdirectory(test) 逻辑
```

### 步骤 3: 验证目录结构

```bash
# 检查关键文件是否都在
ls -la test/benchmark/CMakeLists.txt
ls -la test/benchmark/framework/benchmark_base.h
ls -la test/benchmark/nppi/arithmetic/benchmark_nppi_add.cpp
ls -la test/benchmark/run_comparison.sh
ls -la test/benchmark/compare_results.py
ls -la cmake/BenchmarkConfig.cmake
```

### 步骤 4: 调整 CMake 配置（如果需要）

检查并更新 `CMakeLists.txt`：

```cmake
# 确保这些行存在
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(BenchmarkConfig)

# 确保 benchmark 选项存在
option(BUILD_BENCHMARKS "Build performance benchmarks" OFF)
option(USE_NVIDIA_NPP "Use NVIDIA NPP library instead of MPP" OFF)

# 确保 test 目录被添加
if(BUILD_TESTS OR BUILD_BENCHMARKS)
    enable_testing()
    add_subdirectory(test)
endif()
```

### 步骤 5: 第一次编译测试（MPP 模式）

```bash
# 创建构建目录
mkdir -p build && cd build

# 配置（使用优化版 MPP）
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_BENCHMARKS=ON \
    -DUSE_NVIDIA_NPP=OFF

# 编译
make -j$(nproc)

# 检查编译结果
ls -lh benchmark/nppi_arithmetic_benchmark
```

**预期输出：**
```
-rwxr-xr-x 1 user group 2.3M benchmark/nppi_arithmetic_benchmark
```

### 步骤 6: 第二次编译测试（NVIDIA NPP 模式）

```bash
# 清理并重新配置
cd build
rm -rf *

cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_BENCHMARKS=ON \
    -DUSE_NVIDIA_NPP=ON

# 编译
make -j$(nproc)

# 应该同样成功
ls -lh benchmark/nppi_arithmetic_benchmark
```

### 步骤 7: 运行功能测试

```bash
# 测试 MPP 版本
cd build
./benchmark/nppi_arithmetic_benchmark --benchmark_min_time=0.01

# 测试 NVIDIA NPP 版本（重新编译后）
./benchmark/nppi_arithmetic_benchmark --benchmark_min_time=0.01
```

**预期：两次都能成功运行，输出类似：**
```
Run on (12 X 3000 MHz CPU s)
CPU Caches:
  L1 Data 32 KiB (x6)
  ...
-------------------------------------------------------------------
Benchmark                              Time             CPU   Iterations
-------------------------------------------------------------------
BM_nppiAdd_8u_C1RSfs_Fixed           0.XXX ms        0.XXX ms         XXXX
...
```

### 步骤 8: 运行完整对比测试

```bash
cd test/benchmark
./run_comparison.sh
```

**预期输出：**
```
Building MPP version...
Running MPP benchmarks...
Building NVIDIA NPP version...
Running NVIDIA NPP benchmarks...
Comparing results...

NPP Performance Comparison Report
====================================================================================================
Test Name                          MPP (ms)  NVIDIA (ms)      Perf%       Rating
----------------------------------------------------------------------------------------------------
...

CSV report generated: benchmark_results/comparison_YYYYMMDD_HHMMSS.csv
```

## 🔍 常见问题排查

### ❌ 问题 1: 找不到 "npp" target

**错误信息：**
```
CMake Error: Target "npp" not found in project
```

**解决方案：**
检查优化版 MPP 的 `src/CMakeLists.txt`：

```cmake
# 确保有这一行
add_library(npp ${SOURCES})

# 而不是
add_library(mpp_optimized ${SOURCES})  # ❌ 错误
```

### ❌ 问题 2: 头文件找不到

**错误信息：**
```
fatal error: nppi_arithmetic_and_logical_operations.h: No such file or directory
```

**解决方案：**

```cmake
# 在 src/CMakeLists.txt 中添加
target_include_directories(npp PUBLIC
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)
```

### ❌ 问题 3: 链接错误

**错误信息：**
```
undefined reference to `nppiAdd_8u_C1RSfs`
```

**原因：**
1. 函数实现未编译
2. 函数签名不匹配
3. extern "C" 缺失

**解决方案：**
检查优化版 MPP 的实现：

```cpp
// nppi_add.cu
extern "C" {
NppStatus nppiAdd_8u_C1RSfs(...) {
    // 实现
}
}
```

### ❌ 问题 4: 运行时崩溃

**错误信息：**
```
Segmentation fault
```

**排查步骤：**

```bash
# 使用 cuda-memcheck
cuda-memcheck ./benchmark/nppi_arithmetic_benchmark --benchmark_min_time=0.01

# 使用 gdb
gdb --args ./benchmark/nppi_arithmetic_benchmark --benchmark_min_time=0.01
(gdb) run
(gdb) bt  # 查看堆栈
```

## ✅ 迁移完成验证清单

运行以下命令，全部通过则迁移成功：

```bash
# 1. 编译 MPP 版本
cd build && rm -rf * && cmake .. -DBUILD_BENCHMARKS=ON -DUSE_NVIDIA_NPP=OFF && make -j$(nproc)
echo "✅ MPP version compilation: $?"

# 2. 运行 MPP benchmark
./benchmark/nppi_arithmetic_benchmark --benchmark_min_time=0.01 > /dev/null 2>&1
echo "✅ MPP benchmark execution: $?"

# 3. 编译 NVIDIA NPP 版本
cd build && rm -rf * && cmake .. -DBUILD_BENCHMARKS=ON -DUSE_NVIDIA_NPP=ON && make -j$(nproc)
echo "✅ NVIDIA NPP version compilation: $?"

# 4. 运行 NVIDIA NPP benchmark
./benchmark/nppi_arithmetic_benchmark --benchmark_min_time=0.01 > /dev/null 2>&1
echo "✅ NVIDIA NPP benchmark execution: $?"

# 5. 运行完整对比
cd ../test/benchmark && ./run_comparison.sh > /dev/null 2>&1
echo "✅ Comparison script execution: $?"

# 6. 检查结果文件
ls benchmark_results/comparison_*.csv > /dev/null 2>&1
echo "✅ Result files generated: $?"
```

**全部输出 `✅ ... : 0` 则迁移成功！**

## 📊 性能基准参考

迁移完成后，运行对比测试应该看到：

### 预期性能范围

```
优化版 MPP vs NVIDIA NPP:
- 优秀实现: 90-100% (接近或等于 NVIDIA)
- 良好实现: 70-90%
- 可接受实现: 50-70%
- 需要继续优化: <50%
```

如果优化版 MPP 的性能显著低于预期，检查：
1. CUDA kernel 配置
2. 内存访问模式
3. 编译优化选项

## 🔄 持续维护

### 添加新的 API benchmark

```bash
# 使用生成工具
cd test/benchmark
python3 generate_benchmark.py nppiXXX 8u C1 R --module arithmetic

# 重新编译测试
cd ../../build
make nppi_arithmetic_benchmark

# 运行新的 benchmark
./benchmark/nppi_arithmetic_benchmark --benchmark_filter=XXX
```

### 更新对比脚本

如果添加了新的 benchmark 模块，更新 `CMakeLists.txt`：

```cmake
# 添加新模块
set(NPPI_FILTERING_BENCHMARK_SOURCES
    nppi/filtering/benchmark_nppi_filter.cpp
)

npp_create_benchmark_target(
    nppi_filtering_benchmark
    "${NPPI_FILTERING_BENCHMARK_SOURCES}"
    npp
)
```

## 🎯 下一步

迁移完成后，可以：

1. ✅ **扩展测试覆盖**：使用 `batch_generate_benchmarks.py` 添加更多 API
2. ✅ **性能优化**：根据对比结果优化慢的 API
3. ✅ **CI 集成**：将 benchmark 加入持续集成流程
4. ✅ **文档完善**：记录优化技巧和性能指标

---

## 📞 需要帮助？

如果遇到问题：
1. 查看 [EXPANSION_GUIDE.md](./EXPANSION_GUIDE.md)
2. 查看 [README.md](./README.md)
3. 检查 CMake 配置：`cmake .. -LAH | grep NPP`
4. 查看编译日志：`make VERBOSE=1`
