# NPP Benchmark 扩展指南

本文档说明如何快速扩展 NPP benchmark 覆盖范围，从当前的 1 个函数扩展到 100+ 个函数。

## 📊 当前进度

```
Unit Tests 覆盖: ~120+ NPP APIs
Benchmarks 覆盖: 1 API (nppiAdd)

待扩展: ~119 APIs
```

## 🚀 快速开始

### 方法 1: 使用生成脚本（推荐）

#### 生成单个 benchmark

```bash
cd test/benchmark

# 生成 nppiSub benchmark
python3 generate_benchmark.py nppiSub 8u C1 RSfs --module arithmetic

# 生成 nppiMul benchmark  
python3 generate_benchmark.py nppiMul 32f C1 R --module arithmetic

# 生成 nppiResize benchmark
python3 generate_benchmark.py nppiResize 8u C3 R --module geometry
```

#### 批量生成（基于已有的 unit tests）

```bash
# 扫描并生成所有算术运算的 benchmarks
python3 batch_generate_benchmarks.py --module arithmetic

# 生成所有模块的 benchmarks
python3 batch_generate_benchmarks.py --all

# 预览将要生成的文件（不实际创建）
python3 batch_generate_benchmarks.py --module arithmetic --dry-run
```

### 方法 2: 手动复制模板

1. 复制 `BENCHMARK_TEMPLATE.cpp`
2. 重命名为 `benchmark_nppi_xxx.cpp`
3. 替换所有 `{{占位符}}`
4. 更新 `CMakeLists.txt`

## 📁 目录结构

```
test/benchmark/
├── framework/
│   └── benchmark_base.h        # 基础框架（无需修改）
├── nppi/
│   ├── arithmetic/             # 算术运算 benchmarks
│   │   └── benchmark_nppi_add.cpp
│   ├── filtering/              # 滤波 benchmarks（待添加）
│   ├── geometry/               # 几何变换 benchmarks（待添加）
│   ├── color/                  # 颜色转换 benchmarks（待添加）
│   └── ...
├── CMakeLists.txt              # 注册新的 benchmarks
├── BENCHMARK_TEMPLATE.cpp      # 模板文件
├── generate_benchmark.py       # 单文件生成工具
└── batch_generate_benchmarks.py # 批量生成工具
```

## 🎯 优先级建议

基于 unit test 的覆盖情况，建议按以下顺序添加：

### 高优先级（35 个函数）
**算术运算（nppi_arithmetic_operations）**
- [x] nppiAdd (已完成)
- [ ] nppiSub
- [ ] nppiMul
- [ ] nppiDiv
- [ ] nppiAbs
- [ ] nppiSqr
- [ ] nppiSqrt
- [ ] ... 等 28 个

### 中优先级（~50 个函数）
**滤波（nppi_filtering_functions）**
- [ ] nppiFilter
- [ ] nppiFilterBox
- [ ] nppiFilterGauss
- [ ] nppiFilterMedian
- [ ] ... 等

**几何变换（nppi_geometry_transforms）**
- [ ] nppiResize
- [ ] nppiRemap
- [ ] nppiRotate
- [ ] nppiMirror
- [ ] ... 等

### 低优先级（~35 个函数）
**颜色转换、统计、阈值等**

## 📝 添加新 Benchmark 的步骤

### 步骤 1: 创建 Benchmark 文件

```bash
# 使用脚本生成
python3 generate_benchmark.py nppiSub 8u C1 RSfs --module arithmetic

# 或手动创建
cp BENCHMARK_TEMPLATE.cpp nppi/arithmetic/benchmark_nppi_sub.cpp
# 然后编辑文件，替换占位符
```

### 步骤 2: 更新 CMakeLists.txt

在 `test/benchmark/CMakeLists.txt` 中添加：

```cmake
set(NPPI_ARITHMETIC_BENCHMARK_SOURCES
    nppi/arithmetic/benchmark_nppi_add.cpp
    nppi/arithmetic/benchmark_nppi_sub.cpp  # 新添加
)
```

### 步骤 3: 编译测试

```bash
cd build
cmake .. -DBUILD_BENCHMARKS=ON

# 只编译新的 benchmark
make nppi_arithmetic_benchmark

# 运行测试
./benchmark/nppi_arithmetic_benchmark --benchmark_filter=Sub
```

### 步骤 4: 添加到对比脚本

确保 `run_comparison.sh` 能运行新的 benchmark（通常无需修改，因为脚本会运行所有 benchmark targets）。

## 🔧 特殊函数处理

### 带 Scale Factor 的函数

```cpp
// nppiAdd_8u_C1RSfs
const int scaleFactor = 0;  // 添加参数

NppStatus status = nppiAdd_8u_C1RSfs(
    base.d_src1_, base.step_,
    base.d_src2_, base.step_,
    base.d_dst_, base.step_,
    roi, scaleFactor  // 传递参数
);
```

### In-place 操作

```cpp
// nppiAdd_8u_C1IR (结果写回 src2)
NppStatus status = nppiAdd_8u_C1IR(
    base.d_src1_, base.step_,
    base.d_src2_, base.step_,  // 同时作为输入和输出
    roi
);
```

### 多通道函数

```cpp
// C3 (3通道)
ImageBenchmarkBase<Npp8u> base;
base.SetupImageMemory(width, height, 3);  // 指定通道数

// 字节数计算也需要考虑通道
size_t bytesProcessed = base.ComputeImageBytes(2, 1) * 3;
```

### 带常量的函数

```cpp
// nppiAddC (加常量)
const Npp8u constant = 42;

NppStatus status = nppiAddC_8u_C1RSfs(
    base.d_src1_, base.step_,
    constant,
    base.d_dst_, base.step_,
    roi, scaleFactor
);

// 只有1个输入
size_t bytesProcessed = base.ComputeImageBytes(1, 1);
```

## 🎨 不同模块的头文件

```cpp
// 算术运算
#include <nppi_arithmetic_and_logical_operations.h>

// 滤波
#include <nppi_filtering_functions.h>

// 几何变换
#include <nppi_geometry_transforms.h>

// 颜色转换
#include <nppi_color_conversion.h>

// 统计
#include <nppi_statistics_functions.h>

// 阈值和比较
#include <nppi_threshold_and_compare_operations.h>
```

## 📊 验证 Benchmark

### 编译验证

```bash
cd build
cmake .. -DBUILD_BENCHMARKS=ON -DUSE_NVIDIA_NPP=OFF
make nppi_arithmetic_benchmark
```

### 功能验证

```bash
# 快速测试（最小迭代次数）
./benchmark/nppi_arithmetic_benchmark --benchmark_min_time=0.01

# 测试特定函数
./benchmark/nppi_arithmetic_benchmark --benchmark_filter=Sub

# 输出 JSON 结果
./benchmark/nppi_arithmetic_benchmark --benchmark_out=test.json --benchmark_out_format=json
```

### 对比验证

```bash
cd test/benchmark
./run_comparison.sh
```

## 🚨 常见问题

### Q: 生成的 benchmark 编译失败

**A:** 检查以下几点：
1. 头文件是否正确
2. 函数签名是否匹配（参数顺序、类型）
3. CMakeLists.txt 是否正确添加了源文件

### Q: Benchmark 运行时崩溃

**A:** 可能原因：
1. 内存分配失败（GPU 内存不足）
2. 函数参数错误（检查 step、roi 等）
3. 同步问题（确保调用了 `SyncAndCheckError()`）

### Q: 性能结果异常

**A:** 检查：
1. 是否正确计算了 `bytesProcessed`
2. 是否包含了所有输入和输出
3. 通道数是否正确

## 📈 批量添加计划

建议分批次添加：

### 第一批（1-2 周）
```bash
# 算术运算的核心函数
python3 generate_benchmark.py nppiSub 8u C1 RSfs --module arithmetic
python3 generate_benchmark.py nppiMul 8u C1 RSfs --module arithmetic
python3 generate_benchmark.py nppiDiv 8u C1 RSfs --module arithmetic
# ... 添加 10-15 个核心函数
```

### 第二批（2-3 周）
```bash
# 完成算术运算 + 开始滤波
python3 batch_generate_benchmarks.py --module arithmetic
python3 generate_benchmark.py nppiFilter 8u C1 R --module filtering
# ...
```

### 第三批（3-4 周）
```bash
# 所有剩余模块
python3 batch_generate_benchmarks.py --all
```

## 🔗 相关文档

- [Benchmark 框架说明](./README.md)
- [测试基类 API](./framework/benchmark_base.h)
- [CMake 配置说明](../../cmake/BenchmarkConfig.cmake)
- [运行对比脚本](./run_comparison.sh)

## 🤝 贡献指南

1. 每次添加 benchmark 后提交代码
2. 确保所有 benchmark 都能编译通过
3. 运行对比测试验证结果
4. 更新本文档记录新添加的函数
