# NPP Benchmark 快速开始指南

## 🎯 核心问题解答

### ✅ 测试框架是否可以迁移到优化版 MPP？

**是的，完全可以！** 测试框架设计为完全解耦，只需：

1. **替换 `src/` 目录** - 使用优化版 MPP 的实现
2. **保持 Target 名称** - 确保创建名为 `npp` 的 CMake target
3. **无需修改测试代码** - 所有 benchmark 代码保持不变

### ✅ 如何快速扩展到 100+ API？

**三种方法：**

1. **自动批量生成**（最快）
   ```bash
   cd test/benchmark
   python3 batch_generate_benchmarks.py --module arithmetic
   ```

2. **单个函数生成**
   ```bash
   python3 generate_benchmark.py nppiSub 8u C1 RSfs --module arithmetic
   ```

3. **手动复制模板**
   ```bash
   cp BENCHMARK_TEMPLATE.cpp nppi/arithmetic/benchmark_nppi_xxx.cpp
   # 编辑并替换占位符
   ```

## 📚 关键文档导航

| 文档 | 用途 | 适用场景 |
|------|------|---------|
| [MIGRATION_CHECKLIST.md](./MIGRATION_CHECKLIST.md) | 迁移到优化版 MPP 的完整步骤 | 🔥 **最重要** - 迁移时必读 |
| [EXPANSION_GUIDE.md](./EXPANSION_GUIDE.md) | 扩展 benchmark 覆盖范围 | 添加新的 API benchmark |
| [README.md](./README.md) | 完整的 benchmark 框架说明 | 了解框架架构和使用 |
| [IMPLEMENTATION_SUMMARY.md](./IMPLEMENTATION_SUMMARY.md) | 实现总结和技术细节 | 理解内部实现 |

## 🚀 5 分钟快速上手

### 场景 1: 迁移到优化版 MPP

```bash
# 1. 复制测试框架到优化版 MPP
cd /path/to/optimized-mpp
cp -r /path/to/current-mpp/test/benchmark test/
cp -r /path/to/current-mpp/cmake cmake/

# 2. 确保优化版 MPP 的 src/CMakeLists.txt 创建了 "npp" target
# 编辑 src/CMakeLists.txt:
#   add_library(npp ${SOURCES})

# 3. 编译测试
mkdir build && cd build
cmake .. -DBUILD_BENCHMARKS=ON -DUSE_NVIDIA_NPP=OFF
make -j$(nproc)

# 4. 运行对比
cd ../test/benchmark
./run_comparison.sh
```

**详细步骤** → [MIGRATION_CHECKLIST.md](./MIGRATION_CHECKLIST.md)

### 场景 2: 添加单个 API benchmark

```bash
cd test/benchmark

# 生成 benchmark
python3 generate_benchmark.py nppiSub 8u C1 RSfs --module arithmetic

# 编译
cd ../../build
make nppi_arithmetic_benchmark

# 测试
./benchmark/nppi_arithmetic_benchmark --benchmark_filter=Sub

# 完整对比
cd ../test/benchmark
./run_comparison.sh
```

**详细指南** → [EXPANSION_GUIDE.md](./EXPANSION_GUIDE.md)

### 场景 3: 批量添加某个模块的所有 API

```bash
cd test/benchmark

# 预览将要生成的文件
python3 batch_generate_benchmarks.py --module arithmetic --dry-run

# 实际生成
python3 batch_generate_benchmarks.py --module arithmetic

# 编译所有
cd ../../build
cmake .. -DBUILD_BENCHMARKS=ON
make -j$(nproc)

# 运行对比
cd ../test/benchmark
./run_comparison.sh
```

## 📊 当前进度

```
Unit Tests:    ~120+ APIs  ✅
Benchmarks:    1 API       ⚠️  (nppiAdd)
待添加:        ~119 APIs   📝

模块分布:
├── arithmetic:     35 APIs  (1 完成,  34 待添加)
├── filtering:      ~20 APIs (0 完成,  20 待添加)
├── geometry:       ~15 APIs (0 完成,  15 待添加)
├── color:          ~15 APIs (0 完成,  15 待添加)
├── statistics:     ~15 APIs (0 完成,  15 待添加)
└── others:         ~20 APIs (0 完成,  20 待添加)
```

## 🎯 推荐工作流程

### 阶段 1: 验证迁移（1-2 天）

1. 阅读 [MIGRATION_CHECKLIST.md](./MIGRATION_CHECKLIST.md)
2. 将测试框架复制到优化版 MPP
3. 编译并运行 `nppiAdd` benchmark
4. 验证对比脚本正常工作

**目标：** 确保测试框架在优化版 MPP 上正常工作

### 阶段 2: 快速扩展核心 API（1-2 周）

```bash
# 添加算术运算的核心函数
python3 generate_benchmark.py nppiSub 8u C1 RSfs --module arithmetic
python3 generate_benchmark.py nppiMul 8u C1 RSfs --module arithmetic
python3 generate_benchmark.py nppiDiv 8u C1 RSfs --module arithmetic
# ... 添加 10-15 个核心函数
```

**目标：** 覆盖 10-15 个最常用的 API

### 阶段 3: 批量扩展（2-4 周）

```bash
# 完成整个算术运算模块
python3 batch_generate_benchmarks.py --module arithmetic

# 逐步添加其他模块
python3 batch_generate_benchmarks.py --module filtering
python3 batch_generate_benchmarks.py --module geometry
# ...
```

**目标：** 覆盖所有有 unit test 的 API

### 阶段 4: 性能优化（持续）

1. 运行 `./run_comparison.sh`
2. 分析 CSV 结果，找出性能差的 API
3. 优化实现
4. 重新测试验证

**目标：** 大部分 API 达到 NVIDIA NPP 的 70%+ 性能

## 🔧 关键设计特点

### 1. 完全解耦的架构

```
测试框架              MPP 库实现
    |                     |
    |                     |
    +---> npp target <----+
          (抽象接口)
```

- 测试代码只依赖 `npp` target
- 不关心具体实现来自哪里
- 可以轻松切换 MPP/NVIDIA NPP

### 2. 自动化工具链

```
Unit Tests → batch_generate_benchmarks.py → Benchmark 代码
                                                |
                                                v
                                         CMake 编译
                                                |
                                                v
                                      run_comparison.sh
                                                |
                                                v
                                    compare_results.py
                                                |
                                                v
                                          CSV 报告
```

### 3. 标准化输出

所有 benchmark 产生统一的输出：

- **终端：** 带颜色的表格（Excellent/Good/Acceptable/NeedsOpt）
- **CSV：** 可导入 Excel 的详细数据
- **JSON：** 原始 benchmark 数据（用于自动化分析）

## 🐛 常见问题速查

| 问题 | 解决方案 | 文档链接 |
|------|---------|---------|
| 找不到 `npp` target | 检查 `src/CMakeLists.txt` 是否创建了 `npp` | [迁移清单](./MIGRATION_CHECKLIST.md#问题-1-找不到-npp-target) |
| 头文件找不到 | 添加正确的 include 目录 | [迁移清单](./MIGRATION_CHECKLIST.md#问题-2-头文件找不到) |
| 链接错误 | 检查函数签名和 extern "C" | [迁移清单](./MIGRATION_CHECKLIST.md#问题-3-链接错误) |
| 运行时崩溃 | 使用 cuda-memcheck 排查 | [迁移清单](./MIGRATION_CHECKLIST.md#问题-4-运行时崩溃) |
| 性能异常低 | 检查 kernel 配置和编译选项 | [README](./README.md#性能优化) |

## 📞 获取帮助

1. **查看文档**
   - 迁移问题 → [MIGRATION_CHECKLIST.md](./MIGRATION_CHECKLIST.md)
   - 扩展问题 → [EXPANSION_GUIDE.md](./EXPANSION_GUIDE.md)
   - 使用问题 → [README.md](./README.md)

2. **查看示例**
   - 参考已有的 `benchmark_nppi_add.cpp`
   - 使用 `BENCHMARK_TEMPLATE.cpp` 模板

3. **调试技巧**
   ```bash
   # 查看 CMake 配置
   cmake .. -LAH | grep NPP
   
   # 详细编译输出
   make VERBOSE=1
   
   # 检查链接
   ldd ./benchmark/nppi_arithmetic_benchmark
   
   # GPU 内存检查
   cuda-memcheck ./benchmark/nppi_arithmetic_benchmark
   ```

## 🎉 开始使用

```bash
# 克隆或更新代码
git pull

# 阅读迁移指南（如果需要迁移）
cat test/benchmark/MIGRATION_CHECKLIST.md

# 或直接开始添加 benchmark
cd test/benchmark
python3 generate_benchmark.py --help
```

---

**祝顺利完成 100+ API 的 benchmark 覆盖！** 🚀
