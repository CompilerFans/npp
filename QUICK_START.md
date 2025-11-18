# 🚀 Quick Start Guide

## 一键运行性能测试

### 在服务器上执行（首次使用）

```bash
# 1. 克隆项目
cd ~
git clone --recursive git@github.com:UniBoy222/npp.git
cd npp

# 2. 运行一键脚本
./quick_benchmark.sh
```

### 在服务器上执行（已有项目）

```bash
# 进入项目目录
cd ~/npp

# 运行一键脚本
./quick_benchmark.sh
```

就这么简单！✨

---

## 脚本会自动完成

1. ✅ **环境检查** - CMake, CUDA, GPU, Git
2. ✅ **代码更新** - git pull + submodule update
3. ✅ **清理构建** - 删除旧的 build 目录
4. ✅ **CMake 配置** - Release 模式 + NVIDIA NPP
5. ✅ **编译** - 使用所有 CPU 核心
6. ✅ **运行测试** - 5 次重复取平均值
7. ✅ **保存结果** - JSON 格式结果文件

---

## 输出示例

```
========================================
Step 1: Environment Check
========================================

→ Checking CMake version...
✓ CMake version: 3.28.1
→ Checking CUDA...
✓ CUDA version: 12.5
→ Checking GPU...
✓ GPU: NVIDIA GeForce RTX 4090
→ Checking Git...
✓ Git available

========================================
Step 2: Update Source Code
========================================

→ Pulling latest changes...
Already up to date.
→ Updating submodules...
✓ Source code updated
✓ GoogleTest submodule verified

========================================
Step 3: CMake Configuration
========================================

→ Running CMake configuration...
-- CMake version: 3.28.1
-- Using modern CUDAToolkit detection (CMake >= 3.17)
-- Found CUDA Toolkit: 12.5
✓ CMake configuration successful

========================================
Step 4: Compilation
========================================

→ Compiling with 12 cores...
[ 95%] Building CXX object test/benchmark/...
[100%] Linking CXX executable benchmark/nppi_arithmetic_benchmark
✓ Compilation successful
✓ Benchmark executable created

========================================
Step 5: Running Benchmarks
========================================

→ Running performance tests...

Running benchmark/nppi_arithmetic_benchmark
Run on (12 X 3600 MHz CPU s)
CPU Caches:
  L1 Data 32 KiB (x6)
  L1 Instruction 32 KiB (x6)
  L2 Unified 256 KiB (x6)
  L3 Unified 12288 KiB (x1)

---------------------------------------------------------------------------
Benchmark                                 Time             CPU   Iterations
---------------------------------------------------------------------------
BM_nppiAdd_8u_C1RSfs_Fixed_mean        0.125 ms        0.125 ms            5
BM_nppiAdd_8u_C1RSfs_Fixed_median      0.124 ms        0.124 ms            5
BM_nppiAdd_8u_C1RSfs_Fixed_stddev     0.0015 ms       0.0015 ms            5
...

✓ Benchmarks completed successfully

========================================
Summary
========================================

✓ All tests completed successfully!

→ System Information:
  - CMake: 3.28.1
  - CUDA: 12.5
  - GPU: NVIDIA GeForce RTX 4090

→ Results saved to:
  - JSON: benchmark_results/nvidia_npp_20231118_103245.json

→ Executable location:
  - build/benchmark/nppi_arithmetic_benchmark

→ To run again:
  cd build/benchmark
  ./nppi_arithmetic_benchmark

========================================
     Quick Benchmark Completed! 🎉
========================================
```

---

## 结果文件位置

测试结果自动保存在：

```
npp/benchmark_results/nvidia_npp_YYYYMMDD_HHMMSS.json
```

---

## 其他测试脚本

| 脚本 | 用途 |
|------|------|
| `quick_benchmark.sh` | 🚀 **一键测试** - 最简单（推荐） |
| `test/benchmark/run_nvidia_only.sh` | 只测试 NVIDIA NPP |
| `test/benchmark/run_comparison.sh` | MPP vs NVIDIA 对比 |
| `test/benchmark/run_performance_test.sh` | 完整测试套件 |

---

## 手动运行（高级）

如果需要更多控制：

```bash
cd ~/npp
rm -rf build && mkdir build && cd build

# 配置
cmake .. -DCMAKE_BUILD_TYPE=Release -DBUILD_BENCHMARKS=ON -DUSE_NVIDIA_NPP=ON

# 编译
make nppi_arithmetic_benchmark -j$(nproc)

# 运行
cd benchmark
./nppi_arithmetic_benchmark --help  # 查看所有选项
```

---

## 常见问题

### Q: 权限错误？
```bash
chmod +x quick_benchmark.sh
```

### Q: Git 相关错误？
```bash
cd ~/npp
git pull
git submodule update --init --recursive
```

### Q: CMake 版本太低？
```bash
# 见主 README 的 CMake 升级指南
```

### Q: 找不到 NPP 库？
```bash
# 脚本会自动检测，如果失败请检查 CUDA 安装
find /usr/local/cuda* -name "libnppc.so*"
```

---

## 需要帮助？

查看详细文档：
- `README.md` - 项目总览
- `docs/BENCHMARK_GUIDE.md` - 性能测试指南
- `CONTRIBUTING.md` - 开发指南

---

**祝测试顺利！** 🎉
