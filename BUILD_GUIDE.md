# NPP库构建指南

这是NPP (NVIDIA Performance Primitives) 库的构建和测试指南。

## 快速开始

使用我们的自动化构建脚本：

```bash
# 构建并运行功能测试（推荐新用户）
./build.sh

# 显示所有选项
./build.sh --help
```

## 构建目标

### 1. 功能测试（推荐）
构建并运行独立的功能测试，不依赖外部NVIDIA NPP库：

```bash
./build.sh functional
```

### 2. 对比测试
构建并运行与NVIDIA NPP库的对比测试：

```bash
./build.sh comparison
```

### 3. 全部测试
构建并运行所有类型的测试：

```bash
./build.sh all
```

### 4. 仅构建库
只构建NPP库本身：

```bash
./build.sh library
```

### 5. 清理
清理所有构建文件：

```bash
./build.sh clean
```

## 常用选项

### 并行构建
使用多个CPU核心加速构建：

```bash
./build.sh functional -j 8
```

### Debug模式
构建Debug版本用于调试：

```bash
./build.sh functional --debug
```

### 仅构建不运行测试
```bash
./build.sh functional --no-run
```

### 指定GPU架构
```bash
./build.sh functional --gpu-arch "75,86"
```

## 构建要求

### 系统要求
- Linux系统（推荐Ubuntu 20.04+）
- CMake 3.18+
- C++17支持的编译器（GCC 7+ 或 Clang 10+）
- CUDA Toolkit 11.0+ （推荐12.0+）
- NVIDIA GPU（支持计算能力7.0+）

### 依赖检查
构建脚本会自动检查所有依赖项：

```bash
./build.sh functional  # 会自动检查依赖
```

## 测试结果

### 功能测试状态
- ✅ **算术常量运算**: AddC, SubC, MulC, DivC - 全部通过
- ✅ **NPP核心功能**: 版本、GPU属性、内存管理 - 全部通过  
- ✅ **矩阵转置**: 各种尺寸和边界条件 - 全部通过
- ⚠️ **双源算术**: 部分通过（Sub函数待修复）
- 📊 **总体通过率**: 51/56测试通过（91%）

### 测试套件说明

1. **功能测试**（test/functional/）
   - 纯功能验证，不依赖NVIDIA NPP
   - 测试API正确性和边界条件
   - 适合开发和CI环境

2. **对比测试**（test/comparison/）
   - 与NVIDIA NPP库结果对比
   - 验证数值精度和兼容性
   - 需要安装NVIDIA NPP库

3. **遗留测试**（test/legacy/）
   - 向后兼容的测试代码
   - 包含历史测试用例
   - 仅用于回归测试

## 故障排除

### CMake配置失败
```bash
# 清理后重试
./build.sh clean
./build.sh functional
```

### CUDA编译警告
脚本会显示CUDA编译警告，这些通常不影响功能，可以忽略：
- `incompatible redefinition for option 'optimize'`
- `Support for offline compilation for architectures prior to '<compute/sm/lto>_75'`

### 内存不足
对于内存较小的系统，减少并行作业数：

```bash
./build.sh functional -j 2
```

### GPU架构不匹配
如果GPU架构不匹配，指定正确的架构：

```bash
# 对于GTX 1060 (Pascal)
./build.sh functional --gpu-arch "60"

# 对于RTX 2070 (Turing)  
./build.sh functional --gpu-arch "75"

# 对于RTX 3080 (Ampere)
./build.sh functional --gpu-arch "86"
```

## 开发工作流

### 日常开发
```bash
# 快速功能验证
./build.sh functional --no-run -j 8

# 运行特定测试
cd test/functional/build
./npp_functional_tests --gtest_filter="*AddC*"
```

### 性能测试
```bash
# Release模式构建
./build.sh functional --release

# 与NVIDIA NPP对比
./build.sh comparison
```

### 调试问题
```bash
# Debug模式构建
./build.sh functional --debug

# 详细输出
./build.sh functional --verbose
```

## 贡献指南

### 添加新测试
1. 在`test/functional/`对应模块下添加测试文件
2. 更新`CMakeLists.txt`包含新文件
3. 运行`./build.sh functional`验证

### 修复构建问题
1. 使用`./build.sh clean`清理环境
2. 尝试最小化复现
3. 检查依赖和版本兼容性

## 构建统计

脚本会显示详细的构建统计信息：
- 源文件和头文件数量
- 代码总行数
- 构建时间
- GPU和CUDA信息

这些信息有助于了解项目规模和构建性能。

## 支持和反馈

如果遇到构建问题：

1. 查看控制台错误信息
2. 检查依赖是否正确安装
3. 尝试使用`./build.sh clean`清理重建
4. 参考故障排除章节
5. 提交Issue时包含完整的错误日志