# NVIDIA NPP兼容性测试指南

本文档介绍如何在OpenNPP项目中使用NVIDIA NPP库进行兼容性测试。

## 概述

OpenNPP项目支持在编译时选择使用OpenNPP库或NVIDIA NPP库来链接测试程序。这允许开发者：
- 验证OpenNPP与NVIDIA NPP的API兼容性
- 对比两个库的功能实现
- 在相同的测试代码下运行不同的NPP实现

## 使用方法

### 1. 使用OpenNPP库（默认）

默认情况下，测试程序会链接OpenNPP库：

```bash
./build.sh
```

或者显式指定：

```bash
cmake .. -DUSE_NVIDIA_NPP=OFF
```

### 2. 使用NVIDIA NPP库

要使用NVIDIA NPP库运行测试，使用以下命令：

```bash
./build.sh --use-nvidia-npp
```

或者使用CMake直接配置：

```bash
cmake .. -DUSE_NVIDIA_NPP=ON
```

## 工作原理

当启用`USE_NVIDIA_NPP`选项时：
- 测试程序将链接NVIDIA NPP库（如nppial, nppicc, npps等）
- 使用CUDA Toolkit中的NPP头文件
- 所有测试代码保持不变，只是链接的库不同

## 前提条件

使用NVIDIA NPP库需要：
- 安装CUDA Toolkit（包含NPP库）
- NPP库通常位于`$CUDA_TOOLKIT_ROOT_DIR/lib64`
- NPP头文件位于`$CUDA_TOOLKIT_ROOT_DIR/include`

## 示例

### 编译并运行所有测试（使用OpenNPP）
```bash
./build.sh
./build/unit_tests
```

### 编译并运行所有测试（使用NVIDIA NPP）
```bash
./build.sh --use-nvidia-npp
./build/unit_tests
```

### 对比测试结果
```bash
# 使用OpenNPP运行测试
./build.sh clean
./build.sh
./build/unit_tests > opennpp_results.txt

# 使用NVIDIA NPP运行测试
./build.sh clean
./build.sh --use-nvidia-npp
./build/unit_tests > nvidia_npp_results.txt

# 对比结果
diff opennpp_results.txt nvidia_npp_results.txt
```

## 注意事项

1. **API兼容性**：OpenNPP旨在提供与NVIDIA NPP兼容的API，但可能存在细微差异
2. **性能差异**：两个库的性能可能不同，这是正常的
3. **功能覆盖**：
   - OpenNPP可能尚未实现所有NVIDIA NPP的功能
   - OpenNPP也实现了一些NVIDIA NPP没有的扩展功能（如nppiLog、nppiPow、nppiArcTan等）
4. **测试失败**：
   - 如果使用NVIDIA NPP时测试失败，可能是因为：
     - OpenNPP的实现与NVIDIA NPP存在差异
     - 测试使用了OpenNPP的扩展功能，而NVIDIA NPP不支持

## OpenNPP扩展功能

以下函数是OpenNPP实现的扩展功能，NVIDIA NPP不提供：
- `nppiLog_*` - 常用对数运算
- `nppiPow_*` - 幂运算
- `nppiArcTan_*` - 反正切运算
- `nppiLn_*` - 自然对数运算（如果NVIDIA NPP不支持）
- `nppiExp_*` - 指数运算（如果NVIDIA NPP不支持）
- `nppiMulScale_*` - 乘法缩放运算

使用NVIDIA NPP编译时，包含这些函数的测试将无法编译。

## 调试提示

如果遇到链接错误：
```bash
# 检查NVIDIA NPP库是否存在
ls $CUDA_TOOLKIT_ROOT_DIR/lib64/libnpp*

# 查看具体的链接命令
make VERBOSE=1
```

## 扩展用途

这个功能不仅可用于测试，还可以：
- 验证OpenNPP的正确性
- 性能基准测试对比
- 回归测试
- API兼容性验证