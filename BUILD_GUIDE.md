# NPP构建指南

简单的NPP库构建说明。

## 快速构建

```bash
# 默认Release构建
./build.sh

# Debug构建
./build.sh --debug

# 指定并行作业数
./build.sh -j 8

# 清理构建
./build.sh clean
```

## 运行测试

所有测试配置由CMake控制。构建完成后，在build目录下使用ctest运行：

```bash
./build.sh
cd build
ctest
```

## 要求

- CMake 3.18+
- CUDA Toolkit
- C++17编译器