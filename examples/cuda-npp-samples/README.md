# CUDA NPP Samples

这个目录包含从NVIDIA CUDA Samples中复制的NPP相关示例，展示了如何使用NVIDIA Performance Primitives (NPP) 库进行高性能图像和信号处理。

## 示例概览

### 图像处理示例

1. **histEqualizationNPP** - 直方图均衡化
   - 演示如何使用NPP进行图像直方图均衡化
   - 关键概念：图像处理、性能优化、NPP库

2. **cannyEdgeDetectorNPP** - Canny边缘检测
   - 使用NPP的Canny边缘检测滤波器
   - 展示了多步骤边缘检测技术的组合应用
   - 关键概念：边缘检测、图像滤波、性能优化

3. **boxFilterNPP** - 盒式滤波
   - 演示NPP的盒式滤波功能
   - 基础的图像模糊和噪声减少技术

4. **FilterBorderControlNPP** - 边界控制滤波
   - 展示NPP滤波器的边界处理控制
   - 演示不同边界处理模式的效果

5. **watershedSegmentationNPP** - 分水岭分割
   - 使用NPP实现分水岭图像分割算法
   - 高级图像分割技术

6. **freeImageInteropNPP** - FreeImage互操作
   - 演示NPP与FreeImage库的集成使用
   - 图像I/O和格式转换

## 目录结构

```
cuda-npp-samples/
├── Common/                    # 通用工具和数据
│   ├── UtilNPP/              # NPP工具类和封装
│   ├── data/                 # 测试图像数据
│   └── helper_*.h            # CUDA助手函数
├── histEqualizationNPP/      # 直方图均衡化示例
├── cannyEdgeDetectorNPP/     # Canny边缘检测示例
├── boxFilterNPP/             # 盒式滤波示例
├── FilterBorderControlNPP/   # 边界控制滤波示例
├── watershedSegmentationNPP/ # 分水岭分割示例
└── freeImageInteropNPP/      # FreeImage互操作示例
```

## 构建要求

### 依赖项
- CUDA Toolkit (版本12.0或更高)
- NPP库 (随CUDA Toolkit提供)
- FreeImage库 (某些示例需要)
- CMake 3.10或更高版本

### 支持的架构
- SM 5.0, 5.2, 5.3, 6.0, 6.1, 7.0, 7.2, 7.5, 8.0, 8.6, 8.7, 8.9, 9.0

## 构建指南

### 构建所有示例
```bash
mkdir build && cd build
cmake ..
make -j$(nproc)
```

### 构建特定示例
```bash
cd histEqualizationNPP
mkdir build && cd build
cmake ..
make
```

## 运行示例

每个示例都可以独立运行，大多数示例会：
1. 加载测试图像数据
2. 应用NPP处理函数
3. 保存或显示处理结果
4. 输出性能统计信息

## UtilNPP工具库

Common/UtilNPP目录包含了有用的C++封装类：
- **Image类**: 图像数据管理
- **ImageAllocators**: 内存分配管理
- **Signal类**: 信号数据管理
- **Exceptions**: 错误处理

这些工具类简化了NPP函数的使用，提供了RAII内存管理和异常安全的错误处理。

## 学习建议

1. **从基础开始**: 先学习boxFilterNPP了解基本的NPP使用模式
2. **理解内存管理**: 注意GPU内存分配和数据传输
3. **掌握错误处理**: 学习NPP的错误码和异常处理
4. **性能优化**: 观察示例中的性能测量和优化技巧
5. **扩展应用**: 基于示例代码开发自己的图像处理应用

## 注意事项

- 确保CUDA设备支持所需的计算能力
- 某些示例需要FreeImage库进行图像I/O
- 建议在构建前检查CUDA和NPP的版本兼容性
- 示例数据文件位于Common/data目录中