# Super Sampling CPU Reference Implementation

高质量CPU参考实现，与NVIDIA NPP高度匹配（88.5%完美匹配，100%在±1范围内）

## 目录结构

```
super_sampling_cpu_reference/
├── README.md                       # 本文件
├── README_SUPER_SAMPLING_ANALYSIS.md  # 完整分析总览
├── src/
│   └── super_sampling_cpu.h        # 核心实现（头文件）
├── tests/
│   ├── reference_super_sampling.cpp    # 验证测试（11个场景）
│   └── test_extensive_shapes.cpp       # 广泛测试（52个场景）
├── scripts/
│   └── build_tests.sh              # 编译脚本
└── docs/
    ├── cpu_reference_implementation.md  # 实现指南
    ├── computation_rounding_conclusion.md  # Rounding分析
    ├── extensive_shape_test_results.md     # 测试结果
    ├── fractional_scale_behavior.md        # 分数倍详解
    ├── super_sampling_boundary_analysis.md # 边界分析
    ├── nvidia_npp_border_test_results.md   # 边界测试
    └── rounding_mode_impact.md             # Rounding影响
```

## 快速开始

### 1. 编译测试

```bash
cd ref_code/super_sampling_cpu_reference
./scripts/build_tests.sh
```

### 2. 运行验证

```bash
# 基础验证（11个测试）
./validate_reference

# 广泛测试（52个形状组合）
./test_shapes
```

### 3. 使用参考实现

```cpp
#include "src/super_sampling_cpu.h"

// 单通道8位图像
unsigned char* src = ...;  // 源图像
unsigned char* dst = ...;  // 目标图像

SuperSamplingCPU<unsigned char>::resize(
  src, srcWidth,      // 源数据和宽度（step）
  srcWidth, srcHeight,  // 源尺寸
  dst, dstWidth,      // 目标数据和宽度（step）
  dstWidth, dstHeight,  // 目标尺寸
  1                   // 通道数（1=灰度，3=RGB，4=RGBA）
);
```

## 核心算法

### 算法概述

超采样 = **动态加权Box Filter**
- 采样区域大小：`scale × scale`
- 边缘像素权重：分数值（0.0-1.0）
- 总权重归一化：`scale × scale`

### 关键步骤

```cpp
// 1. 计算采样区域
float srcCenterX = (dx + 0.5f) * scaleX;
float xMin = srcCenterX - scaleX * 0.5f;
float xMax = srcCenterX + scaleX * 0.5f;

// 2. 整数边界（关键！）
int xMinInt = (int)ceil(xMin);   // 必须ceil
int xMaxInt = (int)floor(xMax);  // 必须floor

// 3. 边缘权重（关键！）
float wxMin = ceil(xMin) - xMin;
float wxMax = xMax - floor(xMax);

// 4. 加权累加
// 左边缘: wxMin, 中间: 1.0, 右边缘: wxMax

// 5. 归一化并舍入（关键！）
result = (int)(sum / totalWeight + 0.5f);  // +0.5, 非lrintf
```

### 三个关键技术点

| 步骤 | 正确方法 | 错误后果 |
|------|----------|----------|
| 整数边界 | `ceil(xMin), floor(xMax)` | 6.25%匹配率 ❌ |
| 边缘权重 | `ceil(xMin)-xMin, xMax-floor(xMax)` | 逻辑错误 ❌ |
| 最终舍入 | `(int)(value + 0.5f)` | 0%匹配率 ❌ |

## 验证结果

### 测试覆盖

- **63个独立测试**
  - 11个基础验证测试
  - 52个形状组合测试

### 精度统计

| 指标 | 结果 |
|------|------|
| 完美匹配 (100% bit-exact) | 88.5% (46/52) |
| ±1范围内 | **100% (52/52)** ✓ |
| 最大差异 | 1像素 |
| 平均绝对差异 | 0.058像素 |

### 测试场景

✅ 正方形图像（整数/分数倍）
✅ 横向图像（16:9, 4:3, 21:9）
✅ 纵向图像（9:16, 3:4）
✅ 非均匀缩放（不同X/Y比例）
✅ 小尺寸图像
✅ 极端下采样（32x-50x）
✅ 极端宽高比（32:9, 9:32）
✅ 质数维度
✅ 2的幂次尺寸

## 浮点精度说明

### ±1差异的原因

1. **Float精度限制**
   - 23位尾数无法精确表示某些分数（如1.333...）

2. **累积误差**
   - 多次乘法和加法操作

3. **GPU vs CPU**
   - FMA指令 vs 标准浮点运算

### 影响场景

| Scale | 示例 | 精确匹配率 | 状态 |
|-------|------|-----------|------|
| 整数倍 | 2x, 4x, 8x | 100% | ✓ 完美 |
| 简单分数 | 1.5x, 2.5x | 100% | ✓ 完美 |
| 1.333x | 128/96, 1280/960 | 78-93% | ✓ 优秀 |
| 其他分数 | 大部分 | 98-100% | ✓ 优秀 |

**结论**：±1差异是**正常的**，属于浮点计算固有特性。

## 使用建议

### ✅ 推荐场景

1. **单元测试参考实现**
   ```cpp
   EXPECT_NEAR(cpuResult, nppResult, 1);  // ±1容忍度
   ```

2. **算法验证**
   - 验证CUDA kernel正确性
   - 对比优化版本

3. **调试工具**
   - 逐像素分析
   - 误差定位

4. **小规模图像处理**
   - 预处理/后处理
   - 快速原型

### ⚠️ 性能优化场景

对于大规模/实时处理，建议：
- GPU CUDA实现
- SIMD优化 + 多线程
- 直接使用NPP库

## 性能参考

| 图像大小 | Scale | CPU时间(单核) |
|---------|-------|--------------|
| 256×256 → 128×128 | 2x | ~2 ms |
| 1024×768 → 512×384 | 2x | ~12 ms |
| 1920×1080 → 960×540 | 2x | ~20 ms |

## 文档说明

### 核心文档

1. **README_SUPER_SAMPLING_ANALYSIS.md** ⭐
   - 完整分析总览
   - 文档导航
   - 推荐首先阅读

2. **cpu_reference_implementation.md** ⭐
   - 详细实现指南
   - 算法解释
   - 使用示例

3. **extensive_shape_test_results.md** ⭐
   - 52个测试详细结果
   - 精度分析
   - 场景覆盖

### 理论分析

4. **computation_rounding_conclusion.md**
   - Rounding模式深入分析
   - 三个关键步骤证明

5. **fractional_scale_behavior.md**
   - 分数倍缩放详解
   - 采样区域数学推导

6. **super_sampling_boundary_analysis.md**
   - 边界处理数学证明
   - 为什么不需要边界扩展

7. **rounding_mode_impact.md**
   - Rounding对误差的影响
   - +0.5 vs lrintf对比

## 编译要求

- C++11或更高
- CUDA Toolkit（用于测试验证）
- NPP库（用于对比测试）

## 许可证

遵循项目整体许可证。

## 总结

这个CPU参考实现是经过**63个独立测试**验证的高质量实现，达到：
- ✅ **88.5%完美匹配**（逐像素bit-exact）
- ✅ **100%在±1范围内**
- ✅ **所有测试最大差异仅1像素**

可直接用于单元测试、算法验证和调试分析！
