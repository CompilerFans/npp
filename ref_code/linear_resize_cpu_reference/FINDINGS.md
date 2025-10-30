# NVIDIA NPP Linear Interpolation Algorithm Reverse Engineering Findings

## Current Status

正在对NVIDIA NPP 12.6的linear interpolation算法进行逆向工程。

## 已发现的关键规律

### 1. 上采样修正系数 (Upscale, scale < 1.0)

对于上采样，NVIDIA NPP对标准双线性插值的权重进行了修正。

#### 测试案例：2x2 → 3x3 (scale=0.667)

**Pattern [0,100,50,150]:**
- fx=0.5, fy=0: NPP=42, StdBilin=50 → effective_fx=0.42, **modifier=0.84**
- fx=0, fy=0.5: NPP=21, StdBilin=25 → effective_fy=0.42, **modifier=0.84**
- fx=0.5, fy=0.5: NPP=63, StdBilin=75

**Pattern [0,127,127,255]:**
- fx=0.5, fy=0: NPP=53, StdBilin=63.5 → effective_fx≈0.4173, **modifier≈0.8346**
- fx=0, fy=0.5: NPP=53, StdBilin=63.5 → effective_fy≈0.4173, **modifier≈0.8346**
- fx=0.5, fy=0.5: NPP=106, StdBilin=127.25

**Pattern [0,255,0,255]:**
- fx=0.5, fy=0: NPP=106, StdBilin=127.5 → effective_fx≈0.4157, **modifier≈0.8314**

**关键发现：**
- 对于fx或fy=0.5的1D插值，修正系数在0.83-0.84之间
- 对于[0,100,50,150]模式，修正系数恰好为0.84 (42/50 = 0.84)
- 对于2D插值 (fx=0.5, fy=0.5)，使用mod_x=mod_y=0.84可以完美匹配NPP结果

#### 验证 2D 情况 (fx=0.5, fy=0.5, 修正系数=0.84)

Pattern [0,100,50,150]:
```
fx_eff = fy_eff = 0.5 * 0.84 = 0.42
v0 = p00*(1-fx_eff) + p10*fx_eff = 0*0.58 + 100*0.42 = 42
v1 = p01*(1-fx_eff) + p11*fx_eff = 50*0.58 + 150*0.42 = 29 + 63 = 92
result = v0*(1-fy_eff) + v1*fy_eff = 42*0.58 + 92*0.42 = 24.36 + 38.64 = 63 ✓
```

完美匹配NPP=63！

### 2. 修正系数与fx/fy值的关系

从test_different_scales的输出显示，修正系数不是恒定的：

**scale=0.75 (3x3→4x4):**
- fx=0.625: modifier≈0.813
- fx=0.375: modifier≈0.667

**scale=0.8 (4x4→5x5):**
- fx=0.7: modifier≈0.782
- fx=0.5: modifier≈0.698
- fx=0.3: modifier≈0.476

**关键观察：**
- 修正系数随fx值变化
- 修正系数也随scale值变化
- 需要更多数据来找出准确的公式

### 3. 下采样 (Downscale, scale >= 1.0)

综合测试显示，之前发现的"上采样算法"在下采样时完全失效 (0% match)。

这说明NVIDIA对下采样使用了完全不同的算法。

## 测试结果总结

### 成功的发现

1. **三模式架构**: 我们在`linear_resize_refactored.h`中实现的算法能与MPP完美匹配(100%)
   - UPSCALE mode: scale < 1.0
   - FRACTIONAL_DOWN mode: 1.0 ≤ scale < 2.0
   - LARGE_DOWN mode: scale ≥ 2.0

2. **NVIDIA NPP差异**: NVIDIA NPP与我们的三模式算法有44.4%的匹配率
   - 说明NVIDIA使用了不同的算法
   - 但我们的算法仍然是valid的线性插值变体

### 失败的尝试

1. **简单修正公式**: `modifier = 0.85 - 0.04*fx*fy`
   - 仅对[0,100,50,150] 2x2→3x3有效
   - 对其他分辨率和模式失败 (< 25% match)
   - **结论**: 过度拟合单一测试案例

2. **恒定修正系数0.84**:
   - 仅对fx=fy=0.5有效
   - 对其他fx/fy值不适用

## 下一步计划

### 方法 1: 继续逆向工程NVIDIA算法

需要收集更多数据点来找出NVIDIA的真实公式：
- 系统测试不同的fx值 (0.1, 0.2, 0.3, ..., 0.9)
- 系统测试不同的scale值 (0.5, 0.6, 0.7, 0.8, 0.9)
- 分析modifier(fx, scale)的函数关系

### 方法 2: 采用我们自己的算法

我们已经有一个工作良好的算法 (`linear_resize_refactored.h`):
- 100% 匹配我们的CPU reference
- 算法清晰、可理解、可维护
- 虽然与NVIDIA NPP有差异，但仍是合理的linear interpolation

**建议**:
- 将`linear_resize_refactored.h`的算法作为MPP的official实现
- 在文档中说明与NVIDIA NPP的差异
- 添加选项允许用户在strict NVIDIA compatibility和我们的算法之间选择

## 文件清单

### 分析工具
- `systematic_reverse_engineering.cpp`: 系统化数据收集
- `analyze_weight_patterns.cpp`: 权重模式分析
- `test_different_scales.cpp`: 不同scale测试
- `nvidia_analysis.cpp`: NVIDIA输出提取
- `reverse_engineer_weights.cpp`: 权重逆向工程

### CPU Reference实现
- `linear_resize_refactored.h`: **最终推荐版本** (100% MPP匹配)
- `linear_resize_nvidia_compatible.h`: 尝试的NVIDIA兼容版本 (失败)

### 测试程序
- `comprehensive_test.cpp`: 41个测试案例的全面测试
- `test_nvidia_compat.cpp`: NVIDIA兼容性验证

### 文档
- `NVIDIA_LINEAR_ALGORITHM_DISCOVERED.md`: 之前的发现 (部分错误，需修订)
- `FINDINGS.md`: 本文档

## 技术细节

### 坐标映射
中心对齐映射 (所有测试确认):
```cpp
float srcX = (dstX + 0.5f) * scaleX - 0.5f;
float srcY = (dstY + 0.5f) * scaleY - 0.5f;
```

### 舍入方式
NVIDIA使用`floor()`而不是`round()`:
```cpp
output = (uint8_t)floor(interpolated_value);
```

这与标准的round()+0.5不同，是关键差异之一。

## 结论

我们成功实现了高质量的linear interpolation算法，虽然与NVIDIA NPP不100%一致，但：
1. 算法正确且可验证
2. 性能应该相当
3. 代码清晰易维护

继续追求100% NVIDIA兼容性的成本效益比可能不高，除非有明确的业务需求。

建议采用我们自己的三模式算法作为MPP的标准实现。
