# NVIDIA NPP Gamma Implementation Analysis

## 发现

通过反向工程NVIDIA NPP库的实际行为，我们发现：

**NVIDIA NPP的`nppiGammaFwd`和`nppiGammaInv`函数不使用标准sRGB gamma校正！**

## 实际实现

NVIDIA NPP使用简单的power-law gamma校正：

- **Gamma值**: 1.87
- **Forward**: `output = input^(1/1.87)`
- **Inverse**: `output = input^1.87`

### 与标准sRGB的区别

标准sRGB (IEC 61966-2-1) 使用复杂的分段函数：

```cpp
// 标准sRGB Forward (Linear → sRGB)
if (linear <= 0.0031308)
    srgb = 12.92 * linear;
else
    srgb = 1.055 * pow(linear, 1/2.4) - 0.055;
```

NVIDIA NPP使用简化的power law：

```cpp
// NVIDIA NPP Forward
output = pow(input, 1/1.87);
```

## 验证方法

通过测试NVIDIA NPP实际输出反推：

| Input | NVIDIA Output | Standard sRGB | Simple Gamma 1.87 |
|-------|---------------|---------------|-------------------|
| 28    | 78            | 93            | **78** ✓          |
| 29    | 80            | 95            | **80** ✓          |
| 30    | 81            | 96            | **81** ✓          |
| 50    | 107           | 122           | **107** ✓         |
| 100   | 155           | 168           | **155** ✓         |

## MPP实现策略

为了100%兼容NVIDIA NPP，MPP采用相同的gamma 1.87实现。

## 参考

- 测试代码: `test/unit/nppi/nppi_color_conversion_operations/test_nppi_gamma.cpp`
- GPU实现: `src/nppi/nppi_color_conversion_operations/nppi_gamma.cu`
- 反向工程日期: 2025-12-02
