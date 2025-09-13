# NVIDIA NPP 测试失败列表

## 内存分配失败的测试
基于之前的运行结果，以下测试因为CUDA内存分配失败而失败：

### ExpFunctionalTest
- Exp_8u_C1RSfs_BasicOperation
- Exp_32f_C1R_BasicOperation  
- Exp_16s_C1RSfs_BasicOperation
- Exp_32f_C1R_SpecialValues
- Exp_32f_C1IR_InPlace
- Exp_StreamContext
- Exp_8u_C1RSfs_WithScaling

### LnFunctionalTest
- Ln_8u_C1RSfs_BasicOperation
- Ln_8u_C1RSfs_WithScaling
- Ln_32f_C1R_BasicOperation
- Ln_16s_C1RSfs_BasicOperation
- Ln_32f_C1R_SpecialValues
- Ln_32f_C1IR_InPlace
- Ln_StreamContext

### MulScaleFunctionalTest
- MulScale_8u_C1R_BasicOperation
- MulScale_16u_C1R_BasicOperation
- MulScale_8u_C1R_ExtremeValues
- MulScale_8u_C1IR_InPlace
- MulScale_StreamContext
- MulScale_16u_C1R_OverflowHandling

### SqrFunctionalTest
- Sqr_8u_C1RSfs_BasicOperation
- Sqr_8u_C1RSfs_WithScaling
- Sqr_32f_C1R_BasicOperation
- Sqr_16s_C1RSfs_BasicOperation
- Sqr_32f_C1R_SpecialValues
- Sqr_StreamContext

### SqrtFunctionalTest
- Sqrt_8u_C1RSfs_BasicOperation
- Sqrt_8u_C1RSfs_WithScaling
- Sqrt_32f_C1R_BasicOperation
- Sqrt_16s_C1RSfs_BasicOperation
- Sqrt_32f_C1R_SpecialValues
- Sqrt_StreamContext

### ColorConversionTest
- RGBToYUV_BasicColors
- YUVToRGB_RoundTrip

### ColorConversionParameterTest
- InvalidSize (参数验证错误)

### NV12ToRGBTest
- BasicNV12ToRGB_8u_P2C3R
- NV12ToRGB_709CSC_8u_P2C3R
- NV12ToRGB_709HDTV_8u_P2C3R
- VariousImageSizes

## 问题分析

1. **内存分配失败**: 大多数失败都是因为 `nppiMalloc_*` 函数返回 nullptr
2. **API不兼容**: 可能 NVIDIA NPP 没有某些 OpenNPP 实现的函数 (如 nppiLn, nppiExp, nppiMulScale 等)
3. **参数验证差异**: NVIDIA NPP 和 OpenNPP 对参数的验证可能不同

## 调试策略

1. 逐一测试每个失败的函数，检查其是否存在于 NVIDIA NPP 中
2. 对于存在的函数，分析内存分配和参数使用的差异
3. 对于不存在的函数，从测试中排除或提供替代方案