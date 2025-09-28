# OpenCV NPP API缺失详细清单

## 分析基准
- **OpenCV需要的API总数**: 446个
- **当前MPP已实现**: 约267个 (60%)
- **缺失API总数**: 约179个 (40%)

## 缺失API详细分类

### 1. 直方图相关 (36个缺失)

#### 1.1 HistogramEven系列 (已部分实现)
```
✅ 已实现:
- nppiHistogramEven_8u_C1R
- nppiHistogramEven_8u_C4R  
- nppiHistogramEven_16s_C1R
- nppiHistogramEven_16s_C4R
- nppiHistogramEven_16u_C1R
- nppiHistogramEven_16u_C4R

❌ 缺失 (12个):
- nppiHistogramEvenGetBufferSize_8u_C1R
- nppiHistogramEvenGetBufferSize_8u_C1R_Ctx
- nppiHistogramEvenGetBufferSize_8u_C4R  
- nppiHistogramEvenGetBufferSize_8u_C4R_Ctx
- nppiHistogramEvenGetBufferSize_16s_C1R
- nppiHistogramEvenGetBufferSize_16s_C1R_Ctx
- nppiHistogramEvenGetBufferSize_16s_C4R
- nppiHistogramEvenGetBufferSize_16s_C4R_Ctx
- nppiHistogramEvenGetBufferSize_16u_C1R
- nppiHistogramEvenGetBufferSize_16u_C1R_Ctx
- nppiHistogramEvenGetBufferSize_16u_C4R
- nppiHistogramEvenGetBufferSize_16u_C4R_Ctx
```

#### 1.2 HistogramRange系列 (全部缺失)
```
❌ 缺失 (24个):
主要API:
- nppiHistogramRange_8u_C1R
- nppiHistogramRange_8u_C1R_Ctx
- nppiHistogramRange_8u_C4R
- nppiHistogramRange_8u_C4R_Ctx
- nppiHistogramRange_16s_C1R
- nppiHistogramRange_16s_C1R_Ctx
- nppiHistogramRange_16s_C4R
- nppiHistogramRange_16s_C4R_Ctx
- nppiHistogramRange_16u_C1R
- nppiHistogramRange_16u_C1R_Ctx
- nppiHistogramRange_16u_C4R
- nppiHistogramRange_16u_C4R_Ctx
- nppiHistogramRange_32f_C1R
- nppiHistogramRange_32f_C1R_Ctx
- nppiHistogramRange_32f_C4R
- nppiHistogramRange_32f_C4R_Ctx

BufferSize API:
- nppiHistogramRangeGetBufferSize_*系列 (12个)
```

### 2. 伽马校正 (16个全部缺失)

```
❌ 缺失 (16个):
前向伽马:
- nppiGammaFwd_8u_C3R
- nppiGammaFwd_8u_C3R_Ctx
- nppiGammaFwd_8u_C3IR
- nppiGammaFwd_8u_C3IR_Ctx
- nppiGammaFwd_8u_AC4R
- nppiGammaFwd_8u_AC4R_Ctx
- nppiGammaFwd_8u_AC4IR
- nppiGammaFwd_8u_AC4IR_Ctx

逆向伽马:
- nppiGammaInv_8u_C3R
- nppiGammaInv_8u_C3R_Ctx  
- nppiGammaInv_8u_C3IR
- nppiGammaInv_8u_C3IR_Ctx
- nppiGammaInv_8u_AC4R
- nppiGammaInv_8u_AC4R_Ctx
- nppiGammaInv_8u_AC4IR
- nppiGammaInv_8u_AC4IR_Ctx
```

### 3. 形态学操作 (16个全部缺失)

```
当前实现状态:
✅ 已实现: 基本的3x3和任意kernel的dilate/erode
❌ 缺失OpenCV需要的特定API (16个):

Dilate系列:
- nppiDilate_8u_C1R
- nppiDilate_8u_C1R_Ctx
- nppiDilate_8u_C4R  
- nppiDilate_8u_C4R_Ctx
- nppiDilate_32f_C1R
- nppiDilate_32f_C1R_Ctx
- nppiDilate_32f_C4R
- nppiDilate_32f_C4R_Ctx

Erode系列:
- nppiErode_8u_C1R
- nppiErode_8u_C1R_Ctx
- nppiErode_8u_C4R
- nppiErode_8u_C4R_Ctx
- nppiErode_32f_C1R
- nppiErode_32f_C1R_Ctx
- nppiErode_32f_C4R
- nppiErode_32f_C4R_Ctx
```

### 4. 滤波器扩展 (12个全部缺失)

```
当前实现状态:
✅ 已实现: nppiFilterBox_8u_C1R
❌ 缺失OpenCV需要的 (12个):

FilterBox扩展:
- nppiFilterBox_8u_C4R
- nppiFilterBox_8u_C4R_Ctx
- nppiFilterBox_32f_C1R
- nppiFilterBox_32f_C1R_Ctx

极值滤波:
- nppiFilterMax_8u_C1R
- nppiFilterMax_8u_C1R_Ctx
- nppiFilterMax_8u_C4R
- nppiFilterMax_8u_C4R_Ctx
- nppiFilterMin_8u_C1R
- nppiFilterMin_8u_C1R_Ctx
- nppiFilterMin_8u_C4R
- nppiFilterMin_8u_C4R_Ctx
```

### 5. 逻辑运算扩展 (18个全部缺失)

```
当前实现状态:
✅ 已实现: 基本的And/Or/Xor单通道操作
❌ 缺失OpenCV需要的多通道支持 (18个):

AndC系列:
- nppiAndC_8u_C3R
- nppiAndC_8u_C3R_Ctx
- nppiAndC_16u_C3R
- nppiAndC_16u_C3R_Ctx
- nppiAndC_16u_C4R
- nppiAndC_16u_C4R_Ctx
- nppiAndC_32s_C3R
- nppiAndC_32s_C3R_Ctx
- nppiAndC_32s_C4R
- nppiAndC_32s_C4R_Ctx

OrC系列:
- nppiOrC_8u_C3R
- nppiOrC_8u_C3R_Ctx
- nppiOrC_16u_C3R
- nppiOrC_16u_C3R_Ctx
- nppiOrC_16u_C4R
- nppiOrC_16u_C4R_Ctx
- nppiOrC_32s_C3R
- nppiOrC_32s_C3R_Ctx
- nppiOrC_32s_C4R
- nppiOrC_32s_C4R_Ctx

XorC系列:
- nppiXorC_8u_C3R
- nppiXorC_8u_C3R_Ctx
- nppiXorC_16u_C3R
- nppiXorC_16u_C3R_Ctx
- nppiXorC_16u_C4R
- nppiXorC_16u_C4R_Ctx
- nppiXorC_32s_C3R
- nppiXorC_32s_C3R_Ctx
- nppiXorC_32s_C4R
- nppiXorC_32s_C4R_Ctx
```

### 6. 统计分析扩展 (10个全部缺失)

```
当前实现状态:
✅ 已实现: nppiMean_StdDev_8u_C1R, nppiMean_StdDev_32f_C1R
❌ 缺失OpenCV需要的 (10个):

BufferSize API:
- nppiMeanStdDevGetBufferHostSize_8u_C1R
- nppiMeanStdDevGetBufferHostSize_8u_C1R_Ctx
- nppiMeanStdDevGetBufferHostSize_8u_C1MR
- nppiMeanStdDevGetBufferHostSize_8u_C1MR_Ctx
- nppiMeanStdDevGetBufferHostSize_32f_C1R
- nppiMeanStdDevGetBufferHostSize_32f_C1R_Ctx
- nppiMeanStdDevGetBufferHostSize_32f_C1MR
- nppiMeanStdDevGetBufferHostSize_32f_C1MR_Ctx

区域统计:
- nppiRectStdDev_32s32f_C1R
- nppiRectStdDev_32s32f_C1R_Ctx
```

### 7. 移位操作扩展 (部分缺失)

```
当前实现状态:
✅ 已实现: 基本的LShiftC/RShiftC操作
❌ 需要检查的特定数据类型支持:
- nppiRShiftC_8s_* 系列 (6个)
```

### 8. 其他专用功能 (61个)

#### 8.1 通道操作 (2个)
```
❌ 缺失:
- nppiSwapChannels_8u_C4IR
- nppiSwapChannels_8u_C4IR_Ctx
```

#### 8.2 窗口统计 (4个)
```
❌ 缺失:
- nppiSumWindowColumn_8u32f_C1R
- nppiSumWindowColumn_8u32f_C1R_Ctx
- nppiSumWindowRow_8u32f_C1R
- nppiSumWindowRow_8u32f_C1R_Ctx
```

#### 8.3 Graph Cut分割 (8个)
```
❌ 缺失:
- nppiGraphcut_32f8u
- nppiGraphcut_32s8u
- nppiGraphcut8_32f8u
- nppiGraphcut8_32s8u
- nppiGraphcutGetSize
- nppiGraphcut8GetSize
- nppiGraphcutInitAlloc
- nppiGraphcutFree
```

#### 8.4 专用变换St系列 (47个)
```
❌ 缺失大量St*系列API:
- nppiStDecimate_* 系列 (15个)
- nppiStIntegral_* 系列 (6个)
- nppiStTranspose_* 系列 (14个)
- nppiSt*其他系列 (12个)
```

## 实现优先级建议

### 第一优先级 (P0) - 立即实现
1. **直方图BufferSize API** - 12个 (扩展现有模块)
2. **伽马校正API** - 16个 (新建模块)
3. **统计BufferSize API** - 8个 (扩展现有模块)

### 第二优先级 (P1) - 近期实现  
1. **形态学扩展** - 16个 (适配现有实现)
2. **滤波器扩展** - 12个 (新增极值滤波)
3. **逻辑运算多通道** - 18个 (扩展现有模块)

### 第三优先级 (P2) - 中期实现
1. **通道操作** - 2个
2. **窗口统计** - 4个
3. **直方图Range系列** - 24个

### 第四优先级 (P3) - 长期实现
1. **Graph Cut** - 8个
2. **专用变换St系列** - 47个

## 技术实现策略

### 快速实现方案
1. **BufferSize API**: 大多数可以基于现有算法计算内存需求
2. **多通道扩展**: 复用单通道kernel，添加通道循环
3. **伽马校正**: 相对独立的新功能，可并行开发

### 代码复用策略
1. 利用现有的arithmetic_executor框架
2. 扩展现有morphology和filter实现
3. 统一错误处理和context管理模式

此优先级排序确保首先实现OpenCV最核心的依赖API，逐步完善整个生态系统。