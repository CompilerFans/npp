# MPP 私有开发的NVIDIA Performance Primitives实现


MPP是NVIDIA Performance Primitives (NPP)的私有开发实现，提供与NVIDIA闭源NPP库100%兼容的GPU加速图像和信号处理功能。

### 基本使用

```cpp
#include "npp.h"

// 初始化CUDA设备
cudaSetDevice(0);

// 分配图像内存
Npp8u* pSrc = nppiMalloc_8u_C1(width, height, &srcStep);
Npp8u* pDst = nppiMalloc_8u_C1(width, height, &dstStep);

// 执行图像加法运算
NppiSize roiSize = {width, height};
nppiAddC_8u_C1RSfs(pSrc, srcStep, 50, pDst, dstStep, roiSize, 1);

// 清理资源
nppiFree(pSrc);
nppiFree(pDst);
```

## 测试框架

MPP采用**三方对比验证**确保实现质量:

```
MPP ←→ NVIDIA NPP库 ←→ CPU参考实现
      ↘         ↙         ↗
        验证框架测试
       (逐位精度对比)
```