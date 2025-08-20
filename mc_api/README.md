# MetaX MPP API Headers

这个目录包含了从NVIDIA NPP转换而来的MetaX MPP (MetaX Performance Primitives) API头文件。

## 概述

MetaX MPP是基于NVIDIA NPP的GPU加速图像和信号处理库的MetaX实现版本。这些头文件定义了与NPP API兼容但使用MetaX/MACA命名约定的接口。

## 文件结构

### 核心头文件
- `mpp.h` - 主头文件，聚合所有其他头文件
- `mppcore.h` - 核心功能，库版本和设备属性查询
- `mppdefs.h` - 类型定义、宏定义和常量

### 图像处理头文件 (MPPI)
- `mppi.h` - 图像处理主头文件
- `mppi_arithmetic_and_logical_operations.h` - 算术和逻辑运算
- `mppi_color_conversion.h` - 颜色空间转换
- `mppi_data_exchange_and_initialization.h` - 数据交换和初始化
- `mppi_filtering_functions.h` - 滤波函数
- `mppi_geometry_transforms.h` - 几何变换
- `mppi_linear_transforms.h` - 线性变换
- `mppi_morphological_operations.h` - 形态学操作
- `mppi_statistics_functions.h` - 统计函数
- `mppi_support_functions.h` - 支持函数
- `mppi_threshold_and_compare_operations.h` - 阈值和比较操作

### 信号处理头文件 (MPPS)
- `mpps.h` - 信号处理主头文件
- `mpps_arithmetic_and_logical_operations.h` - 信号算术和逻辑运算
- `mpps_conversion_functions.h` - 信号转换函数
- `mpps_filtering_functions.h` - 信号滤波函数
- `mpps_initialization.h` - 信号初始化
- `mpps_statistics_functions.h` - 信号统计函数
- `mpps_support_functions.h` - 信号支持函数

## 命名转换规则

| NVIDIA NPP | MetaX MPP | 描述 |
|------------|-----------|------|
| NPP* | MPP* | 库前缀 |
| Npp* | Mpp* | 类型前缀 |
| nppi* | mppi* | 图像处理函数 |
| npps* | mpps* | 信号处理函数 |
| NVIDIA | MetaX | 公司名 |
| CUDA | MACA | 计算平台 |
| cuda* | maca* | 运行时函数 |

## 主要数据类型

### 基础数据类型
```c
typedef unsigned char    Mpp8u;     // 8位无符号整数
typedef signed char      Mpp8s;     // 8位有符号整数  
typedef unsigned short   Mpp16u;    // 16位无符号整数
typedef short            Mpp16s;    // 16位有符号整数
typedef unsigned int     Mpp32u;    // 32位无符号整数
typedef int              Mpp32s;    // 32位有符号整数
typedef float            Mpp32f;    // 32位浮点数
typedef double           Mpp64f;    // 64位浮点数
```

### 复数类型
```c
typedef struct { Mpp32f re, im; } Mpp32fc;  // 32位浮点复数
typedef struct { Mpp64f re, im; } Mpp64fc;  // 64位浮点复数
```

### 几何类型
```c
typedef struct { int x, y; } MppiPoint;           // 2D点
typedef struct { int width, height; } MppiSize;   // 2D尺寸
typedef struct { int x, y, width, height; } MppiRect; // 矩形
```

### 状态码
```c
typedef enum {
    MPP_NO_ERROR = 0,              // 成功
    MPP_NULL_POINTER_ERROR = -8,   // 空指针错误
    MPP_SIZE_ERROR = -6,           // 尺寸错误
    // ... 其他错误码
} MppStatus;
```

## 使用示例

### 基础用法
```c
#include "mpp.h"

int main() {
    // 初始化
    MppiSize imageSize = {640, 480};
    Mpp8u *d_src, *d_dst;
    
    // 分配设备内存
    mppiMalloc_8u_C1(&d_src, imageSize.width, imageSize.height);
    mppiMalloc_8u_C1(&d_dst, imageSize.width, imageSize.height);
    
    // 执行图像处理操作
    MppStatus status = mppiAdd_8u_C1RSfs(
        d_src1, imageSize.width, 
        d_src2, imageSize.width,
        d_dst, imageSize.width,
        imageSize, 0);
    
    if (status != MPP_NO_ERROR) {
        // 处理错误
    }
    
    // 清理
    mppiFreeMem(d_src);
    mppiFreeMem(d_dst);
    
    return 0;
}
```

### 流上下文使用
```c
// 使用自定义流
MppStreamContext streamCtx;
mppGetStreamContext(&streamCtx);

MppStatus status = mppiAdd_8u_C1RSfs_Ctx(
    d_src1, step1, d_src2, step2, d_dst, stepDst,
    imageSize, 0, streamCtx);
```

## 工具脚本

### convert_npp_to_mpp.sh
完整的NPP到MPP转换脚本，支持：
- 整个工程的一键转换
- 选择性转换（头文件、源文件、测试文件）
- 干运行模式预览更改
- 自动备份原文件

使用方法：
```bash
# 转换整个工程
./convert_npp_to_mpp.sh /path/to/npp/project

# 只转换头文件
./convert_npp_to_mpp.sh --headers-only /path/to/npp/api

# 预览更改（不实际修改文件）
./convert_npp_to_mpp.sh --dry-run /path/to/npp/project

# 转换到新目录
./convert_npp_to_mpp.sh /path/to/npp/project /path/to/mpp/project
```

## 测试隔离

参见 `test_isolation_design.md` 了解如何分离MPP测试和NVIDIA NPP对比测试：

- MPP专用测试：测试MPP库的功能正确性
- 对比测试：将MPP结果与NVIDIA NPP结果进行对比
- 配置控制：可以选择性地启用/禁用不同类型的测试

## 编译集成

在CMakeLists.txt中：
```cmake
# 设置MPP头文件路径
include_directories(${PROJECT_SOURCE_DIR}/mc_api)

# 链接MPP库
target_link_libraries(your_target mpp)

# 可选：启用NVIDIA对比测试
option(ENABLE_NVIDIA_NPP_COMPARISON "Enable NVIDIA NPP comparison tests" OFF)
```

## 注意事项

1. **头文件兼容性**: 这些头文件与NVIDIA NPP API保持结构兼容，但使用MetaX命名约定
2. **实现依赖**: 头文件只定义接口，需要相应的MPP库实现
3. **测试隔离**: 建议使用提供的测试隔离框架分离MPP测试和对比测试
4. **版本管理**: 基于NVIDIA NPP 12.3.3.65版本转换而来

## 版本信息

- **MPP版本**: 12.3.3 (基于NPP 12.3.3.65)
- **转换日期**: $(date)
- **转换工具**: Claude Code Assistant + convert_npp_to_mpp.sh

## 许可证

版权所有 © 2025 MetaX Corporation & Affiliates. 保留所有权利。