# 快速开始指南

## 5分钟上手

### 1. 最简单的使用

```cpp
#include "linear_resize_cpu_final.h"

// 准备数据
uint8_t src[640 * 480];  // 源图像
uint8_t dst[320 * 240];  // 目标图像

// 调用resize
LinearResizeCPU<uint8_t>::resize(
    src, 640,        // 源数据和步长
    640, 480,       // 源宽高
    dst, 320,       // 目标数据和步长
    320, 240,       // 目标宽高
    1               // 通道数(灰度=1)
);
```

### 2. RGB图像缩放

```cpp
uint8_t srcRGB[640 * 480 * 3];
uint8_t dstRGB[320 * 240 * 3];

LinearResizeCPU<uint8_t>::resize(
    srcRGB, 640 * 3,  // 步长 = 宽度 × 3
    640, 480,
    dstRGB, 320 * 3,
    320, 240,
    3                 // RGB = 3通道
);
```

### 3. 运行测试

```bash
# 编译测试
./compile_cpu_vs_npp.sh

# 运行对比测试
./test_cpu_vs_npp
```

## 期望结果

运行`test_cpu_vs_npp`应该看到：

```
✓ PERFECT 场景:
  • 2x upscale
  • 0.5x/0.33x/0.25x downscale
  • 0.67x fractional downscale
  • 非均匀 2x×0.5x
  • RGB/RGBA 2x和0.5x
  • Same size
  • 大图像downscale

✗ 已知差异:
  • 3x/4x/5x upscale (NPP专有算法)
  • 部分分数下采样 (参数精度)
```

## 常见问题

### Q: 哪个实现文件应该使用？

**A**: 推荐 `linear_resize_cpu_final.h`

- 最清晰的代码结构
- 生产就绪
- 50%场景完美匹配
- 关键场景100%匹配

### Q: 为什么有些场景不是100%匹配？

**A**: NPP在某些场景使用专有算法：

1. **大整数上采样** (3x/4x/5x): NPP使用修改的插值权重
2. **分数下采样** (0.75x/0.8x): 参数精度需要进一步校准
3. **非整数上采样**: NPP可能使用修改权重

对于这些场景，差异很小（最大2-13像素），视觉上几乎无法察觉。

### Q: 性能如何？

**A**: 这是参考实现，优先考虑正确性和清晰度：

- ✅ 算法正确
- ✅ 代码清晰
- ⚠️ 未优化性能
- ⚠️ 适合小图像或低频处理

对于高性能需求，建议：
- 使用NPP GPU实现
- 或基于此实现做SIMD优化

### Q: 如何验证我的使用是正确的？

**A**: 运行test并检查关键场景：

```bash
./test_cpu_vs_npp | grep "✓ PERFECT"
```

应该看到16个PERFECT匹配的场景。

## 文件说明

### 核心文件
- `linear_resize_cpu_final.h` ⭐ - 最终CPU实现（推荐使用）
- `linear_interpolation_cpu.h` - V1 Improved（功能相同）
- `linear_interpolation_v1_ultimate.h` - 实验版本（更多模式）

### 测试文件
- `test_cpu_vs_npp.cpp` ⭐ - 主要对比测试（32个场景）
- `comprehensive_test.cpp` - 完整测试套件（38个场景）
- `debug_075x.cpp` - 调试工具示例

### 文档
- `CPU_IMPLEMENTATION_SUMMARY.md` ⭐ - 实现总结
- `COMPLETE_PROJECT_SUMMARY.md` ⭐⭐⭐ - 完整项目文档
- `FINAL_RESULTS_SUMMARY.md` - 最终结果
- `PROJECT_INDEX.md` - 文件索引

## 下一步

### 深入了解
1. 阅读 `CPU_IMPLEMENTATION_SUMMARY.md` 了解实现细节
2. 阅读 `COMPLETE_PROJECT_SUMMARY.md` 了解完整项目
3. 查看 `NPP_ALGORITHM_DISCOVERED.md` 了解算法原理

### 优化改进
1. 查看 `INTEGER_UPSCALE_WEIGHTS_ANALYSIS.md` 了解权重分析
2. 研究 `reverse_engineer_params.cpp` 工具
3. 基于 `linear_interpolation_v1_ultimate.h` 进行改进

### 添加新功能
1. 参考测试程序的写法
2. 使用 `debug_075x.cpp` 作为调试模板
3. 对比NPP输出验证正确性

## 技术支持

遇到问题？检查：
1. 是否使用了正确的步长（宽度×通道数）
2. 是否正确传递了通道数参数
3. 运行 `test_cpu_vs_npp` 验证环境

查看详细分析：
- 测试失败场景的分析文档
- 使用调试工具生成详细输出
- 对比NPP结果定位问题

## 示例代码

完整的工作示例在 `test_cpu_vs_npp.cpp` 中，包括：
- 内存分配
- NPP调用
- CPU调用
- 结果对比
- 统计分析

可以直接参考或复制使用。

---

**就这么简单！** 包含一个头文件，调用一个函数，即可完成图像缩放。
