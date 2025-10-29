# 分数下采样模式分析

## 关键发现

从深度分析工具的输出中，我们发现了NPP在1.0 < scale < 2.0范围内的分数下采样的重要模式：

### 观察到的模式

1. **完全匹配的像素** - 当floor(srcX)的小数部分接近0时（fx很小）
   - 0.875x: dst[0] 匹配 (fx=0.071)
   - 0.8x: dst[0], dst[4] 匹配 (fx=0.125)
   - 0.75x: dst[0], dst[3] 匹配 (fx=0.167)
   - 0.67x: dst[0] 匹配 (fx=0.250)

   这些点NPP使用**FLOOR方法**（直接取floor(srcX)的值）

2. **不匹配的像素** - 当fx较大时
   - NPP的值介于floor值和bilinear插值之间
   - 但既不是标准的bilinear，也不完全是floor

### 测试案例分析

#### 0.875x (8->7) 分析
```
dst[0]: NPP=0,   floor=0,   bilinear=3    ✓ FLOOR匹配
dst[1]: NPP=41,  floor=36,  bilinear=44   ✗ NPP介于两者之间
dst[2]: NPP=83,  floor=72,  bilinear=85   ✗ 接近bilinear但不完全相同
dst[3]: NPP=124, floor=109, bilinear=127  ✗ 介于两者之间
```

#### 0.8x (10->8) 分析
```
dst[0]: NPP=0,   floor=0,   bilinear=4    ✓ FLOOR匹配
dst[1]: NPP=35,  floor=28,  bilinear=39   ✗ NPP介于两者之间
dst[2]: NPP=71,  floor=56,  bilinear=74   ✗ 找到匹配 fx=0.50 (原始fx=0.62)
dst[4]: NPP=141, floor=141, bilinear=145  ✓ FLOOR匹配
```

#### 0.75x (8->6) 分析
```
dst[0]: NPP=0,   floor=0,   bilinear=6    ✓ FLOOR匹配
dst[1]: NPP=48,  floor=36,  bilinear=54   ✗ NPP介于两者之间
dst[2]: NPP=97,  floor=72,  bilinear=103  ✗ 介于两者之间
dst[3]: NPP=145, floor=145, bilinear=151  ✓ FLOOR匹配
```

### 假设：修正fx的插值

观察到的数据表明，NPP可能在1.0 < scale < 2.0范围内：

1. **当fx很小时（< 某个阈值）**: 使用FLOOR方法
2. **当fx较大时**: 使用修正后的双线性插值

对于0.8x的dst[2]，工具发现原始fx=0.62时，如果使用fx=0.50可以匹配NPP结果。

这暗示NPP可能使用了**衰减的插值权重**：
- 不是完全的bilinear（会给出更大的值）
- 不是完全的floor（会给出更小的值）
- 而是介于两者之间

### 可能的算法

```cpp
// 对于 1.0 < scale < 2.0 的分数下采样
float srcX = (dx + 0.5f) * scale - 0.5f;
int x0 = floor(srcX);
float fx = srcX - x0;

if (fx < THRESHOLD) {
    // 使用floor方法
    result = src[x0];
} else {
    // 使用修正的插值
    float modified_fx = fx * ATTENUATION_FACTOR;
    result = src[x0] * (1 - modified_fx) + src[x0+1] * modified_fx;
}
```

其中：
- `THRESHOLD` 可能在 0.15-0.25 范围内
- `ATTENUATION_FACTOR` 可能在 0.7-0.9 范围内

### 对比工作情况：scale >= 2.0

- **0.5x (scale=2.0)**: 100%匹配 - 使用edge-to-edge + floor
- **0.33x (scale=3.0)**: 100%匹配 - 使用edge-to-edge + floor
- **0.25x (scale=4.0)**: 100%匹配 - 使用edge-to-edge + floor

当scale >= 2.0时，算法清晰：`srcX = dx * scale; result = src[floor(srcX)]`

### 下一步行动

1. ✓ 分析分数下采样模式（已完成）
2. 需要通过更多测试确定：
   - THRESHOLD的精确值
   - ATTENUATION_FACTOR的精确值或函数形式
   - 是否与scale大小相关

3. 实现V1终极版本，包含三种模式：
   - scale < 1.0: 标准bilinear插值（上采样）
   - 1.0 <= scale < 2.0: 修正插值（小下采样）
   - scale >= 2.0: edge-to-edge + floor（大下采样）
