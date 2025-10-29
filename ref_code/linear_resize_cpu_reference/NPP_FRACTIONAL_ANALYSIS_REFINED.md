# NPP分数下采样算法 - 精化分析

## 关键发现的修正

从深度调试发现，NPP的分数下采样算法比我们最初想的更复杂。

### 错误的初始假设

最初假设：
```
threshold = 0.5 / scale
attenuation = 1.0 / scale
```

### 实际观察到的行为

| Scale | 比例 | Floor使用的max fx | 插值开始的min fx | 实际衰减系数观察 |
|-------|------|------------------|------------------|------------------|
| 1.14x | 7/8  | 0.071 | 0.214 | 0.65-0.93 (平均0.84) |
| 1.25x | 4/5  | 0.125 | 0.375 | 0.67-0.87 (平均0.78) |
| 1.33x | 3/4  | 0.167 | 0.500 | 0.67-0.81 (平均0.74) |
| 1.50x | 2/3  | 0.250 | 0.750 | 0.67 (完全一致) |
| 1.67x | 3/5  | 0.333 | 0.667 | 0.48-0.52 (平均0.50) |

### 重要观察

1. **阈值不是单一值**
   - NPP似乎在一个fx范围内从floor平滑过渡到插值
   - 最大使用floor的fx总是小于最小使用插值的fx

2. **衰减系数不是常数**
   - 同一个scale下，不同像素的衰减系数不同
   - 衰减系数从0.65逐渐增加到0.93（对于0.875x）
   - 这表明NPP可能使用**位置相关的衰减系数**

3. **2/3 (0.67x) 是完美情况**
   - scale=1.5时，所有插值像素的衰减系数完全一致：0.667
   - 这正好是1/1.5 = 2/3
   - floor threshold正好是0.25（1/4）

## 新假设：分段线性或自适应算法

NPP可能使用以下策略之一：

### 假设1：基于fx的自适应衰减

```cpp
if (fx < threshold_low) {
    // 使用floor
    result = src[x0];
} else if (fx < threshold_high) {
    // 过渡区：线性混合floor和attenuated interp
    float blend = (fx - threshold_low) / (threshold_high - threshold_low);
    float floor_val = src[x0];
    float interp_val = attenuated_bilinear(src[x0], src[x1], fx, scale);
    result = floor_val * (1 - blend) + interp_val * blend;
} else {
    // 完全插值区
    result = attenuated_bilinear(src[x0], src[x1], fx, scale);
}
```

### 假设2：基于像素位置的渐变衰减

衰减系数可能取决于目标像素在输出中的位置：

```cpp
// 边缘像素使用较小的衰减（更接近floor）
// 中心像素使用较大的衰减（更接近bilinear）
float position_factor = (float)dx / (dstW - 1);  // 0.0 to 1.0
float base_attenuation = 1.0f / scale;
float adjusted_attenuation = base_attenuation + position_factor * adjustment;
```

这可以解释为什么0.875x的衰减从0.65增加到0.93。

### 假设3：基于距离的阈值调整

阈值可能基于源坐标的小数部分动态调整：

```cpp
// 当源坐标接近整数时（fx小），倾向于floor
// 当源坐标远离整数时（fx大），更倾向于插值
float dynamic_threshold = calculate_threshold(fx, scale);
```

## 特殊case：0.6x (scale=1.67) 的异常

0.6x的结果显示V1 Ultimate完全错误。从NPP输出看：
```
Source: 0 28 56 85 113 141 170 198 226 255
NPP:    0 47 94 141 189 236
```

分析srcX映射：
```
dst[0]: srcX=0.333 -> NPP=0   (floor)
dst[1]: srcX=2.0   -> NPP=47  (应该是src[2]=56，但NPP=47？)
dst[2]: srcX=3.667 -> NPP=94  (src[3]=85, src[4]=113, interp应该≈100)
dst[3]: srcX=5.333 -> NPP=141 (floor of src[5]=141 ✓)
dst[4]: srcX=7.0   -> NPP=189 (src[7]=198，不匹配)
dst[5]: srcX=8.667 -> NPP=236 (src[8]=226, src[9]=255, interp应该≈246)
```

这个case的行为非常奇怪，可能表明：
1. NPP在scale接近2.0时切换了算法
2. 或者我们的坐标映射理解有误
3. 或者这是NPP的bug或特殊处理

## 下一步行动

1. ✓ 完成反向工程数据收集
2. 需要更精确地建模NPP的行为：
   - 测试更多scale值确定阈值公式
   - 分析衰减系数的变化模式
   - 特别调查1.6-1.8范围的行为
3. 可能需要接受某些情况无法完美匹配
4. 对于已经能够完美匹配的cases（0.75x, 0.67x），保持现有算法

## 已解决的完美匹配cases

- 整数上采样 2x: 100% ✓
- 大下采样 >= 2.0x: 100% ✓ (0.5x, 0.33x, 0.25x)
- 特定分数: 0.75x (3/4), 0.67x (2/3): 100% ✓
- 非均匀缩放（整数组合）: 100% ✓

## 仍需改进的cases

- 0.875x (7/8): 57% (max diff=5)
- 0.8x (4/5): 50% (max diff=7)
- 0.6x (3/5): 0% (异常，需特殊调查)
- 大整数上采样 3x/4x/5x: 25-44%
- 非整数上采样: 10-33% (视觉差异小)
