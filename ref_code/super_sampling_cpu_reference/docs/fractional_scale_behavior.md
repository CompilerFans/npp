# Fractional Scale Downsampling Behavior

## Overview

This document explains how NVIDIA NPP super sampling handles fractional scale ratios (e.g., 4×4 → 3×3, scale=1.333x).

## Key Question

**When downsampling 4×4 to 3×3, what sampling region does each output pixel use?**

## Answer: Dynamic Weighted Box Filter

Each output pixel at position `(dx, dy)` samples a region of size `scale × scale` centered at the mapped source position.

### Sampling Region Formula

```
scaleX = srcWidth / dstWidth
scaleY = srcHeight / dstHeight

For output pixel (dx, dy):
  centerX = (dx + 0.5) × scaleX
  centerY = (dy + 0.5) × scaleY

  xMin = centerX - scaleX/2
  xMax = centerX + scaleX/2
  yMin = centerY - scaleY/2
  yMax = centerY + scaleY/2

  Sampling region: [xMin, xMax) × [yMin, yMax)
```

### Example: 4×4 → 3×3 (scale = 1.333)

**Source Image (4×4)**:
```
  0  16  32  48
 64  80  96 112
128 144 160 176
192 208 224 240
```

**Output Pixel dst(0,0)**:
```
Center: (0.5 × 1.333, 0.5 × 1.333) = (0.67, 0.67)
Region: [0.00, 1.33) × [0.00, 1.33)

Covers pixels (with fractional weights):
  src(0,0) = 0:   weight = 1.0×1.0 = 1.0000  (fully inside)
  src(1,0) = 16:  weight = 0.33×1.0 = 0.3333 (partial horizontal overlap)
  src(0,1) = 64:  weight = 1.0×0.33 = 0.3333 (partial vertical overlap)
  src(1,1) = 80:  weight = 0.33×0.33 = 0.1111 (corner partial overlap)

Total weight: 1.0 + 0.3333 + 0.3333 + 0.1111 = 1.7777 ≈ 1.333²

Result = (0×1.0 + 16×0.3333 + 64×0.3333 + 80×0.1111) / 1.7777
       = 35.55 / 1.7777
       = 20.0
```

**NPP Result**: 20 ✓ (exact match)

### All Sampling Regions for 4×4 → 3×3

| Output Pixel | Sampling Region X | Sampling Region Y |
|--------------|-------------------|-------------------|
| dst(0,0) | [0.00, 1.33) | [0.00, 1.33) |
| dst(1,0) | [1.33, 2.67) | [0.00, 1.33) |
| dst(2,0) | [2.67, 4.00) | [0.00, 1.33) |
| dst(0,1) | [0.00, 1.33) | [1.33, 2.67) |
| dst(1,1) | [1.33, 2.67) | [1.33, 2.67) |
| dst(2,1) | [2.67, 4.00) | [1.33, 2.67) |
| dst(0,2) | [0.00, 1.33) | [2.67, 4.00) |
| dst(1,2) | [1.33, 2.67) | [2.67, 4.00) |
| dst(2,2) | [2.67, 4.00) | [2.67, 4.00) |

**Result Image (3×3)**:
```
 20  40  60
100 120 140
180 200 220
```

## Visual Representation

### 4×4 → 3×3 Sampling Pattern

```
Source grid (4×4):
+-----+-----+-----+-----+
|  0  | 16  | 32  | 48  |
+-----+-----+-----+-----+
| 64  | 80  | 96  | 112 |
+-----+-----+-----+-----+
| 128 | 144 | 160 | 176 |
+-----+-----+-----+-----+
| 192 | 208 | 224 | 240 |
+-----+-----+-----+-----+

Sampling region for dst(0,0): [0.00, 1.33) × [0.00, 1.33)
+-----+--+
|  0  |16|  ← Full pixel + 1/3 of next
+-----+--+
| 64  |80|  ← 1/3 height of these pixels
+--+--+--+
```

### Weight Calculation Detail

For a source pixel at `(sx, sy)` and sampling region `[xMin, xMax) × [yMin, yMax)`:

```cpp
// Calculate overlap in X direction
float wxMin = max(xMin, (float)sx);
float wxMax = min(xMax, (float)(sx + 1));
float overlapX = wxMax - wxMin;

// Calculate overlap in Y direction
float wyMin = max(yMin, (float)sy);
float wyMax = min(yMax, (float)(sy + 1));
float overlapY = wyMax - wyMin;

// Weight = area of overlap
float weight = overlapX × overlapY;
```

## Additional Test Cases

### Test: 5×5 → 3×3 (scale = 1.667)

**dst(0,0) sampling region**: [0.00, 1.67) × [0.00, 1.67)

```
Weighted contributions:
  src(0,0) = 0:   1.0 × 1.0 = 1.0000
  src(1,0) = 10:  0.67 × 1.0 = 0.6667
  src(0,1) = 50:  1.0 × 0.67 = 0.6667
  src(1,1) = 60:  0.67 × 0.67 = 0.4444

Total weight: 2.7778 ≈ 1.667²
Result: (0 + 6.667 + 33.333 + 26.667) / 2.7778 = 24.0
```

**NPP Result**: 24 ✓

### Test: 6×6 → 4×4 (scale = 1.5)

**dst(0,0) sampling region**: [0.00, 1.50) × [0.00, 1.50)

```
Weighted contributions:
  src(0,0) = 0:   1.0 × 1.0 = 1.0000
  src(1,0) = 7:   0.5 × 1.0 = 0.5000
  src(0,1) = 42:  1.0 × 0.5 = 0.5000
  src(1,1) = 49:  0.5 × 0.5 = 0.2500

Total weight: 2.25 = 1.5²
Result: (0 + 3.5 + 21 + 12.25) / 2.25 = 16.33
```

**NPP Result**: 16 (rounded from 16.33) ✓

### Test: Uniform Image 7×7 → 5×5 (all pixels = 100)

**Expected**: All output pixels should be 100

**NPP Result**:
```
100 100 100 100 100
100 100 100 100 100
100 100 100 100 100
100 100 100 100 100
100 100 100 100 100
```

✓ **Correct**: When all source pixels have the same value, output is identical regardless of weights.

### Test: Checkerboard 8×8 → 5×5

**Source**: Alternating 0 and 255

**Result**:
```
120 120 128 135 135
120 120 128 135 135
128 128 128 128 128
135 135 128 120 120
135 135 128 120 120
```

The result shows smooth blending based on fractional overlap of black and white regions.

## Key Properties

### 1. Total Weight = Theoretical Area

For each output pixel:
```
Total weight ≈ scaleX × scaleY
```

This ensures proper averaging over the sampling region.

### 2. Fractional Pixel Weights

Edge pixels contribute fractionally based on overlap area:
- Full overlap: weight = 1.0
- Half overlap: weight = 0.5
- Quarter overlap: weight = 0.25
- etc.

### 3. Boundary Alignment

For the leftmost pixel (dx=0):
```
xMin = (0 + 0.5) × scale - scale/2 = 0.0
```

For the rightmost pixel (dx=dstWidth-1):
```
xMax = (dstWidth - 0.5) × scale + scale/2 = dstWidth × scale = srcWidth
```

**Sampling regions perfectly align with source boundaries.**

## Implementation Algorithm

```cpp
for each output pixel (dx, dy):
  // Calculate sampling region
  centerX = (dx + 0.5) × scaleX
  centerY = (dy + 0.5) × scaleY
  xMin = centerX - scaleX/2
  xMax = centerX + scaleX/2
  yMin = centerY - scaleY/2
  yMax = centerY + scaleY/2

  // Find integer pixel range
  xMinInt = floor(xMin)
  xMaxInt = ceil(xMax)
  yMinInt = floor(yMin)
  yMaxInt = ceil(yMax)

  // Accumulate weighted sum
  sum = 0
  totalWeight = 0

  for sy in [yMinInt, yMaxInt):
    for sx in [xMinInt, xMaxInt):
      // Calculate overlap area
      wxMin = max(xMin, sx)
      wxMax = min(xMax, sx+1)
      wyMin = max(yMin, sy)
      wyMax = min(yMax, sy+1)
      weight = (wxMax - wxMin) × (wyMax - wyMin)

      // Accumulate
      sum += source[sy][sx] × weight
      totalWeight += weight

  // Normalize and write result
  result[dy][dx] = round(sum / totalWeight)
```

## Comparison: Integer vs Fractional Scales

| Property | Integer Scale (e.g., 4x) | Fractional Scale (e.g., 1.333x) |
|----------|--------------------------|----------------------------------|
| Sampling region size | 4×4 pixels | 1.333×1.333 pixels |
| Edge pixel weights | All 1.0 (full pixels) | Fractional (0.0 to 1.0) |
| Total weight | 16.0 | ≈1.78 |
| Pixel overlap | Clean alignment | Fractional overlap |
| Result precision | Exact average | Weighted average with rounding |

## Validation Code

See `test_fractional_scale.cpp` for complete implementation and test cases.

### Compilation
```bash
g++ -o test_fractional_scale test_fractional_scale.cpp \
    -I/usr/local/cuda/include \
    -L/usr/local/cuda/lib64 \
    -lcudart -lnppc -lnppial -lnppig \
    -std=c++11
```

### Execution
```bash
./test_fractional_scale
```

## Summary

### Fractional Scale Behavior

✅ **Fully Supported**: NVIDIA NPP super sampling handles fractional scales correctly

✅ **Weighted Box Filter**: Uses dynamic kernel size = `scale × scale`

✅ **Fractional Weights**: Edge pixels weighted by overlap area (0.0 to 1.0)

✅ **Proper Normalization**: Sum of weights equals theoretical sampling area

✅ **Boundary Safe**: Sampling regions naturally fit within source boundaries

### Example Summary: 4×4 → 3×3

- **Scale**: 1.333× in both directions
- **Each output pixel**: Samples a 1.333×1.333 region
- **Overlapping pixels**: 2×2 grid with fractional weights
- **Total weight**: ≈1.78 (= 1.333²)
- **Result**: Properly weighted average of 4 source pixels
- **Accuracy**: NPP results match mathematical expectation (±1 due to rounding)

### Applicability

This behavior applies to:
- ✅ All fractional scales (1.1x, 1.5x, 2.5x, 3.333x, etc.)
- ✅ Different X and Y scales (e.g., 1.5x horizontal, 2.0x vertical)
- ✅ All data types (8u, 16u, 32f, etc.)
- ✅ All channel counts (C1, C3, C4)

## Reference

- Test program: `test_fractional_scale.cpp`
- Related docs: `super_sampling_boundary_analysis.md`, `box_filter_visual_example.md`
