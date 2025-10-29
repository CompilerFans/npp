# Non-Integer Scale Analysis - Deep Dive

## Investigation Summary

This document summarizes extensive investigation into why the CPU reference implementation doesn't perfectly match NPP for non-integer scale factors (e.g., 1.5x, 4x).

## Test Case: 2x2 → 3x3 (1.5x Upscale)

### Source Data
```
[  0] [ 10]
[ 20] [ 30]
```

### NPP Result
```
[  0] [  4] [ 10]
[  8] [ 13] [ 18]
[ 20] [ 24] [ 30]
```

### CPU Standard Bilinear Result
```
[  0] [  5] [ 10]
[ 10] [ 15] [ 20]
[ 20] [ 25] [ 30]
```

### Differences
| Position | CPU | NPP | Diff |
|----------|-----|-----|------|
| (0,0)    | 0   | 0   | 0    |
| (1,0)    | 5   | 4   | -1   |
| (2,0)    | 10  | 10  | 0    |
| (0,1)    | 10  | 8   | -2   |
| (1,1)    | 15  | 13  | -2   |
| (2,1)    | 20  | 18  | -2   |
| (0,2)    | 20  | 20  | 0    |
| (1,2)    | 25  | 24  | -1   |
| (2,2)    | 30  | 30  | 0    |

**Observation**: All differences are negative (-1 or -2). Corner pixels match perfectly.

## Hypotheses Tested

### 1. Different Coordinate Mapping Formula
**Tested**: Various offsets in `srcCoord = (dstCoord + offset) * scale - offset`
- Offset values: 0.25, 0.3333, 0.4, 0.45, 0.5, 0.6
- **Result**: All gave 4/9 matches (corners only) ❌

### 2. Different Rounding Method
**Tested**:
- Round half up: `(int)(value + 0.5)`
- Truncate: `(int)value`
- Floor: `floor(value)`
- Ceil: `ceil(value)`
- **Result**: None matched NPP output ❌

### 3. Interpolation Bias
**Tested**: Adding bias before rounding: `(int)(value - bias + 0.5)`
- Bias = 1.6 to 2.5 gives dst(1,1) = 13
- **Result**: Works for one pixel but not a systematic solution ❌

### 4. Fixed-Point Arithmetic
**Tested**: Using integer arithmetic scaled by 256
```cpp
int fx_fixed = (int)(fx * 256);
int result = (p00 * (256-fx) * (256-fy) + ...) / (256*256);
```
- **Result**: Still gives 15 for dst(1,1), not 13 ❌

### 5. Modified Fractional Weights
**Discovered**: Using fx=fy=0.4 instead of 0.5 for interpolated pixels
- **Result**: 8/9 pixels match! ✓✓
- Only dst(1,1) doesn't match (gives 12, need 13)

**Analysis**:
```
dst(1,0) = 4: needs fx=0.4
  0*(1-0.4) + 10*0.4 = 0 + 4 = 4 ✓

dst(0,1) = 8: needs fy=0.4
  0*(1-0.4) + 20*0.4 = 0 + 8 = 8 ✓

dst(1,1) = 13: with fx=fy=0.4
  0*0.36 + 10*0.24 + 20*0.24 + 30*0.16 = 0 + 2.4 + 4.8 + 4.8 = 12 ✗
  (NPP gives 13, off by +1)
```

### 6. Edge-to-Edge Mapping
**Tested**: `srcCoord = dstCoord * (srcSize-1) / (dstSize-1)`
- **Result**: 4/9 matches (corners only) ❌

## Key Findings

1. **Corner Pixels**: Always match perfectly (use exact source values)

2. **Edge Pixels** (one dimension interpolated):
   - NPP appears to use weight ≈ 0.4 instead of 0.5
   - This gives 0.6/0.4 split instead of 0.5/0.5

3. **Center Pixel** (both dimensions interpolated):
   - Cannot be explained by single fx/fy pair
   - NPP uses different logic for fully interpolated pixels

4. **Pattern**:
   - Systematic negative bias in all non-corner pixels
   - Suggests NPP uses a modified interpolation formula

## Conclusion

**NPP LINEAR mode for non-integer upscaling does NOT use standard bilinear interpolation.**

Possible explanations:
1. **Proprietary Algorithm**: NPP may use optimized/specialized interpolation
2. **Quality vs Speed Tradeoff**: Modified weights for better performance
3. **Anti-Aliasing Filter**: Additional filtering not visible in simple analysis
4. **CUDA-specific Optimization**: Hardware-optimized interpolation in GPU

## Recommendations

### For Production Use

**Option 1**: Accept the differences for non-integer scales
- Differences are small (±1-5 pixels out of 0-255 range)
- Visual impact is minimal
- Use current implementation as-is

**Option 2**: Use tolerance-based testing
- Perfect match for integer scales
- ±2 tolerance for non-integer scales
- Document known limitations

**Option 3**: Use NPP as black box
- Don't try to replicate exact behavior
- Use NPP for reference, focus on functional correctness
- Test visual quality instead of pixel-perfect match

### For Future Investigation

1. **Test more patterns**: Linear gradients, single-color regions
2. **Check CUDA source**: If available, examine actual NPP implementation
3. **Binary analysis**: Reverse engineer NPP library
4. **Contact NVIDIA**: Request documentation on exact algorithm

## Files Generated During Investigation

- `analyze_2x2_to_3x3.cpp` - Detailed pixel-level analysis
- `test_coord_offset.cpp` - Coordinate mapping variations
- `test_interpolation_bias.cpp` - Bias and rounding tests
- `test_nearest_for_upscale.cpp` - Pattern analysis
- `find_exact_weights.cpp` - Exhaustive weight search

## Impact Assessment

**Low Impact**:
- Integer scales: 100% match ✓
- Common use cases (2x, 0.5x): Perfect
- Non-integer scales: Small visual differences

**Acceptable for**:
- Unit testing with tolerance
- Visual quality testing
- Performance benchmarking

**Not suitable for**:
- Bit-exact reproduction of NPP
- Forensic image analysis requiring perfect fidelity
