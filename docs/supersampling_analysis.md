# Supersampling Implementation Analysis

## Current Implementation Issues

### 1. Edge Weight Problem
Current implementation uses integer pixel boundaries:
```cpp
int x_start = max((int)src_x_start, oSrcRectROI.x);
int x_end = min((int)src_x_end, oSrcRectROI.x + oSrcRectROI.width);
```

**Problem**: When sampling region boundaries fall between pixels, edge pixels are either fully included (weight=1) or fully excluded (weight=0).

**Example**: Downscaling by 2.5x
- Source region: [0.0, 2.5] pixels
- Current: includes pixels 0, 1, 2 with equal weight (incorrect)
- Correct: pixel 0 (weight=1), pixel 1 (weight=1), pixel 2 (weight=0.5)

### 2. Fractional Scale Factors
- ❌ Inaccurate for non-integer scales (2.5x, 3.3x, etc.)
- ✓ Works for integer scales (2x, 4x, 8x) but not optimal

### 3. Sampling Patterns
Current: Fixed regular grid only
- ✓ Regular Grid (current)
- ❌ Random sampling
- ❌ Jittered grid
- ❌ Poisson disk
- ❌ Rotated grid

## Recommended Algorithm: Weighted Box Filter

Based on kunzmi/mpp implementation, use weighted averaging:

### Algorithm Steps

1. **Calculate sampling region** (float precision):
   ```
   xMin = dx * scaleX
   xMax = (dx + 1) * scaleX
   yMin = dy * scaleY
   yMax = (dy + 1) * scaleY
   ```

2. **Determine pixel ranges**:
   ```
   x_start = ceil(xMin)
   x_end = floor(xMax)
   y_start = ceil(yMin)
   y_end = floor(yMax)
   ```

3. **Calculate edge weights**:
   ```
   wxMin = x_start - xMin  // left edge weight
   wxMax = xMax - x_end    // right edge weight
   wyMin = y_start - yMin  // top edge weight
   wyMax = yMax - y_end    // bottom edge weight
   ```

4. **Weighted accumulation**:
   - Full pixels: weight = 1.0
   - Left/Right edge: weight = wxMin or wxMax
   - Top/Bottom edge: weight = wyMin or wyMax
   - Corners: weight = wxMin * wyMin, etc.

5. **Normalization**:
   ```
   total_weight = scaleX * scaleY
   result = sum / total_weight
   ```

### Code Structure

```cuda
// Full interior pixels (weight = 1.0)
for (int y = y_start; y < y_end; y++) {
  for (int x = x_start; x < x_end; x++) {
    sum += pixel[y][x];
  }
}

// Top edge (weight = wyMin)
if (wyMin > 0) {
  for (int x = x_start; x < x_end; x++) {
    sum += pixel[y_start-1][x] * wyMin;
  }
}

// Left edge (weight = wxMin)
if (wxMin > 0) {
  for (int y = y_start; y < y_end; y++) {
    sum += pixel[y][x_start-1] * wxMin;
  }
}

// Corners (weight = wxMin * wyMin, etc.)
// ... handle 4 corners

result = sum / (scaleX * scaleY);
```

## Integer Scale Factor Optimization

For x2, x4, x8 downsampling, can use optimized paths:

### Fast Path for 2x Downsampling
```cuda
if (scaleX == 2.0f && scaleY == 2.0f) {
  // Direct 2x2 average, no weights needed
  int sx = dx * 2;
  int sy = dy * 2;
  sum = pixel[sy][sx] + pixel[sy][sx+1] +
        pixel[sy+1][sx] + pixel[sy+1][sx+1];
  result = sum * 0.25f;
}
```

### Fast Path for 4x Downsampling
```cuda
if (scaleX == 4.0f && scaleY == 4.0f) {
  // Direct 4x4 average
  int sx = dx * 4;
  int sy = dy * 4;
  sum = 0;
  for (int j = 0; j < 4; j++) {
    for (int i = 0; i < 4; i++) {
      sum += pixel[sy+j][sx+i];
    }
  }
  result = sum * 0.0625f;  // 1/16
}
```

## Advanced Sampling Patterns

### Why Advanced Patterns?
- Reduce aliasing artifacts
- Minimize moiré patterns
- Better visual quality for high-frequency content

### Pattern Comparison

| Pattern | Quality | Performance | Complexity | Use Case |
|---------|---------|-------------|------------|----------|
| Regular Grid | Medium | Fastest | Simple | General purpose |
| Random | Medium | Fast | Simple | Quick antialiasing |
| Jittered | Good | Fast | Medium | Better than random |
| Poisson Disk | Best | Slower | High | High-quality downsampling |
| Rotated Grid | Good | Fast | Medium | Reduce grid artifacts |

### Implementation Complexity

**Regular Grid** (Current):
- ✓ Simple to implement
- ✓ GPU-friendly (cache coherent)
- ✓ Deterministic results

**Random Sampling**:
```cuda
// Requires random number generation per pixel
float rx = random(seed, dx, dy, 0);
float ry = random(seed, dx, dy, 1);
float sx = dx * scaleX + rx;
float sy = dy * scaleY + ry;
```
- ❌ Requires random number generator
- ❌ Non-deterministic (needs fixed seed)
- ❌ May have clustering issues

**Jittered Grid**:
```cuda
// Stratified sampling with jitter
const int N = 4;  // 4x4 samples
for (int j = 0; j < N; j++) {
  for (int i = 0; i < N; i++) {
    float jx = random(seed, i, j, 0) / N;
    float jy = random(seed, i, j, 1) / N;
    float sx = dx * scaleX + (i + jx) * scaleX / N;
    float sy = dy * scaleY + (j + jy) * scaleY / N;
    sample_and_accumulate(sx, sy);
  }
}
```
- ⚠️ More complex
- ⚠️ Fixed sample count

**Poisson Disk**:
- ❌ Requires precomputed sample patterns
- ❌ Increased memory usage
- ❌ Complex implementation

**Rotated Grid**:
```cuda
// Rotate sampling grid by angle (e.g., 26.565 degrees)
float angle = 0.4636f;  // radians
float cos_a = cosf(angle);
float sin_a = sinf(angle);
for each sample (i, j):
  float rx = i * cos_a - j * sin_a;
  float ry = i * sin_a + j * cos_a;
  sample at (dx * scaleX + rx, dy * scaleY + ry);
```
- ⚠️ Moderate complexity
- ✓ Better than regular grid
- ✓ Deterministic

## Recommendations

### Short Term (Phase 1)
1. ✅ Implement weighted box filter for accurate edge handling
2. ✅ Add fast paths for 2x, 4x, 8x integer downsampling
3. ✅ Keep regular grid sampling pattern

### Medium Term (Phase 2)
4. Consider adding rotated grid option for quality-sensitive applications
5. Add compile-time option to choose sampling pattern

### Long Term (Phase 3)
6. Evaluate Poisson disk if high-quality antialiasing is needed
7. Benchmark performance impact of different patterns

## Testing Strategy

1. **Accuracy Tests**:
   - Fractional scales: 2.5x, 3.3x, 4.7x
   - Non-square scales: 2x × 3x
   - Edge cases: very small/large scale factors

2. **Visual Quality Tests**:
   - Checkerboard patterns (aliasing test)
   - High-frequency textures (moiré test)
   - Natural images (perceptual quality)

3. **Performance Tests**:
   - Compare weighted vs unweighted
   - Measure fast path speedup (2x, 4x, 8x)
   - Benchmark different sampling patterns

## References

- kunzmi/mpp: Weighted box filter implementation
- NVIDIA NPP documentation: NPPI_INTER_SUPER specification
- Graphics Gems: Antialiasing techniques
- GPU Gems: Efficient sampling patterns
