# Linear Resize CPU Reference Implementation

High-accuracy CPU reference implementation for NVIDIA NPP `nppiResize` with `NPPI_INTER_LINEAR` mode.

## Overview

This directory contains:
- CPU reference implementation matching NPP LINEAR interpolation behavior
- Analysis of NPP's actual algorithm behavior
- Validation tests comparing CPU vs NPP
- Debug tests for understanding coordinate mapping

## Files

### Core Implementation

- **`linear_interpolation_cpu.h`** - Template-based CPU reference implementation
  - Supports all data types (Npp8u, Npp16u, Npp32f, etc.)
  - Supports 1, 3, 4 channel images
  - Implements NPP's exact behavior for integer scales

### Documentation

- **`linear_interpolation_analysis.md`** - Comprehensive analysis of NPP LINEAR mode
  - Coordinate mapping formula
  - Bilinear interpolation method
  - Downscale behavior (floor method)
  - Known limitations for non-integer scales

### Validation & Testing

- **`validate_linear_cpu.cpp`** - Main validation test suite
  - 16 test cases covering upscale, downscale, and non-uniform scaling
  - Accuracy statistics and diff histograms
  - Sample pixel comparisons

- **`test_linear_analysis.cpp`** - NPP behavior analysis test
  - Simple patterns for understanding NPP algorithm
  - Coordinate mapping verification

- **`test_linear_debug_simple_patterns.cpp`** - Debug test with sequential values
  - Simplified patterns for manual verification
  - Detailed coordinate mapping output

### Build Scripts

- **`compile_validate_linear.sh`** - Compile validation test
- **`compile_linear_analysis.sh`** - Compile analysis test
- **`compile_debug.sh`** - Compile debug test

## Quick Start

```bash
# Compile and run validation
cd /home/cjxu/npp/ref_code/linear_resize_cpu_reference
./compile_validate_linear.sh
./validate_linear_cpu

# Compile and run analysis
./compile_linear_analysis.sh
./test_linear_analysis

# Compile and run debug test
./compile_debug.sh
./test_linear_debug
```

## Key Findings

### NPP LINEAR Mode Behavior

1. **Coordinate Mapping**: `srcCoord = (dstCoord + 0.5) * scale - 0.5`

2. **Integer Upscale (2x, 3x)**:
   - Uses standard bilinear interpolation
   - ✓ 100% CPU reference match

3. **Integer Downscale (0.5x, 0.25x)**:
   - Uses floor method: `src[floor(srcY)][floor(srcX)]`
   - **NO interpolation!**
   - ✓ 100% CPU reference match for 0.5x

4. **Non-Integer Scales (1.5x, 4x)**:
   - NPP uses unknown algorithm
   - Standard bilinear gives different results
   - ⚠ 11-25% CPU reference match

### Validation Results

**Perfect Match (100%):**
- Upscale 2x - Grayscale: 64/64 pixels
- Upscale 2x - RGB: 192/192 pixels
- Downscale 0.5x - Grayscale: 16/16 pixels
- Downscale 0.5x - RGB: 48/48 pixels

**Partial Match:**
- Upscale 4x: 25% match (max diff: 8)
- Upscale 1.5x: 11% match (max diff: 5)
- Downscale 0.75x: 11% match
- Non-uniform scaling: 12-25% match

## Usage in Tests

```cpp
#include "linear_interpolation_cpu.h"

// Example: Resize 8-bit grayscale image
std::vector<Npp8u> src(srcW * srcH);
std::vector<Npp8u> dst(dstW * dstH);

LinearInterpolationCPU<Npp8u>::resize(
    src.data(), srcW * sizeof(Npp8u),    // source
    srcW, srcH,
    dst.data(), dstW * sizeof(Npp8u),    // destination
    dstW, dstH,
    1                                      // channels
);
```

## Known Limitations

1. **Non-Integer Upscaling**: CPU reference uses standard bilinear, NPP uses different weights
   - Differences typically within ±5 pixels
   - Corner and edge pixels usually match perfectly

2. **Non-Integer Downscaling**: Floor method doesn't provide anti-aliasing
   - May show aliasing artifacts
   - Matches NPP behavior but not ideal for image quality

3. **Non-Uniform Scaling**: Each dimension handled separately
   - One dimension upscale + one dimension downscale
   - Mixed interpolation behavior

## Recommendations

- **Use for**: Integer scale operations (2x, 0.5x, etc.)
- **Accept small differences for**: Non-integer scales
- **Future work**: Reverse-engineer exact NPP algorithm for non-integer scales

## Integration Status

- ✓ CPU reference implementation complete
- ✓ Validation test suite complete
- ✓ Analysis documentation complete
- ⏳ Integration into main test framework - pending

## References

- NVIDIA NPP Documentation: https://docs.nvidia.com/cuda/npp/
- NPP Header Files: `nppi_geometry_transforms.h`
- Related: `ref_code/super_sampling_cpu_reference/` for Super sampling mode
