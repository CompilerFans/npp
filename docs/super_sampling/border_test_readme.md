# Border Expansion Impact Test

## Overview

This standalone test program validates whether border expansion affects NVIDIA NPP super sampling downscale operations.

## Quick Start

```bash
# Compile the test
./compile_border_test.sh

# Run the test
./test_border_impact
```

## What This Test Does

The test compares three scenarios:

1. **Direct resize** - No border expansion
2. **Constant border** - Source expanded with constant value (128)
3. **Replicate border** - Source expanded by replicating edge pixels

Each scenario performs 4x downsampling (128×128 → 32×32) using `NPPI_INTER_SUPER`.

## Expected Results

All three scenarios should produce **identical results** with **0% pixel difference**, proving that border expansion is not necessary for super sampling downscale.

```
=== Comparison Results ===
Result 1 (no border) vs Result 2 (constant border) Difference Analysis:
  Pixels with difference: 0 / 1024 (0%)
  No differences found - results are identical!
```

## Test Output

The program outputs:
- Source image samples (top-left and top-right corners)
- Result images for all three scenarios
- Pixel-by-pixel difference statistics
- Edge pixel comparison table

## Files

- `test_border_impact.cpp` - Main test program
- `compile_border_test.sh` - Compilation script
- `docs/nvidia_npp_border_test_results.md` - Detailed analysis results

## Requirements

- CUDA Toolkit (with NVIDIA NPP libraries)
- C++ compiler with C++11 support
- CUDA-capable GPU

## Key Findings

✅ Border expansion is **NOT required** for super sampling downscale
✅ Edge pixels compute correctly without border data
✅ All border modes produce identical results
✅ Applies to all downsampling ratios (scale > 1.0)

## Reference

See `docs/nvidia_npp_border_test_results.md` for detailed analysis and explanation.
