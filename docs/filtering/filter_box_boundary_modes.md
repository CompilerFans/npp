# FilterBox Boundary Handling Modes

## Overview

The MPP FilterBox implementation supports two boundary handling modes that can be selected at compile time through the `MPP_FILTERBOX_ZERO_PADDING` macro.

## Boundary Modes

### 1. Direct Access Mode (Default)

When `MPP_FILTERBOX_ZERO_PADDING` is NOT defined:

- **Behavior**: Assumes valid data exists outside the ROI boundaries
- **Performance**: Faster due to no boundary checks
- **Memory Access**: Directly accesses memory outside specified ROI
- **Use Cases**: 
  - Processing sub-regions of larger images
  - When surrounding context data is available
  - Performance-critical applications

**Requirements**:
- Caller must ensure `pSrc` points to a location within a sufficiently large buffer
- Buffer must accommodate all filter accesses without causing memory violations

### 2. Zero Padding Mode

When `MPP_FILTERBOX_ZERO_PADDING` is defined:

- **Behavior**: Out-of-bounds pixels are treated as zeros
- **Performance**: Slightly slower due to boundary checks
- **Memory Access**: Checks bounds before each pixel access
- **Use Cases**:
  - Processing images without surrounding context
  - Edge detection at image boundaries
  - Safe processing when buffer size is unknown

## Usage

### Compile-time Selection

To use zero padding mode, define the macro during compilation:

```bash
# Using compiler flags
CXXFLAGS="-DMPP_FILTERBOX_ZERO_PADDING" ./build.sh
CUDAFLAGS="-DMPP_FILTERBOX_ZERO_PADDING" ./build.sh

# Or in CMake
add_compile_definitions(MPP_FILTERBOX_ZERO_PADDING)
```

### Code Example

```cpp
// No code changes required - boundary mode is selected at compile time
NppStatus status = nppiFilterBox_8u_C1R(pSrc, srcStep, pDst, dstStep, 
                                       roiSize, maskSize, anchor);
```

## Memory Access Patterns

For a pixel at position (x,y) with mask size (mw,mh) and anchor (ax,ay):
- **Minimum access offset**: (x - ax, y - ay)
- **Maximum access offset**: (x + mw - ax - 1, y + mh - ay - 1)

### Direct Access Mode
- No bounds checking performed
- Accesses may go outside ROI boundaries
- Caller responsible for ensuring valid memory

### Zero Padding Mode
- All accesses checked against ROI bounds
- Out-of-bounds accesses return zero
- Safe for any valid ROI size

## API Behavior Differences

### Parameter Validation

In zero padding mode:
- Mask can be larger than ROI
- No restriction on mask size relative to ROI

In direct access mode:
- Mask size must not exceed ROI size
- Returns `NPP_MASK_SIZE_ERROR` if mask > ROI

## Performance Considerations

- **Direct Access Mode**: ~10-15% faster for typical use cases
- **Zero Padding Mode**: Additional overhead for boundary checking
- Choose based on your specific requirements:
  - Use direct access when processing sub-regions with guaranteed padding
  - Use zero padding for safety and edge processing

## Compatibility

Both modes are fully compatible with NVIDIA NPP API:
- Same function signatures
- Same parameter meanings
- Mode selection is transparent to API users