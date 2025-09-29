# Morphology Operations Boundary Handling

## Overview

The MPP morphology operations support two boundary handling modes that can be selected at compile time.

## Boundary Modes

### 1. Replicate Mode (Default) 

When `MPP_MORPHOLOGY_DIRECT_ACCESS` is NOT defined:

- **Behavior**: Out-of-bounds pixels are clamped to nearest boundary pixel
- **Performance**: Slightly slower due to boundary checks
- **Memory Access**: All accesses are clamped to valid ROI boundaries
- **Use Cases**: 
  - Processing images without padding
  - When safety is more important than performance
  - When boundary behavior needs to be consistent

### 2. Direct Access Mode

When `MPP_MORPHOLOGY_DIRECT_ACCESS` is defined:

- **Behavior**: Assumes valid data exists outside the ROI boundaries
- **Performance**: Faster due to no boundary checks  
- **Memory Access**: Directly accesses memory outside specified ROI
- **Use Cases**:
  - Processing sub-regions of larger images
  - When sufficient padding is guaranteed around the ROI
  - Performance-critical applications

**Requirements**:
- Caller must ensure `pSrc` points to a location within a sufficiently large buffer
- Buffer must accommodate all structural element accesses without memory violations
- For a 3x3 kernel with anchor at (1,1), requires at least 1 pixel padding on all sides

## Usage

### Compile-time Selection

To use direct access mode, define the macro during compilation:

```bash
# Using compiler flags
CXXFLAGS="-DMPP_MORPHOLOGY_DIRECT_ACCESS" ./build.sh
CUDAFLAGS="-DMPP_MORPHOLOGY_DIRECT_ACCESS" ./build.sh

# Or in CMake
add_compile_definitions(MPP_MORPHOLOGY_DIRECT_ACCESS)
```

By default (no macro defined), replicate mode is used for safety.

### Test Code Example

When using direct access mode, ensure proper padding:

```cpp
// Allocate buffer with padding
const int padding = 8;  // Sufficient for 3x3 kernels
int paddedWidth = width + 2 * padding;
int paddedHeight = height + 2 * padding;

Npp32f* d_buffer = nppiMalloc_32f_C1(paddedWidth, paddedHeight, &step);

// Initialize entire buffer (including padding)
cudaMemset2D(d_buffer, step, 0, paddedWidth * sizeof(Npp32f), paddedHeight);

// Get pointer to actual ROI area
Npp32f* d_roi = (Npp32f*)((char*)d_buffer + padding * step + padding * sizeof(Npp32f));

// Copy data to ROI area
cudaMemcpy2D(d_roi, step, hostData, width * sizeof(Npp32f), 
             width * sizeof(Npp32f), height, cudaMemcpyHostToDevice);

// Use d_roi for morphology operations
nppiErode_32f_C1R(d_roi, step, d_dst, dstStep, roiSize, mask, maskSize, anchor);
```

## Memory Access Patterns

For morphological operations with structural element size (sw, sh) and anchor (ax, ay):
- **Maximum negative offset**: (-ax, -ay)
- **Maximum positive offset**: (sw - ax - 1, sh - ay - 1)

### Example: 3x3 kernel with center anchor (1,1)
- Requires access to pixels at offsets: (-1,-1) to (1,1)
- Minimum padding needed: 1 pixel on all sides

### Example: 5x5 kernel with center anchor (2,2)  
- Requires access to pixels at offsets: (-2,-2) to (2,2)
- Minimum padding needed: 2 pixels on all sides

## Performance Considerations

- **Direct Access Mode**: ~5-10% faster for typical morphology operations
- **Replicate Mode**: Additional overhead for boundary checking per pixel
- Choose based on your specific requirements:
  - Use direct access when you can guarantee sufficient padding
  - Use replicate mode for safety or when padding is not feasible

## Compatibility

Both modes aim to be compatible with NVIDIA NPP behavior:
- Direct access mode matches NPP's assumption of valid surrounding data
- Replicate mode provides similar boundary behavior to NPP's clamping mode
- Mode selection is transparent to API users