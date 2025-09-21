# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview
This is the NVIDIA Performance Primitives (NPP) library - a comprehensive GPU-accelerated image and signal processing library. NPP provides high-performance primitives for image processing, signal processing, and computer vision applications on CUDA-enabled GPUs.

Create a open source cuda implement lib, called MPP.

## Architecture
NPP is organized into several distinct modules:
- **Core**: Basic types, error handling, and utility functions
- **Image Processing (NPPI)**: Image filtering, transformations, color conversion, etc.
- **Signal Processing (NPPS)**: Signal filtering, arithmetic operations, statistics
- **Data Types**: Custom types like Npp8u, Npp16f, complex number structures
- **Error Handling**: Comprehensive error codes and status reporting

## Key Components

### Header Structure
- `npp.h`: Main include file aggregating all other headers
- `nppdefs.h`: Type definitions, error codes, constants, and basic structures
- `nppcore.h`: Core functionality and utilities
- `nppi.h`: Image processing functions (aggregates all NPPI headers)
- `npps.h`: Signal processing functions (aggregates all NPPS headers)

### Specialized Headers
- `nppi_*`: Specific image processing domains (color conversion, filtering, transforms)
- `npps_*`: Specific signal processing domains (arithmetic, filtering, statistics)

### Data Types
- Basic types: Npp8u, Npp16u, Npp32f, etc.
- Complex types: Npp32fc, Npp64fc
- Geometric types: NppiPoint, NppiSize, NppiRect
- Image descriptors: NppiImageDescriptor
- Advanced types: Npp16f (CUDA fp16), NppStreamContext

## Development Setup

### Build System
This is a CUDA library that typically uses:
- CMake for build configuration (see `cmake/` directory)
- CUDA toolkit required (version compatibility varies)
- Standard C++ compiler with CUDA support

### Common Build Commands
```bash
# Configure cmake and build by this project mpp
./build.sh

# Configue cmake and build by test depends on nvidia npp
 ./build.sh --use-nvidia-npp
```

### Testing
Tests are located in the `test/` directory. Run with:
```bash
# Run all tests
ctest

# Run specific test categories
ctest -R "nppi_"    # Image processing tests
ctest -R "npps_"    # Signal processing tests
```

### Integration Notes
- Requires CUDA runtime and driver
- Link with `-lnppial -lnppicc -lnppidei -lnppif -lnppig -lnppim -lnppist -lnppisu -lnppitc -lnpps`
- Include path: `api/` directory
- Uses CUDA streams for asynchronous operations

## Key Patterns

### Function Naming
- Prefix indicates domain: `nppi` (image), `npps` (signal)
- Follows pattern: `nppi<operation>_<datatype>`
- Example: `nppiAdd_8u_C1RSfs` - Add operation for 8-bit unsigned 1-channel images

### Error Handling
- All functions return NppStatus
- Check with: `NPP_NO_ERROR` for success
- Negative values = errors, positive = warnings

### Memory Management
- Uses CUDA device pointers
- Requires proper alignment (see NPP_ALIGN_16, NPP_ALIGN_8 macros)
- Stream-based execution supported via NppStreamContext
- npp项目的内存申请和释放需要采用内置的资源管理function api

### ROI Processing
- Most functions operate on rectangular regions of interest (ROI)
- Uses NppiRect and NppiSize structures for region specification
- Border handling controlled via NppiBorderType enum


## Development Best Practices
- 每一个阶段性成果，git commit 提交，不需要push，清理目录中的过时临时代码和目录，对项目做整理

## Testing Guidelines
- 单元测试需要与源码的层次结构匹配对应

## API Usage Guidelines
- 绝对不要修改API的内容，我们需要使用默认的版本
- 绝对不要修改API头文件。API头文件是固定的cuda 12.8版本的NPP include头文件

## Test Precision Handling
- 若是精度的细小差异导致的测试结果，可添加一个选项和宏，选择测试校验是严格匹配还是有一定的宽容度

## Testing Principles
- nvidia npp的结果是黄金标准，如果nvidia 的测试失败，说明测试的代码写的有问题

## References
- 相关代码实现可参考git 仓库https://github.com/kunzmi/mpp和它的deepwiki

## Implementation Guidelines
- 函数实现可参考nv的官方api定义，以及https://github.com/kunzmi/mpp中对相关api的实现

## Priority Management
- 优先级最高P0的是crash问题，算法正确性问题P1，算法精度问题优先级P2，异常处理行为不正确优先级P3

## Development Notes
- 问题修正后需要同时修正build和build-nvidia
- 问题修正后需要同时验证build和build-nvidia

## Testing Restrictions
- GTEST_SKIP禁止使用

## Code Documentation
- Check existing comments in the code
- Modify comments to be brief and concise in English
- Strictly prohibit using Chinese in comments
- Aim for clarity and simplicity in comment language