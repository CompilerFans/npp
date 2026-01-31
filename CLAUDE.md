# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

MPP is an open-source CUDA implementation library that provides 100% API compatibility with NVIDIA Performance Primitives (NPP). It implements GPU-accelerated image and signal processing primitives for CUDA-enabled GPUs.

**Key Goal**: Create a functionally equivalent, open-source alternative to NVIDIA's closed-source NPP library with identical APIs and behavior.

## Architecture

### Module Organization
- **nppcore**: Core functionality including device management, memory allocation, error handling
- **nppi**: Image processing operations (filtering, transforms, color conversion, arithmetic, etc.)
- **npps**: Signal processing operations (arithmetic, filtering, statistics)

### Directory Structure
```
MPP/
├── API/              -> Symlink to API-X.X (current CUDA SDK version headers)
├── API-11.4/         CUDA 11.4 NPP headers (DO NOT MODIFY)
├── API-12.2/         CUDA 12.2 NPP headers (DO NOT MODIFY)
├── API-12.8/         CUDA 12.8 NPP headers (DO NOT MODIFY)
├── src/              MPP implementation
│   ├── nppcore/      Core utilities and memory management
│   ├── nppi/         Image processing implementations
│   └── npps/         Signal processing implementations
├── test/             Test suite
│   ├── unit/         Functional unit tests (GoogleTest)
│   ├── golden/       Golden reference tests
│   └── common/       Shared test utilities
├── cmake/            Modular CMake configuration
├── build/            Build directory for MPP library
└── build-nvidia/     Build directory for NVIDIA NPP validation
```

## Build System

### Build Commands

```bash
# Build MPP library (default)
./build.sh

# Build with NVIDIA NPP for validation testing
./build.sh --use-nvidia-npp

# Debug build
./build.sh --debug

# Build library only (skip tests and examples)
./build.sh --lib-only

# Clean all builds
./build.sh clean

# Build with N parallel jobs
./build.sh -j 8
```

### Dual Build System

The project supports two build configurations:
- **build/**: Builds and tests MPP implementation
- **build-nvidia/**: Builds tests against NVIDIA's NPP library (for validation)

After fixing issues, always verify both builds work correctly.

### Running Tests

```bash
# Run all tests
cd build && ctest

# Run specific test categories
ctest -R "nppi_"      # Image processing tests
ctest -R "npps_"      # Signal processing tests
ctest -R "nppcore_"   # Core function tests

# Run single test
ctest -R "test_add"

# Verbose output
ctest -V -R "test_name"
```

## CUDA SDK Version Management

Switch between CUDA SDK versions using the version switcher:

```bash
./switch_cuda_version.sh 11.4    # Switch to CUDA 11.4 headers
./switch_cuda_version.sh 12.2    # Switch to CUDA 12.2 headers
./switch_cuda_version.sh 12.8    # Switch to CUDA 12.8 headers
./switch_cuda_version.sh current # Show current version
./switch_cuda_version.sh list    # List available versions
```

**Important**: Rebuild the project after switching versions.

## API Compatibility

### API Headers (CRITICAL - READ ONLY)
- **NEVER modify API header files** - these are official CUDA NPP headers
- API headers are version-specific (11.4, 12.2, 12.8)
- MPP library must support all API versions despite interface differences
- Headers located in `API-X.X/` directories are immutable

### Function Naming Convention
- Prefix indicates domain: `nppi` (image), `npps` (signal), `npp` (core)
- Pattern: `nppi<Operation>_<DataType>_<Channels><Modifier>`
- Example: `nppiAdd_8u_C1RSfs` - Add operation, 8-bit unsigned, 1-channel, ROI, scaled

### Data Types
- Basic: `Npp8u`, `Npp16u`, `Npp32f`, `Npp64f`
- Complex: `Npp32fc`, `Npp64fc`
- Geometric: `NppiPoint`, `NppiSize`, `NppiRect`
- Advanced: `Npp16f` (fp16), `NppStreamContext`

## Testing Principles

### Validation Approach
NVIDIA NPP results are the golden standard. The testing follows a triple-validation approach:
```
MPP Implementation ←→ NVIDIA NPP Library ←→ CPU Reference Implementation
         ↘                 ↙                  ↗
                 Test Framework
              (Bit-level precision)
```

### Test Framework Components
The project uses a unified test framework in `test/unit/framework/`:
- **NppTestBase**: Base test class with device setup/teardown and capability checks
- **NppTestMemory**: Unified memory management (host/device allocation, RAII wrappers)
- **NppTestResultValidator**: Result comparison utilities with tolerance support
- **NppTestRandGen**: Random data generation for test inputs
- **NppTestUtils**: Common test utilities and helpers

All new tests should inherit from `NppTestBase` and use the framework utilities.

### Test Organization
- Unit tests must match source code hierarchy
- Test files named: `test_<function>.cpp`
- Test organization mirrors API module structure
- Use namespace `npp_functional_test` for all test code

### Test Restrictions
- **DO NOT use GTEST_SKIP** - tests must run
- **DO NOT add ParameterValidation or error handling tests**
- **DO NOT add _ErrorHandling tests**
- Focus on functional correctness, not error cases

### Precision Handling
- For minor precision differences, add tolerance option/macro
- Use `NPP_STRICT_TESTING` CMake option to control strict vs tolerant matching
- If NVIDIA NPP test fails, the test code is wrong (not the implementation)

## Development Workflow

### Code Standards
- **All comments and documentation in English only** (no Chinese)
- **No emojis** in code, comments, or commit messages
- Keep comments brief and concise
- Use generic naming - avoid "cuda", "nvidia" in symbols/comments where possible
- Write naturally, avoid AI-like patterns in naming and comments

### Memory Management
- Use built-in NPP memory management functions
- Device pointers require proper alignment (NPP_ALIGN_16, NPP_ALIGN_8)
- Stream-based execution via NppStreamContext
- Example: `nppiMalloc_8u_C1()`, `nppiFree()`

### Git Workflow
- Commit after each milestone (no need to push)
- Clean up temporary code and directories regularly
- Keep repository organized

### Priority System
- **P0 (Highest)**: Crash issues
- **P1**: Algorithm correctness issues
- **P2**: Algorithm precision issues
- **P3**: Error handling behavior issues

## Implementation Guidelines

### ROI Processing
- Most functions operate on rectangular regions of interest (ROI)
- Use `NppiRect` and `NppiSize` for region specification
- Border handling via `NppiBorderType` enum

### Error Handling
- All functions return `NppStatus`
- Success: `NPP_NO_ERROR`
- Negative values = errors, positive = warnings

### Reference Implementations
- Consult NVIDIA's official API documentation
- Reference: https://github.com/kunzmi/mpp and its wiki for implementation patterns

## CMake Configuration

The build system uses modular CMake configuration:
- `CommonConfig.cmake`: Common compiler and build settings
- `CudaConfig.cmake`: CUDA-specific configuration
- `FindSources.cmake`: Source file discovery
- `TestConfig.cmake`: Test framework setup
- `InstallConfig.cmake`: Installation rules

### Build Options
```cmake
BUILD_TESTS           # Build test suite (default: ON)
BUILD_EXAMPLES        # Build example programs (default: ON)
BUILD_SHARED_LIBS     # Build shared libraries (default: OFF)
NPP_WARNINGS_AS_ERRORS # Treat warnings as errors (default: ON)
USE_NVIDIA_NPP        # Use NVIDIA NPP instead of MPP (default: OFF)
NPP_STRICT_TESTING    # Strict precision matching (default: OFF)
```

## Integration

### Linking
Link with appropriate NPP modules:
```
-lnppial -lnppicc -lnppidei -lnppif -lnppig -lnppim -lnppist -lnppisu -lnppitc -lnpps
```

### Include Path
Include from `API/` directory (symlink to current version)

### Requirements
- CUDA toolkit (11.4, 12.2, or 12.8)
- CUDA-capable GPU
- C++ compiler with CUDA support
- CMake 3.18+
- Ninja build system (recommended)

## Coverage Tracking

### Checking Implementation Status
Track implementation and test coverage using the automated analysis tools:
- **coverage.csv**: Full function-level coverage data
- **coverage_summary.md**: Module-level statistics and percentages

View coverage status:
```bash
cat api_analysis/coverage_summary.md
```

### Updating Coverage After Implementation
After adding new implementations or tests, update the coverage tracking:
```bash
./update_coverage.sh
```

This generates:
- Updated `api_analysis/coverage.csv`
- Updated `api_analysis/coverage_summary.md`
- Module-specific untested function lists (e.g., `untested_arithmetic.txt`)

## Implementation Patterns

### Avoid Macros
- Prefer templates or classes over preprocessor macros
- Use type-safe abstractions for code reuse

### Type Compatibility (size_t vs int)
- Support both size_t and int versions where API signatures differ across CUDA versions
- Implement shared logic using function templates or common helpers
- Tests should conditionally call the appropriate version based on CUDA version

### Test Parameterization
- Use `TEST_P` (parameterized tests) to reduce code duplication
- Pass expected behavior via functors or test fixtures
- Keep test code DRY (Don't Repeat Yourself)

### File Organization
- Implementation files and test files should be organized by API function name
- Maintain parallel structure between `src/` and `test/unit/` hierarchies


## 开发流程与执行规则
### 开发流程（按顺序执行）
1. 先写测试用例，再写实现代码。
2. 先在 `build-nvidia` 目录开发并运行测试，确保测试通过；测试需包含功能测试与精度测试，优先使用 gtest 参数化覆盖不同数据量、数据类型。需要执行build-nvidia下的unit_tests，使用nvidia的npp库，验证测试代码正确性
2.1 如果遇到执行nvidia下的unit_tests错误，需要优先修正，并且不能因为错误影响其他测试
3. 再开发 `src/` 函数实现；实现需考虑复用性，抽象可复用逻辑。
4. 在 `build` 目录运行测试，确保测试通过。
5. 执行 `./update_coverage.sh` 更新覆盖率数据。
6. 完成后提交 `git commit` 并 `push` 到远程仓库。

## 开发计划
- 扫描并提取npp/api_analysis/coverage.csv中，nppi_color_conversion.h缺失的api，总结至npp/api_analysis目录，再逐步实现
- 基于nppi_color_conversion.h中在实际中的实际使用热度，制定开发计划
- 开发要考虑测试和源码的复用性