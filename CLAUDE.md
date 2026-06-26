# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

MPP is an open-source CUDA implementation library providing 100% API compatibility with NVIDIA Performance Primitives (NPP). It implements GPU-accelerated image and signal processing primitives for CUDA-enabled GPUs.

**Goal**: Functionally equivalent, open-source alternative to NVIDIA's closed-source NPP library with identical APIs and behavior.

## Architecture

### Module Organization
- **nppcore**: Device management, memory allocation, error handling, logging
- **nppi**: Image processing (filtering, transforms, color conversion, arithmetic, morphology, statistics, threshold/compare, data exchange, linear transforms)
- **npps**: Signal processing (arithmetic, filtering, statistics, initialization)

All modules compile into a single library target `npp` (output: `libmpp`).

### Key Architectural Patterns

**Arithmetic Template Framework** (`src/nppi/nppi_arithmetic_operations/`):
- `nppi_arithmetic_api.h` — API dispatch layer mapping NPP function signatures to template instantiations
- `nppi_arithmetic_executor.h` — Generic template executor handling ROI iteration, border handling, stream context
- `nppi_arithmetic_ops.h` — Operation functor definitions (Add, Sub, Mul, etc.) with saturation/clamping logic
- Each operation (e.g., `nppi_add.cu`) instantiates the executor for all data type/channel combinations

**Geometry Transforms** (`src/nppi/nppi_geometry_transforms/interpolators/`):
- Specialized GPU interpolation kernels: `bilinear_interpolator.cuh`, `cubic_interpolator.cuh`, `lanczos_interpolator.cuh`, `nearest_interpolator.cuh`, `super_interpolator.cuh`
- Used by resize, warp affine, warp perspective, and rotate operations

**Version Compatibility Layer** (`src/include/npp_version_compat.h`):
- `CUDA_SDK_AT_LEAST(major, minor)` — compile-time SDK version check
- `NppSignalLength` — `size_t` for SDK 12.8+, `int` for older versions
- Handles API signature differences across CUDA 11.4/12.2/12.8

### API Headers (CRITICAL - READ ONLY)
- **NEVER modify files in `API-X.X/` directories** — these are official NVIDIA NPP headers
- `API` symlink points to the active version (currently `API-12.2`)
- Switch versions: `./switch_cuda_version.sh <11.4|12.2|12.8|current|list>`
- Rebuild after switching versions

### Function Naming Convention
- Pattern: `nppi<Operation>_<DataType>_<Channels><Modifier>`
- Prefix: `nppi` (image), `npps` (signal), `npp` (core)
- Example: `nppiAdd_8u_C1RSfs` — Add, 8-bit unsigned, 1-channel, ROI, scaled
- Modifiers: `R` (ROI), `Sfs` (scale factor), `C` (constant), `I` (in-place), `P` (planar), `AC4` (alpha channel 4)

## Build System

```bash
./build.sh                    # Build MPP (Release, Ninja + ccache, auto-detects GPU arch)
./build.sh --use-nvidia-npp   # Build tests against NVIDIA NPP (build-nvidia/)
./build.sh -d                 # Debug build
./build.sh --lib-only         # Library only (skip tests/examples)
./build.sh --benchmark        # Build benchmark suite
./build.sh -j 8               # Parallel jobs
./build.sh clean              # Remove both build/ and build-nvidia/
```

### Dual Build System
- **build/**: MPP implementation + tests
- **build-nvidia/**: Tests linked against NVIDIA's NPP (validation)

### Running Tests
```bash
cd build && ctest                          # All tests
ctest -R "nppi_"                           # Image processing tests
ctest -R "npps_"                           # Signal processing tests
ctest -V -R "test_add"                     # Verbose single test
./build/unit_tests --gtest_filter="*Add*"  # Direct binary with filter
```

### Build Options (CMake)
```
BUILD_TESTS            ON    Build test suite
BUILD_EXAMPLES         ON    Build examples
BUILD_BENCHMARK        OFF   Build benchmarks
BUILD_SHARED_LIBS      OFF   Shared vs static library
NPP_WARNINGS_AS_ERRORS ON    Treat warnings as errors
USE_NVIDIA_NPP         OFF   Link against NVIDIA NPP instead of MPP
NPP_STRICT_TESTING     OFF   Strict vs tolerant precision matching
```

## Development Workflow

### Mandatory Test-First Process
1. Write tests first in `test/unit/nppi/` or `test/unit/npps/`
2. Build and pass tests with `./build.sh --use-nvidia-npp` (validates test correctness against NVIDIA NPP)
3. Fix any NVIDIA-side test failures before proceeding — never let them block other tests
4. Implement the function in `src/`
5. Build and pass tests with `./build.sh` (validates MPP implementation)
6. Run `./update_coverage.sh` to update coverage tracking
7. Commit and push

### Test Framework (`test/unit/framework/`)
All in namespace `npp_functional_test`. New tests must inherit from `NppTestBase`.

| Component | File | Purpose |
|-----------|------|---------|
| `NppTestBase` | `npp_test_base.h` | Base fixture: CUDA device setup/teardown |
| `DeviceMemory<T>` | `npp_test_memory.h` | RAII device memory with `copyFromHost`/`copyToHost` |
| `NppImageMemory<T>` | `npp_test_memory.h` | RAII NPP image memory (Npp8u/16u/16s/16f/32s/32f/32fc/16sc/32sc, 1-4 channels) |
| `ResultValidator` | `npp_test_result_validator.h` | Array comparison with tolerance, mismatch statistics |
| `NPPTestUtils` | `npp_test_utils.h` | `IsEqual()`, `IsArithmeticEqual()` with function-specific tolerances |

Key macros: `NPP_EXPECT_EQUAL`, `NPP_EXPECT_ARITHMETIC_EQUAL`

### Test Conventions
- File naming: `test_<function_name>.cpp`
- Two patterns per operation: basic `TEST_F` and `TEST_P` (parameterized for data types/sizes)
- Use `TEST_P` with type parameters to cover multiple data types without duplication
- **No GTEST_SKIP** — tests must run
- **No error-handling-only tests** — focus on functional correctness
- If NVIDIA NPP test fails, the test code is wrong (not the implementation)
- Controlled by `NPP_STRICT_TESTING` compile definition for strict vs tolerant matching

## Code Standards

- **English only** for comments and documentation
- **No emojis** in code, comments, or commit messages
- LLVM style, 2-space indent, 120-column limit — run `./format.sh` (clang-format)
- Prefer templates over preprocessor macros
- Use generic naming — avoid "cuda", "nvidia" in symbols/comments where possible
- Commit messages: short, verb-first (e.g., "Optimize nppiCopy_32f_C3P3R performance")

## Coverage Tracking

```bash
cat api_analysis/coverage_summary.md   # View module-level statistics
./update_coverage.sh                   # Regenerate after implementing new functions
```

Coverage data lives in `api_analysis/coverage.csv` (function-level) and `api_analysis/coverage_summary.md` (module-level).

## Integration

```cpp
#include "npp.h"
// Link: -lnppial -lnppicc -lnppidei -lnppif -lnppig -lnppim -lnppist -lnppisu -lnppitc -lnpps
```

### Requirements
- CUDA toolkit (11.4, 12.2, or 12.8)
- CUDA-capable GPU (architectures default to "80;89" — adjust via `--arch` flag)
- C++17 compiler with CUDA support
- CMake 3.18+, Ninja (recommended), ccache (optional)
