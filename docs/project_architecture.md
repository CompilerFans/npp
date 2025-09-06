# OpenNPP Project Architecture

## PROJECT OVERVIEW
```
├── Architecture: GPU-accelerated Image and Signal Processing Library
├── Main Technologies: CUDA, C++17, CMake, Google Test
├── Key Patterns: NVIDIA NPP API Compatible Implementation
└── Entry Point: libopennpp.a/libopennpp.so (Library)
```

## COMPONENT MAP

### API Layer
```
api/
├── Core API
│   ├── npp.h                    # Main aggregator header
│   ├── nppcore.h               # Core functionality (version, device)
│   └── nppdefs.h               # Type definitions, error codes
├── Image Processing (NPPI)
│   ├── nppi.h                   # NPPI aggregator
│   ├── nppi_support_functions.h # Memory allocation
│   ├── nppi_arithmetic_and_logical_operations.h
│   ├── nppi_data_exchange_and_initialization.h
│   ├── nppi_linear_transforms.h
│   └── [other image processing headers]
└── Signal Processing (NPPS)
    ├── npps.h                   # NPPS aggregator
    └── [signal processing headers]
```

### Implementation Layer
```
src/
├── nppcore/
│   └── nppcore.cpp              # Library version, device info
├── nppi/
│   ├── nppi_support_functions.cpp        # Memory management
│   ├── nppi_arithmetic_operations/
│   │   ├── nppi_add.cpp/.cu             # Add functions
│   │   ├── nppi_addc.cpp/.cu            # Add constant
│   │   ├── nppi_sub.cpp/.cu             # Subtract
│   │   ├── nppi_subc.cpp/.cu            # Subtract constant
│   │   ├── nppi_mul.cpp/.cu             # Multiply
│   │   ├── nppi_mulc.cpp/.cu            # Multiply constant
│   │   ├── nppi_div.cpp/.cu             # Divide
│   │   ├── nppi_divc.cpp/.cu            # Divide constant
│   │   ├── nppi_abs.cpp/.cu             # Absolute value
│   │   └── nppi_arithmetic_reference.cpp/h # CPU reference
│   ├── nppi_data_exchange_operations/
│   │   └── nppi_transpose.cpp/.cu       # Matrix transpose
│   └── nppi_linear_transform_operations/
│       └── nppi_magnitude.cpp/.cu        # Complex magnitude
```

### Test Framework
```
test/
├── framework/
│   ├── nvidia_comparison_test_base.h  # NVIDIA NPP comparison
│   ├── nvidia_npp_loader.h           # Dynamic loading of NVIDIA libs
│   ├── opennpp_loader.h              # Dynamic loading of OpenNPP
│   └── dual_library_test_base.h     # Dual library testing
├── unit/                             # Unit tests
│   ├── nppcore/                      # Core function tests
│   ├── nppi/
│   │   ├── arithmetic/               # Arithmetic operation tests
│   │   ├── support/                  # Memory management tests
│   │   ├── data_exchange/           # Transpose tests
│   │   └── linear_transforms/       # Magnitude tests
│   └── dynamic/                      # Dynamic loading tests
├── validation/                       # Three-way validation tests
└── tools/                           # Test utilities
```

### Build System
```
├── CMakeLists.txt              # Main build configuration
├── build.sh                    # Build script
└── run_tests.sh               # Test execution script
```

## KEY INSIGHTS

### 1. **Three-Way Validation Architecture**
```
OpenNPP ←→ NVIDIA NPP ←→ CPU Reference
   ↓           ↓            ↓
   └───────────┴────────────┘
          Validation Tests
```
This ensures pixel-perfect compatibility with NVIDIA's implementation.

### 2. **Function Naming Convention**
- Format: `nppi{Operation}_{DataType}_{Channels}{Modifiers}_{Suffix}`
- Example: `nppiAddC_8u_C1RSfs`
  - `nppi`: Image processing namespace
  - `AddC`: Add constant operation
  - `8u`: 8-bit unsigned
  - `C1`: 1 channel
  - `R`: ROI (Region of Interest)
  - `Sfs`: With scale factor shift
  - `_Ctx`: Context variant (stream support)

### 3. **Memory Management Pattern**
- All NPPI operations require NPP-allocated memory
- Memory allocation: `nppiMalloc_8u_C1(width, height, &step)`
- Step/pitch alignment for GPU optimization
- Dedicated deallocation: `nppiFree(ptr)`

### 4. **Error Handling Pattern**
- All functions return `NppStatus`
- Success: `NPP_NO_ERROR` (0)
- Errors: Negative values
- Warnings: Positive values

### 5. **Testing Strategy**
- **Dynamic Library Loading**: Avoids symbol conflicts between OpenNPP and NVIDIA NPP
- **CPU Reference**: Independent verification of correctness
- **Comprehensive Coverage**: Edge cases, precision, memory alignment

### 6. **Implementation Status**
Current coverage:
- ✅ NPP Core (8 functions)
- ✅ Memory Management (5 functions)
- ✅ Basic Arithmetic (Add, Sub, Mul, Div + constant variants)
- ✅ Abs functions
- ✅ Transpose operation
- ✅ Magnitude (complex to real)
- 🚧 Extended arithmetic (Sqr, Sqrt, Ln, Exp)
- ❌ Filtering, morphology, color conversion

### 7. **Build Artifacts**
- `libopennpp.a`: Static library
- `libopennpp.so`: Shared library
- `opennpp_test`: Unified test executable

### 8. **Key Design Decisions**
- **100% API Compatibility**: Drop-in replacement for NVIDIA NPP
- **No API Modifications**: Original CUDA headers preserved in `api/`
- **Modular Implementation**: One file per operation type
- **CUDA Kernel Optimization**: GPU-specific optimizations
- **Comprehensive Testing**: Three-way validation ensures correctness

### 9. **Development Workflow**
1. Implement `.cpp` wrapper (parameter validation)
2. Implement `.cu` CUDA kernel
3. Add CPU reference implementation
4. Create unit tests
5. Add NVIDIA comparison tests
6. Run three-way validation

### 10. **Current Focus Areas**
- Basic image arithmetic operations
- Memory management functions
- Single and three-channel support
- Fixed-point scaling (Sfs functions)
- Stream context support

## Data Flow Architecture

### Function Call Flow
```
Application Code
       ↓
NPP API (npp.h)
       ↓
OpenNPP Wrapper (.cpp)
   - Parameter validation
   - Error checking
       ↓
CUDA Kernel (.cu)
   - GPU computation
       ↓
Device Memory Result
```

### Memory Management Flow
```
nppiMalloc → cudaMalloc (aligned)
     ↓
GPU Operations
     ↓
nppiFree → cudaFree
```

### Test Validation Flow
```
Test Input Data
    ├─→ OpenNPP Implementation
    ├─→ NVIDIA NPP (via dlopen)
    └─→ CPU Reference
           ↓
    Compare Results
    (tolerance-based)
```

## Code Organization Patterns

### File Structure Pattern
Each operation follows this structure:
```
operation_name.cpp    # C++ wrapper, parameter validation
operation_name.cu     # CUDA kernel implementation
```

### Header Include Hierarchy
```
npp.h
├── nppcore.h
├── nppdefs.h  
├── nppi.h
│   ├── nppi_support_functions.h
│   ├── nppi_arithmetic_and_logical_operations.h
│   └── ...
└── npps.h
    └── ...
```

### Test Organization Pattern
```
test_{module}_{operation}.cpp
test_{module}_{operation}_nvidia_comparison.cpp
test_{module}_{operation}_edge_cases.cpp
```

This architecture ensures maintainability, testability, and 100% compatibility with NVIDIA's NPP library while providing an open-source alternative.