# NPP Unit Tests

This directory contains unit tests for the MPP library using GoogleTest framework.

## Directory Structure

```
test/
├── unit/                           # Unit test root directory
│   ├── nppcore/                   # NPP core functionality tests
│   │   └── test_nppcore.cpp       # Device management, memory, error handling, etc.
│   │
│   └── nppi/                      # NPP image processing module tests
│       ├── arithmetic/            # Arithmetic operation tests
│       │   ├── test_add.cpp       # Add dual source operation tests
│       │   ├── test_constants.cpp  # Constant operation tests (AddC, SubC, MulC, DivC)
│       │   └── test_dual_source.cpp # Dual source operation tests (Add, Sub, Mul, Div)
│       │
│       ├── data_exchange/         # Data exchange operation tests
│       │   └── test_transpose.cpp # Matrix transpose tests
│       │
│       ├── linear_transforms/     # Linear transform tests (reserved)
│       │
│       └── support/               # Support functions and test framework
│           └── framework/         # Test base framework
│               └── npp_test_base.h # Test base class and utility functions
│
├── CMakeLists.txt                 # Test build configuration
└── README.md                      # This document
```

## Testing Principles

### 1. Pure Functional Tests
- All tests are **pure functional tests** without external dependencies
- No performance benchmarking or comparison with NVIDIA NPP
- Focus on verifying API correctness and functional completeness

### 2. Modular Organization
- Organize tests according to NPP official module structure
- `nppcore`: Core device and memory management functions
- `nppi`: Image processing related functions
- Future extensible: `npps` (signal processing)

### 3. Clear Naming
- Test file naming: `test_<function_name>.cpp`
- Test class naming: `<function_name>Test` 
- Test case naming: `<function_name>_<data_type>_<test_scenario>`

## Running Tests

### Build and Run All Tests
```bash
mkdir build && cd build
cmake ..
make -j4
./test/unit/unit_tests
```

### Run with CTest
```bash
cd build
ctest --output-on-failure
```

### Run Specific Tests
```bash
./test/unit/unit_tests --gtest_filter="AddTest.*"
./test/unit/unit_tests --gtest_filter="*Constants*"
```

## Test Framework

### Infrastructure
- **GoogleTest**: Main test framework
- **NppTestBase**: Custom base class providing CUDA device management
- **NppImageMemory**: CUDA memory management template class
- **ResultValidator**: Result validation utility class

### Test Modes
- **Basic functionality tests**: Verify correct output for normal input
- **Boundary condition tests**: Test extreme values, empty input and other boundary cases
- **Error handling tests**: Verify correct error codes for invalid input
- **Data type tests**: Verify support for different data types

## Adding New Tests

### 1. Choose the Correct Module Directory
- Core functionality → `nppcore/`
- Image arithmetic → `nppi/arithmetic/`
- Data exchange → `nppi/data_exchange/`
- Linear transforms → `nppi/linear_transforms/`

### 2. Create Test File
```cpp
#include "gtest/gtest.h"
#include "../support/framework/npp_test_base.h"

class MyFunctionTest : public NppTestBase {
protected:
    void SetUp() override {
        NppTestBase::SetUp();
        // Initialize test data
    }
};

TEST_F(MyFunctionTest, BasicOperation) {
    // Test basic functionality
}

TEST_F(MyFunctionTest, DISABLED_ErrorHandling) {
    // Test error handling
}
```

### 3. Update CMakeLists.txt
Add the new test file to the `UNIT_TEST_SOURCES` list.

## Test Coverage

Currently implemented test modules:
- ✅ **nppcore**: Device management, memory allocation, error handling
- ✅ **nppi/arithmetic**: Add, Sub, Mul, Divand their constant versions
- ✅ **nppi/data_exchange**: Matrix transpose operations

Modules to be extended:
- ⏳ **nppi/linear_transforms**: Linear transforms操作
- ⏳ **nppi/filtering**: 滤波操作  
- ⏳ **nppi/color_conversion**: 颜色空间转换
- ⏳ **npps**: 信号处理模块

## 质量标准

每个测试应该包含：
1. **功能正确性验证** - 基本操作产生预期结果
2. **数据类型支持** - 测试不同的NPP数据类型
3. **边界条件处理** - 最小/最大尺寸、特殊值等
4. **错误处理验证** - 无效参数返回正确错误码
5. **内存安全** - 无内存泄漏，正确的CUDA内存管理

通过这些测试，我们确保MPP库的功能正确性和稳定性。