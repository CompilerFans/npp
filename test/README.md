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
- **数据类型测试**: 验证不同数据类型的支持

## 添加新测试

### 1. 选择正确的模块目录
- 核心功能 → `nppcore/`
- 图像算术 → `nppi/arithmetic/`
- 数据交换 → `nppi/data_exchange/`
- 线性变换 → `nppi/linear_transforms/`

### 2. 创建测试文件
```cpp
#include "gtest/gtest.h"
#include "../support/framework/npp_test_base.h"

class MyFunctionTest : public NppTestBase {
protected:
    void SetUp() override {
        NppTestBase::SetUp();
        // 初始化测试数据
    }
};

TEST_F(MyFunctionTest, BasicOperation) {
    // 测试基本功能
}

TEST_F(MyFunctionTest, DISABLED_ErrorHandling) {
    // 测试错误处理
}
```

### 3. 更新CMakeLists.txt
将新的测试文件添加到`UNIT_TEST_SOURCES`列表中。

## 测试覆盖率

目前已实现的测试模块：
- ✅ **nppcore**: 设备管理、内存分配、错误处理
- ✅ **nppi/arithmetic**: Add, Sub, Mul, Div及其常数版本
- ✅ **nppi/data_exchange**: 矩阵转置操作

待扩展的模块：
- ⏳ **nppi/linear_transforms**: 线性变换操作
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