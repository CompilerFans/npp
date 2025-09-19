# NPP 单元测试

这个目录包含OpenNPP库的单元测试，采用GoogleTest框架。

## 目录结构

```
test/
├── unit/                           # 单元测试根目录
│   ├── nppcore/                   # NPP核心功能测试
│   │   └── test_nppcore.cpp       # 设备管理、内存、错误处理等
│   │
│   └── nppi/                      # NPP图像处理模块测试
│       ├── arithmetic/            # 算术运算测试
│       │   ├── test_add.cpp       # Add双源运算测试
│       │   ├── test_constants.cpp  # 常数运算测试(AddC, SubC, MulC, DivC)
│       │   └── test_dual_source.cpp # 双源运算测试(Add, Sub, Mul, Div)
│       │
│       ├── data_exchange/         # 数据交换操作测试
│       │   └── test_transpose.cpp # 矩阵转置测试
│       │
│       ├── linear_transforms/     # 线性变换测试(预留)
│       │
│       └── support/               # 支持功能和测试框架
│           └── framework/         # 测试基础框架
│               └── npp_test_base.h # 测试基类和工具函数
│
├── CMakeLists.txt                 # 测试构建配置
└── README.md                      # 本文档
```

## 测试原则

### 1. 纯功能测试
- 所有测试都是**纯功能测试**，不依赖外部库
- 不进行性能基准测试或与NVIDIA NPP的对比
- 专注于验证API的正确性和功能完整性

### 2. 模块化组织
- 按照NPP的官方模块结构组织测试
- `nppcore`: 核心设备和内存管理功能
- `nppi`: 图像处理相关功能
- 未来可扩展：`npps`(信号处理)

### 3. 清晰的命名
- 测试文件命名：`test_<功能名>.cpp`
- 测试类命名：`<功能名>Test` 
- 测试用例命名：`<函数名>_<数据类型>_<测试场景>`

## 运行测试

### 构建和运行所有测试
```bash
mkdir build && cd build
cmake ..
make -j4
./test/unit/unit_tests
```

### 使用CTest运行
```bash
cd build
ctest --output-on-failure
```

### 运行特定测试
```bash
./test/unit/unit_tests --gtest_filter="AddTest.*"
./test/unit/unit_tests --gtest_filter="*Constants*"
```

## 测试框架

### 基础设施
- **GoogleTest**: 主要测试框架
- **NppTestBase**: 自定义基类，提供CUDA设备管理
- **NppImageMemory**: CUDA内存管理模板类
- **ResultValidator**: 结果验证工具类

### 测试模式
- **基础功能测试**: 验证正常输入的正确输出
- **边界条件测试**: 测试极限值、空输入等边界情况
- **错误处理测试**: 验证错误输入的正确错误码返回
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

通过这些测试，我们确保OpenNPP库的功能正确性和稳定性。