# NPP 测试架构

## 概述

本文档说明NPP项目的新测试结构，该结构将功能测试与对比测试完全分离，提供清晰的测试职责分工。新架构专注于：

- **功能验证**: 纯单元测试，验证API功能正确性
- **对比测试**: 与NVIDIA NPP的功能和性能对比
- **快速反馈**: 无外部依赖的快速功能测试
- **全面保障**: 完整的质量验证体系

## 测试结构概览

```
test/
├── functional/          # 纯功能单元测试（无外部依赖）
│   ├── framework/       # 功能测试框架
│   ├── arithmetic/      # 算术运算功能测试
│   ├── core/            # 核心功能测试
│   ├── data_exchange/   # 数据交换功能测试
│   ├── linear_transforms/ # 线性变换功能测试
│   ├── memory/          # 内存管理功能测试
│   ├── CMakeLists.txt   # 功能测试构建配置
│   └── build/           # 构建输出目录
├── comparison/          # NVIDIA NPP对比测试
│   ├── framework/       # 对比测试框架
│   ├── arithmetic/      # 算术运算对比测试
│   ├── data_exchange/   # 数据交换对比测试
│   ├── linear_transforms/ # 线性变换对比测试
│   └── CMakeLists.txt   # 对比测试构建配置
├── legacy/              # 遗留测试（向后兼容）
│   └── unit/            # 原有的单元测试
└── README.md            # 本文档
```

## 测试类型详解

### 1. 功能测试 (Functional Tests)

**目标**: 验证NPP API的功能正确性
**特点**:
- 纯单元测试，专注API功能验证
- 无外部依赖，不依赖NVIDIA NPP库
- 使用GoogleTest框架
- 快速执行，适合持续集成

**测试内容**:
- API参数验证
- 基本功能正确性
- 边界条件处理
- 错误处理机制
- 内存管理正确性

### 2. 对比测试 (Comparison Tests)

**目标**: 验证OpenNPP实现与NVIDIA NPP的一致性
**特点**:
- 依赖NVIDIA NPP库进行对比
- 测试功能一致性和性能对比
- 使用动态库加载避免符号冲突
- 需要NVIDIA GPU和NPP库支持

**测试内容**:
- 功能一致性验证
- 性能基准对比
- 大规模数据测试
- 精度对比分析

### 3. 遗留测试 (Legacy Tests)

**目标**: 向后兼容，保持现有测试能力
**特点**:
- 包含原有的混合测试代码
- 可选构建，默认关闭
- 用于渐进式迁移

## 构建和运行

### 构建选项

在根目录的CMakeLists.txt中定义了以下构建选项：

```cmake
option(BUILD_TESTS "Build tests" ON)
option(BUILD_FUNCTIONAL_TESTS "Build functional tests" ON) 
option(BUILD_COMPARISON_TESTS "Build comparison tests" ON)
option(BUILD_LEGACY_TESTS "Build legacy tests" OFF)
option(ENABLE_NVIDIA_NPP_TESTS "Enable NVIDIA NPP comparison tests" ON)
```

### 构建功能测试

```bash
# 方法1: 在项目根目录
mkdir -p build && cd build
cmake .. -DBUILD_FUNCTIONAL_TESTS=ON
make test_functional

# 方法2: 直接在功能测试目录构建
cd test/functional
mkdir -p build && cd build
cmake ..
make -j4
```

### 运行功能测试

```bash
# 运行所有功能测试
./npp_functional_tests

# 运行特定模块测试
./npp_arithmetic_functional_tests

# 使用GoogleTest过滤器
./npp_functional_tests --gtest_filter="*BasicOperation"
./npp_functional_tests --gtest_filter="AddFunctionalTest.*"
```

### 构建对比测试

```bash
# 需要NVIDIA NPP库支持
cd test/comparison
mkdir -p build && cd build
cmake ..
make -j4
```

### 运行对比测试

```bash
# 运行对比测试
./npp_comparison_tests

# 运行算术运算对比测试
./npp_arithmetic_comparison_tests
```

### 查看测试结果

功能测试使用GoogleTest框架，输出标准的测试结果：

```
[==========] Running 10 tests from 1 test suite.
[----------] Global test environment set-up.
[----------] 10 tests from AddFunctionalTest
[ RUN      ] AddFunctionalTest.Add_8u_C1RSfs_BasicOperation
Using GPU: NVIDIA GeForce RTX 4060 Laptop GPU
[       OK ] AddFunctionalTest.Add_8u_C1RSfs_BasicOperation (259 ms)
[ RUN      ] AddFunctionalTest.Add_32f_C1R_BasicOperation
Using GPU: NVIDIA GeForce RTX 4060 Laptop GPU
[       OK ] AddFunctionalTest.Add_32f_C1R_BasicOperation (8 ms)
...
[----------] 10 tests from AddFunctionalTest (500 ms total)
[----------] Global test environment tear-down
[==========] 10 tests from 1 test suite ran. (500 ms total)
[  PASSED  ] 10 tests.
```

## 测试框架

### 功能测试框架

位于 `test/functional/framework/npp_test_base.h`，提供：

- **NppTestBase**: 基础测试类，提供CUDA设备管理
- **DeviceMemory<T>**: 通用GPU内存管理模板类
- **NppImageMemory<T>**: NPP图像内存管理类
- **TestDataGenerator**: 测试数据生成工具类
- **ResultValidator**: 结果验证工具类

### 对比测试框架

位于 `test/comparison/framework/`，提供：

- **NvidiaComparisonTestBase**: NVIDIA NPP对比测试基类
- **NvidiaLoader**: NVIDIA NPP动态库加载器
- **PerformanceTimer**: 性能测试计时器
- **ConsistencyValidator**: 一致性验证工具

## 功能测试示例

```cpp
TEST_F(AddFunctionalTest, Add_32f_C1R_BasicOperation) {
    const int width = 64;
    const int height = 64;
    
    // 准备测试数据
    std::vector<Npp32f> srcData1(width * height);
    std::vector<Npp32f> srcData2(width * height);
    std::vector<Npp32f> expectedData(width * height);
    
    TestDataGenerator::generateConstant(srcData1, 1.5f);
    TestDataGenerator::generateConstant(srcData2, 2.5f);
    TestDataGenerator::generateConstant(expectedData, 4.0f);
    
    // 分配GPU内存
    NppImageMemory<Npp32f> src1(width, height);
    NppImageMemory<Npp32f> src2(width, height);
    NppImageMemory<Npp32f> dst(width, height);
    
    // 复制数据到GPU
    src1.copyFromHost(srcData1);
    src2.copyFromHost(srcData2);
    
    // 执行Add操作
    NppiSize roi = {width, height};
    NppStatus status = nppiAdd_32f_C1R(
        src1.get(), src1.step(),
        src2.get(), src2.step(),
        dst.get(), dst.step(),
        roi);
    
    ASSERT_EQ(status, NPP_NO_ERROR) << "nppiAdd_32f_C1R failed";
    
    // 验证结果
    std::vector<Npp32f> resultData(width * height);
    dst.copyToHost(resultData);
    
    EXPECT_TRUE(ResultValidator::arraysEqual(resultData, expectedData, 1e-5f))
        << "Add 32f operation produced incorrect results";
}
```

## 测试内容覆盖

### 功能测试覆盖
- 不同图像尺寸 (1x1 到 2048x2048)
- 不同数据类型 (8u, 16u, 16s, 32f, 32fc等)
- 不同缩放因子和常量值
- 多种测试模式 (常量、随机、序列、棋盘格等)
- 边界条件和错误处理
- In-place操作测试
- 多通道图像处理

### 对比测试覆盖
- 功能一致性验证
- 性能基准对比
- 大规模数据测试
- 精度对比分析
- 内存使用对比

## 最佳实践

### 编写功能测试

1. **继承NppTestBase**: 所有功能测试都应继承此基类
2. **使用内存管理类**: 使用NppImageMemory或DeviceMemory管理GPU内存
3. **数据生成**: 使用TestDataGenerator生成测试数据
4. **结果验证**: 使用ResultValidator验证计算结果
5. **错误处理**: 测试各种错误条件和边界情况

```cpp
class YourFunctionalTest : public NppTestBase {
protected:
    void SetUp() override {
        NppTestBase::SetUp();
        // 你的初始化代码
    }
};

TEST_F(YourFunctionalTest, TestName) {
    // 使用框架提供的工具类
    NppImageMemory<Npp32f> src(width, height);
    std::vector<Npp32f> data(width * height);
    TestDataGenerator::generateRandom(data, -1.0f, 1.0f);
    src.copyFromHost(data);
    
    // 执行NPP操作
    NppStatus status = nppYourFunction(src.get(), ...);
    ASSERT_EQ(status, NPP_NO_ERROR);
    
    // 验证结果
    std::vector<Npp32f> result;
    src.copyToHost(result);
    EXPECT_TRUE(ResultValidator::arraysEqual(result, expected));
}
```

### 编写对比测试

1. **继承NvidiaComparisonTestBase**: 对比测试继承此基类
2. **动态库加载**: 使用提供的加载器访问NVIDIA NPP
3. **一致性检查**: 比较两个实现的输出一致性
4. **性能基准**: 测量和比较执行时间

## 迁移指南

### 从遗留测试迁移

1. **分析测试意图**: 确定测试是功能测试还是对比测试
2. **重构测试代码**: 使用新的测试框架重写
3. **更新CMakeLists**: 添加到对应的测试目标
4. **验证测试结果**: 确保迁移后测试仍然有效

### 添加新测试

1. **确定测试类型**: 功能测试 vs 对比测试
2. **选择合适目录**: functional/ 或 comparison/
3. **使用测试框架**: 继承相应的基类
4. **更新构建配置**: 添加到CMakeLists.txt

## 持续集成建议

- **快速反馈**: 优先运行功能测试（无外部依赖，执行快速）
- **完整验证**: 在有NVIDIA GPU的环境中运行对比测试
- **渐进式测试**: 可以分阶段运行不同类型的测试
- **自动化**: 使用CTest进行测试自动化

## 故障排除

### 常见问题

1. **CUDA设备初始化失败**
   - 检查NVIDIA驱动和CUDA Runtime
   - 确保有可用的CUDA设备

2. **NVIDIA NPP库找不到**
   - 设置CUDA_PATH环境变量
   - 检查NPP库是否正确安装

3. **编译错误**
   - 检查CUDA架构设置
   - 确保C++17标准支持

4. **测试失败**
   - 检查GPU内存是否足够
   - 验证输入数据的合理性
   - 查看详细的错误信息

## 总结

新的测试架构提供了：

- **清晰的职责分离**: 功能测试vs对比测试
- **快速的功能验证**: 无外部依赖的纯功能测试
- **全面的质量保证**: 与NVIDIA NPP的对比验证
- **灵活的构建选项**: 可选择性构建不同类型测试
- **便于维护**: 结构化的测试框架和工具类

这种架构设计支持敏捷开发，既保证了开发速度（快速功能测试），又确保了质量（全面对比测试）。
