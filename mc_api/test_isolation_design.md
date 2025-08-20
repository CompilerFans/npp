# MPP vs NPP 测试隔离设计

## 概述

为了支持MetaX MPP库与NVIDIA NPP库的对比测试，同时保持两个测试体系的独立性，我们需要设计一个测试隔离机制。

## 设计原则

1. **完全隔离**: MPP测试和NPP对比测试应该能够独立运行
2. **配置驱动**: 通过配置控制启用哪种测试模式
3. **向后兼容**: 现有的NPP对比测试应该继续工作
4. **易于扩展**: 未来可以轻松添加其他GPU库的对比测试

## 目录结构设计

```
test/
├── framework/
│   ├── test_base.h                    # 通用测试基类
│   ├── mpp_test_base.h                # MPP专用测试基类
│   ├── nvidia_comparison_test_base.h  # NVIDIA对比测试基类（现有）
│   ├── test_config.h                  # 测试配置管理
│   └── test_isolation.h               # 测试隔离控制
├── unit/
│   ├── mpp/                          # MPP单元测试
│   │   ├── test_mpp_arithmetic.cpp
│   │   ├── test_mpp_transpose.cpp
│   │   └── ...
│   ├── comparison/                   # 对比测试
│   │   ├── nvidia/                   # NVIDIA NPP对比测试
│   │   │   ├── test_npp_comparison.cpp
│   │   │   └── ...
│   │   └── future/                   # 未来其他库对比测试
│   └── legacy/                       # 现有测试（逐步迁移）
└── config/
    ├── test_modes.json               # 测试模式配置
    └── library_paths.json            # 库路径配置
```

## 配置机制

### 编译时配置

```cmake
# CMake options
option(ENABLE_MPP_TESTS "Enable MPP native tests" ON)
option(ENABLE_NVIDIA_NPP_COMPARISON "Enable NVIDIA NPP comparison tests" ON)  
option(ENABLE_INTEL_IPP_COMPARISON "Enable Intel IPP comparison tests" OFF)
option(TEST_ISOLATION_MODE "Test isolation mode (STRICT|MIXED)" "MIXED")

# 配置定义
if(ENABLE_MPP_TESTS)
    add_definitions(-DENABLE_MPP_TESTS=1)
endif()

if(ENABLE_NVIDIA_NPP_COMPARISON AND NVIDIA_NPP_FOUND)
    add_definitions(-DENABLE_NVIDIA_NPP_COMPARISON=1)
endif()
```

### 运行时配置

```cpp
// test_config.h
#pragma once

enum class TestMode {
    MPP_ONLY,           // 只运行MPP测试
    COMPARISON_ONLY,    // 只运行对比测试  
    MIXED,              // 混合模式（默认）
    ISOLATED            // 严格隔离模式
};

class TestConfig {
public:
    static TestConfig& getInstance();
    
    TestMode getTestMode() const;
    void setTestMode(TestMode mode);
    
    bool isMppTestsEnabled() const;
    bool isNvidiaComparisonEnabled() const;
    bool isTestIsolationStrict() const;
    
    std::string getNvidiaLibraryPath() const;
    void setNvidiaLibraryPath(const std::string& path);
    
private:
    TestMode mode_ = TestMode::MIXED;
    bool mppTestsEnabled_ = true;
    bool nvidiaComparisonEnabled_ = true;
    bool strictIsolation_ = false;
    std::string nvidiaLibPath_;
};
```

## 测试基类设计

### MPP专用测试基类

```cpp
// mpp_test_base.h
#pragma once

#include <gtest/gtest.h>
#include "mpp.h"

class MppTestBase : public ::testing::Test {
protected:
    void SetUp() override;
    void TearDown() override;
    
    // MPP专用工具函数
    MppStatus initializeMpp();
    void cleanupMpp();
    
    template<typename T>
    void generateTestData(T* data, size_t count);
    
    template<typename T>  
    bool compareResults(const T* result1, const T* result2, 
                       size_t count, double tolerance = 1e-6);
    
    // 设备内存管理
    template<typename T>
    T* allocateDeviceMemory(size_t count);
    
    template<typename T>
    void freeDeviceMemory(T* ptr);
    
private:
    bool mppInitialized_ = false;
};

#define MPP_SKIP_IF_DISABLED() \
    do { \
        if (!TestConfig::getInstance().isMppTestsEnabled()) { \
            GTEST_SKIP() << "MPP tests are disabled"; \
        } \
    } while(0)
```

### 隔离测试宏

```cpp
// test_isolation.h
#pragma once

// 测试类型标记
#define MPP_TEST(test_suite_name, test_name) \
    class test_suite_name##_##test_name##_Test : public MppTestBase { \
    public: \
        test_suite_name##_##test_name##_Test() {} \
    private: \
        void TestBody() override; \
        static ::testing::TestInfo* const test_info_ GTEST_ATTRIBUTE_UNUSED_; \
    }; \
    ::testing::TestInfo* const test_suite_name##_##test_name##_Test::test_info_ = \
        ::testing::internal::MakeAndRegisterTestInfo( \
            #test_suite_name, #test_name, nullptr, nullptr, \
            ::testing::internal::CodeLocation(__FILE__, __LINE__), \
            (::testing::internal::GetTestTypeId()), \
            ::testing::Test::SetUpTestCase, \
            ::testing::Test::TearDownTestCase, \
            new ::testing::internal::TestFactoryImpl<test_suite_name##_##test_name##_Test>); \
    void test_suite_name##_##test_name##_Test::TestBody()

#define NVIDIA_COMPARISON_TEST(test_suite_name, test_name) \
    class test_suite_name##_##test_name##_NvidiaComparison : public NvidiaComparisonTestBase { \
    public: \
        test_suite_name##_##test_name##_NvidiaComparison() {} \
    private: \
        void TestBody() override; \
        static ::testing::TestInfo* const test_info_ GTEST_ATTRIBUTE_UNUSED_; \
    }; \
    ::testing::TestInfo* const test_suite_name##_##test_name##_NvidiaComparison::test_info_ = \
        ::testing::internal::MakeAndRegisterTestInfo( \
            #test_suite_name "_NvidiaComparison", #test_name, nullptr, nullptr, \
            ::testing::internal::CodeLocation(__FILE__, __LINE__), \
            (::testing::internal::GetTestTypeId()), \
            ::testing::Test::SetUpTestCase, \
            ::testing::Test::TearDownTestCase, \
            new ::testing::internal::TestFactoryImpl<test_suite_name##_##test_name##_NvidiaComparison>); \
    void test_suite_name##_##test_name##_NvidiaComparison::TestBody()

// 条件跳过宏
#define SKIP_IF_COMPARISON_DISABLED() \
    do { \
        if (!TestConfig::getInstance().isNvidiaComparisonEnabled()) { \
            GTEST_SKIP() << "NVIDIA comparison tests are disabled"; \
        } \
    } while(0)

#define SKIP_IF_MPP_DISABLED() \
    do { \
        if (!TestConfig::getInstance().isMppTestsEnabled()) { \
            GTEST_SKIP() << "MPP tests are disabled"; \
        } \
    } while(0)
```

## 使用示例

### MPP单独测试

```cpp
// test_mpp_arithmetic.cpp
#include "mpp_test_base.h"

MPP_TEST(ArithmeticOperations, AddC_8u_C1) {
    MPP_SKIP_IF_DISABLED();
    
    // 纯MPP测试，不依赖NVIDIA库
    const int width = 256, height = 256;
    Mpp8u* d_src = allocateDeviceMemory<Mpp8u>(width * height);
    Mpp8u* d_dst = allocateDeviceMemory<Mpp8u>(width * height);
    
    // 初始化测试数据
    generateTestData(d_src, width * height);
    
    // 执行MPP操作
    MppiSize roi = {width, height};
    MppStatus status = mppiAddC_8u_C1RSfs(d_src, width, 50, d_dst, width, roi, 0);
    
    EXPECT_EQ(status, MPP_NO_ERROR);
    
    freeDeviceMemory(d_src);
    freeDeviceMemory(d_dst);
}
```

### NVIDIA对比测试

```cpp
// test_nvidia_comparison.cpp
#include "nvidia_comparison_test_base.h"

NVIDIA_COMPARISON_TEST(ArithmeticOperations, AddC_8u_C1_Comparison) {
    SKIP_IF_COMPARISON_DISABLED();
    
    // 对比测试，需要两个库都存在
    const int width = 256, height = 256;
    
    // MPP测试
    auto mpp_result = runMppAddC_8u_C1(width, height, 50);
    
    // NVIDIA NPP测试  
    auto npp_result = runNvidiaAddC_8u_C1(width, height, 50);
    
    // 比较结果
    EXPECT_TRUE(compareResults(mpp_result.data(), npp_result.data(), 
                              width * height, 1e-6));
}
```

## 运行配置

### 命令行参数

```bash
# 只运行MPP测试
./mpp_test --gtest_filter="*-*NvidiaComparison*" --test-mode=mpp-only

# 只运行对比测试
./mpp_test --gtest_filter="*NvidiaComparison*" --test-mode=comparison-only

# 严格隔离模式
./mpp_test --test-isolation=strict

# 自定义NVIDIA库路径
./mpp_test --nvidia-lib-path=/custom/path/to/nvidia/libs
```

### 环境变量控制

```bash
# 禁用NVIDIA对比测试
export DISABLE_NVIDIA_COMPARISON=1

# 指定测试模式
export MPP_TEST_MODE=isolated

# 指定库路径
export NVIDIA_NPP_LIB_PATH=/usr/local/cuda/lib64
```

## 迁移策略

1. **阶段1**: 创建新的测试框架和配置系统
2. **阶段2**: 将现有测试逐步迁移到新框架
3. **阶段3**: 添加MPP专用测试
4. **阶段4**: 完善隔离机制和配置选项
5. **阶段5**: 清理legacy代码

## 好处

1. **清晰分离**: MPP测试和对比测试各自独立
2. **灵活配置**: 可以根据需要启用/禁用不同类型的测试
3. **CI/CD友好**: 在没有NVIDIA驱动的环境中可以只运行MPP测试
4. **开发效率**: 开发者可以专注于相关的测试
5. **维护性**: 代码结构清晰，易于维护和扩展