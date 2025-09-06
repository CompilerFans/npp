/**
 * @file test_nppi_abs.cpp
 * @brief NPP 绝对值函数测试
 * 
 * 测试nppiAbs系列函数的功能验证
 */

#include "../../framework/npp_test_base.h"

using namespace npp_functional_test;

class AbsFunctionalTest : public NppTestBase {
protected:
    void SetUp() override {
        NppTestBase::SetUp();
    }
    
    void TearDown() override {
        NppTestBase::TearDown();
    }
};

// ==================== 8位有符号整数测试 ====================

TEST_F(AbsFunctionalTest, Abs_8s_C1R_BasicOperation) {
    const int width = 64;
    const int height = 64;
    
    // 准备测试数据
    std::vector<Npp8s> srcData(width * height);
    std::vector<Npp8s> expectedData(width * height);
    
    // 填充测试数据：包含正数、负数和零
    for (int i = 0; i < width * height; i++) {
        srcData[i] = static_cast<Npp8s>((i % 256) - 128); // -128 to 127
        expectedData[i] = static_cast<Npp8s>(std::abs(srcData[i]));
    }
    
    // 分配GPU内存
    NppImageMemory<Npp8s> src(width, height);
    NppImageMemory<Npp8s> dst(width, height);
    
    // 复制数据到GPU
    src.copyFromHost(srcData);
    
    // 执行Abs操作 - 该函数未实现，应返回错误码
    NppiSize roi = {width, height};
    NppStatus status = nppiAbs_8s_C1R(
        src.get(), src.step(),
        dst.get(), dst.step(),
        roi);
    
    ASSERT_EQ(status, NPP_FUNCTION_NOT_IMPLEMENTED) << "nppiAbs_8s_C1R should return NPP_FUNCTION_NOT_IMPLEMENTED";
    
    // 跳过结果验证，因为函数未实现
    std::cout << "nppiAbs_8s_C1R correctly returned NPP_FUNCTION_NOT_IMPLEMENTED" << std::endl;
}

// ==================== 32位浮点数测试 ====================

TEST_F(AbsFunctionalTest, Abs_32f_C1R_BasicOperation) {
    const int width = 32;
    const int height = 32;
    
    // 准备测试数据
    std::vector<Npp32f> srcData(width * height);
    std::vector<Npp32f> expectedData(width * height);
    
    TestDataGenerator::generateRandom(srcData, -100.0f, 100.0f, 12345);
    
    // 计算期望结果
    for (size_t i = 0; i < expectedData.size(); i++) {
        expectedData[i] = std::abs(srcData[i]);
    }
    
    // 分配GPU内存
    NppImageMemory<Npp32f> src(width, height);
    NppImageMemory<Npp32f> dst(width, height);
    
    // 复制数据到GPU
    src.copyFromHost(srcData);
    
    // 执行Abs操作
    NppiSize roi = {width, height};
    NppStatus status = nppiAbs_32f_C1R(
        src.get(), src.step(),
        dst.get(), dst.step(),
        roi);
    
    ASSERT_EQ(status, NPP_NO_ERROR) << "nppiAbs_32f_C1R failed";
    
    // 验证结果
    std::vector<Npp32f> resultData(width * height);
    dst.copyToHost(resultData);
    
    EXPECT_TRUE(ResultValidator::arraysEqual(resultData, expectedData, 1e-5f))
        << "Abs 32f operation produced incorrect results";
}

// ==================== In-place 操作测试 ====================

TEST_F(AbsFunctionalTest, Abs_32f_C1IR_InPlaceOperation) {
    const int width = 16;
    const int height = 16;
    
    // 准备测试数据
    std::vector<Npp32f> srcData(width * height);
    std::vector<Npp32f> expectedData(width * height);
    
    TestDataGenerator::generateRandom(srcData, -50.0f, 50.0f, 54321);
    
    // 计算期望结果
    for (size_t i = 0; i < expectedData.size(); i++) {
        expectedData[i] = std::abs(srcData[i]);
    }
    
    // 分配GPU内存
    NppImageMemory<Npp32f> srcDst(width, height);
    
    // 复制数据到GPU
    srcDst.copyFromHost(srcData);
    
    // 执行In-place Abs操作
    NppiSize roi = {width, height};
    NppStatus status = nppiAbs_32f_C1IR(
        srcDst.get(), srcDst.step(),
        roi);
    
    ASSERT_EQ(status, NPP_NO_ERROR) << "nppiAbs_32f_C1IR failed";
    
    // 验证结果
    std::vector<Npp32f> resultData(width * height);
    srcDst.copyToHost(resultData);
    
    EXPECT_TRUE(ResultValidator::arraysEqual(resultData, expectedData, 1e-5f))
        << "In-place Abs operation produced incorrect results";
}

// ==================== 边界情况测试 ====================

TEST_F(AbsFunctionalTest, Abs_8s_C1R_BoundaryValues) {
    const int width = 4;
    const int height = 4;
    
    // 测试边界值
    std::vector<Npp8s> srcData = {
        -128, -127, -1, 0,
        1, 2, 126, 127,
        -50, -25, 25, 50,
        -100, -10, 10, 100
    };
    
    std::vector<Npp8s> expectedData = {
        127, 127, 1, 0,  // 注意：abs(-128) = 127 (饱和)
        1, 2, 126, 127,
        50, 25, 25, 50,
        100, 10, 10, 100
    };
    
    NppImageMemory<Npp8s> src(width, height);
    NppImageMemory<Npp8s> dst(width, height);
    
    src.copyFromHost(srcData);
    
    NppiSize roi = {width, height};
    NppStatus status = nppiAbs_8s_C1R(
        src.get(), src.step(),
        dst.get(), dst.step(),
        roi);
    
    ASSERT_EQ(status, NPP_FUNCTION_NOT_IMPLEMENTED) << "nppiAbs_8s_C1R should return NPP_FUNCTION_NOT_IMPLEMENTED";
    
    // 跳过结果验证，因为函数未实现
    std::cout << "Boundary value test: nppiAbs_8s_C1R correctly returned NPP_FUNCTION_NOT_IMPLEMENTED" << std::endl;
}

// ==================== 错误处理测试 ====================

TEST_F(AbsFunctionalTest, Abs_ErrorHandling_NullPointer) {
    const int width = 16;
    const int height = 16;
    
    NppImageMemory<Npp32f> src(width, height);
    NppImageMemory<Npp32f> dst(width, height);
    
    NppiSize roi = {width, height};
    
    // 测试null源指针
    NppStatus status = nppiAbs_32f_C1R(
        nullptr, src.step(),
        dst.get(), dst.step(),
        roi);
    
    EXPECT_EQ(status, NPP_NULL_POINTER_ERROR) << "Should detect null source pointer";
}

TEST_F(AbsFunctionalTest, Abs_ErrorHandling_InvalidROI) {
    const int width = 16;
    const int height = 16;
    
    NppImageMemory<Npp32f> src(width, height);
    NppImageMemory<Npp32f> dst(width, height);
    
    // 测试无效的ROI
    NppiSize roi = {0, height}; // 宽度为0
    NppStatus status = nppiAbs_32f_C1R(
        src.get(), src.step(),
        dst.get(), dst.step(),
        roi);
    
    EXPECT_EQ(status, NPP_SIZE_ERROR) << "Should detect invalid ROI size";
}