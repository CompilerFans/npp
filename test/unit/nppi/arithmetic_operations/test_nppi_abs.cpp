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

// 注意：nppiAbs_8s_C1R函数在标准NPP API中不存在，已删除相关测试

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

// 注意：nppiAbs_8s_C1R函数在标准NPP API中不存在，相关边界测试已删除

// ==================== 错误处理测试 ====================

// Test error handling for null pointers
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

// Test error handling for invalid ROI
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