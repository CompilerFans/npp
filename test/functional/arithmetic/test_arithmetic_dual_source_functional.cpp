/**
 * @file test_arithmetic_dual_source_functional.cpp
 * @brief NPP 双源算术运算功能测试 (Add, Sub, Mul, Div)
 * 
 * 专注于两个源图像间算术运算的功能验证
 */

#include "test/functional/framework/npp_test_base.h"

using namespace npp_functional_test;

class ArithmeticDualSourceFunctionalTest : public NppTestBase {
protected:
    void SetUp() override {
        NppTestBase::SetUp();
    }
    
    void TearDown() override {
        NppTestBase::TearDown();
    }
};

// ==================== Sub 双源减法测试 ====================

TEST_F(ArithmeticDualSourceFunctionalTest, Sub_8u_C1RSfs_BasicOperation) {
    const int width = 64;
    const int height = 64;
    const int scaleFactor = 0;
    
    // 准备测试数据
    std::vector<Npp8u> src1Data(width * height, 150);
    std::vector<Npp8u> src2Data(width * height, 50);
    std::vector<Npp8u> expectedData(width * height, 100);  // 150 - 50 = 100
    
    NppImageMemory<Npp8u> src1(width, height);
    NppImageMemory<Npp8u> src2(width, height);
    NppImageMemory<Npp8u> dst(width, height);
    
    src1.copyFromHost(src1Data);
    src2.copyFromHost(src2Data);
    
    // 执行Sub操作
    NppiSize roi = {width, height};
    NppStatus status = nppiSub_8u_C1RSfs(
        src1.get(), src1.step(),
        src2.get(), src2.step(),
        dst.get(), dst.step(),
        roi, scaleFactor);
    
    ASSERT_EQ(status, NPP_NO_ERROR) << "nppiSub_8u_C1RSfs failed";
    
    // 验证结果
    std::vector<Npp8u> result(width * height);
    dst.copyToHost(result);
    
    EXPECT_TRUE(ResultValidator::arraysEqual(result, expectedData))
        << "Sub operation produced incorrect results";
}

TEST_F(ArithmeticDualSourceFunctionalTest, Sub_8u_C1RSfs_WithSaturation) {
    const int width = 32;
    const int height = 32;
    const int scaleFactor = 0;
    
    // 测试饱和：小数减大数应该饱和到0
    std::vector<Npp8u> src1Data(width * height, 30);
    std::vector<Npp8u> src2Data(width * height, 100);
    std::vector<Npp8u> expectedData(width * height, 0);  // 30 - 100 = 0 (saturated)
    
    NppImageMemory<Npp8u> src1(width, height);
    NppImageMemory<Npp8u> src2(width, height);
    NppImageMemory<Npp8u> dst(width, height);
    
    src1.copyFromHost(src1Data);
    src2.copyFromHost(src2Data);
    
    NppiSize roi = {width, height};
    NppStatus status = nppiSub_8u_C1RSfs(
        src1.get(), src1.step(),
        src2.get(), src2.step(),
        dst.get(), dst.step(),
        roi, scaleFactor);
    
    ASSERT_EQ(status, NPP_NO_ERROR) << "nppiSub saturation test failed";
    
    std::vector<Npp8u> result(width * height);
    dst.copyToHost(result);
    
    EXPECT_TRUE(ResultValidator::arraysEqual(result, expectedData))
        << "Sub saturation handling failed";
}

TEST_F(ArithmeticDualSourceFunctionalTest, Sub_32f_C1R_PrecisionTest) {
    const int width = 64;
    const int height = 64;
    
    struct TestCase {
        float src1_val;
        float src2_val;
        float expected;
        std::string description;
    };
    
    std::vector<TestCase> testCases = {
        {10.5f, 3.2f, 7.3f, "10.5 - 3.2 = 7.3"},
        {-5.0f, -2.0f, -3.0f, "-5.0 - (-2.0) = -3.0"},
        {0.0f, 5.5f, -5.5f, "0.0 - 5.5 = -5.5"},
        {100.0f, 100.0f, 0.0f, "100.0 - 100.0 = 0.0"},
        {1.5f, -1.5f, 3.0f, "1.5 - (-1.5) = 3.0"}
    };
    
    for (const auto& testCase : testCases) {
        std::vector<Npp32f> src1Data(width * height, testCase.src1_val);
        std::vector<Npp32f> src2Data(width * height, testCase.src2_val);
        std::vector<Npp32f> expectedData(width * height, testCase.expected);
        
        NppImageMemory<Npp32f> src1(width, height);
        NppImageMemory<Npp32f> src2(width, height);
        NppImageMemory<Npp32f> dst(width, height);
        
        src1.copyFromHost(src1Data);
        src2.copyFromHost(src2Data);
        
        NppiSize roi = {width, height};
        NppStatus status = nppiSub_32f_C1R(
            src1.get(), src1.step(),
            src2.get(), src2.step(),
            dst.get(), dst.step(),
            roi);
        
        ASSERT_EQ(status, NPP_NO_ERROR) << "Sub 32f failed for: " << testCase.description;
        
        std::vector<Npp32f> result(width * height);
        dst.copyToHost(result);
        
        EXPECT_TRUE(ResultValidator::arraysEqual(result, expectedData, 1e-5f))
            << "Sub 32f result incorrect for: " << testCase.description;
    }
}

// ==================== Mul 双源乘法测试 ====================

TEST_F(ArithmeticDualSourceFunctionalTest, Mul_8u_C1RSfs_BasicOperation) {
    const int width = 32;
    const int height = 32;
    const int scaleFactor = 0;
    
    // 基本乘法测试
    std::vector<Npp8u> src1Data(width * height, 10);
    std::vector<Npp8u> src2Data(width * height, 5);
    std::vector<Npp8u> expectedData(width * height, 50);  // 10 * 5 = 50
    
    NppImageMemory<Npp8u> src1(width, height);
    NppImageMemory<Npp8u> src2(width, height);
    NppImageMemory<Npp8u> dst(width, height);
    
    src1.copyFromHost(src1Data);
    src2.copyFromHost(src2Data);
    
    NppiSize roi = {width, height};
    NppStatus status = nppiMul_8u_C1RSfs(
        src1.get(), src1.step(),
        src2.get(), src2.step(),
        dst.get(), dst.step(),
        roi, scaleFactor);
    
    ASSERT_EQ(status, NPP_NO_ERROR) << "nppiMul_8u_C1RSfs failed";
    
    std::vector<Npp8u> result(width * height);
    dst.copyToHost(result);
    
    EXPECT_TRUE(ResultValidator::arraysEqual(result, expectedData))
        << "Mul operation produced incorrect results";
}

TEST_F(ArithmeticDualSourceFunctionalTest, Mul_8u_C1RSfs_WithScaling) {
    const int width = 32;
    const int height = 32;
    const int scaleFactor = 2;  // 右移2位
    
    // 乘法后缩放：(20 * 10) >> 2 = 200 >> 2 = 50
    std::vector<Npp8u> src1Data(width * height, 20);
    std::vector<Npp8u> src2Data(width * height, 10);
    std::vector<Npp8u> expectedData(width * height, 50);
    
    NppImageMemory<Npp8u> src1(width, height);
    NppImageMemory<Npp8u> src2(width, height);
    NppImageMemory<Npp8u> dst(width, height);
    
    src1.copyFromHost(src1Data);
    src2.copyFromHost(src2Data);
    
    NppiSize roi = {width, height};
    NppStatus status = nppiMul_8u_C1RSfs(
        src1.get(), src1.step(),
        src2.get(), src2.step(),
        dst.get(), dst.step(),
        roi, scaleFactor);
    
    ASSERT_EQ(status, NPP_NO_ERROR) << "nppiMul with scaling failed";
    
    std::vector<Npp8u> result(width * height);
    dst.copyToHost(result);
    
    EXPECT_TRUE(ResultValidator::arraysEqual(result, expectedData))
        << "Mul with scaling produced incorrect results";
}

TEST_F(ArithmeticDualSourceFunctionalTest, Mul_32f_C1R_PrecisionTest) {
    const int width = 64;
    const int height = 64;
    
    struct TestCase {
        float src1_val;
        float src2_val;
        float expected;
        std::string description;
    };
    
    std::vector<TestCase> testCases = {
        {2.5f, 4.0f, 10.0f, "2.5 * 4.0 = 10.0"},
        {-3.0f, 2.0f, -6.0f, "-3.0 * 2.0 = -6.0"},
        {0.5f, 0.5f, 0.25f, "0.5 * 0.5 = 0.25"},
        {1.0f, 0.0f, 0.0f, "1.0 * 0.0 = 0.0"},
        {-1.0f, -1.0f, 1.0f, "-1.0 * -1.0 = 1.0"}
    };
    
    for (const auto& testCase : testCases) {
        std::vector<Npp32f> src1Data(width * height, testCase.src1_val);
        std::vector<Npp32f> src2Data(width * height, testCase.src2_val);
        std::vector<Npp32f> expectedData(width * height, testCase.expected);
        
        NppImageMemory<Npp32f> src1(width, height);
        NppImageMemory<Npp32f> src2(width, height);
        NppImageMemory<Npp32f> dst(width, height);
        
        src1.copyFromHost(src1Data);
        src2.copyFromHost(src2Data);
        
        NppiSize roi = {width, height};
        NppStatus status = nppiMul_32f_C1R(
            src1.get(), src1.step(),
            src2.get(), src2.step(),
            dst.get(), dst.step(),
            roi);
        
        ASSERT_EQ(status, NPP_NO_ERROR) << "Mul 32f failed for: " << testCase.description;
        
        std::vector<Npp32f> result(width * height);
        dst.copyToHost(result);
        
        EXPECT_TRUE(ResultValidator::arraysEqual(result, expectedData, 1e-5f))
            << "Mul 32f result incorrect for: " << testCase.description;
    }
}

// ==================== Div 双源除法测试 ====================

TEST_F(ArithmeticDualSourceFunctionalTest, Div_8u_C1RSfs_BasicOperation) {
    const int width = 32;
    const int height = 32;
    const int scaleFactor = 0;
    
    // 基本除法测试
    std::vector<Npp8u> src1Data(width * height, 100);
    std::vector<Npp8u> src2Data(width * height, 4);
    std::vector<Npp8u> expectedData(width * height, 25);  // 100 / 4 = 25
    
    NppImageMemory<Npp8u> src1(width, height);
    NppImageMemory<Npp8u> src2(width, height);
    NppImageMemory<Npp8u> dst(width, height);
    
    src1.copyFromHost(src1Data);
    src2.copyFromHost(src2Data);
    
    NppiSize roi = {width, height};
    NppStatus status = nppiDiv_8u_C1RSfs(
        src1.get(), src1.step(),
        src2.get(), src2.step(),
        dst.get(), dst.step(),
        roi, scaleFactor);
    
    ASSERT_EQ(status, NPP_NO_ERROR) << "nppiDiv_8u_C1RSfs failed";
    
    std::vector<Npp8u> result(width * height);
    dst.copyToHost(result);
    
    EXPECT_TRUE(ResultValidator::arraysEqual(result, expectedData))
        << "Div operation produced incorrect results";
}

TEST_F(ArithmeticDualSourceFunctionalTest, Div_32f_C1R_PrecisionTest) {
    const int width = 64;
    const int height = 64;
    
    struct TestCase {
        float src1_val;
        float src2_val;
        float expected;
        std::string description;
    };
    
    std::vector<TestCase> testCases = {
        {10.0f, 2.0f, 5.0f, "10.0 / 2.0 = 5.0"},
        {-12.0f, 3.0f, -4.0f, "-12.0 / 3.0 = -4.0"},
        {7.5f, 2.5f, 3.0f, "7.5 / 2.5 = 3.0"},
        {1.0f, 3.0f, 1.0f/3.0f, "1.0 / 3.0 = 0.333..."},
        {0.0f, 2.0f, 0.0f, "0.0 / 2.0 = 0.0"}
    };
    
    for (const auto& testCase : testCases) {
        std::vector<Npp32f> src1Data(width * height, testCase.src1_val);
        std::vector<Npp32f> src2Data(width * height, testCase.src2_val);
        std::vector<Npp32f> expectedData(width * height, testCase.expected);
        
        NppImageMemory<Npp32f> src1(width, height);
        NppImageMemory<Npp32f> src2(width, height);
        NppImageMemory<Npp32f> dst(width, height);
        
        src1.copyFromHost(src1Data);
        src2.copyFromHost(src2Data);
        
        NppiSize roi = {width, height};
        NppStatus status = nppiDiv_32f_C1R(
            src1.get(), src1.step(),
            src2.get(), src2.step(),
            dst.get(), dst.step(),
            roi);
        
        ASSERT_EQ(status, NPP_NO_ERROR) << "Div 32f failed for: " << testCase.description;
        
        std::vector<Npp32f> result(width * height);
        dst.copyToHost(result);
        
        EXPECT_TRUE(ResultValidator::arraysEqual(result, expectedData, 1e-5f))
            << "Div 32f result incorrect for: " << testCase.description;
    }
}

// ==================== In-place 操作测试 ====================

TEST_F(ArithmeticDualSourceFunctionalTest, Sub_32f_C1IR_InPlaceOperation) {
    const int width = 32;
    const int height = 32;
    
    // 准备测试数据
    std::vector<Npp32f> srcData(width * height, 3.0f);
    std::vector<Npp32f> srcDstData(width * height, 10.0f);
    std::vector<Npp32f> expectedData(width * height, 7.0f);  // 10.0 - 3.0 = 7.0
    
    NppImageMemory<Npp32f> src(width, height);
    NppImageMemory<Npp32f> srcDst(width, height);
    
    src.copyFromHost(srcData);
    srcDst.copyFromHost(srcDstData);
    
    // 执行In-place Sub操作
    NppiSize roi = {width, height};
    NppStatus status = nppiSub_32f_C1IR(
        src.get(), src.step(),
        srcDst.get(), srcDst.step(),
        roi);
    
    ASSERT_EQ(status, NPP_NO_ERROR) << "nppiSub_32f_C1IR failed";
    
    // 验证结果
    std::vector<Npp32f> result(width * height);
    srcDst.copyToHost(result);
    
    EXPECT_TRUE(ResultValidator::arraysEqual(result, expectedData, 1e-5f))
        << "In-place Sub operation produced incorrect results";
}

TEST_F(ArithmeticDualSourceFunctionalTest, Mul_32f_C1IR_InPlaceOperation) {
    const int width = 32;
    const int height = 32;
    
    std::vector<Npp32f> srcData(width * height, 2.5f);
    std::vector<Npp32f> srcDstData(width * height, 4.0f);
    std::vector<Npp32f> expectedData(width * height, 10.0f);  // 4.0 * 2.5 = 10.0
    
    NppImageMemory<Npp32f> src(width, height);
    NppImageMemory<Npp32f> srcDst(width, height);
    
    src.copyFromHost(srcData);
    srcDst.copyFromHost(srcDstData);
    
    NppiSize roi = {width, height};
    NppStatus status = nppiMul_32f_C1IR(
        src.get(), src.step(),
        srcDst.get(), srcDst.step(),
        roi);
    
    ASSERT_EQ(status, NPP_NO_ERROR) << "nppiMul_32f_C1IR failed";
    
    std::vector<Npp32f> result(width * height);
    srcDst.copyToHost(result);
    
    EXPECT_TRUE(ResultValidator::arraysEqual(result, expectedData, 1e-5f))
        << "In-place Mul operation produced incorrect results";
}

// ==================== 随机数据测试 ====================

TEST_F(ArithmeticDualSourceFunctionalTest, RandomData_ComprehensiveTest) {
    const int width = 128;
    const int height = 128;
    const unsigned seed = 54321;
    
    // 生成随机测试数据
    std::vector<Npp32f> src1Data(width * height);
    std::vector<Npp32f> src2Data(width * height);
    
    TestDataGenerator::generateRandom(src1Data, -50.0f, 50.0f, seed);
    TestDataGenerator::generateRandom(src2Data, 1.0f, 10.0f, seed + 1);  // 避免除零
    
    // 计算期望结果
    std::vector<Npp32f> expectedAdd(width * height);
    std::vector<Npp32f> expectedSub(width * height);
    std::vector<Npp32f> expectedMul(width * height);
    std::vector<Npp32f> expectedDiv(width * height);
    
    for (size_t i = 0; i < src1Data.size(); i++) {
        expectedAdd[i] = src1Data[i] + src2Data[i];
        expectedSub[i] = src1Data[i] - src2Data[i];
        expectedMul[i] = src1Data[i] * src2Data[i];
        expectedDiv[i] = src1Data[i] / src2Data[i];
    }
    
    NppImageMemory<Npp32f> src1(width, height);
    NppImageMemory<Npp32f> src2(width, height);
    NppImageMemory<Npp32f> dst(width, height);
    
    src1.copyFromHost(src1Data);
    src2.copyFromHost(src2Data);
    
    NppiSize roi = {width, height};
    
    // 测试Add
    NppStatus status = nppiAdd_32f_C1R(
        src1.get(), src1.step(),
        src2.get(), src2.step(),
        dst.get(), dst.step(),
        roi);
    ASSERT_EQ(status, NPP_NO_ERROR) << "Random Add test failed";
    
    std::vector<Npp32f> resultAdd(width * height);
    dst.copyToHost(resultAdd);
    EXPECT_TRUE(ResultValidator::arraysEqual(resultAdd, expectedAdd, 1e-4f))
        << "Random Add results incorrect";
    
    // 测试Sub
    status = nppiSub_32f_C1R(
        src1.get(), src1.step(),
        src2.get(), src2.step(),
        dst.get(), dst.step(),
        roi);
    ASSERT_EQ(status, NPP_NO_ERROR) << "Random Sub test failed";
    
    std::vector<Npp32f> resultSub(width * height);
    dst.copyToHost(resultSub);
    EXPECT_TRUE(ResultValidator::arraysEqual(resultSub, expectedSub, 1e-4f))
        << "Random Sub results incorrect";
    
    // 测试Mul
    status = nppiMul_32f_C1R(
        src1.get(), src1.step(),
        src2.get(), src2.step(),
        dst.get(), dst.step(),
        roi);
    ASSERT_EQ(status, NPP_NO_ERROR) << "Random Mul test failed";
    
    std::vector<Npp32f> resultMul(width * height);
    dst.copyToHost(resultMul);
    EXPECT_TRUE(ResultValidator::arraysEqual(resultMul, expectedMul, 1e-4f))
        << "Random Mul results incorrect";
    
    // 测试Div
    status = nppiDiv_32f_C1R(
        src1.get(), src1.step(),
        src2.get(), src2.step(),
        dst.get(), dst.step(),
        roi);
    ASSERT_EQ(status, NPP_NO_ERROR) << "Random Div test failed";
    
    std::vector<Npp32f> resultDiv(width * height);
    dst.copyToHost(resultDiv);
    EXPECT_TRUE(ResultValidator::arraysEqual(resultDiv, expectedDiv, 1e-4f))
        << "Random Div results incorrect";
    
    std::cout << "Random data comprehensive test passed for " 
              << width << "x" << height << " images" << std::endl;
}

// ==================== 错误处理测试 ====================

TEST_F(ArithmeticDualSourceFunctionalTest, ErrorHandling_DivideByZero) {
    const int width = 16;
    const int height = 16;
    
    std::vector<Npp32f> src1Data(width * height, 10.0f);
    std::vector<Npp32f> src2Data(width * height, 0.0f);  // 除零
    
    NppImageMemory<Npp32f> src1(width, height);
    NppImageMemory<Npp32f> src2(width, height);
    NppImageMemory<Npp32f> dst(width, height);
    
    src1.copyFromHost(src1Data);
    src2.copyFromHost(src2Data);
    
    NppiSize roi = {width, height};
    NppStatus status = nppiDiv_32f_C1R(
        src1.get(), src1.step(),
        src2.get(), src2.step(),
        dst.get(), dst.step(),
        roi);
    
    // 除零的行为可能因实现而异，这里主要测试不会崩溃
    std::cout << "Div by zero returned status: " << status << std::endl;
    
    if (status == NPP_NO_ERROR) {
        std::vector<Npp32f> result(width * height);
        dst.copyToHost(result);
        
        // 检查结果是否为inf或nan
        bool has_inf_or_nan = false;
        for (const auto& val : result) {
            if (std::isinf(val) || std::isnan(val)) {
                has_inf_or_nan = true;
                break;
            }
        }
        
        std::cout << "Result contains inf/nan: " << (has_inf_or_nan ? "yes" : "no") << std::endl;
    }
}