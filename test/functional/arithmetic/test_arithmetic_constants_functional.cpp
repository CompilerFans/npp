/**
 * @file test_arithmetic_constants_functional.cpp
 * @brief NPP 算术常量运算功能测试 (AddC, SubC, MulC, DivC)
 * 
 * 专注于与常量进行算术运算的功能验证
 */

#include "test/functional/framework/npp_test_base.h"

using namespace npp_functional_test;

class ArithmeticConstantsFunctionalTest : public NppTestBase {
protected:
    void SetUp() override {
        NppTestBase::SetUp();
    }
    
    void TearDown() override {
        NppTestBase::TearDown();
    }
};

// ==================== AddC 常量加法测试 ====================

TEST_F(ArithmeticConstantsFunctionalTest, AddC_8u_C1RSfs_BasicValues) {
    const int width = 64;
    const int height = 64;
    
    struct TestCase {
        Npp8u src_val;
        Npp8u constant;
        int scale_factor;
        Npp8u expected;
        std::string description;
    };
    
    std::vector<TestCase> test_cases = {
        {50, 30, 0, 80, "50 + 30 = 80"},
        {100, 100, 1, 100, "(100 + 100 + 1) >> 1 = 100"},
        {200, 100, 0, 255, "200 + 100 = 300 (saturated to 255)"},
        {0, 0, 0, 0, "0 + 0 = 0"},
        {255, 0, 0, 255, "255 + 0 = 255"}
    };
    
    for (const auto& test : test_cases) {
        // 准备数据
        std::vector<Npp8u> src_data(width * height, test.src_val);
        std::vector<Npp8u> expected_data(width * height, test.expected);
        
        NppImageMemory<Npp8u> src(width, height);
        NppImageMemory<Npp8u> dst(width, height);
        
        src.copyFromHost(src_data);
        
        // 执行AddC操作
        NppiSize roi = {width, height};
        NppStatus status = nppiAddC_8u_C1RSfs(
            src.get(), src.step(),
            test.constant,
            dst.get(), dst.step(),
            roi, test.scale_factor);
        
        ASSERT_EQ(status, NPP_NO_ERROR) << "AddC failed for: " << test.description;
        
        // 验证结果
        std::vector<Npp8u> result(width * height);
        dst.copyToHost(result);
        
        EXPECT_TRUE(ResultValidator::arraysEqual(result, expected_data))
            << "AddC result incorrect for: " << test.description;
    }
}

TEST_F(ArithmeticConstantsFunctionalTest, AddC_32f_C1R_PrecisionTest) {
    const int width = 32;
    const int height = 32;
    
    struct TestCase {
        float src_val;
        float constant;
        float expected;
        std::string description;
    };
    
    std::vector<TestCase> test_cases = {
        {1.5f, 2.5f, 4.0f, "1.5 + 2.5 = 4.0"},
        {-10.5f, 5.25f, -5.25f, "-10.5 + 5.25 = -5.25"},
        {0.0f, 3.14159f, 3.14159f, "0.0 + π = π"},
        {1e6f, 1e-6f, 1000000.000001f, "Large + Small"},
        {-1.0f, 1.0f, 0.0f, "-1.0 + 1.0 = 0.0"}
    };
    
    for (const auto& test : test_cases) {
        std::vector<Npp32f> src_data(width * height, test.src_val);
        std::vector<Npp32f> expected_data(width * height, test.expected);
        
        NppImageMemory<Npp32f> src(width, height);
        NppImageMemory<Npp32f> dst(width, height);
        
        src.copyFromHost(src_data);
        
        NppiSize roi = {width, height};
        NppStatus status = nppiAddC_32f_C1R(
            src.get(), src.step(),
            test.constant,
            dst.get(), dst.step(),
            roi);
        
        ASSERT_EQ(status, NPP_NO_ERROR) << "AddC failed for: " << test.description;
        
        std::vector<Npp32f> result(width * height);
        dst.copyToHost(result);
        
        EXPECT_TRUE(ResultValidator::arraysEqual(result, expected_data, 1e-5f))
            << "AddC result incorrect for: " << test.description;
    }
}

// ==================== SubC 常量减法测试 ====================

TEST_F(ArithmeticConstantsFunctionalTest, SubC_8u_C1RSfs_BasicValues) {
    const int width = 64;
    const int height = 64;
    
    struct TestCase {
        Npp8u src_val;
        Npp8u constant;
        int scale_factor;
        Npp8u expected;
        std::string description;
    };
    
    std::vector<TestCase> test_cases = {
        {100, 50, 0, 50, "100 - 50 = 50"},
        {200, 100, 1, 50, "(200 - 100) >> 1 = 50"},
        {30, 100, 0, 0, "30 - 100 = 0 (saturated)"},
        {255, 0, 0, 255, "255 - 0 = 255"},
        {128, 128, 0, 0, "128 - 128 = 0"}
    };
    
    for (const auto& test : test_cases) {
        std::vector<Npp8u> src_data(width * height, test.src_val);
        std::vector<Npp8u> expected_data(width * height, test.expected);
        
        NppImageMemory<Npp8u> src(width, height);
        NppImageMemory<Npp8u> dst(width, height);
        
        src.copyFromHost(src_data);
        
        NppiSize roi = {width, height};
        NppStatus status = nppiSubC_8u_C1RSfs(
            src.get(), src.step(),
            test.constant,
            dst.get(), dst.step(),
            roi, test.scale_factor);
        
        ASSERT_EQ(status, NPP_NO_ERROR) << "SubC failed for: " << test.description;
        
        std::vector<Npp8u> result(width * height);
        dst.copyToHost(result);
        
        EXPECT_TRUE(ResultValidator::arraysEqual(result, expected_data))
            << "SubC result incorrect for: " << test.description;
    }
}

TEST_F(ArithmeticConstantsFunctionalTest, SubC_32f_C1R_PrecisionTest) {
    const int width = 32;
    const int height = 32;
    
    struct TestCase {
        float src_val;
        float constant;
        float expected;
        std::string description;
    };
    
    std::vector<TestCase> test_cases = {
        {10.0f, 3.5f, 6.5f, "10.0 - 3.5 = 6.5"},
        {-5.0f, -2.5f, -2.5f, "-5.0 - (-2.5) = -2.5"},
        {0.0f, 10.0f, -10.0f, "0.0 - 10.0 = -10.0"},
        {3.14159f, 3.14159f, 0.0f, "π - π = 0.0"},
        {1e6f, 1.0f, 999999.0f, "Large - Small"}
    };
    
    for (const auto& test : test_cases) {
        std::vector<Npp32f> src_data(width * height, test.src_val);
        std::vector<Npp32f> expected_data(width * height, test.expected);
        
        NppImageMemory<Npp32f> src(width, height);
        NppImageMemory<Npp32f> dst(width, height);
        
        src.copyFromHost(src_data);
        
        NppiSize roi = {width, height};
        NppStatus status = nppiSubC_32f_C1R(
            src.get(), src.step(),
            test.constant,
            dst.get(), dst.step(),
            roi);
        
        ASSERT_EQ(status, NPP_NO_ERROR) << "SubC failed for: " << test.description;
        
        std::vector<Npp32f> result(width * height);
        dst.copyToHost(result);
        
        EXPECT_TRUE(ResultValidator::arraysEqual(result, expected_data, 1e-5f))
            << "SubC result incorrect for: " << test.description;
    }
}

// ==================== MulC 常量乘法测试 ====================

TEST_F(ArithmeticConstantsFunctionalTest, MulC_8u_C1RSfs_BasicValues) {
    const int width = 32;
    const int height = 32;
    
    struct TestCase {
        Npp8u src_val;
        Npp8u constant;
        int scale_factor;
        Npp8u expected;
        std::string description;
    };
    
    std::vector<TestCase> test_cases = {
        {10, 5, 0, 50, "10 * 5 = 50"},
        {20, 10, 1, 100, "(20 * 10) >> 1 = 100"},
        {50, 6, 0, 255, "50 * 6 = 300 (saturated to 255)"},
        {0, 100, 0, 0, "0 * 100 = 0"},
        {1, 255, 0, 255, "1 * 255 = 255"}
    };
    
    for (const auto& test : test_cases) {
        std::vector<Npp8u> src_data(width * height, test.src_val);
        std::vector<Npp8u> expected_data(width * height, test.expected);
        
        NppImageMemory<Npp8u> src(width, height);
        NppImageMemory<Npp8u> dst(width, height);
        
        src.copyFromHost(src_data);
        
        NppiSize roi = {width, height};
        NppStatus status = nppiMulC_8u_C1RSfs(
            src.get(), src.step(),
            test.constant,
            dst.get(), dst.step(),
            roi, test.scale_factor);
        
        ASSERT_EQ(status, NPP_NO_ERROR) << "MulC failed for: " << test.description;
        
        std::vector<Npp8u> result(width * height);
        dst.copyToHost(result);
        
        EXPECT_TRUE(ResultValidator::arraysEqual(result, expected_data))
            << "MulC result incorrect for: " << test.description;
    }
}

TEST_F(ArithmeticConstantsFunctionalTest, MulC_32f_C1R_PrecisionTest) {
    const int width = 32;
    const int height = 32;
    
    struct TestCase {
        float src_val;
        float constant;
        float expected;
        std::string description;
    };
    
    std::vector<TestCase> test_cases = {
        {2.0f, 3.5f, 7.0f, "2.0 * 3.5 = 7.0"},
        {-4.0f, 2.5f, -10.0f, "-4.0 * 2.5 = -10.0"},
        {10.0f, 0.0f, 0.0f, "10.0 * 0.0 = 0.0"},
        {1.0f, -1.0f, -1.0f, "1.0 * (-1.0) = -1.0"},
        {0.5f, 0.5f, 0.25f, "0.5 * 0.5 = 0.25"}
    };
    
    for (const auto& test : test_cases) {
        std::vector<Npp32f> src_data(width * height, test.src_val);
        std::vector<Npp32f> expected_data(width * height, test.expected);
        
        NppImageMemory<Npp32f> src(width, height);
        NppImageMemory<Npp32f> dst(width, height);
        
        src.copyFromHost(src_data);
        
        NppiSize roi = {width, height};
        NppStatus status = nppiMulC_32f_C1R(
            src.get(), src.step(),
            test.constant,
            dst.get(), dst.step(),
            roi);
        
        ASSERT_EQ(status, NPP_NO_ERROR) << "MulC failed for: " << test.description;
        
        std::vector<Npp32f> result(width * height);
        dst.copyToHost(result);
        
        EXPECT_TRUE(ResultValidator::arraysEqual(result, expected_data, 1e-5f))
            << "MulC result incorrect for: " << test.description;
    }
}

// ==================== DivC 常量除法测试 ====================

TEST_F(ArithmeticConstantsFunctionalTest, DivC_8u_C1RSfs_BasicValues) {
    const int width = 32;
    const int height = 32;
    
    struct TestCase {
        Npp8u src_val;
        Npp8u constant;
        int scale_factor;
        Npp8u expected;
        std::string description;
    };
    
    std::vector<TestCase> test_cases = {
        {100, 2, 0, 50, "100 / 2 = 50"},
        {200, 4, 0, 50, "200 / 4 = 50"},
        {255, 3, 0, 85, "255 / 3 = 85"},
        {10, 3, 0, 3, "10 / 3 = 3 (integer division)"},
        {0, 10, 0, 0, "0 / 10 = 0"}
    };
    
    for (const auto& test : test_cases) {
        std::vector<Npp8u> src_data(width * height, test.src_val);
        std::vector<Npp8u> expected_data(width * height, test.expected);
        
        NppImageMemory<Npp8u> src(width, height);
        NppImageMemory<Npp8u> dst(width, height);
        
        src.copyFromHost(src_data);
        
        NppiSize roi = {width, height};
        NppStatus status = nppiDivC_8u_C1RSfs(
            src.get(), src.step(),
            test.constant,
            dst.get(), dst.step(),
            roi, test.scale_factor);
        
        ASSERT_EQ(status, NPP_NO_ERROR) << "DivC failed for: " << test.description;
        
        std::vector<Npp8u> result(width * height);
        dst.copyToHost(result);
        
        EXPECT_TRUE(ResultValidator::arraysEqual(result, expected_data))
            << "DivC result incorrect for: " << test.description;
    }
}

TEST_F(ArithmeticConstantsFunctionalTest, DivC_32f_C1R_PrecisionTest) {
    const int width = 32;
    const int height = 32;
    
    struct TestCase {
        float src_val;
        float constant;
        float expected;
        std::string description;
    };
    
    std::vector<TestCase> test_cases = {
        {10.0f, 2.0f, 5.0f, "10.0 / 2.0 = 5.0"},
        {-15.0f, 3.0f, -5.0f, "-15.0 / 3.0 = -5.0"},
        {7.0f, 2.0f, 3.5f, "7.0 / 2.0 = 3.5"},
        {1.0f, 3.0f, 1.0f/3.0f, "1.0 / 3.0 = 0.333..."},
        {0.0f, 5.0f, 0.0f, "0.0 / 5.0 = 0.0"}
    };
    
    for (const auto& test : test_cases) {
        std::vector<Npp32f> src_data(width * height, test.src_val);
        std::vector<Npp32f> expected_data(width * height, test.expected);
        
        NppImageMemory<Npp32f> src(width, height);
        NppImageMemory<Npp32f> dst(width, height);
        
        src.copyFromHost(src_data);
        
        NppiSize roi = {width, height};
        NppStatus status = nppiDivC_32f_C1R(
            src.get(), src.step(),
            test.constant,
            dst.get(), dst.step(),
            roi);
        
        ASSERT_EQ(status, NPP_NO_ERROR) << "DivC failed for: " << test.description;
        
        std::vector<Npp32f> result(width * height);
        dst.copyToHost(result);
        
        EXPECT_TRUE(ResultValidator::arraysEqual(result, expected_data, 1e-5f))
            << "DivC result incorrect for: " << test.description;
    }
}

// ==================== 多通道测试 ====================

TEST_F(ArithmeticConstantsFunctionalTest, AddC_32f_C3R_MultiChannel) {
    const int width = 16;
    const int height = 16;
    const Npp32f constants[3] = {1.0f, 2.0f, 3.0f};  // R, G, B
    
    // 准备测试数据
    std::vector<Npp32f> src_data(width * height * 3);
    std::vector<Npp32f> expected_data(width * height * 3);
    
    for (int i = 0; i < width * height; i++) {
        src_data[i * 3 + 0] = 10.0f;  // R
        src_data[i * 3 + 1] = 20.0f;  // G
        src_data[i * 3 + 2] = 30.0f;  // B
        
        expected_data[i * 3 + 0] = 11.0f;  // R + 1.0
        expected_data[i * 3 + 1] = 22.0f;  // G + 2.0
        expected_data[i * 3 + 2] = 33.0f;  // B + 3.0
    }
    
    NppImageMemory<Npp32f> src(width * 3, height);  // 3通道宽度
    NppImageMemory<Npp32f> dst(width * 3, height);
    
    src.copyFromHost(src_data);
    
    NppiSize roi = {width, height};
    NppStatus status = nppiAddC_32f_C3R(
        src.get(), src.step(),
        constants,
        dst.get(), dst.step(),
        roi);
    
    ASSERT_EQ(status, NPP_NO_ERROR) << "AddC C3R failed";
    
    std::vector<Npp32f> result(width * height * 3);
    dst.copyToHost(result);
    
    EXPECT_TRUE(ResultValidator::arraysEqual(result, expected_data, 1e-5f))
        << "AddC C3R result incorrect";
}

// ==================== 错误处理测试 ====================

TEST_F(ArithmeticConstantsFunctionalTest, ErrorHandling_DivideByZero) {
    const int width = 16;
    const int height = 16;
    
    NppImageMemory<Npp32f> src(width, height);
    NppImageMemory<Npp32f> dst(width, height);
    
    src.fill(10.0f);
    
    NppiSize roi = {width, height};
    NppStatus status = nppiDivC_32f_C1R(
        src.get(), src.step(),
        0.0f,  // 除以零
        dst.get(), dst.step(),
        roi);
    
    // 除以零的行为可能因实现而异，这里主要测试不会崩溃
    std::cout << "DivC by zero returned status: " << status << std::endl;
}

TEST_F(ArithmeticConstantsFunctionalTest, ErrorHandling_InvalidROI) {
    const int width = 16;
    const int height = 16;
    
    NppImageMemory<Npp32f> src(width, height);
    NppImageMemory<Npp32f> dst(width, height);
    
    // 测试无效的ROI
    NppiSize invalid_roi = {0, height};  // 宽度为0
    NppStatus status = nppiAddC_32f_C1R(
        src.get(), src.step(),
        1.0f,
        dst.get(), dst.step(),
        invalid_roi);
    
    EXPECT_EQ(status, NPP_SIZE_ERROR) << "Should detect invalid ROI";
}