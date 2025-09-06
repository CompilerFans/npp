/**
 * @file test_transpose_functional.cpp
 * @brief NPP 矩阵转置功能测试
 * 
 * 专注于矩阵转置操作的功能验证，包括不同数据类型和尺寸
 */

#include "../../framework/npp_test_base.h"
#include <iomanip>

using namespace npp_functional_test;

class TransposeFunctionalTest : public NppTestBase {
protected:
    void SetUp() override {
        NppTestBase::SetUp();
    }
    
    void TearDown() override {
        NppTestBase::TearDown();
    }
    
    // 辅助函数：生成测试矩阵
    template<typename T>
    void generateTestMatrix(std::vector<T>& data, int width, int height) {
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                data[y * width + x] = static_cast<T>(y * width + x + 1);
            }
        }
    }
    
    // 辅助函数：计算转置后的期望结果
    template<typename T>
    void computeExpectedTranspose(const std::vector<T>& src, std::vector<T>& dst, 
                                 int src_width, int src_height) {
        dst.resize(src_width * src_height);
        for (int y = 0; y < src_height; y++) {
            for (int x = 0; x < src_width; x++) {
                dst[x * src_height + y] = src[y * src_width + x];
            }
        }
    }
};

// ==================== 基本转置测试 ====================

TEST_F(TransposeFunctionalTest, Transpose_8u_C1R_BasicSquareMatrix) {
    const int size = 4;  // 4x4矩阵
    
    // 准备测试数据
    std::vector<Npp8u> src_data(size * size);
    generateTestMatrix(src_data, size, size);
    
    std::vector<Npp8u> expected_data;
    computeExpectedTranspose(src_data, expected_data, size, size);
    
    // 分配GPU内存
    NppImageMemory<Npp8u> src(size, size);
    NppImageMemory<Npp8u> dst(size, size);
    
    src.copyFromHost(src_data);
    
    // 执行转置操作
    NppiSize roi = {size, size};
    NppStatus status = nppiTranspose_8u_C1R(
        src.get(), src.step(),
        dst.get(), dst.step(),
        roi);
    
    ASSERT_EQ(status, NPP_NO_ERROR) << "Transpose failed";
    
    // 验证结果
    std::vector<Npp8u> result(size * size);
    dst.copyToHost(result);
    
    EXPECT_TRUE(ResultValidator::arraysEqual(result, expected_data))
        << "Transpose result incorrect for 4x4 matrix";
    
    // 打印矩阵用于调试
    std::cout << "Original 4x4 matrix:" << std::endl;
    for (int y = 0; y < size; y++) {
        for (int x = 0; x < size; x++) {
            std::cout << std::setw(3) << static_cast<int>(src_data[y * size + x]) << " ";
        }
        std::cout << std::endl;
    }
    
    std::cout << "Transposed 4x4 matrix:" << std::endl;
    for (int y = 0; y < size; y++) {
        for (int x = 0; x < size; x++) {
            std::cout << std::setw(3) << static_cast<int>(result[y * size + x]) << " ";
        }
        std::cout << std::endl;
    }
}

TEST_F(TransposeFunctionalTest, Transpose_32f_C1R_BasicSquareMatrix) {
    const int size = 3;  // 3x3矩阵
    
    // 准备测试数据
    std::vector<Npp32f> src_data = {
        1.0f, 2.0f, 3.0f,
        4.0f, 5.0f, 6.0f,
        7.0f, 8.0f, 9.0f
    };
    
    std::vector<Npp32f> expected_data = {
        1.0f, 4.0f, 7.0f,
        2.0f, 5.0f, 8.0f,
        3.0f, 6.0f, 9.0f
    };
    
    NppImageMemory<Npp32f> src(size, size);
    NppImageMemory<Npp32f> dst(size, size);
    
    src.copyFromHost(src_data);
    
    NppiSize roi = {size, size};
    NppStatus status = nppiTranspose_32f_C1R(
        src.get(), src.step(),
        dst.get(), dst.step(),
        roi);
    
    ASSERT_EQ(status, NPP_NO_ERROR) << "Transpose 32f failed";
    
    std::vector<Npp32f> result(size * size);
    dst.copyToHost(result);
    
    EXPECT_TRUE(ResultValidator::arraysEqual(result, expected_data, 1e-5f))
        << "Transpose 32f result incorrect";
}

// ==================== 矩形矩阵转置测试 ====================

TEST_F(TransposeFunctionalTest, Transpose_8u_C1R_RectangularMatrix) {
    const int width = 6;
    const int height = 4;
    
    // 准备测试数据
    std::vector<Npp8u> src_data(width * height);
    generateTestMatrix(src_data, width, height);
    
    std::vector<Npp8u> expected_data;
    computeExpectedTranspose(src_data, expected_data, width, height);
    
    // 注意：转置后尺寸变为height x width
    NppImageMemory<Npp8u> src(width, height);
    NppImageMemory<Npp8u> dst(height, width);  // 转置后的尺寸
    
    src.copyFromHost(src_data);
    
    NppiSize roi = {width, height};
    NppStatus status = nppiTranspose_8u_C1R(
        src.get(), src.step(),
        dst.get(), dst.step(),
        roi);
    
    ASSERT_EQ(status, NPP_NO_ERROR) << "Transpose rectangular failed";
    
    std::vector<Npp8u> result(width * height);
    dst.copyToHost(result);
    
    EXPECT_TRUE(ResultValidator::arraysEqual(result, expected_data))
        << "Transpose rectangular result incorrect";
    
    // 打印矩阵验证
    std::cout << "Original " << width << "x" << height << " matrix:" << std::endl;
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            std::cout << std::setw(3) << static_cast<int>(src_data[y * width + x]) << " ";
        }
        std::cout << std::endl;
    }
    
    std::cout << "Transposed " << height << "x" << width << " matrix:" << std::endl;
    for (int y = 0; y < width; y++) {  // 注意尺寸交换
        for (int x = 0; x < height; x++) {
            std::cout << std::setw(3) << static_cast<int>(result[y * height + x]) << " ";
        }
        std::cout << std::endl;
    }
}

// ==================== 大矩阵转置测试 ====================

TEST_F(TransposeFunctionalTest, Transpose_32f_C1R_LargeMatrix) {
    const int width = 128;
    const int height = 64;
    
    // 生成大矩阵测试数据
    std::vector<Npp32f> src_data(width * height);
    TestDataGenerator::generateRandom(src_data, 0.0f, 100.0f, 12345);
    
    std::vector<Npp32f> expected_data;
    computeExpectedTranspose(src_data, expected_data, width, height);
    
    NppImageMemory<Npp32f> src(width, height);
    NppImageMemory<Npp32f> dst(height, width);
    
    src.copyFromHost(src_data);
    
    NppiSize roi = {width, height};
    NppStatus status = nppiTranspose_32f_C1R(
        src.get(), src.step(),
        dst.get(), dst.step(),
        roi);
    
    ASSERT_EQ(status, NPP_NO_ERROR) << "Transpose large matrix failed";
    
    std::vector<Npp32f> result(width * height);
    dst.copyToHost(result);
    
    EXPECT_TRUE(ResultValidator::arraysEqual(result, expected_data, 1e-5f))
        << "Transpose large matrix result incorrect";
    
    std::cout << "Large matrix transpose (" << width << "x" << height << ") completed successfully" << std::endl;
}

// ==================== 边界条件测试 ====================

TEST_F(TransposeFunctionalTest, Transpose_8u_C1R_SinglePixel) {
    const int size = 1;
    
    std::vector<Npp8u> src_data = {42};
    std::vector<Npp8u> expected_data = {42};
    
    NppImageMemory<Npp8u> src(size, size);
    NppImageMemory<Npp8u> dst(size, size);
    
    src.copyFromHost(src_data);
    
    NppiSize roi = {size, size};
    NppStatus status = nppiTranspose_8u_C1R(
        src.get(), src.step(),
        dst.get(), dst.step(),
        roi);
    
    ASSERT_EQ(status, NPP_NO_ERROR) << "Transpose single pixel failed";
    
    std::vector<Npp8u> result(1);
    dst.copyToHost(result);
    
    EXPECT_EQ(result[0], 42) << "Single pixel transpose failed";
}

TEST_F(TransposeFunctionalTest, Transpose_32f_C1R_SingleRow) {
    const int width = 8;
    const int height = 1;
    
    std::vector<Npp32f> src_data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    std::vector<Npp32f> expected_data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    
    NppImageMemory<Npp32f> src(width, height);
    NppImageMemory<Npp32f> dst(height, width);  // 转置后变成8x1
    
    src.copyFromHost(src_data);
    
    NppiSize roi = {width, height};
    NppStatus status = nppiTranspose_32f_C1R(
        src.get(), src.step(),
        dst.get(), dst.step(),
        roi);
    
    ASSERT_EQ(status, NPP_NO_ERROR) << "Transpose single row failed";
    
    std::vector<Npp32f> result(width * height);
    dst.copyToHost(result);
    
    EXPECT_TRUE(ResultValidator::arraysEqual(result, expected_data, 1e-5f))
        << "Single row transpose result incorrect";
}

TEST_F(TransposeFunctionalTest, Transpose_32f_C1R_SingleColumn) {
    const int width = 1;
    const int height = 6;
    
    std::vector<Npp32f> src_data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    std::vector<Npp32f> expected_data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    
    NppImageMemory<Npp32f> src(width, height);
    NppImageMemory<Npp32f> dst(height, width);  // 转置后变成6x1
    
    src.copyFromHost(src_data);
    
    NppiSize roi = {width, height};
    NppStatus status = nppiTranspose_32f_C1R(
        src.get(), src.step(),
        dst.get(), dst.step(),
        roi);
    
    ASSERT_EQ(status, NPP_NO_ERROR) << "Transpose single column failed";
    
    std::vector<Npp32f> result(width * height);
    dst.copyToHost(result);
    
    EXPECT_TRUE(ResultValidator::arraysEqual(result, expected_data, 1e-5f))
        << "Single column transpose result incorrect";
}

// ==================== 双重转置测试 ====================

TEST_F(TransposeFunctionalTest, Transpose_8u_C1R_DoubleTranspose) {
    const int width = 5;
    const int height = 7;
    
    // 准备原始数据
    std::vector<Npp8u> original_data(width * height);
    generateTestMatrix(original_data, width, height);
    
    // 第一次转置
    NppImageMemory<Npp8u> src(width, height);
    NppImageMemory<Npp8u> temp(height, width);
    NppImageMemory<Npp8u> result(width, height);
    
    src.copyFromHost(original_data);
    
    // 转置1: width x height -> height x width
    NppiSize roi1 = {width, height};
    NppStatus status1 = nppiTranspose_8u_C1R(
        src.get(), src.step(),
        temp.get(), temp.step(),
        roi1);
    ASSERT_EQ(status1, NPP_NO_ERROR) << "First transpose failed";
    
    // 转置2: height x width -> width x height (应该恢复原样)
    NppiSize roi2 = {height, width};
    NppStatus status2 = nppiTranspose_8u_C1R(
        temp.get(), temp.step(),
        result.get(), result.step(),
        roi2);
    ASSERT_EQ(status2, NPP_NO_ERROR) << "Second transpose failed";
    
    // 验证双重转置后的结果与原始数据相同
    std::vector<Npp8u> final_result(width * height);
    result.copyToHost(final_result);
    
    EXPECT_TRUE(ResultValidator::arraysEqual(final_result, original_data))
        << "Double transpose should restore original data";
}

// ==================== 不同数据类型测试 ====================

TEST_F(TransposeFunctionalTest, Transpose_16u_C1R_DataTypeTest) {
    const int size = 4;
    
    std::vector<Npp16u> src_data = {
        1000, 2000, 3000, 4000,
        5000, 6000, 7000, 8000,
        9000, 10000, 11000, 12000,
        13000, 14000, 15000, 16000
    };
    
    std::vector<Npp16u> expected_data = {
        1000, 5000, 9000, 13000,
        2000, 6000, 10000, 14000,
        3000, 7000, 11000, 15000,
        4000, 8000, 12000, 16000
    };
    
    NppImageMemory<Npp16u> src(size, size);
    NppImageMemory<Npp16u> dst(size, size);
    
    src.copyFromHost(src_data);
    
    NppiSize roi = {size, size};
    NppStatus status = nppiTranspose_16u_C1R(
        src.get(), src.step(),
        dst.get(), dst.step(),
        roi);
    
    ASSERT_EQ(status, NPP_NO_ERROR) << "Transpose 16u failed";
    
    std::vector<Npp16u> result(size * size);
    dst.copyToHost(result);
    
    EXPECT_TRUE(ResultValidator::arraysEqual(result, expected_data))
        << "Transpose 16u result incorrect";
}

// ==================== 错误处理测试 ====================

TEST_F(TransposeFunctionalTest, ErrorHandling_NullPointer) {
    const int size = 4;
    
    NppImageMemory<Npp32f> src(size, size);
    NppImageMemory<Npp32f> dst(size, size);
    
    NppiSize roi = {size, size};
    
    // 测试null源指针
    NppStatus status = nppiTranspose_32f_C1R(
        nullptr, src.step(),
        dst.get(), dst.step(),
        roi);
    
    EXPECT_EQ(status, NPP_NULL_POINTER_ERROR) << "Should detect null source pointer";
    
    // 测试null目标指针
    status = nppiTranspose_32f_C1R(
        src.get(), src.step(),
        nullptr, dst.step(),
        roi);
    
    EXPECT_EQ(status, NPP_NULL_POINTER_ERROR) << "Should detect null destination pointer";
}

TEST_F(TransposeFunctionalTest, ErrorHandling_InvalidROI) {
    const int size = 4;
    
    NppImageMemory<Npp32f> src(size, size);
    NppImageMemory<Npp32f> dst(size, size);
    
    // 测试无效的ROI
    NppiSize invalid_roi = {0, size};  // 宽度为0
    NppStatus status = nppiTranspose_32f_C1R(
        src.get(), src.step(),
        dst.get(), dst.step(),
        invalid_roi);
    
    EXPECT_EQ(status, NPP_SIZE_ERROR) << "Should detect invalid ROI width";
    
    // 测试负数ROI
    invalid_roi = {size, -1};  // 负高度
    status = nppiTranspose_32f_C1R(
        src.get(), src.step(),
        dst.get(), dst.step(),
        invalid_roi);
    
    EXPECT_EQ(status, NPP_SIZE_ERROR) << "Should detect negative ROI height";
}

// ==================== 性能基准测试（不测量具体时间，只验证功能）====================

TEST_F(TransposeFunctionalTest, Performance_VariousSizes) {
    // 跳过大尺寸测试如果GPU内存不足
    if (!hasComputeCapability(3, 5)) {
        GTEST_SKIP() << "GPU compute capability too low for performance tests";
    }
    
    struct TestSize {
        int width;
        int height;
        std::string description;
    };
    
    std::vector<TestSize> test_sizes = {
        {32, 32, "Small 32x32"},
        {128, 128, "Medium 128x128"},
        {256, 256, "Large 256x256"},
        {512, 1024, "Rectangular 512x1024"}
    };
    
    for (const auto& test_size : test_sizes) {
        std::vector<Npp32f> src_data(test_size.width * test_size.height);
        TestDataGenerator::generateRandom(src_data, -100.0f, 100.0f);
        
        NppImageMemory<Npp32f> src(test_size.width, test_size.height);
        NppImageMemory<Npp32f> dst(test_size.height, test_size.width);
        
        src.copyFromHost(src_data);
        
        NppiSize roi = {test_size.width, test_size.height};
        NppStatus status = nppiTranspose_32f_C1R(
            src.get(), src.step(),
            dst.get(), dst.step(),
            roi);
        
        EXPECT_EQ(status, NPP_NO_ERROR) << "Failed for " << test_size.description;
        
        std::cout << test_size.description << " transpose completed successfully" << std::endl;
    }
}