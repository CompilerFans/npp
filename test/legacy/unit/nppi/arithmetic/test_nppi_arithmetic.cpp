/**
 * @file test_nppi_arithmetic.cpp
 * @brief NPPI算术运算完整测试套件 - 按源码结构组织
 * 
 * 测试覆盖：
 * - nppi_addc.cpp/cu - 加常数运算
 * - nppi_subc.cpp/cu - 减常数运算  
 * - nppi_mulc.cpp/cu - 乘常数运算
 * - nppi_divc.cpp/cu - 除常数运算
 */

#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <vector>
#include <random>
#include <chrono>
#include <memory>
#include "npp.h"

#ifdef HAVE_NVIDIA_NPP
#include <nppdefs.h>
#include <nppi_arithmetic_and_logical_operations.h>
#include <nppi_support_functions.h>
#endif

// ==================== 测试配置 ====================
namespace TestConfig {
    constexpr int DEFAULT_WIDTH = 1024;
    constexpr int DEFAULT_HEIGHT = 768;
    constexpr int PERF_WIDTH = 1920;
    constexpr int PERF_HEIGHT = 1080;
    constexpr float EPSILON_32F = 1e-5f;
    constexpr int MAX_DIFF_INTEGER = 1;  // 整数运算允许的最大差异
}

// ==================== 测试基类 ====================
class NPPIArithmeticTestBase : public ::testing::Test {
protected:
    void SetUp() override {
        cudaError_t cudaErr = cudaSetDevice(0);
        ASSERT_EQ(cudaErr, cudaSuccess) << "Failed to set CUDA device";
        nppGetStreamContext(&nppStreamCtx);
        
#ifdef HAVE_NVIDIA_NPP
        hasNvidiaNPP = true;
#else
        hasNvidiaNPP = false;
#endif
    }
    
    void TearDown() override {
        cudaDeviceSynchronize();
    }
    
    // 分配NPPI内存
    template<typename T>
    T* AllocateNppiMemory(int width, int height, int channels, int& step) {
        Npp8u* ptr = nullptr;
        int allocWidth = width * channels * sizeof(T);
        
        if (channels == 1) {
            ptr = nppiMalloc_8u_C1(allocWidth, height, &step);
        } else if (channels == 3) {
            ptr = nppiMalloc_8u_C3(width * sizeof(T), height, &step);
        } else if (channels == 4) {
            ptr = nppiMalloc_8u_C4(width * sizeof(T), height, &step);
        }
        
        EXPECT_NE(ptr, nullptr) << "Failed to allocate NPPI memory";
        return reinterpret_cast<T*>(ptr);
    }
    
    // 初始化测试数据
    template<typename T>
    void InitializeTestData(T* devicePtr, int width, int height, int channels, int step) {
        std::vector<T> hostData(width * height * channels);
        std::random_device rd;
        std::mt19937 gen(rd());
        
        if constexpr (std::is_floating_point_v<T>) {
            std::uniform_real_distribution<float> dist(1.0f, 100.0f);
            for (size_t i = 0; i < hostData.size(); i++) {
                hostData[i] = static_cast<T>(dist(gen));
            }
        } else {
            std::uniform_int_distribution<int> dist(1, 100);
            for (size_t i = 0; i < hostData.size(); i++) {
                hostData[i] = static_cast<T>(dist(gen));
            }
        }
        
        cudaMemcpy2D(devicePtr, step, hostData.data(), 
                    width * channels * sizeof(T), 
                    width * channels * sizeof(T), 
                    height, cudaMemcpyHostToDevice);
    }
    
    // 验证结果
    template<typename T>
    void VerifyResult(T* deviceResult, int width, int height, int step, 
                     T expectedValue, int index = 0) {
        std::vector<T> hostResult(width * height);
        cudaMemcpy2D(hostResult.data(), width * sizeof(T), 
                    deviceResult, step, width * sizeof(T), 
                    height, cudaMemcpyDeviceToHost);
        
        if constexpr (std::is_floating_point_v<T>) {
            EXPECT_NEAR(hostResult[index], expectedValue, TestConfig::EPSILON_32F);
        } else {
            EXPECT_NEAR(static_cast<int>(hostResult[index]), 
                       static_cast<int>(expectedValue), 
                       TestConfig::MAX_DIFF_INTEGER);
        }
    }
    
    NppStreamContext nppStreamCtx;
    bool hasNvidiaNPP;
};

// ==================== nppi_addc 测试 ====================
class AddCTest : public NPPIArithmeticTestBase {};

// 8-bit unsigned AddC
TEST_F(AddCTest, AddC_8u_C1RSfs) {
    int width = TestConfig::DEFAULT_WIDTH;
    int height = TestConfig::DEFAULT_HEIGHT;
    int srcStep, dstStep;
    
    Npp8u* src = AllocateNppiMemory<Npp8u>(width, height, 1, srcStep);
    Npp8u* dst = AllocateNppiMemory<Npp8u>(width, height, 1, dstStep);
    
    ASSERT_NE(src, nullptr);
    ASSERT_NE(dst, nullptr);
    
    InitializeTestData(src, width, height, 1, srcStep);
    
    Npp8u constant = 50;
    NppiSize roi = {width, height};
    
    NppStatus status = nppiAddC_8u_C1RSfs(src, srcStep, constant, dst, dstStep, roi, 0);
    EXPECT_EQ(status, NPP_SUCCESS);
    
    // 获取第一个像素进行验证
    std::vector<Npp8u> hostSrc(1);
    cudaMemcpy(hostSrc.data(), src, sizeof(Npp8u), cudaMemcpyDeviceToHost);
    Npp8u expected = (hostSrc[0] + constant > 255) ? 255 : (hostSrc[0] + constant);
    
    VerifyResult(dst, width, height, dstStep, expected, 0);
    
    nppiFree(src);
    nppiFree(dst);
}

// 16-bit unsigned AddC
TEST_F(AddCTest, AddC_16u_C1RSfs) {
    int width = TestConfig::DEFAULT_WIDTH;
    int height = TestConfig::DEFAULT_HEIGHT;
    int srcStep, dstStep;
    
    Npp16u* src = AllocateNppiMemory<Npp16u>(width, height, 1, srcStep);
    Npp16u* dst = AllocateNppiMemory<Npp16u>(width, height, 1, dstStep);
    
    ASSERT_NE(src, nullptr);
    ASSERT_NE(dst, nullptr);
    
    InitializeTestData(src, width, height, 1, srcStep);
    
    Npp16u constant = 1000;
    NppiSize roi = {width, height};
    
    NppStatus status = nppiAddC_16u_C1RSfs(src, srcStep, constant, dst, dstStep, roi, 0);
    EXPECT_EQ(status, NPP_SUCCESS);
    
    nppiFree(src);
    nppiFree(dst);
}

// 16-bit signed AddC
TEST_F(AddCTest, AddC_16s_C1RSfs) {
    int width = TestConfig::DEFAULT_WIDTH;
    int height = TestConfig::DEFAULT_HEIGHT;
    int srcStep, dstStep;
    
    Npp16s* src = AllocateNppiMemory<Npp16s>(width, height, 1, srcStep);
    Npp16s* dst = AllocateNppiMemory<Npp16s>(width, height, 1, dstStep);
    
    ASSERT_NE(src, nullptr);
    ASSERT_NE(dst, nullptr);
    
    InitializeTestData(src, width, height, 1, srcStep);
    
    Npp16s constant = -500;
    NppiSize roi = {width, height};
    
    NppStatus status = nppiAddC_16s_C1RSfs(src, srcStep, constant, dst, dstStep, roi, 0);
    EXPECT_EQ(status, NPP_SUCCESS);
    
    nppiFree(src);
    nppiFree(dst);
}

// 32-bit float AddC
TEST_F(AddCTest, AddC_32f_C1R) {
    int width = TestConfig::DEFAULT_WIDTH;
    int height = TestConfig::DEFAULT_HEIGHT;
    int srcStep, dstStep;
    
    Npp32f* src = AllocateNppiMemory<Npp32f>(width, height, 1, srcStep);
    Npp32f* dst = AllocateNppiMemory<Npp32f>(width, height, 1, dstStep);
    
    ASSERT_NE(src, nullptr);
    ASSERT_NE(dst, nullptr);
    
    InitializeTestData(src, width, height, 1, srcStep);
    
    Npp32f constant = 3.14159f;
    NppiSize roi = {width, height};
    
    NppStatus status = nppiAddC_32f_C1R(src, srcStep, constant, dst, dstStep, roi);
    EXPECT_EQ(status, NPP_SUCCESS);
    
    // 验证第一个像素
    std::vector<Npp32f> hostSrc(1);
    cudaMemcpy(hostSrc.data(), src, sizeof(Npp32f), cudaMemcpyDeviceToHost);
    Npp32f expected = hostSrc[0] + constant;
    
    VerifyResult(dst, width, height, dstStep, expected, 0);
    
    nppiFree(src);
    nppiFree(dst);
}

// ==================== nppi_subc 测试 ====================
class SubCTest : public NPPIArithmeticTestBase {};

// 8-bit unsigned SubC
TEST_F(SubCTest, SubC_8u_C1RSfs) {
    int width = TestConfig::DEFAULT_WIDTH;
    int height = TestConfig::DEFAULT_HEIGHT;
    int srcStep, dstStep;
    
    Npp8u* src = AllocateNppiMemory<Npp8u>(width, height, 1, srcStep);
    Npp8u* dst = AllocateNppiMemory<Npp8u>(width, height, 1, dstStep);
    
    ASSERT_NE(src, nullptr);
    ASSERT_NE(dst, nullptr);
    
    InitializeTestData(src, width, height, 1, srcStep);
    
    Npp8u constant = 25;
    NppiSize roi = {width, height};
    
    NppStatus status = nppiSubC_8u_C1RSfs(src, srcStep, constant, dst, dstStep, roi, 0);
    EXPECT_EQ(status, NPP_SUCCESS);
    
    nppiFree(src);
    nppiFree(dst);
}

// 32-bit signed SubC (扩展功能)
TEST_F(SubCTest, SubC_32s_C1RSfs) {
    int width = TestConfig::DEFAULT_WIDTH;
    int height = TestConfig::DEFAULT_HEIGHT;
    int srcStep, dstStep;
    
    Npp32s* src = AllocateNppiMemory<Npp32s>(width, height, 1, srcStep);
    Npp32s* dst = AllocateNppiMemory<Npp32s>(width, height, 1, dstStep);
    
    ASSERT_NE(src, nullptr);
    ASSERT_NE(dst, nullptr);
    
    InitializeTestData(src, width, height, 1, srcStep);
    
    Npp32s constant = 100000;
    NppiSize roi = {width, height};
    
    NppStatus status = nppiSubC_32s_C1RSfs(src, srcStep, constant, dst, dstStep, roi, 0);
    EXPECT_EQ(status, NPP_SUCCESS);
    
    nppiFree(src);
    nppiFree(dst);
}

// 32-bit float SubC
TEST_F(SubCTest, SubC_32f_C1R) {
    int width = TestConfig::DEFAULT_WIDTH;
    int height = TestConfig::DEFAULT_HEIGHT;
    int srcStep, dstStep;
    
    Npp32f* src = AllocateNppiMemory<Npp32f>(width, height, 1, srcStep);
    Npp32f* dst = AllocateNppiMemory<Npp32f>(width, height, 1, dstStep);
    
    ASSERT_NE(src, nullptr);
    ASSERT_NE(dst, nullptr);
    
    InitializeTestData(src, width, height, 1, srcStep);
    
    Npp32f constant = 2.71828f;
    NppiSize roi = {width, height};
    
    NppStatus status = nppiSubC_32f_C1R(src, srcStep, constant, dst, dstStep, roi);
    EXPECT_EQ(status, NPP_SUCCESS);
    
    nppiFree(src);
    nppiFree(dst);
}

// 多通道SubC测试
TEST_F(SubCTest, SubC_8u_C3RSfs) {
    int width = TestConfig::DEFAULT_WIDTH;
    int height = TestConfig::DEFAULT_HEIGHT;
    int srcStep, dstStep;
    
    Npp8u* src = nppiMalloc_8u_C3(width, height, &srcStep);
    Npp8u* dst = nppiMalloc_8u_C3(width, height, &dstStep);
    
    ASSERT_NE(src, nullptr);
    ASSERT_NE(dst, nullptr);
    
    InitializeTestData(src, width, height, 3, srcStep);
    
    Npp8u constants[3] = {10, 20, 30};
    NppiSize roi = {width, height};
    
    NppStatus status = nppiSubC_8u_C3RSfs(src, srcStep, constants, dst, dstStep, roi, 0);
    EXPECT_EQ(status, NPP_SUCCESS);
    
    nppiFree(src);
    nppiFree(dst);
}

// In-place SubC测试
TEST_F(SubCTest, SubC_8u_C1IRSfs_InPlace) {
    int width = TestConfig::DEFAULT_WIDTH;
    int height = TestConfig::DEFAULT_HEIGHT;
    int step;
    
    Npp8u* data = nppiMalloc_8u_C1(width, height, &step);
    ASSERT_NE(data, nullptr);
    
    InitializeTestData(data, width, height, 1, step);
    
    // 保存原始值用于验证
    std::vector<Npp8u> original(width * height);
    cudaMemcpy2D(original.data(), width, data, step, width, height, cudaMemcpyDeviceToHost);
    
    Npp8u constant = 25;
    NppiSize roi = {width, height};
    
    NppStatus status = nppiSubC_8u_C1IRSfs(constant, data, step, roi, 0);
    EXPECT_EQ(status, NPP_SUCCESS);
    
    // 验证结果
    std::vector<Npp8u> result(width * height);
    cudaMemcpy2D(result.data(), width, data, step, width, height, cudaMemcpyDeviceToHost);
    
    Npp8u expected = (original[0] > constant) ? (original[0] - constant) : 0;
    EXPECT_EQ(result[0], expected);
    
    nppiFree(data);
}

// ==================== nppi_mulc 测试 ====================
class MulCTest : public NPPIArithmeticTestBase {};

// 8-bit unsigned MulC
TEST_F(MulCTest, MulC_8u_C1RSfs) {
    int width = TestConfig::DEFAULT_WIDTH;
    int height = TestConfig::DEFAULT_HEIGHT;
    int srcStep, dstStep;
    
    Npp8u* src = AllocateNppiMemory<Npp8u>(width, height, 1, srcStep);
    Npp8u* dst = AllocateNppiMemory<Npp8u>(width, height, 1, dstStep);
    
    ASSERT_NE(src, nullptr);
    ASSERT_NE(dst, nullptr);
    
    InitializeTestData(src, width, height, 1, srcStep);
    
    Npp8u constant = 2;
    NppiSize roi = {width, height};
    int scaleFactor = 1; // 右移1位，相当于除以2
    
    NppStatus status = nppiMulC_8u_C1RSfs(src, srcStep, constant, dst, dstStep, roi, scaleFactor);
    EXPECT_EQ(status, NPP_SUCCESS);
    
    nppiFree(src);
    nppiFree(dst);
}

// 32-bit float MulC
TEST_F(MulCTest, MulC_32f_C1R) {
    int width = TestConfig::DEFAULT_WIDTH;
    int height = TestConfig::DEFAULT_HEIGHT;
    int srcStep, dstStep;
    
    Npp32f* src = AllocateNppiMemory<Npp32f>(width, height, 1, srcStep);
    Npp32f* dst = AllocateNppiMemory<Npp32f>(width, height, 1, dstStep);
    
    ASSERT_NE(src, nullptr);
    ASSERT_NE(dst, nullptr);
    
    InitializeTestData(src, width, height, 1, srcStep);
    
    Npp32f constant = 1.5f;
    NppiSize roi = {width, height};
    
    NppStatus status = nppiMulC_32f_C1R(src, srcStep, constant, dst, dstStep, roi);
    EXPECT_EQ(status, NPP_SUCCESS);
    
    nppiFree(src);
    nppiFree(dst);
}

// ==================== nppi_divc 测试 ====================
class DivCTest : public NPPIArithmeticTestBase {};

// 8-bit unsigned DivC
TEST_F(DivCTest, DivC_8u_C1RSfs) {
    int width = TestConfig::DEFAULT_WIDTH;
    int height = TestConfig::DEFAULT_HEIGHT;
    int srcStep, dstStep;
    
    Npp8u* src = AllocateNppiMemory<Npp8u>(width, height, 1, srcStep);
    Npp8u* dst = AllocateNppiMemory<Npp8u>(width, height, 1, dstStep);
    
    ASSERT_NE(src, nullptr);
    ASSERT_NE(dst, nullptr);
    
    InitializeTestData(src, width, height, 1, srcStep);
    
    Npp8u constant = 2;
    NppiSize roi = {width, height};
    
    NppStatus status = nppiDivC_8u_C1RSfs(src, srcStep, constant, dst, dstStep, roi, 0);
    EXPECT_EQ(status, NPP_SUCCESS);
    
    nppiFree(src);
    nppiFree(dst);
}

// 32-bit float DivC
TEST_F(DivCTest, DivC_32f_C1R) {
    int width = TestConfig::DEFAULT_WIDTH;
    int height = TestConfig::DEFAULT_HEIGHT;
    int srcStep, dstStep;
    
    Npp32f* src = AllocateNppiMemory<Npp32f>(width, height, 1, srcStep);
    Npp32f* dst = AllocateNppiMemory<Npp32f>(width, height, 1, dstStep);
    
    ASSERT_NE(src, nullptr);
    ASSERT_NE(dst, nullptr);
    
    InitializeTestData(src, width, height, 1, srcStep);
    
    Npp32f constant = 2.0f;
    NppiSize roi = {width, height};
    
    NppStatus status = nppiDivC_32f_C1R(src, srcStep, constant, dst, dstStep, roi);
    EXPECT_EQ(status, NPP_SUCCESS);
    
    nppiFree(src);
    nppiFree(dst);
}

// 除零错误测试
TEST_F(DivCTest, DivC_8u_DivideByZero) {
    int width = 32;
    int height = 32;
    int srcStep, dstStep;
    
    Npp8u* src = AllocateNppiMemory<Npp8u>(width, height, 1, srcStep);
    Npp8u* dst = AllocateNppiMemory<Npp8u>(width, height, 1, dstStep);
    
    ASSERT_NE(src, nullptr);
    ASSERT_NE(dst, nullptr);
    
    InitializeTestData(src, width, height, 1, srcStep);
    
    Npp8u constant = 0; // 除零
    NppiSize roi = {width, height};
    
    NppStatus status = nppiDivC_8u_C1RSfs(src, srcStep, constant, dst, dstStep, roi, 0);
    EXPECT_EQ(status, NPP_DIVIDE_BY_ZERO_ERROR);
    
    nppiFree(src);
    nppiFree(dst);
}

// ==================== 性能测试 ====================
class PerformanceTest : public NPPIArithmeticTestBase {};

TEST_F(PerformanceTest, AddC_32f_Performance) {
    int width = TestConfig::PERF_WIDTH;
    int height = TestConfig::PERF_HEIGHT;
    int srcStep, dstStep;
    
    Npp32f* src = AllocateNppiMemory<Npp32f>(width, height, 1, srcStep);
    Npp32f* dst = AllocateNppiMemory<Npp32f>(width, height, 1, dstStep);
    
    ASSERT_NE(src, nullptr);
    ASSERT_NE(dst, nullptr);
    
    InitializeTestData(src, width, height, 1, srcStep);
    
    Npp32f constant = 2.5f;
    NppiSize roi = {width, height};
    
    // 预热
    for (int i = 0; i < 10; i++) {
        nppiAddC_32f_C1R(src, srcStep, constant, dst, dstStep, roi);
    }
    cudaDeviceSynchronize();
    
    // 性能测试
    const int iterations = 100;
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < iterations; i++) {
        NppStatus status = nppiAddC_32f_C1R(src, srcStep, constant, dst, dstStep, roi);
        EXPECT_EQ(status, NPP_SUCCESS);
    }
    cudaDeviceSynchronize();
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    double avgTimeMs = duration.count() / (1000.0 * iterations);
    double throughputGBps = (width * height * sizeof(Npp32f) * 2.0) / 
                           (duration.count() / iterations * 1000.0);
    
    std::cout << "\nAddC_32f_C1R 性能测试:\n";
    std::cout << "  图像大小: " << width << "x" << height << "\n";
    std::cout << "  平均时间: " << avgTimeMs << " ms\n";
    std::cout << "  吞吐量: " << throughputGBps << " GB/s\n";
    
    EXPECT_GT(throughputGBps, 1.0) << "Performance below expected threshold";
    
    nppiFree(src);
    nppiFree(dst);
}

// ==================== 边界条件测试 ====================
class BoundaryTest : public NPPIArithmeticTestBase {};

TEST_F(BoundaryTest, NonAlignedWidth) {
    int width = 1023;  // 非对齐宽度
    int height = 769;
    int srcStep, dstStep;
    
    Npp8u* src = nppiMalloc_8u_C1(width, height, &srcStep);
    Npp8u* dst = nppiMalloc_8u_C1(width, height, &dstStep);
    
    ASSERT_NE(src, nullptr);
    ASSERT_NE(dst, nullptr);
    
    // 验证pitch对齐
    EXPECT_GE(srcStep, width);
    EXPECT_GE(dstStep, width);
    
    InitializeTestData(src, width, height, 1, srcStep);
    
    NppiSize roi = {width, height};
    NppStatus status = nppiAddC_8u_C1RSfs(src, srcStep, 50, dst, dstStep, roi, 0);
    
    EXPECT_EQ(status, NPP_SUCCESS);
    
    nppiFree(src);
    nppiFree(dst);
}

TEST_F(BoundaryTest, SmallImage) {
    int width = 1;
    int height = 1;
    int srcStep, dstStep;
    
    Npp8u* src = nppiMalloc_8u_C1(width, height, &srcStep);
    Npp8u* dst = nppiMalloc_8u_C1(width, height, &dstStep);
    
    ASSERT_NE(src, nullptr);
    ASSERT_NE(dst, nullptr);
    
    Npp8u srcValue = 100;
    cudaMemcpy(src, &srcValue, sizeof(Npp8u), cudaMemcpyHostToDevice);
    
    Npp8u constant = 55;
    NppiSize roi = {width, height};
    
    NppStatus status = nppiAddC_8u_C1RSfs(src, srcStep, constant, dst, dstStep, roi, 0);
    EXPECT_EQ(status, NPP_SUCCESS);
    
    Npp8u result;
    cudaMemcpy(&result, dst, sizeof(Npp8u), cudaMemcpyDeviceToHost);
    EXPECT_EQ(result, 155);
    
    nppiFree(src);
    nppiFree(dst);
}

// ==================== 与NVIDIA NPP对比测试 ====================
#ifdef HAVE_NVIDIA_NPP
class ComparisonTest : public NPPIArithmeticTestBase {};

TEST_F(ComparisonTest, AddC_8u_CompareWithNvidiaNPP) {
    int width = TestConfig::DEFAULT_WIDTH;
    int height = TestConfig::DEFAULT_HEIGHT;
    int srcStep, dstOpenStep, dstNvidiaStep;
    
    Npp8u* src = AllocateNppiMemory<Npp8u>(width, height, 1, srcStep);
    Npp8u* dstOpen = AllocateNppiMemory<Npp8u>(width, height, 1, dstOpenStep);
    Npp8u* dstNvidia = AllocateNppiMemory<Npp8u>(width, height, 1, dstNvidiaStep);
    
    ASSERT_NE(src, nullptr);
    ASSERT_NE(dstOpen, nullptr);
    ASSERT_NE(dstNvidia, nullptr);
    
    InitializeTestData(src, width, height, 1, srcStep);
    
    Npp8u constant = 50;
    NppiSize roi = {width, height};
    
    // OpenNPP
    NppStatus statusOpen = nppiAddC_8u_C1RSfs(src, srcStep, constant, 
                                              dstOpen, dstOpenStep, roi, 0);
    
    // NVIDIA NPP
    NppStatus statusNvidia = ::nppiAddC_8u_C1RSfs(src, srcStep, constant, 
                                                  dstNvidia, dstNvidiaStep, roi, 0);
    
    EXPECT_EQ(statusOpen, NPP_SUCCESS);
    EXPECT_EQ(statusNvidia, NPP_SUCCESS);
    
    // 比较结果
    std::vector<Npp8u> openResult(width * height);
    std::vector<Npp8u> nvidiaResult(width * height);
    
    cudaMemcpy2D(openResult.data(), width, dstOpen, dstOpenStep, 
                width, height, cudaMemcpyDeviceToHost);
    cudaMemcpy2D(nvidiaResult.data(), width, dstNvidia, dstNvidiaStep, 
                width, height, cudaMemcpyDeviceToHost);
    
    for (int i = 0; i < width * height; i++) {
        EXPECT_EQ(openResult[i], nvidiaResult[i]) 
            << "Mismatch at index " << i;
    }
    
    nppiFree(src);
    nppiFree(dstOpen);
    nppiFree(dstNvidia);
}
#endif

// Main函数由gtest_main提供