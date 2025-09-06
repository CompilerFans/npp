#include <gtest/gtest.h>
#include <vector>
#include <cmath>
#include <algorithm>
#include <random>
#include "npp.h"
#include "../framework/npp_test_base.h"

using namespace npp_functional_test;

/**
 * NPPS Sum Functions Test Suite
 * 测试NPPS信号求和函数
 */
class NPPSSumFunctionalTest : public ::testing::Test {
protected:
    void SetUp() override {
        // 获取GPU设备名称用于测试输出
        const char* deviceName = nppGetGpuName();
        if (deviceName) {
            std::cout << "Using GPU: " << deviceName << std::endl;
        }
    }
};

// ==============================================================================
// nppsSum_32f Tests - 32-bit float summation
// ==============================================================================

TEST_F(NPPSSumFunctionalTest, Sum_32f_BasicOperation) {
    const size_t nLength = 1024;
    
    // 准备测试数据
    std::vector<Npp32f> src(nLength);
    float expectedSum = 0.0f;
    for (size_t i = 0; i < nLength; i++) {
        src[i] = static_cast<float>(i + 1);
        expectedSum += src[i];
    }
    
    // 分配GPU内存
    Npp32f* d_src = nullptr;
    Npp32f* d_sum = nullptr;
    Npp8u* d_buffer = nullptr;
    
    cudaMalloc(&d_src, nLength * sizeof(Npp32f));
    cudaMalloc(&d_sum, sizeof(Npp32f));
    
    // 获取所需缓冲区大小
    size_t bufferSize;
    NppStatus status = nppsSumGetBufferSize_32f(nLength, &bufferSize);
    ASSERT_EQ(status, NPP_NO_ERROR) << "nppsSumGetBufferSize_32f failed";
    
    cudaMalloc(&d_buffer, bufferSize);
    ASSERT_TRUE(d_src && d_sum && d_buffer) << "GPU memory allocation failed";
    
    // 复制数据到GPU
    cudaMemcpy(d_src, src.data(), nLength * sizeof(Npp32f), cudaMemcpyHostToDevice);
    
    // 执行NPP求和运算
    status = nppsSum_32f(d_src, nLength, d_sum, d_buffer);
    ASSERT_EQ(status, NPP_NO_ERROR) << "nppsSum_32f failed";
    
    // 复制结果回主机
    float result;
    cudaMemcpy(&result, d_sum, sizeof(Npp32f), cudaMemcpyDeviceToHost);
    
    // 验证结果（允许浮点精度误差）
    EXPECT_NEAR(result, expectedSum, expectedSum * 1e-6f) << "Sum result mismatch";
    
    // 清理GPU内存
    cudaFree(d_src);
    cudaFree(d_sum);
    cudaFree(d_buffer);
}

TEST_F(NPPSSumFunctionalTest, Sum_32f_SequentialPattern) {
    const size_t nLength = 10;
    
    // 使用简单的序列: [1, 2, 3, ..., 10]
    std::vector<Npp32f> src(nLength);
    for (size_t i = 0; i < nLength; i++) {
        src[i] = static_cast<float>(i + 1);
    }
    float expectedSum = 55.0f; // 1+2+...+10 = 55
    
    // 分配GPU内存
    Npp32f* d_src = nullptr;
    Npp32f* d_sum = nullptr;
    Npp8u* d_buffer = nullptr;
    
    cudaMalloc(&d_src, nLength * sizeof(Npp32f));
    cudaMalloc(&d_sum, sizeof(Npp32f));
    
    size_t bufferSize;
    NppStatus status = nppsSumGetBufferSize_32f(nLength, &bufferSize);
    ASSERT_EQ(status, NPP_NO_ERROR);
    
    cudaMalloc(&d_buffer, bufferSize);
    ASSERT_TRUE(d_src && d_sum && d_buffer);
    
    // 复制数据并执行求和
    cudaMemcpy(d_src, src.data(), nLength * sizeof(Npp32f), cudaMemcpyHostToDevice);
    status = nppsSum_32f(d_src, nLength, d_sum, d_buffer);
    ASSERT_EQ(status, NPP_NO_ERROR);
    
    // 验证结果
    float result;
    cudaMemcpy(&result, d_sum, sizeof(Npp32f), cudaMemcpyDeviceToHost);
    EXPECT_FLOAT_EQ(result, expectedSum);
    
    // 清理GPU内存
    cudaFree(d_src);
    cudaFree(d_sum);
    cudaFree(d_buffer);
}

TEST_F(NPPSSumFunctionalTest, Sum_32f_LargeSignal) {
    const size_t nLength = 1024 * 1024;  // 1M elements
    
    // 使用常数值以便验证结果
    std::vector<Npp32f> src(nLength, 2.5f);
    float expectedSum = static_cast<float>(nLength) * 2.5f;
    
    // 分配GPU内存
    Npp32f* d_src = nullptr;
    Npp32f* d_sum = nullptr;
    Npp8u* d_buffer = nullptr;
    
    cudaMalloc(&d_src, nLength * sizeof(Npp32f));
    cudaMalloc(&d_sum, sizeof(Npp32f));
    
    size_t bufferSize;
    NppStatus status = nppsSumGetBufferSize_32f(nLength, &bufferSize);
    ASSERT_EQ(status, NPP_NO_ERROR);
    
    cudaMalloc(&d_buffer, bufferSize);
    ASSERT_TRUE(d_src && d_sum && d_buffer);
    
    // 复制数据并执行求和
    cudaMemcpy(d_src, src.data(), nLength * sizeof(Npp32f), cudaMemcpyHostToDevice);
    status = nppsSum_32f(d_src, nLength, d_sum, d_buffer);
    ASSERT_EQ(status, NPP_NO_ERROR);
    
    // 验证结果
    float result;
    cudaMemcpy(&result, d_sum, sizeof(Npp32f), cudaMemcpyDeviceToHost);
    EXPECT_NEAR(result, expectedSum, expectedSum * 1e-6f);
    
    // 清理GPU内存
    cudaFree(d_src);
    cudaFree(d_sum);
    cudaFree(d_buffer);
}

// ==============================================================================
// nppsSum_32fc Tests - 32-bit float complex summation
// ==============================================================================

TEST_F(NPPSSumFunctionalTest, Sum_32fc_BasicOperation) {
    const size_t nLength = 100;
    
    // 准备复数测试数据
    std::vector<Npp32fc> src(nLength);
    Npp32fc expectedSum = {0.0f, 0.0f};
    for (size_t i = 0; i < nLength; i++) {
        src[i].re = static_cast<float>(i + 1);
        src[i].im = static_cast<float>(i * 0.5f);
        expectedSum.re += src[i].re;
        expectedSum.im += src[i].im;
    }
    
    // 分配GPU内存
    Npp32fc* d_src = nullptr;
    Npp32fc* d_sum = nullptr;
    Npp8u* d_buffer = nullptr;
    
    cudaMalloc(&d_src, nLength * sizeof(Npp32fc));
    cudaMalloc(&d_sum, sizeof(Npp32fc));
    
    size_t bufferSize;
    NppStatus status = nppsSumGetBufferSize_32fc(nLength, &bufferSize);
    ASSERT_EQ(status, NPP_NO_ERROR);
    
    cudaMalloc(&d_buffer, bufferSize);
    ASSERT_TRUE(d_src && d_sum && d_buffer);
    
    // 复制数据并执行求和
    cudaMemcpy(d_src, src.data(), nLength * sizeof(Npp32fc), cudaMemcpyHostToDevice);
    status = nppsSum_32fc(d_src, nLength, d_sum, d_buffer);
    ASSERT_EQ(status, NPP_NO_ERROR);
    
    // 验证结果
    Npp32fc result;
    cudaMemcpy(&result, d_sum, sizeof(Npp32fc), cudaMemcpyDeviceToHost);
    
    EXPECT_NEAR(result.re, expectedSum.re, std::abs(expectedSum.re) * 1e-6f);
    EXPECT_NEAR(result.im, expectedSum.im, std::abs(expectedSum.im) * 1e-6f);
    
    // 清理GPU内存
    cudaFree(d_src);
    cudaFree(d_sum);
    cudaFree(d_buffer);
}

// ==============================================================================
// Buffer Size Tests
// ==============================================================================

TEST_F(NPPSSumFunctionalTest, GetBufferSize_32f) {
    size_t bufferSize;
    
    // 测试不同长度的缓冲区大小计算
    for (size_t nLength : {1, 100, 1000, 10000, 100000}) {
        NppStatus status = nppsSumGetBufferSize_32f(nLength, &bufferSize);
        ASSERT_EQ(status, NPP_NO_ERROR) << "Failed for length " << nLength;
        EXPECT_GT(bufferSize, 0) << "Buffer size should be > 0 for length " << nLength;
    }
}

// ==============================================================================
// Error Handling Tests
// ==============================================================================

TEST_F(NPPSSumFunctionalTest, ErrorHandling_NullPointers) {
    const size_t nLength = 100;
    size_t bufferSize;
    
    // 测试空指针错误处理
    EXPECT_EQ(nppsSumGetBufferSize_32f(nLength, nullptr), NPP_NULL_POINTER_ERROR);
    EXPECT_EQ(nppsSum_32f(nullptr, nLength, nullptr, nullptr), NPP_NULL_POINTER_ERROR);
    
    // 正常获取缓冲区大小用于后续测试
    NppStatus status = nppsSumGetBufferSize_32f(nLength, &bufferSize);
    ASSERT_EQ(status, NPP_NO_ERROR);
    
    Npp32f* d_dummy = nullptr;
    Npp8u* d_buffer = nullptr;
    cudaMalloc(&d_dummy, nLength * sizeof(Npp32f));
    cudaMalloc(&d_buffer, bufferSize);
    
    EXPECT_EQ(nppsSum_32f(nullptr, nLength, d_dummy, d_buffer), NPP_NULL_POINTER_ERROR);
    EXPECT_EQ(nppsSum_32f(d_dummy, nLength, nullptr, d_buffer), NPP_NULL_POINTER_ERROR);
    EXPECT_EQ(nppsSum_32f(d_dummy, nLength, d_dummy, nullptr), NPP_NULL_POINTER_ERROR);
    
    cudaFree(d_dummy);
    cudaFree(d_buffer);
}

TEST_F(NPPSSumFunctionalTest, ErrorHandling_ZeroLength) {
    size_t bufferSize;
    
    // 测试零长度错误处理
    EXPECT_EQ(nppsSumGetBufferSize_32f(0, &bufferSize), NPP_SIZE_ERROR);
    
    Npp32f* d_dummy = nullptr;
    cudaMalloc(&d_dummy, sizeof(Npp32f));
    
    EXPECT_EQ(nppsSum_32f(d_dummy, 0, d_dummy, reinterpret_cast<Npp8u*>(d_dummy)), NPP_SIZE_ERROR);
    
    cudaFree(d_dummy);
}