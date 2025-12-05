#include "npp.h"
#include "npp_test_base.h"
#include "npp_version_compat.h"

#include <algorithm>
#include <cmath>
#include <gtest/gtest.h>
#include <random>
#include <vector>

// Implementation file
class NPPSSumFunctionalTest : public ::testing::Test {
protected:
  void SetUp() override {}
};

// ==============================================================================
// nppsSum_32f Tests - 32-bit float summation
// ==============================================================================

TEST_F(NPPSSumFunctionalTest, Sum_32f_BasicOperation) {
  const NppSignalLength nLength = 1024;

  // prepare test data
  std::vector<Npp32f> src(nLength);
  float expectedSum = 0.0f;
  for (NppSignalLength i = 0; i < nLength; i++) {
    src[i] = static_cast<float>(i + 1);
    expectedSum += src[i];
  }

  // 分配GPU内存
  Npp32f *d_src = nullptr;
  Npp32f *d_sum = nullptr;
  Npp8u *d_buffer = nullptr;

  cudaMalloc(&d_src, nLength * sizeof(Npp32f));
  cudaMalloc(&d_sum, sizeof(Npp32f));

  // 获取所需缓冲区大小
  NppSignalLength bufferSize;
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

  // Validate结果（允许浮点精度误差）
  EXPECT_NEAR(result, expectedSum, expectedSum * 1e-6f) << "Sum result mismatch";

  // 清理GPU内存
  cudaFree(d_src);
  cudaFree(d_sum);
  cudaFree(d_buffer);
}

TEST_F(NPPSSumFunctionalTest, Sum_32f_SequentialPattern) {
  const NppSignalLength nLength = 10;

  // 使用简单的序列: [1, 2, 3, ..., 10]
  std::vector<Npp32f> src(nLength);
  for (NppSignalLength i = 0; i < nLength; i++) {
    src[i] = static_cast<float>(i + 1);
  }
  float expectedSum = 55.0f; // 1+2+...+10 = 55

  // 分配GPU内存
  Npp32f *d_src = nullptr;
  Npp32f *d_sum = nullptr;
  Npp8u *d_buffer = nullptr;

  cudaMalloc(&d_src, nLength * sizeof(Npp32f));
  cudaMalloc(&d_sum, sizeof(Npp32f));

  NppSignalLength bufferSize;
  NppStatus status = nppsSumGetBufferSize_32f(nLength, &bufferSize);
  ASSERT_EQ(status, NPP_NO_ERROR);

  cudaMalloc(&d_buffer, bufferSize);
  ASSERT_TRUE(d_src && d_sum && d_buffer);

  // 复制数据并执行求和
  cudaMemcpy(d_src, src.data(), nLength * sizeof(Npp32f), cudaMemcpyHostToDevice);
  status = nppsSum_32f(d_src, nLength, d_sum, d_buffer);
  ASSERT_EQ(status, NPP_NO_ERROR);

  // Validate结果
  float result;
  cudaMemcpy(&result, d_sum, sizeof(Npp32f), cudaMemcpyDeviceToHost);
  EXPECT_FLOAT_EQ(result, expectedSum);

  // 清理GPU内存
  cudaFree(d_src);
  cudaFree(d_sum);
  cudaFree(d_buffer);
}

TEST_F(NPPSSumFunctionalTest, Sum_32f_LargeSignal) {
  const NppSignalLength nLength = 1024 * 1024; // 1M elements

  // 使用常数值以便Validate结果
  std::vector<Npp32f> src(nLength, 2.5f);
  float expectedSum = static_cast<float>(nLength) * 2.5f;

  // 分配GPU内存
  Npp32f *d_src = nullptr;
  Npp32f *d_sum = nullptr;
  Npp8u *d_buffer = nullptr;

  cudaMalloc(&d_src, nLength * sizeof(Npp32f));
  cudaMalloc(&d_sum, sizeof(Npp32f));

  NppSignalLength bufferSize;
  NppStatus status = nppsSumGetBufferSize_32f(nLength, &bufferSize);
  ASSERT_EQ(status, NPP_NO_ERROR);

  cudaMalloc(&d_buffer, bufferSize);
  ASSERT_TRUE(d_src && d_sum && d_buffer);

  // 复制数据并执行求和
  cudaMemcpy(d_src, src.data(), nLength * sizeof(Npp32f), cudaMemcpyHostToDevice);
  status = nppsSum_32f(d_src, nLength, d_sum, d_buffer);
  ASSERT_EQ(status, NPP_NO_ERROR);

  // Validate结果
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
  const NppSignalLength nLength = 100;

  // 准备复数测试数据
  std::vector<Npp32fc> src(nLength);
  Npp32fc expectedSum = {0.0f, 0.0f};
  for (NppSignalLength i = 0; i < nLength; i++) {
    src[i].re = static_cast<float>(i + 1);
    src[i].im = static_cast<float>(i * 0.5f);
    expectedSum.re += src[i].re;
    expectedSum.im += src[i].im;
  }

  // 分配GPU内存
  Npp32fc *d_src = nullptr;
  Npp32fc *d_sum = nullptr;
  Npp8u *d_buffer = nullptr;

  cudaMalloc(&d_src, nLength * sizeof(Npp32fc));
  cudaMalloc(&d_sum, sizeof(Npp32fc));

  NppSignalLength bufferSize;
  NppStatus status = nppsSumGetBufferSize_32fc(nLength, &bufferSize);
  ASSERT_EQ(status, NPP_NO_ERROR);

  cudaMalloc(&d_buffer, bufferSize);
  ASSERT_TRUE(d_src && d_sum && d_buffer);

  // 复制数据并执行求和
  cudaMemcpy(d_src, src.data(), nLength * sizeof(Npp32fc), cudaMemcpyHostToDevice);
  status = nppsSum_32fc(d_src, nLength, d_sum, d_buffer);
  ASSERT_EQ(status, NPP_NO_ERROR);

  // Validate结果
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
  NppSignalLength bufferSize;

  // 测试不同长度的缓冲区大小计算
  for (NppSignalLength nLength : {1, 100, 1000, 10000, 100000}) {
    NppStatus status = nppsSumGetBufferSize_32f(nLength, &bufferSize);
    ASSERT_EQ(status, NPP_NO_ERROR) << "Failed for length " << nLength;
    EXPECT_GT(bufferSize, 0) << "Buffer size should be > 0 for length " << nLength;
  }
}
