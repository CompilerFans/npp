#include "../../framework/npp_test_base.h"
#include "npp.h"
#include <algorithm>
#include <cmath>
#include <gtest/gtest.h>
#include <random>
#include <vector>

using namespace npp_functional_test;

/**
 * NPPS Add Functions Test Suite
 * 测试NPPS信号加法运算函数
 */
class NPPSAddFunctionalTest : public ::testing::Test {
protected:
  void SetUp() override {}
};

// ==============================================================================
// nppsAdd_32f Tests - 32-bit float addition
// ==============================================================================

TEST_F(NPPSAddFunctionalTest, Add_32f_BasicOperation) {
  const size_t nLength = 1024;

  // prepare test data
  std::vector<Npp32f> src1(nLength), src2(nLength), expected(nLength);
  for (size_t i = 0; i < nLength; i++) {
    src1[i] = static_cast<Npp32f>(i + 1);
    src2[i] = static_cast<Npp32f>(i * 2);
    expected[i] = src1[i] + src2[i];
  }

  // 分配GPU内存
  Npp32f *d_src1 = nullptr;
  Npp32f *d_src2 = nullptr;
  Npp32f *d_dst = nullptr;

  cudaMalloc(&d_src1, nLength * sizeof(Npp32f));
  cudaMalloc(&d_src2, nLength * sizeof(Npp32f));
  cudaMalloc(&d_dst, nLength * sizeof(Npp32f));

  ASSERT_TRUE(d_src1 && d_src2 && d_dst) << "GPU memory allocation failed";

  // 复制数据到GPU
  cudaMemcpy(d_src1, src1.data(), nLength * sizeof(Npp32f), cudaMemcpyHostToDevice);
  cudaMemcpy(d_src2, src2.data(), nLength * sizeof(Npp32f), cudaMemcpyHostToDevice);

  // 执行NPP加法运算
  NppStatus status = nppsAdd_32f(d_src1, d_src2, d_dst, nLength);
  ASSERT_EQ(status, NPP_NO_ERROR) << "nppsAdd_32f failed";

  // 复制结果回主机
  std::vector<Npp32f> result(nLength);
  cudaMemcpy(result.data(), d_dst, nLength * sizeof(Npp32f), cudaMemcpyDeviceToHost);

  // 验证结果
  for (size_t i = 0; i < nLength; i++) {
    EXPECT_FLOAT_EQ(result[i], expected[i]) << "Mismatch at index " << i;
  }

  // 清理GPU内存
  cudaFree(d_src1);
  cudaFree(d_src2);
  cudaFree(d_dst);
}

TEST_F(NPPSAddFunctionalTest, Add_32f_LargeSignals) {
  const size_t nLength = 1024 * 1024; // 1M elements

  // 使用随机数据
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> dis(-100.0f, 100.0f);

  std::vector<Npp32f> src1(nLength), src2(nLength);
  for (size_t i = 0; i < nLength; i++) {
    src1[i] = dis(gen);
    src2[i] = dis(gen);
  }

  // 分配GPU内存
  Npp32f *d_src1 = nullptr;
  Npp32f *d_src2 = nullptr;
  Npp32f *d_dst = nullptr;

  cudaMalloc(&d_src1, nLength * sizeof(Npp32f));
  cudaMalloc(&d_src2, nLength * sizeof(Npp32f));
  cudaMalloc(&d_dst, nLength * sizeof(Npp32f));

  ASSERT_TRUE(d_src1 && d_src2 && d_dst) << "GPU memory allocation failed";

  // 复制数据到GPU
  cudaMemcpy(d_src1, src1.data(), nLength * sizeof(Npp32f), cudaMemcpyHostToDevice);
  cudaMemcpy(d_src2, src2.data(), nLength * sizeof(Npp32f), cudaMemcpyHostToDevice);

  // 执行NPP加法运算
  NppStatus status = nppsAdd_32f(d_src1, d_src2, d_dst, nLength);
  ASSERT_EQ(status, NPP_NO_ERROR) << "nppsAdd_32f failed";

  // 验证几个采样点
  std::vector<Npp32f> result(100);
  cudaMemcpy(result.data(), d_dst, 100 * sizeof(Npp32f), cudaMemcpyDeviceToHost);

  for (int i = 0; i < 100; i++) {
    float expected = src1[i] + src2[i];
    EXPECT_FLOAT_EQ(result[i], expected) << "Mismatch at index " << i;
  }

  // 清理GPU内存
  cudaFree(d_src1);
  cudaFree(d_src2);
  cudaFree(d_dst);
}

// ==============================================================================
// nppsAdd_16s Tests - 16-bit signed integer addition with saturation
// ==============================================================================

TEST_F(NPPSAddFunctionalTest, Add_16s_BasicOperation) {
  const size_t nLength = 512;

  // prepare test data
  std::vector<Npp16s> src1(nLength), src2(nLength), expected(nLength);
  for (size_t i = 0; i < nLength; i++) {
    src1[i] = static_cast<Npp16s>(i % 100);
    src2[i] = static_cast<Npp16s>(i % 50);

    int result = static_cast<int>(src1[i]) + static_cast<int>(src2[i]);
    expected[i] = static_cast<Npp16s>(std::max(-32768, std::min(32767, result)));
  }

  // 分配GPU内存
  Npp16s *d_src1 = nullptr;
  Npp16s *d_src2 = nullptr;
  Npp16s *d_dst = nullptr;

  cudaMalloc(&d_src1, nLength * sizeof(Npp16s));
  cudaMalloc(&d_src2, nLength * sizeof(Npp16s));
  cudaMalloc(&d_dst, nLength * sizeof(Npp16s));

  ASSERT_TRUE(d_src1 && d_src2 && d_dst) << "GPU memory allocation failed";

  // 复制数据到GPU
  cudaMemcpy(d_src1, src1.data(), nLength * sizeof(Npp16s), cudaMemcpyHostToDevice);
  cudaMemcpy(d_src2, src2.data(), nLength * sizeof(Npp16s), cudaMemcpyHostToDevice);

  // 执行NPP加法运算
  NppStatus status = nppsAdd_16s(d_src1, d_src2, d_dst, nLength);
  ASSERT_EQ(status, NPP_NO_ERROR) << "nppsAdd_16s failed";

  // 复制结果回主机
  std::vector<Npp16s> result(nLength);
  cudaMemcpy(result.data(), d_dst, nLength * sizeof(Npp16s), cudaMemcpyDeviceToHost);

  // 验证结果
  for (size_t i = 0; i < nLength; i++) {
    EXPECT_EQ(result[i], expected[i]) << "Mismatch at index " << i;
  }

  // 清理GPU内存
  cudaFree(d_src1);
  cudaFree(d_src2);
  cudaFree(d_dst);
}

// ==============================================================================
// nppsAdd_32fc Tests - 32-bit float complex addition
// ==============================================================================

TEST_F(NPPSAddFunctionalTest, Add_32fc_BasicOperation) {
  const size_t nLength = 256;

  // prepare test data
  std::vector<Npp32fc> src1(nLength), src2(nLength), expected(nLength);
  for (size_t i = 0; i < nLength; i++) {
    src1[i].re = static_cast<float>(i);
    src1[i].im = static_cast<float>(i * 0.5);
    src2[i].re = static_cast<float>(i * 2);
    src2[i].im = static_cast<float>(i * 1.5);

    expected[i].re = src1[i].re + src2[i].re;
    expected[i].im = src1[i].im + src2[i].im;
  }

  // 分配GPU内存
  Npp32fc *d_src1 = nullptr;
  Npp32fc *d_src2 = nullptr;
  Npp32fc *d_dst = nullptr;

  cudaMalloc(&d_src1, nLength * sizeof(Npp32fc));
  cudaMalloc(&d_src2, nLength * sizeof(Npp32fc));
  cudaMalloc(&d_dst, nLength * sizeof(Npp32fc));

  ASSERT_TRUE(d_src1 && d_src2 && d_dst) << "GPU memory allocation failed";

  // 复制数据到GPU
  cudaMemcpy(d_src1, src1.data(), nLength * sizeof(Npp32fc), cudaMemcpyHostToDevice);
  cudaMemcpy(d_src2, src2.data(), nLength * sizeof(Npp32fc), cudaMemcpyHostToDevice);

  // 执行NPP复数加法运算
  NppStatus status = nppsAdd_32fc(d_src1, d_src2, d_dst, nLength);
  ASSERT_EQ(status, NPP_NO_ERROR) << "nppsAdd_32fc failed";

  // 复制结果回主机
  std::vector<Npp32fc> result(nLength);
  cudaMemcpy(result.data(), d_dst, nLength * sizeof(Npp32fc), cudaMemcpyDeviceToHost);

  // 验证结果
  for (size_t i = 0; i < nLength; i++) {
    EXPECT_FLOAT_EQ(result[i].re, expected[i].re) << "Real part mismatch at index " << i;
    EXPECT_FLOAT_EQ(result[i].im, expected[i].im) << "Imaginary part mismatch at index " << i;
  }

  // 清理GPU内存
  cudaFree(d_src1);
  cudaFree(d_src2);
  cudaFree(d_dst);
}

// ==============================================================================
// Error Handling Tests
// ==============================================================================

// 测试OpenNPP的错误处理机制 - 空指针检测
TEST_F(NPPSAddFunctionalTest, ErrorHandling_NullPointers) {
  const size_t nLength = 100;

  // 测试所有参数都为null的情况
  EXPECT_EQ(nppsAdd_32f(nullptr, nullptr, nullptr, nLength), NPP_NULL_POINTER_ERROR);

  // 分配有效的GPU内存用于测试
  Npp32f *d_valid = nullptr;
  cudaError_t cudaStatus = cudaMalloc(&d_valid, nLength * sizeof(Npp32f));
  ASSERT_EQ(cudaStatus, cudaSuccess) << "Failed to allocate GPU memory";

  // 测试各种单个null指针情况
  EXPECT_EQ(nppsAdd_32f(nullptr, d_valid, d_valid, nLength), NPP_NULL_POINTER_ERROR);
  EXPECT_EQ(nppsAdd_32f(d_valid, nullptr, d_valid, nLength), NPP_NULL_POINTER_ERROR);
  EXPECT_EQ(nppsAdd_32f(d_valid, d_valid, nullptr, nLength), NPP_NULL_POINTER_ERROR);

  // 验证有效参数时函数正常工作
  EXPECT_EQ(nppsAdd_32f(d_valid, d_valid, d_valid, nLength), NPP_NO_ERROR);

  // 清理内存
  cudaFree(d_valid);
}

// 测试OpenNPP的错误处理机制 - 大小验证
TEST_F(NPPSAddFunctionalTest, ErrorHandling_ZeroLength) {
  // 分配有效的GPU内存
  Npp32f *d_valid = nullptr;
  cudaError_t cudaStatus = cudaMalloc(&d_valid, sizeof(Npp32f));
  ASSERT_EQ(cudaStatus, cudaSuccess) << "Failed to allocate GPU memory";

  // 测试零长度应该返回错误
  EXPECT_EQ(nppsAdd_32f(d_valid, d_valid, d_valid, 0), NPP_SIZE_ERROR);

  // 验证正常长度时函数工作
  EXPECT_EQ(nppsAdd_32f(d_valid, d_valid, d_valid, 1), NPP_NO_ERROR);

  // 清理内存
  cudaFree(d_valid);
}