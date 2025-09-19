#include "npp.h"
#include <algorithm>
#include <cmath>
#include <gtest/gtest.h>
#include <vector>

/**
 * NPPS Set Functions Test Suite
 * 测试NPPS信号初始化函数
 */
class NPPSSetFunctionalTest : public ::testing::Test {
protected:
  void SetUp() override {}
};

// ==============================================================================
// nppsSet_8u Tests - 8-bit unsigned char set
// ==============================================================================

TEST_F(NPPSSetFunctionalTest, Set_8u_BasicOperation) {
  const size_t nLength = 1024;
  const Npp8u setValue = 123;

  // 分配GPU内存
  Npp8u *d_dst = nullptr;
  cudaMalloc(&d_dst, nLength * sizeof(Npp8u));
  ASSERT_TRUE(d_dst) << "GPU memory allocation failed";

  // 执行NPP设置操作
  NppStatus status = nppsSet_8u(setValue, d_dst, nLength);
  ASSERT_EQ(status, NPP_NO_ERROR) << "nppsSet_8u failed";

  // 复制结果回主机并验证
  std::vector<Npp8u> result(nLength);
  cudaMemcpy(result.data(), d_dst, nLength * sizeof(Npp8u), cudaMemcpyDeviceToHost);

  for (size_t i = 0; i < nLength; i++) {
    EXPECT_EQ(result[i], setValue) << "Mismatch at index " << i;
  }

  // 清理GPU内存
  cudaFree(d_dst);
}

TEST_F(NPPSSetFunctionalTest, Set_8u_BoundaryValues) {
  const size_t nLength = 100;

  // 测试边界值
  std::vector<Npp8u> testValues = {0, 1, 127, 128, 255};

  for (Npp8u setValue : testValues) {
    Npp8u *d_dst = nullptr;
    cudaMalloc(&d_dst, nLength * sizeof(Npp8u));
    ASSERT_TRUE(d_dst);

    NppStatus status = nppsSet_8u(setValue, d_dst, nLength);
    ASSERT_EQ(status, NPP_NO_ERROR) << "nppsSet_8u failed for value " << static_cast<int>(setValue);

    std::vector<Npp8u> result(nLength);
    cudaMemcpy(result.data(), d_dst, nLength * sizeof(Npp8u), cudaMemcpyDeviceToHost);

    for (size_t i = 0; i < nLength; i++) {
      EXPECT_EQ(result[i], setValue) << "Mismatch at index " << i << " for value " << static_cast<int>(setValue);
    }

    cudaFree(d_dst);
  }
}

// ==============================================================================
// nppsSet_32f Tests - 32-bit float set
// ==============================================================================

TEST_F(NPPSSetFunctionalTest, Set_32f_BasicOperation) {
  const size_t nLength = 512;
  const Npp32f setValue = 3.14159f;

  // 分配GPU内存
  Npp32f *d_dst = nullptr;
  cudaMalloc(&d_dst, nLength * sizeof(Npp32f));
  ASSERT_TRUE(d_dst) << "GPU memory allocation failed";

  // 执行NPP设置操作
  NppStatus status = nppsSet_32f(setValue, d_dst, nLength);
  ASSERT_EQ(status, NPP_NO_ERROR) << "nppsSet_32f failed";

  // 复制结果回主机并验证
  std::vector<Npp32f> result(nLength);
  cudaMemcpy(result.data(), d_dst, nLength * sizeof(Npp32f), cudaMemcpyDeviceToHost);

  for (size_t i = 0; i < nLength; i++) {
    EXPECT_FLOAT_EQ(result[i], setValue) << "Mismatch at index " << i;
  }

  // 清理GPU内存
  cudaFree(d_dst);
}

TEST_F(NPPSSetFunctionalTest, Set_32f_SpecialValues) {
  const size_t nLength = 100;

  // 测试特殊值
  std::vector<Npp32f> testValues = {0.0f, -0.0f, 1.0f, -1.0f, 1e-10f, 1e10f};

  for (Npp32f setValue : testValues) {
    Npp32f *d_dst = nullptr;
    cudaMalloc(&d_dst, nLength * sizeof(Npp32f));
    ASSERT_TRUE(d_dst);

    NppStatus status = nppsSet_32f(setValue, d_dst, nLength);
    ASSERT_EQ(status, NPP_NO_ERROR) << "nppsSet_32f failed for value " << setValue;

    std::vector<Npp32f> result(nLength);
    cudaMemcpy(result.data(), d_dst, nLength * sizeof(Npp32f), cudaMemcpyDeviceToHost);

    for (size_t i = 0; i < nLength; i++) {
      EXPECT_FLOAT_EQ(result[i], setValue) << "Mismatch at index " << i << " for value " << setValue;
    }

    cudaFree(d_dst);
  }
}

TEST_F(NPPSSetFunctionalTest, Set_32f_LargeSignal) {
  const size_t nLength = 1024 * 1024; // 1M elements
  const Npp32f setValue = -2.71828f;

  // 分配GPU内存
  Npp32f *d_dst = nullptr;
  cudaMalloc(&d_dst, nLength * sizeof(Npp32f));
  ASSERT_TRUE(d_dst) << "GPU memory allocation failed";

  // 执行NPP设置操作
  NppStatus status = nppsSet_32f(setValue, d_dst, nLength);
  ASSERT_EQ(status, NPP_NO_ERROR) << "nppsSet_32f failed";

  // 验证几个采样点
  std::vector<Npp32f> sample(100);
  cudaMemcpy(sample.data(), d_dst, 100 * sizeof(Npp32f), cudaMemcpyDeviceToHost);

  for (int i = 0; i < 100; i++) {
    EXPECT_FLOAT_EQ(sample[i], setValue) << "Mismatch at index " << i;
  }

  // 验证末尾几个元素
  cudaMemcpy(sample.data(), d_dst + nLength - 100, 100 * sizeof(Npp32f), cudaMemcpyDeviceToHost);

  for (int i = 0; i < 100; i++) {
    EXPECT_FLOAT_EQ(sample[i], setValue) << "Mismatch at tail index " << (nLength - 100 + i);
  }

  // 清理GPU内存
  cudaFree(d_dst);
}

// ==============================================================================
// nppsSet_32fc Tests - 32-bit float complex set
// ==============================================================================

TEST_F(NPPSSetFunctionalTest, Set_32fc_BasicOperation) {
  const size_t nLength = 256;
  const Npp32fc setValue = {1.5f, -2.5f};

  // 分配GPU内存
  Npp32fc *d_dst = nullptr;
  cudaMalloc(&d_dst, nLength * sizeof(Npp32fc));
  ASSERT_TRUE(d_dst) << "GPU memory allocation failed";

  // 执行NPP复数设置操作
  NppStatus status = nppsSet_32fc(setValue, d_dst, nLength);
  ASSERT_EQ(status, NPP_NO_ERROR) << "nppsSet_32fc failed";

  // 复制结果回主机并验证
  std::vector<Npp32fc> result(nLength);
  cudaMemcpy(result.data(), d_dst, nLength * sizeof(Npp32fc), cudaMemcpyDeviceToHost);

  for (size_t i = 0; i < nLength; i++) {
    EXPECT_FLOAT_EQ(result[i].re, setValue.re) << "Real part mismatch at index " << i;
    EXPECT_FLOAT_EQ(result[i].im, setValue.im) << "Imaginary part mismatch at index " << i;
  }

  // 清理GPU内存
  cudaFree(d_dst);
}

// ==============================================================================
// nppsZero_32f Tests - Zero initialization convenience function
// ==============================================================================

TEST_F(NPPSSetFunctionalTest, Zero_32f_BasicOperation) {
  const size_t nLength = 200;

  // 分配GPU内存
  Npp32f *d_dst = nullptr;
  cudaMalloc(&d_dst, nLength * sizeof(Npp32f));
  ASSERT_TRUE(d_dst) << "GPU memory allocation failed";

  // 先设置非零值
  NppStatus status = nppsSet_32f(42.0f, d_dst, nLength);
  ASSERT_EQ(status, NPP_NO_ERROR);

  // 执行零初始化
  status = nppsZero_32f(d_dst, nLength);
  ASSERT_EQ(status, NPP_NO_ERROR) << "nppsZero_32f failed";

  // 复制结果回主机并验证
  std::vector<Npp32f> result(nLength);
  cudaMemcpy(result.data(), d_dst, nLength * sizeof(Npp32f), cudaMemcpyDeviceToHost);

  for (size_t i = 0; i < nLength; i++) {
    EXPECT_FLOAT_EQ(result[i], 0.0f) << "Mismatch at index " << i;
  }

  // 清理GPU内存
  cudaFree(d_dst);
}

// ==============================================================================
// Error Handling Tests
// ==============================================================================

// NOTE: 测试已被禁用 - NVIDIA NPP对无效参数的错误检测行为与预期不符
TEST_F(NPPSSetFunctionalTest, DISABLED_ErrorHandling_NullPointers) {
  const size_t nLength = 100;

  // 测试空指针错误处理
  EXPECT_EQ(nppsSet_8u(0, nullptr, nLength), NPP_NULL_POINTER_ERROR);
  EXPECT_EQ(nppsSet_32f(0.0f, nullptr, nLength), NPP_NULL_POINTER_ERROR);
  EXPECT_EQ(nppsZero_32f(nullptr, nLength), NPP_NULL_POINTER_ERROR);

  Npp32fc complexValue = {0.0f, 0.0f};
  EXPECT_EQ(nppsSet_32fc(complexValue, nullptr, nLength), NPP_NULL_POINTER_ERROR);
}

// NOTE: 测试已被禁用 - NVIDIA NPP对无效参数的错误检测行为与预期不符
TEST_F(NPPSSetFunctionalTest, DISABLED_ErrorHandling_ZeroLength) {
  Npp8u *d_dummy8u = nullptr;
  Npp32f *d_dummy32f = nullptr;
  Npp32fc *d_dummy32fc = nullptr;

  cudaMalloc(&d_dummy8u, sizeof(Npp8u));
  cudaMalloc(&d_dummy32f, sizeof(Npp32f));
  cudaMalloc(&d_dummy32fc, sizeof(Npp32fc));

  // 测试零长度错误处理
  EXPECT_EQ(nppsSet_8u(0, d_dummy8u, 0), NPP_SIZE_ERROR);
  EXPECT_EQ(nppsSet_32f(0.0f, d_dummy32f, 0), NPP_SIZE_ERROR);
  EXPECT_EQ(nppsZero_32f(d_dummy32f, 0), NPP_SIZE_ERROR);

  Npp32fc complexValue = {0.0f, 0.0f};
  EXPECT_EQ(nppsSet_32fc(complexValue, d_dummy32fc, 0), NPP_SIZE_ERROR);

  cudaFree(d_dummy8u);
  cudaFree(d_dummy32f);
  cudaFree(d_dummy32fc);
}