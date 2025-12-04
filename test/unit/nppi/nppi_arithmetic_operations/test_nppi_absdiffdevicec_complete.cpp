#include <gtest/gtest.h>
#include <npp.h>
#include <cuda_runtime.h>
#include <vector>
#include <cmath>

// Test for nppiAbsDiffDeviceC_8u_C1R_Ctx, nppiAbsDiffDeviceC_16u_C1R_Ctx, nppiAbsDiffDeviceC_32f_C1R_Ctx

struct AbsDiffDeviceCParam {
  std::string name;
  int dataType;  // 0=8u, 1=16u, 2=32f
};

// ============================================================================
// 8u Tests
// ============================================================================

class AbsDiffDeviceC_8u_Test : public ::testing::Test {
protected:
  void SetUp() override { cudaSetDevice(0); }
  void TearDown() override { cudaDeviceSynchronize(); }
};

TEST_F(AbsDiffDeviceC_8u_Test, C1R_BasicOperation) {
  const int width = 4;
  const int height = 3;
  const int totalPixels = width * height;

  std::vector<Npp8u> hostSrc(totalPixels);
  for (int i = 0; i < totalPixels; ++i) {
    hostSrc[i] = static_cast<Npp8u>(50 + (i % 100));
  }
  Npp8u hostConstant = 75;

  int step;
  Npp8u* d_src = nppiMalloc_8u_C1(width, height, &step);
  Npp8u* d_dst = nppiMalloc_8u_C1(width, height, &step);
  Npp8u* d_constant;
  cudaMalloc(&d_constant, sizeof(Npp8u));

  ASSERT_NE(d_src, nullptr);
  ASSERT_NE(d_dst, nullptr);
  ASSERT_NE(d_constant, nullptr);

  int hostStep = width * sizeof(Npp8u);
  cudaMemcpy2D(d_src, step, hostSrc.data(), hostStep, hostStep, height, cudaMemcpyHostToDevice);
  cudaMemcpy(d_constant, &hostConstant, sizeof(Npp8u), cudaMemcpyHostToDevice);

  NppiSize oSizeROI = {width, height};
  NppStreamContext ctx;
  nppGetStreamContext(&ctx);
  NppStatus status = nppiAbsDiffDeviceC_8u_C1R_Ctx(d_src, step, d_dst, step, oSizeROI, d_constant, ctx);
  ASSERT_EQ(status, NPP_SUCCESS);

  std::vector<Npp8u> hostResult(totalPixels);
  cudaMemcpy2D(hostResult.data(), hostStep, d_dst, step, hostStep, height, cudaMemcpyDeviceToHost);

  for (int i = 0; i < totalPixels; ++i) {
    int expected = std::abs(static_cast<int>(hostSrc[i]) - static_cast<int>(hostConstant));
    EXPECT_EQ(hostResult[i], static_cast<Npp8u>(expected)) << "Mismatch at index " << i;
  }

  nppiFree(d_src);
  nppiFree(d_dst);
  cudaFree(d_constant);
}

// ============================================================================
// 16u Tests
// ============================================================================

class AbsDiffDeviceC_16u_Test : public ::testing::Test {
protected:
  void SetUp() override { cudaSetDevice(0); }
  void TearDown() override { cudaDeviceSynchronize(); }
};

TEST_F(AbsDiffDeviceC_16u_Test, C1R_BasicOperation) {
  const int width = 4;
  const int height = 3;
  const int totalPixels = width * height;

  std::vector<Npp16u> hostSrc(totalPixels);
  for (int i = 0; i < totalPixels; ++i) {
    hostSrc[i] = static_cast<Npp16u>(500 + (i % 1000));
  }
  Npp16u hostConstant = 750;

  int step;
  Npp16u* d_src = nppiMalloc_16u_C1(width, height, &step);
  Npp16u* d_dst = nppiMalloc_16u_C1(width, height, &step);
  Npp16u* d_constant;
  cudaMalloc(&d_constant, sizeof(Npp16u));

  ASSERT_NE(d_src, nullptr);
  ASSERT_NE(d_dst, nullptr);
  ASSERT_NE(d_constant, nullptr);

  int hostStep = width * sizeof(Npp16u);
  cudaMemcpy2D(d_src, step, hostSrc.data(), hostStep, hostStep, height, cudaMemcpyHostToDevice);
  cudaMemcpy(d_constant, &hostConstant, sizeof(Npp16u), cudaMemcpyHostToDevice);

  NppiSize oSizeROI = {width, height};
  NppStreamContext ctx;
  nppGetStreamContext(&ctx);
  NppStatus status = nppiAbsDiffDeviceC_16u_C1R_Ctx(d_src, step, d_dst, step, oSizeROI, d_constant, ctx);
  ASSERT_EQ(status, NPP_SUCCESS);

  std::vector<Npp16u> hostResult(totalPixels);
  cudaMemcpy2D(hostResult.data(), hostStep, d_dst, step, hostStep, height, cudaMemcpyDeviceToHost);

  for (int i = 0; i < totalPixels; ++i) {
    int expected = std::abs(static_cast<int>(hostSrc[i]) - static_cast<int>(hostConstant));
    EXPECT_EQ(hostResult[i], static_cast<Npp16u>(expected)) << "Mismatch at index " << i;
  }

  nppiFree(d_src);
  nppiFree(d_dst);
  cudaFree(d_constant);
}

// ============================================================================
// 32f Tests
// ============================================================================

class AbsDiffDeviceC_32f_Test : public ::testing::Test {
protected:
  void SetUp() override { cudaSetDevice(0); }
  void TearDown() override { cudaDeviceSynchronize(); }
};

TEST_F(AbsDiffDeviceC_32f_Test, C1R_BasicOperation) {
  const int width = 4;
  const int height = 3;
  const int totalPixels = width * height;

  std::vector<Npp32f> hostSrc(totalPixels);
  for (int i = 0; i < totalPixels; ++i) {
    hostSrc[i] = 1.0f + (i % 10) * 0.5f;
  }
  Npp32f hostConstant = 3.5f;

  int step;
  Npp32f* d_src = nppiMalloc_32f_C1(width, height, &step);
  Npp32f* d_dst = nppiMalloc_32f_C1(width, height, &step);
  Npp32f* d_constant;
  cudaMalloc(&d_constant, sizeof(Npp32f));

  ASSERT_NE(d_src, nullptr);
  ASSERT_NE(d_dst, nullptr);
  ASSERT_NE(d_constant, nullptr);

  int hostStep = width * sizeof(Npp32f);
  cudaMemcpy2D(d_src, step, hostSrc.data(), hostStep, hostStep, height, cudaMemcpyHostToDevice);
  cudaMemcpy(d_constant, &hostConstant, sizeof(Npp32f), cudaMemcpyHostToDevice);

  NppiSize oSizeROI = {width, height};
  NppStreamContext ctx;
  nppGetStreamContext(&ctx);
  NppStatus status = nppiAbsDiffDeviceC_32f_C1R_Ctx(d_src, step, d_dst, step, oSizeROI, d_constant, ctx);
  ASSERT_EQ(status, NPP_SUCCESS);

  std::vector<Npp32f> hostResult(totalPixels);
  cudaMemcpy2D(hostResult.data(), hostStep, d_dst, step, hostStep, height, cudaMemcpyDeviceToHost);

  for (int i = 0; i < totalPixels; ++i) {
    float expected = std::fabs(hostSrc[i] - hostConstant);
    EXPECT_NEAR(hostResult[i], expected, 1e-5f) << "Mismatch at index " << i;
  }

  nppiFree(d_src);
  nppiFree(d_dst);
  cudaFree(d_constant);
}
