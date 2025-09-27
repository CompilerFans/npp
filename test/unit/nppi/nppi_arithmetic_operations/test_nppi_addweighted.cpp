#include "npp.h"
#include "npp_test_base.h"
#include <algorithm>
#include <gtest/gtest.h>
#include <vector>

class NppiAddWeightedTest : public ::testing::Test {
protected:
  void SetUp() override {
    cudaError_t err = cudaSetDevice(0);
    ASSERT_EQ(err, cudaSuccess);
  }

  void TearDown() override {
    cudaError_t err = cudaDeviceSynchronize();
    EXPECT_EQ(err, cudaSuccess);
  }
};

// ============================================================================
// AddWeighted Tests - In-place operations: result = src * alpha + dst * (1-alpha)
// ============================================================================

TEST_F(NppiAddWeightedTest, AddWeighted_8u32f_C1IR_BasicOperation) {
  const int width = 4;
  const int height = 2;
  const int totalPixels = width * height;

  std::vector<Npp8u> hostSrc = {100, 150, 200, 50, 75, 125, 175, 225};
  std::vector<Npp32f> hostSrcDst = {10.0f, 20.0f, 30.0f, 40.0f, 50.0f, 60.0f, 70.0f, 80.0f};

  Npp32f alpha = 0.3f; // Weight for source

  // Expected result: src * alpha + dst * (1 - alpha)
  std::vector<Npp32f> expected(totalPixels);
  for (int i = 0; i < totalPixels; ++i) {
    expected[i] = static_cast<float>(hostSrc[i]) * alpha + hostSrcDst[i] * (1.0f - alpha);
  }

  // Allocate GPU memory
  int srcStep, srcDstStep;
  Npp8u *d_src = nppiMalloc_8u_C1(width, height, &srcStep);
  Npp32f *d_srcDst = nppiMalloc_32f_C1(width, height, &srcDstStep);

  ASSERT_NE(d_src, nullptr);
  ASSERT_NE(d_srcDst, nullptr);

  // Copy data to GPU
  int hostSrcStep = width * sizeof(Npp8u);
  int hostSrcDstStep = width * sizeof(Npp32f);
  cudaMemcpy2D(d_src, srcStep, hostSrc.data(), hostSrcStep, hostSrcStep, height, cudaMemcpyHostToDevice);
  cudaMemcpy2D(d_srcDst, srcDstStep, hostSrcDst.data(), hostSrcDstStep, hostSrcDstStep, height, cudaMemcpyHostToDevice);

  // Execute operation
  NppiSize oSizeROI = {width, height};
  NppStatus status = nppiAddWeighted_8u32f_C1IR(d_src, srcStep, d_srcDst, srcDstStep, oSizeROI, alpha);
  ASSERT_EQ(status, NPP_SUCCESS);

  // Copy result back
  std::vector<Npp32f> hostResult(totalPixels);
  cudaMemcpy2D(hostResult.data(), hostSrcDstStep, d_srcDst, srcDstStep, hostSrcDstStep, height, cudaMemcpyDeviceToHost);

  // Verify results
  for (int i = 0; i < totalPixels; ++i) {
    EXPECT_NEAR(hostResult[i], expected[i], 1e-5f)
        << "Mismatch at index " << i << " got " << hostResult[i] << " expected " << expected[i];
  }

  // Cleanup
  nppiFree(d_src);
  nppiFree(d_srcDst);
}

TEST_F(NppiAddWeightedTest, AddWeighted_16u32f_C1IR_BasicOperation) {
  const int width = 3;
  const int height = 2;
  const int totalPixels = width * height;

  std::vector<Npp16u> hostSrc = {1000, 2000, 3000, 4000, 5000, 6000};
  std::vector<Npp32f> hostSrcDst = {100.0f, 200.0f, 300.0f, 400.0f, 500.0f, 600.0f};

  Npp32f alpha = 0.7f; // Weight for source

  // Expected result: src * alpha + dst * (1 - alpha)
  std::vector<Npp32f> expected(totalPixels);
  for (int i = 0; i < totalPixels; ++i) {
    expected[i] = static_cast<float>(hostSrc[i]) * alpha + hostSrcDst[i] * (1.0f - alpha);
  }

  // Allocate GPU memory
  int srcStep, srcDstStep;
  Npp16u *d_src = nppiMalloc_16u_C1(width, height, &srcStep);
  Npp32f *d_srcDst = nppiMalloc_32f_C1(width, height, &srcDstStep);

  ASSERT_NE(d_src, nullptr);
  ASSERT_NE(d_srcDst, nullptr);

  // Copy data to GPU
  int hostSrcStep = width * sizeof(Npp16u);
  int hostSrcDstStep = width * sizeof(Npp32f);
  cudaMemcpy2D(d_src, srcStep, hostSrc.data(), hostSrcStep, hostSrcStep, height, cudaMemcpyHostToDevice);
  cudaMemcpy2D(d_srcDst, srcDstStep, hostSrcDst.data(), hostSrcDstStep, hostSrcDstStep, height, cudaMemcpyHostToDevice);

  // Execute operation
  NppiSize oSizeROI = {width, height};
  NppStatus status = nppiAddWeighted_16u32f_C1IR(d_src, srcStep, d_srcDst, srcDstStep, oSizeROI, alpha);
  ASSERT_EQ(status, NPP_SUCCESS);

  // Copy result back
  std::vector<Npp32f> hostResult(totalPixels);
  cudaMemcpy2D(hostResult.data(), hostSrcDstStep, d_srcDst, srcDstStep, hostSrcDstStep, height, cudaMemcpyDeviceToHost);

  // Verify results
  for (int i = 0; i < totalPixels; ++i) {
    EXPECT_NEAR(hostResult[i], expected[i], 1e-3f)
        << "Mismatch at index " << i << " got " << hostResult[i] << " expected " << expected[i];
  }

  // Cleanup
  nppiFree(d_src);
  nppiFree(d_srcDst);
}

TEST_F(NppiAddWeightedTest, AddWeighted_32f_C1IR_FloatingPoint) {
  const int width = 3;
  const int height = 2;
  const int totalPixels = width * height;

  std::vector<Npp32f> hostSrc = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  std::vector<Npp32f> hostSrcDst = {0.5f, 1.5f, 2.5f, 3.5f, 4.5f, 5.5f};

  Npp32f alpha = 0.4f; // Weight for source

  // Expected result: src * alpha + dst * (1 - alpha)
  std::vector<Npp32f> expected(totalPixels);
  for (int i = 0; i < totalPixels; ++i) {
    expected[i] = hostSrc[i] * alpha + hostSrcDst[i] * (1.0f - alpha);
  }

  // Allocate GPU memory
  int srcStep, srcDstStep;
  Npp32f *d_src = nppiMalloc_32f_C1(width, height, &srcStep);
  Npp32f *d_srcDst = nppiMalloc_32f_C1(width, height, &srcDstStep);

  ASSERT_NE(d_src, nullptr);
  ASSERT_NE(d_srcDst, nullptr);

  // Copy data to GPU
  int hostStep = width * sizeof(Npp32f);
  cudaMemcpy2D(d_src, srcStep, hostSrc.data(), hostStep, hostStep, height, cudaMemcpyHostToDevice);
  cudaMemcpy2D(d_srcDst, srcDstStep, hostSrcDst.data(), hostStep, hostStep, height, cudaMemcpyHostToDevice);

  // Execute operation
  NppiSize oSizeROI = {width, height};
  NppStatus status = nppiAddWeighted_32f_C1IR(d_src, srcStep, d_srcDst, srcDstStep, oSizeROI, alpha);
  ASSERT_EQ(status, NPP_SUCCESS);

  // Copy result back
  std::vector<Npp32f> hostResult(totalPixels);
  cudaMemcpy2D(hostResult.data(), hostStep, d_srcDst, srcDstStep, hostStep, height, cudaMemcpyDeviceToHost);

  // Verify results
  for (int i = 0; i < totalPixels; ++i) {
    EXPECT_NEAR(hostResult[i], expected[i], 1e-6f)
        << "Mismatch at index " << i << " got " << hostResult[i] << " expected " << expected[i];
  }

  // Cleanup
  nppiFree(d_src);
  nppiFree(d_srcDst);
}

TEST_F(NppiAddWeightedTest, AddWeighted_Alpha_EdgeCases) {
  const int width = 2;
  const int height = 1;
  const int totalPixels = width * height;

  std::vector<Npp32f> hostSrc = {100.0f, 200.0f};
  std::vector<Npp32f> hostSrcDst = {50.0f, 150.0f};

  // Test alpha = 0.0 (should return dst unchanged)
  {
    Npp32f alpha = 0.0f;
    std::vector<Npp32f> expected = hostSrcDst; // dst * 1.0

    int srcStep, srcDstStep;
    Npp32f *d_src = nppiMalloc_32f_C1(width, height, &srcStep);
    Npp32f *d_srcDst = nppiMalloc_32f_C1(width, height, &srcDstStep);

    int hostStep = width * sizeof(Npp32f);
    cudaMemcpy2D(d_src, srcStep, hostSrc.data(), hostStep, hostStep, height, cudaMemcpyHostToDevice);
    cudaMemcpy2D(d_srcDst, srcDstStep, hostSrcDst.data(), hostStep, hostStep, height, cudaMemcpyHostToDevice);

    NppiSize oSizeROI = {width, height};
    NppStatus status = nppiAddWeighted_32f_C1IR(d_src, srcStep, d_srcDst, srcDstStep, oSizeROI, alpha);
    ASSERT_EQ(status, NPP_SUCCESS);

    std::vector<Npp32f> hostResult(totalPixels);
    cudaMemcpy2D(hostResult.data(), hostStep, d_srcDst, srcDstStep, hostStep, height, cudaMemcpyDeviceToHost);

    for (int i = 0; i < totalPixels; ++i) {
      EXPECT_NEAR(hostResult[i], expected[i], 1e-6f) << "Alpha=0.0 failed at index " << i;
    }

    nppiFree(d_src);
    nppiFree(d_srcDst);
  }

  // Test alpha = 1.0 (should return src)
  {
    Npp32f alpha = 1.0f;
    std::vector<Npp32f> expected = hostSrc; // src * 1.0

    int srcStep, srcDstStep;
    Npp32f *d_src = nppiMalloc_32f_C1(width, height, &srcStep);
    Npp32f *d_srcDst = nppiMalloc_32f_C1(width, height, &srcDstStep);

    int hostStep = width * sizeof(Npp32f);
    cudaMemcpy2D(d_src, srcStep, hostSrc.data(), hostStep, hostStep, height, cudaMemcpyHostToDevice);
    cudaMemcpy2D(d_srcDst, srcDstStep, hostSrcDst.data(), hostStep, hostStep, height, cudaMemcpyHostToDevice);

    NppiSize oSizeROI = {width, height};
    NppStatus status = nppiAddWeighted_32f_C1IR(d_src, srcStep, d_srcDst, srcDstStep, oSizeROI, alpha);
    ASSERT_EQ(status, NPP_SUCCESS);

    std::vector<Npp32f> hostResult(totalPixels);
    cudaMemcpy2D(hostResult.data(), hostStep, d_srcDst, srcDstStep, hostStep, height, cudaMemcpyDeviceToHost);

    for (int i = 0; i < totalPixels; ++i) {
      EXPECT_NEAR(hostResult[i], expected[i], 1e-6f) << "Alpha=1.0 failed at index " << i;
    }

    nppiFree(d_src);
    nppiFree(d_srcDst);
  }
}