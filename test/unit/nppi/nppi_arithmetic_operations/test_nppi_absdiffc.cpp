#include "npp.h"
#include "npp_test_base.h"
#include <gtest/gtest.h>
#include <vector>

class NppiAbsDiffCTest : public ::testing::Test {
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
// AbsDiffC Tests
// ============================================================================

TEST_F(NppiAbsDiffCTest, AbsDiffC_8u_C1R_BasicOperation) {
  const int width = 4;
  const int height = 2;
  const int totalPixels = width * height;

  std::vector<Npp8u> hostSrc = {10, 50, 100, 200, 5, 150, 75, 255};
  Npp8u constant = 100;
  std::vector<Npp8u> expected = {90, 50, 0, 100, 95, 50, 25, 155}; // |src - 100|

  // Allocate GPU memory
  int step;
  Npp8u *d_src = nppiMalloc_8u_C1(width, height, &step);
  Npp8u *d_dst = nppiMalloc_8u_C1(width, height, &step);

  ASSERT_NE(d_src, nullptr);
  ASSERT_NE(d_dst, nullptr);

  // Copy data to GPU
  int hostStep = width * sizeof(Npp8u);
  cudaMemcpy2D(d_src, step, hostSrc.data(), hostStep, hostStep, height, cudaMemcpyHostToDevice);

  // Execute operation
  NppiSize oSizeROI = {width, height};
  NppStatus status = nppiAbsDiffC_8u_C1R(d_src, step, d_dst, step, oSizeROI, constant);
  ASSERT_EQ(status, NPP_SUCCESS);

  // Copy result back
  std::vector<Npp8u> hostResult(totalPixels);
  cudaMemcpy2D(hostResult.data(), hostStep, d_dst, step, hostStep, height, cudaMemcpyDeviceToHost);

  // Verify results
  for (int i = 0; i < totalPixels; ++i) {
    EXPECT_EQ(hostResult[i], expected[i]) << "Mismatch at index " << i;
  }

  // Cleanup
  nppiFree(d_src);
  nppiFree(d_dst);
}

TEST_F(NppiAbsDiffCTest, AbsDiffC_16u_C1R_DataTypes) {
  const int width = 3;
  const int height = 2;
  const int totalPixels = width * height;

  std::vector<Npp16u> hostSrc = {1000, 5000, 10000, 500, 15000, 2000};
  Npp16u constant = 5000;
  std::vector<Npp16u> expected = {4000, 0, 5000, 4500, 10000, 3000}; // |src - 5000|

  // Allocate GPU memory
  int step;
  Npp16u *d_src = nppiMalloc_16u_C1(width, height, &step);
  Npp16u *d_dst = nppiMalloc_16u_C1(width, height, &step);

  ASSERT_NE(d_src, nullptr);
  ASSERT_NE(d_dst, nullptr);

  // Copy data to GPU
  int hostStep = width * sizeof(Npp16u);
  cudaMemcpy2D(d_src, step, hostSrc.data(), hostStep, hostStep, height, cudaMemcpyHostToDevice);

  // Execute operation
  NppiSize oSizeROI = {width, height};
  NppStatus status = nppiAbsDiffC_16u_C1R(d_src, step, d_dst, step, oSizeROI, constant);
  ASSERT_EQ(status, NPP_SUCCESS);

  // Copy result back
  std::vector<Npp16u> hostResult(totalPixels);
  cudaMemcpy2D(hostResult.data(), hostStep, d_dst, step, hostStep, height, cudaMemcpyDeviceToHost);

  // Verify results
  for (int i = 0; i < totalPixels; ++i) {
    EXPECT_EQ(hostResult[i], expected[i]) << "Mismatch at index " << i;
  }

  // Cleanup
  nppiFree(d_src);
  nppiFree(d_dst);
}

TEST_F(NppiAbsDiffCTest, AbsDiffC_32f_C1R_FloatingPoint) {
  const int width = 2;
  const int height = 2;
  const int totalPixels = width * height;

  std::vector<Npp32f> hostSrc = {1.5f, 3.7f, 2.1f, 4.9f};
  Npp32f constant = 3.0f;
  std::vector<Npp32f> expected = {1.5f, 0.7f, 0.9f, 1.9f}; // |src - 3.0|

  // Allocate GPU memory
  int step;
  Npp32f *d_src = nppiMalloc_32f_C1(width, height, &step);
  Npp32f *d_dst = nppiMalloc_32f_C1(width, height, &step);

  ASSERT_NE(d_src, nullptr);
  ASSERT_NE(d_dst, nullptr);

  // Copy data to GPU
  int hostStep = width * sizeof(Npp32f);
  cudaMemcpy2D(d_src, step, hostSrc.data(), hostStep, hostStep, height, cudaMemcpyHostToDevice);

  // Execute operation
  NppiSize oSizeROI = {width, height};
  NppStatus status = nppiAbsDiffC_32f_C1R(d_src, step, d_dst, step, oSizeROI, constant);
  ASSERT_EQ(status, NPP_SUCCESS);

  // Copy result back
  std::vector<Npp32f> hostResult(totalPixels);
  cudaMemcpy2D(hostResult.data(), hostStep, d_dst, step, hostStep, height, cudaMemcpyDeviceToHost);

  // Verify results with tolerance for floating point
  for (int i = 0; i < totalPixels; ++i) {
    EXPECT_NEAR(hostResult[i], expected[i], 1e-5f) << "Mismatch at index " << i;
  }

  // Cleanup
  nppiFree(d_src);
  nppiFree(d_dst);
}