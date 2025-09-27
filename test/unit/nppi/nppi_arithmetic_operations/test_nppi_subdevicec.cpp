#include "npp.h"
#include "npp_test_base.h"
#include <gtest/gtest.h>
#include <vector>

class NppiSubDeviceCTest : public ::testing::Test {
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
// SubDeviceC 8u C1 Tests
// ============================================================================

TEST_F(NppiSubDeviceCTest, SubDeviceC_8u_C1RSfs_BasicOperation) {
  const int width = 4;
  const int height = 2;
  const int totalPixels = width * height;

  std::vector<Npp8u> hostSrc = {50, 60, 70, 80, 90, 100, 110, 120};
  Npp8u hostConstant = 25;
  std::vector<Npp8u> expected = {25, 35, 45, 55, 65, 75, 85, 95}; // src - 25

  // Allocate GPU memory for image data
  int step;
  Npp8u *d_src = nppiMalloc_8u_C1(width, height, &step);
  Npp8u *d_dst = nppiMalloc_8u_C1(width, height, &step);

  // Allocate GPU memory for device constant
  Npp8u *d_constant;
  cudaMalloc(&d_constant, sizeof(Npp8u));

  ASSERT_NE(d_src, nullptr);
  ASSERT_NE(d_dst, nullptr);
  ASSERT_NE(d_constant, nullptr);

  // Copy data to GPU
  int hostStep = width * sizeof(Npp8u);
  cudaMemcpy2D(d_src, step, hostSrc.data(), hostStep, hostStep, height, cudaMemcpyHostToDevice);
  cudaMemcpy(d_constant, &hostConstant, sizeof(Npp8u), cudaMemcpyHostToDevice);

  // Execute operation
  NppiSize oSizeROI = {width, height};
  NppStreamContext nppStreamCtx;
  nppGetStreamContext(&nppStreamCtx);
  NppStatus status = nppiSubDeviceC_8u_C1RSfs_Ctx(d_src, step, d_constant, d_dst, step, oSizeROI, 0, nppStreamCtx);
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
  cudaFree(d_constant);
}

TEST_F(NppiSubDeviceCTest, SubDeviceC_8u_C1RSfs_WithUnderflow) {
  const int width = 3;
  const int height = 2;
  const int totalPixels = width * height;

  std::vector<Npp8u> hostSrc = {20, 40, 80, 15, 50, 100};
  Npp8u hostConstant = 30;
  std::vector<Npp8u> expected = {0, 10, 50, 0, 20, 70}; // src - 30 with underflow protection

  // Allocate GPU memory
  int step;
  Npp8u *d_src = nppiMalloc_8u_C1(width, height, &step);
  Npp8u *d_dst = nppiMalloc_8u_C1(width, height, &step);
  Npp8u *d_constant;
  cudaMalloc(&d_constant, sizeof(Npp8u));

  ASSERT_NE(d_src, nullptr);
  ASSERT_NE(d_dst, nullptr);
  ASSERT_NE(d_constant, nullptr);

  // Copy data to GPU
  int hostStep = width * sizeof(Npp8u);
  cudaMemcpy2D(d_src, step, hostSrc.data(), hostStep, hostStep, height, cudaMemcpyHostToDevice);
  cudaMemcpy(d_constant, &hostConstant, sizeof(Npp8u), cudaMemcpyHostToDevice);

  // Execute operation
  NppiSize oSizeROI = {width, height};
  NppStreamContext nppStreamCtx;
  nppGetStreamContext(&nppStreamCtx);
  NppStatus status = nppiSubDeviceC_8u_C1RSfs_Ctx(d_src, step, d_constant, d_dst, step, oSizeROI, 0, nppStreamCtx);
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
  cudaFree(d_constant);
}

// ============================================================================
// SubDeviceC 8u C3 Tests
// ============================================================================

TEST_F(NppiSubDeviceCTest, SubDeviceC_8u_C3RSfs_MultiChannel) {
  const int width = 2;
  const int height = 2;
  const int channels = 3;
  const int totalElements = width * height * channels;

  // Test data: 2x2 image with 3 channels (RGB)
  std::vector<Npp8u> hostSrc = {// Pixel (0,0): R=50, G=60, B=70
                                50, 60, 70,
                                // Pixel (1,0): R=80, G=90, B=100
                                80, 90, 100,
                                // Pixel (0,1): R=110, G=120, B=130
                                110, 120, 130,
                                // Pixel (1,1): R=140, G=150, B=160
                                140, 150, 160};

  std::vector<Npp8u> hostConstants = {10, 20, 30}; // Subtract R-10, G-20, B-30
  int scaleFactor = 0;                             // No scaling

  // Expected results
  std::vector<Npp8u> expected = {// Pixel (0,0): R=50-10=40, G=60-20=40, B=70-30=40
                                 40, 40, 40,
                                 // Pixel (1,0): R=80-10=70, G=90-20=70, B=100-30=70
                                 70, 70, 70,
                                 // Pixel (0,1): R=110-10=100, G=120-20=100, B=130-30=100
                                 100, 100, 100,
                                 // Pixel (1,1): R=140-10=130, G=150-20=130, B=160-30=130
                                 130, 130, 130};

  // Allocate GPU memory
  int step;
  Npp8u *d_src = nppiMalloc_8u_C3(width, height, &step);
  Npp8u *d_dst = nppiMalloc_8u_C3(width, height, &step);
  Npp8u *d_constants;
  cudaMalloc(&d_constants, channels * sizeof(Npp8u));

  ASSERT_NE(d_src, nullptr);
  ASSERT_NE(d_dst, nullptr);
  ASSERT_NE(d_constants, nullptr);

  // Copy data to GPU
  int hostStep = width * channels * sizeof(Npp8u);
  cudaMemcpy2D(d_src, step, hostSrc.data(), hostStep, hostStep, height, cudaMemcpyHostToDevice);
  cudaMemcpy(d_constants, hostConstants.data(), channels * sizeof(Npp8u), cudaMemcpyHostToDevice);

  // Execute operation
  NppiSize oSizeROI = {width, height};
  NppStreamContext nppStreamCtx;
  nppGetStreamContext(&nppStreamCtx);
  NppStatus status =
      nppiSubDeviceC_8u_C3RSfs_Ctx(d_src, step, d_constants, d_dst, step, oSizeROI, scaleFactor, nppStreamCtx);
  ASSERT_EQ(status, NPP_SUCCESS);

  // Copy result back
  std::vector<Npp8u> hostResult(totalElements);
  cudaMemcpy2D(hostResult.data(), hostStep, d_dst, step, hostStep, height, cudaMemcpyDeviceToHost);

  // Verify results
  for (int i = 0; i < totalElements; ++i) {
    EXPECT_EQ(hostResult[i], expected[i]) << "Mismatch at index " << i;
  }

  // Cleanup
  nppiFree(d_src);
  nppiFree(d_dst);
  cudaFree(d_constants);
}

// ============================================================================
// SubDeviceC 32f Tests
// ============================================================================

TEST_F(NppiSubDeviceCTest, SubDeviceC_32f_C1R_FloatingPoint) {
  const int width = 2;
  const int height = 2;
  const int totalPixels = width * height;

  std::vector<Npp32f> hostSrc = {5.5f, 7.25f, 9.0f, 10.75f};
  Npp32f hostConstant = 2.25f;
  std::vector<Npp32f> expected = {3.25f, 5.0f, 6.75f, 8.5f}; // src - 2.25

  // Allocate GPU memory
  int step;
  Npp32f *d_src = nppiMalloc_32f_C1(width, height, &step);
  Npp32f *d_dst = nppiMalloc_32f_C1(width, height, &step);
  Npp32f *d_constant;
  cudaMalloc(&d_constant, sizeof(Npp32f));

  ASSERT_NE(d_src, nullptr);
  ASSERT_NE(d_dst, nullptr);
  ASSERT_NE(d_constant, nullptr);

  // Copy data to GPU
  int hostStep = width * sizeof(Npp32f);
  cudaMemcpy2D(d_src, step, hostSrc.data(), hostStep, hostStep, height, cudaMemcpyHostToDevice);
  cudaMemcpy(d_constant, &hostConstant, sizeof(Npp32f), cudaMemcpyHostToDevice);

  // Execute operation
  NppiSize oSizeROI = {width, height};
  NppStreamContext nppStreamCtx;
  nppGetStreamContext(&nppStreamCtx);
  NppStatus status = nppiSubDeviceC_32f_C1R_Ctx(d_src, step, d_constant, d_dst, step, oSizeROI, nppStreamCtx);
  ASSERT_EQ(status, NPP_SUCCESS);

  // Copy result back
  std::vector<Npp32f> hostResult(totalPixels);
  cudaMemcpy2D(hostResult.data(), hostStep, d_dst, step, hostStep, height, cudaMemcpyDeviceToHost);

  // Verify results with floating point tolerance
  for (int i = 0; i < totalPixels; ++i) {
    EXPECT_NEAR(hostResult[i], expected[i], 1e-5f) << "Mismatch at index " << i;
  }

  // Cleanup
  nppiFree(d_src);
  nppiFree(d_dst);
  cudaFree(d_constant);
}
