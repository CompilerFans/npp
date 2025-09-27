#include "npp.h"
#include "npp_test_base.h"
#include <gtest/gtest.h>
#include <vector>

#ifndef USE_NVIDIA_NPP_TESTS
extern "C" {
// AbsDiffDeviceC function declarations (MPP implementation only)
NppStatus nppiAbsDiffDeviceC_8u_C1RSfs_Ctx(const Npp8u *pSrc1, int nSrc1Step, const Npp8u *pConstant, Npp8u *pDst,
                                           int nDstStep, NppiSize oSizeROI, int nScaleFactor,
                                           NppStreamContext nppStreamCtx);

NppStatus nppiAbsDiffDeviceC_8u_C3RSfs_Ctx(const Npp8u *pSrc1, int nSrc1Step, const Npp8u *pConstants, Npp8u *pDst,
                                           int nDstStep, NppiSize oSizeROI, int nScaleFactor,
                                           NppStreamContext nppStreamCtx);

NppStatus nppiAbsDiffDeviceC_32f_C1R_Ctx(const Npp32f *pSrc1, int nSrc1Step, Npp32f *pDst, int nDstStep,
                                         NppiSize oSizeROI, Npp32f *pConstant, NppStreamContext nppStreamCtx);
}
#endif

#ifndef USE_NVIDIA_NPP_TESTS

class NppiAbsDiffDeviceCTest : public ::testing::Test {
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
// AbsDiffDeviceC 8u C1 Tests
// ============================================================================

TEST_F(NppiAbsDiffDeviceCTest, AbsDiffDeviceC_8u_C1RSfs_BasicOperation) {
  const int width = 4;
  const int height = 2;
  const int totalPixels = width * height;

  std::vector<Npp8u> hostSrc = {10, 20, 30, 40, 50, 60, 70, 80};
  Npp8u hostConstant = 35;
  std::vector<Npp8u> expected = {25, 15, 5, 5, 15, 25, 35, 45}; // |src - 35|

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
  NppStatus status = nppiAbsDiffDeviceC_8u_C1RSfs_Ctx(d_src, step, d_constant, d_dst, step, oSizeROI, 0, nppStreamCtx);
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

TEST_F(NppiAbsDiffDeviceCTest, AbsDiffDeviceC_8u_C1RSfs_WithScaling) {
  const int width = 3;
  const int height = 2;
  const int totalPixels = width * height;

  std::vector<Npp8u> hostSrc = {100, 150, 200, 50, 75, 125};
  Npp8u hostConstant = 100;
  int scaleFactor = 1;                                   // Divide by 2 after absolute difference
  std::vector<Npp8u> expected = {0, 25, 50, 25, 13, 13}; // |src - 100| / 2 (rounded)

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
  NppStatus status =
      nppiAbsDiffDeviceC_8u_C1RSfs_Ctx(d_src, step, d_constant, d_dst, step, oSizeROI, scaleFactor, nppStreamCtx);
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
// AbsDiffDeviceC 8u C3 Tests
// ============================================================================

TEST_F(NppiAbsDiffDeviceCTest, AbsDiffDeviceC_8u_C3RSfs_MultiChannel) {
  const int width = 2;
  const int height = 2;
  const int channels = 3;
  const int totalPixels = width * height * channels;

  std::vector<Npp8u> hostSrc = {
      100, 150, 200, // Pixel 1: R=100, G=150, B=200
      50,  75,  125, // Pixel 2: R=50,  G=75,  B=125
      80,  120, 160, // Pixel 3: R=80,  G=120, B=160
      40,  60,  100  // Pixel 4: R=40,  G=60,  B=100
  };

  std::vector<Npp8u> hostConstants = {100, 100, 150}; // Constants for R, G, B
  std::vector<Npp8u> expected = {
      0,  50, 50, // |100-100|, |150-100|, |200-150|
      50, 25, 25, // |50-100|,  |75-100|,  |125-150|
      20, 20, 10, // |80-100|,  |120-100|, |160-150|
      60, 40, 50  // |40-100|,  |60-100|,  |100-150|
  };

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
  NppStatus status = nppiAbsDiffDeviceC_8u_C3RSfs_Ctx(d_src, step, d_constants, d_dst, step, oSizeROI, 0, nppStreamCtx);
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
  cudaFree(d_constants);
}

// ============================================================================
// AbsDiffDeviceC 32f C1 Tests
// ============================================================================

TEST_F(NppiAbsDiffDeviceCTest, AbsDiffDeviceC_32f_C1R_FloatingPoint) {
  const int width = 3;
  const int height = 2;
  const int totalPixels = width * height;

  std::vector<Npp32f> hostSrc = {1.5f, 2.5f, 3.5f, 4.5f, 5.5f, 6.5f};
  Npp32f hostConstant = 3.0f;
  std::vector<Npp32f> expected = {1.5f, 0.5f, 0.5f, 1.5f, 2.5f, 3.5f}; // |src - 3.0|

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
  NppStatus status = nppiAbsDiffDeviceC_32f_C1R_Ctx(d_src, step, d_dst, step, oSizeROI, d_constant, nppStreamCtx);
  ASSERT_EQ(status, NPP_SUCCESS);

  // Copy result back
  std::vector<Npp32f> hostResult(totalPixels);
  cudaMemcpy2D(hostResult.data(), hostStep, d_dst, step, hostStep, height, cudaMemcpyDeviceToHost);

  // Verify results with tolerance
  for (int i = 0; i < totalPixels; ++i) {
    EXPECT_NEAR(hostResult[i], expected[i], 1e-6f) << "Mismatch at index " << i;
  }

  // Cleanup
  nppiFree(d_src);
  nppiFree(d_dst);
  cudaFree(d_constant);
}

#endif // USE_NVIDIA_NPP_TESTS