#include "npp.h"
#include "npp_test_base.h"

#include <algorithm>
#include <cmath>
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <random>
#include <vector>

class NppiMeanStdDevMaskTest : public ::testing::Test {
protected:
  void SetUp() override {
    cudaError_t err = cudaSetDevice(0);
    ASSERT_EQ(err, cudaSuccess);
  }

  void TearDown() override {
    cudaError_t err = cudaDeviceSynchronize();
    EXPECT_EQ(err, cudaSuccess);
  }

  // Calculate reference mean and standard deviation with mask
  template <typename T>
  void calculateReferenceMeanStdDevWithMask(const std::vector<T> &data, const std::vector<Npp8u> &mask, double &mean,
                                            double &stddev) {
    double sum = 0.0;
    int validCount = 0;

    for (size_t i = 0; i < data.size(); ++i) {
      if (mask[i] > 0) {
        sum += static_cast<double>(data[i]);
        validCount++;
      }
    }

    if (validCount == 0) {
      mean = 0.0;
      stddev = 0.0;
      return;
    }

    mean = sum / validCount;

    double sumSquaredDiff = 0.0;
    for (size_t i = 0; i < data.size(); ++i) {
      if (mask[i] > 0) {
        double diff = static_cast<double>(data[i]) - mean;
        sumSquaredDiff += diff * diff;
      }
    }
    stddev = std::sqrt(sumSquaredDiff / validCount);
  }

  // Generate various mask patterns
  std::vector<Npp8u> generateFullMask(int width, int height) { return std::vector<Npp8u>(width * height, 255); }

  std::vector<Npp8u> generateCheckerboardMask(int width, int height, int blockSize = 1) {
    std::vector<Npp8u> mask(width * height);
    for (int y = 0; y < height; y++) {
      for (int x = 0; x < width; x++) {
        bool enabled = ((x / blockSize) + (y / blockSize)) % 2 == 0;
        mask[y * width + x] = enabled ? 255 : 0;
      }
    }
    return mask;
  }

  std::vector<Npp8u> generateCircularMask(int width, int height) {
    std::vector<Npp8u> mask(width * height);
    int centerX = width / 2;
    int centerY = height / 2;
    int radius = std::min(width, height) / 3;

    for (int y = 0; y < height; y++) {
      for (int x = 0; x < width; x++) {
        int dx = x - centerX;
        int dy = y - centerY;
        bool inside = (dx * dx + dy * dy) <= (radius * radius);
        mask[y * width + x] = inside ? 255 : 0;
      }
    }
    return mask;
  }
};

// Test 1: Buffer size calculation for 8u_C1MR
TEST_F(NppiMeanStdDevMaskTest, MeanStdDevGetBufferHostSize_8u_C1MR) {
  NppiSize oSizeROI = {256, 256};
  SIZE_TYPE bufferSize = 0;

  NppStatus status = nppiMeanStdDevGetBufferHostSize_8u_C1MR(oSizeROI, &bufferSize);
  ASSERT_EQ(status, NPP_SUCCESS);
  EXPECT_GT(bufferSize, 0);

  // Test with context version
  cudaStream_t stream;
  cudaStreamCreate(&stream);
  NppStreamContext nppStreamCtx;
  nppGetStreamContext(&nppStreamCtx);
  nppStreamCtx.hStream = stream;

  SIZE_TYPE bufferSizeCtx = 0;
  status = nppiMeanStdDevGetBufferHostSize_8u_C1MR_Ctx(oSizeROI, &bufferSizeCtx, nppStreamCtx);
  ASSERT_EQ(status, NPP_SUCCESS);
  EXPECT_GT(bufferSizeCtx, 0);
  EXPECT_EQ(bufferSize, bufferSizeCtx);

  cudaStreamDestroy(stream);
}

// Test 2: Buffer size calculation for 32f_C1MR
TEST_F(NppiMeanStdDevMaskTest, MeanStdDevGetBufferHostSize_32f_C1MR) {
  NppiSize oSizeROI = {512, 512};
  SIZE_TYPE bufferSize = 0;

  NppStatus status = nppiMeanStdDevGetBufferHostSize_32f_C1MR(oSizeROI, &bufferSize);
  ASSERT_EQ(status, NPP_SUCCESS);
  EXPECT_GT(bufferSize, 0);

  // Test with context version
  cudaStream_t stream;
  cudaStreamCreate(&stream);
  NppStreamContext nppStreamCtx;
  nppGetStreamContext(&nppStreamCtx);
  nppStreamCtx.hStream = stream;

  SIZE_TYPE bufferSizeCtx = 0;
  status = nppiMeanStdDevGetBufferHostSize_32f_C1MR_Ctx(oSizeROI, &bufferSizeCtx, nppStreamCtx);
  ASSERT_EQ(status, NPP_SUCCESS);
  EXPECT_GT(bufferSizeCtx, 0);
  EXPECT_EQ(bufferSize, bufferSizeCtx);

  cudaStreamDestroy(stream);
}

// Test 3: Mean and StdDev calculation for 8u_C1MR with full mask
TEST_F(NppiMeanStdDevMaskTest, Mean_StdDev_8u_C1MR_FullMask) {
  const int width = 128;
  const int height = 128;
  const Npp8u constantValue = 100;

  // Create test data and mask
  std::vector<Npp8u> hostSrc(width * height, constantValue);
  std::vector<Npp8u> hostMask = generateFullMask(width, height);

  // Calculate reference values
  double refMean, refStdDev;
  calculateReferenceMeanStdDevWithMask(hostSrc, hostMask, refMean, refStdDev);

  // Allocate GPU memory
  Npp8u *d_src = nullptr;
  Npp8u *d_mask = nullptr;
  int srcStep, maskStep;

  d_src = nppiMalloc_8u_C1(width, height, &srcStep);
  d_mask = nppiMalloc_8u_C1(width, height, &maskStep);
  ASSERT_NE(d_src, nullptr);
  ASSERT_NE(d_mask, nullptr);

  // Upload data
  cudaMemcpy2D(d_src, srcStep, hostSrc.data(), width, width, height, cudaMemcpyHostToDevice);
  cudaMemcpy2D(d_mask, maskStep, hostMask.data(), width, width, height, cudaMemcpyHostToDevice);

  // Get buffer size and allocate buffer
  NppiSize oSizeROI = {width, height};
  SIZE_TYPE bufferSize = 0;
  NppStatus status = nppiMeanStdDevGetBufferHostSize_8u_C1MR(oSizeROI, &bufferSize);
  ASSERT_EQ(status, NPP_SUCCESS);

  Npp8u *pDeviceBuffer = nullptr;
  cudaMalloc(&pDeviceBuffer, bufferSize);
  ASSERT_NE(pDeviceBuffer, nullptr);

  // Allocate device memory for results
  Npp64f *pMean = nullptr;
  Npp64f *pStdDev = nullptr;
  cudaMalloc(&pMean, sizeof(Npp64f));
  cudaMalloc(&pStdDev, sizeof(Npp64f));
  ASSERT_NE(pMean, nullptr);
  ASSERT_NE(pStdDev, nullptr);

  // Calculate mean and standard deviation
  status = nppiMean_StdDev_8u_C1MR(d_src, srcStep, d_mask, maskStep, oSizeROI, pDeviceBuffer, pMean, pStdDev);
  ASSERT_EQ(status, NPP_SUCCESS);

  // Copy results to host and verify
  Npp64f hostMean, hostStdDev;
  cudaMemcpy(&hostMean, pMean, sizeof(Npp64f), cudaMemcpyDeviceToHost);
  cudaMemcpy(&hostStdDev, pStdDev, sizeof(Npp64f), cudaMemcpyDeviceToHost);

  EXPECT_NEAR(hostMean, refMean, 0.001);
  EXPECT_NEAR(hostStdDev, refStdDev, 0.001);

  // Cleanup
  nppiFree(d_src);
  nppiFree(d_mask);
  cudaFree(pDeviceBuffer);
  cudaFree(pMean);
  cudaFree(pStdDev);
}

// Test 4: Mean and StdDev calculation for 8u_C1MR with checkerboard mask
TEST_F(NppiMeanStdDevMaskTest, Mean_StdDev_8u_C1MR_CheckerboardMask) {
  const int width = 64;
  const int height = 64;

  // Create gradient data
  std::vector<Npp8u> hostSrc(width * height);
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      hostSrc[y * width + x] = static_cast<Npp8u>((x + y) % 256);
    }
  }

  // Create checkerboard mask
  std::vector<Npp8u> hostMask = generateCheckerboardMask(width, height, 2);

  // Calculate reference values
  double refMean, refStdDev;
  calculateReferenceMeanStdDevWithMask(hostSrc, hostMask, refMean, refStdDev);

  // Allocate GPU memory
  Npp8u *d_src = nullptr;
  Npp8u *d_mask = nullptr;
  int srcStep, maskStep;

  d_src = nppiMalloc_8u_C1(width, height, &srcStep);
  d_mask = nppiMalloc_8u_C1(width, height, &maskStep);
  ASSERT_NE(d_src, nullptr);
  ASSERT_NE(d_mask, nullptr);

  // Upload data
  cudaMemcpy2D(d_src, srcStep, hostSrc.data(), width, width, height, cudaMemcpyHostToDevice);
  cudaMemcpy2D(d_mask, maskStep, hostMask.data(), width, width, height, cudaMemcpyHostToDevice);

  // Get buffer size and allocate buffer
  NppiSize oSizeROI = {width, height};
  SIZE_TYPE bufferSize = 0;
  NppStatus status = nppiMeanStdDevGetBufferHostSize_8u_C1MR(oSizeROI, &bufferSize);
  ASSERT_EQ(status, NPP_SUCCESS);

  Npp8u *pDeviceBuffer = nullptr;
  cudaMalloc(&pDeviceBuffer, bufferSize);
  ASSERT_NE(pDeviceBuffer, nullptr);

  // Allocate device memory for results
  Npp64f *pMean = nullptr;
  Npp64f *pStdDev = nullptr;
  cudaMalloc(&pMean, sizeof(Npp64f));
  cudaMalloc(&pStdDev, sizeof(Npp64f));
  ASSERT_NE(pMean, nullptr);
  ASSERT_NE(pStdDev, nullptr);

  // Calculate mean and standard deviation
  status = nppiMean_StdDev_8u_C1MR(d_src, srcStep, d_mask, maskStep, oSizeROI, pDeviceBuffer, pMean, pStdDev);
  ASSERT_EQ(status, NPP_SUCCESS);

  // Copy results to host and verify
  Npp64f hostMean, hostStdDev;
  cudaMemcpy(&hostMean, pMean, sizeof(Npp64f), cudaMemcpyDeviceToHost);
  cudaMemcpy(&hostStdDev, pStdDev, sizeof(Npp64f), cudaMemcpyDeviceToHost);

  EXPECT_NEAR(hostMean, refMean, 1.0); // Allow some tolerance for complex patterns
  EXPECT_NEAR(hostStdDev, refStdDev, 1.0);

  // Cleanup
  nppiFree(d_src);
  nppiFree(d_mask);
  cudaFree(pDeviceBuffer);
  cudaFree(pMean);
  cudaFree(pStdDev);
}

// Test 5: Mean and StdDev calculation for 32f_C1MR with circular mask
TEST_F(NppiMeanStdDevMaskTest, Mean_StdDev_32f_C1MR_CircularMask) {
  const int width = 128;
  const int height = 128;

  // Create gradient data
  std::vector<Npp32f> hostSrc(width * height);
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      hostSrc[y * width + x] = static_cast<float>(x * 0.1f + y * 0.2f);
    }
  }

  // Create circular mask
  std::vector<Npp8u> hostMask = generateCircularMask(width, height);

  // Calculate reference values
  double refMean, refStdDev;
  calculateReferenceMeanStdDevWithMask(hostSrc, hostMask, refMean, refStdDev);

  // Allocate GPU memory
  Npp32f *d_src = nullptr;
  Npp8u *d_mask = nullptr;
  int srcStep, maskStep;

  d_src = nppiMalloc_32f_C1(width, height, &srcStep);
  d_mask = nppiMalloc_8u_C1(width, height, &maskStep);
  ASSERT_NE(d_src, nullptr);
  ASSERT_NE(d_mask, nullptr);

  // Upload data
  cudaMemcpy2D(d_src, srcStep, hostSrc.data(), width * sizeof(Npp32f), width * sizeof(Npp32f), height,
               cudaMemcpyHostToDevice);
  cudaMemcpy2D(d_mask, maskStep, hostMask.data(), width, width, height, cudaMemcpyHostToDevice);

  // Get buffer size and allocate buffer
  NppiSize oSizeROI = {width, height};
  SIZE_TYPE bufferSize = 0;
  NppStatus status = nppiMeanStdDevGetBufferHostSize_32f_C1MR(oSizeROI, &bufferSize);
  ASSERT_EQ(status, NPP_SUCCESS);

  Npp8u *pDeviceBuffer = nullptr;
  cudaMalloc(&pDeviceBuffer, bufferSize);
  ASSERT_NE(pDeviceBuffer, nullptr);

  // Allocate device memory for results
  Npp64f *pMean = nullptr;
  Npp64f *pStdDev = nullptr;
  cudaMalloc(&pMean, sizeof(Npp64f));
  cudaMalloc(&pStdDev, sizeof(Npp64f));
  ASSERT_NE(pMean, nullptr);
  ASSERT_NE(pStdDev, nullptr);

  // Calculate mean and standard deviation
  status = nppiMean_StdDev_32f_C1MR(d_src, srcStep, d_mask, maskStep, oSizeROI, pDeviceBuffer, pMean, pStdDev);
  ASSERT_EQ(status, NPP_SUCCESS);

  // Copy results to host and verify
  Npp64f hostMean, hostStdDev;
  cudaMemcpy(&hostMean, pMean, sizeof(Npp64f), cudaMemcpyDeviceToHost);
  cudaMemcpy(&hostStdDev, pStdDev, sizeof(Npp64f), cudaMemcpyDeviceToHost);

  EXPECT_NEAR(hostMean, refMean, 0.1);
  EXPECT_NEAR(hostStdDev, refStdDev, 0.1);

  // Cleanup
  nppiFree(d_src);
  nppiFree(d_mask);
  cudaFree(pDeviceBuffer);
  cudaFree(pMean);
  cudaFree(pStdDev);
}

// Test 6: Context version for 8u_C1MR
TEST_F(NppiMeanStdDevMaskTest, Mean_StdDev_8u_C1MR_Ctx) {
  const int width = 64;
  const int height = 64;

  // Create test data
  std::vector<Npp8u> hostSrc(width * height, 150);
  std::vector<Npp8u> hostMask = generateCheckerboardMask(width, height, 1);

  // Calculate reference values
  double refMean, refStdDev;
  calculateReferenceMeanStdDevWithMask(hostSrc, hostMask, refMean, refStdDev);

  // Create stream
  cudaStream_t stream;
  cudaStreamCreate(&stream);
  NppStreamContext nppStreamCtx;
  nppGetStreamContext(&nppStreamCtx);
  nppStreamCtx.hStream = stream;

  // Allocate GPU memory
  Npp8u *d_src = nullptr;
  Npp8u *d_mask = nullptr;
  int srcStep, maskStep;

  d_src = nppiMalloc_8u_C1(width, height, &srcStep);
  d_mask = nppiMalloc_8u_C1(width, height, &maskStep);
  ASSERT_NE(d_src, nullptr);
  ASSERT_NE(d_mask, nullptr);

  // Upload data asynchronously
  cudaMemcpy2DAsync(d_src, srcStep, hostSrc.data(), width, width, height, cudaMemcpyHostToDevice, stream);
  cudaMemcpy2DAsync(d_mask, maskStep, hostMask.data(), width, width, height, cudaMemcpyHostToDevice, stream);

  // Get buffer size and allocate buffer
  NppiSize oSizeROI = {width, height};
  SIZE_TYPE bufferSize = 0;
  NppStatus status = nppiMeanStdDevGetBufferHostSize_8u_C1MR_Ctx(oSizeROI, &bufferSize, nppStreamCtx);
  ASSERT_EQ(status, NPP_SUCCESS);

  Npp8u *pDeviceBuffer = nullptr;
  cudaMalloc(&pDeviceBuffer, bufferSize);
  ASSERT_NE(pDeviceBuffer, nullptr);

  // Allocate device memory for results
  Npp64f *pMean = nullptr;
  Npp64f *pStdDev = nullptr;
  cudaMalloc(&pMean, sizeof(Npp64f));
  cudaMalloc(&pStdDev, sizeof(Npp64f));
  ASSERT_NE(pMean, nullptr);
  ASSERT_NE(pStdDev, nullptr);

  // Calculate mean and standard deviation with stream context
  status = nppiMean_StdDev_8u_C1MR_Ctx(d_src, srcStep, d_mask, maskStep, oSizeROI, pDeviceBuffer, pMean, pStdDev,
                                       nppStreamCtx);
  ASSERT_EQ(status, NPP_SUCCESS);

  // Synchronize stream and get results
  cudaStreamSynchronize(stream);

  Npp64f hostMean, hostStdDev;
  cudaMemcpy(&hostMean, pMean, sizeof(Npp64f), cudaMemcpyDeviceToHost);
  cudaMemcpy(&hostStdDev, pStdDev, sizeof(Npp64f), cudaMemcpyDeviceToHost);

  EXPECT_NEAR(hostMean, refMean, 0.001);
  EXPECT_NEAR(hostStdDev, refStdDev, 0.001);

  // Cleanup
  cudaStreamDestroy(stream);
  nppiFree(d_src);
  nppiFree(d_mask);
  cudaFree(pDeviceBuffer);
  cudaFree(pMean);
  cudaFree(pStdDev);
}

// Test 7: Context version for 32f_C1MR
TEST_F(NppiMeanStdDevMaskTest, Mean_StdDev_32f_C1MR_Ctx) {
  const int width = 64;
  const int height = 64;

  // Create test data
  std::vector<Npp32f> hostSrc(width * height);
  for (int i = 0; i < width * height; i++) {
    hostSrc[i] = static_cast<float>(i % 100) / 10.0f;
  }

  std::vector<Npp8u> hostMask = generateCircularMask(width, height);

  // Calculate reference values
  double refMean, refStdDev;
  calculateReferenceMeanStdDevWithMask(hostSrc, hostMask, refMean, refStdDev);

  // Create stream
  cudaStream_t stream;
  cudaStreamCreate(&stream);
  NppStreamContext nppStreamCtx;
  nppGetStreamContext(&nppStreamCtx);
  nppStreamCtx.hStream = stream;

  // Allocate GPU memory
  Npp32f *d_src = nullptr;
  Npp8u *d_mask = nullptr;
  int srcStep, maskStep;

  d_src = nppiMalloc_32f_C1(width, height, &srcStep);
  d_mask = nppiMalloc_8u_C1(width, height, &maskStep);
  ASSERT_NE(d_src, nullptr);
  ASSERT_NE(d_mask, nullptr);

  // Upload data asynchronously
  cudaMemcpy2DAsync(d_src, srcStep, hostSrc.data(), width * sizeof(Npp32f), width * sizeof(Npp32f), height,
                    cudaMemcpyHostToDevice, stream);
  cudaMemcpy2DAsync(d_mask, maskStep, hostMask.data(), width, width, height, cudaMemcpyHostToDevice, stream);

  // Get buffer size and allocate buffer
  NppiSize oSizeROI = {width, height};
  SIZE_TYPE bufferSize = 0;
  NppStatus status = nppiMeanStdDevGetBufferHostSize_32f_C1MR_Ctx(oSizeROI, &bufferSize, nppStreamCtx);
  ASSERT_EQ(status, NPP_SUCCESS);

  Npp8u *pDeviceBuffer = nullptr;
  cudaMalloc(&pDeviceBuffer, bufferSize);
  ASSERT_NE(pDeviceBuffer, nullptr);

  // Allocate device memory for results
  Npp64f *pMean = nullptr;
  Npp64f *pStdDev = nullptr;
  cudaMalloc(&pMean, sizeof(Npp64f));
  cudaMalloc(&pStdDev, sizeof(Npp64f));
  ASSERT_NE(pMean, nullptr);
  ASSERT_NE(pStdDev, nullptr);

  // Calculate mean and standard deviation with stream context
  status = nppiMean_StdDev_32f_C1MR_Ctx(d_src, srcStep, d_mask, maskStep, oSizeROI, pDeviceBuffer, pMean, pStdDev,
                                        nppStreamCtx);
  ASSERT_EQ(status, NPP_SUCCESS);

  // Synchronize stream and get results
  cudaStreamSynchronize(stream);

  Npp64f hostMean, hostStdDev;
  cudaMemcpy(&hostMean, pMean, sizeof(Npp64f), cudaMemcpyDeviceToHost);
  cudaMemcpy(&hostStdDev, pStdDev, sizeof(Npp64f), cudaMemcpyDeviceToHost);

  EXPECT_NEAR(hostMean, refMean, 0.1);
  EXPECT_NEAR(hostStdDev, refStdDev, 0.1);

  // Cleanup
  cudaStreamDestroy(stream);
  nppiFree(d_src);
  nppiFree(d_mask);
  cudaFree(pDeviceBuffer);
  cudaFree(pMean);
  cudaFree(pStdDev);
}

// ============================================================================
// Numerical stability tests
// ============================================================================

// High-precision CPU reference using long double and two-pass variance (Welford-friendly)
static void calculateReferenceMeanStdDevWithMaskPrecise(const std::vector<Npp8u> &data, int srcStep,
                                                        const std::vector<Npp8u> &mask, int maskStep,
                                                        int width, int height, double &mean, double &stddev,
                                                        int &validCount) {
  long double sum = 0.0L;
  int count = 0;

  for (int y = 0; y < height; ++y) {
    const Npp8u *srcRow = data.data() + y * srcStep;
    const Npp8u *maskRow = mask.data() + y * maskStep;
    for (int x = 0; x < width; ++x) {
      if (maskRow[x] > 0) {
        sum += static_cast<long double>(srcRow[x]);
        ++count;
      }
    }
  }

  validCount = count;
  if (count == 0) {
    mean = 0.0;
    stddev = 0.0;
    return;
  }

  long double lmean = sum / static_cast<long double>(count);

  // Two-pass variance: numerically stable for constant-valued images
  long double sqDiffSum = 0.0L;
  for (int y = 0; y < height; ++y) {
    const Npp8u *srcRow = data.data() + y * srcStep;
    const Npp8u *maskRow = mask.data() + y * maskStep;
    for (int x = 0; x < width; ++x) {
      if (maskRow[x] > 0) {
        long double diff = static_cast<long double>(srcRow[x]) - lmean;
        sqDiffSum += diff * diff;
      }
    }
  }

  long double variance = sqDiffSum / static_cast<long double>(count);
  if (variance < 0.0L && fabsl(variance) < 1e-15L)
    variance = 0.0L;

  mean = static_cast<double>(lmean);
  stddev = static_cast<double>(sqrtl(variance));
}

// Helper that runs a single mean/stddev case with pitched memory and returns results
static void runMeanStdDev8uC1MR(const std::vector<Npp8u> &hostSrc, const std::vector<Npp8u> &hostMask,
                                int width, int height, double &outMean, double &outStdDev) {
  Npp8u *d_src = nullptr;
  Npp8u *d_mask = nullptr;
  int srcStep, maskStep;

  d_src = nppiMalloc_8u_C1(width, height, &srcStep);
  d_mask = nppiMalloc_8u_C1(width, height, &maskStep);
  ASSERT_NE(d_src, nullptr);
  ASSERT_NE(d_mask, nullptr);

  cudaMemcpy2D(d_src, srcStep, hostSrc.data(), width, width, height, cudaMemcpyHostToDevice);
  cudaMemcpy2D(d_mask, maskStep, hostMask.data(), width, width, height, cudaMemcpyHostToDevice);

  NppiSize oSizeROI = {width, height};
  SIZE_TYPE bufferSize = 0;
  NppStatus status = nppiMeanStdDevGetBufferHostSize_8u_C1MR(oSizeROI, &bufferSize);
  ASSERT_EQ(status, NPP_SUCCESS);

  Npp8u *pDeviceBuffer = nullptr;
  cudaMalloc(&pDeviceBuffer, bufferSize);
  ASSERT_NE(pDeviceBuffer, nullptr);

  Npp64f *pMean = nullptr;
  Npp64f *pStdDev = nullptr;
  cudaMalloc(&pMean, sizeof(Npp64f));
  cudaMalloc(&pStdDev, sizeof(Npp64f));
  ASSERT_NE(pMean, nullptr);
  ASSERT_NE(pStdDev, nullptr);

  status = nppiMean_StdDev_8u_C1MR(d_src, srcStep, d_mask, maskStep, oSizeROI, pDeviceBuffer, pMean, pStdDev);
  ASSERT_EQ(status, NPP_SUCCESS);

  Npp64f hostMean = 0.0, hostStdDev = 0.0;
  cudaMemcpy(&hostMean, pMean, sizeof(Npp64f), cudaMemcpyDeviceToHost);
  cudaMemcpy(&hostStdDev, pStdDev, sizeof(Npp64f), cudaMemcpyDeviceToHost);

  outMean = static_cast<double>(hostMean);
  outStdDev = static_cast<double>(hostStdDev);

  nppiFree(d_src);
  nppiFree(d_mask);
  cudaFree(pDeviceBuffer);
  cudaFree(pMean);
  cudaFree(pStdDev);
}

// Numerical stability: constant image with large pixel count
// When all pixels are the same value, mean should equal that value exactly
// and stddev should be 0. The E[X^2] - (E[X])^2 formula can produce
// negative or non-zero variance due to catastrophic cancellation.
TEST_F(NppiMeanStdDevMaskTest, NumericalStability_ConstantImage_LargeFullMask) {
  const int width = 512;
  const int height = 512;
  const Npp8u constantValue = 37;

  std::vector<Npp8u> hostSrc(width * height, constantValue);
  std::vector<Npp8u> hostMask(width * height, 255);

  double refMean, refStdDev;
  int validCount;
  calculateReferenceMeanStdDevWithMaskPrecise(hostSrc, width, hostMask, width, width, height, refMean, refStdDev,
                                              validCount);

  double nppMean = 0.0, nppStdDev = 0.0;
  runMeanStdDev8uC1MR(hostSrc, hostMask, width, height, nppMean, nppStdDev);

  // For a constant image, mean should be exact and stddev should be 0
  EXPECT_NEAR(nppMean, refMean, 1e-6) << "constant37_512x512: mean mismatch";
  EXPECT_NEAR(nppStdDev, refStdDev, 1e-6) << "constant37_512x512: stddev should be 0 for constant image";
}

// Numerical stability: constant image at max value (255) with large pixel count
// E[X^2] = 255^2 = 65025, which is large, making cancellation worse
TEST_F(NppiMeanStdDevMaskTest, NumericalStability_Constant255_LargeFullMask) {
  const int width = 512;
  const int height = 512;
  const Npp8u constantValue = 255;

  std::vector<Npp8u> hostSrc(width * height, constantValue);
  std::vector<Npp8u> hostMask(width * height, 255);

  double refMean, refStdDev;
  int validCount;
  calculateReferenceMeanStdDevWithMaskPrecise(hostSrc, width, hostMask, width, width, height, refMean, refStdDev,
                                              validCount);

  double nppMean = 0.0, nppStdDev = 0.0;
  runMeanStdDev8uC1MR(hostSrc, hostMask, width, height, nppMean, nppStdDev);

  EXPECT_NEAR(nppMean, refMean, 1e-6) << "constant255_512x512: mean mismatch";
  EXPECT_NEAR(nppStdDev, refStdDev, 1e-6) << "constant255_512x512: stddev should be 0 for constant image";
}

// Numerical stability: extreme 0/255 checkerboard with full mask
// Both values are extreme, variance is maximal, E[X^2] and E[X]^2 are close
TEST_F(NppiMeanStdDevMaskTest, NumericalStability_Extreme01_LargeFullMask) {
  const int width = 512;
  const int height = 512;

  std::vector<Npp8u> hostSrc(width * height);
  for (int y = 0; y < height; ++y)
    for (int x = 0; x < width; ++x)
      hostSrc[y * width + x] = ((x + y) & 1) ? 255 : 0;

  std::vector<Npp8u> hostMask(width * height, 255);

  double refMean, refStdDev;
  int validCount;
  calculateReferenceMeanStdDevWithMaskPrecise(hostSrc, width, hostMask, width, width, height, refMean, refStdDev,
                                              validCount);

  double nppMean = 0.0, nppStdDev = 0.0;
  runMeanStdDev8uC1MR(hostSrc, hostMask, width, height, nppMean, nppStdDev);

  EXPECT_NEAR(nppMean, refMean, 1e-6) << "extreme_0_255_512x512: mean mismatch";
  EXPECT_NEAR(nppStdDev, refStdDev, 1e-6) << "extreme_0_255_512x512: stddev mismatch";
}

// Numerical stability: large ramp image with full mask
TEST_F(NppiMeanStdDevMaskTest, NumericalStability_Ramp_LargeFullMask) {
  const int width = 512;
  const int height = 512;

  std::vector<Npp8u> hostSrc(width * height);
  for (int y = 0; y < height; ++y)
    for (int x = 0; x < width; ++x)
      hostSrc[y * width + x] = static_cast<Npp8u>((x + y) & 0xFF);

  std::vector<Npp8u> hostMask(width * height, 255);

  double refMean, refStdDev;
  int validCount;
  calculateReferenceMeanStdDevWithMaskPrecise(hostSrc, width, hostMask, width, width, height, refMean, refStdDev,
                                              validCount);

  double nppMean = 0.0, nppStdDev = 0.0;
  runMeanStdDev8uC1MR(hostSrc, hostMask, width, height, nppMean, nppStdDev);

  EXPECT_NEAR(nppMean, refMean, 1e-6) << "ramp_512x512: mean mismatch";
  EXPECT_NEAR(nppStdDev, refStdDev, 1e-6) << "ramp_512x512: stddev mismatch";
}

// Numerical stability: random image with full mask (large)
TEST_F(NppiMeanStdDevMaskTest, NumericalStability_Random_LargeFullMask) {
  const int width = 512;
  const int height = 512;

  std::mt19937 rng(12345);
  std::uniform_int_distribution<int> dist(0, 255);

  std::vector<Npp8u> hostSrc(width * height);
  for (auto &v : hostSrc)
    v = static_cast<Npp8u>(dist(rng));

  std::vector<Npp8u> hostMask(width * height, 255);

  double refMean, refStdDev;
  int validCount;
  calculateReferenceMeanStdDevWithMaskPrecise(hostSrc, width, hostMask, width, width, height, refMean, refStdDev,
                                              validCount);

  double nppMean = 0.0, nppStdDev = 0.0;
  runMeanStdDev8uC1MR(hostSrc, hostMask, width, height, nppMean, nppStdDev);

  EXPECT_NEAR(nppMean, refMean, 1e-4) << "random_512x512: mean mismatch";
  EXPECT_NEAR(nppStdDev, refStdDev, 1e-4) << "random_512x512: stddev mismatch";
}

// Numerical stability: random image with random mask (large, ~50% valid)
TEST_F(NppiMeanStdDevMaskTest, NumericalStability_Random_LargeRandomMask) {
  const int width = 512;
  const int height = 512;

  std::mt19937 srcRng(777), maskRng(888);
  std::uniform_int_distribution<int> pixelDist(0, 255);
  std::uniform_int_distribution<int> maskDist(0, 1);

  std::vector<Npp8u> hostSrc(width * height);
  std::vector<Npp8u> hostMask(width * height);
  for (int i = 0; i < width * height; ++i) {
    hostSrc[i] = static_cast<Npp8u>(pixelDist(srcRng));
    hostMask[i] = maskDist(maskRng) ? 255 : 0;
  }

  double refMean, refStdDev;
  int validCount;
  calculateReferenceMeanStdDevWithMaskPrecise(hostSrc, width, hostMask, width, width, height, refMean, refStdDev,
                                              validCount);

  double nppMean = 0.0, nppStdDev = 0.0;
  runMeanStdDev8uC1MR(hostSrc, hostMask, width, height, nppMean, nppStdDev);

  EXPECT_NEAR(nppMean, refMean, 1e-4) << "random_random_mask_512x512: mean mismatch";
  EXPECT_NEAR(nppStdDev, refStdDev, 1e-4) << "random_random_mask_512x512: stddev mismatch";
}

// Numerical stability: constant image with sparse mask
// Very few valid pixels — tests whether valid count accumulation is correct
TEST_F(NppiMeanStdDevMaskTest, NumericalStability_ConstantImage_SparseMask) {
  const int width = 512;
  const int height = 512;
  const Npp8u constantValue = 100;

  std::vector<Npp8u> hostSrc(width * height, constantValue);
  std::vector<Npp8u> hostMask(width * height, 0);
  // Only every 16th pixel in both x and y is valid
  for (int y = 0; y < height; ++y)
    for (int x = 0; x < width; ++x)
      hostMask[y * width + x] = ((x % 16 == 0) && (y % 16 == 0)) ? 255 : 0;

  double refMean, refStdDev;
  int validCount;
  calculateReferenceMeanStdDevWithMaskPrecise(hostSrc, width, hostMask, width, width, height, refMean, refStdDev,
                                              validCount);

  double nppMean = 0.0, nppStdDev = 0.0;
  runMeanStdDev8uC1MR(hostSrc, hostMask, width, height, nppMean, nppStdDev);

  EXPECT_NEAR(nppMean, refMean, 1e-6) << "constant_sparse_mask: mean mismatch";
  EXPECT_NEAR(nppStdDev, refStdDev, 1e-6) << "constant_sparse_mask: stddev should be 0";
}

// Numerical stability: repeat consistency
// Running the same computation multiple times should yield identical results
TEST_F(NppiMeanStdDevMaskTest, NumericalStability_RepeatConsistency) {
  const int width = 512;
  const int height = 512;
  const int repeatCount = 100;

  std::vector<Npp8u> hostSrc(width * height, 37);
  std::vector<Npp8u> hostMask(width * height, 255);

  Npp8u *d_src = nullptr;
  Npp8u *d_mask = nullptr;
  int srcStep, maskStep;
  d_src = nppiMalloc_8u_C1(width, height, &srcStep);
  d_mask = nppiMalloc_8u_C1(width, height, &maskStep);
  ASSERT_NE(d_src, nullptr);
  ASSERT_NE(d_mask, nullptr);

  cudaMemcpy2D(d_src, srcStep, hostSrc.data(), width, width, height, cudaMemcpyHostToDevice);
  cudaMemcpy2D(d_mask, maskStep, hostMask.data(), width, width, height, cudaMemcpyHostToDevice);

  NppiSize oSizeROI = {width, height};
  SIZE_TYPE bufferSize = 0;
  NppStatus status = nppiMeanStdDevGetBufferHostSize_8u_C1MR(oSizeROI, &bufferSize);
  ASSERT_EQ(status, NPP_SUCCESS);

  Npp8u *pDeviceBuffer = nullptr;
  cudaMalloc(&pDeviceBuffer, bufferSize);
  ASSERT_NE(pDeviceBuffer, nullptr);

  Npp64f *pMean = nullptr;
  Npp64f *pStdDev = nullptr;
  cudaMalloc(&pMean, sizeof(Npp64f));
  cudaMalloc(&pStdDev, sizeof(Npp64f));
  ASSERT_NE(pMean, nullptr);
  ASSERT_NE(pStdDev, nullptr);

  double firstMean = 0.0, firstStdDev = 0.0;
  for (int i = 0; i < repeatCount; ++i) {
    status = nppiMean_StdDev_8u_C1MR(d_src, srcStep, d_mask, maskStep, oSizeROI, pDeviceBuffer, pMean, pStdDev);
    ASSERT_EQ(status, NPP_SUCCESS);
    cudaDeviceSynchronize();

    Npp64f mean = 0.0, stddev = 0.0;
    cudaMemcpy(&mean, pMean, sizeof(Npp64f), cudaMemcpyDeviceToHost);
    cudaMemcpy(&stddev, pStdDev, sizeof(Npp64f), cudaMemcpyDeviceToHost);

    if (i == 0) {
      firstMean = static_cast<double>(mean);
      firstStdDev = static_cast<double>(stddev);
    } else {
      EXPECT_DOUBLE_EQ(static_cast<double>(mean), firstMean)
          << "repeat inconsistency at iter " << i << ": mean changed";
      EXPECT_DOUBLE_EQ(static_cast<double>(stddev), firstStdDev)
          << "repeat inconsistency at iter " << i << ": stddev changed";
    }
  }

  nppiFree(d_src);
  nppiFree(d_mask);
  cudaFree(pDeviceBuffer);
  cudaFree(pMean);
  cudaFree(pStdDev);
}

// Numerical stability: single valid pixel under mask
// Edge case where validCount = 1, stddev must be 0
TEST_F(NppiMeanStdDevMaskTest, NumericalStability_SinglePixelMask) {
  const int width = 512;
  const int height = 512;

  std::mt19937 rng(45678);
  std::uniform_int_distribution<int> dist(0, 255);

  std::vector<Npp8u> hostSrc(width * height);
  for (auto &v : hostSrc)
    v = static_cast<Npp8u>(dist(rng));

  std::vector<Npp8u> hostMask(width * height, 0);
  // Only one valid pixel
  hostMask[234 * width + 123] = 255;

  double refMean, refStdDev;
  int validCount;
  calculateReferenceMeanStdDevWithMaskPrecise(hostSrc, width, hostMask, width, width, height, refMean, refStdDev,
                                              validCount);

  double nppMean = 0.0, nppStdDev = 0.0;
  runMeanStdDev8uC1MR(hostSrc, hostMask, width, height, nppMean, nppStdDev);

  EXPECT_EQ(validCount, 1);
  EXPECT_NEAR(nppMean, refMean, 1e-6) << "single_pixel: mean should equal the pixel value";
  EXPECT_NEAR(nppStdDev, 0.0, 1e-6) << "single_pixel: stddev must be 0";
}

// Test 8: Edge case - empty mask (all zeros)
TEST_F(NppiMeanStdDevMaskTest, Mean_StdDev_8u_C1MR_EmptyMask) {
  const int width = 32;
  const int height = 32;

  // Create test data and empty mask
  std::vector<Npp8u> hostSrc(width * height, 100);
  std::vector<Npp8u> hostMask(width * height, 0); // All mask pixels are zero

  // Allocate GPU memory
  Npp8u *d_src = nullptr;
  Npp8u *d_mask = nullptr;
  int srcStep, maskStep;

  d_src = nppiMalloc_8u_C1(width, height, &srcStep);
  d_mask = nppiMalloc_8u_C1(width, height, &maskStep);
  ASSERT_NE(d_src, nullptr);
  ASSERT_NE(d_mask, nullptr);

  // Upload data
  cudaMemcpy2D(d_src, srcStep, hostSrc.data(), width, width, height, cudaMemcpyHostToDevice);
  cudaMemcpy2D(d_mask, maskStep, hostMask.data(), width, width, height, cudaMemcpyHostToDevice);

  // Get buffer size and allocate buffer
  NppiSize oSizeROI = {width, height};
  SIZE_TYPE bufferSize = 0;
  NppStatus status = nppiMeanStdDevGetBufferHostSize_8u_C1MR(oSizeROI, &bufferSize);
  ASSERT_EQ(status, NPP_SUCCESS);

  Npp8u *pDeviceBuffer = nullptr;
  cudaMalloc(&pDeviceBuffer, bufferSize);
  ASSERT_NE(pDeviceBuffer, nullptr);

  // Allocate device memory for results
  Npp64f *pMean = nullptr;
  Npp64f *pStdDev = nullptr;
  cudaMalloc(&pMean, sizeof(Npp64f));
  cudaMalloc(&pStdDev, sizeof(Npp64f));
  ASSERT_NE(pMean, nullptr);
  ASSERT_NE(pStdDev, nullptr);

  // Calculate mean and standard deviation - should handle empty mask gracefully
  status = nppiMean_StdDev_8u_C1MR(d_src, srcStep, d_mask, maskStep, oSizeROI, pDeviceBuffer, pMean, pStdDev);

  // NPP behavior with empty mask varies - some implementations return error, others return 0
  // We'll accept both behaviors
  if (status == NPP_SUCCESS) {
    Npp64f hostMean, hostStdDev;
    cudaMemcpy(&hostMean, pMean, sizeof(Npp64f), cudaMemcpyDeviceToHost);
    cudaMemcpy(&hostStdDev, pStdDev, sizeof(Npp64f), cudaMemcpyDeviceToHost);
    // With empty mask, results may be 0 or undefined
    std::cout << "Empty mask result - Mean: " << hostMean << ", StdDev: " << hostStdDev << std::endl;
  } else {
    // Some NPP implementations return error for empty mask
    std::cout << "Empty mask returned error status: " << status << std::endl;
  }

  // Cleanup
  nppiFree(d_src);
  nppiFree(d_mask);
  cudaFree(pDeviceBuffer);
  cudaFree(pMean);
  cudaFree(pStdDev);
}