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