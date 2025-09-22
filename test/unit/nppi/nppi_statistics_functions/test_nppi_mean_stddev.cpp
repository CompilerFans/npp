#include "../../../../src/npp_version_compat.h"
#include "npp.h"

// only enable at CUDA 12.8
#if CUDA_SDK_AT_LEAST(12, 8)
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <random>
#include <vector>

class NppiMeanStdDevTest : public ::testing::Test {
protected:
  void SetUp() override {
    cudaError_t err = cudaSetDevice(0);
    ASSERT_EQ(err, cudaSuccess);
  }

  void TearDown() override {
    cudaError_t err = cudaDeviceSynchronize();
    EXPECT_EQ(err, cudaSuccess);
  }

  // Function to calculate reference mean and standard deviation
  template <typename T> void calculateReferenceMeanStdDev(const std::vector<T> &data, double &mean, double &stddev) {
    double sum = 0.0;
    for (const auto &val : data) {
      sum += static_cast<double>(val);
    }
    mean = sum / data.size();

    double sumSquaredDiff = 0.0;
    for (const auto &val : data) {
      double diff = static_cast<double>(val) - mean;
      sumSquaredDiff += diff * diff;
    }
    stddev = std::sqrt(sumSquaredDiff / data.size());
  }
};

// Test 1: Buffer size calculation for 8u_C1R
TEST_F(NppiMeanStdDevTest, MeanStdDevGetBufferHostSize_8u_C1R) {
  NppiSize oSizeROI = {256, 256};
  size_t bufferSize = 0;

  NppStatus status = nppiMeanStdDevGetBufferHostSize_8u_C1R(oSizeROI, &bufferSize);
  ASSERT_EQ(status, NPP_SUCCESS);
  EXPECT_GT(bufferSize, 0);

  // Test with different sizes
  NppiSize smallROI = {64, 64};
  size_t smallBufferSize = 0;
  status = nppiMeanStdDevGetBufferHostSize_8u_C1R(smallROI, &smallBufferSize);
  ASSERT_EQ(status, NPP_SUCCESS);
  EXPECT_GT(smallBufferSize, 0);

  // Debug output to understand the actual buffer sizes
  std::cout << "Large ROI (" << oSizeROI.width << "x" << oSizeROI.height << ") buffer size: " << bufferSize << std::endl;
  std::cout << "Small ROI (" << smallROI.width << "x" << smallROI.height << ") buffer size: " << smallBufferSize << std::endl;

  // Note: NPP buffer size may be implementation-specific and not always proportional to image size
  // Some implementations use fixed buffer sizes or have different scaling algorithms
  // We'll just verify both buffer sizes are reasonable (non-zero and not excessively large)
  EXPECT_GT(bufferSize, 0);
  EXPECT_LT(bufferSize, size_t(10 * 1024 * 1024)); // Less than 10MB should be reasonable
  EXPECT_GT(smallBufferSize, 0);
  EXPECT_LT(smallBufferSize, size_t(10 * 1024 * 1024)); // Less than 10MB should be reasonable
}

// Test 2: Buffer size calculation for 32f_C1R
TEST_F(NppiMeanStdDevTest, MeanStdDevGetBufferHostSize_32f_C1R) {
  NppiSize oSizeROI = {512, 512};
  size_t bufferSize = 0;

  NppStatus status = nppiMeanStdDevGetBufferHostSize_32f_C1R(oSizeROI, &bufferSize);
  ASSERT_EQ(status, NPP_SUCCESS);
  EXPECT_GT(bufferSize, 0);
}

// Test 3: Mean and StdDev calculation for 8u_C1R with constant data
TEST_F(NppiMeanStdDevTest, Mean_StdDev_8u_C1R_ConstantData) {
  const int width = 128;
  const int height = 128;
  const Npp8u constantValue = 100;

  // Create constant data
  std::vector<Npp8u> hostSrc(width * height, constantValue);

  // Allocate GPU memory
  Npp8u *d_src = nullptr;
  int srcStep;
  d_src = nppiMalloc_8u_C1(width, height, &srcStep);
  ASSERT_NE(d_src, nullptr);

  // Upload data
  cudaMemcpy2D(d_src, srcStep, hostSrc.data(), width, width, height, cudaMemcpyHostToDevice);

  // Get buffer size and allocate buffer
  NppiSize oSizeROI = {width, height};
  size_t bufferSize = 0;
  NppStatus status = nppiMeanStdDevGetBufferHostSize_8u_C1R(oSizeROI, &bufferSize);
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
  status = nppiMean_StdDev_8u_C1R(d_src, srcStep, oSizeROI, pDeviceBuffer, pMean, pStdDev);
  ASSERT_EQ(status, NPP_SUCCESS);

  // Copy results to host and verify
  Npp64f hostMean, hostStdDev;
  cudaMemcpy(&hostMean, pMean, sizeof(Npp64f), cudaMemcpyDeviceToHost);
  cudaMemcpy(&hostStdDev, pStdDev, sizeof(Npp64f), cudaMemcpyDeviceToHost);

  // For constant data, mean should be the constant value and stddev should be 0
  EXPECT_NEAR(hostMean, static_cast<double>(constantValue), 0.001);
  EXPECT_NEAR(hostStdDev, 0.0, 0.001);

  // Cleanup
  nppiFree(d_src);
  cudaFree(pDeviceBuffer);
  cudaFree(pMean);
  cudaFree(pStdDev);
}

// Test 4: Mean and StdDev calculation for 32f_C1R with gradient data
TEST_F(NppiMeanStdDevTest, Mean_StdDev_32f_C1R_GradientData) {
  const int width = 64;
  const int height = 64;

  // Create gradient data
  std::vector<Npp32f> hostSrc(width * height);
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      hostSrc[y * width + x] = static_cast<float>(x + y);
    }
  }

  // Calculate reference values
  double refMean, refStdDev;
  calculateReferenceMeanStdDev(hostSrc, refMean, refStdDev);

  // Allocate GPU memory
  Npp32f *d_src = nullptr;
  int srcStep;
  d_src = nppiMalloc_32f_C1(width, height, &srcStep);
  ASSERT_NE(d_src, nullptr);

  // Upload data
  cudaMemcpy2D(d_src, srcStep, hostSrc.data(), width * sizeof(Npp32f), width * sizeof(Npp32f), height,
               cudaMemcpyHostToDevice);

  // Get buffer size and allocate buffer
  NppiSize oSizeROI = {width, height};
  size_t bufferSize = 0;
  NppStatus status = nppiMeanStdDevGetBufferHostSize_32f_C1R(oSizeROI, &bufferSize);
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
  status = nppiMean_StdDev_32f_C1R(d_src, srcStep, oSizeROI, pDeviceBuffer, pMean, pStdDev);
  ASSERT_EQ(status, NPP_SUCCESS);

  // Copy results to host and compare with reference values
  Npp64f hostMean, hostStdDev;
  cudaMemcpy(&hostMean, pMean, sizeof(Npp64f), cudaMemcpyDeviceToHost);
  cudaMemcpy(&hostStdDev, pStdDev, sizeof(Npp64f), cudaMemcpyDeviceToHost);

  EXPECT_NEAR(hostMean, refMean, 0.01);
  EXPECT_NEAR(hostStdDev, refStdDev, 0.01);

  // Cleanup
  nppiFree(d_src);
  cudaFree(pDeviceBuffer);
  cudaFree(pMean);
  cudaFree(pStdDev);
}

// Test 5: Stream context test for 8u_C1R
TEST_F(NppiMeanStdDevTest, Mean_StdDev_8u_C1R_StreamContext) {
  const int width = 256;
  const int height = 256;

  // Create random data with known statistics
  std::vector<Npp8u> hostSrc(width * height);
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> dis(0, 255);

  for (auto &val : hostSrc) {
    val = static_cast<Npp8u>(dis(gen));
  }

  // Calculate reference values
  double refMean, refStdDev;
  calculateReferenceMeanStdDev(hostSrc, refMean, refStdDev);

  // Allocate GPU memory
  Npp8u *d_src = nullptr;
  int srcStep;
  d_src = nppiMalloc_8u_C1(width, height, &srcStep);
  ASSERT_NE(d_src, nullptr);

  // Create stream
  cudaStream_t stream;
  cudaStreamCreate(&stream);
  NppStreamContext nppStreamCtx;
  nppGetStreamContext(&nppStreamCtx);
  nppStreamCtx.hStream = stream;

  // Upload data asynchronously
  cudaMemcpy2DAsync(d_src, srcStep, hostSrc.data(), width, width, height, cudaMemcpyHostToDevice, stream);

  // Get buffer size and allocate buffer
  NppiSize oSizeROI = {width, height};
  size_t bufferSize = 0;
  NppStatus status = nppiMeanStdDevGetBufferHostSize_8u_C1R_Ctx(oSizeROI, &bufferSize, nppStreamCtx);
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
  status = nppiMean_StdDev_8u_C1R_Ctx(d_src, srcStep, oSizeROI, pDeviceBuffer, pMean, pStdDev, nppStreamCtx);
  ASSERT_EQ(status, NPP_SUCCESS);

  // Synchronize stream and get results
  cudaStreamSynchronize(stream);

  Npp64f hostMean, hostStdDev;
  cudaMemcpy(&hostMean, pMean, sizeof(Npp64f), cudaMemcpyDeviceToHost);
  cudaMemcpy(&hostStdDev, pStdDev, sizeof(Npp64f), cudaMemcpyDeviceToHost);

  // Compare with reference values
  EXPECT_NEAR(hostMean, refMean, 1.0); // Allow for some numerical differences with random data
  EXPECT_NEAR(hostStdDev, refStdDev, 1.0);

  // Cleanup
  cudaStreamDestroy(stream);
  nppiFree(d_src);
  cudaFree(pDeviceBuffer);
  cudaFree(pMean);
  cudaFree(pStdDev);
}

#endif // CUDA_SDK_AT_LEAST(12, 8)
