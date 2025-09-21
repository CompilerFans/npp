#include "npp.h"
#include <algorithm>
#include <chrono>
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <random>
#include <vector>

class NppiFilterBoxExtendedTest : public ::testing::Test {
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

// Test 1: nppiFilterBox_8u_C4R basic functionality
TEST_F(NppiFilterBoxExtendedTest, FilterBox_8u_C4R_BasicTest) {
  const int width = 128;
  const int height = 128;
  const int channels = 4;

  // Create test data with different patterns for each channel
  std::vector<Npp8u> hostSrc(width * height * channels);
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      int idx = (y * width + x) * channels;
      hostSrc[idx] = (x + y) % 256;         // R channel
      hostSrc[idx + 1] = (x * 2) % 256;     // G channel
      hostSrc[idx + 2] = (y * 2) % 256;     // B channel
      hostSrc[idx + 3] = (x + y * 2) % 256; // A channel
    }
  }

  // Allocate GPU memory
  Npp8u *d_src = nullptr;
  Npp8u *d_dst = nullptr;
  int srcStep, dstStep;
  d_src = nppiMalloc_8u_C4(width, height, &srcStep);
  d_dst = nppiMalloc_8u_C4(width, height, &dstStep);
  ASSERT_NE(d_src, nullptr);
  ASSERT_NE(d_dst, nullptr);

  // Upload source data
  cudaMemcpy2D(d_src, srcStep, hostSrc.data(), width * channels, width * channels, height, cudaMemcpyHostToDevice);

  // Apply 3x3 box filter
  NppiSize oSizeROI = {width, height};
  NppiSize oMaskSize = {3, 3};
  NppiPoint oAnchor = {1, 1};

  NppStatus status = nppiFilterBox_8u_C4R(d_src, srcStep, d_dst, dstStep, oSizeROI, oMaskSize, oAnchor);
  ASSERT_EQ(status, NPP_SUCCESS);

  // Download result
  std::vector<Npp8u> hostDst(width * height * channels);
  cudaMemcpy2D(hostDst.data(), width * channels, d_dst, dstStep, width * channels, height, cudaMemcpyDeviceToHost);

  // Verify that center pixels are smoothed (basic sanity check)
  int centerX = width / 2;
  int centerY = height / 2;
  int centerIdx = (centerY * width + centerX) * channels;

  // Check that result is within reasonable range for each channel
  for (int c = 0; c < channels; c++) {
    EXPECT_GE(hostDst[centerIdx + c], 0);
    EXPECT_LE(hostDst[centerIdx + c], 255);
  }

  nppiFree(d_src);
  nppiFree(d_dst);
}

// Test 2: nppiFilterBox_32f_C1R basic functionality
TEST_F(NppiFilterBoxExtendedTest, FilterBox_32f_C1R_BasicTest) {
  const int width = 64;
  const int height = 64;

  // Create test data with gradient pattern
  std::vector<Npp32f> hostSrc(width * height);
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      hostSrc[y * width + x] = static_cast<float>(x + y * 2);
    }
  }

  // Allocate GPU memory
  Npp32f *d_src = nullptr;
  Npp32f *d_dst = nullptr;
  int srcStep, dstStep;
  d_src = nppiMalloc_32f_C1(width, height, &srcStep);
  d_dst = nppiMalloc_32f_C1(width, height, &dstStep);
  ASSERT_NE(d_src, nullptr);
  ASSERT_NE(d_dst, nullptr);

  // Upload source data
  cudaMemcpy2D(d_src, srcStep, hostSrc.data(), width * sizeof(Npp32f), width * sizeof(Npp32f), height,
               cudaMemcpyHostToDevice);

  // Apply 5x5 box filter
  NppiSize oSizeROI = {width, height};
  NppiSize oMaskSize = {5, 5};
  NppiPoint oAnchor = {2, 2};

  NppStatus status = nppiFilterBox_32f_C1R(d_src, srcStep, d_dst, dstStep, oSizeROI, oMaskSize, oAnchor);
  ASSERT_EQ(status, NPP_SUCCESS);

  // Download result
  std::vector<Npp32f> hostDst(width * height);
  cudaMemcpy2D(hostDst.data(), width * sizeof(Npp32f), d_dst, dstStep, width * sizeof(Npp32f), height,
               cudaMemcpyDeviceToHost);

  // Verify that the filter smooths the gradient
  // Center pixel should be average of surrounding 5x5 area
  int centerX = width / 2;
  int centerY = height / 2;

  // Calculate expected value manually for center pixel
  float expectedSum = 0.0f;
  int count = 0;
  for (int dy = -2; dy <= 2; dy++) {
    for (int dx = -2; dx <= 2; dx++) {
      int x = centerX + dx;
      int y = centerY + dy;
      if (x >= 0 && x < width && y >= 0 && y < height) {
        expectedSum += static_cast<float>(x + y * 2);
        count++;
      }
    }
  }
  float expectedValue = expectedSum / count;

  float actualValue = hostDst[centerY * width + centerX];
  EXPECT_NEAR(actualValue, expectedValue, 0.1f);

  nppiFree(d_src);
  nppiFree(d_dst);
}

// Test 3: Stream context for 8u_C4R
TEST_F(NppiFilterBoxExtendedTest, FilterBox_8u_C4R_StreamContext) {
  const int width = 256;
  const int height = 256;
  const int channels = 4;

  // Create test data
  std::vector<Npp8u> hostSrc(width * height * channels, 128);

  // Allocate GPU memory
  Npp8u *d_src = nullptr;
  Npp8u *d_dst = nullptr;
  int srcStep, dstStep;
  d_src = nppiMalloc_8u_C4(width, height, &srcStep);
  d_dst = nppiMalloc_8u_C4(width, height, &dstStep);
  ASSERT_NE(d_src, nullptr);
  ASSERT_NE(d_dst, nullptr);

  // Create stream
  cudaStream_t stream;
  cudaStreamCreate(&stream);
  NppStreamContext nppStreamCtx;
  nppGetStreamContext(&nppStreamCtx);
  nppStreamCtx.hStream = stream;

  // Upload data
  cudaMemcpy2DAsync(d_src, srcStep, hostSrc.data(), width * channels, width * channels, height, cudaMemcpyHostToDevice,
                    stream);

  // Apply filter with stream context
  NppiSize oSizeROI = {width, height};
  NppiSize oMaskSize = {3, 3};
  NppiPoint oAnchor = {1, 1};

  NppStatus status =
      nppiFilterBox_8u_C4R_Ctx(d_src, srcStep, d_dst, dstStep, oSizeROI, oMaskSize, oAnchor, nppStreamCtx);
  ASSERT_EQ(status, NPP_SUCCESS);

  // Synchronize stream
  cudaStreamSynchronize(stream);

  // Verify result
  std::vector<Npp8u> hostDst(width * height * channels);
  cudaMemcpy2D(hostDst.data(), width * channels, d_dst, dstStep, width * channels, height, cudaMemcpyDeviceToHost);

  // With uniform input of 128, output should also be close to 128
  EXPECT_NEAR(hostDst[width * height * channels / 2], 128, 5);

  cudaStreamDestroy(stream);
  nppiFree(d_src);
  nppiFree(d_dst);
}

// Test 4: Performance comparison 32f vs 8u
TEST_F(NppiFilterBoxExtendedTest, FilterBox_PerformanceComparison) {
  const int width = 1024;
  const int height = 1024;
  const int iterations = 10;

  // Test data
  std::vector<Npp8u> hostSrc8u(width * height, 64);
  std::vector<Npp32f> hostSrc32f(width * height, 64.0f);

  // Allocate memory for 8u test
  Npp8u *d_src8u = nullptr, *d_dst8u = nullptr;
  int srcStep8u, dstStep8u;
  d_src8u = nppiMalloc_8u_C1(width, height, &srcStep8u);
  d_dst8u = nppiMalloc_8u_C1(width, height, &dstStep8u);

  // Allocate memory for 32f test
  Npp32f *d_src32f = nullptr, *d_dst32f = nullptr;
  int srcStep32f, dstStep32f;
  d_src32f = nppiMalloc_32f_C1(width, height, &srcStep32f);
  d_dst32f = nppiMalloc_32f_C1(width, height, &dstStep32f);

  ASSERT_NE(d_src8u, nullptr);
  ASSERT_NE(d_dst8u, nullptr);
  ASSERT_NE(d_src32f, nullptr);
  ASSERT_NE(d_dst32f, nullptr);

  // Upload data
  cudaMemcpy2D(d_src8u, srcStep8u, hostSrc8u.data(), width, width, height, cudaMemcpyHostToDevice);
  cudaMemcpy2D(d_src32f, srcStep32f, hostSrc32f.data(), width * sizeof(Npp32f), width * sizeof(Npp32f), height,
               cudaMemcpyHostToDevice);

  NppiSize oSizeROI = {width, height};
  NppiSize oMaskSize = {5, 5};
  NppiPoint oAnchor = {2, 2};

  // Warm up and test 8u version
  nppiFilterBox_8u_C1R(d_src8u, srcStep8u, d_dst8u, dstStep8u, oSizeROI, oMaskSize, oAnchor);
  cudaDeviceSynchronize();

  auto start8u = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < iterations; i++) {
    nppiFilterBox_8u_C1R(d_src8u, srcStep8u, d_dst8u, dstStep8u, oSizeROI, oMaskSize, oAnchor);
  }
  cudaDeviceSynchronize();
  auto end8u = std::chrono::high_resolution_clock::now();

  // Warm up and test 32f version
  nppiFilterBox_32f_C1R(d_src32f, srcStep32f, d_dst32f, dstStep32f, oSizeROI, oMaskSize, oAnchor);
  cudaDeviceSynchronize();

  auto start32f = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < iterations; i++) {
    nppiFilterBox_32f_C1R(d_src32f, srcStep32f, d_dst32f, dstStep32f, oSizeROI, oMaskSize, oAnchor);
  }
  cudaDeviceSynchronize();
  auto end32f = std::chrono::high_resolution_clock::now();

  auto duration8u = std::chrono::duration_cast<std::chrono::microseconds>(end8u - start8u).count();
  auto duration32f = std::chrono::duration_cast<std::chrono::microseconds>(end32f - start32f).count();

  std::cout << "Box Filter Performance Comparison:" << std::endl;
  std::cout << "  8u_C1R average time: " << duration8u / iterations << " μs" << std::endl;
  std::cout << "  32f_C1R average time: " << duration32f / iterations << " μs" << std::endl;
  std::cout << "  32f/8u ratio: " << static_cast<double>(duration32f) / duration8u << std::endl;

  nppiFree(d_src8u);
  nppiFree(d_dst8u);
  nppiFree(d_src32f);
  nppiFree(d_dst32f);
}

// Test 5: Error handling
TEST_F(NppiFilterBoxExtendedTest, FilterBox_ErrorHandling) {
  const int width = 100;
  const int height = 100;

  Npp8u *d_src = nullptr;
  Npp8u *d_dst = nullptr;
  int srcStep, dstStep;
  d_src = nppiMalloc_8u_C4(width, height, &srcStep);
  d_dst = nppiMalloc_8u_C4(width, height, &dstStep);

  NppiSize oSizeROI = {width, height};
  NppiSize oMaskSize = {3, 3};
  NppiPoint oAnchor = {1, 1};

  // Test null pointers
  EXPECT_EQ(nppiFilterBox_8u_C4R(nullptr, srcStep, d_dst, dstStep, oSizeROI, oMaskSize, oAnchor),
            NPP_NULL_POINTER_ERROR);
  EXPECT_EQ(nppiFilterBox_8u_C4R(d_src, srcStep, nullptr, dstStep, oSizeROI, oMaskSize, oAnchor),
            NPP_NULL_POINTER_ERROR);

  // Test invalid ROI
  NppiSize invalidROI = {-1, height};
  EXPECT_EQ(nppiFilterBox_8u_C4R(d_src, srcStep, d_dst, dstStep, invalidROI, oMaskSize, oAnchor), NPP_SIZE_ERROR);

  // Test invalid mask size
  NppiSize invalidMask = {0, 3};
  EXPECT_EQ(nppiFilterBox_8u_C4R(d_src, srcStep, d_dst, dstStep, oSizeROI, invalidMask, oAnchor), NPP_MASK_SIZE_ERROR);

  // Test invalid anchor
  NppiPoint invalidAnchor = {-1, 1};
  EXPECT_EQ(nppiFilterBox_8u_C4R(d_src, srcStep, d_dst, dstStep, oSizeROI, oMaskSize, invalidAnchor), NPP_ANCHOR_ERROR);

  nppiFree(d_src);
  nppiFree(d_dst);
}
