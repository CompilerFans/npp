// Test suite for morphology operations

#include "npp.h"
#include <algorithm>
#include <chrono>
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <iostream>
#include <vector>

class NppiMorphologyExtendedTest : public ::testing::Test {
protected:
  void SetUp() override {
    cudaError_t err = cudaSetDevice(0);
    ASSERT_EQ(err, cudaSuccess);
  }

  void TearDown() override {
    cudaError_t err = cudaDeviceSynchronize();
    EXPECT_EQ(err, cudaSuccess);
  }

  // Create 3x3 cross structure element
  std::vector<Npp8u> createCrossKernel() { return {0, 1, 0, 1, 1, 1, 0, 1, 0}; }

  // Create 3x3 box structure element
  std::vector<Npp8u> createBoxKernel() { return {1, 1, 1, 1, 1, 1, 1, 1, 1}; }

  // Create 5x5 circle structure element
  std::vector<Npp8u> createCircleKernel() {
    return {0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0};
  }
};

// Test 1: Basic erosion for 8u_C1R with cross kernel
TEST_F(NppiMorphologyExtendedTest, Erode_8u_C1R_CrossKernel) {
  const int width = 16;
  const int height = 16;

  // Create test image with a white square in the center
  std::vector<Npp8u> hostSrc(width * height, 0);
  for (int y = 4; y < 12; y++) {
    for (int x = 4; x < 12; x++) {
      hostSrc[y * width + x] = 255;
    }
  }

  // Allocate GPU memory
  Npp8u *d_src = nullptr, *d_dst = nullptr;
  int srcStep, dstStep;
  d_src = nppiMalloc_8u_C1(width, height, &srcStep);
  d_dst = nppiMalloc_8u_C1(width, height, &dstStep);
  ASSERT_NE(d_src, nullptr);
  ASSERT_NE(d_dst, nullptr);

  // Upload source data
  cudaMemcpy2D(d_src, srcStep, hostSrc.data(), width, width, height, cudaMemcpyHostToDevice);

  // Create structure element
  auto kernel = createCrossKernel();
  Npp8u *d_mask = nullptr;
  cudaMalloc(&d_mask, kernel.size());
  cudaMemcpy(d_mask, kernel.data(), kernel.size(), cudaMemcpyHostToDevice);

  NppiSize oSizeROI = {width, height};
  NppiSize oMaskSize = {3, 3};
  NppiPoint oAnchor = {1, 1};

  // Perform erosion
  NppStatus status = nppiErode_8u_C1R(d_src, srcStep, d_dst, dstStep, oSizeROI, d_mask, oMaskSize, oAnchor);
  ASSERT_EQ(status, NPP_SUCCESS);

  // Copy result back to host
  std::vector<Npp8u> hostDst(width * height);
  cudaMemcpy2D(hostDst.data(), width, d_dst, dstStep, width, height, cudaMemcpyDeviceToHost);

  // Verify erosion occurred - the white square should be smaller
  bool foundWhiteInCenter = false;
  bool foundZeroAtEdge = false;
  for (int y = 5; y < 11; y++) {
    for (int x = 5; x < 11; x++) {
      if (hostDst[y * width + x] == 255) {
        foundWhiteInCenter = true;
      }
    }
  }
  // Check edge of original square should now be black
  if (hostDst[4 * width + 4] == 0) {
    foundZeroAtEdge = true;
  }

  EXPECT_TRUE(foundWhiteInCenter);
  EXPECT_TRUE(foundZeroAtEdge);

  // Cleanup
  nppiFree(d_src);
  nppiFree(d_dst);
  cudaFree(d_mask);
}

// Test 2: Basic dilation for 8u_C4R with box kernel
TEST_F(NppiMorphologyExtendedTest, Dilate_8u_C4R_BoxKernel) {
  const int width = 16;
  const int height = 16;

  // Create test image with a small white square in the center (4 channels)
  std::vector<Npp8u> hostSrc(width * height * 4, 0);
  for (int y = 6; y < 10; y++) {
    for (int x = 6; x < 10; x++) {
      for (int c = 0; c < 4; c++) {
        hostSrc[(y * width + x) * 4 + c] = 255;
      }
    }
  }

  // Allocate GPU memory
  Npp8u *d_src = nullptr, *d_dst = nullptr;
  int srcStep, dstStep;
  d_src = nppiMalloc_8u_C4(width, height, &srcStep);
  d_dst = nppiMalloc_8u_C4(width, height, &dstStep);
  ASSERT_NE(d_src, nullptr);
  ASSERT_NE(d_dst, nullptr);

  // Upload source data
  cudaMemcpy2D(d_src, srcStep, hostSrc.data(), width * 4, width * 4, height, cudaMemcpyHostToDevice);

  // Create structure element
  auto kernel = createBoxKernel();
  Npp8u *d_mask = nullptr;
  cudaMalloc(&d_mask, kernel.size());
  cudaMemcpy(d_mask, kernel.data(), kernel.size(), cudaMemcpyHostToDevice);

  NppiSize oSizeROI = {width, height};
  NppiSize oMaskSize = {3, 3};
  NppiPoint oAnchor = {1, 1};

  // Perform dilation
  NppStatus status = nppiDilate_8u_C4R(d_src, srcStep, d_dst, dstStep, oSizeROI, d_mask, oMaskSize, oAnchor);
  ASSERT_EQ(status, NPP_SUCCESS);

  // Copy result back to host
  std::vector<Npp8u> hostDst(width * height * 4);
  cudaMemcpy2D(hostDst.data(), width * 4, d_dst, dstStep, width * 4, height, cudaMemcpyDeviceToHost);

  // Verify dilation occurred - the white area should be larger
  bool foundExpandedWhite = false;
  // Check if dilation expanded to (5,5) which was originally black
  for (int c = 0; c < 4; c++) {
    if (hostDst[(5 * width + 5) * 4 + c] == 255) {
      foundExpandedWhite = true;
      break;
    }
  }

  EXPECT_TRUE(foundExpandedWhite);

  // Cleanup
  nppiFree(d_src);
  nppiFree(d_dst);
  cudaFree(d_mask);
}

// Test 3: Erosion and dilation with 32f_C1R
TEST_F(NppiMorphologyExtendedTest, Erode_Dilate_32f_C1R_Comparison) {
  const int width = 12;
  const int height = 12;

  // Create test image with gradient values
  std::vector<Npp32f> hostSrc(width * height);
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      hostSrc[y * width + x] = static_cast<float>(x + y * 10);
    }
  }

  // Allocate GPU memory
  Npp32f *d_src = nullptr, *d_eroded = nullptr, *d_dilated = nullptr;
  int srcStep, erodedStep, dilatedStep;
  d_src = nppiMalloc_32f_C1(width, height, &srcStep);
  d_eroded = nppiMalloc_32f_C1(width, height, &erodedStep);
  d_dilated = nppiMalloc_32f_C1(width, height, &dilatedStep);
  ASSERT_NE(d_src, nullptr);
  ASSERT_NE(d_eroded, nullptr);
  ASSERT_NE(d_dilated, nullptr);

  // Upload source data
  cudaMemcpy2D(d_src, srcStep, hostSrc.data(), width * sizeof(Npp32f), width * sizeof(Npp32f), height,
               cudaMemcpyHostToDevice);

  // Create structure element
  auto kernel = createCrossKernel();
  Npp8u *d_mask = nullptr;
  cudaMalloc(&d_mask, kernel.size());
  cudaMemcpy(d_mask, kernel.data(), kernel.size(), cudaMemcpyHostToDevice);

  NppiSize oSizeROI = {width, height};
  NppiSize oMaskSize = {3, 3};
  NppiPoint oAnchor = {1, 1};

  // Perform erosion and dilation
  NppStatus status1 = nppiErode_32f_C1R(d_src, srcStep, d_eroded, erodedStep, oSizeROI, d_mask, oMaskSize, oAnchor);
  NppStatus status2 = nppiDilate_32f_C1R(d_src, srcStep, d_dilated, dilatedStep, oSizeROI, d_mask, oMaskSize, oAnchor);
  ASSERT_EQ(status1, NPP_SUCCESS);
  ASSERT_EQ(status2, NPP_SUCCESS);

  // Copy results back to host
  std::vector<Npp32f> hostEroded(width * height);
  std::vector<Npp32f> hostDilated(width * height);
  cudaMemcpy2D(hostEroded.data(), width * sizeof(Npp32f), d_eroded, erodedStep, width * sizeof(Npp32f), height,
               cudaMemcpyDeviceToHost);
  cudaMemcpy2D(hostDilated.data(), width * sizeof(Npp32f), d_dilated, dilatedStep, width * sizeof(Npp32f), height,
               cudaMemcpyDeviceToHost);

  // Verify morphological properties: eroded <= original <= dilated
  bool morphologyValid = true;
  for (int i = 1; i < (width - 1) * (height - 1); i++) {
    if (hostEroded[i] > hostSrc[i] || hostSrc[i] > hostDilated[i]) {
      morphologyValid = false;
      break;
    }
  }

  EXPECT_TRUE(morphologyValid);

  // Cleanup
  nppiFree(d_src);
  nppiFree(d_eroded);
  nppiFree(d_dilated);
  cudaFree(d_mask);
}

// Test 4: Stream context test for 32f_C4R
TEST_F(NppiMorphologyExtendedTest, Erode_32f_C4R_StreamContext) {
  const int width = 20;
  const int height = 20;

  // Create test image (4 channels)
  std::vector<Npp32f> hostSrc(width * height * 4);
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      for (int c = 0; c < 4; c++) {
        hostSrc[(y * width + x) * 4 + c] = static_cast<float>(100 + c * 10);
      }
    }
  }

  // Allocate GPU memory
  Npp32f *d_src = nullptr, *d_dst = nullptr;
  int srcStep, dstStep;
  d_src = nppiMalloc_32f_C4(width, height, &srcStep);
  d_dst = nppiMalloc_32f_C4(width, height, &dstStep);
  ASSERT_NE(d_src, nullptr);
  ASSERT_NE(d_dst, nullptr);

  // Create stream
  cudaStream_t stream;
  cudaStreamCreate(&stream);
  NppStreamContext nppStreamCtx;
  nppGetStreamContext(&nppStreamCtx);
  nppStreamCtx.hStream = stream;

  // Upload data asynchronously
  cudaMemcpy2DAsync(d_src, srcStep, hostSrc.data(), width * 4 * sizeof(Npp32f), width * 4 * sizeof(Npp32f), height,
                    cudaMemcpyHostToDevice, stream);

  // Create structure element
  auto kernel = createCircleKernel();
  Npp8u *d_mask = nullptr;
  cudaMalloc(&d_mask, kernel.size());
  cudaMemcpyAsync(d_mask, kernel.data(), kernel.size(), cudaMemcpyHostToDevice, stream);

  NppiSize oSizeROI = {width, height};
  NppiSize oMaskSize = {5, 5};
  NppiPoint oAnchor = {2, 2};

  // Perform erosion with stream context
  NppStatus status =
      nppiErode_32f_C4R_Ctx(d_src, srcStep, d_dst, dstStep, oSizeROI, d_mask, oMaskSize, oAnchor, nppStreamCtx);
  ASSERT_EQ(status, NPP_SUCCESS);

  // Synchronize stream
  cudaStreamSynchronize(stream);

  // Copy result back to host
  std::vector<Npp32f> hostDst(width * height * 4);
  cudaMemcpy2D(hostDst.data(), width * 4 * sizeof(Npp32f), d_dst, dstStep, width * 4 * sizeof(Npp32f), height,
               cudaMemcpyDeviceToHost);

  // Basic verification - results should not be all zeros
  bool hasNonZeroResults = false;
  for (const auto &val : hostDst) {
    if (val > 0.0f) {
      hasNonZeroResults = true;
      break;
    }
  }

  EXPECT_TRUE(hasNonZeroResults);

  // Cleanup
  cudaStreamDestroy(stream);
  nppiFree(d_src);
  nppiFree(d_dst);
  cudaFree(d_mask);
}

// Test 5: Error handling
TEST_F(NppiMorphologyExtendedTest, Morphology_ErrorHandling) {
  NppiSize oSizeROI = {100, 100};
  NppiSize oMaskSize = {3, 3};
  NppiPoint oAnchor = {1, 1};

  // Allocate minimal valid data for testing
  Npp8u *d_src = nullptr, *d_dst = nullptr, *d_mask = nullptr;
  int srcStep, dstStep;
  d_src = nppiMalloc_8u_C1(100, 100, &srcStep);
  d_dst = nppiMalloc_8u_C1(100, 100, &dstStep);
  cudaMalloc(&d_mask, 9);

  // Test null pointer errors
  EXPECT_EQ(nppiErode_8u_C1R(nullptr, srcStep, d_dst, dstStep, oSizeROI, d_mask, oMaskSize, oAnchor),
            NPP_NULL_POINTER_ERROR);
  EXPECT_EQ(nppiErode_8u_C1R(d_src, srcStep, nullptr, dstStep, oSizeROI, d_mask, oMaskSize, oAnchor),
            NPP_NULL_POINTER_ERROR);
  EXPECT_EQ(nppiErode_8u_C1R(d_src, srcStep, d_dst, dstStep, oSizeROI, nullptr, oMaskSize, oAnchor),
            NPP_NULL_POINTER_ERROR);

  // Test invalid ROI size
  NppiSize invalidROI = {-1, 100};
  EXPECT_EQ(nppiErode_8u_C1R(d_src, srcStep, d_dst, dstStep, invalidROI, d_mask, oMaskSize, oAnchor), NPP_SIZE_ERROR);

  // Test invalid mask size
  NppiSize invalidMaskSize = {0, 3};
  EXPECT_EQ(nppiErode_8u_C1R(d_src, srcStep, d_dst, dstStep, oSizeROI, d_mask, invalidMaskSize, oAnchor),
            NPP_MASK_SIZE_ERROR);

  // Test invalid anchor
  NppiPoint invalidAnchor = {-1, 1};
  EXPECT_EQ(nppiErode_8u_C1R(d_src, srcStep, d_dst, dstStep, oSizeROI, d_mask, oMaskSize, invalidAnchor),
            NPP_ANCHOR_ERROR);

  invalidAnchor = {3, 1}; // Anchor outside mask bounds
  EXPECT_EQ(nppiErode_8u_C1R(d_src, srcStep, d_dst, dstStep, oSizeROI, d_mask, oMaskSize, invalidAnchor),
            NPP_ANCHOR_ERROR);

  // Cleanup
  nppiFree(d_src);
  nppiFree(d_dst);
  cudaFree(d_mask);
}

// Test 6: Performance comparison between different data types and operations
TEST_F(NppiMorphologyExtendedTest, Morphology_PerformanceComparison) {
  const int width = 512;
  const int height = 512;
  const int iterations = 5;

  // Create test data
  std::vector<Npp8u> hostSrc8u(width * height, 128);
  std::vector<Npp32f> hostSrc32f(width * height, 128.0f);

  // Allocate GPU memory
  Npp8u *d_src8u = nullptr, *d_dst8u = nullptr;
  Npp32f *d_src32f = nullptr, *d_dst32f = nullptr;
  int srcStep8u, dstStep8u, srcStep32f, dstStep32f;

  d_src8u = nppiMalloc_8u_C1(width, height, &srcStep8u);
  d_dst8u = nppiMalloc_8u_C1(width, height, &dstStep8u);
  d_src32f = nppiMalloc_32f_C1(width, height, &srcStep32f);
  d_dst32f = nppiMalloc_32f_C1(width, height, &dstStep32f);

  // Upload data
  cudaMemcpy2D(d_src8u, srcStep8u, hostSrc8u.data(), width, width, height, cudaMemcpyHostToDevice);
  cudaMemcpy2D(d_src32f, srcStep32f, hostSrc32f.data(), width * sizeof(Npp32f), width * sizeof(Npp32f), height,
               cudaMemcpyHostToDevice);

  // Create structure element
  auto kernel = createBoxKernel();
  Npp8u *d_mask = nullptr;
  cudaMalloc(&d_mask, kernel.size());
  cudaMemcpy(d_mask, kernel.data(), kernel.size(), cudaMemcpyHostToDevice);

  NppiSize oSizeROI = {width, height};
  NppiSize oMaskSize = {3, 3};
  NppiPoint oAnchor = {1, 1};

  // Warm up and benchmark 8u erosion
  nppiErode_8u_C1R(d_src8u, srcStep8u, d_dst8u, dstStep8u, oSizeROI, d_mask, oMaskSize, oAnchor);
  cudaDeviceSynchronize();

  auto start8u = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < iterations; i++) {
    nppiErode_8u_C1R(d_src8u, srcStep8u, d_dst8u, dstStep8u, oSizeROI, d_mask, oMaskSize, oAnchor);
  }
  cudaDeviceSynchronize();
  auto end8u = std::chrono::high_resolution_clock::now();

  // Warm up and benchmark 32f erosion
  nppiErode_32f_C1R(d_src32f, srcStep32f, d_dst32f, dstStep32f, oSizeROI, d_mask, oMaskSize, oAnchor);
  cudaDeviceSynchronize();

  auto start32f = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < iterations; i++) {
    nppiErode_32f_C1R(d_src32f, srcStep32f, d_dst32f, dstStep32f, oSizeROI, d_mask, oMaskSize, oAnchor);
  }
  cudaDeviceSynchronize();
  auto end32f = std::chrono::high_resolution_clock::now();

  auto duration8u = std::chrono::duration_cast<std::chrono::microseconds>(end8u - start8u).count();
  auto duration32f = std::chrono::duration_cast<std::chrono::microseconds>(end32f - start32f).count();

  std::cout << "Morphology Performance Comparison:" << std::endl;
  std::cout << "  Erode 8u_C1R average time: " << duration8u / iterations << " μs" << std::endl;
  std::cout << "  Erode 32f_C1R average time: " << duration32f / iterations << " μs" << std::endl;

  // Cleanup
  nppiFree(d_src8u);
  nppiFree(d_dst8u);
  nppiFree(d_src32f);
  nppiFree(d_dst32f);
  cudaFree(d_mask);
}

// Test 7: Large kernel test
TEST_F(NppiMorphologyExtendedTest, Morphology_LargeKernel) {
  const int width = 64;
  const int height = 64;

  // Create test image
  std::vector<Npp8u> hostSrc(width * height, 100);
  // Add a bright region in the center
  for (int y = 25; y < 40; y++) {
    for (int x = 25; x < 40; x++) {
      hostSrc[y * width + x] = 255;
    }
  }

  // Allocate GPU memory
  Npp8u *d_src = nullptr, *d_dst = nullptr;
  int srcStep, dstStep;
  d_src = nppiMalloc_8u_C1(width, height, &srcStep);
  d_dst = nppiMalloc_8u_C1(width, height, &dstStep);

  // Upload source data
  cudaMemcpy2D(d_src, srcStep, hostSrc.data(), width, width, height, cudaMemcpyHostToDevice);

  // Create a large 7x7 structure element
  std::vector<Npp8u> largeKernel(49, 1); // 7x7 all ones
  Npp8u *d_mask = nullptr;
  cudaMalloc(&d_mask, largeKernel.size());
  cudaMemcpy(d_mask, largeKernel.data(), largeKernel.size(), cudaMemcpyHostToDevice);

  NppiSize oSizeROI = {width, height};
  NppiSize oMaskSize = {7, 7};
  NppiPoint oAnchor = {3, 3};

  // Perform dilation with large kernel
  NppStatus status = nppiDilate_8u_C1R(d_src, srcStep, d_dst, dstStep, oSizeROI, d_mask, oMaskSize, oAnchor);
  ASSERT_EQ(status, NPP_SUCCESS);

  // Copy result back to host
  std::vector<Npp8u> hostDst(width * height);
  cudaMemcpy2D(hostDst.data(), width, d_dst, dstStep, width, height, cudaMemcpyDeviceToHost);

  // Verify that dilation expanded the bright region
  bool foundExpansion = false;
  // Check a position that should be affected by 7x7 dilation
  if (hostDst[22 * width + 22] > 100) {
    foundExpansion = true;
  }

  EXPECT_TRUE(foundExpansion);

  // Cleanup
  nppiFree(d_src);
  nppiFree(d_dst);
  cudaFree(d_mask);
}