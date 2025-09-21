/**
 * @file test_nppiConvert_8u32f_C3R_comprehensive.cpp
 * @brief Comprehensive test suite for nppiConvert_8u32f_C3R
 *
 * Tests cover:
 * - Basic functionality
 * - Boundary conditions
 * - Memory alignment
 * - Large images
 * - ROI operations
 * - Error handling
 * - Performance scenarios
 */

#include "npp.h"
#include <chrono>
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <random>
#include <vector>

class NppiConvert8u32fC3RTest : public ::testing::Test {
protected:
  void SetUp() override {
    // Initialize
    cudaError_t err = cudaSetDevice(0);
    ASSERT_EQ(err, cudaSuccess) << "Failed to set device";
  }

  void TearDown() override {
    // Synchronize to catch any errors
    cudaError_t err = cudaDeviceSynchronize();
    EXPECT_EQ(err, cudaSuccess);
  }

  // Helper function to allocate aligned memory
  void allocateAlignedImage(Npp8u **ppSrc, Npp32f **ppDst, int width, int height, int *pSrcStep, int *pDstStep) {
    *ppSrc = nppiMalloc_8u_C3(width, height, pSrcStep);
    ASSERT_NE(*ppSrc, nullptr);
    *ppDst = nppiMalloc_32f_C3(width, height, pDstStep);
    ASSERT_NE(*ppDst, nullptr);
  }

  // Helper to verify conversion results
  bool verifyConversion(const std::vector<Npp8u> &src, const std::vector<Npp32f> &dst, int width, int height) {
    for (int i = 0; i < width * height * 3; i++) {
      if (dst[i] != static_cast<float>(src[i])) {
        return false;
      }
    }
    return true;
  }
};

// Test 1: Basic functionality
TEST_F(NppiConvert8u32fC3RTest, BasicConversion) {
  const int width = 128;
  const int height = 128;

  // Prepare test data
  std::vector<Npp8u> hostSrc(width * height * 3);
  for (int i = 0; i < width * height * 3; i++) {
    hostSrc[i] = i % 256;
  }

  // Allocate GPU memory
  Npp8u *d_src = nullptr;
  Npp32f *d_dst = nullptr;
  int srcStep, dstStep;
  allocateAlignedImage(&d_src, &d_dst, width, height, &srcStep, &dstStep);

  // Copy to GPU
  cudaMemcpy2D(d_src, srcStep, hostSrc.data(), width * 3, width * 3, height, cudaMemcpyHostToDevice);

  // Execute conversion
  NppiSize roi = {width, height};
  NppStatus status = nppiConvert_8u32f_C3R(d_src, srcStep, d_dst, dstStep, roi);
  ASSERT_EQ(status, NPP_SUCCESS);

  // Copy back and verify
  std::vector<Npp32f> hostDst(width * height * 3);
  cudaMemcpy2D(hostDst.data(), width * 3 * sizeof(Npp32f), d_dst, dstStep, width * 3 * sizeof(Npp32f), height,
               cudaMemcpyDeviceToHost);

  EXPECT_TRUE(verifyConversion(hostSrc, hostDst, width, height));

  nppiFree(d_src);
  nppiFree(d_dst);
}

// Test 2: All possible 8-bit values
TEST_F(NppiConvert8u32fC3RTest, AllPossibleValues) {
  const int width = 16;
  const int height = 16;

  std::vector<Npp8u> hostSrc(width * height * 3);
  // Fill with all possible 8-bit values
  for (int i = 0; i < 256 && i < width * height; i++) {
    hostSrc[i * 3] = i;
    hostSrc[i * 3 + 1] = i;
    hostSrc[i * 3 + 2] = i;
  }

  Npp8u *d_src = nullptr;
  Npp32f *d_dst = nullptr;
  int srcStep, dstStep;
  allocateAlignedImage(&d_src, &d_dst, width, height, &srcStep, &dstStep);

  cudaMemcpy2D(d_src, srcStep, hostSrc.data(), width * 3, width * 3, height, cudaMemcpyHostToDevice);

  NppiSize roi = {width, height};
  NppStatus status = nppiConvert_8u32f_C3R(d_src, srcStep, d_dst, dstStep, roi);
  ASSERT_EQ(status, NPP_SUCCESS);

  std::vector<Npp32f> hostDst(width * height * 3);
  cudaMemcpy2D(hostDst.data(), width * 3 * sizeof(Npp32f), d_dst, dstStep, width * 3 * sizeof(Npp32f), height,
               cudaMemcpyDeviceToHost);

  // Verify exact conversion
  for (int i = 0; i < std::min(256, width * height); i++) {
    EXPECT_EQ(hostDst[i * 3], static_cast<float>(i));
    EXPECT_EQ(hostDst[i * 3 + 1], static_cast<float>(i));
    EXPECT_EQ(hostDst[i * 3 + 2], static_cast<float>(i));
  }

  nppiFree(d_src);
  nppiFree(d_dst);
}

// Test 3: Large image handling
TEST_F(NppiConvert8u32fC3RTest, LargeImage) {
  const int width = 4096;
  const int height = 2160; // 4K resolution

  Npp8u *d_src = nullptr;
  Npp32f *d_dst = nullptr;
  int srcStep, dstStep;
  allocateAlignedImage(&d_src, &d_dst, width, height, &srcStep, &dstStep);

  // Fill with pattern
  cudaMemset(d_src, 128, srcStep * height);

  NppiSize roi = {width, height};
  NppStatus status = nppiConvert_8u32f_C3R(d_src, srcStep, d_dst, dstStep, roi);
  EXPECT_EQ(status, NPP_SUCCESS);

  // Verify a few samples
  Npp32f sample[3];
  cudaMemcpy(sample, d_dst, 3 * sizeof(Npp32f), cudaMemcpyDeviceToHost);
  EXPECT_EQ(sample[0], 128.0f);
  EXPECT_EQ(sample[1], 128.0f);
  EXPECT_EQ(sample[2], 128.0f);

  nppiFree(d_src);
  nppiFree(d_dst);
}

// Test 4: ROI operations
TEST_F(NppiConvert8u32fC3RTest, ROIOperations) {
  const int width = 256;
  const int height = 256;
  const int roiX = 64;
  const int roiY = 64;
  const int roiWidth = 128;
  const int roiHeight = 128;

  std::vector<Npp8u> hostSrc(width * height * 3);
  std::vector<Npp32f> hostDst(width * height * 3, -1.0f); // Initialize with -1

  // Create checkerboard pattern
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      int idx = (y * width + x) * 3;
      Npp8u value = ((x / 32 + y / 32) % 2) * 255;
      hostSrc[idx] = value;
      hostSrc[idx + 1] = value;
      hostSrc[idx + 2] = value;
    }
  }

  Npp8u *d_src = nullptr;
  Npp32f *d_dst = nullptr;
  int srcStep, dstStep;
  allocateAlignedImage(&d_src, &d_dst, width, height, &srcStep, &dstStep);

  cudaMemcpy2D(d_src, srcStep, hostSrc.data(), width * 3, width * 3, height, cudaMemcpyHostToDevice);
  cudaMemcpy2D(d_dst, dstStep, hostDst.data(), width * 3 * sizeof(Npp32f), width * 3 * sizeof(Npp32f), height,
               cudaMemcpyHostToDevice);

  // Apply conversion only to ROI
  Npp8u *pSrcROI = d_src + roiY * srcStep + roiX * 3;
  Npp32f *pDstROI = (Npp32f *)((Npp8u *)d_dst + roiY * dstStep) + roiX * 3;

  NppiSize roi = {roiWidth, roiHeight};
  NppStatus status = nppiConvert_8u32f_C3R(pSrcROI, srcStep, pDstROI, dstStep, roi);
  ASSERT_EQ(status, NPP_SUCCESS);

  cudaMemcpy2D(hostDst.data(), width * 3 * sizeof(Npp32f), d_dst, dstStep, width * 3 * sizeof(Npp32f), height,
               cudaMemcpyDeviceToHost);

  // Verify ROI was converted and outside ROI remains -1
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      int idx = (y * width + x) * 3;
      if (x >= roiX && x < roiX + roiWidth && y >= roiY && y < roiY + roiHeight) {
        // Inside ROI - should be converted
        EXPECT_EQ(hostDst[idx], static_cast<float>(hostSrc[idx]));
      } else {
        // Outside ROI - should remain -1
        EXPECT_EQ(hostDst[idx], -1.0f);
      }
    }
  }

  nppiFree(d_src);
  nppiFree(d_dst);
}

// Test 5: Non-standard stride
TEST_F(NppiConvert8u32fC3RTest, NonStandardStride) {
  const int width = 100;
  const int height = 100;
  const int extraBytes = 128; // Extra bytes per row

  int srcStep = width * 3 + extraBytes;
  int dstStep = width * 3 * sizeof(Npp32f) + extraBytes;

  // Allocate with custom stride
  Npp8u *d_src = nullptr;
  Npp32f *d_dst = nullptr;
  cudaMalloc(&d_src, srcStep * height);
  cudaMalloc(&d_dst, dstStep * height);
  ASSERT_NE(d_src, nullptr);
  ASSERT_NE(d_dst, nullptr);

  // Fill source with pattern
  std::vector<Npp8u> hostSrc(srcStep * height);
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width * 3; x++) {
      hostSrc[y * srcStep + x] = (y * width * 3 + x) % 256;
    }
  }

  cudaMemcpy(d_src, hostSrc.data(), srcStep * height, cudaMemcpyHostToDevice);

  NppiSize roi = {width, height};
  NppStatus status = nppiConvert_8u32f_C3R(d_src, srcStep, d_dst, dstStep, roi);
  ASSERT_EQ(status, NPP_SUCCESS);

  // Verify a sample
  std::vector<Npp32f> hostDst(dstStep * height / sizeof(Npp32f));
  cudaMemcpy(hostDst.data(), d_dst, dstStep * height, cudaMemcpyDeviceToHost);

  // Check first row
  for (int x = 0; x < width * 3; x++) {
    EXPECT_EQ(hostDst[x], static_cast<float>(hostSrc[x]));
  }

  cudaFree(d_src);
  cudaFree(d_dst);
}

// Test 6: Performance with stream context
TEST_F(NppiConvert8u32fC3RTest, StreamContextPerformance) {
  const int width = 1920;
  const int height = 1080;
  const int iterations = 100;

  Npp8u *d_src = nullptr;
  Npp32f *d_dst = nullptr;
  int srcStep, dstStep;
  allocateAlignedImage(&d_src, &d_dst, width, height, &srcStep, &dstStep);

  // Create stream
  cudaStream_t stream;
  cudaStreamCreate(&stream);

  NppStreamContext nppStreamCtx;
  nppGetStreamContext(&nppStreamCtx);
  nppStreamCtx.hStream = stream;

  NppiSize roi = {width, height};

  // Warm up
  nppiConvert_8u32f_C3R_Ctx(d_src, srcStep, d_dst, dstStep, roi, nppStreamCtx);
  cudaStreamSynchronize(stream);

  // Measure performance
  auto start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < iterations; i++) {
    nppiConvert_8u32f_C3R_Ctx(d_src, srcStep, d_dst, dstStep, roi, nppStreamCtx);
  }
  cudaStreamSynchronize(stream);
  auto end = std::chrono::high_resolution_clock::now();

  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
  double avgTime = duration / (double)iterations;

  // Calculate throughput (MB/s)
  double dataSize = width * height * 3 * (1 + 4) / 1024.0 / 1024.0; // Input + output
  double throughput = dataSize * 1000000.0 / avgTime;

  std::cout << "Average time per conversion: " << avgTime << " us" << std::endl;
  std::cout << "Throughput: " << throughput << " MB/s" << std::endl;

  cudaStreamDestroy(stream);
  nppiFree(d_src);
  nppiFree(d_dst);
}

// Test 7: Error handling
TEST_F(NppiConvert8u32fC3RTest, DISABLED_ErrorHandling) {
  const int width = 64;
  const int height = 64;

  Npp8u *d_src = nullptr;
  Npp32f *d_dst = nullptr;
  int srcStep, dstStep;
  allocateAlignedImage(&d_src, &d_dst, width, height, &srcStep, &dstStep);

  NppiSize roi = {width, height};

  // Test null pointers
  EXPECT_EQ(nppiConvert_8u32f_C3R(nullptr, srcStep, d_dst, dstStep, roi), NPP_NULL_POINTER_ERROR);
  EXPECT_EQ(nppiConvert_8u32f_C3R(d_src, srcStep, nullptr, dstStep, roi), NPP_NULL_POINTER_ERROR);

  // Test invalid ROI
  NppiSize invalidRoi = {0, height};
  EXPECT_EQ(nppiConvert_8u32f_C3R(d_src, srcStep, d_dst, dstStep, invalidRoi), NPP_SUCCESS);

  invalidRoi = {width, -1};
  EXPECT_EQ(nppiConvert_8u32f_C3R(d_src, srcStep, d_dst, dstStep, invalidRoi), NPP_SIZE_ERROR);

  // Test invalid step
  EXPECT_EQ(nppiConvert_8u32f_C3R(d_src, 0, d_dst, dstStep, roi), NPP_SUCCESS);
  EXPECT_EQ(nppiConvert_8u32f_C3R(d_src, srcStep, d_dst, 0, roi), NPP_STEP_ERROR);

  nppiFree(d_src);
  nppiFree(d_dst);
}

// Test 8: Memory alignment edge cases
TEST_F(NppiConvert8u32fC3RTest, MemoryAlignmentEdgeCases) {
  // Test various widths that might cause alignment issues
  const int testWidths[] = {1, 3, 7, 15, 31, 63, 127, 255, 333, 1023};
  const int height = 16;

  for (int width : testWidths) {
    Npp8u *d_src = nullptr;
    Npp32f *d_dst = nullptr;
    int srcStep, dstStep;

    d_src = nppiMalloc_8u_C3(width, height, &srcStep);
    d_dst = nppiMalloc_32f_C3(width, height, &dstStep);

    ASSERT_NE(d_src, nullptr) << "Failed to allocate for width " << width;
    ASSERT_NE(d_dst, nullptr) << "Failed to allocate for width " << width;

    // Fill with test pattern
    cudaMemset(d_src, 42, srcStep * height);

    NppiSize roi = {width, height};
    NppStatus status = nppiConvert_8u32f_C3R(d_src, srcStep, d_dst, dstStep, roi);
    EXPECT_EQ(status, NPP_SUCCESS) << "Failed for width " << width;

    // Verify first pixel
    Npp32f firstPixel[3];
    cudaMemcpy(firstPixel, d_dst, 3 * sizeof(Npp32f), cudaMemcpyDeviceToHost);
    EXPECT_EQ(firstPixel[0], 42.0f) << "Wrong value for width " << width;
    EXPECT_EQ(firstPixel[1], 42.0f) << "Wrong value for width " << width;
    EXPECT_EQ(firstPixel[2], 42.0f) << "Wrong value for width " << width;

    nppiFree(d_src);
    nppiFree(d_dst);
  }
}

// Test 9: Concurrent execution
TEST_F(NppiConvert8u32fC3RTest, ConcurrentExecution) {
  const int numStreams = 4;
  const int width = 512;
  const int height = 512;

  std::vector<cudaStream_t> streams(numStreams);
  std::vector<Npp8u *> srcBuffers(numStreams);
  std::vector<Npp32f *> dstBuffers(numStreams);
  std::vector<int> srcSteps(numStreams);
  std::vector<int> dstSteps(numStreams);

  // Create streams and allocate buffers
  for (int i = 0; i < numStreams; i++) {
    cudaStreamCreate(&streams[i]);
    srcBuffers[i] = nppiMalloc_8u_C3(width, height, &srcSteps[i]);
    dstBuffers[i] = nppiMalloc_32f_C3(width, height, &dstSteps[i]);
    ASSERT_NE(srcBuffers[i], nullptr);
    ASSERT_NE(dstBuffers[i], nullptr);

    // Initialize with different values
    cudaMemset(srcBuffers[i], i * 50, srcSteps[i] * height);
  }

  // Launch conversions concurrently
  NppiSize roi = {width, height};
  for (int i = 0; i < numStreams; i++) {
    NppStreamContext ctx;
    nppGetStreamContext(&ctx);
    ctx.hStream = streams[i];

    NppStatus status = nppiConvert_8u32f_C3R_Ctx(srcBuffers[i], srcSteps[i], dstBuffers[i], dstSteps[i], roi, ctx);
    EXPECT_EQ(status, NPP_SUCCESS);
  }

  // Wait for all to complete
  for (int i = 0; i < numStreams; i++) {
    cudaStreamSynchronize(streams[i]);
  }

  // Verify results
  for (int i = 0; i < numStreams; i++) {
    Npp32f sample[3];
    cudaMemcpy(sample, dstBuffers[i], 3 * sizeof(Npp32f), cudaMemcpyDeviceToHost);
    float expected = static_cast<float>(i * 50);
    EXPECT_EQ(sample[0], expected) << "Stream " << i << " failed";
    EXPECT_EQ(sample[1], expected) << "Stream " << i << " failed";
    EXPECT_EQ(sample[2], expected) << "Stream " << i << " failed";
  }

  // Cleanup
  for (int i = 0; i < numStreams; i++) {
    cudaStreamDestroy(streams[i]);
    nppiFree(srcBuffers[i]);
    nppiFree(dstBuffers[i]);
  }
}

// Test 10: Extreme dimensions
TEST_F(NppiConvert8u32fC3RTest, ExtremeDimensions) {
  // Test very wide but short image
  {
    const int width = 16384;
    const int height = 2;

    Npp8u *d_src = nullptr;
    Npp32f *d_dst = nullptr;
    int srcStep, dstStep;

    d_src = nppiMalloc_8u_C3(width, height, &srcStep);
    d_dst = nppiMalloc_32f_C3(width, height, &dstStep);

    if (d_src && d_dst) {
      NppiSize roi = {width, height};
      NppStatus status = nppiConvert_8u32f_C3R(d_src, srcStep, d_dst, dstStep, roi);
      EXPECT_EQ(status, NPP_SUCCESS);

      nppiFree(d_src);
      nppiFree(d_dst);
    }
  }

  // Test very tall but narrow image
  {
    const int width = 2;
    const int height = 16384;

    Npp8u *d_src = nullptr;
    Npp32f *d_dst = nullptr;
    int srcStep, dstStep;

    d_src = nppiMalloc_8u_C3(width, height, &srcStep);
    d_dst = nppiMalloc_32f_C3(width, height, &dstStep);

    if (d_src && d_dst) {
      NppiSize roi = {width, height};
      NppStatus status = nppiConvert_8u32f_C3R(d_src, srcStep, d_dst, dstStep, roi);
      EXPECT_EQ(status, NPP_SUCCESS);

      nppiFree(d_src);
      nppiFree(d_dst);
    }
  }
}

// Test 11: Gradient Pattern Conversion
TEST_F(NppiConvert8u32fC3RTest, GradientPatternConversion) {
  const int width = 64, height = 64;
  
  Npp8u *d_src = nullptr;
  Npp32f *d_dst = nullptr;
  int srcStep, dstStep;
  allocateAlignedImage(&d_src, &d_dst, width, height, &srcStep, &dstStep);
  
  // Create gradient pattern
  std::vector<Npp8u> hostSrc(width * height * 3);
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      int idx = (y * width + x) * 3;
      hostSrc[idx] = static_cast<Npp8u>((x + y) % 256);     // R channel
      hostSrc[idx + 1] = static_cast<Npp8u>((x * 2) % 256); // G channel
      hostSrc[idx + 2] = static_cast<Npp8u>((y * 2) % 256); // B channel
    }
  }
  
  cudaMemcpy2D(d_src, srcStep, hostSrc.data(), width * 3, width * 3, height, cudaMemcpyHostToDevice);
  
  NppiSize roi = {width, height};
  NppStatus status = nppiConvert_8u32f_C3R(d_src, srcStep, d_dst, dstStep, roi);
  ASSERT_EQ(status, NPP_SUCCESS);
  
  std::vector<Npp32f> hostDst(width * height * 3);
  cudaMemcpy2D(hostDst.data(), width * 3 * sizeof(Npp32f), d_dst, dstStep, 
               width * 3 * sizeof(Npp32f), height, cudaMemcpyDeviceToHost);
  
  // Verify gradient conversion
  EXPECT_TRUE(verifyConversion(hostSrc, hostDst, width, height));
  
  nppiFree(d_src);
  nppiFree(d_dst);
}

// Test 12: Channel Independence Verification
TEST_F(NppiConvert8u32fC3RTest, ChannelIndependenceVerification) {
  const int width = 32, height = 32;
  
  Npp8u *d_src = nullptr;
  Npp32f *d_dst = nullptr;
  int srcStep, dstStep;
  allocateAlignedImage(&d_src, &d_dst, width, height, &srcStep, &dstStep);
  
  // Create data where each channel has distinct values
  std::vector<Npp8u> hostSrc(width * height * 3);
  for (int i = 0; i < width * height; i++) {
    hostSrc[i * 3] = 100;     // R channel constant
    hostSrc[i * 3 + 1] = 150; // G channel constant
    hostSrc[i * 3 + 2] = 200; // B channel constant
  }
  
  cudaMemcpy2D(d_src, srcStep, hostSrc.data(), width * 3, width * 3, height, cudaMemcpyHostToDevice);
  
  NppiSize roi = {width, height};
  NppStatus status = nppiConvert_8u32f_C3R(d_src, srcStep, d_dst, dstStep, roi);
  ASSERT_EQ(status, NPP_SUCCESS);
  
  std::vector<Npp32f> hostDst(width * height * 3);
  cudaMemcpy2D(hostDst.data(), width * 3 * sizeof(Npp32f), d_dst, dstStep,
               width * 3 * sizeof(Npp32f), height, cudaMemcpyDeviceToHost);
  
  // Verify each channel maintains its distinct value
  for (int i = 0; i < width * height; i++) {
    EXPECT_FLOAT_EQ(hostDst[i * 3], 100.0f) << "R channel mismatch at pixel " << i;
    EXPECT_FLOAT_EQ(hostDst[i * 3 + 1], 150.0f) << "G channel mismatch at pixel " << i;
    EXPECT_FLOAT_EQ(hostDst[i * 3 + 2], 200.0f) << "B channel mismatch at pixel " << i;
  }
  
  nppiFree(d_src);
  nppiFree(d_dst);
}