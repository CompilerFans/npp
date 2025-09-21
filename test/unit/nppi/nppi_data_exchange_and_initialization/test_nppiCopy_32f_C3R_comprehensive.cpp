// Implementation file

#include "npp.h"
#include <algorithm>
#include <chrono>
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <random>
#include <thread>
#include <vector>

class NppiCopy32fC3RTest : public ::testing::Test {
protected:
  void SetUp() override {
    cudaError_t err = cudaSetDevice(0);
    ASSERT_EQ(err, cudaSuccess);
  }

  void TearDown() override {
    cudaError_t err = cudaDeviceSynchronize();
    EXPECT_EQ(err, cudaSuccess);
  }

  // Helper to allocate aligned memory
  void allocateImage(Npp32f **ppImg, int width, int height, int *pStep) {
    *ppImg = nppiMalloc_32f_C3(width, height, pStep);
    ASSERT_NE(*ppImg, nullptr);
  }

  // Helper to verify data integrity
  bool verifyData(const std::vector<Npp32f> &expected, const std::vector<Npp32f> &actual, float tolerance = 0.0f) {
    if (expected.size() != actual.size())
      return false;

    for (size_t i = 0; i < expected.size(); i++) {
      if (std::abs(expected[i] - actual[i]) > tolerance) {
        return false;
      }
    }
    return true;
  }

  // Generate test pattern for 3-channel data
  void generatePattern(std::vector<Npp32f> &data, int width, int height, float scale = 1.0f) {
    data.resize(width * height * 3);
    for (int y = 0; y < height; y++) {
      for (int x = 0; x < width; x++) {
        int idx = (y * width + x) * 3;
        data[idx] = ((y * width + x) % 1000) * scale;            // R
        data[idx + 1] = ((y * width + x) % 1000) * scale + 0.1f; // G
        data[idx + 2] = ((y * width + x) % 1000) * scale + 0.2f; // B
      }
    }
  }
};

// Test 1: Basic 3-channel copy functionality
TEST_F(NppiCopy32fC3RTest, Basic3ChannelCopy) {
  const int width = 256;
  const int height = 256;

  std::vector<Npp32f> hostSrc(width * height * 3);
  generatePattern(hostSrc, width, height);

  Npp32f *d_src = nullptr;
  Npp32f *d_dst = nullptr;
  int srcStep, dstStep;
  allocateImage(&d_src, width, height, &srcStep);
  allocateImage(&d_dst, width, height, &dstStep);

  // Upload source data
  cudaMemcpy2D(d_src, srcStep, hostSrc.data(), width * 3 * sizeof(Npp32f), width * 3 * sizeof(Npp32f), height,
               cudaMemcpyHostToDevice);

  // Perform copy
  NppiSize roi = {width, height};
  NppStatus status = nppiCopy_32f_C3R(d_src, srcStep, d_dst, dstStep, roi);
  ASSERT_EQ(status, NPP_SUCCESS);

  // Verify result
  std::vector<Npp32f> hostDst(width * height * 3);
  cudaMemcpy2D(hostDst.data(), width * 3 * sizeof(Npp32f), d_dst, dstStep, width * 3 * sizeof(Npp32f), height,
               cudaMemcpyDeviceToHost);

  EXPECT_TRUE(verifyData(hostSrc, hostDst));

  nppiFree(d_src);
  nppiFree(d_dst);
}

// Test 2: Partial ROI copy with 3 channels
TEST_F(NppiCopy32fC3RTest, PartialROI3ChannelCopy) {
  const int width = 512;
  const int height = 512;
  const int roiX = 100;
  const int roiY = 100;
  const int roiWidth = 300;
  const int roiHeight = 300;

  std::vector<Npp32f> hostSrc(width * height * 3);
  std::vector<Npp32f> hostDst(width * height * 3, -999.0f);
  generatePattern(hostSrc, width, height);

  Npp32f *d_src = nullptr;
  Npp32f *d_dst = nullptr;
  int srcStep, dstStep;
  allocateImage(&d_src, width, height, &srcStep);
  allocateImage(&d_dst, width, height, &dstStep);

  // Upload data
  cudaMemcpy2D(d_src, srcStep, hostSrc.data(), width * 3 * sizeof(Npp32f), width * 3 * sizeof(Npp32f), height,
               cudaMemcpyHostToDevice);
  cudaMemcpy2D(d_dst, dstStep, hostDst.data(), width * 3 * sizeof(Npp32f), width * 3 * sizeof(Npp32f), height,
               cudaMemcpyHostToDevice);

  // Copy ROI
  Npp32f *pSrcROI = (Npp32f *)((Npp8u *)d_src + roiY * srcStep) + roiX * 3;
  Npp32f *pDstROI = (Npp32f *)((Npp8u *)d_dst + roiY * dstStep) + roiX * 3;

  NppiSize roi = {roiWidth, roiHeight};
  NppStatus status = nppiCopy_32f_C3R(pSrcROI, srcStep, pDstROI, dstStep, roi);
  ASSERT_EQ(status, NPP_SUCCESS);

  // Verify result
  cudaMemcpy2D(hostDst.data(), width * 3 * sizeof(Npp32f), d_dst, dstStep, width * 3 * sizeof(Npp32f), height,
               cudaMemcpyDeviceToHost);

  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      int idx = (y * width + x) * 3;
      if (x >= roiX && x < roiX + roiWidth && y >= roiY && y < roiY + roiHeight) {
        // Inside ROI - all 3 channels should be copied
        EXPECT_FLOAT_EQ(hostDst[idx], hostSrc[idx]);
        EXPECT_FLOAT_EQ(hostDst[idx + 1], hostSrc[idx + 1]);
        EXPECT_FLOAT_EQ(hostDst[idx + 2], hostSrc[idx + 2]);
      } else {
        // Outside ROI - should remain -999
        EXPECT_FLOAT_EQ(hostDst[idx], -999.0f);
        EXPECT_FLOAT_EQ(hostDst[idx + 1], -999.0f);
        EXPECT_FLOAT_EQ(hostDst[idx + 2], -999.0f);
      }
    }
  }

  nppiFree(d_src);
  nppiFree(d_dst);
}

// Test 3: Color patterns verification
TEST_F(NppiCopy32fC3RTest, ColorPatternsVerification) {
  const int width = 256;
  const int height = 256;

  // Create distinct color patterns
  std::vector<Npp32f> hostSrc(width * height * 3);
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      int idx = (y * width + x) * 3;
      // Create gradient in each channel
      hostSrc[idx] = (x / (float)width) * 255.0f;                      // R gradient left to right
      hostSrc[idx + 1] = (y / (float)height) * 255.0f;                 // G gradient top to bottom
      hostSrc[idx + 2] = ((x + y) / (float)(width + height)) * 255.0f; // B diagonal
    }
  }

  Npp32f *d_src = nullptr;
  Npp32f *d_dst = nullptr;
  int srcStep, dstStep;
  allocateImage(&d_src, width, height, &srcStep);
  allocateImage(&d_dst, width, height, &dstStep);

  cudaMemcpy2D(d_src, srcStep, hostSrc.data(), width * 3 * sizeof(Npp32f), width * 3 * sizeof(Npp32f), height,
               cudaMemcpyHostToDevice);

  NppiSize roi = {width, height};
  NppStatus status = nppiCopy_32f_C3R(d_src, srcStep, d_dst, dstStep, roi);
  ASSERT_EQ(status, NPP_SUCCESS);

  std::vector<Npp32f> hostDst(width * height * 3);
  cudaMemcpy2D(hostDst.data(), width * 3 * sizeof(Npp32f), d_dst, dstStep, width * 3 * sizeof(Npp32f), height,
               cudaMemcpyDeviceToHost);

  // Verify specific points
  int testPoints[][2] = {{0, 0}, {width - 1, 0}, {0, height - 1}, {width - 1, height - 1}, {width / 2, height / 2}};
  for (auto &pt : testPoints) {
    int x = pt[0], y = pt[1];
    int idx = (y * width + x) * 3;
    EXPECT_FLOAT_EQ(hostDst[idx], hostSrc[idx]) << "R channel mismatch at (" << x << "," << y << ")";
    EXPECT_FLOAT_EQ(hostDst[idx + 1], hostSrc[idx + 1]) << "G channel mismatch at (" << x << "," << y << ")";
    EXPECT_FLOAT_EQ(hostDst[idx + 2], hostSrc[idx + 2]) << "B channel mismatch at (" << x << "," << y << ")";
  }

  nppiFree(d_src);
  nppiFree(d_dst);
}

// Test 4: Different source and destination strides
TEST_F(NppiCopy32fC3RTest, DifferentStrides3Channel) {
  const int width = 200;
  const int height = 200;
  const int srcExtraBytes = 512; // Extra padding for source
  const int dstExtraBytes = 256; // Different padding for destination

  int srcStep = width * 3 * sizeof(Npp32f) + srcExtraBytes;
  int dstStep = width * 3 * sizeof(Npp32f) + dstExtraBytes;

  // Allocate with custom strides
  Npp32f *d_src = nullptr;
  Npp32f *d_dst = nullptr;
  cudaMalloc(&d_src, srcStep * height);
  cudaMalloc(&d_dst, dstStep * height);
  ASSERT_NE(d_src, nullptr);
  ASSERT_NE(d_dst, nullptr);

  // Create test data
  std::vector<Npp8u> hostSrcBuffer(srcStep * height, 0);
  for (int y = 0; y < height; y++) {
    Npp32f *row = (Npp32f *)(hostSrcBuffer.data() + y * srcStep);
    for (int x = 0; x < width; x++) {
      row[x * 3] = y * 1000.0f + x;
      row[x * 3 + 1] = y * 1000.0f + x + 500.0f;
      row[x * 3 + 2] = y * 1000.0f + x + 1000.0f;
    }
  }

  cudaMemcpy(d_src, hostSrcBuffer.data(), srcStep * height, cudaMemcpyHostToDevice);

  // Clear destination
  cudaMemset(d_dst, 0, dstStep * height);

  // Perform copy
  NppiSize roi = {width, height};
  NppStatus status = nppiCopy_32f_C3R(d_src, srcStep, d_dst, dstStep, roi);
  ASSERT_EQ(status, NPP_SUCCESS);

  // Verify
  std::vector<Npp8u> hostDstBuffer(dstStep * height);
  cudaMemcpy(hostDstBuffer.data(), d_dst, dstStep * height, cudaMemcpyDeviceToHost);

  for (int y = 0; y < height; y++) {
    Npp32f *srcRow = (Npp32f *)(hostSrcBuffer.data() + y * srcStep);
    Npp32f *dstRow = (Npp32f *)(hostDstBuffer.data() + y * dstStep);
    for (int x = 0; x < width; x++) {
      EXPECT_FLOAT_EQ(dstRow[x * 3], srcRow[x * 3]);
      EXPECT_FLOAT_EQ(dstRow[x * 3 + 1], srcRow[x * 3 + 1]);
      EXPECT_FLOAT_EQ(dstRow[x * 3 + 2], srcRow[x * 3 + 2]);
    }
  }

  cudaFree(d_src);
  cudaFree(d_dst);
}

// Test 5: Large image copy performance for 3 channels
TEST_F(NppiCopy32fC3RTest, LargeImage3ChannelPerformance) {
  const int width = 3840;  // 4K width
  const int height = 2160; // 4K height
  const int iterations = 10;

  Npp32f *d_src = nullptr;
  Npp32f *d_dst = nullptr;
  int srcStep, dstStep;
  allocateImage(&d_src, width, height, &srcStep);
  allocateImage(&d_dst, width, height, &dstStep);

  // Fill source with test pattern
  cudaMemset(d_src, 0x42, srcStep * height);

  NppiSize roi = {width, height};

  // Warm up
  nppiCopy_32f_C3R(d_src, srcStep, d_dst, dstStep, roi);
  cudaDeviceSynchronize();

  // Measure performance
  auto start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < iterations; i++) {
    NppStatus status = nppiCopy_32f_C3R(d_src, srcStep, d_dst, dstStep, roi);
    ASSERT_EQ(status, NPP_SUCCESS);
  }
  cudaDeviceSynchronize();
  auto end = std::chrono::high_resolution_clock::now();

  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
  double avgTime = duration / (double)iterations;
  double dataSize = width * height * 3 * sizeof(Npp32f) * 2 / 1024.0 / 1024.0 / 1024.0; // GB
  double bandwidth = dataSize * 1000000.0 / avgTime;                                    // GB/s

  std::cout << "3-channel large image copy performance:" << std::endl;
  std::cout << "  Image size: " << width << "x" << height << " (3 channels)" << std::endl;
  std::cout << "  Average time: " << avgTime << " us" << std::endl;
  std::cout << "  Bandwidth: " << bandwidth << " GB/s" << std::endl;

  nppiFree(d_src);
  nppiFree(d_dst);
}

// Test 7: Edge alignment cases for 3-channel data
TEST_F(NppiCopy32fC3RTest, EdgeAlignmentCases3Channel) {
  const int testWidths[] = {1, 3, 5, 7, 13, 17, 31, 63, 85, 127, 171, 255, 341, 511};
  const int height = 13; // Prime number for edge case

  for (int width : testWidths) {
    Npp32f *d_src = nullptr;
    Npp32f *d_dst = nullptr;
    int srcStep, dstStep;

    d_src = nppiMalloc_32f_C3(width, height, &srcStep);
    d_dst = nppiMalloc_32f_C3(width, height, &dstStep);

    if (!d_src || !d_dst) {
      if (d_src)
        nppiFree(d_src);
      if (d_dst)
        nppiFree(d_dst);
      continue;
    }

    // Create unique pattern
    std::vector<Npp32f> pattern(width * height * 3);
    for (int i = 0; i < width * height; i++) {
      pattern[i * 3] = static_cast<float>(i * 7 + width);
      pattern[i * 3 + 1] = static_cast<float>(i * 11 + width);
      pattern[i * 3 + 2] = static_cast<float>(i * 13 + width);
    }

    cudaMemcpy2D(d_src, srcStep, pattern.data(), width * 3 * sizeof(Npp32f), width * 3 * sizeof(Npp32f), height,
                 cudaMemcpyHostToDevice);

    NppiSize roi = {width, height};
    NppStatus status = nppiCopy_32f_C3R(d_src, srcStep, d_dst, dstStep, roi);
    EXPECT_EQ(status, NPP_SUCCESS) << "Copy failed for width " << width;

    // Verify
    std::vector<Npp32f> result(width * height * 3);
    cudaMemcpy2D(result.data(), width * 3 * sizeof(Npp32f), d_dst, dstStep, width * 3 * sizeof(Npp32f), height,
                 cudaMemcpyDeviceToHost);

    EXPECT_TRUE(verifyData(pattern, result)) << "Data mismatch for width " << width;

    nppiFree(d_src);
    nppiFree(d_dst);
  }
}

// Test 8: Error handling for 3-channel copy
TEST_F(NppiCopy32fC3RTest, DISABLED_ErrorHandling3Channel) {
  const int width = 100;
  const int height = 100;

  Npp32f *d_src = nullptr;
  Npp32f *d_dst = nullptr;
  int srcStep, dstStep;
  allocateImage(&d_src, width, height, &srcStep);
  allocateImage(&d_dst, width, height, &dstStep);

  NppiSize roi = {width, height};

  // Test null source
  EXPECT_EQ(nppiCopy_32f_C3R(nullptr, srcStep, d_dst, dstStep, roi), NPP_NULL_POINTER_ERROR);

  // Test null destination
  EXPECT_EQ(nppiCopy_32f_C3R(d_src, srcStep, nullptr, dstStep, roi), NPP_NULL_POINTER_ERROR);

  // Test invalid ROI
  NppiSize invalidRoi = {0, height};
  EXPECT_EQ(nppiCopy_32f_C3R(d_src, srcStep, d_dst, dstStep, invalidRoi), NPP_SUCCESS);

  invalidRoi = {width, -5};
  EXPECT_EQ(nppiCopy_32f_C3R(d_src, srcStep, d_dst, dstStep, invalidRoi), NPP_SIZE_ERROR);

  nppiFree(d_src);
  nppiFree(d_dst);
}

// Test 9: Concurrent stream operations with 3 channels
TEST_F(NppiCopy32fC3RTest, DISABLED_ConcurrentStreams3Channel) {
  const int numStreams = 6;
  const int width = 800;
  const int height = 600;

  std::vector<cudaStream_t> streams(numStreams);
  std::vector<Npp32f *> srcBuffers(numStreams);
  std::vector<Npp32f *> dstBuffers(numStreams);
  std::vector<int> srcSteps(numStreams);
  std::vector<int> dstSteps(numStreams);

  // Create streams and allocate buffers
  for (int i = 0; i < numStreams; i++) {
    cudaStreamCreate(&streams[i]);
    srcBuffers[i] = nppiMalloc_32f_C3(width, height, &srcSteps[i]);
    dstBuffers[i] = nppiMalloc_32f_C3(width, height, &dstSteps[i]);
    ASSERT_NE(srcBuffers[i], nullptr);
    ASSERT_NE(dstBuffers[i], nullptr);

    // Initialize with different patterns
    std::vector<Npp32f> pattern(width * height * 3);
    generatePattern(pattern, width, height, static_cast<float>(i + 1));
    cudaMemcpy2DAsync(srcBuffers[i], srcSteps[i], pattern.data(), width * 3 * sizeof(Npp32f),
                      width * 3 * sizeof(Npp32f), height, cudaMemcpyHostToDevice, streams[i]);
  }

  // Launch concurrent copies
  NppiSize roi = {width, height};
  for (int i = 0; i < numStreams; i++) {
    NppStreamContext ctx;
    nppGetStreamContext(&ctx);
    ctx.hStream = streams[i];

    NppStatus status = nppiCopy_32f_C3R_Ctx(srcBuffers[i], srcSteps[i], dstBuffers[i], dstSteps[i], roi, ctx);
    EXPECT_EQ(status, NPP_SUCCESS);
  }

  // Verify results
  for (int i = 0; i < numStreams; i++) {
    cudaStreamSynchronize(streams[i]);

    std::vector<Npp32f> result(width * height * 3);
    cudaMemcpy2D(result.data(), width * 3 * sizeof(Npp32f), dstBuffers[i], dstSteps[i], width * 3 * sizeof(Npp32f),
                 height, cudaMemcpyDeviceToHost);

    // Check first and last pixel (3 channels each)
    float expectedFirst[3] = {0.0f, 0.1f, 0.2f};
    float expectedLast[3];
    for (int c = 0; c < 3; c++) {
      expectedLast[c] = ((height - 1) * width + (width - 1)) * (i + 1) + c * 0.1f;
      EXPECT_FLOAT_EQ(result[c], expectedFirst[c] * (i + 1))
          << "Stream " << i << " channel " << c << " first pixel failed";
      EXPECT_NEAR(result[(width * height - 1) * 3 + c], expectedLast[c], 0.1f)
          << "Stream " << i << " channel " << c << " last pixel failed";
    }
  }

  // Cleanup
  for (int i = 0; i < numStreams; i++) {
    cudaStreamDestroy(streams[i]);
    nppiFree(srcBuffers[i]);
    nppiFree(dstBuffers[i]);
  }
}

// Test 10: Extreme dimensions for 3-channel data
TEST_F(NppiCopy32fC3RTest, DISABLED_ExtremeDimensions3Channel) {
  // Test very wide but short image
  {
    const int width = 8192;
    const int height = 2;

    Npp32f *d_src = nullptr;
    Npp32f *d_dst = nullptr;
    int srcStep, dstStep;

    d_src = nppiMalloc_32f_C3(width, height, &srcStep);
    d_dst = nppiMalloc_32f_C3(width, height, &dstStep);

    if (d_src && d_dst) {
      // Initialize with pattern
      std::vector<Npp32f> pattern(width * height * 3);
      for (int i = 0; i < width * height * 3; i++) {
        pattern[i] = i * 0.1f;
      }

      cudaMemcpy2D(d_src, srcStep, pattern.data(), width * 3 * sizeof(Npp32f), width * 3 * sizeof(Npp32f), height,
                   cudaMemcpyHostToDevice);

      NppiSize roi = {width, height};
      NppStatus status = nppiCopy_32f_C3R(d_src, srcStep, d_dst, dstStep, roi);
      EXPECT_EQ(status, NPP_SUCCESS);

      // Verify corner pixels
      Npp32f corners[12]; // 4 corners * 3 channels
      cudaMemcpy(&corners[0], d_dst, 3 * sizeof(Npp32f), cudaMemcpyDeviceToHost);
      cudaMemcpy(&corners[3], (Npp32f *)((Npp8u *)d_dst) + (width - 1) * 3 * sizeof(Npp32f), 3 * sizeof(Npp32f),
                 cudaMemcpyDeviceToHost);
      cudaMemcpy(&corners[6], (Npp32f *)((Npp8u *)d_dst + dstStep) + 0, 3 * sizeof(Npp32f), cudaMemcpyDeviceToHost);
      cudaMemcpy(&corners[9], (Npp32f *)((Npp8u *)d_dst + dstStep) + (width - 1) * 3 * sizeof(Npp32f),
                 3 * sizeof(Npp32f), cudaMemcpyDeviceToHost);

      // Verify expected values
      EXPECT_FLOAT_EQ(corners[0], 0.0f);
      EXPECT_FLOAT_EQ(corners[1], 0.1f);
      EXPECT_FLOAT_EQ(corners[2], 0.2f);

      nppiFree(d_src);
      nppiFree(d_dst);
    }
  }

  // Test very tall but narrow image
  {
    const int width = 3;
    const int height = 8192;

    Npp32f *d_src = nullptr;
    Npp32f *d_dst = nullptr;
    int srcStep, dstStep;

    d_src = nppiMalloc_32f_C3(width, height, &srcStep);
    d_dst = nppiMalloc_32f_C3(width, height, &dstStep);

    if (d_src && d_dst) {
      NppiSize roi = {width, height};
      NppStatus status = nppiCopy_32f_C3R(d_src, srcStep, d_dst, dstStep, roi);
      EXPECT_EQ(status, NPP_SUCCESS);

      nppiFree(d_src);
      nppiFree(d_dst);
    }
  }
}

// Test 11: Channel-Specific Pattern Verification
TEST_F(NppiCopy32fC3RTest, ChannelSpecificPatternVerification) {
  const int width = 64, height = 64;
  
  Npp32f *d_src = nullptr;
  Npp32f *d_dst = nullptr;
  int srcStep, dstStep;
  allocateImage(&d_src, width, height, &srcStep);
  allocateImage(&d_dst, width, height, &dstStep);
  
  // Create distinct patterns for each channel to verify channel integrity
  std::vector<Npp32f> hostSrc(width * height * 3);
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      int idx = (y * width + x) * 3;
      // R channel: sine wave pattern
      hostSrc[idx] = 128.0f + 127.0f * std::sin(2.0f * M_PI * x / width);
      // G channel: cosine wave pattern  
      hostSrc[idx + 1] = 128.0f + 127.0f * std::cos(2.0f * M_PI * y / height);
      // B channel: checkerboard pattern
      hostSrc[idx + 2] = ((x / 8 + y / 8) % 2) ? 255.0f : 0.0f;
    }
  }
  
  cudaMemcpy2D(d_src, srcStep, hostSrc.data(), width * 3 * sizeof(Npp32f),
               width * 3 * sizeof(Npp32f), height, cudaMemcpyHostToDevice);
  
  NppiSize roi = {width, height};
  NppStatus status = nppiCopy_32f_C3R(d_src, srcStep, d_dst, dstStep, roi);
  ASSERT_EQ(status, NPP_SUCCESS);
  
  std::vector<Npp32f> hostDst(width * height * 3);
  cudaMemcpy2D(hostDst.data(), width * 3 * sizeof(Npp32f), d_dst, dstStep,
               width * 3 * sizeof(Npp32f), height, cudaMemcpyDeviceToHost);
  
  // Verify pattern integrity at strategic points
  int testPoints[][2] = {{0, 0}, {width/4, height/4}, {width/2, height/2}, 
                         {3*width/4, 3*height/4}, {width-1, height-1}};
  
  for (auto &pt : testPoints) {
    int x = pt[0], y = pt[1];
    int idx = (y * width + x) * 3;
    
    // Verify R channel (sine wave)
    float expectedR = 128.0f + 127.0f * std::sin(2.0f * M_PI * x / width);
    EXPECT_NEAR(hostDst[idx], expectedR, 0.01f) 
      << "R channel pattern mismatch at (" << x << "," << y << ")";
    
    // Verify G channel (cosine wave)
    float expectedG = 128.0f + 127.0f * std::cos(2.0f * M_PI * y / height);
    EXPECT_NEAR(hostDst[idx + 1], expectedG, 0.01f)
      << "G channel pattern mismatch at (" << x << "," << y << ")";
    
    // Verify B channel (checkerboard)
    float expectedB = ((x / 8 + y / 8) % 2) ? 255.0f : 0.0f;
    EXPECT_FLOAT_EQ(hostDst[idx + 2], expectedB)
      << "B channel pattern mismatch at (" << x << "," << y << ")";
  }
  
  nppiFree(d_src);
  nppiFree(d_dst);
}

// Test 12: Multi-ROI Performance and Correctness
TEST_F(NppiCopy32fC3RTest, MultiROIPerformanceAndCorrectness) {
  const int width = 512, height = 512;
  const int numROIs = 9; // 3x3 grid of ROIs
  
  Npp32f *d_src = nullptr;
  Npp32f *d_dst = nullptr;
  int srcStep, dstStep;
  allocateImage(&d_src, width, height, &srcStep);
  allocateImage(&d_dst, width, height, &dstStep);
  
  // Initialize source with position-dependent values
  std::vector<Npp32f> hostSrc(width * height * 3);
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      int idx = (y * width + x) * 3;
      hostSrc[idx] = y * 1000.0f + x;          // R: position encoding
      hostSrc[idx + 1] = y * 1000.0f + x + 0.5f; // G: slight offset
      hostSrc[idx + 2] = y * 1000.0f + x + 1.0f; // B: larger offset
    }
  }
  
  cudaMemcpy2D(d_src, srcStep, hostSrc.data(), width * 3 * sizeof(Npp32f),
               width * 3 * sizeof(Npp32f), height, cudaMemcpyHostToDevice);
  
  // Clear destination
  cudaMemset(d_dst, 0, dstStep * height);
  
  // Define 3x3 grid of ROIs
  const int roiWidth = width / 3;
  const int roiHeight = height / 3;
  
  auto start = std::chrono::high_resolution_clock::now();
  
  // Copy each ROI
  for (int roiY = 0; roiY < 3; roiY++) {
    for (int roiX = 0; roiX < 3; roiX++) {
      int startX = roiX * roiWidth;
      int startY = roiY * roiHeight;
      
      // Handle edge cases for last ROI
      int actualWidth = (roiX == 2) ? width - startX : roiWidth;
      int actualHeight = (roiY == 2) ? height - startY : roiHeight;
      
      Npp32f *pSrcROI = (Npp32f *)((Npp8u *)d_src + startY * srcStep) + startX * 3;
      Npp32f *pDstROI = (Npp32f *)((Npp8u *)d_dst + startY * dstStep) + startX * 3;
      
      NppiSize roi = {actualWidth, actualHeight};
      NppStatus status = nppiCopy_32f_C3R(pSrcROI, srcStep, pDstROI, dstStep, roi);
      ASSERT_EQ(status, NPP_SUCCESS) << "ROI (" << roiX << "," << roiY << ") failed";
    }
  }
  
  auto end = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
  
  std::cout << "Multi-ROI copy (" << numROIs << " ROIs) completed in " 
            << duration << " microseconds" << std::endl;
  
  // Verify correctness
  std::vector<Npp32f> hostDst(width * height * 3);
  cudaMemcpy2D(hostDst.data(), width * 3 * sizeof(Npp32f), d_dst, dstStep,
               width * 3 * sizeof(Npp32f), height, cudaMemcpyDeviceToHost);
  
  // Sample verification across all ROIs
  for (int roiY = 0; roiY < 3; roiY++) {
    for (int roiX = 0; roiX < 3; roiX++) {
      int startX = roiX * roiWidth;
      int startY = roiY * roiHeight;
      int endX = (roiX == 2) ? width : (roiX + 1) * roiWidth;
      int endY = (roiY == 2) ? height : (roiY + 1) * roiHeight;
      
      // Test a few sample points in each ROI
      for (int sy = startY; sy < endY; sy += (endY - startY) / 3 + 1) {
        for (int sx = startX; sx < endX; sx += (endX - startX) / 3 + 1) {
          int idx = (sy * width + sx) * 3;
          
          EXPECT_FLOAT_EQ(hostDst[idx], hostSrc[idx])
            << "R mismatch at ROI(" << roiX << "," << roiY << ") pos(" << sx << "," << sy << ")";
          EXPECT_FLOAT_EQ(hostDst[idx + 1], hostSrc[idx + 1])
            << "G mismatch at ROI(" << roiX << "," << roiY << ") pos(" << sx << "," << sy << ")";
          EXPECT_FLOAT_EQ(hostDst[idx + 2], hostSrc[idx + 2])
            << "B mismatch at ROI(" << roiX << "," << roiY << ") pos(" << sx << "," << sy << ")";
        }
      }
    }
  }
  
  nppiFree(d_src);
  nppiFree(d_dst);
}