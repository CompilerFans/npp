// Implementation file

#include "npp.h"
#include <algorithm>
#include <chrono>
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <random>
#include <thread>
#include <vector>

class NppiCopy32fC1RTest : public ::testing::Test {
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
    *ppImg = nppiMalloc_32f_C1(width, height, pStep);
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

  // Generate test pattern
  void generatePattern(std::vector<Npp32f> &data, int width, int height, float scale = 1.0f) {
    data.resize(width * height);
    for (int y = 0; y < height; y++) {
      for (int x = 0; x < width; x++) {
        data[y * width + x] = ((y * width + x) % 1000) * scale;
      }
    }
  }
};

// Test 1: Basic copy functionality
TEST_F(NppiCopy32fC1RTest, BasicCopy) {
  const int width = 256;
  const int height = 256;

  std::vector<Npp32f> hostSrc(width * height);
  generatePattern(hostSrc, width, height);

  Npp32f *d_src = nullptr;
  Npp32f *d_dst = nullptr;
  int srcStep, dstStep;
  allocateImage(&d_src, width, height, &srcStep);
  allocateImage(&d_dst, width, height, &dstStep);

  // Upload source data
  cudaMemcpy2D(d_src, srcStep, hostSrc.data(), width * sizeof(Npp32f), width * sizeof(Npp32f), height,
               cudaMemcpyHostToDevice);

  // Perform copy
  NppiSize roi = {width, height};
  NppStatus status = nppiCopy_32f_C1R(d_src, srcStep, d_dst, dstStep, roi);
  ASSERT_EQ(status, NPP_SUCCESS);

  // Verify result
  std::vector<Npp32f> hostDst(width * height);
  cudaMemcpy2D(hostDst.data(), width * sizeof(Npp32f), d_dst, dstStep, width * sizeof(Npp32f), height,
               cudaMemcpyDeviceToHost);

  EXPECT_TRUE(verifyData(hostSrc, hostDst));

  nppiFree(d_src);
  nppiFree(d_dst);
}

// Test 2: Partial ROI copy
TEST_F(NppiCopy32fC1RTest, PartialROICopy) {
  const int width = 512;
  const int height = 512;
  const int roiX = 128;
  const int roiY = 128;
  const int roiWidth = 256;
  const int roiHeight = 256;

  std::vector<Npp32f> hostSrc(width * height);
  std::vector<Npp32f> hostDst(width * height, -999.0f);
  generatePattern(hostSrc, width, height);

  Npp32f *d_src = nullptr;
  Npp32f *d_dst = nullptr;
  int srcStep, dstStep;
  allocateImage(&d_src, width, height, &srcStep);
  allocateImage(&d_dst, width, height, &dstStep);

  // Upload data
  cudaMemcpy2D(d_src, srcStep, hostSrc.data(), width * sizeof(Npp32f), width * sizeof(Npp32f), height,
               cudaMemcpyHostToDevice);
  cudaMemcpy2D(d_dst, dstStep, hostDst.data(), width * sizeof(Npp32f), width * sizeof(Npp32f), height,
               cudaMemcpyHostToDevice);

  // Copy ROI
  Npp32f *pSrcROI = (Npp32f *)((Npp8u *)d_src + roiY * srcStep) + roiX;
  Npp32f *pDstROI = (Npp32f *)((Npp8u *)d_dst + roiY * dstStep) + roiX;

  NppiSize roi = {roiWidth, roiHeight};
  NppStatus status = nppiCopy_32f_C1R(pSrcROI, srcStep, pDstROI, dstStep, roi);
  ASSERT_EQ(status, NPP_SUCCESS);

  // Verify result
  cudaMemcpy2D(hostDst.data(), width * sizeof(Npp32f), d_dst, dstStep, width * sizeof(Npp32f), height,
               cudaMemcpyDeviceToHost);

  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      int idx = y * width + x;
      if (x >= roiX && x < roiX + roiWidth && y >= roiY && y < roiY + roiHeight) {
        EXPECT_FLOAT_EQ(hostDst[idx], hostSrc[idx]);
      } else {
        EXPECT_FLOAT_EQ(hostDst[idx], -999.0f);
      }
    }
  }

  nppiFree(d_src);
  nppiFree(d_dst);
}

// Test 4: Different source and destination strides
TEST_F(NppiCopy32fC1RTest, DifferentStrides) {
  const int width = 200;
  const int height = 200;
  const int srcExtraBytes = 256;
  const int dstExtraBytes = 128;

  int srcStep = width * sizeof(Npp32f) + srcExtraBytes;
  int dstStep = width * sizeof(Npp32f) + dstExtraBytes;

  // Allocate with custom strides
  Npp32f *d_src = nullptr;
  Npp32f *d_dst = nullptr;
  cudaMalloc(&d_src, srcStep * height);
  cudaMalloc(&d_dst, dstStep * height);
  ASSERT_NE(d_src, nullptr);
  ASSERT_NE(d_dst, nullptr);

  // Create test data
  std::vector<Npp8u> hostSrcBuffer(srcStep * height);
  for (int y = 0; y < height; y++) {
    Npp32f *row = (Npp32f *)(hostSrcBuffer.data() + y * srcStep);
    for (int x = 0; x < width; x++) {
      row[x] = y * 1000.0f + x;
    }
  }

  cudaMemcpy(d_src, hostSrcBuffer.data(), srcStep * height, cudaMemcpyHostToDevice);

  // Perform copy
  NppiSize roi = {width, height};
  NppStatus status = nppiCopy_32f_C1R(d_src, srcStep, d_dst, dstStep, roi);
  ASSERT_EQ(status, NPP_SUCCESS);

  // Verify
  std::vector<Npp8u> hostDstBuffer(dstStep * height);
  cudaMemcpy(hostDstBuffer.data(), d_dst, dstStep * height, cudaMemcpyDeviceToHost);

  for (int y = 0; y < height; y++) {
    Npp32f *srcRow = (Npp32f *)(hostSrcBuffer.data() + y * srcStep);
    Npp32f *dstRow = (Npp32f *)(hostDstBuffer.data() + y * dstStep);
    for (int x = 0; x < width; x++) {
      EXPECT_FLOAT_EQ(dstRow[x], srcRow[x]);
    }
  }

  cudaFree(d_src);
  cudaFree(d_dst);
}

// Test 5: Large image copy performance
TEST_F(NppiCopy32fC1RTest, LargeImagePerformance) {
  const int width = 8192;
  const int height = 4096;
  const int iterations = 10;

  Npp32f *d_src = nullptr;
  Npp32f *d_dst = nullptr;
  int srcStep, dstStep;
  allocateImage(&d_src, width, height, &srcStep);
  allocateImage(&d_dst, width, height, &dstStep);

  // Fill source with test pattern
  cudaMemset(d_src, 0x3F, srcStep * height); // ~0.5f pattern

  NppiSize roi = {width, height};

  // Warm up
  nppiCopy_32f_C1R(d_src, srcStep, d_dst, dstStep, roi);
  cudaDeviceSynchronize();

  // Measure performance
  auto start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < iterations; i++) {
    NppStatus status = nppiCopy_32f_C1R(d_src, srcStep, d_dst, dstStep, roi);
    ASSERT_EQ(status, NPP_SUCCESS);
  }
  cudaDeviceSynchronize();
  auto end = std::chrono::high_resolution_clock::now();

  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
  double avgTime = duration / (double)iterations;
  double dataSize = width * height * sizeof(Npp32f) * 2 / 1024.0 / 1024.0 / 1024.0; // GB
  double bandwidth = dataSize * 1000000.0 / avgTime;                                // GB/s

  std::cout << "Large image copy performance:" << std::endl;
  std::cout << "  Image size: " << width << "x" << height << std::endl;
  std::cout << "  Average time: " << avgTime << " us" << std::endl;
  std::cout << "  Bandwidth: " << bandwidth << " GB/s" << std::endl;

  nppiFree(d_src);
  nppiFree(d_dst);
}

// Test 6: Concurrent stream operations
TEST_F(NppiCopy32fC1RTest, DISABLED_ConcurrentStreams) {
  const int numStreams = 8;
  const int width = 1024;
  const int height = 1024;

  std::vector<cudaStream_t> streams(numStreams);
  std::vector<Npp32f *> srcBuffers(numStreams);
  std::vector<Npp32f *> dstBuffers(numStreams);
  std::vector<int> srcSteps(numStreams);
  std::vector<int> dstSteps(numStreams);

  // Create streams and allocate buffers
  for (int i = 0; i < numStreams; i++) {
    cudaStreamCreate(&streams[i]);
    allocateImage(&srcBuffers[i], width, height, &srcSteps[i]);
    allocateImage(&dstBuffers[i], width, height, &dstSteps[i]);

    // Initialize with different patterns
    std::vector<Npp32f> pattern(width * height);
    generatePattern(pattern, width, height, static_cast<float>(i + 1));
    cudaMemcpy2DAsync(srcBuffers[i], srcSteps[i], pattern.data(), width * sizeof(Npp32f), width * sizeof(Npp32f),
                      height, cudaMemcpyHostToDevice, streams[i]);
  }

  // Launch concurrent copies
  NppiSize roi = {width, height};
  for (int i = 0; i < numStreams; i++) {
    NppStreamContext ctx;
    nppGetStreamContext(&ctx);
    ctx.hStream = streams[i];

    NppStatus status = nppiCopy_32f_C1R_Ctx(srcBuffers[i], srcSteps[i], dstBuffers[i], dstSteps[i], roi, ctx);
    EXPECT_EQ(status, NPP_SUCCESS);
  }

  // Verify results
  for (int i = 0; i < numStreams; i++) {
    cudaStreamSynchronize(streams[i]);

    std::vector<Npp32f> result(width * height);
    cudaMemcpy2D(result.data(), width * sizeof(Npp32f), dstBuffers[i], dstSteps[i], width * sizeof(Npp32f), height,
                 cudaMemcpyDeviceToHost);

    // Check first and last values
    float expectedFirst = 0.0f;
    float expectedLast = ((height - 1) * width + (width - 1)) * (i + 1);
    EXPECT_FLOAT_EQ(result[0], expectedFirst);
    EXPECT_NEAR(result[width * height - 1], expectedLast, 0.01f);
  }

  // Cleanup
  for (int i = 0; i < numStreams; i++) {
    cudaStreamDestroy(streams[i]);
    nppiFree(srcBuffers[i]);
    nppiFree(dstBuffers[i]);
  }
}

// Test 7: Edge alignment cases
TEST_F(NppiCopy32fC1RTest, EdgeAlignmentCases) {
  const int testWidths[] = {1, 3, 7, 13, 17, 31, 63, 127, 251, 509, 1021, 2047};
  const int height = 17; // Prime number for extra edge case

  for (int width : testWidths) {
    Npp32f *d_src = nullptr;
    Npp32f *d_dst = nullptr;
    int srcStep, dstStep;

    d_src = nppiMalloc_32f_C1(width, height, &srcStep);
    d_dst = nppiMalloc_32f_C1(width, height, &dstStep);

    ASSERT_NE(d_src, nullptr) << "Allocation failed for width " << width;
    ASSERT_NE(d_dst, nullptr) << "Allocation failed for width " << width;

    // Create unique pattern
    std::vector<Npp32f> pattern(width * height);
    for (int i = 0; i < width * height; i++) {
      pattern[i] = static_cast<float>(i * 7 + width);
    }

    cudaMemcpy2D(d_src, srcStep, pattern.data(), width * sizeof(Npp32f), width * sizeof(Npp32f), height,
                 cudaMemcpyHostToDevice);

    NppiSize roi = {width, height};
    NppStatus status = nppiCopy_32f_C1R(d_src, srcStep, d_dst, dstStep, roi);
    EXPECT_EQ(status, NPP_SUCCESS) << "Copy failed for width " << width;

    // Verify
    std::vector<Npp32f> result(width * height);
    cudaMemcpy2D(result.data(), width * sizeof(Npp32f), d_dst, dstStep, width * sizeof(Npp32f), height,
                 cudaMemcpyDeviceToHost);

    EXPECT_TRUE(verifyData(pattern, result)) << "Data mismatch for width " << width;

    nppiFree(d_src);
    nppiFree(d_dst);
  }
}

// Test 8: Error handling
TEST_F(NppiCopy32fC1RTest, DISABLED_ErrorHandling) {
  const int width = 100;
  const int height = 100;

  Npp32f *d_src = nullptr;
  Npp32f *d_dst = nullptr;
  int srcStep, dstStep;
  allocateImage(&d_src, width, height, &srcStep);
  allocateImage(&d_dst, width, height, &dstStep);

  NppiSize roi = {width, height};

  // Test null source
  EXPECT_EQ(nppiCopy_32f_C1R(nullptr, srcStep, d_dst, dstStep, roi), NPP_NULL_POINTER_ERROR);

  // Test null destination
  EXPECT_EQ(nppiCopy_32f_C1R(d_src, srcStep, nullptr, dstStep, roi), NPP_NULL_POINTER_ERROR);

  // Test invalid ROI
  NppiSize invalidRoi = {0, height};
  EXPECT_EQ(nppiCopy_32f_C1R(d_src, srcStep, d_dst, dstStep, invalidRoi), NPP_SUCCESS);

  invalidRoi = {width, -5};
  EXPECT_EQ(nppiCopy_32f_C1R(d_src, srcStep, d_dst, dstStep, invalidRoi), NPP_SIZE_ERROR);

  // Test invalid step
  EXPECT_EQ(nppiCopy_32f_C1R(d_src, 0, d_dst, dstStep, roi), NPP_SUCCESS);
  EXPECT_NE(nppiCopy_32f_C1R(d_src, srcStep, d_dst, -1, roi), NPP_SUCCESS);

  nppiFree(d_src);
  nppiFree(d_dst);
}

// Test 9: Stress test with random operations
TEST_F(NppiCopy32fC1RTest, StressTest) {
  const int numOperations = 100;

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> sizeDist(1, 100);
  std::uniform_int_distribution<> coordDist(0, 50);

  for (int op = 0; op < numOperations; op++) {
    // Random dimensions
    int srcWidth = sizeDist(gen) * 4;
    int srcHeight = sizeDist(gen) * 4;
    int dstWidth = srcWidth + coordDist(gen) * 4;
    int dstHeight = srcHeight + coordDist(gen) * 4;

    // Random ROI
    int roiWidth = std::min(srcWidth, sizeDist(gen) * 2);
    int roiHeight = std::min(srcHeight, sizeDist(gen) * 2);
    int roiX = coordDist(gen) % (srcWidth - roiWidth + 1);
    int roiY = coordDist(gen) % (srcHeight - roiHeight + 1);

    Npp32f *d_src = nullptr;
    Npp32f *d_dst = nullptr;
    int srcStep, dstStep;

    d_src = nppiMalloc_32f_C1(srcWidth, srcHeight, &srcStep);
    d_dst = nppiMalloc_32f_C1(dstWidth, dstHeight, &dstStep);

    if (d_src && d_dst) {
      // Initialize with pattern
      std::vector<Npp32f> pattern(srcWidth * srcHeight);
      generatePattern(pattern, srcWidth, srcHeight, op * 0.1f);
      cudaMemcpy2D(d_src, srcStep, pattern.data(), srcWidth * sizeof(Npp32f), srcWidth * sizeof(Npp32f), srcHeight,
                   cudaMemcpyHostToDevice);

      // Perform copy
      Npp32f *pSrcROI = (Npp32f *)((Npp8u *)d_src + roiY * srcStep) + roiX;
      Npp32f *pDstROI = d_dst;

      NppiSize roi = {roiWidth, roiHeight};
      NppStatus status = nppiCopy_32f_C1R(pSrcROI, srcStep, pDstROI, dstStep, roi);
      EXPECT_EQ(status, NPP_SUCCESS) << "Operation " << op << " failed";

      nppiFree(d_src);
      nppiFree(d_dst);
    }
  }
}

// Test 10: Zero-copy scenarios (device to device)
TEST_F(NppiCopy32fC1RTest, DeviceToDeviceCopy) {
  const int width = 512;
  const int height = 512;

  // Create multiple GPU buffers
  Npp32f *d_buf1 = nullptr;
  Npp32f *d_buf2 = nullptr;
  Npp32f *d_buf3 = nullptr;
  int step1, step2, step3;

  allocateImage(&d_buf1, width, height, &step1);
  allocateImage(&d_buf2, width, height, &step2);
  allocateImage(&d_buf3, width, height, &step3);

  // Initialize first buffer
  std::vector<Npp32f> initialData(width * height);
  generatePattern(initialData, width, height, 3.14f);
  cudaMemcpy2D(d_buf1, step1, initialData.data(), width * sizeof(Npp32f), width * sizeof(Npp32f), height,
               cudaMemcpyHostToDevice);

  NppiSize roi = {width, height};

  // Chain of copies: buf1 -> buf2 -> buf3
  NppStatus status1 = nppiCopy_32f_C1R(d_buf1, step1, d_buf2, step2, roi);
  NppStatus status2 = nppiCopy_32f_C1R(d_buf2, step2, d_buf3, step3, roi);

  ASSERT_EQ(status1, NPP_SUCCESS);
  ASSERT_EQ(status2, NPP_SUCCESS);

  // Verify final result
  std::vector<Npp32f> finalData(width * height);
  cudaMemcpy2D(finalData.data(), width * sizeof(Npp32f), d_buf3, step3, width * sizeof(Npp32f), height,
               cudaMemcpyDeviceToHost);

  EXPECT_TRUE(verifyData(initialData, finalData, 0.001f));

  nppiFree(d_buf1);
  nppiFree(d_buf2);
  nppiFree(d_buf3);
}

// Test 11: Special Float Values Handling
TEST_F(NppiCopy32fC1RTest, SpecialFloatValuesHandling) {
  const int width = 64, height = 64;

  Npp32f *d_src = nullptr;
  Npp32f *d_dst = nullptr;
  int srcStep, dstStep;
  allocateImage(&d_src, width, height, &srcStep);
  allocateImage(&d_dst, width, height, &dstStep);

  std::vector<Npp32f> srcData(width * height);

  // Fill with special float values including infinity and extreme values
  for (int i = 0; i < width * height; i++) {
    switch (i % 8) {
    case 0:
      srcData[i] = 0.0f;
      break;
    case 1:
      srcData[i] = -0.0f;
      break;
    case 2:
      srcData[i] = std::numeric_limits<float>::max();
      break;
    case 3:
      srcData[i] = std::numeric_limits<float>::lowest();
      break;
    case 4:
      srcData[i] = std::numeric_limits<float>::infinity();
      break;
    case 5:
      srcData[i] = -std::numeric_limits<float>::infinity();
      break;
    case 6:
      srcData[i] = std::numeric_limits<float>::epsilon();
      break;
    case 7:
      srcData[i] = std::numeric_limits<float>::denorm_min();
      break;
    }
  }

  cudaMemcpy2D(d_src, srcStep, srcData.data(), width * sizeof(Npp32f), width * sizeof(Npp32f), height,
               cudaMemcpyHostToDevice);

  NppiSize roi = {width, height};
  NppStatus status = nppiCopy_32f_C1R(d_src, srcStep, d_dst, dstStep, roi);
  ASSERT_EQ(status, NPP_SUCCESS);

  std::vector<Npp32f> dstData(width * height);
  cudaMemcpy2D(dstData.data(), width * sizeof(Npp32f), d_dst, dstStep, width * sizeof(Npp32f), height,
               cudaMemcpyDeviceToHost);

  // Verify special values are preserved
  for (int i = 0; i < width * height; i++) {
    if (std::isfinite(srcData[i])) {
      EXPECT_FLOAT_EQ(dstData[i], srcData[i]) << "Finite value mismatch at index " << i;
    } else {
      EXPECT_EQ(std::signbit(dstData[i]), std::signbit(srcData[i])) << "Sign mismatch at index " << i;
      EXPECT_EQ(std::isinf(dstData[i]), std::isinf(srcData[i])) << "Infinity state mismatch at index " << i;
    }
  }

  nppiFree(d_src);
  nppiFree(d_dst);
}

// Test 12: Stream Context Performance
TEST_F(NppiCopy32fC1RTest, StreamContextPerformance) {
  const int width = 1024, height = 1024;
  const int iterations = 10;

  Npp32f *d_src = nullptr;
  Npp32f *d_dst = nullptr;
  int srcStep, dstStep;
  allocateImage(&d_src, width, height, &srcStep);
  allocateImage(&d_dst, width, height, &dstStep);

  std::vector<Npp32f> srcData(width * height);
  generatePattern(srcData, width, height, 1.5f);
  cudaMemcpy2D(d_src, srcStep, srcData.data(), width * sizeof(Npp32f), width * sizeof(Npp32f), height,
               cudaMemcpyHostToDevice);

  NppiSize roi = {width, height};
  NppStreamContext ctx;
  nppGetStreamContext(&ctx);

  // Measure performance with stream context
  auto start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < iterations; i++) {
    NppStatus status = nppiCopy_32f_C1R_Ctx(d_src, srcStep, d_dst, dstStep, roi, ctx);
    ASSERT_EQ(status, NPP_SUCCESS);
  }
  cudaStreamSynchronize(ctx.hStream);
  auto end = std::chrono::high_resolution_clock::now();

  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
  double avgTime = duration / (double)iterations;
  double dataSize = width * height * sizeof(Npp32f) * 2 / 1024.0 / 1024.0; // MB (read + write)
  double bandwidth = dataSize * 1000000.0 / avgTime;                       // MB/s

  std::cout << "Stream context copy performance:" << std::endl;
  std::cout << "  Average time: " << avgTime << " μs" << std::endl;
  std::cout << "  Bandwidth: " << bandwidth << " MB/s" << std::endl;

  // Verify correctness
  std::vector<Npp32f> dstData(width * height);
  cudaMemcpy2D(dstData.data(), width * sizeof(Npp32f), d_dst, dstStep, width * sizeof(Npp32f), height,
               cudaMemcpyDeviceToHost);
  EXPECT_TRUE(verifyData(srcData, dstData));

  nppiFree(d_src);
  nppiFree(d_dst);
}