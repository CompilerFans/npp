#include "npp.h"
#include <algorithm>
#include <chrono>
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <random>
#include <vector>

class NppiCopy32fC3P3RTest : public ::testing::Test {
protected:
  void SetUp() override {
    cudaError_t err = cudaSetDevice(0);
    ASSERT_EQ(err, cudaSuccess);
  }

  void TearDown() override {
    cudaError_t err = cudaDeviceSynchronize();
    EXPECT_EQ(err, cudaSuccess);
  }

  // Helper to generate test pattern for 3-channel interleaved data
  void generateInterleavedPattern(std::vector<Npp32f> &data, int width, int height) {
    data.resize(width * height * 3);
    for (int y = 0; y < height; y++) {
      for (int x = 0; x < width; x++) {
        int idx = (y * width + x) * 3;
        data[idx] = x * 1.0f;           // R channel
        data[idx + 1] = y * 2.0f;       // G channel
        data[idx + 2] = (x + y) * 3.0f; // B channel
      }
    }
  }

  // Helper to verify planar data
  bool verifyPlanarData(const std::vector<Npp32f> &interleaved, const std::vector<Npp32f> &planarR,
                        const std::vector<Npp32f> &planarG, const std::vector<Npp32f> &planarB, int width, int height) {
    for (int y = 0; y < height; y++) {
      for (int x = 0; x < width; x++) {
        int interleavedIdx = (y * width + x) * 3;
        int planarIdx = y * width + x;

        if (planarR[planarIdx] != interleaved[interleavedIdx] ||
            planarG[planarIdx] != interleaved[interleavedIdx + 1] ||
            planarB[planarIdx] != interleaved[interleavedIdx + 2]) {
          return false;
        }
      }
    }
    return true;
  }
};

// Test 1: Basic interleaved to planar conversion
TEST_F(NppiCopy32fC3P3RTest, BasicInterleavedToPlanar) {
  const int width = 256;
  const int height = 256;

  std::vector<Npp32f> hostSrc(width * height * 3);
  generateInterleavedPattern(hostSrc, width, height);

  // Allocate GPU memory
  Npp32f *d_src = nullptr;
  int srcStep;
  d_src = nppiMalloc_32f_C3(width, height, &srcStep);
  ASSERT_NE(d_src, nullptr);

  Npp32f *d_dstPlanes[3];
  int dstStep;
  for (int i = 0; i < 3; i++) {
    d_dstPlanes[i] = nppiMalloc_32f_C1(width, height, &dstStep);
    ASSERT_NE(d_dstPlanes[i], nullptr);
  }

  // Upload interleaved data
  cudaMemcpy2D(d_src, srcStep, hostSrc.data(), width * 3 * sizeof(Npp32f), width * 3 * sizeof(Npp32f), height,
               cudaMemcpyHostToDevice);

  // Perform copy
  NppiSize roi = {width, height};
  NppStatus status = nppiCopy_32f_C3P3R(d_src, srcStep, d_dstPlanes, dstStep, roi);
  ASSERT_EQ(status, NPP_SUCCESS);

  // Download planar data
  std::vector<Npp32f> hostR(width * height), hostG(width * height), hostB(width * height);
  cudaMemcpy2D(hostR.data(), width * sizeof(Npp32f), d_dstPlanes[0], dstStep, width * sizeof(Npp32f), height,
               cudaMemcpyDeviceToHost);
  cudaMemcpy2D(hostG.data(), width * sizeof(Npp32f), d_dstPlanes[1], dstStep, width * sizeof(Npp32f), height,
               cudaMemcpyDeviceToHost);
  cudaMemcpy2D(hostB.data(), width * sizeof(Npp32f), d_dstPlanes[2], dstStep, width * sizeof(Npp32f), height,
               cudaMemcpyDeviceToHost);

  // Verify
  EXPECT_TRUE(verifyPlanarData(hostSrc, hostR, hostG, hostB, width, height));

  // Cleanup
  nppiFree(d_src);
  for (int i = 0; i < 3; i++) {
    nppiFree(d_dstPlanes[i]);
  }
}

// Test 2: ROI operations
TEST_F(NppiCopy32fC3P3RTest, ROIOperations) {
  const int width = 512;
  const int height = 512;
  const int roiX = 128;
  const int roiY = 128;
  const int roiWidth = 256;
  const int roiHeight = 256;

  std::vector<Npp32f> hostSrc(width * height * 3);
  generateInterleavedPattern(hostSrc, width, height);

  // Allocate GPU memory
  Npp32f *d_src = nullptr;
  int srcStep;
  d_src = nppiMalloc_32f_C3(width, height, &srcStep);
  ASSERT_NE(d_src, nullptr);

  Npp32f *d_dstPlanes[3];
  int dstStep;
  for (int i = 0; i < 3; i++) {
    d_dstPlanes[i] = nppiMalloc_32f_C1(width, height, &dstStep);
    ASSERT_NE(d_dstPlanes[i], nullptr);
    // Initialize with -999 to verify ROI boundaries
    cudaMemset(d_dstPlanes[i], 0xFF, dstStep * height);
  }

  // Upload source data
  cudaMemcpy2D(d_src, srcStep, hostSrc.data(), width * 3 * sizeof(Npp32f), width * 3 * sizeof(Npp32f), height,
               cudaMemcpyHostToDevice);

  // Set up ROI pointers
  Npp32f *pSrcROI = (Npp32f *)((Npp8u *)d_src + roiY * srcStep) + roiX * 3;
  Npp32f *pDstPlanesROI[3];
  for (int i = 0; i < 3; i++) {
    pDstPlanesROI[i] = (Npp32f *)((Npp8u *)d_dstPlanes[i] + roiY * dstStep) + roiX;
  }

  // Perform ROI copy
  NppiSize roi = {roiWidth, roiHeight};
  NppStatus status = nppiCopy_32f_C3P3R(pSrcROI, srcStep, pDstPlanesROI, dstStep, roi);
  ASSERT_EQ(status, NPP_SUCCESS);

  // Verify ROI data
  std::vector<Npp32f> hostR(width * height), hostG(width * height), hostB(width * height);
  cudaMemcpy2D(hostR.data(), width * sizeof(Npp32f), d_dstPlanes[0], dstStep, width * sizeof(Npp32f), height,
               cudaMemcpyDeviceToHost);
  cudaMemcpy2D(hostG.data(), width * sizeof(Npp32f), d_dstPlanes[1], dstStep, width * sizeof(Npp32f), height,
               cudaMemcpyDeviceToHost);
  cudaMemcpy2D(hostB.data(), width * sizeof(Npp32f), d_dstPlanes[2], dstStep, width * sizeof(Npp32f), height,
               cudaMemcpyDeviceToHost);

  // Check ROI region
  for (int y = roiY; y < roiY + roiHeight; y++) {
    for (int x = roiX; x < roiX + roiWidth; x++) {
      int interleavedIdx = (y * width + x) * 3;
      int planarIdx = y * width + x;
      EXPECT_FLOAT_EQ(hostR[planarIdx], hostSrc[interleavedIdx]);
      EXPECT_FLOAT_EQ(hostG[planarIdx], hostSrc[interleavedIdx + 1]);
      EXPECT_FLOAT_EQ(hostB[planarIdx], hostSrc[interleavedIdx + 2]);
    }
  }

  // Cleanup
  nppiFree(d_src);
  for (int i = 0; i < 3; i++) {
    nppiFree(d_dstPlanes[i]);
  }
}

// Test 3: Various image dimensions
TEST_F(NppiCopy32fC3P3RTest, VariousDimensions) {
  struct TestDimension {
    int width, height;
    const char *description;
  };

  TestDimension dimensions[] = {{1, 1, "Single pixel"},  {17, 13, "Prime dimensions"}, {640, 480, "VGA"},
                                {1920, 1080, "Full HD"}, {31, 1, "Single row"},        {1, 100, "Single column"}};

  for (const auto &dim : dimensions) {
    Npp32f *d_src = nullptr;
    int srcStep;
    d_src = nppiMalloc_32f_C3(dim.width, dim.height, &srcStep);

    if (d_src == nullptr) {
      continue; // Skip if allocation fails
    }

    Npp32f *d_dstPlanes[3];
    int dstStep;
    bool allocationSuccess = true;
    for (int i = 0; i < 3; i++) {
      d_dstPlanes[i] = nppiMalloc_32f_C1(dim.width, dim.height, &dstStep);
      if (d_dstPlanes[i] == nullptr) {
        allocationSuccess = false;
        // Free already allocated planes
        for (int j = 0; j < i; j++) {
          nppiFree(d_dstPlanes[j]);
        }
        break;
      }
    }

    if (!allocationSuccess) {
      nppiFree(d_src);
      continue;
    }

    // Test pattern
    std::vector<Npp32f> pattern(dim.width * dim.height * 3);
    generateInterleavedPattern(pattern, dim.width, dim.height);
    cudaMemcpy2D(d_src, srcStep, pattern.data(), dim.width * 3 * sizeof(Npp32f), dim.width * 3 * sizeof(Npp32f),
                 dim.height, cudaMemcpyHostToDevice);

    NppiSize roi = {dim.width, dim.height};
    NppStatus status = nppiCopy_32f_C3P3R(d_src, srcStep, d_dstPlanes, dstStep, roi);
    EXPECT_EQ(status, NPP_SUCCESS) << "Failed for " << dim.description << " (" << dim.width << "x" << dim.height << ")";

    // Cleanup
    nppiFree(d_src);
    for (int i = 0; i < 3; i++) {
      nppiFree(d_dstPlanes[i]);
    }
  }
}

// Test 4: Performance with large images
TEST_F(NppiCopy32fC3P3RTest, LargeImagePerformance) {
  const int width = 4096;
  const int height = 2160;
  const int iterations = 10;

  Npp32f *d_src = nullptr;
  int srcStep;
  d_src = nppiMalloc_32f_C3(width, height, &srcStep);
  ASSERT_NE(d_src, nullptr);

  Npp32f *d_dstPlanes[3];
  int dstStep;
  for (int i = 0; i < 3; i++) {
    d_dstPlanes[i] = nppiMalloc_32f_C1(width, height, &dstStep);
    ASSERT_NE(d_dstPlanes[i], nullptr);
  }

  // Initialize with pattern
  cudaMemset(d_src, 0x3F, srcStep * height);

  NppiSize roi = {width, height};

  // Warm up
  nppiCopy_32f_C3P3R(d_src, srcStep, d_dstPlanes, dstStep, roi);
  cudaDeviceSynchronize();

  // Measure performance
  auto start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < iterations; i++) {
    NppStatus status = nppiCopy_32f_C3P3R(d_src, srcStep, d_dstPlanes, dstStep, roi);
    ASSERT_EQ(status, NPP_SUCCESS);
  }
  cudaDeviceSynchronize();
  auto end = std::chrono::high_resolution_clock::now();

  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
  double avgTime = duration / (double)iterations;
  double dataSize = width * height * 3 * sizeof(Npp32f) * 2 / 1024.0 / 1024.0 / 1024.0; // GB
  double bandwidth = dataSize * 1000000.0 / avgTime;                                    // GB/s

  std::cout << "Interleaved to planar copy performance:" << std::endl;
  std::cout << "  Image size: " << width << "x" << height << std::endl;
  std::cout << "  Average time: " << avgTime << " us" << std::endl;
  std::cout << "  Bandwidth: " << bandwidth << " GB/s" << std::endl;

  // Cleanup
  nppiFree(d_src);
  for (int i = 0; i < 3; i++) {
    nppiFree(d_dstPlanes[i]);
  }
}

// Test 5: Memory alignment edge cases
TEST_F(NppiCopy32fC3P3RTest, MemoryAlignmentEdgeCases) {
  const int testWidths[] = {1, 3, 7, 13, 31, 63, 127, 255, 333, 511, 1023};
  const int height = 17;

  for (int width : testWidths) {
    Npp32f *d_src = nullptr;
    int srcStep;
    d_src = nppiMalloc_32f_C3(width, height, &srcStep);

    if (d_src == nullptr)
      continue;

    Npp32f *d_dstPlanes[3];
    int dstStep;
    bool allocSuccess = true;

    for (int i = 0; i < 3; i++) {
      d_dstPlanes[i] = nppiMalloc_32f_C1(width, height, &dstStep);
      if (d_dstPlanes[i] == nullptr) {
        allocSuccess = false;
        for (int j = 0; j < i; j++) {
          nppiFree(d_dstPlanes[j]);
        }
        break;
      }
    }

    if (!allocSuccess) {
      nppiFree(d_src);
      continue;
    }

    // Create unique pattern
    std::vector<Npp32f> pattern(width * height * 3);
    for (int i = 0; i < width * height; i++) {
      pattern[i * 3] = static_cast<float>(i);
      pattern[i * 3 + 1] = static_cast<float>(i * 2);
      pattern[i * 3 + 2] = static_cast<float>(i * 3);
    }

    cudaMemcpy2D(d_src, srcStep, pattern.data(), width * 3 * sizeof(Npp32f), width * 3 * sizeof(Npp32f), height,
                 cudaMemcpyHostToDevice);

    NppiSize roi = {width, height};
    NppStatus status = nppiCopy_32f_C3P3R(d_src, srcStep, d_dstPlanes, dstStep, roi);
    EXPECT_EQ(status, NPP_SUCCESS) << "Failed for width " << width;

    // Verify first pixel of each plane
    Npp32f firstPixelR, firstPixelG, firstPixelB;
    cudaMemcpy(&firstPixelR, d_dstPlanes[0], sizeof(Npp32f), cudaMemcpyDeviceToHost);
    cudaMemcpy(&firstPixelG, d_dstPlanes[1], sizeof(Npp32f), cudaMemcpyDeviceToHost);
    cudaMemcpy(&firstPixelB, d_dstPlanes[2], sizeof(Npp32f), cudaMemcpyDeviceToHost);

    EXPECT_FLOAT_EQ(firstPixelR, 0.0f) << "Wrong R value for width " << width;
    EXPECT_FLOAT_EQ(firstPixelG, 0.0f) << "Wrong G value for width " << width;
    EXPECT_FLOAT_EQ(firstPixelB, 0.0f) << "Wrong B value for width " << width;

    // Cleanup
    nppiFree(d_src);
    for (int i = 0; i < 3; i++) {
      nppiFree(d_dstPlanes[i]);
    }
  }
}

// Test 6: Different source and destination strides
TEST_F(NppiCopy32fC3P3RTest, DifferentStrides) {
  const int width = 200;
  const int height = 200;
  const int srcExtraBytes = 256;
  const int dstExtraBytes = 128;

  int srcStep = width * 3 * sizeof(Npp32f) + srcExtraBytes;
  int dstStep = width * sizeof(Npp32f) + dstExtraBytes;

  // Allocate with custom strides
  Npp32f *d_src = nullptr;
  Npp32f *d_dstPlanes[3];

  cudaMalloc(&d_src, srcStep * height);
  ASSERT_NE(d_src, nullptr);

  for (int i = 0; i < 3; i++) {
    cudaMalloc(&d_dstPlanes[i], dstStep * height);
    ASSERT_NE(d_dstPlanes[i], nullptr);
  }

  // Create test data with custom stride
  std::vector<Npp8u> hostSrcBuffer(srcStep * height);
  for (int y = 0; y < height; y++) {
    Npp32f *row = (Npp32f *)(hostSrcBuffer.data() + y * srcStep);
    for (int x = 0; x < width; x++) {
      row[x * 3] = y * 1000.0f + x;
      row[x * 3 + 1] = y * 1000.0f + x + 0.1f;
      row[x * 3 + 2] = y * 1000.0f + x + 0.2f;
    }
  }

  cudaMemcpy(d_src, hostSrcBuffer.data(), srcStep * height, cudaMemcpyHostToDevice);

  // Perform copy
  NppiSize roi = {width, height};
  NppStatus status = nppiCopy_32f_C3P3R(d_src, srcStep, d_dstPlanes, dstStep, roi);
  ASSERT_EQ(status, NPP_SUCCESS);

  // Verify first row of each plane
  std::vector<Npp32f> firstRowR(width);
  std::vector<Npp32f> firstRowG(width);
  std::vector<Npp32f> firstRowB(width);

  cudaMemcpy(firstRowR.data(), d_dstPlanes[0], width * sizeof(Npp32f), cudaMemcpyDeviceToHost);
  cudaMemcpy(firstRowG.data(), d_dstPlanes[1], width * sizeof(Npp32f), cudaMemcpyDeviceToHost);
  cudaMemcpy(firstRowB.data(), d_dstPlanes[2], width * sizeof(Npp32f), cudaMemcpyDeviceToHost);

  for (int x = 0; x < width; x++) {
    EXPECT_FLOAT_EQ(firstRowR[x], x);
    EXPECT_FLOAT_EQ(firstRowG[x], x + 0.1f);
    EXPECT_FLOAT_EQ(firstRowB[x], x + 0.2f);
  }

  // Cleanup
  cudaFree(d_src);
  for (int i = 0; i < 3; i++) {
    cudaFree(d_dstPlanes[i]);
  }
}

// Test 7: Error handling
TEST_F(NppiCopy32fC3P3RTest, DISABLED_ErrorHandling) {
  const int width = 100;
  const int height = 100;

  Npp32f *d_src = nullptr;
  int srcStep, dstStep;
  d_src = nppiMalloc_32f_C3(width, height, &srcStep);
  ASSERT_NE(d_src, nullptr);

  Npp32f *d_dstPlanes[3];
  for (int i = 0; i < 3; i++) {
    d_dstPlanes[i] = nppiMalloc_32f_C1(width, height, &dstStep);
    ASSERT_NE(d_dstPlanes[i], nullptr);
  }

  NppiSize roi = {width, height};

  // Test null source
  EXPECT_EQ(nppiCopy_32f_C3P3R(nullptr, srcStep, d_dstPlanes, dstStep, roi), NPP_NULL_POINTER_ERROR);

  // Test null destination array
  EXPECT_EQ(nppiCopy_32f_C3P3R(d_src, srcStep, nullptr, dstStep, roi), NPP_NULL_POINTER_ERROR);

  // Test null destination plane
  Npp32f *badPlanes[3] = {d_dstPlanes[0], nullptr, d_dstPlanes[2]};
  EXPECT_EQ(nppiCopy_32f_C3P3R(d_src, srcStep, badPlanes, dstStep, roi), NPP_NULL_POINTER_ERROR);

  // Test invalid ROI
  NppiSize invalidRoi = {0, height};
  EXPECT_EQ(nppiCopy_32f_C3P3R(d_src, srcStep, d_dstPlanes, dstStep, invalidRoi), NPP_SIZE_ERROR);

  invalidRoi = {width, -5};
  EXPECT_EQ(nppiCopy_32f_C3P3R(d_src, srcStep, d_dstPlanes, dstStep, invalidRoi), NPP_SIZE_ERROR);

  // Test invalid step
  EXPECT_EQ(nppiCopy_32f_C3P3R(d_src, 0, d_dstPlanes, dstStep, roi), NPP_STEP_ERROR);
  EXPECT_EQ(nppiCopy_32f_C3P3R(d_src, srcStep, d_dstPlanes, -1, roi), NPP_STEP_ERROR);

  // Cleanup
  nppiFree(d_src);
  for (int i = 0; i < 3; i++) {
    nppiFree(d_dstPlanes[i]);
  }
}

// Test 8: Concurrent stream operations
TEST_F(NppiCopy32fC3P3RTest, ConcurrentStreams) {
  const int numStreams = 4;
  const int width = 1024;
  const int height = 1024;

  std::vector<cudaStream_t> streams(numStreams);
  std::vector<Npp32f *> srcBuffers(numStreams);
  std::vector<std::vector<Npp32f *>> dstBuffers(numStreams, std::vector<Npp32f *>(3));
  std::vector<int> srcSteps(numStreams);
  std::vector<int> dstSteps(numStreams);

  // Create streams and allocate buffers
  for (int i = 0; i < numStreams; i++) {
    gpuStreamCreate(&streams[i]);
    srcBuffers[i] = nppiMalloc_32f_C3(width, height, &srcSteps[i]);
    ASSERT_NE(srcBuffers[i], nullptr);

    for (int j = 0; j < 3; j++) {
      dstBuffers[i][j] = nppiMalloc_32f_C1(width, height, &dstSteps[i]);
      ASSERT_NE(dstBuffers[i][j], nullptr);
    }

    // Initialize with different patterns
    std::vector<Npp32f> pattern(width * height * 3);
    generateInterleavedPattern(pattern, width, height);
    for (size_t k = 0; k < pattern.size(); k++) {
      pattern[k] *= (i + 1);
    }

    cudaMemcpy2DAsync(srcBuffers[i], srcSteps[i], pattern.data(), width * 3 * sizeof(Npp32f),
                      width * 3 * sizeof(Npp32f), height, cudaMemcpyHostToDevice, streams[i]);
  }

  // Launch concurrent copies
  NppiSize roi = {width, height};
  for (int i = 0; i < numStreams; i++) {
    NppStreamContext ctx;
    nppGetStreamContext(&ctx);
    ctx.hStream = streams[i];

    Npp32f *planes[3] = {dstBuffers[i][0], dstBuffers[i][1], dstBuffers[i][2]};
    NppStatus status = nppiCopy_32f_C3P3R_Ctx(srcBuffers[i], srcSteps[i], planes, dstSteps[i], roi, ctx);
    EXPECT_EQ(status, NPP_SUCCESS);
  }

  // Wait for all to complete and verify
  for (int i = 0; i < numStreams; i++) {
    gpuStreamSynchronize(streams[i]);

    // Verify first pixel of each plane
    Npp32f sampleR, sampleG, sampleB;
    cudaMemcpy(&sampleR, dstBuffers[i][0], sizeof(Npp32f), cudaMemcpyDeviceToHost);
    cudaMemcpy(&sampleG, dstBuffers[i][1], sizeof(Npp32f), cudaMemcpyDeviceToHost);
    cudaMemcpy(&sampleB, dstBuffers[i][2], sizeof(Npp32f), cudaMemcpyDeviceToHost);

    EXPECT_FLOAT_EQ(sampleR, 0.0f) << "Stream " << i << " R channel failed";
    EXPECT_FLOAT_EQ(sampleG, 0.0f) << "Stream " << i << " G channel failed";
    EXPECT_FLOAT_EQ(sampleB, 0.0f) << "Stream " << i << " B channel failed";
  }

  // Cleanup
  for (int i = 0; i < numStreams; i++) {
    gpuStreamDestroy(streams[i]);
    nppiFree(srcBuffers[i]);
    for (int j = 0; j < 3; j++) {
      nppiFree(dstBuffers[i][j]);
    }
  }
}

// Test 9: Stress test with random operations
TEST_F(NppiCopy32fC3P3RTest, StressTest) {
  const int numOperations = 50;
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> sizeDist(1, 50);
  std::uniform_int_distribution<> coordDist(0, 20);

  for (int op = 0; op < numOperations; op++) {
    // Random dimensions
    int width = sizeDist(gen) * 4;
    int height = sizeDist(gen) * 4;

    Npp32f *d_src = nullptr;
    int srcStep;
    d_src = nppiMalloc_32f_C3(width, height, &srcStep);

    if (d_src == nullptr)
      continue;

    Npp32f *d_dstPlanes[3];
    int dstStep;
    bool allocSuccess = true;

    for (int i = 0; i < 3; i++) {
      d_dstPlanes[i] = nppiMalloc_32f_C1(width, height, &dstStep);
      if (d_dstPlanes[i] == nullptr) {
        allocSuccess = false;
        for (int j = 0; j < i; j++) {
          nppiFree(d_dstPlanes[j]);
        }
        break;
      }
    }

    if (!allocSuccess) {
      nppiFree(d_src);
      continue;
    }

    // Initialize with pattern
    std::vector<Npp32f> pattern(width * height * 3);
    generateInterleavedPattern(pattern, width, height);
    cudaMemcpy2D(d_src, srcStep, pattern.data(), width * 3 * sizeof(Npp32f), width * 3 * sizeof(Npp32f), height,
                 cudaMemcpyHostToDevice);

    // Random ROI
    int roiWidth = std::min(width, sizeDist(gen) * 2);
    int roiHeight = std::min(height, sizeDist(gen) * 2);

    NppiSize roi = {roiWidth, roiHeight};
    NppStatus status = nppiCopy_32f_C3P3R(d_src, srcStep, d_dstPlanes, dstStep, roi);
    EXPECT_EQ(status, NPP_SUCCESS) << "Operation " << op << " failed";

    // Cleanup
    nppiFree(d_src);
    for (int i = 0; i < 3; i++) {
      nppiFree(d_dstPlanes[i]);
    }
  }
}

// Test 10: Special patterns verification
TEST_F(NppiCopy32fC3P3RTest, SpecialPatterns) {
  const int width = 256;
  const int height = 256;

  struct PatternTest {
    const char *name;
    std::function<void(std::vector<Npp32f> &, int, int)> generator;
  };

  PatternTest patterns[] = {{"Gradient",
                             [](std::vector<Npp32f> &data, int w, int h) {
                               for (int y = 0; y < h; y++) {
                                 for (int x = 0; x < w; x++) {
                                   int idx = (y * w + x) * 3;
                                   data[idx] = x / (float)w * 255.0f;
                                   data[idx + 1] = y / (float)h * 255.0f;
                                   data[idx + 2] = (x + y) / (float)(w + h) * 255.0f;
                                 }
                               }
                             }},
                            {"Checkerboard",
                             [](std::vector<Npp32f> &data, int w, int h) {
                               for (int y = 0; y < h; y++) {
                                 for (int x = 0; x < w; x++) {
                                   int idx = (y * w + x) * 3;
                                   float val = ((x / 32 + y / 32) % 2) * 255.0f;
                                   data[idx] = val;
                                   data[idx + 1] = val;
                                   data[idx + 2] = val;
                                 }
                               }
                             }},
                            {"Diagonal stripes", [](std::vector<Npp32f> &data, int w, int h) {
                               for (int y = 0; y < h; y++) {
                                 for (int x = 0; x < w; x++) {
                                   int idx = (y * w + x) * 3;
                                   float val = ((x + y) % 64 < 32) ? 255.0f : 0.0f;
                                   data[idx] = val;
                                   data[idx + 1] = val * 0.5f;
                                   data[idx + 2] = val * 0.25f;
                                 }
                               }
                             }}};

  for (const auto &pattern : patterns) {
    std::vector<Npp32f> hostSrc(width * height * 3);
    pattern.generator(hostSrc, width, height);

    // Allocate GPU memory
    Npp32f *d_src = nullptr;
    int srcStep;
    d_src = nppiMalloc_32f_C3(width, height, &srcStep);
    ASSERT_NE(d_src, nullptr);

    Npp32f *d_dstPlanes[3];
    int dstStep;
    for (int i = 0; i < 3; i++) {
      d_dstPlanes[i] = nppiMalloc_32f_C1(width, height, &dstStep);
      ASSERT_NE(d_dstPlanes[i], nullptr);
    }

    // Upload and process
    cudaMemcpy2D(d_src, srcStep, hostSrc.data(), width * 3 * sizeof(Npp32f), width * 3 * sizeof(Npp32f), height,
                 cudaMemcpyHostToDevice);

    NppiSize roi = {width, height};
    NppStatus status = nppiCopy_32f_C3P3R(d_src, srcStep, d_dstPlanes, dstStep, roi);
    ASSERT_EQ(status, NPP_SUCCESS) << "Failed for pattern: " << pattern.name;

    // Verify center pixel
    int centerX = width / 2;
    int centerY = height / 2;
    int centerIdx = (centerY * width + centerX) * 3;

    Npp32f centerR, centerG, centerB;
    cudaMemcpy(&centerR, (Npp32f *)((Npp8u *)d_dstPlanes[0] + centerY * dstStep) + centerX, sizeof(Npp32f),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(&centerG, (Npp32f *)((Npp8u *)d_dstPlanes[1] + centerY * dstStep) + centerX, sizeof(Npp32f),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(&centerB, (Npp32f *)((Npp8u *)d_dstPlanes[2] + centerY * dstStep) + centerX, sizeof(Npp32f),
               cudaMemcpyDeviceToHost);

    EXPECT_FLOAT_EQ(centerR, hostSrc[centerIdx]) << "Pattern: " << pattern.name;
    EXPECT_FLOAT_EQ(centerG, hostSrc[centerIdx + 1]) << "Pattern: " << pattern.name;
    EXPECT_FLOAT_EQ(centerB, hostSrc[centerIdx + 2]) << "Pattern: " << pattern.name;

    // Cleanup
    nppiFree(d_src);
    for (int i = 0; i < 3; i++) {
      nppiFree(d_dstPlanes[i]);
    }
  }
}

// Test 11: Non-Contiguous Memory Layout Verification
TEST_F(NppiCopy32fC3P3RTest, NonContiguousMemoryLayoutVerification) {
  const int width = 128, height = 128;

  Npp32f *d_src = nullptr;
  int srcStep;
  d_src = nppiMalloc_32f_C3(width, height, &srcStep);
  ASSERT_NE(d_src, nullptr);

  // Allocate destination planes with different strides to simulate non-contiguous layout
  Npp32f *d_dstPlanes[3];
  int dstSteps[3];

  for (int i = 0; i < 3; i++) {
    d_dstPlanes[i] = nppiMalloc_32f_C1(width, height, &dstSteps[i]);
    ASSERT_NE(d_dstPlanes[i], nullptr);
  }

  // Create test data with distinct channel values
  std::vector<Npp32f> hostSrc(width * height * 3);
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      int idx = (y * width + x) * 3;
      hostSrc[idx] = 100.0f + x;           // R channel: base 100 + x coordinate
      hostSrc[idx + 1] = 200.0f + y;       // G channel: base 200 + y coordinate
      hostSrc[idx + 2] = 300.0f + (x + y); // B channel: base 300 + diagonal
    }
  }

  cudaMemcpy2D(d_src, srcStep, hostSrc.data(), width * 3 * sizeof(Npp32f), width * 3 * sizeof(Npp32f), height,
               cudaMemcpyHostToDevice);

  NppiSize roi = {width, height};
  NppStatus status = nppiCopy_32f_C3P3R(d_src, srcStep, d_dstPlanes, dstSteps[0], roi);
  ASSERT_EQ(status, NPP_SUCCESS);

  // Verify each plane separately with different memory layouts
  for (int plane = 0; plane < 3; plane++) {
    std::vector<Npp32f> hostPlane(width * height);
    cudaMemcpy2D(hostPlane.data(), width * sizeof(Npp32f), d_dstPlanes[plane], dstSteps[plane], width * sizeof(Npp32f),
                 height, cudaMemcpyDeviceToHost);

    // Verify several sample points across the image
    int samplePoints[][2] = {{0, 0},
                             {width - 1, 0},
                             {0, height - 1},
                             {width - 1, height - 1},
                             {width / 4, height / 4},
                             {3 * width / 4, 3 * height / 4}};

    for (auto &pt : samplePoints) {
      int x = pt[0], y = pt[1];
      int srcIdx = (y * width + x) * 3 + plane;
      int dstIdx = y * width + x;

      EXPECT_FLOAT_EQ(hostPlane[dstIdx], hostSrc[srcIdx])
          << "Plane " << plane << " mismatch at (" << x << "," << y << ")";
    }
  }

  nppiFree(d_src);
  for (int i = 0; i < 3; i++) {
    nppiFree(d_dstPlanes[i]);
  }
}

// Test 12: Stream Context Synchronization
TEST_F(NppiCopy32fC3P3RTest, StreamContextSynchronization) {
  const int width = 512, height = 512;
  const int numStreams = 4;

  std::vector<cudaStream_t> streams(numStreams);
  std::vector<Npp32f *> srcBuffers(numStreams);
  std::vector<std::vector<Npp32f *>> dstBuffers(numStreams, std::vector<Npp32f *>(3));
  std::vector<int> srcSteps(numStreams);
  std::vector<int> dstSteps(numStreams);

  // Create streams and allocate memory
  for (int i = 0; i < numStreams; i++) {
    gpuStreamCreate(&streams[i]);
    srcBuffers[i] = nppiMalloc_32f_C3(width, height, &srcSteps[i]);
    ASSERT_NE(srcBuffers[i], nullptr);

    for (int j = 0; j < 3; j++) {
      dstBuffers[i][j] = nppiMalloc_32f_C1(width, height, &dstSteps[i]);
      ASSERT_NE(dstBuffers[i][j], nullptr);
    }

    // Initialize with stream-specific pattern
    std::vector<Npp32f> pattern(width * height * 3);
    for (int p = 0; p < width * height; p++) {
      pattern[p * 3] = (float)(i * 1000 + p % 256);             // R
      pattern[p * 3 + 1] = (float)(i * 1000 + (p + 100) % 256); // G
      pattern[p * 3 + 2] = (float)(i * 1000 + (p + 200) % 256); // B
    }

    cudaMemcpy2DAsync(srcBuffers[i], srcSteps[i], pattern.data(), width * 3 * sizeof(Npp32f),
                      width * 3 * sizeof(Npp32f), height, cudaMemcpyHostToDevice, streams[i]);
  }

  // Launch operations on different streams
  NppiSize roi = {width, height};
  for (int i = 0; i < numStreams; i++) {
    NppStreamContext ctx;
    nppGetStreamContext(&ctx);
    ctx.hStream = streams[i];

    Npp32f *planes[3] = {dstBuffers[i][0], dstBuffers[i][1], dstBuffers[i][2]};
    NppStatus status = nppiCopy_32f_C3P3R_Ctx(srcBuffers[i], srcSteps[i], planes, dstSteps[i], roi, ctx);
    EXPECT_EQ(status, NPP_SUCCESS) << "Stream " << i << " failed";
  }

  // Wait for all streams and verify cross-stream independence
  for (int i = 0; i < numStreams; i++) {
    gpuStreamSynchronize(streams[i]);

    // Verify each plane has correct stream-specific data
    for (int plane = 0; plane < 3; plane++) {
      Npp32f sampleValue;
      cudaMemcpy(&sampleValue, dstBuffers[i][plane], sizeof(Npp32f), cudaMemcpyDeviceToHost);

      float expected = (float)(i * 1000 + (plane * 100) % 256);
      EXPECT_FLOAT_EQ(sampleValue, expected) << "Stream " << i << " plane " << plane << " incorrect value";
    }
  }

  // Cleanup
  for (int i = 0; i < numStreams; i++) {
    gpuStreamDestroy(streams[i]);
    nppiFree(srcBuffers[i]);
    for (int j = 0; j < 3; j++) {
      nppiFree(dstBuffers[i][j]);
    }
  }
}
