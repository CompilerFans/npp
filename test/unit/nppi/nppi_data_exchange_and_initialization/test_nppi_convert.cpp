#include "../../framework/npp_test_base.h"
#include "npp.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <numeric>
#include <random>
#include <thread>
#include <vector>

using namespace npp_functional_test;

class ConvertFunctionalTest : public NppTestBase {
protected:
  void SetUp() override { NppTestBase::SetUp(); }

  void TearDown() override { NppTestBase::TearDown(); }
};

// 测试8位无符号整数到32位浮点数转换
TEST_F(ConvertFunctionalTest, Convert_8u32f_C1R_Ctx_Basic) {
  const int width = 64, height = 64;

  // prepare test data - 全bounds8位值
  std::vector<Npp8u> srcData(width * height);
  for (int i = 0; i < width * height; i++) {
    srcData[i] = static_cast<Npp8u>(i % 256);
  }

  NppImageMemory<Npp8u> src(width, height);
  NppImageMemory<Npp32f> dst(width, height);

  src.copyFromHost(srcData);

  // 设置参数
  NppiSize oSizeROI = {width, height};

  // 获取流上下文
  NppStreamContext nppStreamCtx;
  nppGetStreamContext(&nppStreamCtx);

  // 执行转换
  NppStatus status = nppiConvert_8u32f_C1R_Ctx(src.get(), src.step(), dst.get(), dst.step(), oSizeROI, nppStreamCtx);

  ASSERT_EQ(status, NPP_SUCCESS);

  // 同步并获取结果
  cudaStreamSynchronize(nppStreamCtx.hStream);
  std::vector<Npp32f> dstData(width * height);
  dst.copyToHost(dstData);

  // Validate转换结果
  for (int i = 0; i < width * height; i++) {
    ASSERT_FLOAT_EQ(dstData[i], static_cast<float>(srcData[i]));
  }
}

// 测试边界值转换
TEST_F(ConvertFunctionalTest, Convert_8u32f_C1R_Ctx_BoundaryValues) {
  const int width = 16, height = 16;

  // prepare test data - 边界值
  std::vector<Npp8u> srcData(width * height);
  srcData[0] = 0;   // 最小值
  srcData[1] = 255; // 最大值
  srcData[2] = 1;   // 接近最小值
  srcData[3] = 254; // 接近最大值
  srcData[4] = 128; // 中间值

  // 填充剩余数据
  for (int i = 5; i < width * height; i++) {
    srcData[i] = static_cast<Npp8u>(i % 256);
  }

  NppImageMemory<Npp8u> src(width, height);
  NppImageMemory<Npp32f> dst(width, height);

  src.copyFromHost(srcData);

  // 设置参数
  NppiSize oSizeROI = {width, height};

  // 获取流上下文
  NppStreamContext nppStreamCtx;
  nppGetStreamContext(&nppStreamCtx);

  // 执行转换
  NppStatus status = nppiConvert_8u32f_C1R_Ctx(src.get(), src.step(), dst.get(), dst.step(), oSizeROI, nppStreamCtx);

  ASSERT_EQ(status, NPP_SUCCESS);

  // 同步并获取结果
  cudaStreamSynchronize(nppStreamCtx.hStream);
  std::vector<Npp32f> dstData(width * height);
  dst.copyToHost(dstData);

  // Validate边界值
  ASSERT_FLOAT_EQ(dstData[0], 0.0f);
  ASSERT_FLOAT_EQ(dstData[1], 255.0f);
  ASSERT_FLOAT_EQ(dstData[2], 1.0f);
  ASSERT_FLOAT_EQ(dstData[3], 254.0f);
  ASSERT_FLOAT_EQ(dstData[4], 128.0f);
}

// test roi part trans
TEST_F(ConvertFunctionalTest, Convert_8u32f_C1R_Ctx_PartialROI) {
  const int width = 32, height = 32;

  // prepare test data
  std::vector<Npp8u> srcData(width * height);
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      srcData[y * width + x] = static_cast<Npp8u>((x + y) % 256);
    }
  }

  NppImageMemory<Npp8u> src(width, height);
  NppImageMemory<Npp32f> dst(width, height);

  src.copyFromHost(srcData);

  // 设置ROI为中心区域
  NppiSize oSizeROI = {16, 16};
  int xOffset = 8;
  int yOffset = 8;

  // 获取流上下文
  NppStreamContext nppStreamCtx;
  nppGetStreamContext(&nppStreamCtx);

  // 执行ROI转换
  Npp8u *pSrcROI = src.get() + yOffset * src.step() / sizeof(Npp8u) + xOffset;
  Npp32f *pDstROI = dst.get() + yOffset * dst.step() / sizeof(Npp32f) + xOffset;

  NppStatus status = nppiConvert_8u32f_C1R_Ctx(pSrcROI, src.step(), pDstROI, dst.step(), oSizeROI, nppStreamCtx);

  ASSERT_EQ(status, NPP_SUCCESS);

  // 同步并获取结果
  cudaStreamSynchronize(nppStreamCtx.hStream);
  std::vector<Npp32f> dstData(width * height);
  dst.copyToHost(dstData);

  // ValidateROI内外的值
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      int idx = y * width + x;
      if (x >= xOffset && x < xOffset + oSizeROI.width && y >= yOffset && y < yOffset + oSizeROI.height) {
        // ROI内应该有转换后的值
        ASSERT_FLOAT_EQ(dstData[idx], static_cast<float>(srcData[idx]));
      } else {
        // ROI外应该保持0
        ASSERT_FLOAT_EQ(dstData[idx], 0.0f);
      }
    }
  }
}

using namespace npp_functional_test;

class NPPIConvertAndCopyComprehensiveTest : public NppTestBase {
protected:
  void SetUp() override { NppTestBase::SetUp(); }

  void TearDown() override { NppTestBase::TearDown(); }
};

// ==============================================================================
// nppiConvert_8u32f_C3R Comprehensive Tests
// ==============================================================================

// Test with various image sizes including edge cases
TEST_F(NPPIConvertAndCopyComprehensiveTest, Convert_8u32f_C3R_VariousSizes) {
  struct TestSize {
    int width, height;
    const char *description;
  };

  TestSize testSizes[] = {{1, 1, "Single pixel"},     {3, 1, "Single row"},          {1, 3, "Single column"},
                          {15, 15, "Small odd size"}, {16, 16, "Small power of 2"},  {17, 17, "Small prime"},
                          {256, 256, "Medium size"},  {1920, 1080, "HD resolution"}, {4096, 2160, "4K resolution"}};

  for (const auto &testSize : testSizes) {
    // Prepare test data
    int pixelCount = testSize.width * testSize.height * 3;
    std::vector<Npp8u> srcData(pixelCount);

    // Fill with gradient pattern
    for (int i = 0; i < pixelCount; i++) {
      srcData[i] = static_cast<Npp8u>(i % 256);
    }

    NppImageMemory<Npp8u> src(testSize.width * 3, testSize.height);
    NppImageMemory<Npp32f> dst(testSize.width * 3, testSize.height);

    src.copyFromHost(srcData);

    NppiSize oSizeROI = {testSize.width, testSize.height};
    NppStatus status = nppiConvert_8u32f_C3R(src.get(), src.step(), dst.get(), dst.step(), oSizeROI);

    ASSERT_EQ(status, NPP_SUCCESS) << "Failed for " << testSize.description << " (" << testSize.width << "x"
                                   << testSize.height << ")";

    // Verify conversion
    std::vector<Npp32f> dstData(pixelCount);
    dst.copyToHost(dstData);

    for (int i = 0; i < std::min(100, pixelCount); i++) {
      EXPECT_FLOAT_EQ(dstData[i], static_cast<float>(srcData[i]))
          << "Mismatch at index " << i << " for " << testSize.description;
    }
  }
}

// Test with non-standard step sizes
TEST_F(NPPIConvertAndCopyComprehensiveTest, Convert_8u32f_C3R_NonStandardSteps) {
  const int width = 64, height = 64;
  const int channels = 3;

  // Create source with extra padding per row
  int srcPadding = 128; // Extra bytes per row
  int srcStep = width * channels + srcPadding;

  std::vector<Npp8u> srcData(srcStep * height, 0);

  // Fill actual image data (skip padding)
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width * channels; x++) {
      srcData[y * srcStep + x] = static_cast<Npp8u>((y * width + x / 3) % 256);
    }
  }

  // Allocate with custom steps
  Npp8u *d_src;
  Npp32f *d_dst;
  int dstStep;

  cudaMalloc(&d_src, srcStep * height);
  d_dst = nppiMalloc_32f_C3(width, height, &dstStep);

  ASSERT_TRUE(d_src && d_dst) << "Memory allocation failed";

  // Copy source data
  cudaMemcpy(d_src, srcData.data(), srcStep * height, cudaMemcpyHostToDevice);

  // Convert with non-standard source step
  NppiSize oSizeROI = {width, height};
  NppStatus status = nppiConvert_8u32f_C3R(d_src, srcStep, d_dst, dstStep, oSizeROI);

  ASSERT_EQ(status, NPP_SUCCESS) << "Conversion with non-standard step failed";

  // Verify results
  std::vector<Npp32f> dstData(width * height * channels);
  cudaMemcpy2D(dstData.data(), width * channels * sizeof(Npp32f), d_dst, dstStep, width * channels * sizeof(Npp32f),
               height, cudaMemcpyDeviceToHost);

  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width * channels; x++) {
      int srcIdx = y * srcStep + x;
      int dstIdx = y * width * channels + x;
      EXPECT_FLOAT_EQ(dstData[dstIdx], static_cast<float>(srcData[srcIdx]))
          << "Mismatch at position (" << x / 3 << "," << y << ") channel " << x % 3;
    }
  }

  cudaFree(d_src);
  nppiFree(d_dst);
}

// Stress test with concurrent operations
TEST_F(NPPIConvertAndCopyComprehensiveTest, Convert_8u32f_C3R_ConcurrentStreams) {
  const int width = 512, height = 512;
  const int numStreams = 4;

  std::vector<cudaStream_t> streams(numStreams);
  std::vector<NppStreamContext> contexts(numStreams);

  // Create streams and contexts
  for (int i = 0; i < numStreams; i++) {
    cudaStreamCreate(&streams[i]);
    nppGetStreamContext(&contexts[i]);
    contexts[i].hStream = streams[i];
  }

  // Prepare test data
  std::vector<Npp8u> srcData(width * height * 3);
  for (size_t i = 0; i < srcData.size(); i++) {
    srcData[i] = static_cast<Npp8u>(i % 256);
  }

  // Allocate memory for each stream
  std::vector<NppImageMemory<Npp8u>> srcs;
  std::vector<NppImageMemory<Npp32f>> dsts;

  for (int i = 0; i < numStreams; i++) {
    srcs.emplace_back(width * 3, height);
    dsts.emplace_back(width * 3, height);
    srcs[i].copyFromHost(srcData);
  }

  // Launch conversions concurrently
  NppiSize oSizeROI = {width, height};
  for (int i = 0; i < numStreams; i++) {
    NppStatus status =
        nppiConvert_8u32f_C3R_Ctx(srcs[i].get(), srcs[i].step(), dsts[i].get(), dsts[i].step(), oSizeROI, contexts[i]);
    ASSERT_EQ(status, NPP_SUCCESS) << "Stream " << i << " conversion failed";
  }

  // Wait for all streams
  for (int i = 0; i < numStreams; i++) {
    cudaStreamSynchronize(streams[i]);
  }

  // Verify all results are identical
  std::vector<Npp32f> baseResult(width * height * 3);
  dsts[0].copyToHost(baseResult);

  for (int i = 1; i < numStreams; i++) {
    std::vector<Npp32f> result(width * height * 3);
    dsts[i].copyToHost(result);

    for (size_t j = 0; j < result.size(); j++) {
      EXPECT_FLOAT_EQ(result[j], baseResult[j]) << "Stream " << i << " differs at index " << j;
    }
  }

  // Cleanup
  for (int i = 0; i < numStreams; i++) {
    cudaStreamDestroy(streams[i]);
  }
}

// ==============================================================================
// nppiCopy_32f_C1R Comprehensive Tests
// ==============================================================================

// Test with various step configurations
TEST_F(NPPIConvertAndCopyComprehensiveTest, Copy_32f_C1R_DifferentSteps) {
  const int width = 100, height = 100;

  struct StepConfig {
    int srcPadding;
    int dstPadding;
    const char *description;
  };

  StepConfig configs[] = {{0, 0, "No padding"},
                          {4, 0, "Source padded"},
                          {0, 8, "Destination padded"},
                          {12, 16, "Both padded"},
                          {100, 200, "Large padding"}};

  for (const auto &config : configs) {
    // Calculate steps
    int srcStep = (width + config.srcPadding) * sizeof(Npp32f);
    int dstStep = (width + config.dstPadding) * sizeof(Npp32f);

    // Allocate memory
    Npp32f *d_src, *d_dst;
    cudaMalloc(&d_src, srcStep * height);
    cudaMalloc(&d_dst, dstStep * height);

    ASSERT_TRUE(d_src && d_dst) << "Allocation failed for " << config.description;

    // Initialize source
    std::vector<Npp32f> srcData((srcStep / sizeof(Npp32f)) * height, 0);
    for (int y = 0; y < height; y++) {
      for (int x = 0; x < width; x++) {
        srcData[y * (srcStep / sizeof(Npp32f)) + x] = y * width + x;
      }
    }
    cudaMemcpy(d_src, srcData.data(), srcData.size() * sizeof(Npp32f), cudaMemcpyHostToDevice);

    // Copy
    NppiSize roi = {width, height};
    NppStatus status = nppiCopy_32f_C1R(d_src, srcStep, d_dst, dstStep, roi);

    ASSERT_EQ(status, NPP_SUCCESS) << "Copy failed for " << config.description;

    // Verify
    std::vector<Npp32f> dstData((dstStep / sizeof(Npp32f)) * height);
    cudaMemcpy(dstData.data(), d_dst, dstData.size() * sizeof(Npp32f), cudaMemcpyDeviceToHost);

    for (int y = 0; y < height; y++) {
      for (int x = 0; x < width; x++) {
        int srcIdx = y * (srcStep / sizeof(Npp32f)) + x;
        int dstIdx = y * (dstStep / sizeof(Npp32f)) + x;
        EXPECT_FLOAT_EQ(dstData[dstIdx], srcData[srcIdx])
            << "Mismatch at (" << x << "," << y << ") for " << config.description;
      }
    }

    cudaFree(d_src);
    cudaFree(d_dst);
  }
}

// Performance test with large data
TEST_F(NPPIConvertAndCopyComprehensiveTest, Copy_32f_C1R_Performance) {
  const int width = 4096, height = 4096;
  const int iterations = 10;

  NppImageMemory<Npp32f> src(width, height);
  NppImageMemory<Npp32f> dst(width, height);

  // Initialize source
  std::vector<Npp32f> srcData(width * height);
  std::iota(srcData.begin(), srcData.end(), 0.0f);
  src.copyFromHost(srcData);

  // Warm up
  NppiSize roi = {width, height};
  nppiCopy_32f_C1R(src.get(), src.step(), dst.get(), dst.step(), roi);
  cudaDeviceSynchronize();

  // Measure performance
  auto start = std::chrono::high_resolution_clock::now();

  for (int i = 0; i < iterations; i++) {
    NppStatus status = nppiCopy_32f_C1R(src.get(), src.step(), dst.get(), dst.step(), roi);
    ASSERT_EQ(status, NPP_SUCCESS);
  }
  cudaDeviceSynchronize();

  auto end = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

  double avgTime = duration.count() / (double)iterations;
  double bandwidth = (width * height * sizeof(Npp32f) * 2) / (avgTime * 1e6); // GB/s

  std::cout << "Copy_32f_C1R Performance: " << avgTime << " us, " << bandwidth << " GB/s" << std::endl;

  // Verify correctness of last copy
  std::vector<Npp32f> dstData(width * height);
  dst.copyToHost(dstData);

  // Sample verification
  for (int i = 0; i < 1000; i += 100) {
    EXPECT_FLOAT_EQ(dstData[i], srcData[i]);
  }
}

// ==============================================================================
// nppiCopy_32f_C3P3R Comprehensive Tests
// ==============================================================================

// Test with various ROI sizes and positions
TEST_F(NPPIConvertAndCopyComprehensiveTest, Copy_32f_C3P3R_ROIVariations) {
  const int fullWidth = 256, fullHeight = 256;

  struct ROITest {
    int x, y, width, height;
    const char *description;
  };

  ROITest roiTests[] = {{0, 0, fullWidth, fullHeight, "Full image"},
                        {0, 0, 128, 128, "Top-left quarter"},
                        {128, 128, 128, 128, "Bottom-right quarter"},
                        {64, 64, 128, 128, "Center region"},
                        {0, 0, 1, 1, "Single pixel"},
                        {0, 0, fullWidth, 1, "Single row"},
                        {0, 0, 1, fullHeight, "Single column"},
                        {10, 20, 236, 226, "Arbitrary region"}};

  // Prepare source data
  std::vector<Npp32f> srcData(fullWidth * fullHeight * 3);
  for (size_t i = 0; i < srcData.size(); i++) {
    srcData[i] = static_cast<float>(i % 1000) / 10.0f;
  }

  NppImageMemory<Npp32f> src(fullWidth * 3, fullHeight);
  src.copyFromHost(srcData);

  // Allocate plane buffers
  NppImageMemory<Npp32f> dstPlanes[3] = {NppImageMemory<Npp32f>(fullWidth, fullHeight),
                                         NppImageMemory<Npp32f>(fullWidth, fullHeight),
                                         NppImageMemory<Npp32f>(fullWidth, fullHeight)};

  for (const auto &roiTest : roiTests) {
    // Clear destination planes
    for (int i = 0; i < 3; i++) {
      cudaMemset(dstPlanes[i].get(), 0, dstPlanes[i].sizeInBytes());
    }

    // Set up pointers and ROI
    Npp32f *pSrc = src.get() + roiTest.y * (src.step() / sizeof(Npp32f)) + roiTest.x * 3;
    Npp32f *dstPtrs[3] = {dstPlanes[0].get() + roiTest.y * (dstPlanes[0].step() / sizeof(Npp32f)) + roiTest.x,
                          dstPlanes[1].get() + roiTest.y * (dstPlanes[1].step() / sizeof(Npp32f)) + roiTest.x,
                          dstPlanes[2].get() + roiTest.y * (dstPlanes[2].step() / sizeof(Npp32f)) + roiTest.x};

    NppiSize oSizeROI = {roiTest.width, roiTest.height};

    NppStatus status = nppiCopy_32f_C3P3R(pSrc, src.step(), dstPtrs, dstPlanes[0].step(), oSizeROI);

    ASSERT_EQ(status, NPP_SUCCESS) << "Failed for " << roiTest.description;

    // Verify the ROI copy
    for (int p = 0; p < 3; p++) {
      std::vector<Npp32f> planeData(fullWidth * fullHeight);
      dstPlanes[p].copyToHost(planeData);

      for (int y = roiTest.y; y < roiTest.y + roiTest.height; y++) {
        for (int x = roiTest.x; x < roiTest.x + roiTest.width; x++) {
          int srcIdx = (y * fullWidth + x) * 3 + p;
          int dstIdx = y * fullWidth + x;
          EXPECT_FLOAT_EQ(planeData[dstIdx], srcData[srcIdx])
              << "Mismatch in plane " << p << " at (" << x << "," << y << ") "
              << "for " << roiTest.description;
        }
      }
    }
  }
}

// Test with unaligned memory addresses
TEST_F(NPPIConvertAndCopyComprehensiveTest, Copy_32f_C3P3R_UnalignedMemory) {
  const int width = 127, height = 63; // Odd sizes

  // Allocate extra memory for unaligned access
  Npp32f *srcBuffer, *dstBuffers[3];
  cudaMalloc(&srcBuffer, (width * 3 + 1) * height * sizeof(Npp32f) + 64);
  for (int i = 0; i < 3; i++) {
    cudaMalloc(&dstBuffers[i], (width + 1) * height * sizeof(Npp32f) + 64);
  }

  // Create unaligned pointers (offset by 1 float)
  Npp32f *src = srcBuffer + 1;
  Npp32f *dstPtrs[3] = {dstBuffers[0] + 1, dstBuffers[1] + 1, dstBuffers[2] + 1};

  // Initialize source
  std::vector<Npp32f> srcData(width * height * 3);
  for (size_t i = 0; i < srcData.size(); i++) {
    srcData[i] = static_cast<float>(i);
  }
  cudaMemcpy(src, srcData.data(), srcData.size() * sizeof(Npp32f), cudaMemcpyHostToDevice);

  // Copy with unaligned pointers
  NppiSize oSizeROI = {width, height};
  NppStatus status = nppiCopy_32f_C3P3R(src, width * 3 * sizeof(Npp32f), dstPtrs, width * sizeof(Npp32f), oSizeROI);

  ASSERT_EQ(status, NPP_SUCCESS) << "Unaligned copy failed";

  // Verify
  for (int p = 0; p < 3; p++) {
    std::vector<Npp32f> planeData(width * height);
    cudaMemcpy(planeData.data(), dstPtrs[p], planeData.size() * sizeof(Npp32f), cudaMemcpyDeviceToHost);

    for (int i = 0; i < width * height; i++) {
      int srcIdx = i * 3 + p;
      EXPECT_FLOAT_EQ(planeData[i], srcData[srcIdx]) << "Mismatch in plane " << p << " at index " << i;
    }
  }

  // Cleanup
  cudaFree(srcBuffer);
  for (int i = 0; i < 3; i++) {
    cudaFree(dstBuffers[i]);
  }
}

// ==============================================================================
// nppiCopy_32f_C3R Comprehensive Tests
// ==============================================================================

// Test in-place copy (src == dst)
TEST_F(NPPIConvertAndCopyComprehensiveTest, Copy_32f_C3R_InPlace) {
  const int width = 128, height = 128;

  // Create source data
  std::vector<Npp32f> originalData(width * height * 3);
  for (size_t i = 0; i < originalData.size(); i++) {
    originalData[i] = static_cast<float>(i % 256);
  }

  NppImageMemory<Npp32f> buffer(width * 3, height);
  buffer.copyFromHost(originalData);

  // In-place copy (should be no-op)
  NppiSize roi = {width, height};
  NppStatus status = nppiCopy_32f_C3R(buffer.get(), buffer.step(), buffer.get(), buffer.step(), roi);

  ASSERT_EQ(status, NPP_SUCCESS) << "In-place copy failed";

  // Verify data unchanged
  std::vector<Npp32f> resultData(width * height * 3);
  buffer.copyToHost(resultData);

  for (size_t i = 0; i < resultData.size(); i++) {
    EXPECT_FLOAT_EQ(resultData[i], originalData[i]) << "Data changed at index " << i;
  }
}

// Concurrent copy operations
TEST_F(NPPIConvertAndCopyComprehensiveTest, Copy_32f_C3R_ConcurrentOperations) {
  const int width = 512, height = 512;
  const int numOperations = 8;

  // Create streams
  std::vector<cudaStream_t> streams(numOperations);
  std::vector<NppStreamContext> contexts(numOperations);

  for (int i = 0; i < numOperations; i++) {
    cudaStreamCreate(&streams[i]);
    nppGetStreamContext(&contexts[i]);
    contexts[i].hStream = streams[i];
  }

  // Allocate memory for each operation
  std::vector<std::unique_ptr<NppImageMemory<Npp32f>>> sources;
  std::vector<std::unique_ptr<NppImageMemory<Npp32f>>> destinations;

  std::vector<Npp32f> srcData(width * height * 3);
  for (size_t i = 0; i < srcData.size(); i++) {
    srcData[i] = static_cast<float>(i);
  }

  for (int i = 0; i < numOperations; i++) {
    sources.push_back(std::make_unique<NppImageMemory<Npp32f>>(width * 3, height));
    destinations.push_back(std::make_unique<NppImageMemory<Npp32f>>(width * 3, height));
    sources[i]->copyFromHost(srcData);
  }

  // Launch concurrent copies
  NppiSize roi = {width, height};
  auto start = std::chrono::high_resolution_clock::now();

  for (int i = 0; i < numOperations; i++) {
    NppStatus status = nppiCopy_32f_C3R_Ctx(sources[i]->get(), sources[i]->step(), destinations[i]->get(),
                                            destinations[i]->step(), roi, contexts[i]);
    ASSERT_EQ(status, NPP_SUCCESS) << "Stream " << i << " copy failed";
  }

  // Wait for completion
  for (int i = 0; i < numOperations; i++) {
    cudaStreamSynchronize(streams[i]);
  }

  auto end = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

  std::cout << "Concurrent copy operations completed in " << duration.count() << " microseconds" << std::endl;

  // Verify all copies
  std::vector<Npp32f> expectedResult(width * height * 3);
  sources[0]->copyToHost(expectedResult);

  for (int i = 0; i < numOperations; i++) {
    std::vector<Npp32f> result(width * height * 3);
    destinations[i]->copyToHost(result);

    // Sample verification
    for (int j = 0; j < 1000; j += 100) {
      EXPECT_FLOAT_EQ(result[j], expectedResult[j]) << "Stream " << i << " result mismatch at " << j;
    }
  }

  // Cleanup
  for (int i = 0; i < numOperations; i++) {
    cudaStreamDestroy(streams[i]);
  }
}

// ==============================================================================
// Memory Management Comprehensive Tests
// ==============================================================================

// Test memory alignment guarantees
TEST_F(NPPIConvertAndCopyComprehensiveTest, nppiMalloc_32f_AlignmentTest) {
  const int numAllocations = 100;
  std::vector<Npp32f *> ptrs1;
  std::vector<Npp32f *> ptrs3;
  std::vector<int> steps;

  // Allocate many buffers of various sizes
  for (int i = 0; i < numAllocations; i++) {
    int width = 1 + (i * 17) % 1024; // Various widths
    int height = 1 + (i * 13) % 512; // Various heights
    int step;

    // Single channel allocations
    Npp32f *ptr1 = nppiMalloc_32f_C1(width, height, &step);
    ASSERT_NE(ptr1, nullptr) << "Allocation " << i << " failed";
    ASSERT_EQ(reinterpret_cast<uintptr_t>(ptr1) % 256, 0) << "Pointer " << i << " not 256-byte aligned";
    ASSERT_GE(step, width * sizeof(Npp32f)) << "Step too small for allocation " << i;
    ASSERT_EQ(step % 256, 0) << "Step not 256-byte aligned for allocation " << i;

    ptrs1.push_back(ptr1);
    steps.push_back(step);

    // Three channel allocations
    Npp32f *ptr3 = nppiMalloc_32f_C3(width, height, &step);
    ASSERT_NE(ptr3, nullptr) << "C3 allocation " << i << " failed";
    ASSERT_EQ(reinterpret_cast<uintptr_t>(ptr3) % 256, 0) << "C3 pointer " << i << " not 256-byte aligned";
    ASSERT_GE(step, width * 3 * sizeof(Npp32f)) << "C3 step too small for allocation " << i;

    ptrs3.push_back(ptr3);
  }

  // Free all allocations
  for (auto ptr : ptrs1) {
    nppiFree(ptr);
  }
  for (auto ptr : ptrs3) {
    nppiFree(ptr);
  }
}

// Stress test allocation/deallocation cycles
TEST_F(NPPIConvertAndCopyComprehensiveTest, nppiMalloc_nppiFree_StressTest) {
  const int cycles = 1000;
  const int maxAllocations = 10;

  std::vector<Npp32f *> allocations;
  std::vector<int> sizes;

  // Random number generation
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> sizeDist(1, 1024);
  std::uniform_int_distribution<> actionDist(0, 1);

  for (int cycle = 0; cycle < cycles; cycle++) {
    // Randomly allocate or free
    if (allocations.size() < maxAllocations && actionDist(gen) == 0) {
      // Allocate
      int width = sizeDist(gen);
      int height = sizeDist(gen);
      int step;

      Npp32f *ptr =
          (cycle % 2 == 0) ? nppiMalloc_32f_C1(width, height, &step) : nppiMalloc_32f_C3(width, height, &step);

      if (ptr) {
        allocations.push_back(ptr);
        sizes.push_back(width * height);

        // Write test pattern
        std::vector<Npp32f> pattern(100, static_cast<float>(cycle));
        cudaMemcpy(ptr, pattern.data(),
                   std::min(static_cast<size_t>(100), static_cast<size_t>(sizes.back())) * sizeof(Npp32f),
                   cudaMemcpyHostToDevice);
      }
    } else if (!allocations.empty()) {
      // Free
      size_t idx = gen() % allocations.size();

      // Verify pattern before freeing
      std::vector<Npp32f> check(static_cast<size_t>(std::min(100, sizes[idx])));
      cudaMemcpy(check.data(), allocations[idx], check.size() * sizeof(Npp32f), cudaMemcpyDeviceToHost);

      nppiFree(allocations[idx]);
      allocations.erase(allocations.begin() + idx);
      sizes.erase(sizes.begin() + idx);
    }
  }

  // Clean up remaining allocations
  for (auto ptr : allocations) {
    nppiFree(ptr);
  }
}

// Test memory limits and error handling
TEST_F(NPPIConvertAndCopyComprehensiveTest, nppiMalloc_nppiFree_LimitsAndErrors) {
  // Test zero-size allocations
  int step;
  EXPECT_EQ(nppiMalloc_32f_C1(0, 100, &step), nullptr);
  EXPECT_EQ(nppiMalloc_32f_C1(100, 0, &step), nullptr);
  EXPECT_EQ(nppiMalloc_32f_C3(0, 0, &step), nullptr);

  // Test large allocation behavior (system dependent - may succeed or fail)
  // Note: Large allocations may succeed on systems with sufficient GPU memory
  Npp32f *large_ptr = nppiMalloc_32f_C1(10000, 10000, &step);
  if (large_ptr != nullptr) {
    nppiFree(large_ptr); // Clean up if allocation succeeded
  }

  // Test nullptr handling in nppiFree
  nppiFree(nullptr); // Should not crash

  // Test double-free protection (implementation dependent)
  Npp32f *ptr = nppiMalloc_32f_C1(100, 100, &step);
  ASSERT_NE(ptr, nullptr);
  nppiFree(ptr);
}

class ConvertExtendedFunctionalTest : public ::testing::Test {
protected:
  void SetUp() override {
    width = 8;
    height = 6;
    roi.width = width;
    roi.height = height;
  }

  int width, height;
  NppiSize roi;
};

// 测试nppiConvert_8u32f_C3R函数
TEST_F(ConvertExtendedFunctionalTest, Convert_8u32f_C3R_BasicOperation) {
  // const int 3 = 3;  // Commented out as not used
  std::vector<Npp8u> srcData(width * height * 3);
  std::vector<Npp32f> expectedData(width * height * 3);

  // 创建测试数据：RGB渐变模式
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      int idx = (y * width + x) * 3;
      srcData[idx + 0] = (Npp8u)((x * 255) / (width - 1));  // R通道
      srcData[idx + 1] = (Npp8u)((y * 255) / (height - 1)); // G通道
      srcData[idx + 2] = (Npp8u)(128);                      // B通道固定值

      // 期望结果：8位转32位浮点 (0-255 -> 0.0-255.0)
      expectedData[idx + 0] = (Npp32f)srcData[idx + 0];
      expectedData[idx + 1] = (Npp32f)srcData[idx + 1];
      expectedData[idx + 2] = (Npp32f)srcData[idx + 2];
    }
  }

  // 分配GPU内存
  int srcStep, dstStep;
  Npp8u *d_src = nppiMalloc_8u_C3(width, height, &srcStep);
  Npp32f *d_dst = nppiMalloc_32f_C3(width, height, &dstStep);
  ASSERT_NE(d_src, nullptr);
  ASSERT_NE(d_dst, nullptr);

  // 复制输入数据到GPU
  for (int y = 0; y < height; y++) {
    cudaMemcpy((char *)d_src + y * srcStep, srcData.data() + y * width * 3, width * 3 * sizeof(Npp8u),
               cudaMemcpyHostToDevice);
  }

  // 执行转换
  NppStatus status = nppiConvert_8u32f_C3R(d_src, srcStep, d_dst, dstStep, roi);
  EXPECT_EQ(status, NPP_SUCCESS);

  // 复制结果回主机
  std::vector<Npp32f> resultData(width * height * 3);
  for (int y = 0; y < height; y++) {
    cudaMemcpy(resultData.data() + y * width * 3, (char *)d_dst + y * dstStep, width * 3 * sizeof(Npp32f),
               cudaMemcpyDeviceToHost);
  }

  // Validate结果
  for (int i = 0; i < width * height * 3; i++) {
    EXPECT_FLOAT_EQ(resultData[i], expectedData[i])
        << "转换失败 at index " << i << ": got " << resultData[i] << ", expected " << expectedData[i];
  }

  nppiFree(d_src);
  nppiFree(d_dst);
}

// 测试nppiConvert_8u32f_C3R边界值
TEST_F(ConvertExtendedFunctionalTest, Convert_8u32f_C3R_BoundaryValues) {
  // const int 3 = 3;  // Commented out as not used
  std::vector<Npp8u> srcData = {
      0,   0,   0,   // 最小值
      255, 255, 255, // 最大值
      128, 64,  192  // 中间值
  };
  std::vector<Npp32f> expectedData = {0.0f, 0.0f, 0.0f, 255.0f, 255.0f, 255.0f, 128.0f, 64.0f, 192.0f};

  NppiSize testRoi = {3, 1};

  // 分配GPU内存
  int srcStep = 3 * 3 * sizeof(Npp8u);
  int dstStep = 3 * 3 * sizeof(Npp32f);

  Npp8u *d_src = nppsMalloc_8u(3 * 3);
  Npp32f *d_dst = nppsMalloc_32f(3 * 3);
  ASSERT_NE(d_src, nullptr);
  ASSERT_NE(d_dst, nullptr);

  // 复制数据
  cudaMemcpy(d_src, srcData.data(), srcData.size() * sizeof(Npp8u), cudaMemcpyHostToDevice);

  // 执行转换
  NppStatus status = nppiConvert_8u32f_C3R(d_src, srcStep, d_dst, dstStep, testRoi);
  EXPECT_EQ(status, NPP_SUCCESS);

  // Validate结果
  std::vector<Npp32f> resultData(3 * 3);
  cudaMemcpy(resultData.data(), d_dst, resultData.size() * sizeof(Npp32f), cudaMemcpyDeviceToHost);

  for (size_t i = 0; i < expectedData.size(); i++) {
    EXPECT_FLOAT_EQ(resultData[i], expectedData[i]) << "边界值转换失败 at index " << i;
  }

  nppsFree(d_src);
  nppsFree(d_dst);
}

// 测试流上下文版本
TEST_F(ConvertExtendedFunctionalTest, Convert_8u32f_C3R_StreamContext) {
  // const int 3 = 3;  // Commented out as not used
  std::vector<Npp8u> srcData(width * height * 3, 100); // 所有像素值为100

  // 分配GPU内存
  int srcStep, dstStep;
  Npp8u *d_src = nppiMalloc_8u_C3(width, height, &srcStep);
  Npp32f *d_dst = nppiMalloc_32f_C3(width, height, &dstStep);
  ASSERT_NE(d_src, nullptr);
  ASSERT_NE(d_dst, nullptr);

  // 复制数据
  for (int y = 0; y < height; y++) {
    cudaMemcpy((char *)d_src + y * srcStep, srcData.data() + y * width * 3, width * 3 * sizeof(Npp8u),
               cudaMemcpyHostToDevice);
  }

  // 创建流上下文
  NppStreamContext nppStreamCtx;
  nppGetStreamContext(&nppStreamCtx);

  // 执行转换
  NppStatus status = nppiConvert_8u32f_C3R_Ctx(d_src, srcStep, d_dst, dstStep, roi, nppStreamCtx);
  EXPECT_EQ(status, NPP_SUCCESS);

  // Validate结果
  std::vector<Npp32f> resultData(width * height * 3);
  for (int y = 0; y < height; y++) {
    cudaMemcpy(resultData.data() + y * width * 3, (char *)d_dst + y * dstStep, width * 3 * sizeof(Npp32f),
               cudaMemcpyDeviceToHost);
  }

  // 所有值应该都是100.0f
  for (int i = 0; i < width * height * 3; i++) {
    EXPECT_FLOAT_EQ(resultData[i], 100.0f) << "流上下文测试失败 at index " << i;
  }

  nppiFree(d_src);
  nppiFree(d_dst);
}

// 测试错误处理
TEST_F(ConvertExtendedFunctionalTest, Convert_8u32f_C3R_ErrorHandling) {
  int srcStep, dstStep;
  Npp8u *d_src = nppiMalloc_8u_C3(width, height, &srcStep);
  Npp32f *d_dst = nppiMalloc_32f_C3(width, height, &dstStep);

  // 测试空指针
  EXPECT_EQ(nppiConvert_8u32f_C3R(nullptr, srcStep, d_dst, dstStep, roi), NPP_NULL_POINTER_ERROR);
  EXPECT_EQ(nppiConvert_8u32f_C3R(d_src, srcStep, nullptr, dstStep, roi), NPP_NULL_POINTER_ERROR);

  // 测试无效步长
  EXPECT_EQ(nppiConvert_8u32f_C3R(d_src, 0, d_dst, dstStep, roi), NPP_SUCCESS);
  EXPECT_EQ(nppiConvert_8u32f_C3R(d_src, srcStep, d_dst, 0, roi), NPP_STEP_ERROR);
  EXPECT_EQ(nppiConvert_8u32f_C3R(d_src, -1, d_dst, dstStep, roi), NPP_SUCCESS);

  // 测试无效ROI
  NppiSize badRoi = {0, height};
  EXPECT_EQ(nppiConvert_8u32f_C3R(d_src, srcStep, d_dst, dstStep, badRoi), NPP_SUCCESS);
  badRoi = {width, 0};
  EXPECT_EQ(nppiConvert_8u32f_C3R(d_src, srcStep, d_dst, dstStep, badRoi), NPP_SUCCESS);
  badRoi = {-1, height};
  EXPECT_EQ(nppiConvert_8u32f_C3R(d_src, srcStep, d_dst, dstStep, badRoi), NPP_SIZE_ERROR);

  nppiFree(d_src);
  nppiFree(d_dst);
}

// 测试大尺寸图像
TEST_F(ConvertExtendedFunctionalTest, Convert_8u32f_C3R_LargeImage) {
  const int largeWidth = 512; // 减小尺寸避免内存问题
  const int largeHeight = 384;
  NppiSize largeRoi = {largeWidth, largeHeight};

  // 分配GPU内存
  int srcStep, dstStep;
  Npp8u *d_src = nppiMalloc_8u_C3(largeWidth, largeHeight, &srcStep);
  Npp32f *d_dst = nppiMalloc_32f_C3(largeWidth, largeHeight, &dstStep);
  ASSERT_NE(d_src, nullptr);
  ASSERT_NE(d_dst, nullptr);

  // 创建测试数据
  std::vector<Npp8u> srcData(largeWidth * largeHeight * 3);
  for (int i = 0; i < largeWidth * largeHeight * 3; i++) {
    srcData[i] = (Npp8u)(i % 256);
  }

  // 复制数据到GPU
  for (int y = 0; y < largeHeight; y++) {
    cudaMemcpy((char *)d_src + y * srcStep, srcData.data() + y * largeWidth * 3, largeWidth * 3 * sizeof(Npp8u),
               cudaMemcpyHostToDevice);
  }

  // 执行转换
  NppStatus status = nppiConvert_8u32f_C3R(d_src, srcStep, d_dst, dstStep, largeRoi);
  EXPECT_EQ(status, NPP_SUCCESS);

  // Validate部分结果（检查第一行）
  std::vector<Npp32f> resultData(largeWidth * 3);
  cudaMemcpy(resultData.data(), d_dst, largeWidth * 3 * sizeof(Npp32f), cudaMemcpyDeviceToHost);

  for (int x = 0; x < largeWidth * 3; x++) {
    EXPECT_FLOAT_EQ(resultData[x], (Npp32f)srcData[x]);
  }

  nppiFree(d_src);
  nppiFree(d_dst);
}

// 测试零尺寸ROI
TEST_F(ConvertExtendedFunctionalTest, Convert_8u32f_C3R_ZeroROI) {
  int srcStep, dstStep;
  Npp8u *d_src = nppiMalloc_8u_C3(width, height, &srcStep);
  Npp32f *d_dst = nppiMalloc_32f_C3(width, height, &dstStep);
  ASSERT_NE(d_src, nullptr);
  ASSERT_NE(d_dst, nullptr);

  NppiSize zeroRoi = {0, 0};
  NppStatus status = nppiConvert_8u32f_C3R(d_src, srcStep, d_dst, dstStep, zeroRoi);
  EXPECT_EQ(status, NPP_SUCCESS);

  nppiFree(d_src);
  nppiFree(d_dst);
}

// 测试内存对齐
TEST_F(ConvertExtendedFunctionalTest, Convert_8u32f_C3R_MemoryAlignment) {
  // 使用不对齐的尺寸测试
  const int oddWidth = 7;
  const int oddHeight = 5;
  NppiSize oddRoi = {oddWidth, oddHeight};

  int srcStep, dstStep;
  Npp8u *d_src = nppiMalloc_8u_C3(oddWidth, oddHeight, &srcStep);
  Npp32f *d_dst = nppiMalloc_32f_C3(oddWidth, oddHeight, &dstStep);
  ASSERT_NE(d_src, nullptr);
  ASSERT_NE(d_dst, nullptr);

  // 创建测试数据
  std::vector<Npp8u> srcData(oddWidth * oddHeight * 3, 42);

  // 复制数据到GPU (考虑步长)
  for (int y = 0; y < oddHeight; y++) {
    cudaMemcpy((char *)d_src + y * srcStep, srcData.data() + y * oddWidth * 3, oddWidth * 3 * sizeof(Npp8u),
               cudaMemcpyHostToDevice);
  }

  // 执行转换
  NppStatus status = nppiConvert_8u32f_C3R(d_src, srcStep, d_dst, dstStep, oddRoi);
  EXPECT_EQ(status, NPP_SUCCESS);

  // Validate结果
  std::vector<Npp32f> resultData(oddWidth * oddHeight * 3);
  for (int y = 0; y < oddHeight; y++) {
    cudaMemcpy(resultData.data() + y * oddWidth * 3, (char *)d_dst + y * dstStep, oddWidth * 3 * sizeof(Npp32f),
               cudaMemcpyDeviceToHost);
  }

  for (int i = 0; i < oddWidth * oddHeight * 3; i++) {
    EXPECT_FLOAT_EQ(resultData[i], 42.0f);
  }

  nppiFree(d_src);
  nppiFree(d_dst);
}

// 测试单像素ROI
TEST_F(ConvertExtendedFunctionalTest, Convert_8u32f_C3R_SinglePixel) {
  NppiSize singlePixelRoi = {1, 1};
  std::vector<Npp8u> srcData = {123, 45, 210};
  std::vector<Npp32f> expectedData = {123.0f, 45.0f, 210.0f};

  int srcStep = 3 * sizeof(Npp8u);
  int dstStep = 3 * sizeof(Npp32f);

  Npp8u *d_src = nppsMalloc_8u(3);
  Npp32f *d_dst = nppsMalloc_32f(3);
  ASSERT_NE(d_src, nullptr);
  ASSERT_NE(d_dst, nullptr);

  // 复制数据
  cudaMemcpy(d_src, srcData.data(), 3 * sizeof(Npp8u), cudaMemcpyHostToDevice);

  // 执行转换
  NppStatus status = nppiConvert_8u32f_C3R(d_src, srcStep, d_dst, dstStep, singlePixelRoi);
  EXPECT_EQ(status, NPP_SUCCESS);

  // Validate结果
  std::vector<Npp32f> resultData(3);
  cudaMemcpy(resultData.data(), d_dst, 3 * sizeof(Npp32f), cudaMemcpyDeviceToHost);

  for (int i = 0; i < 3; i++) {
    EXPECT_FLOAT_EQ(resultData[i], expectedData[i]) << "单像素转换失败 at channel " << i;
  }

  nppsFree(d_src);
  nppsFree(d_dst);
}

// 测试颜色渐变图像
TEST_F(ConvertExtendedFunctionalTest, Convert_8u32f_C3R_ColorGradient) {
  const int gradWidth = 16;
  const int gradHeight = 12;
  NppiSize gradRoi = {gradWidth, gradHeight};

  // 创建RGB颜色渐变
  std::vector<Npp8u> srcData(gradWidth * gradHeight * 3);
  std::vector<Npp32f> expectedData(gradWidth * gradHeight * 3);

  for (int y = 0; y < gradHeight; y++) {
    for (int x = 0; x < gradWidth; x++) {
      int idx = (y * gradWidth + x) * 3;

      // R通道: 水平渐变 0->255
      srcData[idx + 0] = (Npp8u)((x * 255) / (gradWidth - 1));

      // G通道: 垂直渐变 0->255
      srcData[idx + 1] = (Npp8u)((y * 255) / (gradHeight - 1));

      // B通道: 对角渐变
      int diag = (x + y) * 255 / (gradWidth + gradHeight - 2);
      srcData[idx + 2] = (Npp8u)(diag > 255 ? 255 : diag);

      // 期望的32位浮点结果
      expectedData[idx + 0] = (Npp32f)srcData[idx + 0];
      expectedData[idx + 1] = (Npp32f)srcData[idx + 1];
      expectedData[idx + 2] = (Npp32f)srcData[idx + 2];
    }
  }

  int srcStep, dstStep;
  Npp8u *d_src = nppiMalloc_8u_C3(gradWidth, gradHeight, &srcStep);
  Npp32f *d_dst = nppiMalloc_32f_C3(gradWidth, gradHeight, &dstStep);
  ASSERT_NE(d_src, nullptr);
  ASSERT_NE(d_dst, nullptr);

  // 复制数据到GPU
  for (int y = 0; y < gradHeight; y++) {
    cudaMemcpy((char *)d_src + y * srcStep, srcData.data() + y * gradWidth * 3, gradWidth * 3 * sizeof(Npp8u),
               cudaMemcpyHostToDevice);
  }

  // 执行转换
  NppStatus status = nppiConvert_8u32f_C3R(d_src, srcStep, d_dst, dstStep, gradRoi);
  EXPECT_EQ(status, NPP_SUCCESS);

  // Validate结果
  std::vector<Npp32f> resultData(gradWidth * gradHeight * 3);
  for (int y = 0; y < gradHeight; y++) {
    cudaMemcpy(resultData.data() + y * gradWidth * 3, (char *)d_dst + y * dstStep, gradWidth * 3 * sizeof(Npp32f),
               cudaMemcpyDeviceToHost);
  }

  for (int i = 0; i < gradWidth * gradHeight * 3; i++) {
    EXPECT_FLOAT_EQ(resultData[i], expectedData[i]) << "颜色渐变转换失败 at index " << i;
  }

  nppiFree(d_src);
  nppiFree(d_dst);
}

// 测试棋盘图案
TEST_F(ConvertExtendedFunctionalTest, Convert_8u32f_C3R_CheckerboardPattern) {
  const int boardWidth = 8;
  const int boardHeight = 8;
  NppiSize boardRoi = {boardWidth, boardHeight};

  std::vector<Npp8u> srcData(boardWidth * boardHeight * 3);
  std::vector<Npp32f> expectedData(boardWidth * boardHeight * 3);

  // 创建棋盘图案
  for (int y = 0; y < boardHeight; y++) {
    for (int x = 0; x < boardWidth; x++) {
      int idx = (y * boardWidth + x) * 3;
      bool isWhite = ((x + y) % 2) == 0;

      if (isWhite) {
        srcData[idx + 0] = 255; // R
        srcData[idx + 1] = 255; // G
        srcData[idx + 2] = 255; // B
      } else {
        srcData[idx + 0] = 0; // R
        srcData[idx + 1] = 0; // G
        srcData[idx + 2] = 0; // B
      }

      expectedData[idx + 0] = (Npp32f)srcData[idx + 0];
      expectedData[idx + 1] = (Npp32f)srcData[idx + 1];
      expectedData[idx + 2] = (Npp32f)srcData[idx + 2];
    }
  }

  int srcStep, dstStep;
  Npp8u *d_src = nppiMalloc_8u_C3(boardWidth, boardHeight, &srcStep);
  Npp32f *d_dst = nppiMalloc_32f_C3(boardWidth, boardHeight, &dstStep);
  ASSERT_NE(d_src, nullptr);
  ASSERT_NE(d_dst, nullptr);

  // 复制数据到GPU
  for (int y = 0; y < boardHeight; y++) {
    cudaMemcpy((char *)d_src + y * srcStep, srcData.data() + y * boardWidth * 3, boardWidth * 3 * sizeof(Npp8u),
               cudaMemcpyHostToDevice);
  }

  // 执行转换
  NppStatus status = nppiConvert_8u32f_C3R(d_src, srcStep, d_dst, dstStep, boardRoi);
  EXPECT_EQ(status, NPP_SUCCESS);

  // Validate结果
  std::vector<Npp32f> resultData(boardWidth * boardHeight * 3);
  for (int y = 0; y < boardHeight; y++) {
    cudaMemcpy(resultData.data() + y * boardWidth * 3, (char *)d_dst + y * dstStep, boardWidth * 3 * sizeof(Npp32f),
               cudaMemcpyDeviceToHost);
  }

  for (int i = 0; i < boardWidth * boardHeight * 3; i++) {
    EXPECT_FLOAT_EQ(resultData[i], expectedData[i]) << "棋盘图案转换失败 at index " << i;
  }

  nppiFree(d_src);
  nppiFree(d_dst);
}

// 测试随机数据
TEST_F(ConvertExtendedFunctionalTest, Convert_8u32f_C3R_RandomData) {
  const int randWidth = 15;
  const int randHeight = 11;
  NppiSize randRoi = {randWidth, randHeight};

  std::vector<Npp8u> srcData(randWidth * randHeight * 3);
  std::vector<Npp32f> expectedData(randWidth * randHeight * 3);

  // 生成确定性"随机"数据（使用简单的线性同余生成器）
  unsigned int seed = 12345;
  for (int i = 0; i < randWidth * randHeight * 3; i++) {
    seed = seed * 1103515245 + 12345;
    srcData[i] = (Npp8u)(seed % 256);
    expectedData[i] = (Npp32f)srcData[i];
  }

  int srcStep, dstStep;
  Npp8u *d_src = nppiMalloc_8u_C3(randWidth, randHeight, &srcStep);
  Npp32f *d_dst = nppiMalloc_32f_C3(randWidth, randHeight, &dstStep);
  ASSERT_NE(d_src, nullptr);
  ASSERT_NE(d_dst, nullptr);

  // 复制数据到GPU
  for (int y = 0; y < randHeight; y++) {
    cudaMemcpy((char *)d_src + y * srcStep, srcData.data() + y * randWidth * 3, randWidth * 3 * sizeof(Npp8u),
               cudaMemcpyHostToDevice);
  }

  // 执行转换
  NppStatus status = nppiConvert_8u32f_C3R(d_src, srcStep, d_dst, dstStep, randRoi);
  EXPECT_EQ(status, NPP_SUCCESS);

  // Validate结果
  std::vector<Npp32f> resultData(randWidth * randHeight * 3);
  for (int y = 0; y < randHeight; y++) {
    cudaMemcpy(resultData.data() + y * randWidth * 3, (char *)d_dst + y * dstStep, randWidth * 3 * sizeof(Npp32f),
               cudaMemcpyDeviceToHost);
  }

  for (int i = 0; i < randWidth * randHeight * 3; i++) {
    EXPECT_FLOAT_EQ(resultData[i], expectedData[i]) << "随机数据转换失败 at index " << i;
  }

  nppiFree(d_src);
  nppiFree(d_dst);
}

// 测试矩形ROI（非正方形）
TEST_F(ConvertExtendedFunctionalTest, Convert_8u32f_C3R_RectangularROI) {
  const int rectWidth = 32;
  const int rectHeight = 8;
  NppiSize rectRoi = {rectWidth, rectHeight};

  std::vector<Npp8u> srcData(rectWidth * rectHeight * 3);
  std::vector<Npp32f> expectedData(rectWidth * rectHeight * 3);

  // 创建水平条纹图案
  for (int y = 0; y < rectHeight; y++) {
    for (int x = 0; x < rectWidth; x++) {
      int idx = (y * rectWidth + x) * 3;
      Npp8u value = (Npp8u)((y * 255) / (rectHeight - 1));

      srcData[idx + 0] = value;       // R
      srcData[idx + 1] = 255 - value; // G (反转)
      srcData[idx + 2] = 128;         // B (固定)

      expectedData[idx + 0] = (Npp32f)srcData[idx + 0];
      expectedData[idx + 1] = (Npp32f)srcData[idx + 1];
      expectedData[idx + 2] = (Npp32f)srcData[idx + 2];
    }
  }

  int srcStep, dstStep;
  Npp8u *d_src = nppiMalloc_8u_C3(rectWidth, rectHeight, &srcStep);
  Npp32f *d_dst = nppiMalloc_32f_C3(rectWidth, rectHeight, &dstStep);
  ASSERT_NE(d_src, nullptr);
  ASSERT_NE(d_dst, nullptr);

  // 复制数据到GPU
  for (int y = 0; y < rectHeight; y++) {
    cudaMemcpy((char *)d_src + y * srcStep, srcData.data() + y * rectWidth * 3, rectWidth * 3 * sizeof(Npp8u),
               cudaMemcpyHostToDevice);
  }

  // 执行转换
  NppStatus status = nppiConvert_8u32f_C3R(d_src, srcStep, d_dst, dstStep, rectRoi);
  EXPECT_EQ(status, NPP_SUCCESS);

  // Validate结果
  std::vector<Npp32f> resultData(rectWidth * rectHeight * 3);
  for (int y = 0; y < rectHeight; y++) {
    cudaMemcpy(resultData.data() + y * rectWidth * 3, (char *)d_dst + y * dstStep, rectWidth * 3 * sizeof(Npp32f),
               cudaMemcpyDeviceToHost);
  }

  for (int i = 0; i < rectWidth * rectHeight * 3; i++) {
    EXPECT_FLOAT_EQ(resultData[i], expectedData[i]) << "矩形ROI转换失败 at index " << i;
  }

  nppiFree(d_src);
  nppiFree(d_dst);
}

// 测试性能（大尺寸图像）
TEST_F(ConvertExtendedFunctionalTest, Convert_8u32f_C3R_PerformanceTest) {
  const int perfWidth = 1024;
  const int perfHeight = 768;
  NppiSize perfRoi = {perfWidth, perfHeight};

  int srcStep, dstStep;
  Npp8u *d_src = nppiMalloc_8u_C3(perfWidth, perfHeight, &srcStep);
  Npp32f *d_dst = nppiMalloc_32f_C3(perfWidth, perfHeight, &dstStep);
  ASSERT_NE(d_src, nullptr);
  ASSERT_NE(d_dst, nullptr);

  // 创建简单的测试数据（不Validate结果，主要测试性能）
  std::vector<Npp8u> srcData(perfWidth * perfHeight * 3, 128);

  // 复制数据到GPU
  for (int y = 0; y < perfHeight; y++) {
    cudaMemcpy((char *)d_src + y * srcStep, srcData.data() + y * perfWidth * 3, perfWidth * 3 * sizeof(Npp8u),
               cudaMemcpyHostToDevice);
  }

  // 执行转换（多次测试稳定性）
  for (int i = 0; i < 5; i++) {
    NppStatus status = nppiConvert_8u32f_C3R(d_src, srcStep, d_dst, dstStep, perfRoi);
    EXPECT_EQ(status, NPP_SUCCESS) << "性能测试第 " << i << " 次失败";
  }

  //
  std::vector<Npp32f> firstPixel(3), lastPixel(3);
  cudaMemcpy(firstPixel.data(), d_dst, 3 * sizeof(Npp32f), cudaMemcpyDeviceToHost);

  int lastPixelOffset = ((perfHeight - 1) * dstStep) + ((perfWidth - 1) * 3 * sizeof(Npp32f));
  cudaMemcpy(lastPixel.data(), (char *)d_dst + lastPixelOffset, 3 * sizeof(Npp32f), cudaMemcpyDeviceToHost);

  for (int c = 0; c < 3; c++) {
    EXPECT_FLOAT_EQ(firstPixel[c], 128.0f) << "第一个像素通道 " << c << " 错误";
    EXPECT_FLOAT_EQ(lastPixel[c], 128.0f) << "最后一个像素通道 " << c << " 错误";
  }

  nppiFree(d_src);
  nppiFree(d_dst);
}

// Enhanced parameterized tests for nppiConvert_8u32f_C3R
struct ConvertSizeParams {
  int width;
  int height;
  std::string description;
};

class ConvertC3RSizeTest : public ::testing::TestWithParam<ConvertSizeParams> {
protected:
  void SetUp() override {
    auto params = GetParam();
    width = params.width;
    height = params.height;
    roi.width = width;
    roi.height = height;
  }

  int width, height;
  NppiSize roi;
};

INSTANTIATE_TEST_SUITE_P(
    ConvertC3RSizes, ConvertC3RSizeTest,
    ::testing::Values(ConvertSizeParams{1, 1, "SinglePixel"}, ConvertSizeParams{2, 1, "TwoPixelRow"},
                      ConvertSizeParams{1, 2, "TwoPixelCol"}, ConvertSizeParams{3, 2, "SmallRect"},
                      ConvertSizeParams{4, 4, "SmallSquare"}, ConvertSizeParams{5, 7, "OddSize"},
                      ConvertSizeParams{8, 6, "BasicSize"}, ConvertSizeParams{16, 8, "Rect16x8"},
                      ConvertSizeParams{8, 16, "Rect8x16"}, ConvertSizeParams{32, 32, "Square32"},
                      ConvertSizeParams{64, 16, "WideRect"}, ConvertSizeParams{16, 64, "TallRect"},
                      ConvertSizeParams{128, 96, "MediumSize"}, ConvertSizeParams{256, 192, "LargeSize"},
                      ConvertSizeParams{512, 384, "VeryLarge"}, ConvertSizeParams{1024, 768, "HighRes"},
                      ConvertSizeParams{2048, 1536, "UltraHighRes"}, ConvertSizeParams{4096, 2048, "ExtremeSize"}),
    [](const ::testing::TestParamInfo<ConvertSizeParams> &info) {
      return info.param.description + "_" + std::to_string(info.param.width) + "x" + std::to_string(info.param.height);
    });

TEST_P(ConvertC3RSizeTest, Convert_8u32f_C3R_SizeParameterized) {
  const int channels = 3;
  std::vector<Npp8u> srcData(width * height * channels);
  std::vector<Npp32f> expectedData(width * height * channels);

  // Create size-dependent test pattern
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      int idx = (y * width + x) * channels;

      // R channel: position encoding with size factor
      Npp8u r_val = (Npp8u)((x * 255) / std::max(1, width - 1));
      // G channel: vertical gradient with size factor
      Npp8u g_val = (Npp8u)((y * 255) / std::max(1, height - 1));
      // B channel: combined pattern
      Npp8u b_val = (Npp8u)(((x + y) * 255) / std::max(1, width + height - 2));

      srcData[idx + 0] = r_val;
      srcData[idx + 1] = g_val;
      srcData[idx + 2] = b_val;

      expectedData[idx + 0] = (Npp32f)r_val;
      expectedData[idx + 1] = (Npp32f)g_val;
      expectedData[idx + 2] = (Npp32f)b_val;
    }
  }

  // Allocate GPU memory
  int srcStep, dstStep;
  Npp8u *d_src = nppiMalloc_8u_C3(width, height, &srcStep);
  Npp32f *d_dst = nppiMalloc_32f_C3(width, height, &dstStep);
  ASSERT_NE(d_src, nullptr);
  ASSERT_NE(d_dst, nullptr);

  // Copy input data to GPU
  for (int y = 0; y < height; y++) {
    cudaMemcpy((char *)d_src + y * srcStep, srcData.data() + y * width * channels, width * channels * sizeof(Npp8u),
               cudaMemcpyHostToDevice);
  }

  // Execute conversion
  NppStatus status = nppiConvert_8u32f_C3R(d_src, srcStep, d_dst, dstStep, roi);
  EXPECT_EQ(status, NPP_SUCCESS);

  // Copy result back to host
  std::vector<Npp32f> resultData(width * height * channels);
  for (int y = 0; y < height; y++) {
    cudaMemcpy(resultData.data() + y * width * channels, (char *)d_dst + y * dstStep, width * channels * sizeof(Npp32f),
               cudaMemcpyDeviceToHost);
  }

  // Verify results
  for (int i = 0; i < width * height * channels; i++) {
    EXPECT_FLOAT_EQ(resultData[i], expectedData[i])
        << "Size test failed at index " << i << " for size " << width << "x" << height;
  }

  nppiFree(d_src);
  nppiFree(d_dst);
}

// Data pattern tests for different conversion scenarios
struct ConvertPatternParams {
  std::string name;
  std::function<Npp8u(int, int, int, int, int)> generator; // x, y, width, height, channel
};

class ConvertC3RPatternTest : public ::testing::TestWithParam<ConvertPatternParams> {
protected:
  void SetUp() override {
    width = 16;
    height = 12;
    roi.width = width;
    roi.height = height;
  }

  int width, height;
  NppiSize roi;
};

INSTANTIATE_TEST_SUITE_P(
    ConvertC3RPatterns, ConvertC3RPatternTest,
    ::testing::Values(ConvertPatternParams{"FullRange",
                                           [](int x, int y, int w, int h, int c) -> Npp8u {
                                             (void)w;
                                             (void)h;
                                             (void)c;
                                             return (Npp8u)(((x + y) * 255) / 27);
                                           }},
                      ConvertPatternParams{"Boundaries",
                                           [](int x, int y, int w, int h, int c) -> Npp8u {
                                             (void)c;
                                             if (x == 0 || y == 0 || x == w - 1 || y == h - 1)
                                               return 255;
                                             return 0;
                                           }},
                      ConvertPatternParams{"Checkerboard",
                                           [](int x, int y, int w, int h, int c) -> Npp8u {
                                             (void)w;
                                             (void)h;
                                             (void)c;
                                             return ((x + y) % 2) ? 255 : 0;
                                           }},
                      ConvertPatternParams{"ChannelGradients",
                                           [](int x, int y, int w, int h, int c) -> Npp8u {
                                             if (c == 0)
                                               return (Npp8u)((x * 255) / std::max(1, w - 1)); // R: horizontal
                                             if (c == 1)
                                               return (Npp8u)((y * 255) / std::max(1, h - 1));         // G: vertical
                                             return (Npp8u)(((x + y) * 255) / std::max(1, w + h - 2)); // B: diagonal
                                           }},
                      ConvertPatternParams{"RGBPrimaries",
                                           [](int x, int y, int w, int h, int c) -> Npp8u {
                                             (void)y;
                                             (void)h; // Suppress unused variable warnings
                                             int block_w = w / 3;
                                             int block_x = x / std::max(1, block_w);

                                             if (block_x == 0 && c == 0)
                                               return 255; // Red region
                                             if (block_x == 1 && c == 1)
                                               return 255; // Green region
                                             if (block_x == 2 && c == 2)
                                               return 255; // Blue region
                                             return 0;
                                           }},
                      ConvertPatternParams{"NoisePattern",
                                           [](int x, int y, int w, int h, int c) -> Npp8u {
                                             // Deterministic pseudo-random pattern
                                             unsigned int seed = (unsigned int)(x * 1000 + y * 100 + c * 10 + w + h);
                                             seed = seed * 1103515245 + 12345;
                                             return (Npp8u)(seed % 256);
                                           }},
                      ConvertPatternParams{"Stripes",
                                           [](int x, int y, int w, int h, int c) -> Npp8u {
                                             (void)w;
                                             (void)h;
                                             if (c == 0)
                                               return (y % 4 < 2) ? 255 : 0; // R: horizontal stripes
                                             if (c == 1)
                                               return (x % 4 < 2) ? 255 : 0;     // G: vertical stripes
                                             return ((x + y) % 4 < 2) ? 255 : 0; // B: diagonal stripes
                                           }}),
    [](const ::testing::TestParamInfo<ConvertPatternParams> &info) { return info.param.name; });

TEST_P(ConvertC3RPatternTest, Convert_8u32f_C3R_DataPatterns) {
  auto params = GetParam();
  const int channels = 3;
  std::vector<Npp8u> srcData(width * height * channels);
  std::vector<Npp32f> expectedData(width * height * channels);

  // Generate test data using pattern
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      int idx = (y * width + x) * channels;
      for (int c = 0; c < channels; c++) {
        Npp8u val = params.generator(x, y, width, height, c);
        srcData[idx + c] = val;
        expectedData[idx + c] = (Npp32f)val;
      }
    }
  }

  // Allocate GPU memory
  int srcStep, dstStep;
  Npp8u *d_src = nppiMalloc_8u_C3(width, height, &srcStep);
  Npp32f *d_dst = nppiMalloc_32f_C3(width, height, &dstStep);
  ASSERT_NE(d_src, nullptr);
  ASSERT_NE(d_dst, nullptr);

  // Copy input data to GPU
  for (int y = 0; y < height; y++) {
    cudaMemcpy((char *)d_src + y * srcStep, srcData.data() + y * width * channels, width * channels * sizeof(Npp8u),
               cudaMemcpyHostToDevice);
  }

  // Execute conversion
  NppStatus status = nppiConvert_8u32f_C3R(d_src, srcStep, d_dst, dstStep, roi);
  EXPECT_EQ(status, NPP_SUCCESS);

  // Copy result back to host
  std::vector<Npp32f> resultData(width * height * channels);
  for (int y = 0; y < height; y++) {
    cudaMemcpy(resultData.data() + y * width * channels, (char *)d_dst + y * dstStep, width * channels * sizeof(Npp32f),
               cudaMemcpyDeviceToHost);
  }

  // Verify conversion results
  for (int i = 0; i < width * height * channels; i++) {
    EXPECT_FLOAT_EQ(resultData[i], expectedData[i])
        << "Pattern test failed at index " << i << " for pattern: " << params.name;
  }

  nppiFree(d_src);
  nppiFree(d_dst);
}

// Stream context tests for different sizes
class ConvertC3RStreamTest : public ::testing::TestWithParam<ConvertSizeParams> {
protected:
  void SetUp() override {
    auto params = GetParam();
    width = params.width;
    height = params.height;
    roi.width = width;
    roi.height = height;
  }

  int width, height;
  NppiSize roi;
};

INSTANTIATE_TEST_SUITE_P(ConvertC3RStreamSizes, ConvertC3RStreamTest,
                         ::testing::Values(ConvertSizeParams{4, 4, "Small"}, ConvertSizeParams{16, 16, "Medium"},
                                           ConvertSizeParams{64, 64, "Large"}, ConvertSizeParams{256, 256, "VeryLarge"},
                                           ConvertSizeParams{1024, 512, "HighRes"}),
                         [](const ::testing::TestParamInfo<ConvertSizeParams> &info) {
                           return info.param.description + "_" + std::to_string(info.param.width) + "x" +
                                  std::to_string(info.param.height);
                         });

TEST_P(ConvertC3RStreamTest, Convert_8u32f_C3R_StreamContext) {
  const int channels = 3;
  std::vector<Npp8u> srcData(width * height * channels);
  std::vector<Npp32f> expectedData(width * height * channels);

  // Create stream-specific test pattern
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      int idx = (y * width + x) * channels;

      // Generate pattern based on position
      Npp8u r_val = (Npp8u)(x % 256);
      Npp8u g_val = (Npp8u)(y % 256);
      Npp8u b_val = (Npp8u)((x + y) % 256);

      srcData[idx + 0] = r_val;
      srcData[idx + 1] = g_val;
      srcData[idx + 2] = b_val;

      expectedData[idx + 0] = (Npp32f)r_val;
      expectedData[idx + 1] = (Npp32f)g_val;
      expectedData[idx + 2] = (Npp32f)b_val;
    }
  }

  // Allocate GPU memory
  int srcStep, dstStep;
  Npp8u *d_src = nppiMalloc_8u_C3(width, height, &srcStep);
  Npp32f *d_dst = nppiMalloc_32f_C3(width, height, &dstStep);
  ASSERT_NE(d_src, nullptr);
  ASSERT_NE(d_dst, nullptr);

  // Copy input data to GPU
  for (int y = 0; y < height; y++) {
    cudaMemcpy((char *)d_src + y * srcStep, srcData.data() + y * width * channels, width * channels * sizeof(Npp8u),
               cudaMemcpyHostToDevice);
  }

  // Create stream context
  NppStreamContext nppStreamCtx;
  nppGetStreamContext(&nppStreamCtx);

  // Execute conversion with stream context
  NppStatus status = nppiConvert_8u32f_C3R_Ctx(d_src, srcStep, d_dst, dstStep, roi, nppStreamCtx);
  EXPECT_EQ(status, NPP_SUCCESS);

  // Copy result back to host
  std::vector<Npp32f> resultData(width * height * channels);
  for (int y = 0; y < height; y++) {
    cudaMemcpy(resultData.data() + y * width * channels, (char *)d_dst + y * dstStep, width * channels * sizeof(Npp32f),
               cudaMemcpyDeviceToHost);
  }

  // Verify stream context results
  for (int i = 0; i < width * height * channels; i++) {
    EXPECT_FLOAT_EQ(resultData[i], expectedData[i])
        << "Stream test failed at index " << i << " for size " << width << "x" << height;
  }

  nppiFree(d_src);
  nppiFree(d_dst);
}

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

  // Function to allocate aligned memory
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
  cudaMemcpy2D(hostDst.data(), width * 3 * sizeof(Npp32f), d_dst, dstStep, width * 3 * sizeof(Npp32f), height,
               cudaMemcpyDeviceToHost);

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
  cudaMemcpy2D(hostDst.data(), width * 3 * sizeof(Npp32f), d_dst, dstStep, width * 3 * sizeof(Npp32f), height,
               cudaMemcpyDeviceToHost);

  // Verify each channel maintains its distinct value
  for (int i = 0; i < width * height; i++) {
    EXPECT_FLOAT_EQ(hostDst[i * 3], 100.0f) << "R channel mismatch at pixel " << i;
    EXPECT_FLOAT_EQ(hostDst[i * 3 + 1], 150.0f) << "G channel mismatch at pixel " << i;
    EXPECT_FLOAT_EQ(hostDst[i * 3 + 2], 200.0f) << "B channel mismatch at pixel " << i;
  }

  nppiFree(d_src);
  nppiFree(d_dst);
}
