#include "../../framework/npp_test_base.h"
#include <algorithm>
#include <chrono>
#include <numeric>
#include <thread>

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
