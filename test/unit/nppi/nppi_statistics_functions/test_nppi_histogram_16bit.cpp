#include <algorithm>
#include <cstdlib>
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <npp.h>
#include <vector>

class HistogramEven16BitTest : public ::testing::Test {
protected:
  void SetUp() override {
    // Common test parameters
    width = 64;
    height = 64;
    totalPixels = width * height;
    nLevels = 256;
    nLowerLevel = 0;
    nUpperLevel_16u = 65535; // Max 16-bit unsigned value
    nUpperLevel_16s = 32767; // Max 16-bit signed value

    oSizeROI = {width, height};
  }

  void TearDown() override {
    // Cleanup device memory if allocated
    cleanupDeviceMemory();
  }

  void cleanupDeviceMemory() {
    if (d_src_16u) {
      cudaFree(d_src_16u);
      d_src_16u = nullptr;
    }
    if (d_src_16s) {
      cudaFree(d_src_16s);
      d_src_16s = nullptr;
    }
    if (d_hist) {
      cudaFree(d_hist);
      d_hist = nullptr;
    }
    if (d_buffer) {
      cudaFree(d_buffer);
      d_buffer = nullptr;
    }
  }

  void allocateTestData16u() {
    // Create test data for 16u
    h_src_16u.resize(totalPixels);
    for (int i = 0; i < totalPixels; i++) {
      h_src_16u[i] = static_cast<Npp16u>(i % 65536); // Create varied data
    }

    // Allocate device memory
    nSrcStep_16u = width * sizeof(Npp16u);
    ASSERT_EQ(cudaMalloc(&d_src_16u, totalPixels * sizeof(Npp16u)), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(d_src_16u, h_src_16u.data(), totalPixels * sizeof(Npp16u), cudaMemcpyHostToDevice),
              cudaSuccess);
  }

  void allocateTestData16s() {
    // Create test data for 16s
    h_src_16s.resize(totalPixels);
    for (int i = 0; i < totalPixels; i++) {
      h_src_16s[i] = static_cast<Npp16s>((i % 65536) - 32768); // Range [-32768, 32767]
    }

    // Allocate device memory
    nSrcStep_16s = width * sizeof(Npp16s);
    ASSERT_EQ(cudaMalloc(&d_src_16s, totalPixels * sizeof(Npp16s)), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(d_src_16s, h_src_16s.data(), totalPixels * sizeof(Npp16s), cudaMemcpyHostToDevice),
              cudaSuccess);
  }

  void allocateHistogramMemory() {
    h_hist.resize(nLevels - 1, 0);
    ASSERT_EQ(cudaMalloc(&d_hist, (nLevels - 1) * sizeof(Npp32s)), cudaSuccess);

    // Get buffer size and allocate
#ifdef USE_NVIDIA_NPP_TESTS
    size_t bufferSize;
#else
    int bufferSize;
#endif
    ASSERT_EQ(nppiHistogramEvenGetBufferSize_16u_C1R(oSizeROI, nLevels, &bufferSize), NPP_SUCCESS);
    ASSERT_EQ(cudaMalloc(&d_buffer, bufferSize), cudaSuccess);
  }

protected:
  int width, height, totalPixels;
  int nLevels, nLowerLevel, nUpperLevel_16u, nUpperLevel_16s;
  int nSrcStep_16u, nSrcStep_16s;
  NppiSize oSizeROI;

  std::vector<Npp16u> h_src_16u;
  std::vector<Npp16s> h_src_16s;
  std::vector<Npp32s> h_hist;

  Npp16u *d_src_16u = nullptr;
  Npp16s *d_src_16s = nullptr;
  Npp32s *d_hist = nullptr;
  Npp8u *d_buffer = nullptr;
};

TEST_F(HistogramEven16BitTest, BufferSize_16u_C1R_Basic) {
#ifdef USE_NVIDIA_NPP_TESTS
  size_t bufferSize;
#else
  int bufferSize;
#endif

  // Test basic buffer size calculation
  EXPECT_EQ(nppiHistogramEvenGetBufferSize_16u_C1R(oSizeROI, nLevels, &bufferSize), NPP_SUCCESS);
  EXPECT_GT(bufferSize, 0);

  // Test with stream context
  EXPECT_EQ(nppiHistogramEvenGetBufferSize_16u_C1R_Ctx(oSizeROI, nLevels, &bufferSize, NppStreamContext{}),
            NPP_SUCCESS);
  EXPECT_GT(bufferSize, 0);
}

TEST_F(HistogramEven16BitTest, BufferSize_16s_C1R_Basic) {
#ifdef USE_NVIDIA_NPP_TESTS
  size_t bufferSize;
#else
  int bufferSize;
#endif

  EXPECT_EQ(nppiHistogramEvenGetBufferSize_16s_C1R(oSizeROI, nLevels, &bufferSize), NPP_SUCCESS);
  EXPECT_GT(bufferSize, 0);

  EXPECT_EQ(nppiHistogramEvenGetBufferSize_16s_C1R_Ctx(oSizeROI, nLevels, &bufferSize, NppStreamContext{}),
            NPP_SUCCESS);
  EXPECT_GT(bufferSize, 0);
}

TEST_F(HistogramEven16BitTest, HistogramEven_16u_C1R_Basic) {
  allocateTestData16u();
  allocateHistogramMemory();

  // Compute histogram
  EXPECT_EQ(nppiHistogramEven_16u_C1R(d_src_16u, nSrcStep_16u, oSizeROI, d_hist, nLevels, nLowerLevel, nUpperLevel_16u,
                                      d_buffer),
            NPP_SUCCESS);

  // Copy result back to host
  ASSERT_EQ(cudaMemcpy(h_hist.data(), d_hist, (nLevels - 1) * sizeof(Npp32s), cudaMemcpyDeviceToHost), cudaSuccess);

  // Verify total count
  int totalCount = 0;
  for (int i = 0; i < nLevels - 1; i++) {
    totalCount += h_hist[i];
  }
  EXPECT_EQ(totalCount, totalPixels);

  // Verify histogram has reasonable distribution
  int nonZeroBins = 0;
  for (int i = 0; i < nLevels - 1; i++) {
    if (h_hist[i] > 0)
      nonZeroBins++;
  }
  EXPECT_GT(nonZeroBins, 0);
}

TEST_F(HistogramEven16BitTest, HistogramEven_16s_C1R_Basic) {
  allocateTestData16s();
  allocateHistogramMemory();

  // For 16s, test with range [-1000, 1000] for better distribution
  int testLower = -1000;
  int testUpper = 1000;

  EXPECT_EQ(
      nppiHistogramEven_16s_C1R(d_src_16s, nSrcStep_16s, oSizeROI, d_hist, nLevels, testLower, testUpper, d_buffer),
      NPP_SUCCESS);

  ASSERT_EQ(cudaMemcpy(h_hist.data(), d_hist, (nLevels - 1) * sizeof(Npp32s), cudaMemcpyDeviceToHost), cudaSuccess);

  // Count pixels in range
  int expectedCount = 0;
  for (int i = 0; i < totalPixels; i++) {
    if (h_src_16s[i] >= testLower && h_src_16s[i] < testUpper) {
      expectedCount++;
    }
  }

  int actualCount = 0;
  for (int i = 0; i < nLevels - 1; i++) {
    actualCount += h_hist[i];
  }
  EXPECT_EQ(actualCount, expectedCount);
}

TEST_F(HistogramEven16BitTest, HistogramEven_16u_C1R_WithContext) {
  allocateTestData16u();
  allocateHistogramMemory();

  NppStreamContext nppStreamCtx;
  nppGetStreamContext(&nppStreamCtx);

  EXPECT_EQ(nppiHistogramEven_16u_C1R_Ctx(d_src_16u, nSrcStep_16u, oSizeROI, d_hist, nLevels, nLowerLevel,
                                          nUpperLevel_16u, d_buffer, nppStreamCtx),
            NPP_SUCCESS);

  cudaStreamSynchronize(nppStreamCtx.hStream);

  ASSERT_EQ(cudaMemcpy(h_hist.data(), d_hist, (nLevels - 1) * sizeof(Npp32s), cudaMemcpyDeviceToHost), cudaSuccess);

  int totalCount = 0;
  for (int i = 0; i < nLevels - 1; i++) {
    totalCount += h_hist[i];
  }
  EXPECT_EQ(totalCount, totalPixels);
}

TEST_F(HistogramEven16BitTest, ParameterValidation_16u) {
  allocateTestData16u();
  allocateHistogramMemory();

  // Test null pointer errors
  EXPECT_EQ(nppiHistogramEven_16u_C1R(nullptr, nSrcStep_16u, oSizeROI, d_hist, nLevels, nLowerLevel, nUpperLevel_16u,
                                      d_buffer),
            NPP_NULL_POINTER_ERROR);

  EXPECT_EQ(nppiHistogramEven_16u_C1R(d_src_16u, nSrcStep_16u, oSizeROI, nullptr, nLevels, nLowerLevel, nUpperLevel_16u,
                                      d_buffer),
            NPP_NULL_POINTER_ERROR);

  // Test size error
  NppiSize invalidSize = {0, height};
  EXPECT_EQ(nppiHistogramEven_16u_C1R(d_src_16u, nSrcStep_16u, invalidSize, d_hist, nLevels, nLowerLevel,
                                      nUpperLevel_16u, d_buffer),
            NPP_SIZE_ERROR);

  // Test range error
  EXPECT_EQ(nppiHistogramEven_16u_C1R(d_src_16u, nSrcStep_16u, oSizeROI, d_hist, nLevels, nUpperLevel_16u, nLowerLevel,
                                      d_buffer),
            NPP_RANGE_ERROR);
}

TEST_F(HistogramEven16BitTest, ParameterValidation_16s) {
  allocateTestData16s();
  allocateHistogramMemory();

  // Test null pointer errors
  EXPECT_EQ(nppiHistogramEven_16s_C1R(nullptr, nSrcStep_16s, oSizeROI, d_hist, nLevels, nLowerLevel, nUpperLevel_16s,
                                      d_buffer),
            NPP_NULL_POINTER_ERROR);

  // Test step error - too small step
  int invalidStep = width * sizeof(Npp16s) - 1;
  EXPECT_EQ(nppiHistogramEven_16s_C1R(d_src_16s, invalidStep, oSizeROI, d_hist, nLevels, nLowerLevel, nUpperLevel_16s,
                                      d_buffer),
            NPP_STEP_ERROR);
}

TEST_F(HistogramEven16BitTest, SmallLevelsOptimization) {
  allocateTestData16u();

  // Test with small number of levels to trigger shared memory optimization
  int smallLevels = 16;
#ifdef USE_NVIDIA_NPP_TESTS
  size_t bufferSize;
#else
  int bufferSize;
#endif
  ASSERT_EQ(nppiHistogramEvenGetBufferSize_16u_C1R(oSizeROI, smallLevels, &bufferSize), NPP_SUCCESS);
  ASSERT_EQ(cudaMalloc(&d_buffer, bufferSize), cudaSuccess);

  std::vector<Npp32s> h_hist_small(smallLevels - 1, 0);
  ASSERT_EQ(cudaMalloc(&d_hist, (smallLevels - 1) * sizeof(Npp32s)), cudaSuccess);

  EXPECT_EQ(nppiHistogramEven_16u_C1R(d_src_16u, nSrcStep_16u, oSizeROI, d_hist, smallLevels, 0, 65535, d_buffer),
            NPP_SUCCESS);

  ASSERT_EQ(cudaMemcpy(h_hist_small.data(), d_hist, (smallLevels - 1) * sizeof(Npp32s), cudaMemcpyDeviceToHost),
            cudaSuccess);

  int totalCount = 0;
  for (int i = 0; i < smallLevels - 1; i++) {
    totalCount += h_hist_small[i];
  }
  EXPECT_EQ(totalCount, totalPixels);
}

TEST_F(HistogramEven16BitTest, UniformDistribution) {
  // Test with uniform data distribution
  h_src_16u.resize(totalPixels);
  for (int i = 0; i < totalPixels; i++) {
    h_src_16u[i] = static_cast<Npp16u>((i * 256) % 65536); // Spread values evenly
  }

  nSrcStep_16u = width * sizeof(Npp16u);
  ASSERT_EQ(cudaMalloc(&d_src_16u, totalPixels * sizeof(Npp16u)), cudaSuccess);
  ASSERT_EQ(cudaMemcpy(d_src_16u, h_src_16u.data(), totalPixels * sizeof(Npp16u), cudaMemcpyHostToDevice), cudaSuccess);

  allocateHistogramMemory();

  // Use smaller range for better distribution visibility
  int testUpper = 1024;
  EXPECT_EQ(
      nppiHistogramEven_16u_C1R(d_src_16u, nSrcStep_16u, oSizeROI, d_hist, nLevels, nLowerLevel, testUpper, d_buffer),
      NPP_SUCCESS);

  ASSERT_EQ(cudaMemcpy(h_hist.data(), d_hist, (nLevels - 1) * sizeof(Npp32s), cudaMemcpyDeviceToHost), cudaSuccess);

  // Count expected pixels in range
  int expectedInRange = 0;
  for (int i = 0; i < totalPixels; i++) {
    if (h_src_16u[i] >= nLowerLevel && h_src_16u[i] < testUpper) {
      expectedInRange++;
    }
  }

  int actualCount = 0;
  for (int i = 0; i < nLevels - 1; i++) {
    actualCount += h_hist[i];
  }
  EXPECT_EQ(actualCount, expectedInRange);
}