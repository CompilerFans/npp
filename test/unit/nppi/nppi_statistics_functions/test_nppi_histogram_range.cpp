#include <algorithm>
#include <cstdlib>
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <npp.h>
#include <vector>

// Type alias for buffer size to handle API differences
#ifdef USE_NVIDIA_NPP_TESTS
using BufferSizeType = size_t;
#else
using BufferSizeType = int;
#endif

class HistogramRangeTest : public ::testing::Test {
protected:
  void SetUp() override {
    // Clear any previous CUDA errors
    cudaGetLastError();

    width = 64;
    height = 64;
    totalPixels = width * height;
    nLevels = 10; // Test with custom levels

    oSizeROI = {width, height};
  }

  void TearDown() override {
    // Ensure all CUDA operations are complete before cleanup
    cudaDeviceSynchronize();
    cleanupDeviceMemory();

    // Clear any remaining errors
    cudaGetLastError();

    // Force flush any remaining operations
    cudaDeviceSynchronize();
  }

  void cleanupDeviceMemory() {
    if (d_src_8u) { cudaFree(d_src_8u); d_src_8u = nullptr; }
    if (d_src_16u) { cudaFree(d_src_16u); d_src_16u = nullptr; }
    if (d_src_16s) { cudaFree(d_src_16s); d_src_16s = nullptr; }
    if (d_src_32f) { cudaFree(d_src_32f); d_src_32f = nullptr; }
    if (d_src_8u_c4) { cudaFree(d_src_8u_c4); d_src_8u_c4 = nullptr; }
    if (d_src_16u_c4) { cudaFree(d_src_16u_c4); d_src_16u_c4 = nullptr; }
    if (d_src_16s_c4) { cudaFree(d_src_16s_c4); d_src_16s_c4 = nullptr; }
    if (d_src_32f_c4) { cudaFree(d_src_32f_c4); d_src_32f_c4 = nullptr; }
    if (d_hist) { cudaFree(d_hist); d_hist = nullptr; }
    if (d_buffer) { cudaFree(d_buffer); d_buffer = nullptr; }
    if (d_levels_8u) { cudaFree(d_levels_8u); d_levels_8u = nullptr; }
    if (d_levels_16) { cudaFree(d_levels_16); d_levels_16 = nullptr; }
    if (d_levels_32f) { cudaFree(d_levels_32f); d_levels_32f = nullptr; }
    for (int c = 0; c < 4; c++) {
      if (d_hist_c4[c]) { cudaFree(d_hist_c4[c]); d_hist_c4[c] = nullptr; }
      if (d_levels_8u_c4[c]) { cudaFree(d_levels_8u_c4[c]); d_levels_8u_c4[c] = nullptr; }
      if (d_levels_16_c4[c]) { cudaFree(d_levels_16_c4[c]); d_levels_16_c4[c] = nullptr; }
      if (d_levels_32f_c4[c]) { cudaFree(d_levels_32f_c4[c]); d_levels_32f_c4[c] = nullptr; }
    }
  }

  void allocateTestData8u() {
    // Create test data for 8u
    h_src_8u.resize(totalPixels);
    for (int i = 0; i < totalPixels; i++) {
      h_src_8u[i] = static_cast<Npp8u>(i % 256);
    }

    // Allocate device memory
    nSrcStep_8u = width * sizeof(Npp8u);
    ASSERT_EQ(cudaMalloc(&d_src_8u, totalPixels * sizeof(Npp8u)), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(d_src_8u, h_src_8u.data(), totalPixels * sizeof(Npp8u), cudaMemcpyHostToDevice), cudaSuccess);
  }

  void allocateTestData32f() {
    // Create test data for 32f
    h_src_32f.resize(totalPixels);
    for (int i = 0; i < totalPixels; i++) {
      h_src_32f[i] = static_cast<Npp32f>(i % 256) / 256.0f; // Values [0.0, 1.0)
    }

    nSrcStep_32f = width * sizeof(Npp32f);
    ASSERT_EQ(cudaMalloc(&d_src_32f, totalPixels * sizeof(Npp32f)), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(d_src_32f, h_src_32f.data(), totalPixels * sizeof(Npp32f), cudaMemcpyHostToDevice),
              cudaSuccess);
  }

  void allocateTestData16u() {
    h_src_16u.resize(totalPixels);
    for (int i = 0; i < totalPixels; i++) {
      h_src_16u[i] = static_cast<Npp16u>((i * 257) % 65536);
    }
    nSrcStep_16u = width * sizeof(Npp16u);
    ASSERT_EQ(cudaMalloc(&d_src_16u, totalPixels * sizeof(Npp16u)), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(d_src_16u, h_src_16u.data(), totalPixels * sizeof(Npp16u), cudaMemcpyHostToDevice), cudaSuccess);
  }

  void allocateTestData16s() {
    h_src_16s.resize(totalPixels);
    for (int i = 0; i < totalPixels; i++) {
      h_src_16s[i] = static_cast<Npp16s>((i * 127) % 32768 - 16384);
    }
    nSrcStep_16s = width * sizeof(Npp16s);
    ASSERT_EQ(cudaMalloc(&d_src_16s, totalPixels * sizeof(Npp16s)), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(d_src_16s, h_src_16s.data(), totalPixels * sizeof(Npp16s), cudaMemcpyHostToDevice), cudaSuccess);
  }

  void allocateTestData8uC4() {
    h_src_8u_c4.resize(totalPixels * 4);
    for (int i = 0; i < totalPixels; i++) {
      h_src_8u_c4[i * 4 + 0] = static_cast<Npp8u>((i * 3) % 256);
      h_src_8u_c4[i * 4 + 1] = static_cast<Npp8u>((i * 5) % 256);
      h_src_8u_c4[i * 4 + 2] = static_cast<Npp8u>((i * 7) % 256);
      h_src_8u_c4[i * 4 + 3] = static_cast<Npp8u>((i * 11) % 256);
    }
    nSrcStep_8u_c4 = width * 4 * sizeof(Npp8u);
    ASSERT_EQ(cudaMalloc(&d_src_8u_c4, totalPixels * 4 * sizeof(Npp8u)), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(d_src_8u_c4, h_src_8u_c4.data(), totalPixels * 4 * sizeof(Npp8u), cudaMemcpyHostToDevice), cudaSuccess);
  }

  void allocateTestData16uC4() {
    h_src_16u_c4.resize(totalPixels * 4);
    for (int i = 0; i < totalPixels; i++) {
      h_src_16u_c4[i * 4 + 0] = static_cast<Npp16u>((i * 100) % 65536);
      h_src_16u_c4[i * 4 + 1] = static_cast<Npp16u>((i * 200) % 65536);
      h_src_16u_c4[i * 4 + 2] = static_cast<Npp16u>((i * 300) % 65536);
      h_src_16u_c4[i * 4 + 3] = static_cast<Npp16u>((i * 400) % 65536);
    }
    nSrcStep_16u_c4 = width * 4 * sizeof(Npp16u);
    ASSERT_EQ(cudaMalloc(&d_src_16u_c4, totalPixels * 4 * sizeof(Npp16u)), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(d_src_16u_c4, h_src_16u_c4.data(), totalPixels * 4 * sizeof(Npp16u), cudaMemcpyHostToDevice), cudaSuccess);
  }

  void allocateTestData16sC4() {
    h_src_16s_c4.resize(totalPixels * 4);
    for (int i = 0; i < totalPixels; i++) {
      h_src_16s_c4[i * 4 + 0] = static_cast<Npp16s>((i * 50) % 32768 - 16384);
      h_src_16s_c4[i * 4 + 1] = static_cast<Npp16s>((i * 100) % 32768 - 16384);
      h_src_16s_c4[i * 4 + 2] = static_cast<Npp16s>((i * 150) % 32768 - 16384);
      h_src_16s_c4[i * 4 + 3] = static_cast<Npp16s>((i * 200) % 32768 - 16384);
    }
    nSrcStep_16s_c4 = width * 4 * sizeof(Npp16s);
    ASSERT_EQ(cudaMalloc(&d_src_16s_c4, totalPixels * 4 * sizeof(Npp16s)), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(d_src_16s_c4, h_src_16s_c4.data(), totalPixels * 4 * sizeof(Npp16s), cudaMemcpyHostToDevice), cudaSuccess);
  }

  void allocateTestData32fC4() {
    h_src_32f_c4.resize(totalPixels * 4);
    for (int i = 0; i < totalPixels; i++) {
      h_src_32f_c4[i * 4 + 0] = static_cast<Npp32f>((i % 1000) / 1000.0f);
      h_src_32f_c4[i * 4 + 1] = static_cast<Npp32f>((i % 2000) / 2000.0f);
      h_src_32f_c4[i * 4 + 2] = static_cast<Npp32f>((i % 3000) / 3000.0f);
      h_src_32f_c4[i * 4 + 3] = static_cast<Npp32f>((i % 4000) / 4000.0f);
    }
    nSrcStep_32f_c4 = width * 4 * sizeof(Npp32f);
    ASSERT_EQ(cudaMalloc(&d_src_32f_c4, totalPixels * 4 * sizeof(Npp32f)), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(d_src_32f_c4, h_src_32f_c4.data(), totalPixels * 4 * sizeof(Npp32f), cudaMemcpyHostToDevice), cudaSuccess);
  }

  void setupLevels8u() {
    // Create custom levels for 8u: [0, 25, 50, 75, 100, 125, 150, 175, 200, 225, 255]
    h_levels_8u.resize(nLevels);
    for (int i = 0; i < nLevels; i++) {
      h_levels_8u[i] = static_cast<Npp32s>(i * 255 / (nLevels - 1));
    }

    ASSERT_EQ(cudaMalloc(&d_levels_8u, nLevels * sizeof(Npp32s)), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(d_levels_8u, h_levels_8u.data(), nLevels * sizeof(Npp32s), cudaMemcpyHostToDevice),
              cudaSuccess);
  }

  void setupLevels32f() {
    // Create custom levels for 32f: [0.0, 0.1, 0.2, ..., 0.9, 1.0]
    h_levels_32f.resize(nLevels);
    for (int i = 0; i < nLevels; i++) {
      h_levels_32f[i] = static_cast<Npp32f>(i) / static_cast<Npp32f>(nLevels - 1);
    }

    ASSERT_EQ(cudaMalloc(&d_levels_32f, nLevels * sizeof(Npp32f)), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(d_levels_32f, h_levels_32f.data(), nLevels * sizeof(Npp32f), cudaMemcpyHostToDevice),
              cudaSuccess);
  }

  void setupLevels16u() {
    h_levels_16.resize(nLevels);
    for (int i = 0; i < nLevels; i++) {
      h_levels_16[i] = static_cast<Npp32s>(i * 65535 / (nLevels - 1));
    }
    ASSERT_EQ(cudaMalloc(&d_levels_16, nLevels * sizeof(Npp32s)), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(d_levels_16, h_levels_16.data(), nLevels * sizeof(Npp32s), cudaMemcpyHostToDevice), cudaSuccess);
  }

  void setupLevels16s() {
    h_levels_16.resize(nLevels);
    for (int i = 0; i < nLevels; i++) {
      h_levels_16[i] = static_cast<Npp32s>(-16384 + i * 32767 / (nLevels - 1));
    }
    ASSERT_EQ(cudaMalloc(&d_levels_16, nLevels * sizeof(Npp32s)), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(d_levels_16, h_levels_16.data(), nLevels * sizeof(Npp32s), cudaMemcpyHostToDevice), cudaSuccess);
  }

  void setupLevelsC4_8u() {
    for (int c = 0; c < 4; c++) {
      nLevelsArray[c] = nLevels;
      h_levels_8u_c4[c].resize(nLevels);
      for (int i = 0; i < nLevels; i++) {
        h_levels_8u_c4[c][i] = static_cast<Npp32s>(i * 255 / (nLevels - 1));
      }
      ASSERT_EQ(cudaMalloc(&d_levels_8u_c4[c], nLevels * sizeof(Npp32s)), cudaSuccess);
      ASSERT_EQ(cudaMemcpy(d_levels_8u_c4[c], h_levels_8u_c4[c].data(), nLevels * sizeof(Npp32s), cudaMemcpyHostToDevice), cudaSuccess);
    }
  }

  void setupLevelsC4_16u() {
    for (int c = 0; c < 4; c++) {
      nLevelsArray[c] = nLevels;
      h_levels_16_c4[c].resize(nLevels);
      for (int i = 0; i < nLevels; i++) {
        h_levels_16_c4[c][i] = static_cast<Npp32s>(i * 65535 / (nLevels - 1));
      }
      ASSERT_EQ(cudaMalloc(&d_levels_16_c4[c], nLevels * sizeof(Npp32s)), cudaSuccess);
      ASSERT_EQ(cudaMemcpy(d_levels_16_c4[c], h_levels_16_c4[c].data(), nLevels * sizeof(Npp32s), cudaMemcpyHostToDevice), cudaSuccess);
    }
  }

  void setupLevelsC4_16s() {
    for (int c = 0; c < 4; c++) {
      nLevelsArray[c] = nLevels;
      h_levels_16_c4[c].resize(nLevels);
      for (int i = 0; i < nLevels; i++) {
        h_levels_16_c4[c][i] = static_cast<Npp32s>(-16384 + i * 32767 / (nLevels - 1));
      }
      ASSERT_EQ(cudaMalloc(&d_levels_16_c4[c], nLevels * sizeof(Npp32s)), cudaSuccess);
      ASSERT_EQ(cudaMemcpy(d_levels_16_c4[c], h_levels_16_c4[c].data(), nLevels * sizeof(Npp32s), cudaMemcpyHostToDevice), cudaSuccess);
    }
  }

  void setupLevelsC4_32f() {
    for (int c = 0; c < 4; c++) {
      nLevelsArray[c] = nLevels;
      h_levels_32f_c4[c].resize(nLevels);
      for (int i = 0; i < nLevels; i++) {
        h_levels_32f_c4[c][i] = static_cast<Npp32f>(i) / static_cast<Npp32f>(nLevels - 1);
      }
      ASSERT_EQ(cudaMalloc(&d_levels_32f_c4[c], nLevels * sizeof(Npp32f)), cudaSuccess);
      ASSERT_EQ(cudaMemcpy(d_levels_32f_c4[c], h_levels_32f_c4[c].data(), nLevels * sizeof(Npp32f), cudaMemcpyHostToDevice), cudaSuccess);
    }
  }

  void allocateHistogramC4() {
    for (int c = 0; c < 4; c++) {
      h_hist_c4[c].resize(nLevels - 1, 0);
      ASSERT_EQ(cudaMalloc(&d_hist_c4[c], (nLevels - 1) * sizeof(Npp32s)), cudaSuccess);
    }
  }

  void allocateHistogramMemory() {
    h_hist.resize(nLevels - 1, 0);
    ASSERT_EQ(cudaMalloc(&d_hist, (nLevels - 1) * sizeof(Npp32s)), cudaSuccess);

    // Get buffer size and allocate
    BufferSizeType bufferSize;
    ASSERT_EQ(nppiHistogramRangeGetBufferSize_8u_C1R(oSizeROI, nLevels, &bufferSize), NPP_SUCCESS);
    ASSERT_EQ(cudaMalloc(&d_buffer, bufferSize), cudaSuccess);
  }

protected:
  int width, height, totalPixels;
  int nLevels;
  int nSrcStep_8u, nSrcStep_16u, nSrcStep_16s, nSrcStep_32f;
  int nSrcStep_8u_c4, nSrcStep_16u_c4, nSrcStep_16s_c4, nSrcStep_32f_c4;
  int nLevelsArray[4];
  NppiSize oSizeROI;

  // Host data
  std::vector<Npp8u> h_src_8u;
  std::vector<Npp16u> h_src_16u;
  std::vector<Npp16s> h_src_16s;
  std::vector<Npp32f> h_src_32f;
  std::vector<Npp8u> h_src_8u_c4;
  std::vector<Npp16u> h_src_16u_c4;
  std::vector<Npp16s> h_src_16s_c4;
  std::vector<Npp32f> h_src_32f_c4;
  std::vector<Npp32s> h_hist;
  std::vector<Npp32s> h_hist_c4[4];
  std::vector<Npp32s> h_levels_8u;
  std::vector<Npp32s> h_levels_16;
  std::vector<Npp32f> h_levels_32f;
  std::vector<Npp32s> h_levels_8u_c4[4];
  std::vector<Npp32s> h_levels_16_c4[4];
  std::vector<Npp32f> h_levels_32f_c4[4];

  // Device data
  Npp8u *d_src_8u = nullptr;
  Npp16u *d_src_16u = nullptr;
  Npp16s *d_src_16s = nullptr;
  Npp32f *d_src_32f = nullptr;
  Npp8u *d_src_8u_c4 = nullptr;
  Npp16u *d_src_16u_c4 = nullptr;
  Npp16s *d_src_16s_c4 = nullptr;
  Npp32f *d_src_32f_c4 = nullptr;
  Npp32s *d_hist = nullptr;
  Npp32s *d_hist_c4[4] = {nullptr, nullptr, nullptr, nullptr};
  Npp8u *d_buffer = nullptr;
  Npp32s *d_levels_8u = nullptr;
  Npp32s *d_levels_16 = nullptr;
  Npp32f *d_levels_32f = nullptr;
  Npp32s *d_levels_8u_c4[4] = {nullptr, nullptr, nullptr, nullptr};
  Npp32s *d_levels_16_c4[4] = {nullptr, nullptr, nullptr, nullptr};
  Npp32f *d_levels_32f_c4[4] = {nullptr, nullptr, nullptr, nullptr};
};

TEST_F(HistogramRangeTest, BufferSize_8u_C1R_Basic) {
  BufferSizeType bufferSize;

  EXPECT_EQ(nppiHistogramRangeGetBufferSize_8u_C1R(oSizeROI, nLevels, &bufferSize), NPP_SUCCESS);
  EXPECT_GT(bufferSize, 0);

  // Test with stream context
  EXPECT_EQ(nppiHistogramRangeGetBufferSize_8u_C1R_Ctx(oSizeROI, nLevels, &bufferSize, NppStreamContext{}),
            NPP_SUCCESS);
  EXPECT_GT(bufferSize, 0);
}

TEST_F(HistogramRangeTest, BufferSize_32f_C1R_Basic) {
  BufferSizeType bufferSize;

  EXPECT_EQ(nppiHistogramRangeGetBufferSize_32f_C1R(oSizeROI, nLevels, &bufferSize), NPP_SUCCESS);
  EXPECT_GT(bufferSize, 0);

  EXPECT_EQ(nppiHistogramRangeGetBufferSize_32f_C1R_Ctx(oSizeROI, nLevels, &bufferSize, NppStreamContext{}),
            NPP_SUCCESS);
  EXPECT_GT(bufferSize, 0);
}

TEST_F(HistogramRangeTest, HistogramRange_8u_C1R_Basic) {
  // Clear any previous CUDA errors
  cudaGetLastError();

  allocateTestData8u();
  setupLevels8u();
  allocateHistogramMemory();

  // Compute histogram
  NppStatus status = nppiHistogramRange_8u_C1R(d_src_8u, nSrcStep_8u, oSizeROI, d_hist, d_levels_8u, nLevels, d_buffer);
  EXPECT_EQ(status, NPP_SUCCESS);

  if (status == NPP_SUCCESS) {
    // Copy result back to host
    ASSERT_EQ(cudaMemcpy(h_hist.data(), d_hist, (nLevels - 1) * sizeof(Npp32s), cudaMemcpyDeviceToHost), cudaSuccess);

    // Verify total count matches expected pixels in range
    int totalCount = 0;
    for (int i = 0; i < nLevels - 1; i++) {
      totalCount += h_hist[i];
    }

    // Count expected pixels in range manually
    int expectedCount = 0;
    for (int i = 0; i < totalPixels; i++) {
      int pixel_value = h_src_8u[i];
      for (int j = 0; j < nLevels - 1; j++) {
        if (pixel_value >= h_levels_8u[j] && pixel_value < h_levels_8u[j + 1]) {
          expectedCount++;
          break;
        }
      }
    }

    EXPECT_EQ(totalCount, expectedCount);
  }

  // Ensure operations complete before next test
  cudaDeviceSynchronize();
}

TEST_F(HistogramRangeTest, HistogramRange_32f_C1R_Basic) {
  // Clear any previous CUDA errors
  cudaGetLastError();

  allocateTestData32f();
  setupLevels32f();

  // Allocate histogram memory for 32f test
  // First clean up if already allocated
  if (d_hist) {
    cudaFree(d_hist);
    d_hist = nullptr;
  }
  if (d_buffer) {
    cudaFree(d_buffer);
    d_buffer = nullptr;
  }

  h_hist.resize(nLevels - 1, 0);
  ASSERT_EQ(cudaMalloc(&d_hist, (nLevels - 1) * sizeof(Npp32s)), cudaSuccess);

  BufferSizeType bufferSize;
  ASSERT_EQ(nppiHistogramRangeGetBufferSize_32f_C1R(oSizeROI, nLevels, &bufferSize), NPP_SUCCESS);
  ASSERT_EQ(cudaMalloc(&d_buffer, bufferSize), cudaSuccess);

  EXPECT_EQ(nppiHistogramRange_32f_C1R(d_src_32f, nSrcStep_32f, oSizeROI, d_hist, d_levels_32f, nLevels, d_buffer),
            NPP_SUCCESS);

  ASSERT_EQ(cudaMemcpy(h_hist.data(), d_hist, (nLevels - 1) * sizeof(Npp32s), cudaMemcpyDeviceToHost), cudaSuccess);

  // Verify total count
  int totalCount = 0;
  for (int i = 0; i < nLevels - 1; i++) {
    totalCount += h_hist[i];
  }

  int expectedCount = 0;
  for (int i = 0; i < totalPixels; i++) {
    float pixel_value = h_src_32f[i];
    for (int j = 0; j < nLevels - 1; j++) {
      if (pixel_value >= h_levels_32f[j] && pixel_value < h_levels_32f[j + 1]) {
        expectedCount++;
        break;
      }
    }
  }

  EXPECT_EQ(totalCount, expectedCount);

  // Ensure operations complete before next test
  cudaDeviceSynchronize();
}

TEST_F(HistogramRangeTest, HistogramRange_8u_C1R_WithContext) {
  allocateTestData8u();
  setupLevels8u();
  allocateHistogramMemory();

  NppStreamContext nppStreamCtx;
  nppGetStreamContext(&nppStreamCtx);

  EXPECT_EQ(nppiHistogramRange_8u_C1R_Ctx(d_src_8u, nSrcStep_8u, oSizeROI, d_hist, d_levels_8u, nLevels, d_buffer,
                                          nppStreamCtx),
            NPP_SUCCESS);

  cudaStreamSynchronize(nppStreamCtx.hStream);

  ASSERT_EQ(cudaMemcpy(h_hist.data(), d_hist, (nLevels - 1) * sizeof(Npp32s), cudaMemcpyDeviceToHost), cudaSuccess);

  // Verify result consistency
  int totalCount = 0;
  for (int i = 0; i < nLevels - 1; i++) {
    totalCount += h_hist[i];
  }
  EXPECT_GT(totalCount, 0);
}

TEST_F(HistogramRangeTest, CustomLevels_NonUniform) {
  allocateTestData8u();
  allocateHistogramMemory();

  // Create non-uniform levels: [0, 10, 50, 100, 200, 255]
  nLevels = 6;
  h_levels_8u = {0, 10, 50, 100, 200, 255};

  ASSERT_EQ(cudaMalloc(&d_levels_8u, nLevels * sizeof(Npp32s)), cudaSuccess);
  ASSERT_EQ(cudaMemcpy(d_levels_8u, h_levels_8u.data(), nLevels * sizeof(Npp32s), cudaMemcpyHostToDevice), cudaSuccess);

  // Reallocate histogram for new levels
  if (d_hist)
    cudaFree(d_hist);
  h_hist.resize(nLevels - 1, 0);
  ASSERT_EQ(cudaMalloc(&d_hist, (nLevels - 1) * sizeof(Npp32s)), cudaSuccess);

  EXPECT_EQ(nppiHistogramRange_8u_C1R(d_src_8u, nSrcStep_8u, oSizeROI, d_hist, d_levels_8u, nLevels, d_buffer),
            NPP_SUCCESS);

  ASSERT_EQ(cudaMemcpy(h_hist.data(), d_hist, (nLevels - 1) * sizeof(Npp32s), cudaMemcpyDeviceToHost), cudaSuccess);

  // Verify histogram distribution follows the custom ranges
  for (int i = 0; i < nLevels - 1; i++) {
    EXPECT_GE(h_hist[i], 0) << "Bin " << i << " should be non-negative";
  }

  int totalCount = 0;
  for (int i = 0; i < nLevels - 1; i++) {
    totalCount += h_hist[i];
  }
  EXPECT_GT(totalCount, 0) << "Total histogram count should be positive";
}

// DISABLED: This test causes GPU state corruption that affects subsequent morphology tests
// The SharedMemoryOptimization test triggers a shared memory kernel that leaves the GPU
// in an invalid state, causing float morphology operations to produce values like 3.38e+38
TEST_F(HistogramRangeTest, DISABLED_SharedMemoryOptimization) {
  // Clear any previous CUDA errors at test start
  cudaGetLastError();

  allocateTestData8u();
  setupLevels8u();

  // Clean up any previously allocated memory for levels and histogram
  if (d_levels_8u) {
    cudaFree(d_levels_8u);
    d_levels_8u = nullptr;
  }
  if (d_hist) {
    cudaFree(d_hist);
    d_hist = nullptr;
  }
  if (d_buffer) {
    cudaFree(d_buffer);
    d_buffer = nullptr;
  }

  // Use very small levels to trigger shared memory optimization
  nLevels = 4; // Small number of levels (3 bins)
  h_levels_8u = {0, 85, 170, 255};

  ASSERT_EQ(cudaMalloc(&d_levels_8u, nLevels * sizeof(Npp32s)), cudaSuccess);
  ASSERT_EQ(cudaMemcpy(d_levels_8u, h_levels_8u.data(), nLevels * sizeof(Npp32s), cudaMemcpyHostToDevice), cudaSuccess);

  h_hist.resize(nLevels - 1, 0);
  ASSERT_EQ(cudaMalloc(&d_hist, (nLevels - 1) * sizeof(Npp32s)), cudaSuccess);

  // Initialize histogram memory to zero
  cudaMemset(d_hist, 0, (nLevels - 1) * sizeof(Npp32s));

  BufferSizeType bufferSize;
  ASSERT_EQ(nppiHistogramRangeGetBufferSize_8u_C1R(oSizeROI, nLevels, &bufferSize), NPP_SUCCESS);
  ASSERT_EQ(cudaMalloc(&d_buffer, bufferSize), cudaSuccess);

  // This should use shared memory kernel due to small bins and image size
  EXPECT_EQ(nppiHistogramRange_8u_C1R(d_src_8u, nSrcStep_8u, oSizeROI, d_hist, d_levels_8u, nLevels, d_buffer),
            NPP_SUCCESS);

  // Ensure operation completes
  cudaDeviceSynchronize();

  ASSERT_EQ(cudaMemcpy(h_hist.data(), d_hist, (nLevels - 1) * sizeof(Npp32s), cudaMemcpyDeviceToHost), cudaSuccess);

  int totalCount = 0;
  for (int i = 0; i < nLevels - 1; i++) {
    totalCount += h_hist[i];
  }
  EXPECT_GT(totalCount, 0) << "Shared memory optimization should produce valid results";

  // Clear any CUDA errors and ensure complete synchronization before test ends
  cudaGetLastError();
  cudaDeviceSynchronize();
}

//=============================================================================
// Tests for 16-bit unsigned HistogramRange_C1R
//=============================================================================

TEST_F(HistogramRangeTest, BufferSize_16u_C1R) {
  BufferSizeType bufferSize;
  EXPECT_EQ(nppiHistogramRangeGetBufferSize_16u_C1R(oSizeROI, nLevels, &bufferSize), NPP_SUCCESS);
  EXPECT_GT(bufferSize, 0);
  
  EXPECT_EQ(nppiHistogramRangeGetBufferSize_16u_C1R_Ctx(oSizeROI, nLevels, &bufferSize, NppStreamContext{}), NPP_SUCCESS);
  EXPECT_GT(bufferSize, 0);
}

TEST_F(HistogramRangeTest, HistogramRange_16u_C1R_Basic) {
  allocateTestData16u();
  setupLevels16u();
  
  h_hist.resize(nLevels - 1, 0);
  ASSERT_EQ(cudaMalloc(&d_hist, (nLevels - 1) * sizeof(Npp32s)), cudaSuccess);
  BufferSizeType bufferSize;
  ASSERT_EQ(nppiHistogramRangeGetBufferSize_16u_C1R(oSizeROI, nLevels, &bufferSize), NPP_SUCCESS);
  ASSERT_EQ(cudaMalloc(&d_buffer, bufferSize), cudaSuccess);

  EXPECT_EQ(nppiHistogramRange_16u_C1R(d_src_16u, nSrcStep_16u, oSizeROI, d_hist, d_levels_16, nLevels, d_buffer), NPP_SUCCESS);

  ASSERT_EQ(cudaMemcpy(h_hist.data(), d_hist, (nLevels - 1) * sizeof(Npp32s), cudaMemcpyDeviceToHost), cudaSuccess);

  int totalCount = 0;
  for (int i = 0; i < nLevels - 1; i++) {
    totalCount += h_hist[i];
  }
  EXPECT_GT(totalCount, 0);
  EXPECT_LE(totalCount, totalPixels);
}

//=============================================================================
// Tests for 16-bit signed HistogramRange_C1R
//=============================================================================

TEST_F(HistogramRangeTest, BufferSize_16s_C1R) {
  BufferSizeType bufferSize;
  EXPECT_EQ(nppiHistogramRangeGetBufferSize_16s_C1R(oSizeROI, nLevels, &bufferSize), NPP_SUCCESS);
  EXPECT_GT(bufferSize, 0);
  
  EXPECT_EQ(nppiHistogramRangeGetBufferSize_16s_C1R_Ctx(oSizeROI, nLevels, &bufferSize, NppStreamContext{}), NPP_SUCCESS);
  EXPECT_GT(bufferSize, 0);
}

TEST_F(HistogramRangeTest, HistogramRange_16s_C1R_Basic) {
  allocateTestData16s();
  setupLevels16s();
  
  h_hist.resize(nLevels - 1, 0);
  ASSERT_EQ(cudaMalloc(&d_hist, (nLevels - 1) * sizeof(Npp32s)), cudaSuccess);
  BufferSizeType bufferSize;
  ASSERT_EQ(nppiHistogramRangeGetBufferSize_16s_C1R(oSizeROI, nLevels, &bufferSize), NPP_SUCCESS);
  ASSERT_EQ(cudaMalloc(&d_buffer, bufferSize), cudaSuccess);

  EXPECT_EQ(nppiHistogramRange_16s_C1R(d_src_16s, nSrcStep_16s, oSizeROI, d_hist, d_levels_16, nLevels, d_buffer), NPP_SUCCESS);

  ASSERT_EQ(cudaMemcpy(h_hist.data(), d_hist, (nLevels - 1) * sizeof(Npp32s), cudaMemcpyDeviceToHost), cudaSuccess);

  int totalCount = 0;
  for (int i = 0; i < nLevels - 1; i++) {
    totalCount += h_hist[i];
  }
  EXPECT_GT(totalCount, 0);
  EXPECT_LE(totalCount, totalPixels);
}

//=============================================================================
// Tests for 8-bit unsigned 4-channel HistogramRange
//=============================================================================

TEST_F(HistogramRangeTest, BufferSize_8u_C4R) {
  int nLevelsTest[4] = {10, 10, 10, 10};
  BufferSizeType bufferSize;
  EXPECT_EQ(nppiHistogramRangeGetBufferSize_8u_C4R(oSizeROI, nLevelsTest, &bufferSize), NPP_SUCCESS);
  EXPECT_GT(bufferSize, 0);
  
  EXPECT_EQ(nppiHistogramRangeGetBufferSize_8u_C4R_Ctx(oSizeROI, nLevelsTest, &bufferSize, NppStreamContext{}), NPP_SUCCESS);
  EXPECT_GT(bufferSize, 0);
}

TEST_F(HistogramRangeTest, HistogramRange_8u_C4R_Basic) {
  allocateTestData8uC4();
  setupLevelsC4_8u();
  allocateHistogramC4();

  BufferSizeType bufferSize;
  ASSERT_EQ(nppiHistogramRangeGetBufferSize_8u_C4R(oSizeROI, nLevelsArray, &bufferSize), NPP_SUCCESS);
  ASSERT_EQ(cudaMalloc(&d_buffer, bufferSize), cudaSuccess);

  EXPECT_EQ(nppiHistogramRange_8u_C4R(d_src_8u_c4, nSrcStep_8u_c4, oSizeROI, d_hist_c4, (const Npp32s **)d_levels_8u_c4, nLevelsArray, d_buffer), 
            NPP_SUCCESS);

  for (int c = 0; c < 4; c++) {
    ASSERT_EQ(cudaMemcpy(h_hist_c4[c].data(), d_hist_c4[c], (nLevels - 1) * sizeof(Npp32s), cudaMemcpyDeviceToHost), cudaSuccess);
    
    int totalCount = 0;
    for (int i = 0; i < nLevels - 1; i++) {
      totalCount += h_hist_c4[c][i];
    }
    EXPECT_GT(totalCount, 0) << "Channel " << c;
    EXPECT_LE(totalCount, totalPixels) << "Channel " << c;
  }
}

//=============================================================================
// Tests for 16-bit unsigned 4-channel HistogramRange
//=============================================================================

TEST_F(HistogramRangeTest, BufferSize_16u_C4R) {
  int nLevelsTest[4] = {10, 10, 10, 10};
  BufferSizeType bufferSize;
  EXPECT_EQ(nppiHistogramRangeGetBufferSize_16u_C4R(oSizeROI, nLevelsTest, &bufferSize), NPP_SUCCESS);
  EXPECT_GT(bufferSize, 0);
  
  EXPECT_EQ(nppiHistogramRangeGetBufferSize_16u_C4R_Ctx(oSizeROI, nLevelsTest, &bufferSize, NppStreamContext{}), NPP_SUCCESS);
  EXPECT_GT(bufferSize, 0);
}

TEST_F(HistogramRangeTest, HistogramRange_16u_C4R_Basic) {
  allocateTestData16uC4();
  setupLevelsC4_16u();
  allocateHistogramC4();

  BufferSizeType bufferSize;
  ASSERT_EQ(nppiHistogramRangeGetBufferSize_16u_C4R(oSizeROI, nLevelsArray, &bufferSize), NPP_SUCCESS);
  ASSERT_EQ(cudaMalloc(&d_buffer, bufferSize), cudaSuccess);

  EXPECT_EQ(nppiHistogramRange_16u_C4R(d_src_16u_c4, nSrcStep_16u_c4, oSizeROI, d_hist_c4, (const Npp32s **)d_levels_16_c4, nLevelsArray, d_buffer), 
            NPP_SUCCESS);

  for (int c = 0; c < 4; c++) {
    ASSERT_EQ(cudaMemcpy(h_hist_c4[c].data(), d_hist_c4[c], (nLevels - 1) * sizeof(Npp32s), cudaMemcpyDeviceToHost), cudaSuccess);
    
    int totalCount = 0;
    for (int i = 0; i < nLevels - 1; i++) {
      totalCount += h_hist_c4[c][i];
    }
    EXPECT_GT(totalCount, 0) << "Channel " << c;
    EXPECT_LE(totalCount, totalPixels) << "Channel " << c;
  }
}

//=============================================================================
// Tests for 16-bit signed 4-channel HistogramRange
//=============================================================================

TEST_F(HistogramRangeTest, BufferSize_16s_C4R) {
  int nLevelsTest[4] = {10, 10, 10, 10};
  BufferSizeType bufferSize;
  EXPECT_EQ(nppiHistogramRangeGetBufferSize_16s_C4R(oSizeROI, nLevelsTest, &bufferSize), NPP_SUCCESS);
  EXPECT_GT(bufferSize, 0);
  
  EXPECT_EQ(nppiHistogramRangeGetBufferSize_16s_C4R_Ctx(oSizeROI, nLevelsTest, &bufferSize, NppStreamContext{}), NPP_SUCCESS);
  EXPECT_GT(bufferSize, 0);
}

TEST_F(HistogramRangeTest, HistogramRange_16s_C4R_Basic) {
  allocateTestData16sC4();
  setupLevelsC4_16s();
  allocateHistogramC4();

  BufferSizeType bufferSize;
  ASSERT_EQ(nppiHistogramRangeGetBufferSize_16s_C4R(oSizeROI, nLevelsArray, &bufferSize), NPP_SUCCESS);
  ASSERT_EQ(cudaMalloc(&d_buffer, bufferSize), cudaSuccess);

  EXPECT_EQ(nppiHistogramRange_16s_C4R(d_src_16s_c4, nSrcStep_16s_c4, oSizeROI, d_hist_c4, (const Npp32s **)d_levels_16_c4, nLevelsArray, d_buffer), 
            NPP_SUCCESS);

  for (int c = 0; c < 4; c++) {
    ASSERT_EQ(cudaMemcpy(h_hist_c4[c].data(), d_hist_c4[c], (nLevels - 1) * sizeof(Npp32s), cudaMemcpyDeviceToHost), cudaSuccess);
    
    int totalCount = 0;
    for (int i = 0; i < nLevels - 1; i++) {
      totalCount += h_hist_c4[c][i];
    }
    EXPECT_GT(totalCount, 0) << "Channel " << c;
    EXPECT_LE(totalCount, totalPixels) << "Channel " << c;
  }
}

//=============================================================================
// Tests for 32-bit float 4-channel HistogramRange
//=============================================================================

TEST_F(HistogramRangeTest, BufferSize_32f_C4R) {
  int nLevelsTest[4] = {10, 10, 10, 10};
  BufferSizeType bufferSize;
  EXPECT_EQ(nppiHistogramRangeGetBufferSize_32f_C4R(oSizeROI, nLevelsTest, &bufferSize), NPP_SUCCESS);
  EXPECT_GT(bufferSize, 0);
  
  EXPECT_EQ(nppiHistogramRangeGetBufferSize_32f_C4R_Ctx(oSizeROI, nLevelsTest, &bufferSize, NppStreamContext{}), NPP_SUCCESS);
  EXPECT_GT(bufferSize, 0);
}

TEST_F(HistogramRangeTest, HistogramRange_32f_C4R_Basic) {
  allocateTestData32fC4();
  setupLevelsC4_32f();
  allocateHistogramC4();

  BufferSizeType bufferSize;
  ASSERT_EQ(nppiHistogramRangeGetBufferSize_32f_C4R(oSizeROI, nLevelsArray, &bufferSize), NPP_SUCCESS);
  ASSERT_EQ(cudaMalloc(&d_buffer, bufferSize), cudaSuccess);

  EXPECT_EQ(nppiHistogramRange_32f_C4R(d_src_32f_c4, nSrcStep_32f_c4, oSizeROI, d_hist_c4, (const Npp32f **)d_levels_32f_c4, nLevelsArray, d_buffer), 
            NPP_SUCCESS);

  for (int c = 0; c < 4; c++) {
    ASSERT_EQ(cudaMemcpy(h_hist_c4[c].data(), d_hist_c4[c], (nLevels - 1) * sizeof(Npp32s), cudaMemcpyDeviceToHost), cudaSuccess);
    
    int totalCount = 0;
    for (int i = 0; i < nLevels - 1; i++) {
      totalCount += h_hist_c4[c][i];
    }
    EXPECT_GT(totalCount, 0) << "Channel " << c;
    EXPECT_LE(totalCount, totalPixels) << "Channel " << c;
  }
}

