#include "npp.h"
#include "npp_test_base.h"
#include <algorithm>
#include <gtest/gtest.h>
#include <vector>

class NppiAlphaPremulTest : public ::testing::Test {
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

// ============================================================================
// AlphaPremulC Tests - Alpha premultiplication with constant alpha values
// ============================================================================

TEST_F(NppiAlphaPremulTest, AlphaPremulC_8u_C1R_BasicOperation) {
  const int width = 4;
  const int height = 2;
  const int totalPixels = width * height;

  std::vector<Npp8u> hostSrc = {100, 150, 200, 50, 75, 125, 175, 225};
  Npp8u alpha = 128; // 0.5 in 8-bit

  // Expected result: src * alpha / 255 (NVIDIA NPP AlphaPremulC behavior)
  std::vector<Npp8u> expected(totalPixels);
  for (int i = 0; i < totalPixels; ++i) {
    int result = (hostSrc[i] * alpha) / 255;
    expected[i] = static_cast<Npp8u>(result);
  }

  // Allocate GPU memory
  int srcStep, dstStep;
  Npp8u *d_src = nppiMalloc_8u_C1(width, height, &srcStep);
  Npp8u *d_dst = nppiMalloc_8u_C1(width, height, &dstStep);

  ASSERT_NE(d_src, nullptr);
  ASSERT_NE(d_dst, nullptr);

  // Copy data to GPU
  int hostStep = width * sizeof(Npp8u);
  cudaMemcpy2D(d_src, srcStep, hostSrc.data(), hostStep, hostStep, height, cudaMemcpyHostToDevice);

  // Execute operation
  NppiSize oSizeROI = {width, height};
  NppStatus status = nppiAlphaPremulC_8u_C1R(d_src, srcStep, alpha, d_dst, dstStep, oSizeROI);
  ASSERT_EQ(status, NPP_SUCCESS);

  // Copy result back
  std::vector<Npp8u> hostResult(totalPixels);
  cudaMemcpy2D(hostResult.data(), hostStep, d_dst, dstStep, hostStep, height, cudaMemcpyDeviceToHost);

  // Verify results
  for (int i = 0; i < totalPixels; ++i) {
    EXPECT_EQ(hostResult[i], expected[i]) << "Mismatch at index " << i << " got " << static_cast<int>(hostResult[i])
                                          << " expected " << static_cast<int>(expected[i]);
  }

  // Cleanup
  nppiFree(d_src);
  nppiFree(d_dst);
}

TEST_F(NppiAlphaPremulTest, AlphaPremulC_8u_C1IR_InPlaceOperation) {
  const int width = 3;
  const int height = 2;
  const int totalPixels = width * height;

  std::vector<Npp8u> hostSrc = {100, 150, 200, 50, 75, 125};
  Npp8u alpha = 192; // 0.75 in 8-bit

  // Expected result: src * alpha / 255 (NVIDIA NPP AlphaPremulC behavior)
  std::vector<Npp8u> expected(totalPixels);
  for (int i = 0; i < totalPixels; ++i) {
    int result = (hostSrc[i] * alpha) / 255;
    expected[i] = static_cast<Npp8u>(result);
  }

  // Allocate GPU memory
  int srcDstStep;
  Npp8u *d_srcDst = nppiMalloc_8u_C1(width, height, &srcDstStep);

  ASSERT_NE(d_srcDst, nullptr);

  // Copy data to GPU
  int hostStep = width * sizeof(Npp8u);
  cudaMemcpy2D(d_srcDst, srcDstStep, hostSrc.data(), hostStep, hostStep, height, cudaMemcpyHostToDevice);

  // Execute operation
  NppiSize oSizeROI = {width, height};
  NppStatus status = nppiAlphaPremulC_8u_C1IR(alpha, d_srcDst, srcDstStep, oSizeROI);
  ASSERT_EQ(status, NPP_SUCCESS);

  // Copy result back
  std::vector<Npp8u> hostResult(totalPixels);
  cudaMemcpy2D(hostResult.data(), hostStep, d_srcDst, srcDstStep, hostStep, height, cudaMemcpyDeviceToHost);

  // Verify results
  for (int i = 0; i < totalPixels; ++i) {
    EXPECT_EQ(hostResult[i], expected[i]) << "Mismatch at index " << i << " got " << static_cast<int>(hostResult[i])
                                          << " expected " << static_cast<int>(expected[i]);
  }

  // Cleanup
  nppiFree(d_srcDst);
}

TEST_F(NppiAlphaPremulTest, AlphaPremulC_8u_C3R_MultiChannel) {
  const int width = 2;
  const int height = 2;
  const int channels = 3;
  const int totalPixels = width * height * channels;

  // RGB data: R,G,B,R,G,B,...
  std::vector<Npp8u> hostSrc = {255, 128, 64, 192, 96, 32,   // Pixel 1, 2
                                128, 255, 0,  64,  32, 255}; // Pixel 3, 4
  Npp8u alpha = 128;                                         // 0.5 in 8-bit

  // Expected result: each channel * alpha / 255 (NVIDIA NPP AlphaPremulC behavior)
  std::vector<Npp8u> expected(totalPixels);
  for (int i = 0; i < totalPixels; ++i) {
    int result = (hostSrc[i] * alpha) / 255;
    expected[i] = static_cast<Npp8u>(result);
  }

  // Allocate GPU memory
  int srcStep, dstStep;
  Npp8u *d_src = nppiMalloc_8u_C3(width, height, &srcStep);
  Npp8u *d_dst = nppiMalloc_8u_C3(width, height, &dstStep);

  ASSERT_NE(d_src, nullptr);
  ASSERT_NE(d_dst, nullptr);

  // Copy data to GPU
  int hostStep = width * channels * sizeof(Npp8u);
  cudaMemcpy2D(d_src, srcStep, hostSrc.data(), hostStep, hostStep, height, cudaMemcpyHostToDevice);

  // Execute operation
  NppiSize oSizeROI = {width, height};
  NppStatus status = nppiAlphaPremulC_8u_C3R(d_src, srcStep, alpha, d_dst, dstStep, oSizeROI);
  ASSERT_EQ(status, NPP_SUCCESS);

  // Copy result back
  std::vector<Npp8u> hostResult(totalPixels);
  cudaMemcpy2D(hostResult.data(), hostStep, d_dst, dstStep, hostStep, height, cudaMemcpyDeviceToHost);

  // Verify results (with tolerance for rounding)
  for (int i = 0; i < totalPixels; ++i) {
    EXPECT_NEAR(hostResult[i], expected[i], 1) << "Mismatch at index " << i << " got " << static_cast<int>(hostResult[i])
                                               << " expected " << static_cast<int>(expected[i]);
  }

  // Cleanup
  nppiFree(d_src);
  nppiFree(d_dst);
}

TEST_F(NppiAlphaPremulTest, AlphaPremulC_16u_C1R_BasicOperation) {
  const int width = 3;
  const int height = 2;
  const int totalPixels = width * height;

  std::vector<Npp16u> hostSrc = {10000, 20000, 30000, 40000, 50000, 60000};
  Npp16u alpha = 32768; // 0.5 in 16-bit

  // Expected result: src * alpha / 65535 (NVIDIA NPP AlphaPremulC behavior)
  std::vector<Npp16u> expected(totalPixels);
  for (int i = 0; i < totalPixels; ++i) {
    int64_t result = (static_cast<int64_t>(hostSrc[i]) * alpha) / 65535;
    expected[i] = static_cast<Npp16u>(result);
  }

  // Allocate GPU memory
  int srcStep, dstStep;
  Npp16u *d_src = nppiMalloc_16u_C1(width, height, &srcStep);
  Npp16u *d_dst = nppiMalloc_16u_C1(width, height, &dstStep);

  ASSERT_NE(d_src, nullptr);
  ASSERT_NE(d_dst, nullptr);

  // Copy data to GPU
  int hostStep = width * sizeof(Npp16u);
  cudaMemcpy2D(d_src, srcStep, hostSrc.data(), hostStep, hostStep, height, cudaMemcpyHostToDevice);

  // Execute operation
  NppiSize oSizeROI = {width, height};
  NppStatus status = nppiAlphaPremulC_16u_C1R(d_src, srcStep, alpha, d_dst, dstStep, oSizeROI);
  ASSERT_EQ(status, NPP_SUCCESS);

  // Copy result back
  std::vector<Npp16u> hostResult(totalPixels);
  cudaMemcpy2D(hostResult.data(), hostStep, d_dst, dstStep, hostStep, height, cudaMemcpyDeviceToHost);

  // Verify results
  for (int i = 0; i < totalPixels; ++i) {
    EXPECT_EQ(hostResult[i], expected[i])
        << "Mismatch at index " << i << " got " << hostResult[i] << " expected " << expected[i];
  }

  // Cleanup
  nppiFree(d_src);
  nppiFree(d_dst);
}

// ============================================================================
// AlphaPremul Tests - Alpha premultiplication with pixel alpha values (AC4)
// ============================================================================

TEST_F(NppiAlphaPremulTest, AlphaPremul_8u_AC4R_PixelAlpha) {
  const int width = 2;
  const int height = 1;
  const int channels = 4;
  const int totalPixels = width * height * channels;

  // RGBA data with alpha in 4th channel
  std::vector<Npp8u> hostSrc = {255, 128, 64, 192,  // Pixel 1: R=255, G=128, B=64, A=192
                                128, 255, 32, 128}; // Pixel 2: R=128, G=255, B=32, A=128

  // Expected result: R,G,B channels premultiplied by A, A unchanged
  std::vector<Npp8u> expected(totalPixels);
  for (int p = 0; p < width * height; ++p) {
    int idx = p * 4;
    Npp8u alpha = hostSrc[idx + 3];

    // Premultiply RGB with alpha (NVIDIA uses right shift for pixel alpha)
    for (int c = 0; c < 3; ++c) {
      int result = (hostSrc[idx + c] * alpha) >> 8;
      expected[idx + c] = static_cast<Npp8u>(result);
    }
    // Alpha channel unchanged
    expected[idx + 3] = hostSrc[idx + 3];
  }

  // Allocate GPU memory
  int srcStep, dstStep;
  Npp8u *d_src = nppiMalloc_8u_C4(width, height, &srcStep);
  Npp8u *d_dst = nppiMalloc_8u_C4(width, height, &dstStep);

  ASSERT_NE(d_src, nullptr);
  ASSERT_NE(d_dst, nullptr);

  // Copy data to GPU
  int hostStep = width * channels * sizeof(Npp8u);
  cudaMemcpy2D(d_src, srcStep, hostSrc.data(), hostStep, hostStep, height, cudaMemcpyHostToDevice);

  // Execute operation
  NppiSize oSizeROI = {width, height};
  NppStatus status = nppiAlphaPremul_8u_AC4R(d_src, srcStep, d_dst, dstStep, oSizeROI);
  ASSERT_EQ(status, NPP_SUCCESS);

  // Copy result back
  std::vector<Npp8u> hostResult(totalPixels);
  cudaMemcpy2D(hostResult.data(), hostStep, d_dst, dstStep, hostStep, height, cudaMemcpyDeviceToHost);

  // Verify results
  for (int i = 0; i < totalPixels; ++i) {
    EXPECT_EQ(hostResult[i], expected[i]) << "Mismatch at index " << i << " got " << static_cast<int>(hostResult[i])
                                          << " expected " << static_cast<int>(expected[i]);
  }

  // Cleanup
  nppiFree(d_src);
  nppiFree(d_dst);
}

TEST_F(NppiAlphaPremulTest, AlphaPremul_8u_AC4IR_InPlacePixelAlpha) {
  const int width = 2;
  const int height = 1;
  const int channels = 4;
  const int totalPixels = width * height * channels;

  // RGBA data with alpha in 4th channel
  std::vector<Npp8u> hostSrc = {200, 100, 50,  255, // Pixel 1: R=200, G=100, B=50, A=255
                                100, 200, 150, 64}; // Pixel 2: R=100, G=200, B=150, A=64

  // Expected result: R,G,B channels premultiplied by A, A unchanged
  std::vector<Npp8u> expected(totalPixels);
  for (int p = 0; p < width * height; ++p) {
    int idx = p * 4;
    Npp8u alpha = hostSrc[idx + 3];

    // Premultiply RGB with alpha (NVIDIA uses right shift for pixel alpha)
    for (int c = 0; c < 3; ++c) {
      int result = (hostSrc[idx + c] * alpha) >> 8;
      expected[idx + c] = static_cast<Npp8u>(result);
    }
    // Alpha channel unchanged
    expected[idx + 3] = hostSrc[idx + 3];
  }

  // Allocate GPU memory
  int srcDstStep;
  Npp8u *d_srcDst = nppiMalloc_8u_C4(width, height, &srcDstStep);

  ASSERT_NE(d_srcDst, nullptr);

  // Copy data to GPU
  int hostStep = width * channels * sizeof(Npp8u);
  cudaMemcpy2D(d_srcDst, srcDstStep, hostSrc.data(), hostStep, hostStep, height, cudaMemcpyHostToDevice);

  // Execute operation
  NppiSize oSizeROI = {width, height};
  NppStatus status = nppiAlphaPremul_8u_AC4IR(d_srcDst, srcDstStep, oSizeROI);
  ASSERT_EQ(status, NPP_SUCCESS);

  // Copy result back
  std::vector<Npp8u> hostResult(totalPixels);
  cudaMemcpy2D(hostResult.data(), hostStep, d_srcDst, srcDstStep, hostStep, height, cudaMemcpyDeviceToHost);

  // Verify results (with tolerance for rounding)
  for (int i = 0; i < totalPixels; ++i) {
    EXPECT_NEAR(hostResult[i], expected[i], 1) << "Mismatch at index " << i << " got " << static_cast<int>(hostResult[i])
                                               << " expected " << static_cast<int>(expected[i]);
  }

  // Cleanup
  nppiFree(d_srcDst);
}

TEST_F(NppiAlphaPremulTest, AlphaPremul_16u_AC4R_PixelAlpha) {
  const int width = 2;
  const int height = 1;
  const int channels = 4;
  const int totalPixels = width * height * channels;

  // RGBA data with alpha in 4th channel
  std::vector<Npp16u> hostSrc = {50000, 30000, 20000, 40000,  // Pixel 1
                                 25000, 35000, 15000, 20000}; // Pixel 2

  // Expected result: R,G,B channels premultiplied by A, A unchanged
  std::vector<Npp16u> expected(totalPixels);
  for (int p = 0; p < width * height; ++p) {
    int idx = p * 4;
    Npp16u alpha = hostSrc[idx + 3];

    // Premultiply RGB with alpha (NVIDIA uses division for 16u pixel alpha)
    for (int c = 0; c < 3; ++c) {
      int64_t result = (static_cast<int64_t>(hostSrc[idx + c]) * alpha) / 65535;
      expected[idx + c] = static_cast<Npp16u>(result);
    }
    // Alpha channel unchanged
    expected[idx + 3] = hostSrc[idx + 3];
  }

  // Allocate GPU memory
  int srcStep, dstStep;
  Npp16u *d_src = nppiMalloc_16u_C4(width, height, &srcStep);
  Npp16u *d_dst = nppiMalloc_16u_C4(width, height, &dstStep);

  ASSERT_NE(d_src, nullptr);
  ASSERT_NE(d_dst, nullptr);

  // Copy data to GPU
  int hostStep = width * channels * sizeof(Npp16u);
  cudaMemcpy2D(d_src, srcStep, hostSrc.data(), hostStep, hostStep, height, cudaMemcpyHostToDevice);

  // Execute operation
  NppiSize oSizeROI = {width, height};
  NppStatus status = nppiAlphaPremul_16u_AC4R(d_src, srcStep, d_dst, dstStep, oSizeROI);
  ASSERT_EQ(status, NPP_SUCCESS);

  // Copy result back
  std::vector<Npp16u> hostResult(totalPixels);
  cudaMemcpy2D(hostResult.data(), hostStep, d_dst, dstStep, hostStep, height, cudaMemcpyDeviceToHost);

  // Verify results (with tolerance for rounding)
  for (int i = 0; i < totalPixels; ++i) {
    EXPECT_NEAR(hostResult[i], expected[i], 1)
        << "Mismatch at index " << i << " got " << hostResult[i] << " expected " << expected[i];
  }

  // Cleanup
  nppiFree(d_src);
  nppiFree(d_dst);
}

// Test error handling
TEST_F(NppiAlphaPremulTest, AlphaPremulC_NullPointer) {
  NppiSize oSizeROI = {2, 2};
  Npp8u alpha = 128;

  // Test null source pointer
  NppStatus status = nppiAlphaPremulC_8u_C1R(nullptr, 0, alpha, nullptr, 0, oSizeROI);
  EXPECT_EQ(status, NPP_NULL_POINTER_ERROR);

  // Test null destination pointer for in-place operation
  status = nppiAlphaPremulC_8u_C1IR(alpha, nullptr, 0, oSizeROI);
  EXPECT_EQ(status, NPP_NULL_POINTER_ERROR);
}

TEST_F(NppiAlphaPremulTest, AlphaPremul_EdgeCases) {
  const int width = 2;
  const int height = 1;

  // Test alpha = 0 (should result in 0)
  std::vector<Npp8u> hostSrc = {255, 128};
  Npp8u alpha = 0;

  int srcStep, dstStep;
  Npp8u *d_src = nppiMalloc_8u_C1(width, height, &srcStep);
  Npp8u *d_dst = nppiMalloc_8u_C1(width, height, &dstStep);

  int hostStep = width * sizeof(Npp8u);
  cudaMemcpy2D(d_src, srcStep, hostSrc.data(), hostStep, hostStep, height, cudaMemcpyHostToDevice);

  NppiSize oSizeROI = {width, height};
  NppStatus status = nppiAlphaPremulC_8u_C1R(d_src, srcStep, alpha, d_dst, dstStep, oSizeROI);
  ASSERT_EQ(status, NPP_SUCCESS);

  std::vector<Npp8u> hostResult(width * height);
  cudaMemcpy2D(hostResult.data(), hostStep, d_dst, dstStep, hostStep, height, cudaMemcpyDeviceToHost);

  // With alpha=0, all results should be 0
  for (int i = 0; i < width * height; ++i) {
    EXPECT_EQ(hostResult[i], 0) << "Mismatch at index " << i;
  }

  nppiFree(d_src);
  nppiFree(d_dst);
}