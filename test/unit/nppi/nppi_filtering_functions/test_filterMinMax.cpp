// NPP FilterMax and FilterMin Tests
// Tests for morphological dilation (max) and erosion (min) filters
//
// NOTE: nppiFilterMax/Min functions require source image to have sufficient
// boundary data outside the ROI. These tests use boundary expansion to ensure
// proper results.

#include "npp.h"
#include <algorithm>
#include <cmath>
#include <cuda_runtime.h>
#include <functional>
#include <gtest/gtest.h>
#include <limits>
#include <random>
#include <vector>

// Test fixture for FilterMax/FilterMin tests
class FilterMinMaxTest : public ::testing::Test {
protected:
  void SetUp() override {
    cudaError_t err = cudaSetDevice(0);
    ASSERT_EQ(err, cudaSuccess);
  }

  void TearDown() override { cudaDeviceSynchronize(); }
};

// Helper function to compute CPU reference for max filter with border replication
template <typename T>
std::vector<T> computeMaxFilterCPU(const std::vector<T> &input, int width, int height, int channels, int maskW,
                                   int maskH, int anchorX, int anchorY) {
  std::vector<T> output(width * height * channels);

  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      for (int c = 0; c < channels; c++) {
        T maxVal = std::numeric_limits<T>::lowest();

        for (int ky = 0; ky < maskH; ky++) {
          for (int kx = 0; kx < maskW; kx++) {
            int srcX = x + kx - anchorX;
            int srcY = y + ky - anchorY;

            // Clamp to image boundaries (replicate border)
            srcX = std::max(0, std::min(width - 1, srcX));
            srcY = std::max(0, std::min(height - 1, srcY));

            T val = input[(srcY * width + srcX) * channels + c];
            maxVal = std::max(maxVal, val);
          }
        }

        output[(y * width + x) * channels + c] = maxVal;
      }
    }
  }

  return output;
}

// Helper function to compute CPU reference for min filter with border replication
template <typename T>
std::vector<T> computeMinFilterCPU(const std::vector<T> &input, int width, int height, int channels, int maskW,
                                   int maskH, int anchorX, int anchorY) {
  std::vector<T> output(width * height * channels);

  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      for (int c = 0; c < channels; c++) {
        T minVal = std::numeric_limits<T>::max();

        for (int ky = 0; ky < maskH; ky++) {
          for (int kx = 0; kx < maskW; kx++) {
            int srcX = x + kx - anchorX;
            int srcY = y + ky - anchorY;

            // Clamp to image boundaries (replicate border)
            srcX = std::max(0, std::min(width - 1, srcX));
            srcY = std::max(0, std::min(height - 1, srcY));

            T val = input[(srcY * width + srcX) * channels + c];
            minVal = std::min(minVal, val);
          }
        }

        output[(y * width + x) * channels + c] = minVal;
      }
    }
  }

  return output;
}

// ============================================================================
// Helper functions to run filters with proper boundary expansion
// ============================================================================

// Run FilterMax with boundary expansion for 8u_C1R
std::vector<Npp8u> runFilterMax8uC1R(const std::vector<Npp8u> &input, int width, int height, int maskW, int maskH,
                                     int anchorX, int anchorY) {
  int padLeft = anchorX;
  int padRight = maskW - anchorX - 1;
  int padTop = anchorY;
  int padBottom = maskH - anchorY - 1;

  int expandedWidth = width + padLeft + padRight;
  int expandedHeight = height + padTop + padBottom;

  // Create expanded input with edge replication
  std::vector<Npp8u> expandedInput(expandedWidth * expandedHeight);
  for (int y = 0; y < expandedHeight; y++) {
    for (int x = 0; x < expandedWidth; x++) {
      int srcX = std::max(0, std::min(width - 1, x - padLeft));
      int srcY = std::max(0, std::min(height - 1, y - padTop));
      expandedInput[y * expandedWidth + x] = input[srcY * width + srcX];
    }
  }

  int srcStep, dstStep;
  Npp8u *d_src = nppiMalloc_8u_C1(expandedWidth, expandedHeight, &srcStep);
  Npp8u *d_dst = nppiMalloc_8u_C1(expandedWidth, expandedHeight, &dstStep);

  cudaMemcpy2D(d_src, srcStep, expandedInput.data(), expandedWidth, expandedWidth, expandedHeight,
               cudaMemcpyHostToDevice);

  NppiSize oSizeROI = {expandedWidth, expandedHeight};
  NppiSize oMaskSize = {maskW, maskH};
  NppiPoint oAnchor = {anchorX, anchorY};

  nppiFilterMax_8u_C1R(d_src, srcStep, d_dst, dstStep, oSizeROI, oMaskSize, oAnchor);

  std::vector<Npp8u> expandedOutput(expandedWidth * expandedHeight);
  cudaMemcpy2D(expandedOutput.data(), expandedWidth, d_dst, dstStep, expandedWidth, expandedHeight,
               cudaMemcpyDeviceToHost);

  nppiFree(d_src);
  nppiFree(d_dst);

  // Crop to original size
  std::vector<Npp8u> output(width * height);
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      output[y * width + x] = expandedOutput[(y + padTop) * expandedWidth + (x + padLeft)];
    }
  }

  return output;
}

// Run FilterMin with boundary expansion for 8u_C1R
std::vector<Npp8u> runFilterMin8uC1R(const std::vector<Npp8u> &input, int width, int height, int maskW, int maskH,
                                     int anchorX, int anchorY) {
  int padLeft = anchorX;
  int padRight = maskW - anchorX - 1;
  int padTop = anchorY;
  int padBottom = maskH - anchorY - 1;

  int expandedWidth = width + padLeft + padRight;
  int expandedHeight = height + padTop + padBottom;

  std::vector<Npp8u> expandedInput(expandedWidth * expandedHeight);
  for (int y = 0; y < expandedHeight; y++) {
    for (int x = 0; x < expandedWidth; x++) {
      int srcX = std::max(0, std::min(width - 1, x - padLeft));
      int srcY = std::max(0, std::min(height - 1, y - padTop));
      expandedInput[y * expandedWidth + x] = input[srcY * width + srcX];
    }
  }

  int srcStep, dstStep;
  Npp8u *d_src = nppiMalloc_8u_C1(expandedWidth, expandedHeight, &srcStep);
  Npp8u *d_dst = nppiMalloc_8u_C1(expandedWidth, expandedHeight, &dstStep);

  cudaMemcpy2D(d_src, srcStep, expandedInput.data(), expandedWidth, expandedWidth, expandedHeight,
               cudaMemcpyHostToDevice);

  NppiSize oSizeROI = {expandedWidth, expandedHeight};
  NppiSize oMaskSize = {maskW, maskH};
  NppiPoint oAnchor = {anchorX, anchorY};

  nppiFilterMin_8u_C1R(d_src, srcStep, d_dst, dstStep, oSizeROI, oMaskSize, oAnchor);

  std::vector<Npp8u> expandedOutput(expandedWidth * expandedHeight);
  cudaMemcpy2D(expandedOutput.data(), expandedWidth, d_dst, dstStep, expandedWidth, expandedHeight,
               cudaMemcpyDeviceToHost);

  nppiFree(d_src);
  nppiFree(d_dst);

  std::vector<Npp8u> output(width * height);
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      output[y * width + x] = expandedOutput[(y + padTop) * expandedWidth + (x + padLeft)];
    }
  }

  return output;
}

// ============================================================================
// FilterMax 8u_C1R Tests
// ============================================================================

TEST_F(FilterMinMaxTest, FilterMax_8u_C1R_Basic) {
  const int width = 16;
  const int height = 16;

  std::vector<Npp8u> h_src(width * height);
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      h_src[y * width + x] = static_cast<Npp8u>((x + y * 16) % 256);
    }
  }

  auto result = runFilterMax8uC1R(h_src, width, height, 3, 3, 1, 1);
  auto expected = computeMaxFilterCPU<Npp8u>(h_src, width, height, 1, 3, 3, 1, 1);

  for (int i = 0; i < width * height; i++) {
    EXPECT_EQ(result[i], expected[i]) << "Mismatch at index " << i;
  }
}

TEST_F(FilterMinMaxTest, FilterMax_8u_C1R_Ctx) {
  const int width = 16;
  const int height = 16;

  // Use boundary expansion approach
  int maskW = 3, maskH = 3, anchorX = 1, anchorY = 1;
  int padLeft = anchorX, padRight = maskW - anchorX - 1;
  int padTop = anchorY, padBottom = maskH - anchorY - 1;
  int expandedWidth = width + padLeft + padRight;
  int expandedHeight = height + padTop + padBottom;

  std::vector<Npp8u> h_src(width * height);
  for (int i = 0; i < width * height; i++) {
    h_src[i] = static_cast<Npp8u>(i % 256);
  }

  std::vector<Npp8u> expandedInput(expandedWidth * expandedHeight);
  for (int y = 0; y < expandedHeight; y++) {
    for (int x = 0; x < expandedWidth; x++) {
      int srcX = std::max(0, std::min(width - 1, x - padLeft));
      int srcY = std::max(0, std::min(height - 1, y - padTop));
      expandedInput[y * expandedWidth + x] = h_src[srcY * width + srcX];
    }
  }

  int srcStep, dstStep;
  Npp8u *d_src = nppiMalloc_8u_C1(expandedWidth, expandedHeight, &srcStep);
  Npp8u *d_dst = nppiMalloc_8u_C1(expandedWidth, expandedHeight, &dstStep);
  ASSERT_NE(d_src, nullptr);
  ASSERT_NE(d_dst, nullptr);

  cudaMemcpy2D(d_src, srcStep, expandedInput.data(), expandedWidth, expandedWidth, expandedHeight,
               cudaMemcpyHostToDevice);

  NppiSize oSizeROI = {expandedWidth, expandedHeight};
  NppiSize oMaskSize = {maskW, maskH};
  NppiPoint oAnchor = {anchorX, anchorY};
  NppStreamContext ctx;
  ctx.hStream = 0;

  NppStatus status = nppiFilterMax_8u_C1R_Ctx(d_src, srcStep, d_dst, dstStep, oSizeROI, oMaskSize, oAnchor, ctx);
  EXPECT_EQ(status, NPP_SUCCESS);

  std::vector<Npp8u> expandedOutput(expandedWidth * expandedHeight);
  cudaMemcpy2D(expandedOutput.data(), expandedWidth, d_dst, dstStep, expandedWidth, expandedHeight,
               cudaMemcpyDeviceToHost);

  // Crop and verify
  std::vector<Npp8u> result(width * height);
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      result[y * width + x] = expandedOutput[(y + padTop) * expandedWidth + (x + padLeft)];
    }
  }

  auto expected = computeMaxFilterCPU<Npp8u>(h_src, width, height, 1, maskW, maskH, anchorX, anchorY);
  for (int i = 0; i < width * height; i++) {
    EXPECT_EQ(result[i], expected[i]) << "Mismatch at index " << i;
  }

  nppiFree(d_src);
  nppiFree(d_dst);
}

TEST_F(FilterMinMaxTest, FilterMax_8u_C1R_5x5Mask) {
  const int width = 32;
  const int height = 32;

  std::vector<Npp8u> h_src(width * height);
  std::mt19937 rng(42);
  for (int i = 0; i < width * height; i++) {
    h_src[i] = static_cast<Npp8u>(rng() % 256);
  }

  auto result = runFilterMax8uC1R(h_src, width, height, 5, 5, 2, 2);
  auto expected = computeMaxFilterCPU<Npp8u>(h_src, width, height, 1, 5, 5, 2, 2);

  for (int i = 0; i < width * height; i++) {
    EXPECT_EQ(result[i], expected[i]) << "Mismatch at index " << i;
  }
}

// ============================================================================
// FilterMin 8u_C1R Tests
// ============================================================================

TEST_F(FilterMinMaxTest, FilterMin_8u_C1R_Basic) {
  const int width = 16;
  const int height = 16;

  std::vector<Npp8u> h_src(width * height);
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      h_src[y * width + x] = static_cast<Npp8u>((x + y * 16) % 256);
    }
  }

  auto result = runFilterMin8uC1R(h_src, width, height, 3, 3, 1, 1);
  auto expected = computeMinFilterCPU<Npp8u>(h_src, width, height, 1, 3, 3, 1, 1);

  for (int i = 0; i < width * height; i++) {
    EXPECT_EQ(result[i], expected[i]) << "Mismatch at index " << i;
  }
}

TEST_F(FilterMinMaxTest, FilterMin_8u_C1R_Ctx) {
  const int width = 16;
  const int height = 16;
  int maskW = 3, maskH = 3, anchorX = 1, anchorY = 1;
  int padLeft = anchorX, padRight = maskW - anchorX - 1;
  int padTop = anchorY, padBottom = maskH - anchorY - 1;
  int expandedWidth = width + padLeft + padRight;
  int expandedHeight = height + padTop + padBottom;

  std::vector<Npp8u> h_src(width * height);
  for (int i = 0; i < width * height; i++) {
    h_src[i] = static_cast<Npp8u>(i % 256);
  }

  std::vector<Npp8u> expandedInput(expandedWidth * expandedHeight);
  for (int y = 0; y < expandedHeight; y++) {
    for (int x = 0; x < expandedWidth; x++) {
      int srcX = std::max(0, std::min(width - 1, x - padLeft));
      int srcY = std::max(0, std::min(height - 1, y - padTop));
      expandedInput[y * expandedWidth + x] = h_src[srcY * width + srcX];
    }
  }

  int srcStep, dstStep;
  Npp8u *d_src = nppiMalloc_8u_C1(expandedWidth, expandedHeight, &srcStep);
  Npp8u *d_dst = nppiMalloc_8u_C1(expandedWidth, expandedHeight, &dstStep);
  ASSERT_NE(d_src, nullptr);
  ASSERT_NE(d_dst, nullptr);

  cudaMemcpy2D(d_src, srcStep, expandedInput.data(), expandedWidth, expandedWidth, expandedHeight,
               cudaMemcpyHostToDevice);

  NppiSize oSizeROI = {expandedWidth, expandedHeight};
  NppiSize oMaskSize = {maskW, maskH};
  NppiPoint oAnchor = {anchorX, anchorY};
  NppStreamContext ctx;
  ctx.hStream = 0;

  NppStatus status = nppiFilterMin_8u_C1R_Ctx(d_src, srcStep, d_dst, dstStep, oSizeROI, oMaskSize, oAnchor, ctx);
  EXPECT_EQ(status, NPP_SUCCESS);

  std::vector<Npp8u> expandedOutput(expandedWidth * expandedHeight);
  cudaMemcpy2D(expandedOutput.data(), expandedWidth, d_dst, dstStep, expandedWidth, expandedHeight,
               cudaMemcpyDeviceToHost);

  std::vector<Npp8u> result(width * height);
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      result[y * width + x] = expandedOutput[(y + padTop) * expandedWidth + (x + padLeft)];
    }
  }

  auto expected = computeMinFilterCPU<Npp8u>(h_src, width, height, 1, maskW, maskH, anchorX, anchorY);
  for (int i = 0; i < width * height; i++) {
    EXPECT_EQ(result[i], expected[i]) << "Mismatch at index " << i;
  }

  nppiFree(d_src);
  nppiFree(d_dst);
}

// ============================================================================
// FilterMax 16u_C1R Tests
// ============================================================================

TEST_F(FilterMinMaxTest, FilterMax_16u_C1R_Basic) {
  const int width = 16;
  const int height = 16;
  int maskW = 3, maskH = 3, anchorX = 1, anchorY = 1;
  int padLeft = anchorX, padRight = maskW - anchorX - 1;
  int padTop = anchorY, padBottom = maskH - anchorY - 1;
  int expandedWidth = width + padLeft + padRight;
  int expandedHeight = height + padTop + padBottom;

  std::vector<Npp16u> h_src(width * height);
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      h_src[y * width + x] = static_cast<Npp16u>((x + y * 256) % 65536);
    }
  }

  std::vector<Npp16u> expandedInput(expandedWidth * expandedHeight);
  for (int y = 0; y < expandedHeight; y++) {
    for (int x = 0; x < expandedWidth; x++) {
      int srcX = std::max(0, std::min(width - 1, x - padLeft));
      int srcY = std::max(0, std::min(height - 1, y - padTop));
      expandedInput[y * expandedWidth + x] = h_src[srcY * width + srcX];
    }
  }

  int srcStep, dstStep;
  Npp16u *d_src = nppiMalloc_16u_C1(expandedWidth, expandedHeight, &srcStep);
  Npp16u *d_dst = nppiMalloc_16u_C1(expandedWidth, expandedHeight, &dstStep);
  ASSERT_NE(d_src, nullptr);
  ASSERT_NE(d_dst, nullptr);

  cudaMemcpy2D(d_src, srcStep, expandedInput.data(), expandedWidth * sizeof(Npp16u), expandedWidth * sizeof(Npp16u),
               expandedHeight, cudaMemcpyHostToDevice);

  NppiSize oSizeROI = {expandedWidth, expandedHeight};
  NppiSize oMaskSize = {maskW, maskH};
  NppiPoint oAnchor = {anchorX, anchorY};

  NppStatus status = nppiFilterMax_16u_C1R(d_src, srcStep, d_dst, dstStep, oSizeROI, oMaskSize, oAnchor);
  EXPECT_EQ(status, NPP_SUCCESS);

  std::vector<Npp16u> expandedOutput(expandedWidth * expandedHeight);
  cudaMemcpy2D(expandedOutput.data(), expandedWidth * sizeof(Npp16u), d_dst, dstStep, expandedWidth * sizeof(Npp16u),
               expandedHeight, cudaMemcpyDeviceToHost);

  std::vector<Npp16u> result(width * height);
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      result[y * width + x] = expandedOutput[(y + padTop) * expandedWidth + (x + padLeft)];
    }
  }

  auto expected = computeMaxFilterCPU<Npp16u>(h_src, width, height, 1, maskW, maskH, anchorX, anchorY);
  for (int i = 0; i < width * height; i++) {
    EXPECT_EQ(result[i], expected[i]) << "Mismatch at index " << i;
  }

  nppiFree(d_src);
  nppiFree(d_dst);
}

// ============================================================================
// FilterMin 16u_C1R Tests
// ============================================================================

TEST_F(FilterMinMaxTest, FilterMin_16u_C1R_Basic) {
  const int width = 16;
  const int height = 16;
  int maskW = 3, maskH = 3, anchorX = 1, anchorY = 1;
  int padLeft = anchorX, padRight = maskW - anchorX - 1;
  int padTop = anchorY, padBottom = maskH - anchorY - 1;
  int expandedWidth = width + padLeft + padRight;
  int expandedHeight = height + padTop + padBottom;

  std::vector<Npp16u> h_src(width * height);
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      h_src[y * width + x] = static_cast<Npp16u>((x + y * 256) % 65536);
    }
  }

  std::vector<Npp16u> expandedInput(expandedWidth * expandedHeight);
  for (int y = 0; y < expandedHeight; y++) {
    for (int x = 0; x < expandedWidth; x++) {
      int srcX = std::max(0, std::min(width - 1, x - padLeft));
      int srcY = std::max(0, std::min(height - 1, y - padTop));
      expandedInput[y * expandedWidth + x] = h_src[srcY * width + srcX];
    }
  }

  int srcStep, dstStep;
  Npp16u *d_src = nppiMalloc_16u_C1(expandedWidth, expandedHeight, &srcStep);
  Npp16u *d_dst = nppiMalloc_16u_C1(expandedWidth, expandedHeight, &dstStep);
  ASSERT_NE(d_src, nullptr);
  ASSERT_NE(d_dst, nullptr);

  cudaMemcpy2D(d_src, srcStep, expandedInput.data(), expandedWidth * sizeof(Npp16u), expandedWidth * sizeof(Npp16u),
               expandedHeight, cudaMemcpyHostToDevice);

  NppiSize oSizeROI = {expandedWidth, expandedHeight};
  NppiSize oMaskSize = {maskW, maskH};
  NppiPoint oAnchor = {anchorX, anchorY};

  NppStatus status = nppiFilterMin_16u_C1R(d_src, srcStep, d_dst, dstStep, oSizeROI, oMaskSize, oAnchor);
  EXPECT_EQ(status, NPP_SUCCESS);

  std::vector<Npp16u> expandedOutput(expandedWidth * expandedHeight);
  cudaMemcpy2D(expandedOutput.data(), expandedWidth * sizeof(Npp16u), d_dst, dstStep, expandedWidth * sizeof(Npp16u),
               expandedHeight, cudaMemcpyDeviceToHost);

  std::vector<Npp16u> result(width * height);
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      result[y * width + x] = expandedOutput[(y + padTop) * expandedWidth + (x + padLeft)];
    }
  }

  auto expected = computeMinFilterCPU<Npp16u>(h_src, width, height, 1, maskW, maskH, anchorX, anchorY);
  for (int i = 0; i < width * height; i++) {
    EXPECT_EQ(result[i], expected[i]) << "Mismatch at index " << i;
  }

  nppiFree(d_src);
  nppiFree(d_dst);
}

// ============================================================================
// FilterMax 32f_C1R Tests
// ============================================================================

TEST_F(FilterMinMaxTest, FilterMax_32f_C1R_Basic) {
  const int width = 16;
  const int height = 16;
  int maskW = 3, maskH = 3, anchorX = 1, anchorY = 1;
  int padLeft = anchorX, padRight = maskW - anchorX - 1;
  int padTop = anchorY, padBottom = maskH - anchorY - 1;
  int expandedWidth = width + padLeft + padRight;
  int expandedHeight = height + padTop + padBottom;

  std::vector<Npp32f> h_src(width * height);
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      h_src[y * width + x] = static_cast<Npp32f>(x + y * 0.1f);
    }
  }

  std::vector<Npp32f> expandedInput(expandedWidth * expandedHeight);
  for (int y = 0; y < expandedHeight; y++) {
    for (int x = 0; x < expandedWidth; x++) {
      int srcX = std::max(0, std::min(width - 1, x - padLeft));
      int srcY = std::max(0, std::min(height - 1, y - padTop));
      expandedInput[y * expandedWidth + x] = h_src[srcY * width + srcX];
    }
  }

  int srcStep, dstStep;
  Npp32f *d_src = nppiMalloc_32f_C1(expandedWidth, expandedHeight, &srcStep);
  Npp32f *d_dst = nppiMalloc_32f_C1(expandedWidth, expandedHeight, &dstStep);
  ASSERT_NE(d_src, nullptr);
  ASSERT_NE(d_dst, nullptr);

  cudaMemcpy2D(d_src, srcStep, expandedInput.data(), expandedWidth * sizeof(Npp32f), expandedWidth * sizeof(Npp32f),
               expandedHeight, cudaMemcpyHostToDevice);

  NppiSize oSizeROI = {expandedWidth, expandedHeight};
  NppiSize oMaskSize = {maskW, maskH};
  NppiPoint oAnchor = {anchorX, anchorY};

  NppStatus status = nppiFilterMax_32f_C1R(d_src, srcStep, d_dst, dstStep, oSizeROI, oMaskSize, oAnchor);
  EXPECT_EQ(status, NPP_SUCCESS);

  std::vector<Npp32f> expandedOutput(expandedWidth * expandedHeight);
  cudaMemcpy2D(expandedOutput.data(), expandedWidth * sizeof(Npp32f), d_dst, dstStep, expandedWidth * sizeof(Npp32f),
               expandedHeight, cudaMemcpyDeviceToHost);

  std::vector<Npp32f> result(width * height);
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      result[y * width + x] = expandedOutput[(y + padTop) * expandedWidth + (x + padLeft)];
    }
  }

  auto expected = computeMaxFilterCPU<Npp32f>(h_src, width, height, 1, maskW, maskH, anchorX, anchorY);
  for (int i = 0; i < width * height; i++) {
    EXPECT_NEAR(result[i], expected[i], 1e-5f) << "Mismatch at index " << i;
  }

  nppiFree(d_src);
  nppiFree(d_dst);
}

// ============================================================================
// FilterMin 32f_C1R Tests
// ============================================================================

TEST_F(FilterMinMaxTest, FilterMin_32f_C1R_Basic) {
  const int width = 16;
  const int height = 16;
  int maskW = 3, maskH = 3, anchorX = 1, anchorY = 1;
  int padLeft = anchorX, padRight = maskW - anchorX - 1;
  int padTop = anchorY, padBottom = maskH - anchorY - 1;
  int expandedWidth = width + padLeft + padRight;
  int expandedHeight = height + padTop + padBottom;

  std::vector<Npp32f> h_src(width * height);
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      h_src[y * width + x] = static_cast<Npp32f>(x + y * 0.1f);
    }
  }

  std::vector<Npp32f> expandedInput(expandedWidth * expandedHeight);
  for (int y = 0; y < expandedHeight; y++) {
    for (int x = 0; x < expandedWidth; x++) {
      int srcX = std::max(0, std::min(width - 1, x - padLeft));
      int srcY = std::max(0, std::min(height - 1, y - padTop));
      expandedInput[y * expandedWidth + x] = h_src[srcY * width + srcX];
    }
  }

  int srcStep, dstStep;
  Npp32f *d_src = nppiMalloc_32f_C1(expandedWidth, expandedHeight, &srcStep);
  Npp32f *d_dst = nppiMalloc_32f_C1(expandedWidth, expandedHeight, &dstStep);
  ASSERT_NE(d_src, nullptr);
  ASSERT_NE(d_dst, nullptr);

  cudaMemcpy2D(d_src, srcStep, expandedInput.data(), expandedWidth * sizeof(Npp32f), expandedWidth * sizeof(Npp32f),
               expandedHeight, cudaMemcpyHostToDevice);

  NppiSize oSizeROI = {expandedWidth, expandedHeight};
  NppiSize oMaskSize = {maskW, maskH};
  NppiPoint oAnchor = {anchorX, anchorY};

  NppStatus status = nppiFilterMin_32f_C1R(d_src, srcStep, d_dst, dstStep, oSizeROI, oMaskSize, oAnchor);
  EXPECT_EQ(status, NPP_SUCCESS);

  std::vector<Npp32f> expandedOutput(expandedWidth * expandedHeight);
  cudaMemcpy2D(expandedOutput.data(), expandedWidth * sizeof(Npp32f), d_dst, dstStep, expandedWidth * sizeof(Npp32f),
               expandedHeight, cudaMemcpyDeviceToHost);

  std::vector<Npp32f> result(width * height);
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      result[y * width + x] = expandedOutput[(y + padTop) * expandedWidth + (x + padLeft)];
    }
  }

  auto expected = computeMinFilterCPU<Npp32f>(h_src, width, height, 1, maskW, maskH, anchorX, anchorY);
  for (int i = 0; i < width * height; i++) {
    EXPECT_NEAR(result[i], expected[i], 1e-5f) << "Mismatch at index " << i;
  }

  nppiFree(d_src);
  nppiFree(d_dst);
}

// ============================================================================
// Multi-channel tests (C3R, C4R)
// ============================================================================

TEST_F(FilterMinMaxTest, FilterMax_8u_C3R_Basic) {
  const int width = 16;
  const int height = 16;
  const int channels = 3;
  int maskW = 3, maskH = 3, anchorX = 1, anchorY = 1;
  int padLeft = anchorX, padRight = maskW - anchorX - 1;
  int padTop = anchorY, padBottom = maskH - anchorY - 1;
  int expandedWidth = width + padLeft + padRight;
  int expandedHeight = height + padTop + padBottom;

  std::vector<Npp8u> h_src(width * height * channels);
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      int idx = (y * width + x) * channels;
      h_src[idx + 0] = static_cast<Npp8u>((x * 10) % 256);
      h_src[idx + 1] = static_cast<Npp8u>((y * 10) % 256);
      h_src[idx + 2] = static_cast<Npp8u>((x + y) % 256);
    }
  }

  std::vector<Npp8u> expandedInput(expandedWidth * expandedHeight * channels);
  for (int y = 0; y < expandedHeight; y++) {
    for (int x = 0; x < expandedWidth; x++) {
      int srcX = std::max(0, std::min(width - 1, x - padLeft));
      int srcY = std::max(0, std::min(height - 1, y - padTop));
      for (int c = 0; c < channels; c++) {
        expandedInput[(y * expandedWidth + x) * channels + c] = h_src[(srcY * width + srcX) * channels + c];
      }
    }
  }

  int srcStep, dstStep;
  Npp8u *d_src = nppiMalloc_8u_C3(expandedWidth, expandedHeight, &srcStep);
  Npp8u *d_dst = nppiMalloc_8u_C3(expandedWidth, expandedHeight, &dstStep);
  ASSERT_NE(d_src, nullptr);
  ASSERT_NE(d_dst, nullptr);

  cudaMemcpy2D(d_src, srcStep, expandedInput.data(), expandedWidth * channels, expandedWidth * channels, expandedHeight,
               cudaMemcpyHostToDevice);

  NppiSize oSizeROI = {expandedWidth, expandedHeight};
  NppiSize oMaskSize = {maskW, maskH};
  NppiPoint oAnchor = {anchorX, anchorY};

  NppStatus status = nppiFilterMax_8u_C3R(d_src, srcStep, d_dst, dstStep, oSizeROI, oMaskSize, oAnchor);
  EXPECT_EQ(status, NPP_SUCCESS);

  std::vector<Npp8u> expandedOutput(expandedWidth * expandedHeight * channels);
  cudaMemcpy2D(expandedOutput.data(), expandedWidth * channels, d_dst, dstStep, expandedWidth * channels,
               expandedHeight, cudaMemcpyDeviceToHost);

  std::vector<Npp8u> result(width * height * channels);
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      for (int c = 0; c < channels; c++) {
        result[(y * width + x) * channels + c] =
            expandedOutput[((y + padTop) * expandedWidth + (x + padLeft)) * channels + c];
      }
    }
  }

  auto expected = computeMaxFilterCPU<Npp8u>(h_src, width, height, channels, maskW, maskH, anchorX, anchorY);
  for (int i = 0; i < width * height * channels; i++) {
    EXPECT_EQ(result[i], expected[i]) << "Mismatch at index " << i;
  }

  nppiFree(d_src);
  nppiFree(d_dst);
}

TEST_F(FilterMinMaxTest, FilterMin_8u_C3R_Basic) {
  const int width = 16;
  const int height = 16;
  const int channels = 3;
  int maskW = 3, maskH = 3, anchorX = 1, anchorY = 1;
  int padLeft = anchorX, padRight = maskW - anchorX - 1;
  int padTop = anchorY, padBottom = maskH - anchorY - 1;
  int expandedWidth = width + padLeft + padRight;
  int expandedHeight = height + padTop + padBottom;

  std::vector<Npp8u> h_src(width * height * channels);
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      int idx = (y * width + x) * channels;
      h_src[idx + 0] = static_cast<Npp8u>((x * 10) % 256);
      h_src[idx + 1] = static_cast<Npp8u>((y * 10) % 256);
      h_src[idx + 2] = static_cast<Npp8u>((x + y) % 256);
    }
  }

  std::vector<Npp8u> expandedInput(expandedWidth * expandedHeight * channels);
  for (int y = 0; y < expandedHeight; y++) {
    for (int x = 0; x < expandedWidth; x++) {
      int srcX = std::max(0, std::min(width - 1, x - padLeft));
      int srcY = std::max(0, std::min(height - 1, y - padTop));
      for (int c = 0; c < channels; c++) {
        expandedInput[(y * expandedWidth + x) * channels + c] = h_src[(srcY * width + srcX) * channels + c];
      }
    }
  }

  int srcStep, dstStep;
  Npp8u *d_src = nppiMalloc_8u_C3(expandedWidth, expandedHeight, &srcStep);
  Npp8u *d_dst = nppiMalloc_8u_C3(expandedWidth, expandedHeight, &dstStep);
  ASSERT_NE(d_src, nullptr);
  ASSERT_NE(d_dst, nullptr);

  cudaMemcpy2D(d_src, srcStep, expandedInput.data(), expandedWidth * channels, expandedWidth * channels, expandedHeight,
               cudaMemcpyHostToDevice);

  NppiSize oSizeROI = {expandedWidth, expandedHeight};
  NppiSize oMaskSize = {maskW, maskH};
  NppiPoint oAnchor = {anchorX, anchorY};

  NppStatus status = nppiFilterMin_8u_C3R(d_src, srcStep, d_dst, dstStep, oSizeROI, oMaskSize, oAnchor);
  EXPECT_EQ(status, NPP_SUCCESS);

  std::vector<Npp8u> expandedOutput(expandedWidth * expandedHeight * channels);
  cudaMemcpy2D(expandedOutput.data(), expandedWidth * channels, d_dst, dstStep, expandedWidth * channels,
               expandedHeight, cudaMemcpyDeviceToHost);

  std::vector<Npp8u> result(width * height * channels);
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      for (int c = 0; c < channels; c++) {
        result[(y * width + x) * channels + c] =
            expandedOutput[((y + padTop) * expandedWidth + (x + padLeft)) * channels + c];
      }
    }
  }

  auto expected = computeMinFilterCPU<Npp8u>(h_src, width, height, channels, maskW, maskH, anchorX, anchorY);
  for (int i = 0; i < width * height * channels; i++) {
    EXPECT_EQ(result[i], expected[i]) << "Mismatch at index " << i;
  }

  nppiFree(d_src);
  nppiFree(d_dst);
}

TEST_F(FilterMinMaxTest, FilterMax_8u_C4R_Basic) {
  const int width = 16;
  const int height = 16;
  const int channels = 4;
  int maskW = 3, maskH = 3, anchorX = 1, anchorY = 1;
  int padLeft = anchorX, padRight = maskW - anchorX - 1;
  int padTop = anchorY, padBottom = maskH - anchorY - 1;
  int expandedWidth = width + padLeft + padRight;
  int expandedHeight = height + padTop + padBottom;

  std::vector<Npp8u> h_src(width * height * channels);
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      int idx = (y * width + x) * channels;
      h_src[idx + 0] = static_cast<Npp8u>((x * 10) % 256);
      h_src[idx + 1] = static_cast<Npp8u>((y * 10) % 256);
      h_src[idx + 2] = static_cast<Npp8u>((x + y) % 256);
      h_src[idx + 3] = static_cast<Npp8u>(255);
    }
  }

  std::vector<Npp8u> expandedInput(expandedWidth * expandedHeight * channels);
  for (int y = 0; y < expandedHeight; y++) {
    for (int x = 0; x < expandedWidth; x++) {
      int srcX = std::max(0, std::min(width - 1, x - padLeft));
      int srcY = std::max(0, std::min(height - 1, y - padTop));
      for (int c = 0; c < channels; c++) {
        expandedInput[(y * expandedWidth + x) * channels + c] = h_src[(srcY * width + srcX) * channels + c];
      }
    }
  }

  int srcStep, dstStep;
  Npp8u *d_src = nppiMalloc_8u_C4(expandedWidth, expandedHeight, &srcStep);
  Npp8u *d_dst = nppiMalloc_8u_C4(expandedWidth, expandedHeight, &dstStep);
  ASSERT_NE(d_src, nullptr);
  ASSERT_NE(d_dst, nullptr);

  cudaMemcpy2D(d_src, srcStep, expandedInput.data(), expandedWidth * channels, expandedWidth * channels, expandedHeight,
               cudaMemcpyHostToDevice);

  NppiSize oSizeROI = {expandedWidth, expandedHeight};
  NppiSize oMaskSize = {maskW, maskH};
  NppiPoint oAnchor = {anchorX, anchorY};

  NppStatus status = nppiFilterMax_8u_C4R(d_src, srcStep, d_dst, dstStep, oSizeROI, oMaskSize, oAnchor);
  EXPECT_EQ(status, NPP_SUCCESS);

  std::vector<Npp8u> expandedOutput(expandedWidth * expandedHeight * channels);
  cudaMemcpy2D(expandedOutput.data(), expandedWidth * channels, d_dst, dstStep, expandedWidth * channels,
               expandedHeight, cudaMemcpyDeviceToHost);

  std::vector<Npp8u> result(width * height * channels);
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      for (int c = 0; c < channels; c++) {
        result[(y * width + x) * channels + c] =
            expandedOutput[((y + padTop) * expandedWidth + (x + padLeft)) * channels + c];
      }
    }
  }

  auto expected = computeMaxFilterCPU<Npp8u>(h_src, width, height, channels, maskW, maskH, anchorX, anchorY);
  for (int i = 0; i < width * height * channels; i++) {
    EXPECT_EQ(result[i], expected[i]) << "Mismatch at index " << i;
  }

  nppiFree(d_src);
  nppiFree(d_dst);
}

TEST_F(FilterMinMaxTest, FilterMin_8u_C4R_Basic) {
  const int width = 16;
  const int height = 16;
  const int channels = 4;
  int maskW = 3, maskH = 3, anchorX = 1, anchorY = 1;
  int padLeft = anchorX, padRight = maskW - anchorX - 1;
  int padTop = anchorY, padBottom = maskH - anchorY - 1;
  int expandedWidth = width + padLeft + padRight;
  int expandedHeight = height + padTop + padBottom;

  std::vector<Npp8u> h_src(width * height * channels);
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      int idx = (y * width + x) * channels;
      h_src[idx + 0] = static_cast<Npp8u>((x * 10) % 256);
      h_src[idx + 1] = static_cast<Npp8u>((y * 10) % 256);
      h_src[idx + 2] = static_cast<Npp8u>((x + y) % 256);
      h_src[idx + 3] = static_cast<Npp8u>(255);
    }
  }

  std::vector<Npp8u> expandedInput(expandedWidth * expandedHeight * channels);
  for (int y = 0; y < expandedHeight; y++) {
    for (int x = 0; x < expandedWidth; x++) {
      int srcX = std::max(0, std::min(width - 1, x - padLeft));
      int srcY = std::max(0, std::min(height - 1, y - padTop));
      for (int c = 0; c < channels; c++) {
        expandedInput[(y * expandedWidth + x) * channels + c] = h_src[(srcY * width + srcX) * channels + c];
      }
    }
  }

  int srcStep, dstStep;
  Npp8u *d_src = nppiMalloc_8u_C4(expandedWidth, expandedHeight, &srcStep);
  Npp8u *d_dst = nppiMalloc_8u_C4(expandedWidth, expandedHeight, &dstStep);
  ASSERT_NE(d_src, nullptr);
  ASSERT_NE(d_dst, nullptr);

  cudaMemcpy2D(d_src, srcStep, expandedInput.data(), expandedWidth * channels, expandedWidth * channels, expandedHeight,
               cudaMemcpyHostToDevice);

  NppiSize oSizeROI = {expandedWidth, expandedHeight};
  NppiSize oMaskSize = {maskW, maskH};
  NppiPoint oAnchor = {anchorX, anchorY};

  NppStatus status = nppiFilterMin_8u_C4R(d_src, srcStep, d_dst, dstStep, oSizeROI, oMaskSize, oAnchor);
  EXPECT_EQ(status, NPP_SUCCESS);

  std::vector<Npp8u> expandedOutput(expandedWidth * expandedHeight * channels);
  cudaMemcpy2D(expandedOutput.data(), expandedWidth * channels, d_dst, dstStep, expandedWidth * channels,
               expandedHeight, cudaMemcpyDeviceToHost);

  std::vector<Npp8u> result(width * height * channels);
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      for (int c = 0; c < channels; c++) {
        result[(y * width + x) * channels + c] =
            expandedOutput[((y + padTop) * expandedWidth + (x + padLeft)) * channels + c];
      }
    }
  }

  auto expected = computeMinFilterCPU<Npp8u>(h_src, width, height, channels, maskW, maskH, anchorX, anchorY);
  for (int i = 0; i < width * height * channels; i++) {
    EXPECT_EQ(result[i], expected[i]) << "Mismatch at index " << i;
  }

  nppiFree(d_src);
  nppiFree(d_dst);
}

// ============================================================================
// Edge cases
// ============================================================================

TEST_F(FilterMinMaxTest, FilterMax_8u_C1R_AsymmetricMask) {
  const int width = 16;
  const int height = 16;

  std::vector<Npp8u> h_src(width * height);
  std::mt19937 rng(123);
  for (int i = 0; i < width * height; i++) {
    h_src[i] = static_cast<Npp8u>(rng() % 256);
  }

  auto result = runFilterMax8uC1R(h_src, width, height, 5, 3, 2, 1);
  auto expected = computeMaxFilterCPU<Npp8u>(h_src, width, height, 1, 5, 3, 2, 1);

  for (int i = 0; i < width * height; i++) {
    EXPECT_EQ(result[i], expected[i]) << "Mismatch at index " << i;
  }
}

TEST_F(FilterMinMaxTest, FilterMin_8u_C1R_AsymmetricMask) {
  const int width = 16;
  const int height = 16;

  std::vector<Npp8u> h_src(width * height);
  std::mt19937 rng(123);
  for (int i = 0; i < width * height; i++) {
    h_src[i] = static_cast<Npp8u>(rng() % 256);
  }

  auto result = runFilterMin8uC1R(h_src, width, height, 3, 5, 1, 2);
  auto expected = computeMinFilterCPU<Npp8u>(h_src, width, height, 1, 3, 5, 1, 2);

  for (int i = 0; i < width * height; i++) {
    EXPECT_EQ(result[i], expected[i]) << "Mismatch at index " << i;
  }
}

TEST_F(FilterMinMaxTest, FilterMax_8u_C1R_LargeImage) {
  const int width = 256;
  const int height = 256;

  std::vector<Npp8u> h_src(width * height);
  std::mt19937 rng(456);
  for (int i = 0; i < width * height; i++) {
    h_src[i] = static_cast<Npp8u>(rng() % 256);
  }

  auto result = runFilterMax8uC1R(h_src, width, height, 5, 5, 2, 2);
  auto expected = computeMaxFilterCPU<Npp8u>(h_src, width, height, 1, 5, 5, 2, 2);

  // Sample verification for large image
  int errorCount = 0;
  for (int i = 0; i < width * height; i++) {
    if (result[i] != expected[i]) {
      errorCount++;
    }
  }

  EXPECT_EQ(errorCount, 0) << "Found " << errorCount << " mismatches";
}
