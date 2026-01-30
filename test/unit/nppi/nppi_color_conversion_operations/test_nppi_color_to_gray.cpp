#include "npp.h"
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <vector>

class ColorToGrayTest : public ::testing::Test {
protected:
  void SetUp() override {
    width = 12;
    height = 8;
    roi.width = width;
    roi.height = height;
  }

  int width = 0;
  int height = 0;
  NppiSize roi{};
};

TEST_F(ColorToGrayTest, ColorToGray_8u_C3C1R_IdentityCoeff) {
  const Npp32f coeffs[3] = {1.0f, 0.0f, 0.0f};
  std::vector<Npp8u> srcData(width * height * 3);
  std::vector<Npp8u> expected(width * height);

  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      int idx = (y * width + x) * 3;
      Npp8u r = static_cast<Npp8u>(((x + y) * 4) & 0xFF);
      Npp8u g = static_cast<Npp8u>(((x * 3 + y) * 7) & 0xFF);
      Npp8u b = static_cast<Npp8u>(((x * 5 + y * 2) * 3) & 0xFF);
      srcData[idx + 0] = r;
      srcData[idx + 1] = g;
      srcData[idx + 2] = b;
      expected[y * width + x] = r;
    }
  }

  int srcStep = 0;
  int dstStep = 0;
  Npp8u *d_src = nppiMalloc_8u_C3(width, height, &srcStep);
  Npp8u *d_dst = nppiMalloc_8u_C1(width, height, &dstStep);
  ASSERT_NE(d_src, nullptr);
  ASSERT_NE(d_dst, nullptr);

  cudaMemcpy2D(d_src, srcStep, srcData.data(), width * 3, width * 3, height, cudaMemcpyHostToDevice);

  NppStatus status = nppiColorToGray_8u_C3C1R(d_src, srcStep, d_dst, dstStep, roi, coeffs);
  EXPECT_EQ(status, NPP_SUCCESS);

  std::vector<Npp8u> result(width * height);
  cudaMemcpy2D(result.data(), width, d_dst, dstStep, width, height, cudaMemcpyDeviceToHost);

  for (int i = 0; i < width * height; ++i) {
    EXPECT_EQ(result[i], expected[i]) << "Mismatch at pixel " << i;
  }

  nppiFree(d_src);
  nppiFree(d_dst);
}

TEST_F(ColorToGrayTest, ColorToGray_8u_AC4C1R_WeightedNoAlpha) {
  const Npp32f coeffs[3] = {0.25f, 0.5f, 0.25f};
  std::vector<Npp8u> srcData(width * height * 4);
  std::vector<Npp8u> expected(width * height);

  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      int idx = (y * width + x) * 4;
      Npp8u r = static_cast<Npp8u>(((x + y) * 4) & 0xFF);
      Npp8u g = static_cast<Npp8u>(((x * 2 + y) * 4) & 0xFF);
      Npp8u b = static_cast<Npp8u>(((x + y * 2) * 4) & 0xFF);
      Npp8u a = static_cast<Npp8u>((x * 17 + y * 13) & 0xFF);
      srcData[idx + 0] = r;
      srcData[idx + 1] = g;
      srcData[idx + 2] = b;
      srcData[idx + 3] = a;
      expected[y * width + x] = static_cast<Npp8u>((r + 2 * g + b) / 4);
    }
  }

  int srcStep = 0;
  int dstStep = 0;
  Npp8u *d_src = nppiMalloc_8u_C4(width, height, &srcStep);
  Npp8u *d_dst = nppiMalloc_8u_C1(width, height, &dstStep);
  ASSERT_NE(d_src, nullptr);
  ASSERT_NE(d_dst, nullptr);

  cudaMemcpy2D(d_src, srcStep, srcData.data(), width * 4, width * 4, height, cudaMemcpyHostToDevice);

  NppStatus status = nppiColorToGray_8u_AC4C1R(d_src, srcStep, d_dst, dstStep, roi, coeffs);
  EXPECT_EQ(status, NPP_SUCCESS);

  std::vector<Npp8u> result(width * height);
  cudaMemcpy2D(result.data(), width, d_dst, dstStep, width, height, cudaMemcpyDeviceToHost);

  for (int i = 0; i < width * height; ++i) {
    EXPECT_EQ(result[i], expected[i]) << "Mismatch at pixel " << i;
  }

  nppiFree(d_src);
  nppiFree(d_dst);
}

TEST_F(ColorToGrayTest, ColorToGray_8u_C4C1R_UsesAlphaCoeff) {
  const Npp32f coeffs[4] = {0.25f, 0.25f, 0.25f, 0.25f};
  std::vector<Npp8u> srcData(width * height * 4);
  std::vector<Npp8u> expected(width * height);

  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      int idx = (y * width + x) * 4;
      Npp8u r = static_cast<Npp8u>(((x + y) * 4) & 0xFF);
      Npp8u g = static_cast<Npp8u>(((x * 3 + y) * 4) & 0xFF);
      Npp8u b = static_cast<Npp8u>(((x + y * 2) * 4) & 0xFF);
      Npp8u a = static_cast<Npp8u>(((x * 2 + y * 3) * 4) & 0xFF);
      srcData[idx + 0] = r;
      srcData[idx + 1] = g;
      srcData[idx + 2] = b;
      srcData[idx + 3] = a;
      expected[y * width + x] = static_cast<Npp8u>((r + g + b + a) / 4);
    }
  }

  int srcStep = 0;
  int dstStep = 0;
  Npp8u *d_src = nppiMalloc_8u_C4(width, height, &srcStep);
  Npp8u *d_dst = nppiMalloc_8u_C1(width, height, &dstStep);
  ASSERT_NE(d_src, nullptr);
  ASSERT_NE(d_dst, nullptr);

  cudaMemcpy2D(d_src, srcStep, srcData.data(), width * 4, width * 4, height, cudaMemcpyHostToDevice);

  NppStatus status = nppiColorToGray_8u_C4C1R(d_src, srcStep, d_dst, dstStep, roi, coeffs);
  EXPECT_EQ(status, NPP_SUCCESS);

  std::vector<Npp8u> result(width * height);
  cudaMemcpy2D(result.data(), width, d_dst, dstStep, width, height, cudaMemcpyDeviceToHost);

  for (int i = 0; i < width * height; ++i) {
    EXPECT_EQ(result[i], expected[i]) << "Mismatch at pixel " << i;
  }

  nppiFree(d_src);
  nppiFree(d_dst);
}
