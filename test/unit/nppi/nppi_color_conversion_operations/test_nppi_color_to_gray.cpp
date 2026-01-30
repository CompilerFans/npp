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

TEST_F(ColorToGrayTest, ColorToGray_16u_C3C1R_IdentityCoeff) {
  const Npp32f coeffs[3] = {1.0f, 0.0f, 0.0f};
  std::vector<Npp16u> srcData(width * height * 3);
  std::vector<Npp16u> expected(width * height);

  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      int idx = (y * width + x) * 3;
      Npp16u r = static_cast<Npp16u>(((x + y) * 64) & 0xFFFF);
      Npp16u g = static_cast<Npp16u>(((x * 3 + y) * 32) & 0xFFFF);
      Npp16u b = static_cast<Npp16u>(((x + y * 2) * 48) & 0xFFFF);
      srcData[idx + 0] = r;
      srcData[idx + 1] = g;
      srcData[idx + 2] = b;
      expected[y * width + x] = r;
    }
  }

  int srcStep = 0;
  int dstStep = 0;
  Npp16u *d_src = nppiMalloc_16u_C3(width, height, &srcStep);
  Npp16u *d_dst = nppiMalloc_16u_C1(width, height, &dstStep);
  ASSERT_NE(d_src, nullptr);
  ASSERT_NE(d_dst, nullptr);

  cudaMemcpy2D(d_src, srcStep, srcData.data(), width * sizeof(Npp16u) * 3, width * sizeof(Npp16u) * 3, height,
               cudaMemcpyHostToDevice);

  NppStatus status = nppiColorToGray_16u_C3C1R(d_src, srcStep, d_dst, dstStep, roi, coeffs);
  EXPECT_EQ(status, NPP_SUCCESS);

  std::vector<Npp16u> result(width * height);
  cudaMemcpy2D(result.data(), width * sizeof(Npp16u), d_dst, dstStep, width * sizeof(Npp16u), height,
               cudaMemcpyDeviceToHost);

  for (int i = 0; i < width * height; ++i) {
    EXPECT_EQ(result[i], expected[i]) << "Mismatch at pixel " << i;
  }

  nppiFree(d_src);
  nppiFree(d_dst);
}

TEST_F(ColorToGrayTest, ColorToGray_16u_AC4C1R_WeightedNoAlpha) {
  const Npp32f coeffs[3] = {0.25f, 0.5f, 0.25f};
  std::vector<Npp16u> srcData(width * height * 4);
  std::vector<Npp16u> expected(width * height);

  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      int idx = (y * width + x) * 4;
      Npp16u r = static_cast<Npp16u>(((x + y) * 64) & 0xFFFF);
      Npp16u g = static_cast<Npp16u>(((x * 2 + y) * 64) & 0xFFFF);
      Npp16u b = static_cast<Npp16u>(((x + y * 2) * 64) & 0xFFFF);
      Npp16u a = static_cast<Npp16u>(((x * 5 + y * 7) * 64) & 0xFFFF);
      srcData[idx + 0] = r;
      srcData[idx + 1] = g;
      srcData[idx + 2] = b;
      srcData[idx + 3] = a;
      expected[y * width + x] = static_cast<Npp16u>((r + 2 * g + b) / 4);
    }
  }

  int srcStep = 0;
  int dstStep = 0;
  Npp16u *d_src = nppiMalloc_16u_C4(width, height, &srcStep);
  Npp16u *d_dst = nppiMalloc_16u_C1(width, height, &dstStep);
  ASSERT_NE(d_src, nullptr);
  ASSERT_NE(d_dst, nullptr);

  cudaMemcpy2D(d_src, srcStep, srcData.data(), width * sizeof(Npp16u) * 4, width * sizeof(Npp16u) * 4, height,
               cudaMemcpyHostToDevice);

  NppStatus status = nppiColorToGray_16u_AC4C1R(d_src, srcStep, d_dst, dstStep, roi, coeffs);
  EXPECT_EQ(status, NPP_SUCCESS);

  std::vector<Npp16u> result(width * height);
  cudaMemcpy2D(result.data(), width * sizeof(Npp16u), d_dst, dstStep, width * sizeof(Npp16u), height,
               cudaMemcpyDeviceToHost);

  for (int i = 0; i < width * height; ++i) {
    EXPECT_EQ(result[i], expected[i]) << "Mismatch at pixel " << i;
  }

  nppiFree(d_src);
  nppiFree(d_dst);
}

TEST_F(ColorToGrayTest, ColorToGray_16u_C4C1R_UsesAlphaCoeff) {
  const Npp32f coeffs[4] = {0.25f, 0.25f, 0.25f, 0.25f};
  std::vector<Npp16u> srcData(width * height * 4);
  std::vector<Npp16u> expected(width * height);

  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      int idx = (y * width + x) * 4;
      Npp16u r = static_cast<Npp16u>(((x + y) * 64) & 0xFFFF);
      Npp16u g = static_cast<Npp16u>(((x * 3 + y) * 64) & 0xFFFF);
      Npp16u b = static_cast<Npp16u>(((x + y * 2) * 64) & 0xFFFF);
      Npp16u a = static_cast<Npp16u>(((x * 2 + y * 3) * 64) & 0xFFFF);
      srcData[idx + 0] = r;
      srcData[idx + 1] = g;
      srcData[idx + 2] = b;
      srcData[idx + 3] = a;
      expected[y * width + x] = static_cast<Npp16u>((r + g + b + a) / 4);
    }
  }

  int srcStep = 0;
  int dstStep = 0;
  Npp16u *d_src = nppiMalloc_16u_C4(width, height, &srcStep);
  Npp16u *d_dst = nppiMalloc_16u_C1(width, height, &dstStep);
  ASSERT_NE(d_src, nullptr);
  ASSERT_NE(d_dst, nullptr);

  cudaMemcpy2D(d_src, srcStep, srcData.data(), width * sizeof(Npp16u) * 4, width * sizeof(Npp16u) * 4, height,
               cudaMemcpyHostToDevice);

  NppStatus status = nppiColorToGray_16u_C4C1R(d_src, srcStep, d_dst, dstStep, roi, coeffs);
  EXPECT_EQ(status, NPP_SUCCESS);

  std::vector<Npp16u> result(width * height);
  cudaMemcpy2D(result.data(), width * sizeof(Npp16u), d_dst, dstStep, width * sizeof(Npp16u), height,
               cudaMemcpyDeviceToHost);

  for (int i = 0; i < width * height; ++i) {
    EXPECT_EQ(result[i], expected[i]) << "Mismatch at pixel " << i;
  }

  nppiFree(d_src);
  nppiFree(d_dst);
}

TEST_F(ColorToGrayTest, ColorToGray_32f_C3C1R_IdentityCoeff) {
  const Npp32f coeffs[3] = {1.0f, 0.0f, 0.0f};
  std::vector<Npp32f> srcData(width * height * 3);
  std::vector<Npp32f> expected(width * height);

  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      int idx = (y * width + x) * 3;
      float r = static_cast<float>((x + y) * 1.25f);
      float g = static_cast<float>((x * 3 + y) * 0.5f);
      float b = static_cast<float>((x + y * 2) * 0.75f);
      srcData[idx + 0] = r;
      srcData[idx + 1] = g;
      srcData[idx + 2] = b;
      expected[y * width + x] = r;
    }
  }

  int srcStep = 0;
  int dstStep = 0;
  Npp32f *d_src = nppiMalloc_32f_C3(width, height, &srcStep);
  Npp32f *d_dst = nppiMalloc_32f_C1(width, height, &dstStep);
  ASSERT_NE(d_src, nullptr);
  ASSERT_NE(d_dst, nullptr);

  cudaMemcpy2D(d_src, srcStep, srcData.data(), width * sizeof(Npp32f) * 3, width * sizeof(Npp32f) * 3, height,
               cudaMemcpyHostToDevice);

  NppStatus status = nppiColorToGray_32f_C3C1R(d_src, srcStep, d_dst, dstStep, roi, coeffs);
  EXPECT_EQ(status, NPP_SUCCESS);

  std::vector<Npp32f> result(width * height);
  cudaMemcpy2D(result.data(), width * sizeof(Npp32f), d_dst, dstStep, width * sizeof(Npp32f), height,
               cudaMemcpyDeviceToHost);

  for (int i = 0; i < width * height; ++i) {
    EXPECT_NEAR(result[i], expected[i], 1e-5f) << "Mismatch at pixel " << i;
  }

  nppiFree(d_src);
  nppiFree(d_dst);
}

TEST_F(ColorToGrayTest, ColorToGray_32f_AC4C1R_WeightedNoAlpha) {
  const Npp32f coeffs[3] = {0.25f, 0.5f, 0.25f};
  std::vector<Npp32f> srcData(width * height * 4);
  std::vector<Npp32f> expected(width * height);

  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      int idx = (y * width + x) * 4;
      float r = static_cast<float>((x + y) * 1.0f);
      float g = static_cast<float>((x * 2 + y) * 1.0f);
      float b = static_cast<float>((x + y * 2) * 1.0f);
      float a = static_cast<float>((x * 3 + y * 4) * 1.0f);
      srcData[idx + 0] = r;
      srcData[idx + 1] = g;
      srcData[idx + 2] = b;
      srcData[idx + 3] = a;
      expected[y * width + x] = 0.25f * r + 0.5f * g + 0.25f * b;
    }
  }

  int srcStep = 0;
  int dstStep = 0;
  Npp32f *d_src = nppiMalloc_32f_C4(width, height, &srcStep);
  Npp32f *d_dst = nppiMalloc_32f_C1(width, height, &dstStep);
  ASSERT_NE(d_src, nullptr);
  ASSERT_NE(d_dst, nullptr);

  cudaMemcpy2D(d_src, srcStep, srcData.data(), width * sizeof(Npp32f) * 4, width * sizeof(Npp32f) * 4, height,
               cudaMemcpyHostToDevice);

  NppStatus status = nppiColorToGray_32f_AC4C1R(d_src, srcStep, d_dst, dstStep, roi, coeffs);
  EXPECT_EQ(status, NPP_SUCCESS);

  std::vector<Npp32f> result(width * height);
  cudaMemcpy2D(result.data(), width * sizeof(Npp32f), d_dst, dstStep, width * sizeof(Npp32f), height,
               cudaMemcpyDeviceToHost);

  for (int i = 0; i < width * height; ++i) {
    EXPECT_NEAR(result[i], expected[i], 1e-5f) << "Mismatch at pixel " << i;
  }

  nppiFree(d_src);
  nppiFree(d_dst);
}

TEST_F(ColorToGrayTest, ColorToGray_32f_C4C1R_UsesAlphaCoeff) {
  const Npp32f coeffs[4] = {0.25f, 0.25f, 0.25f, 0.25f};
  std::vector<Npp32f> srcData(width * height * 4);
  std::vector<Npp32f> expected(width * height);

  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      int idx = (y * width + x) * 4;
      float r = static_cast<float>((x + y) * 1.25f);
      float g = static_cast<float>((x * 3 + y) * 0.5f);
      float b = static_cast<float>((x + y * 2) * 0.75f);
      float a = static_cast<float>((x * 2 + y * 3) * 0.25f);
      srcData[idx + 0] = r;
      srcData[idx + 1] = g;
      srcData[idx + 2] = b;
      srcData[idx + 3] = a;
      expected[y * width + x] = 0.25f * (r + g + b + a);
    }
  }

  int srcStep = 0;
  int dstStep = 0;
  Npp32f *d_src = nppiMalloc_32f_C4(width, height, &srcStep);
  Npp32f *d_dst = nppiMalloc_32f_C1(width, height, &dstStep);
  ASSERT_NE(d_src, nullptr);
  ASSERT_NE(d_dst, nullptr);

  cudaMemcpy2D(d_src, srcStep, srcData.data(), width * sizeof(Npp32f) * 4, width * sizeof(Npp32f) * 4, height,
               cudaMemcpyHostToDevice);

  NppStatus status = nppiColorToGray_32f_C4C1R(d_src, srcStep, d_dst, dstStep, roi, coeffs);
  EXPECT_EQ(status, NPP_SUCCESS);

  std::vector<Npp32f> result(width * height);
  cudaMemcpy2D(result.data(), width * sizeof(Npp32f), d_dst, dstStep, width * sizeof(Npp32f), height,
               cudaMemcpyDeviceToHost);

  for (int i = 0; i < width * height; ++i) {
    EXPECT_NEAR(result[i], expected[i], 1e-5f) << "Mismatch at pixel " << i;
  }

  nppiFree(d_src);
  nppiFree(d_dst);
}

TEST_F(ColorToGrayTest, ColorToGray_16s_C3C1R_IdentityCoeff) {
  const Npp32f coeffs[3] = {1.0f, 0.0f, 0.0f};
  std::vector<Npp16s> srcData(width * height * 3);
  std::vector<Npp16s> expected(width * height);

  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      int idx = (y * width + x) * 3;
      Npp16s r = static_cast<Npp16s>((x + y) * 64 - 512);
      Npp16s g = static_cast<Npp16s>((x * 3 + y) * 32 - 256);
      Npp16s b = static_cast<Npp16s>((x + y * 2) * 48 - 384);
      srcData[idx + 0] = r;
      srcData[idx + 1] = g;
      srcData[idx + 2] = b;
      expected[y * width + x] = r;
    }
  }

  size_t srcStepBytes = 0;
  int dstStep = 0;
  Npp16s *d_src = nullptr;
  cudaError_t cudaStatus = cudaMallocPitch(reinterpret_cast<void **>(&d_src), &srcStepBytes,
                                           width * sizeof(Npp16s) * 3, height);
  ASSERT_EQ(cudaStatus, cudaSuccess);
  Npp16s *d_dst = nppiMalloc_16s_C1(width, height, &dstStep);
  ASSERT_NE(d_dst, nullptr);

  cudaMemcpy2D(d_src, srcStepBytes, srcData.data(), width * sizeof(Npp16s) * 3, width * sizeof(Npp16s) * 3, height,
               cudaMemcpyHostToDevice);

  NppStatus status = nppiColorToGray_16s_C3C1R(d_src, static_cast<int>(srcStepBytes), d_dst, dstStep, roi, coeffs);
  EXPECT_EQ(status, NPP_SUCCESS);

  std::vector<Npp16s> result(width * height);
  cudaMemcpy2D(result.data(), width * sizeof(Npp16s), d_dst, dstStep, width * sizeof(Npp16s), height,
               cudaMemcpyDeviceToHost);

  for (int i = 0; i < width * height; ++i) {
    EXPECT_EQ(result[i], expected[i]) << "Mismatch at pixel " << i;
  }

  cudaFree(d_src);
  nppiFree(d_dst);
}

TEST_F(ColorToGrayTest, ColorToGray_16s_AC4C1R_WeightedNoAlpha) {
  const Npp32f coeffs[3] = {0.25f, 0.5f, 0.25f};
  std::vector<Npp16s> srcData(width * height * 4);
  std::vector<Npp16s> expected(width * height);

  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      int idx = (y * width + x) * 4;
      Npp16s r = static_cast<Npp16s>((x + y) * 64 - 512);
      Npp16s g = static_cast<Npp16s>((x * 2 + y) * 64 - 512);
      Npp16s b = static_cast<Npp16s>((x + y * 2) * 64 - 512);
      Npp16s a = static_cast<Npp16s>((x * 5 + y * 7) * 64 - 512);
      srcData[idx + 0] = r;
      srcData[idx + 1] = g;
      srcData[idx + 2] = b;
      srcData[idx + 3] = a;
      expected[y * width + x] = static_cast<Npp16s>((r + 2 * g + b) / 4);
    }
  }

  int srcStep = 0;
  int dstStep = 0;
  Npp16s *d_src = nppiMalloc_16s_C4(width, height, &srcStep);
  Npp16s *d_dst = nppiMalloc_16s_C1(width, height, &dstStep);
  ASSERT_NE(d_src, nullptr);
  ASSERT_NE(d_dst, nullptr);

  cudaMemcpy2D(d_src, srcStep, srcData.data(), width * sizeof(Npp16s) * 4, width * sizeof(Npp16s) * 4, height,
               cudaMemcpyHostToDevice);

  NppStatus status = nppiColorToGray_16s_AC4C1R(d_src, srcStep, d_dst, dstStep, roi, coeffs);
  EXPECT_EQ(status, NPP_SUCCESS);

  std::vector<Npp16s> result(width * height);
  cudaMemcpy2D(result.data(), width * sizeof(Npp16s), d_dst, dstStep, width * sizeof(Npp16s), height,
               cudaMemcpyDeviceToHost);

  for (int i = 0; i < width * height; ++i) {
    EXPECT_EQ(result[i], expected[i]) << "Mismatch at pixel " << i;
  }

  nppiFree(d_src);
  nppiFree(d_dst);
}

TEST_F(ColorToGrayTest, ColorToGray_16s_C4C1R_UsesAlphaCoeff) {
  const Npp32f coeffs[4] = {0.25f, 0.25f, 0.25f, 0.25f};
  std::vector<Npp16s> srcData(width * height * 4);
  std::vector<Npp16s> expected(width * height);

  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      int idx = (y * width + x) * 4;
      Npp16s r = static_cast<Npp16s>((x + y) * 64 - 512);
      Npp16s g = static_cast<Npp16s>((x * 3 + y) * 64 - 512);
      Npp16s b = static_cast<Npp16s>((x + y * 2) * 64 - 512);
      Npp16s a = static_cast<Npp16s>((x * 2 + y * 3) * 64 - 512);
      srcData[idx + 0] = r;
      srcData[idx + 1] = g;
      srcData[idx + 2] = b;
      srcData[idx + 3] = a;
      expected[y * width + x] = static_cast<Npp16s>((r + g + b + a) / 4);
    }
  }

  int srcStep = 0;
  int dstStep = 0;
  Npp16s *d_src = nppiMalloc_16s_C4(width, height, &srcStep);
  Npp16s *d_dst = nppiMalloc_16s_C1(width, height, &dstStep);
  ASSERT_NE(d_src, nullptr);
  ASSERT_NE(d_dst, nullptr);

  cudaMemcpy2D(d_src, srcStep, srcData.data(), width * sizeof(Npp16s) * 4, width * sizeof(Npp16s) * 4, height,
               cudaMemcpyHostToDevice);

  NppStatus status = nppiColorToGray_16s_C4C1R(d_src, srcStep, d_dst, dstStep, roi, coeffs);
  EXPECT_EQ(status, NPP_SUCCESS);

  std::vector<Npp16s> result(width * height);
  cudaMemcpy2D(result.data(), width * sizeof(Npp16s), d_dst, dstStep, width * sizeof(Npp16s), height,
               cudaMemcpyDeviceToHost);

  for (int i = 0; i < width * height; ++i) {
    EXPECT_EQ(result[i], expected[i]) << "Mismatch at pixel " << i;
  }

  nppiFree(d_src);
  nppiFree(d_dst);
}
