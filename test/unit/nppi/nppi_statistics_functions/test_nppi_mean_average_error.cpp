#include "npp.h"
#include "npp_version_compat.h"

#include <cmath>
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <vector>

TEST(NppiMeanAndAverageErrorTest, Mean_8u_C1R_And_Ctx) {
  const int width = 29;
  const int height = 11;
  const NppiSize roi{width, height};
  std::vector<Npp8u> source(static_cast<size_t>(width) * height);
  double expected = 0.0;
  for (size_t i = 0; i < source.size(); ++i) {
    source[i] = static_cast<Npp8u>((i * 19 + 5) % 256);
    expected += source[i];
  }
  expected /= source.size();

  int step = 0;
  Npp8u *dSource = nppiMalloc_8u_C1(width, height, &step);
  ASSERT_NE(dSource, nullptr);
  ASSERT_EQ(cudaMemcpy2D(dSource, step, source.data(), width, width, height, cudaMemcpyHostToDevice), cudaSuccess);
  NppBufferSize bufferSize = 0;
  ASSERT_EQ(nppiMeanGetBufferHostSize_8u_C1R(roi, &bufferSize), NPP_SUCCESS);
  ASSERT_GT(bufferSize, 0);
  NppStreamContext context{};
  ASSERT_EQ(nppGetStreamContext(&context), NPP_SUCCESS);
  NppBufferSize contextBufferSize = 0;
  ASSERT_EQ(nppiMeanGetBufferHostSize_8u_C1R_Ctx(roi, &contextBufferSize, context), NPP_SUCCESS);
  EXPECT_GE(contextBufferSize, bufferSize);
  Npp8u *dBuffer = nullptr;
  Npp64f *dMean = nullptr;
  ASSERT_EQ(cudaMalloc(&dBuffer, contextBufferSize > bufferSize ? contextBufferSize : bufferSize), cudaSuccess);
  ASSERT_EQ(cudaMalloc(&dMean, sizeof(Npp64f)), cudaSuccess);

  ASSERT_EQ(nppiMean_8u_C1R(dSource, step, roi, dBuffer, dMean), NPP_SUCCESS);
  double mean = 0.0;
  ASSERT_EQ(cudaMemcpy(&mean, dMean, sizeof(mean), cudaMemcpyDeviceToHost), cudaSuccess);
  EXPECT_NEAR(mean, expected, 1e-12);

  ASSERT_EQ(nppiMean_8u_C1R_Ctx(dSource, step, roi, dBuffer, dMean, context), NPP_SUCCESS);
  EXPECT_EQ(nppiMean_8u_C1R(nullptr, step, roi, dBuffer, dMean), NPP_NULL_POINTER_ERROR);
  EXPECT_EQ(nppiMean_8u_C1R(dSource, width - 1, roi, dBuffer, dMean), NPP_STEP_ERROR);

  nppiFree(dSource);
  cudaFree(dBuffer);
  cudaFree(dMean);
}

TEST(NppiMeanAndAverageErrorTest, AverageError_8u_C1R_And_Ctx) {
  const int width = 31;
  const int height = 9;
  const NppiSize roi{width, height};
  std::vector<Npp8u> source1(static_cast<size_t>(width) * height);
  std::vector<Npp8u> source2(source1.size());
  double expected = 0.0;
  for (size_t i = 0; i < source1.size(); ++i) {
    source1[i] = static_cast<Npp8u>((i * 7 + 23) % 256);
    source2[i] = static_cast<Npp8u>((i * 13 + 3) % 256);
    expected += std::abs(static_cast<int>(source1[i]) - static_cast<int>(source2[i]));
  }
  expected /= source1.size();

  int step1 = 0;
  int step2 = 0;
  Npp8u *dSource1 = nppiMalloc_8u_C1(width, height, &step1);
  Npp8u *dSource2 = nppiMalloc_8u_C1(width, height, &step2);
  ASSERT_NE(dSource1, nullptr);
  ASSERT_NE(dSource2, nullptr);
  ASSERT_EQ(cudaMemcpy2D(dSource1, step1, source1.data(), width, width, height, cudaMemcpyHostToDevice), cudaSuccess);
  ASSERT_EQ(cudaMemcpy2D(dSource2, step2, source2.data(), width, width, height, cudaMemcpyHostToDevice), cudaSuccess);
  NppBufferSize bufferSize = 0;
  ASSERT_EQ(nppiAverageErrorGetBufferHostSize_8u_C1R(roi, &bufferSize), NPP_SUCCESS);
  ASSERT_GT(bufferSize, 0);
  Npp8u *dBuffer = nullptr;
  Npp64f *dError = nullptr;
  ASSERT_EQ(cudaMalloc(&dBuffer, bufferSize), cudaSuccess);
  ASSERT_EQ(cudaMalloc(&dError, sizeof(Npp64f)), cudaSuccess);

  ASSERT_EQ(nppiAverageError_8u_C1R(dSource1, step1, dSource2, step2, roi, dError, dBuffer), NPP_SUCCESS);
  double error = 0.0;
  ASSERT_EQ(cudaMemcpy(&error, dError, sizeof(error), cudaMemcpyDeviceToHost), cudaSuccess);
  EXPECT_NEAR(error, expected, 1e-12);

  NppStreamContext context{};
  ASSERT_EQ(nppGetStreamContext(&context), NPP_SUCCESS);
  NppBufferSize contextBufferSize = 0;
  ASSERT_EQ(nppiAverageErrorGetBufferHostSize_8u_C1R_Ctx(roi, &contextBufferSize, context), NPP_SUCCESS);
  EXPECT_GE(contextBufferSize, bufferSize);
  EXPECT_EQ(nppiAverageError_8u_C1R_Ctx(dSource1, step1, dSource2, step2, roi, dError, dBuffer, context), NPP_SUCCESS);
  EXPECT_EQ(nppiAverageError_8u_C1R(nullptr, step1, dSource2, step2, roi, dError, dBuffer), NPP_NULL_POINTER_ERROR);
  EXPECT_EQ(nppiAverageError_8u_C1R(dSource1, width - 1, dSource2, step2, roi, dError, dBuffer), NPP_STEP_ERROR);

  nppiFree(dSource1);
  nppiFree(dSource2);
  cudaFree(dBuffer);
  cudaFree(dError);
}

// ==============================================================================
// Business-scenario shapes: camera resolutions, even/odd sweep
// ==============================================================================

class MeanAverageErrorSizeTest : public ::testing::TestWithParam<std::pair<int, int>> {};

INSTANTIATE_TEST_SUITE_P(BusinessSizes, MeanAverageErrorSizeTest,
                         ::testing::Values(std::make_pair(640, 480), std::make_pair(16, 16), std::make_pair(15, 15)));

// Verify nppiMeanGetBufferHostSize formula (ceil(pixels/256)*8) and actual usage.
// 15x15=225 pixels exercises the non-multiple-of-256 buffer size path.
TEST_P(MeanAverageErrorSizeTest, Mean_8u_C1R_BusinessSizes) {
  const auto [width, height] = GetParam();
  const NppiSize roi{width, height};
  const size_t pixels = static_cast<size_t>(width) * height;
  std::vector<Npp8u> source(pixels);
  double expected = 0.0;
  for (size_t i = 0; i < source.size(); ++i) {
    source[i] = static_cast<Npp8u>((i * 37 + 11) % 256);
    expected += source[i];
  }
  expected /= source.size();

  int step = 0;
  Npp8u *dSource = nppiMalloc_8u_C1(width, height, &step);
  ASSERT_NE(dSource, nullptr);
  ASSERT_EQ(cudaMemcpy2D(dSource, step, source.data(), width, width, height, cudaMemcpyHostToDevice), cudaSuccess);

  NppBufferSize bufferSize = 0;
  ASSERT_EQ(nppiMeanGetBufferHostSize_8u_C1R(roi, &bufferSize), NPP_SUCCESS);
  ASSERT_GT(bufferSize, 0);

  Npp8u *dBuffer = nullptr;
  Npp64f *dMean = nullptr;
  ASSERT_EQ(cudaMalloc(&dBuffer, bufferSize), cudaSuccess);
  ASSERT_EQ(cudaMalloc(&dMean, sizeof(Npp64f)), cudaSuccess);

  ASSERT_EQ(nppiMean_8u_C1R(dSource, step, roi, dBuffer, dMean), NPP_SUCCESS);
  double mean = 0.0;
  ASSERT_EQ(cudaMemcpy(&mean, dMean, sizeof(mean), cudaMemcpyDeviceToHost), cudaSuccess);
  EXPECT_NEAR(mean, expected, 1e-12);

  NppStreamContext context{};
  ASSERT_EQ(nppGetStreamContext(&context), NPP_SUCCESS);
  ASSERT_EQ(nppiMean_8u_C1R_Ctx(dSource, step, roi, dBuffer, dMean, context), NPP_SUCCESS);
  ASSERT_EQ(cudaMemcpy(&mean, dMean, sizeof(mean), cudaMemcpyDeviceToHost), cudaSuccess);
  EXPECT_NEAR(mean, expected, 1e-12);

  nppiFree(dSource);
  cudaFree(dBuffer);
  cudaFree(dMean);
}

// Partial ROI with pointer offset: only the ROI region contributes to the mean.
TEST(NppiMeanAndAverageErrorTest, Mean_8u_C1R_PartialROI) {
  const int width = 128;
  const int height = 128;
  const int roiX = 32;
  const int roiY = 16;
  const int roiW = 64;
  const int roiH = 48;
  const NppiSize roi{roiW, roiH};
  std::vector<Npp8u> source(static_cast<size_t>(width) * height);
  for (size_t i = 0; i < source.size(); ++i) {
    source[i] = static_cast<Npp8u>((i * 17 + 9) % 256);
  }
  double expected = 0.0;
  for (int y = roiY; y < roiY + roiH; ++y) {
    for (int x = roiX; x < roiX + roiW; ++x) {
      expected += source[static_cast<size_t>(y) * width + x];
    }
  }
  expected /= roiW * roiH;

  int step = 0;
  Npp8u *dSource = nppiMalloc_8u_C1(width, height, &step);
  ASSERT_NE(dSource, nullptr);
  ASSERT_EQ(cudaMemcpy2D(dSource, step, source.data(), width, width, height, cudaMemcpyHostToDevice), cudaSuccess);
  Npp8u *dRoi = dSource + static_cast<size_t>(roiY) * step + roiX;

  NppBufferSize bufferSize = 0;
  ASSERT_EQ(nppiMeanGetBufferHostSize_8u_C1R(roi, &bufferSize), NPP_SUCCESS);
  Npp8u *dBuffer = nullptr;
  Npp64f *dMean = nullptr;
  ASSERT_EQ(cudaMalloc(&dBuffer, bufferSize), cudaSuccess);
  ASSERT_EQ(cudaMalloc(&dMean, sizeof(Npp64f)), cudaSuccess);

  ASSERT_EQ(nppiMean_8u_C1R(dRoi, step, roi, dBuffer, dMean), NPP_SUCCESS);
  double mean = 0.0;
  ASSERT_EQ(cudaMemcpy(&mean, dMean, sizeof(mean), cudaMemcpyDeviceToHost), cudaSuccess);
  EXPECT_NEAR(mean, expected, 1e-12);

  nppiFree(dSource);
  cudaFree(dBuffer);
  cudaFree(dMean);
}

TEST_P(MeanAverageErrorSizeTest, AverageError_8u_C1R_BusinessSizes) {
  const auto [width, height] = GetParam();
  const NppiSize roi{width, height};
  const size_t pixels = static_cast<size_t>(width) * height;
  std::vector<Npp8u> source1(pixels);
  std::vector<Npp8u> source2(pixels);
  double expected = 0.0;
  for (size_t i = 0; i < pixels; ++i) {
    source1[i] = static_cast<Npp8u>((i * 5 + 41) % 256);
    source2[i] = static_cast<Npp8u>((i * 11 + 7) % 256);
    expected += std::abs(static_cast<int>(source1[i]) - static_cast<int>(source2[i]));
  }
  expected /= pixels;

  int step1 = 0;
  int step2 = 0;
  Npp8u *dSource1 = nppiMalloc_8u_C1(width, height, &step1);
  Npp8u *dSource2 = nppiMalloc_8u_C1(width, height, &step2);
  ASSERT_NE(dSource1, nullptr);
  ASSERT_NE(dSource2, nullptr);
  ASSERT_EQ(cudaMemcpy2D(dSource1, step1, source1.data(), width, width, height, cudaMemcpyHostToDevice), cudaSuccess);
  ASSERT_EQ(cudaMemcpy2D(dSource2, step2, source2.data(), width, width, height, cudaMemcpyHostToDevice), cudaSuccess);

  NppBufferSize bufferSize = 0;
  ASSERT_EQ(nppiAverageErrorGetBufferHostSize_8u_C1R(roi, &bufferSize), NPP_SUCCESS);
  ASSERT_GT(bufferSize, 0);

  Npp8u *dBuffer = nullptr;
  Npp64f *dError = nullptr;
  ASSERT_EQ(cudaMalloc(&dBuffer, bufferSize), cudaSuccess);
  ASSERT_EQ(cudaMalloc(&dError, sizeof(Npp64f)), cudaSuccess);

  ASSERT_EQ(nppiAverageError_8u_C1R(dSource1, step1, dSource2, step2, roi, dError, dBuffer), NPP_SUCCESS);
  double error = 0.0;
  ASSERT_EQ(cudaMemcpy(&error, dError, sizeof(error), cudaMemcpyDeviceToHost), cudaSuccess);
  EXPECT_NEAR(error, expected, 1e-12);

  nppiFree(dSource1);
  nppiFree(dSource2);
  cudaFree(dBuffer);
  cudaFree(dError);
}

TEST(NppiMeanAndAverageErrorTest, AverageError_8u_C1R_PartialROI) {
  const int width = 128;
  const int height = 128;
  const int roiX = 16;
  const int roiY = 24;
  const int roiW = 96;
  const int roiH = 64;
  const NppiSize roi{roiW, roiH};
  std::vector<Npp8u> source1(static_cast<size_t>(width) * height);
  std::vector<Npp8u> source2(source1.size());
  for (size_t i = 0; i < source1.size(); ++i) {
    source1[i] = static_cast<Npp8u>((i * 29 + 13) % 256);
    source2[i] = static_cast<Npp8u>((i * 3 + 61) % 256);
  }
  double expected = 0.0;
  for (int y = roiY; y < roiY + roiH; ++y) {
    for (int x = roiX; x < roiX + roiW; ++x) {
      const size_t idx = static_cast<size_t>(y) * width + x;
      expected += std::abs(static_cast<int>(source1[idx]) - static_cast<int>(source2[idx]));
    }
  }
  expected /= roiW * roiH;

  int step1 = 0;
  int step2 = 0;
  Npp8u *dSource1 = nppiMalloc_8u_C1(width, height, &step1);
  Npp8u *dSource2 = nppiMalloc_8u_C1(width, height, &step2);
  ASSERT_NE(dSource1, nullptr);
  ASSERT_NE(dSource2, nullptr);
  ASSERT_EQ(cudaMemcpy2D(dSource1, step1, source1.data(), width, width, height, cudaMemcpyHostToDevice), cudaSuccess);
  ASSERT_EQ(cudaMemcpy2D(dSource2, step2, source2.data(), width, width, height, cudaMemcpyHostToDevice), cudaSuccess);
  Npp8u *dRoi1 = dSource1 + static_cast<size_t>(roiY) * step1 + roiX;
  Npp8u *dRoi2 = dSource2 + static_cast<size_t>(roiY) * step2 + roiX;

  NppBufferSize bufferSize = 0;
  ASSERT_EQ(nppiAverageErrorGetBufferHostSize_8u_C1R(roi, &bufferSize), NPP_SUCCESS);
  Npp8u *dBuffer = nullptr;
  Npp64f *dError = nullptr;
  ASSERT_EQ(cudaMalloc(&dBuffer, bufferSize), cudaSuccess);
  ASSERT_EQ(cudaMalloc(&dError, sizeof(Npp64f)), cudaSuccess);

  ASSERT_EQ(nppiAverageError_8u_C1R(dRoi1, step1, dRoi2, step2, roi, dError, dBuffer), NPP_SUCCESS);
  double error = 0.0;
  ASSERT_EQ(cudaMemcpy(&error, dError, sizeof(error), cudaMemcpyDeviceToHost), cudaSuccess);
  EXPECT_NEAR(error, expected, 1e-12);

  nppiFree(dSource1);
  nppiFree(dSource2);
  cudaFree(dBuffer);
  cudaFree(dError);
}
