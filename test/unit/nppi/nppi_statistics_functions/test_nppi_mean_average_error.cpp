#include "npp.h"

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
  int bufferSize = 0;
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
  int contextBufferSize = 0;
  ASSERT_EQ(nppiMeanGetBufferHostSize_8u_C1R_Ctx(roi, &contextBufferSize, context), NPP_SUCCESS);
  EXPECT_EQ(contextBufferSize, bufferSize);
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
  int bufferSize = 0;
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
  int contextBufferSize = 0;
  ASSERT_EQ(nppiAverageErrorGetBufferHostSize_8u_C1R_Ctx(roi, &contextBufferSize, context), NPP_SUCCESS);
  EXPECT_EQ(contextBufferSize, bufferSize);
  EXPECT_EQ(nppiAverageError_8u_C1R_Ctx(dSource1, step1, dSource2, step2, roi, dError, dBuffer, context),
            NPP_SUCCESS);
  EXPECT_EQ(nppiAverageError_8u_C1R(nullptr, step1, dSource2, step2, roi, dError, dBuffer),
            NPP_NULL_POINTER_ERROR);
  EXPECT_EQ(nppiAverageError_8u_C1R(dSource1, width - 1, dSource2, step2, roi, dError, dBuffer), NPP_STEP_ERROR);

  nppiFree(dSource1);
  nppiFree(dSource2);
  cudaFree(dBuffer);
  cudaFree(dError);
}
