#include "npp.h"
#include "npp_test_base.h"

#include <algorithm>
#include <cmath>
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <vector>

namespace {

struct MeanStdDevC3Params {
  int coi;
  bool useContext;
};

class NppiMeanStdDevC3Test : public ::testing::TestWithParam<MeanStdDevC3Params> {
protected:
  void SetUp() override { ASSERT_EQ(cudaSetDevice(0), cudaSuccess); }

  void TearDown() override { EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess); }

  static std::vector<Npp8u> makeSource(int width, int height) {
    std::vector<Npp8u> source(static_cast<size_t>(width) * height * 3);
    for (int y = 0; y < height; ++y) {
      for (int x = 0; x < width; ++x) {
        const size_t offset = (static_cast<size_t>(y) * width + x) * 3;
        source[offset] = static_cast<Npp8u>((x * 3 + y * 5 + 7) % 251);
        source[offset + 1] = static_cast<Npp8u>((x * 11 + y * 2 + 19) % 253);
        source[offset + 2] = static_cast<Npp8u>((x * 7 + y * 13 + 31) % 255);
      }
    }
    return source;
  }

  static std::vector<Npp8u> makeMask(int width, int height) {
    std::vector<Npp8u> mask(static_cast<size_t>(width) * height);
    for (int y = 0; y < height; ++y) {
      for (int x = 0; x < width; ++x) {
        mask[static_cast<size_t>(y) * width + x] = ((x + 2 * y) % 4 == 0) ? 0 : 255;
      }
    }
    return mask;
  }

  static void reference(const std::vector<Npp8u> &source, const std::vector<Npp8u> *mask, int width, int height,
                        int coi, double &mean, double &stddev) {
    double sum = 0.0;
    double sumSquares = 0.0;
    size_t count = 0;
    for (int y = 0; y < height; ++y) {
      for (int x = 0; x < width; ++x) {
        const size_t pixel = static_cast<size_t>(y) * width + x;
        if (mask && (*mask)[pixel] == 0) {
          continue;
        }
        const double value = source[pixel * 3 + static_cast<size_t>(coi - 1)];
        sum += value;
        sumSquares += value * value;
        ++count;
      }
    }
    mean = sum / count;
    stddev = std::sqrt(std::max(0.0, sumSquares / count - mean * mean));
  }
};

TEST_P(NppiMeanStdDevC3Test, Mean_StdDev_8u_C3CR_Accuracy) {
  const int width = 37;
  const int height = 19;
  const NppiSize roi{width, height};
  const auto source = makeSource(width, height);
  const auto param = GetParam();

  int srcStep = 0;
  Npp8u *dSource = nppiMalloc_8u_C3(width, height, &srcStep);
  ASSERT_NE(dSource, nullptr);
  ASSERT_EQ(cudaMemcpy2D(dSource, srcStep, source.data(), width * 3, width * 3, height, cudaMemcpyHostToDevice),
            cudaSuccess);

  SIZE_TYPE bufferSize = 0;
  NppStreamContext context{};
  ASSERT_EQ(nppGetStreamContext(&context), NPP_SUCCESS);
  NppStatus status = param.useContext ? nppiMeanStdDevGetBufferHostSize_8u_C3CR_Ctx(roi, &bufferSize, context)
                                      : nppiMeanStdDevGetBufferHostSize_8u_C3CR(roi, &bufferSize);
  ASSERT_EQ(status, NPP_SUCCESS);
  ASSERT_GT(bufferSize, 0);

  Npp8u *dBuffer = nullptr;
  Npp64f *dMean = nullptr;
  Npp64f *dStdDev = nullptr;
  ASSERT_EQ(cudaMalloc(&dBuffer, bufferSize), cudaSuccess);
  ASSERT_EQ(cudaMalloc(&dMean, sizeof(Npp64f)), cudaSuccess);
  ASSERT_EQ(cudaMalloc(&dStdDev, sizeof(Npp64f)), cudaSuccess);

  status = param.useContext
               ? nppiMean_StdDev_8u_C3CR_Ctx(dSource, srcStep, roi, param.coi, dBuffer, dMean, dStdDev, context)
               : nppiMean_StdDev_8u_C3CR(dSource, srcStep, roi, param.coi, dBuffer, dMean, dStdDev);
  ASSERT_EQ(status, NPP_SUCCESS);

  double mean = 0.0;
  double stddev = 0.0;
  ASSERT_EQ(cudaMemcpy(&mean, dMean, sizeof(mean), cudaMemcpyDeviceToHost), cudaSuccess);
  ASSERT_EQ(cudaMemcpy(&stddev, dStdDev, sizeof(stddev), cudaMemcpyDeviceToHost), cudaSuccess);
  double expectedMean = 0.0;
  double expectedStdDev = 0.0;
  reference(source, nullptr, width, height, param.coi, expectedMean, expectedStdDev);
  EXPECT_NEAR(mean, expectedMean, 1e-9);
  EXPECT_NEAR(stddev, expectedStdDev, 1e-9);

  nppiFree(dSource);
  cudaFree(dBuffer);
  cudaFree(dMean);
  cudaFree(dStdDev);
}

TEST_P(NppiMeanStdDevC3Test, Mean_StdDev_8u_C3CMR_Accuracy) {
  const int width = 41;
  const int height = 23;
  const NppiSize roi{width, height};
  const auto source = makeSource(width, height);
  const auto mask = makeMask(width, height);
  const auto param = GetParam();

  int srcStep = 0;
  int maskStep = 0;
  Npp8u *dSource = nppiMalloc_8u_C3(width, height, &srcStep);
  Npp8u *dMask = nppiMalloc_8u_C1(width, height, &maskStep);
  ASSERT_NE(dSource, nullptr);
  ASSERT_NE(dMask, nullptr);
  ASSERT_EQ(cudaMemcpy2D(dSource, srcStep, source.data(), width * 3, width * 3, height, cudaMemcpyHostToDevice),
            cudaSuccess);
  ASSERT_EQ(cudaMemcpy2D(dMask, maskStep, mask.data(), width, width, height, cudaMemcpyHostToDevice), cudaSuccess);

  SIZE_TYPE bufferSize = 0;
  NppStreamContext context{};
  ASSERT_EQ(nppGetStreamContext(&context), NPP_SUCCESS);
  NppStatus status = param.useContext ? nppiMeanStdDevGetBufferHostSize_8u_C3CMR_Ctx(roi, &bufferSize, context)
                                      : nppiMeanStdDevGetBufferHostSize_8u_C3CMR(roi, &bufferSize);
  ASSERT_EQ(status, NPP_SUCCESS);
  ASSERT_GT(bufferSize, 0);

  Npp8u *dBuffer = nullptr;
  Npp64f *dMean = nullptr;
  Npp64f *dStdDev = nullptr;
  ASSERT_EQ(cudaMalloc(&dBuffer, bufferSize), cudaSuccess);
  ASSERT_EQ(cudaMalloc(&dMean, sizeof(Npp64f)), cudaSuccess);
  ASSERT_EQ(cudaMalloc(&dStdDev, sizeof(Npp64f)), cudaSuccess);

  status = param.useContext ? nppiMean_StdDev_8u_C3CMR_Ctx(dSource, srcStep, dMask, maskStep, roi, param.coi,
                                                            dBuffer, dMean, dStdDev, context)
                            : nppiMean_StdDev_8u_C3CMR(dSource, srcStep, dMask, maskStep, roi, param.coi, dBuffer,
                                                       dMean, dStdDev);
  ASSERT_EQ(status, NPP_SUCCESS);

  double mean = 0.0;
  double stddev = 0.0;
  ASSERT_EQ(cudaMemcpy(&mean, dMean, sizeof(mean), cudaMemcpyDeviceToHost), cudaSuccess);
  ASSERT_EQ(cudaMemcpy(&stddev, dStdDev, sizeof(stddev), cudaMemcpyDeviceToHost), cudaSuccess);
  double expectedMean = 0.0;
  double expectedStdDev = 0.0;
  reference(source, &mask, width, height, param.coi, expectedMean, expectedStdDev);
  EXPECT_NEAR(mean, expectedMean, 1e-9);
  EXPECT_NEAR(stddev, expectedStdDev, 1e-9);

  nppiFree(dSource);
  nppiFree(dMask);
  cudaFree(dBuffer);
  cudaFree(dMean);
  cudaFree(dStdDev);
}

INSTANTIATE_TEST_SUITE_P(AllChannelsAndEntryPoints, NppiMeanStdDevC3Test,
                         ::testing::Values(MeanStdDevC3Params{1, false}, MeanStdDevC3Params{2, false},
                                           MeanStdDevC3Params{3, false}, MeanStdDevC3Params{1, true},
                                           MeanStdDevC3Params{2, true}, MeanStdDevC3Params{3, true}));

class NppiMeanStdDevC3ErrorTest : public ::testing::Test {};

TEST_F(NppiMeanStdDevC3ErrorTest, BufferHelpersValidateArguments) {
  NppStreamContext context{};
  ASSERT_EQ(nppGetStreamContext(&context), NPP_SUCCESS);
  SIZE_TYPE size = 0;
  EXPECT_EQ(nppiMeanStdDevGetBufferHostSize_8u_C3CR({0, 8}, &size), NPP_SIZE_ERROR);
  EXPECT_EQ(nppiMeanStdDevGetBufferHostSize_8u_C3CMR({8, 0}, &size), NPP_SIZE_ERROR);
  EXPECT_EQ(nppiMeanStdDevGetBufferHostSize_8u_C3CR_Ctx({8, 8}, nullptr, context), NPP_NULL_POINTER_ERROR);
  EXPECT_EQ(nppiMeanStdDevGetBufferHostSize_8u_C3CMR_Ctx({8, 8}, nullptr, context), NPP_NULL_POINTER_ERROR);
}

TEST_F(NppiMeanStdDevC3ErrorTest, ComputeFunctionsValidateArguments) {
  const NppiSize roi{8, 8};
  int srcStep = 0;
  int maskStep = 0;
  Npp8u *source = nppiMalloc_8u_C3(roi.width, roi.height, &srcStep);
  Npp8u *mask = nppiMalloc_8u_C1(roi.width, roi.height, &maskStep);
  ASSERT_NE(source, nullptr);
  ASSERT_NE(mask, nullptr);

  SIZE_TYPE bufferSize = 0;
  ASSERT_EQ(nppiMeanStdDevGetBufferHostSize_8u_C3CMR(roi, &bufferSize), NPP_SUCCESS);
  Npp8u *buffer = nullptr;
  Npp64f *mean = nullptr;
  Npp64f *stddev = nullptr;
  ASSERT_EQ(cudaMalloc(&buffer, bufferSize), cudaSuccess);
  ASSERT_EQ(cudaMalloc(&mean, sizeof(Npp64f)), cudaSuccess);
  ASSERT_EQ(cudaMalloc(&stddev, sizeof(Npp64f)), cudaSuccess);

  EXPECT_EQ(nppiMean_StdDev_8u_C3CR(nullptr, srcStep, roi, 1, buffer, mean, stddev), NPP_NULL_POINTER_ERROR);
  EXPECT_EQ(nppiMean_StdDev_8u_C3CMR(source, srcStep, nullptr, maskStep, roi, 1, buffer, mean, stddev),
            NPP_NULL_POINTER_ERROR);
  EXPECT_EQ(nppiMean_StdDev_8u_C3CR(source, roi.width * 3 - 1, roi, 1, buffer, mean, stddev), NPP_STEP_ERROR);
  EXPECT_EQ(nppiMean_StdDev_8u_C3CMR(source, srcStep, mask, roi.width - 1, roi, 1, buffer, mean, stddev),
            NPP_STEP_ERROR);
  EXPECT_EQ(nppiMean_StdDev_8u_C3CR(source, srcStep, roi, 0, buffer, mean, stddev), NPP_COI_ERROR);
  EXPECT_EQ(nppiMean_StdDev_8u_C3CR(source, srcStep, roi, 4, buffer, mean, stddev), NPP_COI_ERROR);
  EXPECT_EQ(nppiMean_StdDev_8u_C3CMR(source, srcStep, mask, maskStep, roi, -1, buffer, mean, stddev), NPP_COI_ERROR);

  nppiFree(source);
  nppiFree(mask);
  cudaFree(buffer);
  cudaFree(mean);
  cudaFree(stddev);
}

} // namespace
