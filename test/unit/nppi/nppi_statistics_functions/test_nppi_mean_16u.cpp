#include "npp.h"

#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <string>
#include <vector>

namespace {

enum class Mean16uLayout { C1, C3, C4, AC4 };

struct Mean16uCase {
  Mean16uLayout layout;
  int width;
  int height;
  const char *name;
};

class NppiMean16uTest : public ::testing::TestWithParam<Mean16uCase> {};

NppStatus getBufferSize(const Mean16uCase &testCase, NppiSize roi, int *bufferSize, NppStreamContext context,
                        bool useContext) {
  switch (testCase.layout) {
  case Mean16uLayout::C1:
    return useContext ? nppiMeanGetBufferHostSize_16u_C1R_Ctx(roi, bufferSize, context)
                      : nppiMeanGetBufferHostSize_16u_C1R(roi, bufferSize);
  case Mean16uLayout::C3:
    return useContext ? nppiMeanGetBufferHostSize_16u_C3R_Ctx(roi, bufferSize, context)
                      : nppiMeanGetBufferHostSize_16u_C3R(roi, bufferSize);
  case Mean16uLayout::C4:
    return useContext ? nppiMeanGetBufferHostSize_16u_C4R_Ctx(roi, bufferSize, context)
                      : nppiMeanGetBufferHostSize_16u_C4R(roi, bufferSize);
  case Mean16uLayout::AC4:
    return useContext ? nppiMeanGetBufferHostSize_16u_AC4R_Ctx(roi, bufferSize, context)
                      : nppiMeanGetBufferHostSize_16u_AC4R(roi, bufferSize);
  }
  return NPP_BAD_ARGUMENT_ERROR;
}

NppStatus computeMean(const Mean16uCase &testCase, const Npp16u *source, int sourceStep, NppiSize roi,
                      Npp8u *deviceBuffer, Npp64f *mean, NppStreamContext context, bool useContext) {
  switch (testCase.layout) {
  case Mean16uLayout::C1:
    return useContext ? nppiMean_16u_C1R_Ctx(source, sourceStep, roi, deviceBuffer, mean, context)
                      : nppiMean_16u_C1R(source, sourceStep, roi, deviceBuffer, mean);
  case Mean16uLayout::C3:
    return useContext ? nppiMean_16u_C3R_Ctx(source, sourceStep, roi, deviceBuffer, mean, context)
                      : nppiMean_16u_C3R(source, sourceStep, roi, deviceBuffer, mean);
  case Mean16uLayout::C4:
    return useContext ? nppiMean_16u_C4R_Ctx(source, sourceStep, roi, deviceBuffer, mean, context)
                      : nppiMean_16u_C4R(source, sourceStep, roi, deviceBuffer, mean);
  case Mean16uLayout::AC4:
    return useContext ? nppiMean_16u_AC4R_Ctx(source, sourceStep, roi, deviceBuffer, mean, context)
                      : nppiMean_16u_AC4R(source, sourceStep, roi, deviceBuffer, mean);
  }
  return NPP_BAD_ARGUMENT_ERROR;
}

TEST_P(NppiMean16uTest, AccuracyEntryPointsAndParameters) {
  const Mean16uCase testCase = GetParam();
  const int sourceChannels = testCase.layout == Mean16uLayout::C1 ? 1 : testCase.layout == Mean16uLayout::C3 ? 3 : 4;
  const int outputChannels = testCase.layout == Mean16uLayout::C4 ? 4 : testCase.layout == Mean16uLayout::C1 ? 1 : 3;
  const NppiSize roi{testCase.width, testCase.height};
  const int hostStep = testCase.width * sourceChannels * static_cast<int>(sizeof(Npp16u));
  std::vector<Npp16u> source(static_cast<size_t>(testCase.width) * testCase.height * sourceChannels);
  std::vector<double> expected(outputChannels, 0.0);
  for (int y = 0; y < testCase.height; ++y) {
    for (int x = 0; x < testCase.width; ++x) {
      for (int channel = 0; channel < sourceChannels; ++channel) {
        const Npp16u value = static_cast<Npp16u>((x * 977 + y * 2053 + channel * 7919 + 17) % 65536);
        source[(static_cast<size_t>(y) * testCase.width + x) * sourceChannels + channel] = value;
        if (channel < outputChannels) {
          expected[channel] += value;
        }
      }
    }
  }
  for (double &value : expected) {
    value /= static_cast<double>(testCase.width * testCase.height);
  }

  Npp16u *deviceSource = nullptr;
  size_t sourceStep = 0;
  ASSERT_EQ(cudaMallocPitch(&deviceSource, &sourceStep, hostStep, testCase.height), cudaSuccess);
  ASSERT_EQ(cudaMemcpy2D(deviceSource, sourceStep, source.data(), hostStep, hostStep, testCase.height,
                         cudaMemcpyHostToDevice),
            cudaSuccess);
  NppStreamContext context{};
  ASSERT_EQ(nppGetStreamContext(&context), NPP_SUCCESS);
  int bufferSize = 0;
  int contextBufferSize = 0;
  ASSERT_EQ(getBufferSize(testCase, roi, &bufferSize, context, false), NPP_SUCCESS);
  ASSERT_EQ(getBufferSize(testCase, roi, &contextBufferSize, context, true), NPP_SUCCESS);
  ASSERT_EQ(contextBufferSize, bufferSize);

  Npp8u *deviceBuffer = nullptr;
  Npp64f *deviceMean = nullptr;
  ASSERT_EQ(cudaMalloc(&deviceBuffer, bufferSize), cudaSuccess);
  ASSERT_EQ(cudaMalloc(&deviceMean, outputChannels * sizeof(Npp64f)), cudaSuccess);
  for (bool useContext : {false, true}) {
    ASSERT_EQ(computeMean(testCase, deviceSource, static_cast<int>(sourceStep), roi, deviceBuffer, deviceMean,
                          context, useContext),
              NPP_SUCCESS);
    std::vector<Npp64f> actual(outputChannels);
    ASSERT_EQ(cudaMemcpy(actual.data(), deviceMean, actual.size() * sizeof(Npp64f), cudaMemcpyDeviceToHost),
              cudaSuccess);
    for (int channel = 0; channel < outputChannels; ++channel) {
      EXPECT_NEAR(actual[channel], expected[channel], 1e-12) << "channel=" << channel;
    }
  }

  EXPECT_EQ(computeMean(testCase, nullptr, static_cast<int>(sourceStep), roi, deviceBuffer, deviceMean, context,
                        false),
            NPP_NULL_POINTER_ERROR);
  EXPECT_EQ(computeMean(testCase, deviceSource, hostStep - 1, roi, deviceBuffer, deviceMean, context, false),
            NPP_STEP_ERROR);
  EXPECT_EQ(computeMean(testCase, deviceSource, static_cast<int>(sourceStep), {-1, testCase.height}, deviceBuffer,
                        deviceMean, context, true),
            NPP_SIZE_ERROR);
  EXPECT_EQ(getBufferSize(testCase, roi, nullptr, context, false), NPP_NULL_POINTER_ERROR);

  cudaFree(deviceSource);
  cudaFree(deviceBuffer);
  cudaFree(deviceMean);
}

INSTANTIATE_TEST_SUITE_P(
    LayoutsAndSizes, NppiMean16uTest,
    ::testing::Values(Mean16uCase{Mean16uLayout::C1, 1, 1, "C1_1x1"},
                      Mean16uCase{Mean16uLayout::C1, 37, 5, "C1_37x5"},
                      Mean16uCase{Mean16uLayout::C3, 1, 1, "C3_1x1"},
                      Mean16uCase{Mean16uLayout::C3, 19, 8, "C3_19x8"},
                      Mean16uCase{Mean16uLayout::C4, 1, 1, "C4_1x1"},
                      Mean16uCase{Mean16uLayout::C4, 21, 7, "C4_21x7"},
                      Mean16uCase{Mean16uLayout::AC4, 1, 1, "AC4_1x1"},
                      Mean16uCase{Mean16uLayout::AC4, 23, 6, "AC4_23x6"}),
    [](const testing::TestParamInfo<Mean16uCase> &info) { return std::string(info.param.name); });

} // namespace
