#include "npp.h"

#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <string>
#include <vector>

namespace {

enum class MeanLayout { C3, C4, AC4 };

struct MeanCase {
  MeanLayout layout;
  int width;
  int height;
  const char *name;
};

class NppiMeanMultiChannelTest : public ::testing::TestWithParam<MeanCase> {};

NppStatus getBufferSize(const MeanCase &testCase, NppiSize roi, int *bufferSize, NppStreamContext context,
                        bool useContext) {
  switch (testCase.layout) {
  case MeanLayout::C3:
    return useContext ? nppiMeanGetBufferHostSize_8u_C3R_Ctx(roi, bufferSize, context)
                      : nppiMeanGetBufferHostSize_8u_C3R(roi, bufferSize);
  case MeanLayout::C4:
    return useContext ? nppiMeanGetBufferHostSize_8u_C4R_Ctx(roi, bufferSize, context)
                      : nppiMeanGetBufferHostSize_8u_C4R(roi, bufferSize);
  case MeanLayout::AC4:
    return useContext ? nppiMeanGetBufferHostSize_8u_AC4R_Ctx(roi, bufferSize, context)
                      : nppiMeanGetBufferHostSize_8u_AC4R(roi, bufferSize);
  }
  return NPP_BAD_ARGUMENT_ERROR;
}

NppStatus computeMean(const MeanCase &testCase, const Npp8u *source, int sourceStep, NppiSize roi,
                      Npp8u *deviceBuffer, Npp64f *mean, NppStreamContext context, bool useContext) {
  switch (testCase.layout) {
  case MeanLayout::C3:
    return useContext ? nppiMean_8u_C3R_Ctx(source, sourceStep, roi, deviceBuffer, mean, context)
                      : nppiMean_8u_C3R(source, sourceStep, roi, deviceBuffer, mean);
  case MeanLayout::C4:
    return useContext ? nppiMean_8u_C4R_Ctx(source, sourceStep, roi, deviceBuffer, mean, context)
                      : nppiMean_8u_C4R(source, sourceStep, roi, deviceBuffer, mean);
  case MeanLayout::AC4:
    return useContext ? nppiMean_8u_AC4R_Ctx(source, sourceStep, roi, deviceBuffer, mean, context)
                      : nppiMean_8u_AC4R(source, sourceStep, roi, deviceBuffer, mean);
  }
  return NPP_BAD_ARGUMENT_ERROR;
}

TEST_P(NppiMeanMultiChannelTest, AccuracyEntryPointsAndParameters) {
  const MeanCase testCase = GetParam();
  const int sourceChannels = testCase.layout == MeanLayout::C3 ? 3 : 4;
  const int outputChannels = testCase.layout == MeanLayout::C4 ? 4 : 3;
  const NppiSize roi{testCase.width, testCase.height};
  const int hostStep = testCase.width * sourceChannels;
  std::vector<Npp8u> hostSource(static_cast<size_t>(hostStep) * testCase.height);
  std::vector<double> expected(outputChannels, 0.0);
  for (int y = 0; y < testCase.height; ++y) {
    for (int x = 0; x < testCase.width; ++x) {
      for (int channel = 0; channel < sourceChannels; ++channel) {
        const Npp8u value = static_cast<Npp8u>((x * 13 + y * 29 + channel * 47 + 5) % 256);
        hostSource[static_cast<size_t>(y) * hostStep + x * sourceChannels + channel] = value;
        if (channel < outputChannels) {
          expected[channel] += value;
        }
      }
    }
  }
  for (double &value : expected) {
    value /= static_cast<double>(testCase.width * testCase.height);
  }

  Npp8u *deviceSource = nullptr;
  size_t sourceStep = 0;
  ASSERT_EQ(cudaMallocPitch(&deviceSource, &sourceStep, hostStep, testCase.height), cudaSuccess);
  ASSERT_EQ(cudaMemcpy2D(deviceSource, sourceStep, hostSource.data(), hostStep, hostStep, testCase.height,
                         cudaMemcpyHostToDevice),
            cudaSuccess);

  NppStreamContext context{};
  ASSERT_EQ(nppGetStreamContext(&context), NPP_SUCCESS);
  int bufferSize = 0;
  int contextBufferSize = 0;
  ASSERT_EQ(getBufferSize(testCase, roi, &bufferSize, context, false), NPP_SUCCESS);
  ASSERT_EQ(getBufferSize(testCase, roi, &contextBufferSize, context, true), NPP_SUCCESS);
  ASSERT_EQ(contextBufferSize, bufferSize);
  ASSERT_GT(bufferSize, 0);

  Npp8u *deviceBuffer = nullptr;
  Npp64f *deviceMean = nullptr;
  ASSERT_EQ(cudaMalloc(&deviceBuffer, bufferSize), cudaSuccess);
  ASSERT_EQ(cudaMalloc(&deviceMean, outputChannels * sizeof(Npp64f)), cudaSuccess);

  for (bool useContext : {false, true}) {
    SCOPED_TRACE(useContext ? "Ctx" : "default");
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
  EXPECT_EQ(computeMean(testCase, deviceSource, static_cast<int>(sourceStep), {0, testCase.height}, deviceBuffer,
                        deviceMean, context, false),
            NPP_SIZE_ERROR);
  EXPECT_EQ(getBufferSize(testCase, roi, nullptr, context, false), NPP_NULL_POINTER_ERROR);
  EXPECT_EQ(getBufferSize(testCase, {testCase.width, 0}, &bufferSize, context, true), NPP_SIZE_ERROR);

  cudaFree(deviceSource);
  cudaFree(deviceBuffer);
  cudaFree(deviceMean);
}

INSTANTIATE_TEST_SUITE_P(
    LayoutsAndSizes, NppiMeanMultiChannelTest,
    ::testing::Values(MeanCase{MeanLayout::C3, 1, 1, "C3_1x1"}, MeanCase{MeanLayout::C3, 17, 9, "C3_17x9"},
                      MeanCase{MeanLayout::C3, 65, 5, "C3_65x5"}, MeanCase{MeanLayout::C4, 1, 1, "C4_1x1"},
                      MeanCase{MeanLayout::C4, 19, 7, "C4_19x7"}, MeanCase{MeanLayout::C4, 67, 4, "C4_67x4"},
                      MeanCase{MeanLayout::AC4, 1, 1, "AC4_1x1"}, MeanCase{MeanLayout::AC4, 23, 6, "AC4_23x6"},
                      MeanCase{MeanLayout::AC4, 69, 3, "AC4_69x3"}),
    [](const testing::TestParamInfo<MeanCase> &info) { return std::string(info.param.name); });

} // namespace
