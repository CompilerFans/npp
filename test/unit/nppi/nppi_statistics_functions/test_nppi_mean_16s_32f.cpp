#include "npp.h"

#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <type_traits>
#include <vector>

namespace {

enum class MeanLayout { C1, C3, C4, AC4 };

struct MeanCase {
  MeanLayout layout;
  int width;
  int height;
};

struct Mean16sApi {
  static NppStatus buffer(MeanLayout layout, NppiSize roi, int *size, NppStreamContext context, bool useContext) {
    switch (layout) {
    case MeanLayout::C1:
      return useContext ? nppiMeanGetBufferHostSize_16s_C1R_Ctx(roi, size, context)
                        : nppiMeanGetBufferHostSize_16s_C1R(roi, size);
    case MeanLayout::C3:
      return useContext ? nppiMeanGetBufferHostSize_16s_C3R_Ctx(roi, size, context)
                        : nppiMeanGetBufferHostSize_16s_C3R(roi, size);
    case MeanLayout::C4:
      return useContext ? nppiMeanGetBufferHostSize_16s_C4R_Ctx(roi, size, context)
                        : nppiMeanGetBufferHostSize_16s_C4R(roi, size);
    case MeanLayout::AC4:
      return useContext ? nppiMeanGetBufferHostSize_16s_AC4R_Ctx(roi, size, context)
                        : nppiMeanGetBufferHostSize_16s_AC4R(roi, size);
    }
    return NPP_BAD_ARGUMENT_ERROR;
  }

  static NppStatus compute(MeanLayout layout, const Npp16s *source, int step, NppiSize roi, Npp8u *buffer,
                           Npp64f *mean, NppStreamContext context, bool useContext) {
    switch (layout) {
    case MeanLayout::C1:
      return useContext ? nppiMean_16s_C1R_Ctx(source, step, roi, buffer, mean, context)
                        : nppiMean_16s_C1R(source, step, roi, buffer, mean);
    case MeanLayout::C3:
      return useContext ? nppiMean_16s_C3R_Ctx(source, step, roi, buffer, mean, context)
                        : nppiMean_16s_C3R(source, step, roi, buffer, mean);
    case MeanLayout::C4:
      return useContext ? nppiMean_16s_C4R_Ctx(source, step, roi, buffer, mean, context)
                        : nppiMean_16s_C4R(source, step, roi, buffer, mean);
    case MeanLayout::AC4:
      return useContext ? nppiMean_16s_AC4R_Ctx(source, step, roi, buffer, mean, context)
                        : nppiMean_16s_AC4R(source, step, roi, buffer, mean);
    }
    return NPP_BAD_ARGUMENT_ERROR;
  }
};

struct Mean32fApi {
  static NppStatus buffer(MeanLayout layout, NppiSize roi, int *size, NppStreamContext context, bool useContext) {
    switch (layout) {
    case MeanLayout::C1:
      return useContext ? nppiMeanGetBufferHostSize_32f_C1R_Ctx(roi, size, context)
                        : nppiMeanGetBufferHostSize_32f_C1R(roi, size);
    case MeanLayout::C3:
      return useContext ? nppiMeanGetBufferHostSize_32f_C3R_Ctx(roi, size, context)
                        : nppiMeanGetBufferHostSize_32f_C3R(roi, size);
    case MeanLayout::C4:
      return useContext ? nppiMeanGetBufferHostSize_32f_C4R_Ctx(roi, size, context)
                        : nppiMeanGetBufferHostSize_32f_C4R(roi, size);
    case MeanLayout::AC4:
      return useContext ? nppiMeanGetBufferHostSize_32f_AC4R_Ctx(roi, size, context)
                        : nppiMeanGetBufferHostSize_32f_AC4R(roi, size);
    }
    return NPP_BAD_ARGUMENT_ERROR;
  }

  static NppStatus compute(MeanLayout layout, const Npp32f *source, int step, NppiSize roi, Npp8u *buffer,
                           Npp64f *mean, NppStreamContext context, bool useContext) {
    switch (layout) {
    case MeanLayout::C1:
      return useContext ? nppiMean_32f_C1R_Ctx(source, step, roi, buffer, mean, context)
                        : nppiMean_32f_C1R(source, step, roi, buffer, mean);
    case MeanLayout::C3:
      return useContext ? nppiMean_32f_C3R_Ctx(source, step, roi, buffer, mean, context)
                        : nppiMean_32f_C3R(source, step, roi, buffer, mean);
    case MeanLayout::C4:
      return useContext ? nppiMean_32f_C4R_Ctx(source, step, roi, buffer, mean, context)
                        : nppiMean_32f_C4R(source, step, roi, buffer, mean);
    case MeanLayout::AC4:
      return useContext ? nppiMean_32f_AC4R_Ctx(source, step, roi, buffer, mean, context)
                        : nppiMean_32f_AC4R(source, step, roi, buffer, mean);
    }
    return NPP_BAD_ARGUMENT_ERROR;
  }
};

template <typename T> T sourceValue(int x, int y, int channel);

template <> Npp16s sourceValue<Npp16s>(int x, int y, int channel) {
  return static_cast<Npp16s>(((x * 977 + y * 2053 + channel * 7919 + 17) % 65536) - 32768);
}

template <> Npp32f sourceValue<Npp32f>(int x, int y, int channel) {
  return static_cast<Npp32f>((x - 11) * 0.75f + (y - 5) * 1.125f + channel * 3.0625f);
}

template <typename T, typename Api> void runMeanCase(const MeanCase &testCase, double tolerance) {
  const int sourceChannels = testCase.layout == MeanLayout::C1 ? 1 : testCase.layout == MeanLayout::C3 ? 3 : 4;
  const int outputChannels = testCase.layout == MeanLayout::C4 ? 4 : testCase.layout == MeanLayout::C1 ? 1 : 3;
  const NppiSize roi{testCase.width, testCase.height};
  const int hostStep = testCase.width * sourceChannels * static_cast<int>(sizeof(T));
  std::vector<T> source(static_cast<size_t>(testCase.width) * testCase.height * sourceChannels);
  std::vector<double> expected(outputChannels, 0.0);
  for (int y = 0; y < testCase.height; ++y) {
    for (int x = 0; x < testCase.width; ++x) {
      for (int channel = 0; channel < sourceChannels; ++channel) {
        const T value = sourceValue<T>(x, y, channel);
        source[(static_cast<size_t>(y) * testCase.width + x) * sourceChannels + channel] = value;
        if (channel < outputChannels) {
          expected[channel] += static_cast<double>(value);
        }
      }
    }
  }
  for (double &value : expected) {
    value /= static_cast<double>(testCase.width * testCase.height);
  }

  T *deviceSource = nullptr;
  size_t sourceStep = 0;
  ASSERT_EQ(cudaMallocPitch(&deviceSource, &sourceStep, hostStep, testCase.height), cudaSuccess);
  ASSERT_EQ(cudaMemcpy2D(deviceSource, sourceStep, source.data(), hostStep, hostStep, testCase.height,
                         cudaMemcpyHostToDevice),
            cudaSuccess);
  NppStreamContext context{};
  ASSERT_EQ(nppGetStreamContext(&context), NPP_SUCCESS);
  int bufferSize = 0;
  int contextBufferSize = 0;
  ASSERT_EQ(Api::buffer(testCase.layout, roi, &bufferSize, context, false), NPP_SUCCESS);
  ASSERT_EQ(Api::buffer(testCase.layout, roi, &contextBufferSize, context, true), NPP_SUCCESS);
  ASSERT_EQ(contextBufferSize, bufferSize);

  Npp8u *deviceBuffer = nullptr;
  Npp64f *deviceMean = nullptr;
  ASSERT_EQ(cudaMalloc(&deviceBuffer, bufferSize), cudaSuccess);
  ASSERT_EQ(cudaMalloc(&deviceMean, outputChannels * sizeof(Npp64f)), cudaSuccess);
  for (bool useContext : {false, true}) {
    ASSERT_EQ(Api::compute(testCase.layout, deviceSource, static_cast<int>(sourceStep), roi, deviceBuffer, deviceMean,
                           context, useContext),
              NPP_SUCCESS);
    std::vector<Npp64f> actual(outputChannels);
    ASSERT_EQ(cudaMemcpy(actual.data(), deviceMean, actual.size() * sizeof(Npp64f), cudaMemcpyDeviceToHost),
              cudaSuccess);
    for (int channel = 0; channel < outputChannels; ++channel) {
      EXPECT_NEAR(actual[channel], expected[channel], tolerance) << "channel=" << channel;
    }
  }

  EXPECT_EQ(Api::compute(testCase.layout, nullptr, static_cast<int>(sourceStep), roi, deviceBuffer, deviceMean,
                         context, false),
            NPP_NULL_POINTER_ERROR);
  EXPECT_EQ(Api::compute(testCase.layout, deviceSource, hostStep - 1, roi, deviceBuffer, deviceMean, context, false),
            NPP_STEP_ERROR);
  EXPECT_EQ(Api::compute(testCase.layout, deviceSource, static_cast<int>(sourceStep), {testCase.width, 0},
                         deviceBuffer, deviceMean, context, true),
            NPP_SIZE_ERROR);
  EXPECT_EQ(Api::buffer(testCase.layout, roi, nullptr, context, false), NPP_NULL_POINTER_ERROR);

  cudaFree(deviceSource);
  cudaFree(deviceBuffer);
  cudaFree(deviceMean);
}

const MeanCase kCases[] = {{MeanLayout::C1, 1, 1},   {MeanLayout::C1, 37, 5}, {MeanLayout::C3, 1, 1},
                           {MeanLayout::C3, 19, 8},  {MeanLayout::C4, 1, 1},  {MeanLayout::C4, 21, 7},
                           {MeanLayout::AC4, 1, 1}, {MeanLayout::AC4, 23, 6}};

TEST(NppiMean16sTest, LayoutsSizesEntryPointsAndParameters) {
  for (const MeanCase &testCase : kCases) {
    SCOPED_TRACE(testCase.width * 1000 + testCase.height);
    runMeanCase<Npp16s, Mean16sApi>(testCase, 1e-12);
  }
}

TEST(NppiMean32fTest, LayoutsSizesEntryPointsAndParameters) {
  for (const MeanCase &testCase : kCases) {
    SCOPED_TRACE(testCase.width * 1000 + testCase.height);
    runMeanCase<Npp32f, Mean32fApi>(testCase, 1e-10);
  }
}

} // namespace
