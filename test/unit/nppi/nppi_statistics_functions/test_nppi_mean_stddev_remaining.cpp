#include "npp.h"

#include <cmath>
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <vector>

namespace {

enum class StdDevLayout { C1R, C1MR, C3CR, C3CMR };

struct MeanStdDev8sApi {
  using Value = Npp8s;
  static NppStatus buffer(StdDevLayout layout, NppiSize roi, int *size, NppStreamContext context, bool useCtx) {
    switch (layout) {
    case StdDevLayout::C1R:
      return useCtx ? nppiMeanStdDevGetBufferHostSize_8s_C1R_Ctx(roi, size, context)
                    : nppiMeanStdDevGetBufferHostSize_8s_C1R(roi, size);
    case StdDevLayout::C1MR:
      return useCtx ? nppiMeanStdDevGetBufferHostSize_8s_C1MR_Ctx(roi, size, context)
                    : nppiMeanStdDevGetBufferHostSize_8s_C1MR(roi, size);
    case StdDevLayout::C3CR:
      return useCtx ? nppiMeanStdDevGetBufferHostSize_8s_C3CR_Ctx(roi, size, context)
                    : nppiMeanStdDevGetBufferHostSize_8s_C3CR(roi, size);
    case StdDevLayout::C3CMR:
      return useCtx ? nppiMeanStdDevGetBufferHostSize_8s_C3CMR_Ctx(roi, size, context)
                    : nppiMeanStdDevGetBufferHostSize_8s_C3CMR(roi, size);
    }
    return NPP_BAD_ARGUMENT_ERROR;
  }
  static NppStatus compute(StdDevLayout layout, const Npp8s *src, int srcStep, const Npp8u *mask, int maskStep,
                           NppiSize roi, int coi, Npp8u *buffer, Npp64f *mean, Npp64f *stddev,
                           NppStreamContext context, bool useCtx) {
    switch (layout) {
    case StdDevLayout::C1R:
      return useCtx ? nppiMean_StdDev_8s_C1R_Ctx(src, srcStep, roi, buffer, mean, stddev, context)
                    : nppiMean_StdDev_8s_C1R(src, srcStep, roi, buffer, mean, stddev);
    case StdDevLayout::C1MR:
      return useCtx ? nppiMean_StdDev_8s_C1MR_Ctx(src, srcStep, mask, maskStep, roi, buffer, mean, stddev, context)
                    : nppiMean_StdDev_8s_C1MR(src, srcStep, mask, maskStep, roi, buffer, mean, stddev);
    case StdDevLayout::C3CR:
      return useCtx ? nppiMean_StdDev_8s_C3CR_Ctx(src, srcStep, roi, coi, buffer, mean, stddev, context)
                    : nppiMean_StdDev_8s_C3CR(src, srcStep, roi, coi, buffer, mean, stddev);
    case StdDevLayout::C3CMR:
      return useCtx
                 ? nppiMean_StdDev_8s_C3CMR_Ctx(src, srcStep, mask, maskStep, roi, coi, buffer, mean, stddev, context)
                 : nppiMean_StdDev_8s_C3CMR(src, srcStep, mask, maskStep, roi, coi, buffer, mean, stddev);
    }
    return NPP_BAD_ARGUMENT_ERROR;
  }
};

struct MeanStdDev16uApi {
  using Value = Npp16u;
  static NppStatus buffer(StdDevLayout layout, NppiSize roi, int *size, NppStreamContext context, bool useCtx) {
    switch (layout) {
    case StdDevLayout::C1R:
      return useCtx ? nppiMeanStdDevGetBufferHostSize_16u_C1R_Ctx(roi, size, context)
                    : nppiMeanStdDevGetBufferHostSize_16u_C1R(roi, size);
    case StdDevLayout::C1MR:
      return useCtx ? nppiMeanStdDevGetBufferHostSize_16u_C1MR_Ctx(roi, size, context)
                    : nppiMeanStdDevGetBufferHostSize_16u_C1MR(roi, size);
    case StdDevLayout::C3CR:
      return useCtx ? nppiMeanStdDevGetBufferHostSize_16u_C3CR_Ctx(roi, size, context)
                    : nppiMeanStdDevGetBufferHostSize_16u_C3CR(roi, size);
    case StdDevLayout::C3CMR:
      return useCtx ? nppiMeanStdDevGetBufferHostSize_16u_C3CMR_Ctx(roi, size, context)
                    : nppiMeanStdDevGetBufferHostSize_16u_C3CMR(roi, size);
    }
    return NPP_BAD_ARGUMENT_ERROR;
  }
  static NppStatus compute(StdDevLayout layout, const Npp16u *src, int srcStep, const Npp8u *mask, int maskStep,
                           NppiSize roi, int coi, Npp8u *buffer, Npp64f *mean, Npp64f *stddev,
                           NppStreamContext context, bool useCtx) {
    switch (layout) {
    case StdDevLayout::C1R:
      return useCtx ? nppiMean_StdDev_16u_C1R_Ctx(src, srcStep, roi, buffer, mean, stddev, context)
                    : nppiMean_StdDev_16u_C1R(src, srcStep, roi, buffer, mean, stddev);
    case StdDevLayout::C1MR:
      return useCtx ? nppiMean_StdDev_16u_C1MR_Ctx(src, srcStep, mask, maskStep, roi, buffer, mean, stddev, context)
                    : nppiMean_StdDev_16u_C1MR(src, srcStep, mask, maskStep, roi, buffer, mean, stddev);
    case StdDevLayout::C3CR:
      return useCtx ? nppiMean_StdDev_16u_C3CR_Ctx(src, srcStep, roi, coi, buffer, mean, stddev, context)
                    : nppiMean_StdDev_16u_C3CR(src, srcStep, roi, coi, buffer, mean, stddev);
    case StdDevLayout::C3CMR:
      return useCtx ? nppiMean_StdDev_16u_C3CMR_Ctx(src, srcStep, mask, maskStep, roi, coi, buffer, mean, stddev,
                                                    context)
                    : nppiMean_StdDev_16u_C3CMR(src, srcStep, mask, maskStep, roi, coi, buffer, mean, stddev);
    }
    return NPP_BAD_ARGUMENT_ERROR;
  }
};

struct MeanStdDev32fC3Api {
  using Value = Npp32f;
  static NppStatus buffer(StdDevLayout layout, NppiSize roi, int *size, NppStreamContext context, bool useCtx) {
    if (layout == StdDevLayout::C3CR) {
      return useCtx ? nppiMeanStdDevGetBufferHostSize_32f_C3CR_Ctx(roi, size, context)
                    : nppiMeanStdDevGetBufferHostSize_32f_C3CR(roi, size);
    }
    return useCtx ? nppiMeanStdDevGetBufferHostSize_32f_C3CMR_Ctx(roi, size, context)
                  : nppiMeanStdDevGetBufferHostSize_32f_C3CMR(roi, size);
  }
  static NppStatus compute(StdDevLayout layout, const Npp32f *src, int srcStep, const Npp8u *mask, int maskStep,
                           NppiSize roi, int coi, Npp8u *buffer, Npp64f *mean, Npp64f *stddev,
                           NppStreamContext context, bool useCtx) {
    if (layout == StdDevLayout::C3CR) {
      return useCtx ? nppiMean_StdDev_32f_C3CR_Ctx(src, srcStep, roi, coi, buffer, mean, stddev, context)
                    : nppiMean_StdDev_32f_C3CR(src, srcStep, roi, coi, buffer, mean, stddev);
    }
    return useCtx ? nppiMean_StdDev_32f_C3CMR_Ctx(src, srcStep, mask, maskStep, roi, coi, buffer, mean, stddev,
                                                  context)
                  : nppiMean_StdDev_32f_C3CMR(src, srcStep, mask, maskStep, roi, coi, buffer, mean, stddev);
  }
};

template <typename T> T sourceValue(int x, int y, int channel);
template <> Npp8s sourceValue<Npp8s>(int x, int y, int channel) {
  return static_cast<Npp8s>(((x * 17 + y * 31 + channel * 53 + 7) % 256) - 128);
}
template <> Npp16u sourceValue<Npp16u>(int x, int y, int channel) {
  return static_cast<Npp16u>((x * 977 + y * 2053 + channel * 7919 + 17) % 65536);
}
template <> Npp32f sourceValue<Npp32f>(int x, int y, int channel) {
  return static_cast<Npp32f>((x - 9) * 0.625f + (y - 4) * 1.375f + channel * 2.0625f);
}

template <typename Api>
void runStdDevCase(StdDevLayout layout, int coi, bool emptyMask, double tolerance) {
  using T = typename Api::Value;
  const int width = 19;
  const int height = 7;
  const bool masked = layout == StdDevLayout::C1MR || layout == StdDevLayout::C3CMR;
  const int channels = layout == StdDevLayout::C1R || layout == StdDevLayout::C1MR ? 1 : 3;
  const NppiSize roi{width, height};
  const int hostStep = width * channels * static_cast<int>(sizeof(T));
  std::vector<T> source(static_cast<size_t>(width) * height * channels);
  std::vector<Npp8u> mask(static_cast<size_t>(width) * height);
  double sum = 0.0;
  double sumSquares = 0.0;
  int count = 0;
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      const bool selected = !masked || (!emptyMask && ((x + 2 * y) % 3 != 0));
      mask[static_cast<size_t>(y) * width + x] = selected ? static_cast<Npp8u>(1 + ((x + y) % 3) * 127) : 0;
      for (int channel = 0; channel < channels; ++channel) {
        const T value = sourceValue<T>(x, y, channel);
        source[(static_cast<size_t>(y) * width + x) * channels + channel] = value;
        if (selected && channel == coi - 1) {
          const double converted = static_cast<double>(value);
          sum += converted;
          sumSquares += converted * converted;
        }
      }
      count += selected ? 1 : 0;
    }
  }
  const double expectedMean = count > 0 ? sum / count : 0.0;
  const double expectedStdDev =
      count > 0 ? std::sqrt(std::max(0.0, sumSquares / count - expectedMean * expectedMean)) : 0.0;

  T *deviceSource = nullptr;
  Npp8u *deviceMask = nullptr;
  size_t sourceStep = 0;
  size_t maskStep = 0;
  ASSERT_EQ(cudaMallocPitch(&deviceSource, &sourceStep, hostStep, height), cudaSuccess);
  ASSERT_EQ(cudaMallocPitch(&deviceMask, &maskStep, width, height), cudaSuccess);
  ASSERT_EQ(cudaMemcpy2D(deviceSource, sourceStep, source.data(), hostStep, hostStep, height, cudaMemcpyHostToDevice),
            cudaSuccess);
  ASSERT_EQ(cudaMemcpy2D(deviceMask, maskStep, mask.data(), width, width, height, cudaMemcpyHostToDevice), cudaSuccess);
  NppStreamContext context{};
  ASSERT_EQ(nppGetStreamContext(&context), NPP_SUCCESS);
  int bufferSize = 0;
  int contextBufferSize = 0;
  ASSERT_EQ(Api::buffer(layout, roi, &bufferSize, context, false), NPP_SUCCESS);
  ASSERT_EQ(Api::buffer(layout, roi, &contextBufferSize, context, true), NPP_SUCCESS);
  ASSERT_EQ(contextBufferSize, bufferSize);
  Npp8u *deviceBuffer = nullptr;
  Npp64f *deviceMean = nullptr;
  Npp64f *deviceStdDev = nullptr;
  ASSERT_EQ(cudaMalloc(&deviceBuffer, bufferSize), cudaSuccess);
  ASSERT_EQ(cudaMalloc(&deviceMean, sizeof(Npp64f)), cudaSuccess);
  ASSERT_EQ(cudaMalloc(&deviceStdDev, sizeof(Npp64f)), cudaSuccess);

  for (bool useCtx : {false, true}) {
    ASSERT_EQ(Api::compute(layout, deviceSource, static_cast<int>(sourceStep), deviceMask,
                           static_cast<int>(maskStep), roi, coi, deviceBuffer, deviceMean, deviceStdDev, context,
                           useCtx),
              NPP_SUCCESS);
    Npp64f actualMean = 0.0;
    Npp64f actualStdDev = 0.0;
    ASSERT_EQ(cudaMemcpy(&actualMean, deviceMean, sizeof(actualMean), cudaMemcpyDeviceToHost), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(&actualStdDev, deviceStdDev, sizeof(actualStdDev), cudaMemcpyDeviceToHost), cudaSuccess);
    EXPECT_NEAR(actualMean, expectedMean, tolerance);
    EXPECT_NEAR(actualStdDev, expectedStdDev, tolerance);
  }

  EXPECT_EQ(Api::compute(layout, nullptr, static_cast<int>(sourceStep), deviceMask, static_cast<int>(maskStep), roi,
                         coi, deviceBuffer, deviceMean, deviceStdDev, context, false),
            NPP_NULL_POINTER_ERROR);
  EXPECT_EQ(Api::compute(layout, deviceSource, hostStep - 1, deviceMask, static_cast<int>(maskStep), roi, coi,
                         deviceBuffer, deviceMean, deviceStdDev, context, false),
            NPP_STEP_ERROR);
  EXPECT_EQ(Api::compute(layout, deviceSource, static_cast<int>(sourceStep), deviceMask,
                         static_cast<int>(maskStep), {width, 0}, coi, deviceBuffer, deviceMean, deviceStdDev, context,
                         true),
            NPP_SIZE_ERROR);
  EXPECT_EQ(Api::compute(layout, deviceSource, static_cast<int>(sourceStep), deviceMask,
                         static_cast<int>(maskStep), roi, coi, nullptr, deviceMean, deviceStdDev, context, false),
            NPP_NULL_POINTER_ERROR);
  EXPECT_EQ(Api::compute(layout, deviceSource, static_cast<int>(sourceStep), deviceMask,
                         static_cast<int>(maskStep), roi, coi, deviceBuffer, nullptr, deviceStdDev, context, false),
            NPP_NULL_POINTER_ERROR);
  EXPECT_EQ(Api::compute(layout, deviceSource, static_cast<int>(sourceStep), deviceMask,
                         static_cast<int>(maskStep), roi, coi, deviceBuffer, deviceMean, nullptr, context, true),
            NPP_NULL_POINTER_ERROR);
  EXPECT_EQ(Api::buffer(layout, roi, nullptr, context, false), NPP_NULL_POINTER_ERROR);
  EXPECT_EQ(Api::buffer(layout, roi, nullptr, context, true), NPP_NULL_POINTER_ERROR);
  if (masked) {
    EXPECT_EQ(Api::compute(layout, deviceSource, static_cast<int>(sourceStep), nullptr,
                           static_cast<int>(maskStep), roi, coi, deviceBuffer, deviceMean, deviceStdDev, context,
                           false),
              NPP_NULL_POINTER_ERROR);
    EXPECT_EQ(Api::compute(layout, deviceSource, static_cast<int>(sourceStep), deviceMask, width - 1, roi, coi,
                           deviceBuffer, deviceMean, deviceStdDev, context, true),
              NPP_STEP_ERROR);
  }
  if (channels == 3) {
    EXPECT_EQ(Api::compute(layout, deviceSource, static_cast<int>(sourceStep), deviceMask,
                           static_cast<int>(maskStep), roi, 0, deviceBuffer, deviceMean, deviceStdDev, context, false),
              NPP_COI_ERROR);
    EXPECT_EQ(Api::compute(layout, deviceSource, static_cast<int>(sourceStep), deviceMask,
                           static_cast<int>(maskStep), roi, 4, deviceBuffer, deviceMean, deviceStdDev, context, true),
              NPP_COI_ERROR);
  }

  cudaFree(deviceSource);
  cudaFree(deviceMask);
  cudaFree(deviceBuffer);
  cudaFree(deviceMean);
  cudaFree(deviceStdDev);
}

template <typename Api> void runAllLayouts(double tolerance, bool includeC1) {
  if (includeC1) {
    runStdDevCase<Api>(StdDevLayout::C1R, 1, false, tolerance);
    runStdDevCase<Api>(StdDevLayout::C1MR, 1, false, tolerance);
    runStdDevCase<Api>(StdDevLayout::C1MR, 1, true, tolerance);
  }
  for (int coi = 1; coi <= 3; ++coi) {
    runStdDevCase<Api>(StdDevLayout::C3CR, coi, false, tolerance);
    runStdDevCase<Api>(StdDevLayout::C3CMR, coi, false, tolerance);
  }
  runStdDevCase<Api>(StdDevLayout::C3CMR, 2, true, tolerance);
}

TEST(NppiMeanStdDevRemainingTest, MeanStdDev_8s_AllLayoutsAndParameters) {
  runAllLayouts<MeanStdDev8sApi>(1e-10, true);
}
TEST(NppiMeanStdDevRemainingTest, MeanStdDev_16u_AllLayoutsAndParameters) {
  runAllLayouts<MeanStdDev16uApi>(1e-10, true);
}
TEST(NppiMeanStdDevRemainingTest, MeanStdDev_32f_C3LayoutsAndParameters) {
  runAllLayouts<MeanStdDev32fC3Api>(1e-9, false);
}

} // namespace
