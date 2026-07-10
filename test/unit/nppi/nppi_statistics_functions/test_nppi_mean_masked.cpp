#include "npp.h"

#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <vector>

namespace {

enum class MaskedLayout { C1MR, C3CMR };

struct Mean8uApi {
  using Value = Npp8u;
  static NppStatus buffer(MaskedLayout layout, NppiSize roi, int *size, NppStreamContext context, bool useCtx) {
    if (layout == MaskedLayout::C1MR) {
      return useCtx ? nppiMeanGetBufferHostSize_8u_C1MR_Ctx(roi, size, context)
                    : nppiMeanGetBufferHostSize_8u_C1MR(roi, size);
    }
    return useCtx ? nppiMeanGetBufferHostSize_8u_C3CMR_Ctx(roi, size, context)
                  : nppiMeanGetBufferHostSize_8u_C3CMR(roi, size);
  }
  static NppStatus compute(MaskedLayout layout, const Npp8u *source, int sourceStep, const Npp8u *mask,
                           int maskStep, NppiSize roi, int coi, Npp8u *buffer, Npp64f *mean,
                           NppStreamContext context, bool useCtx) {
    if (layout == MaskedLayout::C1MR) {
      return useCtx ? nppiMean_8u_C1MR_Ctx(source, sourceStep, mask, maskStep, roi, buffer, mean, context)
                    : nppiMean_8u_C1MR(source, sourceStep, mask, maskStep, roi, buffer, mean);
    }
    return useCtx ? nppiMean_8u_C3CMR_Ctx(source, sourceStep, mask, maskStep, roi, coi, buffer, mean, context)
                  : nppiMean_8u_C3CMR(source, sourceStep, mask, maskStep, roi, coi, buffer, mean);
  }
};

struct Mean8sApi {
  using Value = Npp8s;
  static NppStatus buffer(MaskedLayout layout, NppiSize roi, int *size, NppStreamContext context, bool useCtx) {
    if (layout == MaskedLayout::C1MR) {
      return useCtx ? nppiMeanGetBufferHostSize_8s_C1MR_Ctx(roi, size, context)
                    : nppiMeanGetBufferHostSize_8s_C1MR(roi, size);
    }
    return useCtx ? nppiMeanGetBufferHostSize_8s_C3CMR_Ctx(roi, size, context)
                  : nppiMeanGetBufferHostSize_8s_C3CMR(roi, size);
  }
  static NppStatus compute(MaskedLayout layout, const Npp8s *source, int sourceStep, const Npp8u *mask,
                           int maskStep, NppiSize roi, int coi, Npp8u *buffer, Npp64f *mean,
                           NppStreamContext context, bool useCtx) {
    if (layout == MaskedLayout::C1MR) {
      return useCtx ? nppiMean_8s_C1MR_Ctx(source, sourceStep, mask, maskStep, roi, buffer, mean, context)
                    : nppiMean_8s_C1MR(source, sourceStep, mask, maskStep, roi, buffer, mean);
    }
    return useCtx ? nppiMean_8s_C3CMR_Ctx(source, sourceStep, mask, maskStep, roi, coi, buffer, mean, context)
                  : nppiMean_8s_C3CMR(source, sourceStep, mask, maskStep, roi, coi, buffer, mean);
  }
};

struct Mean16uApi {
  using Value = Npp16u;
  static NppStatus buffer(MaskedLayout layout, NppiSize roi, int *size, NppStreamContext context, bool useCtx) {
    if (layout == MaskedLayout::C1MR) {
      return useCtx ? nppiMeanGetBufferHostSize_16u_C1MR_Ctx(roi, size, context)
                    : nppiMeanGetBufferHostSize_16u_C1MR(roi, size);
    }
    return useCtx ? nppiMeanGetBufferHostSize_16u_C3CMR_Ctx(roi, size, context)
                  : nppiMeanGetBufferHostSize_16u_C3CMR(roi, size);
  }
  static NppStatus compute(MaskedLayout layout, const Npp16u *source, int sourceStep, const Npp8u *mask,
                           int maskStep, NppiSize roi, int coi, Npp8u *buffer, Npp64f *mean,
                           NppStreamContext context, bool useCtx) {
    if (layout == MaskedLayout::C1MR) {
      return useCtx ? nppiMean_16u_C1MR_Ctx(source, sourceStep, mask, maskStep, roi, buffer, mean, context)
                    : nppiMean_16u_C1MR(source, sourceStep, mask, maskStep, roi, buffer, mean);
    }
    return useCtx ? nppiMean_16u_C3CMR_Ctx(source, sourceStep, mask, maskStep, roi, coi, buffer, mean, context)
                  : nppiMean_16u_C3CMR(source, sourceStep, mask, maskStep, roi, coi, buffer, mean);
  }
};

struct Mean32fApi {
  using Value = Npp32f;
  static NppStatus buffer(MaskedLayout layout, NppiSize roi, int *size, NppStreamContext context, bool useCtx) {
    if (layout == MaskedLayout::C1MR) {
      return useCtx ? nppiMeanGetBufferHostSize_32f_C1MR_Ctx(roi, size, context)
                    : nppiMeanGetBufferHostSize_32f_C1MR(roi, size);
    }
    return useCtx ? nppiMeanGetBufferHostSize_32f_C3CMR_Ctx(roi, size, context)
                  : nppiMeanGetBufferHostSize_32f_C3CMR(roi, size);
  }
  static NppStatus compute(MaskedLayout layout, const Npp32f *source, int sourceStep, const Npp8u *mask,
                           int maskStep, NppiSize roi, int coi, Npp8u *buffer, Npp64f *mean,
                           NppStreamContext context, bool useCtx) {
    if (layout == MaskedLayout::C1MR) {
      return useCtx ? nppiMean_32f_C1MR_Ctx(source, sourceStep, mask, maskStep, roi, buffer, mean, context)
                    : nppiMean_32f_C1MR(source, sourceStep, mask, maskStep, roi, buffer, mean);
    }
    return useCtx ? nppiMean_32f_C3CMR_Ctx(source, sourceStep, mask, maskStep, roi, coi, buffer, mean, context)
                  : nppiMean_32f_C3CMR(source, sourceStep, mask, maskStep, roi, coi, buffer, mean);
  }
};

template <typename T> T sourceValue(int x, int y, int channel);
template <> Npp8u sourceValue<Npp8u>(int x, int y, int channel) {
  return static_cast<Npp8u>((x * 17 + y * 31 + channel * 53 + 7) % 256);
}
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
void runMaskedCase(MaskedLayout layout, int coi, bool emptyMask, double tolerance) {
  using T = typename Api::Value;
  const int width = 19;
  const int height = 7;
  const int channels = layout == MaskedLayout::C1MR ? 1 : 3;
  const NppiSize roi{width, height};
  const int hostSourceStep = width * channels * static_cast<int>(sizeof(T));
  std::vector<T> source(static_cast<size_t>(width) * height * channels);
  std::vector<Npp8u> mask(static_cast<size_t>(width) * height);
  double expected = 0.0;
  int selectedCount = 0;
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      const bool selected = !emptyMask && ((x + 2 * y) % 3 != 0);
      mask[static_cast<size_t>(y) * width + x] = selected ? static_cast<Npp8u>(1 + ((x + y) % 3) * 127) : 0;
      for (int channel = 0; channel < channels; ++channel) {
        const T value = sourceValue<T>(x, y, channel);
        source[(static_cast<size_t>(y) * width + x) * channels + channel] = value;
        if (selected && channel == coi - 1) {
          expected += static_cast<double>(value);
        }
      }
      selectedCount += selected ? 1 : 0;
    }
  }
  if (selectedCount > 0) {
    expected /= selectedCount;
  }

  T *deviceSource = nullptr;
  Npp8u *deviceMask = nullptr;
  size_t sourceStep = 0;
  size_t maskStep = 0;
  ASSERT_EQ(cudaMallocPitch(&deviceSource, &sourceStep, hostSourceStep, height), cudaSuccess);
  ASSERT_EQ(cudaMallocPitch(&deviceMask, &maskStep, width, height), cudaSuccess);
  ASSERT_EQ(cudaMemcpy2D(deviceSource, sourceStep, source.data(), hostSourceStep, hostSourceStep, height,
                         cudaMemcpyHostToDevice),
            cudaSuccess);
  ASSERT_EQ(cudaMemcpy2D(deviceMask, maskStep, mask.data(), width, width, height, cudaMemcpyHostToDevice),
            cudaSuccess);

  NppStreamContext context{};
  ASSERT_EQ(nppGetStreamContext(&context), NPP_SUCCESS);
  int bufferSize = 0;
  int contextBufferSize = 0;
  ASSERT_EQ(Api::buffer(layout, roi, &bufferSize, context, false), NPP_SUCCESS);
  ASSERT_EQ(Api::buffer(layout, roi, &contextBufferSize, context, true), NPP_SUCCESS);
  ASSERT_EQ(contextBufferSize, bufferSize);
  ASSERT_GT(bufferSize, 0);
  Npp8u *deviceBuffer = nullptr;
  Npp64f *deviceMean = nullptr;
  ASSERT_EQ(cudaMalloc(&deviceBuffer, bufferSize), cudaSuccess);
  ASSERT_EQ(cudaMalloc(&deviceMean, sizeof(Npp64f)), cudaSuccess);

  for (bool useCtx : {false, true}) {
    ASSERT_EQ(Api::compute(layout, deviceSource, static_cast<int>(sourceStep), deviceMask,
                           static_cast<int>(maskStep), roi, coi, deviceBuffer, deviceMean, context, useCtx),
              NPP_SUCCESS);
    Npp64f actual = 0.0;
    ASSERT_EQ(cudaMemcpy(&actual, deviceMean, sizeof(actual), cudaMemcpyDeviceToHost), cudaSuccess);
    EXPECT_NEAR(actual, expected, tolerance);
  }

  EXPECT_EQ(Api::compute(layout, nullptr, static_cast<int>(sourceStep), deviceMask, static_cast<int>(maskStep), roi,
                         coi, deviceBuffer, deviceMean, context, false),
            NPP_NULL_POINTER_ERROR);
  EXPECT_EQ(Api::compute(layout, deviceSource, static_cast<int>(sourceStep), nullptr, static_cast<int>(maskStep), roi,
                         coi, deviceBuffer, deviceMean, context, false),
            NPP_NULL_POINTER_ERROR);
  EXPECT_EQ(Api::compute(layout, deviceSource, hostSourceStep - 1, deviceMask, static_cast<int>(maskStep), roi, coi,
                         deviceBuffer, deviceMean, context, false),
            NPP_STEP_ERROR);
  EXPECT_EQ(Api::compute(layout, deviceSource, static_cast<int>(sourceStep), deviceMask, width - 1, roi, coi,
                         deviceBuffer, deviceMean, context, true),
            NPP_STEP_ERROR);
  EXPECT_EQ(Api::compute(layout, deviceSource, static_cast<int>(sourceStep), deviceMask,
                         static_cast<int>(maskStep), {0, height}, coi, deviceBuffer, deviceMean, context, false),
            NPP_SIZE_ERROR);
  EXPECT_EQ(Api::compute(layout, deviceSource, static_cast<int>(sourceStep), deviceMask,
                         static_cast<int>(maskStep), roi, coi, nullptr, deviceMean, context, false),
            NPP_NULL_POINTER_ERROR);
  EXPECT_EQ(Api::compute(layout, deviceSource, static_cast<int>(sourceStep), deviceMask,
                         static_cast<int>(maskStep), roi, coi, deviceBuffer, nullptr, context, true),
            NPP_NULL_POINTER_ERROR);
  EXPECT_EQ(Api::buffer(layout, roi, nullptr, context, false), NPP_NULL_POINTER_ERROR);
  EXPECT_EQ(Api::buffer(layout, roi, nullptr, context, true), NPP_NULL_POINTER_ERROR);
  if (layout == MaskedLayout::C3CMR) {
    EXPECT_EQ(Api::compute(layout, deviceSource, static_cast<int>(sourceStep), deviceMask,
                           static_cast<int>(maskStep), roi, 0, deviceBuffer, deviceMean, context, false),
              NPP_COI_ERROR);
    EXPECT_EQ(Api::compute(layout, deviceSource, static_cast<int>(sourceStep), deviceMask,
                           static_cast<int>(maskStep), roi, 4, deviceBuffer, deviceMean, context, true),
              NPP_COI_ERROR);
  }

  cudaFree(deviceSource);
  cudaFree(deviceMask);
  cudaFree(deviceBuffer);
  cudaFree(deviceMean);
}

template <typename Api> void runMaskedType(double tolerance) {
  runMaskedCase<Api>(MaskedLayout::C1MR, 1, false, tolerance);
  runMaskedCase<Api>(MaskedLayout::C1MR, 1, true, tolerance);
  for (int coi = 1; coi <= 3; ++coi) {
    runMaskedCase<Api>(MaskedLayout::C3CMR, coi, false, tolerance);
  }
  runMaskedCase<Api>(MaskedLayout::C3CMR, 2, true, tolerance);
}

TEST(NppiMeanMaskedTest, Mean_8u_AllLayoutsAndParameters) { runMaskedType<Mean8uApi>(1e-12); }
TEST(NppiMeanMaskedTest, Mean_8s_AllLayoutsAndParameters) { runMaskedType<Mean8sApi>(1e-12); }
TEST(NppiMeanMaskedTest, Mean_16u_AllLayoutsAndParameters) { runMaskedType<Mean16uApi>(1e-12); }
TEST(NppiMeanMaskedTest, Mean_32f_AllLayoutsAndParameters) { runMaskedType<Mean32fApi>(1e-10); }

} // namespace
