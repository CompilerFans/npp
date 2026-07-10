#include "npp.h"

#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <type_traits>
#include <vector>

namespace {

enum class BoxLayout { C1, C3, C4, AC4 };

struct FilterBox8uApi {
  using Value = Npp8u;
  static NppStatus call(BoxLayout layout, const Npp8u *src, int srcStep, Npp8u *dst, int dstStep, NppiSize roi,
                        NppiSize mask, NppiPoint anchor, NppStreamContext context, bool useCtx) {
    switch (layout) {
    case BoxLayout::C1:
      return useCtx ? nppiFilterBox_8u_C1R_Ctx(src, srcStep, dst, dstStep, roi, mask, anchor, context)
                    : nppiFilterBox_8u_C1R(src, srcStep, dst, dstStep, roi, mask, anchor);
    case BoxLayout::C3:
      return useCtx ? nppiFilterBox_8u_C3R_Ctx(src, srcStep, dst, dstStep, roi, mask, anchor, context)
                    : nppiFilterBox_8u_C3R(src, srcStep, dst, dstStep, roi, mask, anchor);
    case BoxLayout::C4:
      return useCtx ? nppiFilterBox_8u_C4R_Ctx(src, srcStep, dst, dstStep, roi, mask, anchor, context)
                    : nppiFilterBox_8u_C4R(src, srcStep, dst, dstStep, roi, mask, anchor);
    case BoxLayout::AC4:
      return useCtx ? nppiFilterBox_8u_AC4R_Ctx(src, srcStep, dst, dstStep, roi, mask, anchor, context)
                    : nppiFilterBox_8u_AC4R(src, srcStep, dst, dstStep, roi, mask, anchor);
    }
    return NPP_BAD_ARGUMENT_ERROR;
  }
};

struct FilterBox16uApi {
  using Value = Npp16u;
  static NppStatus call(BoxLayout layout, const Npp16u *src, int srcStep, Npp16u *dst, int dstStep, NppiSize roi,
                        NppiSize mask, NppiPoint anchor, NppStreamContext context, bool useCtx) {
    switch (layout) {
    case BoxLayout::C1:
      return useCtx ? nppiFilterBox_16u_C1R_Ctx(src, srcStep, dst, dstStep, roi, mask, anchor, context)
                    : nppiFilterBox_16u_C1R(src, srcStep, dst, dstStep, roi, mask, anchor);
    case BoxLayout::C3:
      return useCtx ? nppiFilterBox_16u_C3R_Ctx(src, srcStep, dst, dstStep, roi, mask, anchor, context)
                    : nppiFilterBox_16u_C3R(src, srcStep, dst, dstStep, roi, mask, anchor);
    case BoxLayout::C4:
      return useCtx ? nppiFilterBox_16u_C4R_Ctx(src, srcStep, dst, dstStep, roi, mask, anchor, context)
                    : nppiFilterBox_16u_C4R(src, srcStep, dst, dstStep, roi, mask, anchor);
    case BoxLayout::AC4:
      return useCtx ? nppiFilterBox_16u_AC4R_Ctx(src, srcStep, dst, dstStep, roi, mask, anchor, context)
                    : nppiFilterBox_16u_AC4R(src, srcStep, dst, dstStep, roi, mask, anchor);
    }
    return NPP_BAD_ARGUMENT_ERROR;
  }
};

struct FilterBox16sApi {
  using Value = Npp16s;
  static NppStatus call(BoxLayout layout, const Npp16s *src, int srcStep, Npp16s *dst, int dstStep, NppiSize roi,
                        NppiSize mask, NppiPoint anchor, NppStreamContext context, bool useCtx) {
    switch (layout) {
    case BoxLayout::C1:
      return useCtx ? nppiFilterBox_16s_C1R_Ctx(src, srcStep, dst, dstStep, roi, mask, anchor, context)
                    : nppiFilterBox_16s_C1R(src, srcStep, dst, dstStep, roi, mask, anchor);
    case BoxLayout::C3:
      return useCtx ? nppiFilterBox_16s_C3R_Ctx(src, srcStep, dst, dstStep, roi, mask, anchor, context)
                    : nppiFilterBox_16s_C3R(src, srcStep, dst, dstStep, roi, mask, anchor);
    case BoxLayout::C4:
      return useCtx ? nppiFilterBox_16s_C4R_Ctx(src, srcStep, dst, dstStep, roi, mask, anchor, context)
                    : nppiFilterBox_16s_C4R(src, srcStep, dst, dstStep, roi, mask, anchor);
    case BoxLayout::AC4:
      return useCtx ? nppiFilterBox_16s_AC4R_Ctx(src, srcStep, dst, dstStep, roi, mask, anchor, context)
                    : nppiFilterBox_16s_AC4R(src, srcStep, dst, dstStep, roi, mask, anchor);
    }
    return NPP_BAD_ARGUMENT_ERROR;
  }
};

struct FilterBox32fApi {
  using Value = Npp32f;
  static NppStatus call(BoxLayout layout, const Npp32f *src, int srcStep, Npp32f *dst, int dstStep, NppiSize roi,
                        NppiSize mask, NppiPoint anchor, NppStreamContext context, bool useCtx) {
    switch (layout) {
    case BoxLayout::C1:
      return useCtx ? nppiFilterBox_32f_C1R_Ctx(src, srcStep, dst, dstStep, roi, mask, anchor, context)
                    : nppiFilterBox_32f_C1R(src, srcStep, dst, dstStep, roi, mask, anchor);
    case BoxLayout::C3:
      return useCtx ? nppiFilterBox_32f_C3R_Ctx(src, srcStep, dst, dstStep, roi, mask, anchor, context)
                    : nppiFilterBox_32f_C3R(src, srcStep, dst, dstStep, roi, mask, anchor);
    case BoxLayout::C4:
      return useCtx ? nppiFilterBox_32f_C4R_Ctx(src, srcStep, dst, dstStep, roi, mask, anchor, context)
                    : nppiFilterBox_32f_C4R(src, srcStep, dst, dstStep, roi, mask, anchor);
    case BoxLayout::AC4:
      return useCtx ? nppiFilterBox_32f_AC4R_Ctx(src, srcStep, dst, dstStep, roi, mask, anchor, context)
                    : nppiFilterBox_32f_AC4R(src, srcStep, dst, dstStep, roi, mask, anchor);
    }
    return NPP_BAD_ARGUMENT_ERROR;
  }
};

struct FilterBox64fApi {
  using Value = Npp64f;
  static NppStatus call(BoxLayout, const Npp64f *src, int srcStep, Npp64f *dst, int dstStep, NppiSize roi,
                        NppiSize mask, NppiPoint anchor, NppStreamContext context, bool useCtx) {
    return useCtx ? nppiFilterBox_64f_C1R_Ctx(src, srcStep, dst, dstStep, roi, mask, anchor, context)
                  : nppiFilterBox_64f_C1R(src, srcStep, dst, dstStep, roi, mask, anchor);
  }
};

template <typename T> T sourceValue(int x, int y, int channel);
template <> Npp8u sourceValue<Npp8u>(int x, int y, int channel) {
  return static_cast<Npp8u>(40 + x * x * 2 + y * y * 3 + x * y + channel * 13);
}
template <> Npp16u sourceValue<Npp16u>(int x, int y, int channel) {
  return static_cast<Npp16u>(1000 + x * x * 17 + y * y * 29 + x * y * 7 + channel * 197);
}
template <> Npp16s sourceValue<Npp16s>(int x, int y, int channel) {
  return static_cast<Npp16s>(-2000 + x * x * 13 + y * y * 19 + x * y * 5 + channel * 197);
}
template <> Npp32f sourceValue<Npp32f>(int x, int y, int channel) {
  return static_cast<Npp32f>(-4.5f + x * x * 0.17f + y * y * 0.23f + x * y * 0.11f + channel * 2.25f);
}
template <> Npp64f sourceValue<Npp64f>(int x, int y, int channel) {
  return -4.5 + x * x * 0.17 + y * y * 0.23 + x * y * 0.11 + channel * 2.25;
}

template <typename T> T averagedValue(double value) { return static_cast<T>(value); }

template <typename Api>
void runBoxCase(BoxLayout layout, double tolerance, NppiSize maskSize, NppiPoint anchor) {
  using T = typename Api::Value;
  const int width = 5;
  const int height = 4;
  const int channels = layout == BoxLayout::C1 ? 1 : layout == BoxLayout::C3 ? 3 : 4;
  const int filteredChannels = layout == BoxLayout::AC4 ? 3 : channels;
  const NppiSize roi{width, height};
  const int haloLeft = anchor.x;
  const int haloTop = anchor.y;
  const int expandedWidth = width + maskSize.width - 1;
  const int expandedHeight = height + maskSize.height - 1;
  const int sourceRowBytes = expandedWidth * channels * static_cast<int>(sizeof(T));
  const int destinationRowBytes = width * channels * static_cast<int>(sizeof(T));
  std::vector<T> source(static_cast<size_t>(expandedWidth) * expandedHeight * channels);
  std::vector<T> initialDestination(static_cast<size_t>(width) * height * channels, static_cast<T>(73));
  for (int y = -haloTop; y < height + maskSize.height - anchor.y - 1; ++y) {
    for (int x = -haloLeft; x < width + maskSize.width - anchor.x - 1; ++x) {
      for (int channel = 0; channel < channels; ++channel) {
        source[(static_cast<size_t>(y + haloTop) * expandedWidth + x + haloLeft) * channels + channel] =
            sourceValue<T>(x, y, channel);
      }
    }
  }

  T *deviceSourceBase = nullptr;
  T *deviceDestination = nullptr;
  size_t sourceStep = 0;
  size_t destinationStep = 0;
  ASSERT_EQ(cudaMallocPitch(&deviceSourceBase, &sourceStep, sourceRowBytes, expandedHeight), cudaSuccess);
  ASSERT_EQ(cudaMallocPitch(&deviceDestination, &destinationStep, destinationRowBytes, height), cudaSuccess);
  ASSERT_EQ(cudaMemcpy2D(deviceSourceBase, sourceStep, source.data(), sourceRowBytes, sourceRowBytes, expandedHeight,
                         cudaMemcpyHostToDevice),
            cudaSuccess);
  T *deviceSource = reinterpret_cast<T *>(reinterpret_cast<char *>(deviceSourceBase) + haloTop * sourceStep) +
                    haloLeft * channels;
  const cudaStream_t originalStream = nppGetStream();
  cudaStream_t managedStream = nullptr;
  cudaStream_t contextStream = nullptr;
  ASSERT_EQ(cudaStreamCreate(&managedStream), cudaSuccess);
  ASSERT_EQ(cudaStreamCreate(&contextStream), cudaSuccess);
  ASSERT_EQ(nppSetStream(managedStream), NPP_SUCCESS);
  NppStreamContext context{};
  ASSERT_EQ(nppGetStreamContext(&context), NPP_SUCCESS);
  ASSERT_EQ(context.hStream, managedStream);
  context.hStream = contextStream;

  for (bool useCtx : {false, true}) {
    ASSERT_EQ(cudaMemcpy2D(deviceDestination, destinationStep, initialDestination.data(), destinationRowBytes,
                           destinationRowBytes, height, cudaMemcpyHostToDevice),
              cudaSuccess);
    ASSERT_EQ(Api::call(layout, deviceSource, static_cast<int>(sourceStep), deviceDestination,
                        static_cast<int>(destinationStep), roi, maskSize, anchor, context, useCtx),
              NPP_SUCCESS);
    ASSERT_EQ(cudaStreamSynchronize(useCtx ? contextStream : managedStream), cudaSuccess);
    std::vector<T> actual(initialDestination.size());
    ASSERT_EQ(cudaMemcpy2D(actual.data(), destinationRowBytes, deviceDestination, destinationStep,
                           destinationRowBytes, height, cudaMemcpyDeviceToHost),
              cudaSuccess);
    for (int y = 0; y < height; ++y) {
      for (int x = 0; x < width; ++x) {
        for (int channel = 0; channel < channels; ++channel) {
          const size_t index = (static_cast<size_t>(y) * width + x) * channels + channel;
          if (channel >= filteredChannels) {
            EXPECT_EQ(actual[index], initialDestination[index]);
            continue;
          }
          double sum = 0.0;
          for (int maskY = 0; maskY < maskSize.height; ++maskY) {
            for (int maskX = 0; maskX < maskSize.width; ++maskX) {
              const int sourceX = x + maskX;
              const int sourceY = y + maskY;
              sum += static_cast<double>(
                  source[(static_cast<size_t>(sourceY) * expandedWidth + sourceX) * channels + channel]);
            }
          }
          const T expected = averagedValue<T>(sum / (maskSize.width * maskSize.height));
          if (std::is_floating_point<T>::value) {
            EXPECT_NEAR(actual[index], expected, tolerance);
          } else {
            EXPECT_EQ(actual[index], expected);
          }
        }
      }
    }
  }

  EXPECT_EQ(Api::call(layout, nullptr, static_cast<int>(sourceStep), deviceDestination,
                      static_cast<int>(destinationStep), roi, maskSize, anchor, context, false),
            NPP_NULL_POINTER_ERROR);
  EXPECT_EQ(Api::call(layout, deviceSource, static_cast<int>(sourceStep), nullptr,
                      static_cast<int>(destinationStep), roi, maskSize, anchor, context, true),
            NPP_NULL_POINTER_ERROR);
  EXPECT_EQ(Api::call(layout, deviceSource, destinationRowBytes - 1, deviceDestination,
                      static_cast<int>(destinationStep), roi, maskSize, anchor, context, false),
            NPP_STEP_ERROR);
  EXPECT_EQ(Api::call(layout, deviceSource, static_cast<int>(sourceStep), deviceDestination,
                      destinationRowBytes - 1, roi, maskSize, anchor, context, true),
            NPP_STEP_ERROR);
  EXPECT_EQ(Api::call(layout, deviceSource, static_cast<int>(sourceStep), deviceDestination,
                      static_cast<int>(destinationStep), {0, height}, maskSize, anchor, context, true),
            NPP_SIZE_ERROR);
  EXPECT_EQ(Api::call(layout, deviceSource, static_cast<int>(sourceStep), deviceDestination,
                      static_cast<int>(destinationStep), {width, -1}, maskSize, anchor, context, false),
            NPP_SIZE_ERROR);
  EXPECT_EQ(Api::call(layout, deviceSource, static_cast<int>(sourceStep), deviceDestination,
                      static_cast<int>(destinationStep), roi, {0, 3}, anchor, context, false),
            NPP_MASK_SIZE_ERROR);
  EXPECT_EQ(Api::call(layout, deviceSource, static_cast<int>(sourceStep), deviceDestination,
                      static_cast<int>(destinationStep), roi, {3, -1}, anchor, context, true),
            NPP_MASK_SIZE_ERROR);
  EXPECT_EQ(Api::call(layout, deviceSource, static_cast<int>(sourceStep), deviceDestination,
                      static_cast<int>(destinationStep), roi, maskSize, {maskSize.width, anchor.y}, context, true),
            NPP_ANCHOR_ERROR);
  EXPECT_EQ(Api::call(layout, deviceSource, static_cast<int>(sourceStep), deviceDestination,
                      static_cast<int>(destinationStep), roi, maskSize, {-1, anchor.y}, context, false),
            NPP_ANCHOR_ERROR);

  EXPECT_EQ(nppSetStream(originalStream), NPP_SUCCESS);
  cudaStreamDestroy(contextStream);
  cudaStreamDestroy(managedStream);
  cudaFree(deviceSourceBase);
  cudaFree(deviceDestination);
}

template <typename Api> void runAllLayouts(double tolerance) {
  for (BoxLayout layout : {BoxLayout::C1, BoxLayout::C3, BoxLayout::C4, BoxLayout::AC4}) {
    runBoxCase<Api>(layout, tolerance, {3, 3}, {1, 1});
    runBoxCase<Api>(layout, tolerance, {4, 2}, {0, 1});
  }
}

TEST(NppiFilterBoxAllTypesTest, FilterBox_8u_AllLayoutsAndParameters) { runAllLayouts<FilterBox8uApi>(0.0); }
TEST(NppiFilterBoxAllTypesTest, FilterBox_16u_AllLayoutsAndParameters) { runAllLayouts<FilterBox16uApi>(0.0); }
TEST(NppiFilterBoxAllTypesTest, FilterBox_16s_AllLayoutsAndParameters) { runAllLayouts<FilterBox16sApi>(0.0); }
TEST(NppiFilterBoxAllTypesTest, FilterBox_32f_AllLayoutsAndParameters) { runAllLayouts<FilterBox32fApi>(1e-6); }
TEST(NppiFilterBoxAllTypesTest, FilterBox_64f_C1AndParameters) {
  runBoxCase<FilterBox64fApi>(BoxLayout::C1, 1e-12, {3, 3}, {1, 1});
  runBoxCase<FilterBox64fApi>(BoxLayout::C1, 1e-12, {4, 2}, {0, 1});
}

} // namespace
