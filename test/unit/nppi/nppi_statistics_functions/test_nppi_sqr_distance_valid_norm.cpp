#include "npp.h"

#include <algorithm>
#include <cmath>
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <limits>
#include <type_traits>
#include <vector>

namespace {

enum class ProximityLayout { C1, C3, C4, AC4 };

int channelCount(ProximityLayout layout) {
  return layout == ProximityLayout::C1 ? 1 : layout == ProximityLayout::C3 ? 3 : 4;
}

int activeChannelCount(ProximityLayout layout) {
  return layout == ProximityLayout::AC4 ? 3 : channelCount(layout);
}

struct SqrDistance8uApi {
  using Src = Npp8u;
  using Dst = Npp8u;

  static NppStatus call(ProximityLayout layout, const Src *src, int srcStep, NppiSize srcSize, const Src *tpl,
                        int tplStep, NppiSize tplSize, Dst *dst, int dstStep, int scaleFactor,
                        NppStreamContext context, bool useContext) {
    switch (layout) {
    case ProximityLayout::C1:
      return useContext ? nppiSqrDistanceValid_Norm_8u_C1RSfs_Ctx(src, srcStep, srcSize, tpl, tplStep, tplSize, dst,
                                                                  dstStep, scaleFactor, context)
                        : nppiSqrDistanceValid_Norm_8u_C1RSfs(src, srcStep, srcSize, tpl, tplStep, tplSize, dst,
                                                              dstStep, scaleFactor);
    case ProximityLayout::C3:
      return useContext ? nppiSqrDistanceValid_Norm_8u_C3RSfs_Ctx(src, srcStep, srcSize, tpl, tplStep, tplSize, dst,
                                                                  dstStep, scaleFactor, context)
                        : nppiSqrDistanceValid_Norm_8u_C3RSfs(src, srcStep, srcSize, tpl, tplStep, tplSize, dst,
                                                              dstStep, scaleFactor);
    case ProximityLayout::C4:
      return useContext ? nppiSqrDistanceValid_Norm_8u_C4RSfs_Ctx(src, srcStep, srcSize, tpl, tplStep, tplSize, dst,
                                                                  dstStep, scaleFactor, context)
                        : nppiSqrDistanceValid_Norm_8u_C4RSfs(src, srcStep, srcSize, tpl, tplStep, tplSize, dst,
                                                              dstStep, scaleFactor);
    case ProximityLayout::AC4:
      return useContext ? nppiSqrDistanceValid_Norm_8u_AC4RSfs_Ctx(src, srcStep, srcSize, tpl, tplStep, tplSize, dst,
                                                                   dstStep, scaleFactor, context)
                        : nppiSqrDistanceValid_Norm_8u_AC4RSfs(src, srcStep, srcSize, tpl, tplStep, tplSize, dst,
                                                               dstStep, scaleFactor);
    }
    return NPP_BAD_ARGUMENT_ERROR;
  }
};

struct SqrDistance8u32fApi {
  using Src = Npp8u;
  using Dst = Npp32f;

  static NppStatus call(ProximityLayout layout, const Src *src, int srcStep, NppiSize srcSize, const Src *tpl,
                        int tplStep, NppiSize tplSize, Dst *dst, int dstStep, int, NppStreamContext context,
                        bool useContext) {
    switch (layout) {
    case ProximityLayout::C1:
      return useContext ? nppiSqrDistanceValid_Norm_8u32f_C1R_Ctx(src, srcStep, srcSize, tpl, tplStep, tplSize, dst,
                                                                  dstStep, context)
                        : nppiSqrDistanceValid_Norm_8u32f_C1R(src, srcStep, srcSize, tpl, tplStep, tplSize, dst,
                                                              dstStep);
    case ProximityLayout::C3:
      return useContext ? nppiSqrDistanceValid_Norm_8u32f_C3R_Ctx(src, srcStep, srcSize, tpl, tplStep, tplSize, dst,
                                                                  dstStep, context)
                        : nppiSqrDistanceValid_Norm_8u32f_C3R(src, srcStep, srcSize, tpl, tplStep, tplSize, dst,
                                                              dstStep);
    case ProximityLayout::C4:
      return useContext ? nppiSqrDistanceValid_Norm_8u32f_C4R_Ctx(src, srcStep, srcSize, tpl, tplStep, tplSize, dst,
                                                                  dstStep, context)
                        : nppiSqrDistanceValid_Norm_8u32f_C4R(src, srcStep, srcSize, tpl, tplStep, tplSize, dst,
                                                              dstStep);
    case ProximityLayout::AC4:
      return useContext ? nppiSqrDistanceValid_Norm_8u32f_AC4R_Ctx(src, srcStep, srcSize, tpl, tplStep, tplSize, dst,
                                                                   dstStep, context)
                        : nppiSqrDistanceValid_Norm_8u32f_AC4R(src, srcStep, srcSize, tpl, tplStep, tplSize, dst,
                                                               dstStep);
    }
    return NPP_BAD_ARGUMENT_ERROR;
  }
};

struct SqrDistance8s32fApi {
  using Src = Npp8s;
  using Dst = Npp32f;

  static NppStatus call(ProximityLayout layout, const Src *src, int srcStep, NppiSize srcSize, const Src *tpl,
                        int tplStep, NppiSize tplSize, Dst *dst, int dstStep, int, NppStreamContext context,
                        bool useContext) {
    switch (layout) {
    case ProximityLayout::C1:
      return useContext ? nppiSqrDistanceValid_Norm_8s32f_C1R_Ctx(src, srcStep, srcSize, tpl, tplStep, tplSize, dst,
                                                                  dstStep, context)
                        : nppiSqrDistanceValid_Norm_8s32f_C1R(src, srcStep, srcSize, tpl, tplStep, tplSize, dst,
                                                              dstStep);
    case ProximityLayout::C3:
      return useContext ? nppiSqrDistanceValid_Norm_8s32f_C3R_Ctx(src, srcStep, srcSize, tpl, tplStep, tplSize, dst,
                                                                  dstStep, context)
                        : nppiSqrDistanceValid_Norm_8s32f_C3R(src, srcStep, srcSize, tpl, tplStep, tplSize, dst,
                                                              dstStep);
    case ProximityLayout::C4:
      return useContext ? nppiSqrDistanceValid_Norm_8s32f_C4R_Ctx(src, srcStep, srcSize, tpl, tplStep, tplSize, dst,
                                                                  dstStep, context)
                        : nppiSqrDistanceValid_Norm_8s32f_C4R(src, srcStep, srcSize, tpl, tplStep, tplSize, dst,
                                                              dstStep);
    case ProximityLayout::AC4:
      return useContext ? nppiSqrDistanceValid_Norm_8s32f_AC4R_Ctx(src, srcStep, srcSize, tpl, tplStep, tplSize, dst,
                                                                   dstStep, context)
                        : nppiSqrDistanceValid_Norm_8s32f_AC4R(src, srcStep, srcSize, tpl, tplStep, tplSize, dst,
                                                               dstStep);
    }
    return NPP_BAD_ARGUMENT_ERROR;
  }
};

struct SqrDistance16u32fApi {
  using Src = Npp16u;
  using Dst = Npp32f;

  static NppStatus call(ProximityLayout layout, const Src *src, int srcStep, NppiSize srcSize, const Src *tpl,
                        int tplStep, NppiSize tplSize, Dst *dst, int dstStep, int, NppStreamContext context,
                        bool useContext) {
    switch (layout) {
    case ProximityLayout::C1:
      return useContext ? nppiSqrDistanceValid_Norm_16u32f_C1R_Ctx(src, srcStep, srcSize, tpl, tplStep, tplSize, dst,
                                                                   dstStep, context)
                        : nppiSqrDistanceValid_Norm_16u32f_C1R(src, srcStep, srcSize, tpl, tplStep, tplSize, dst,
                                                               dstStep);
    case ProximityLayout::C3:
      return useContext ? nppiSqrDistanceValid_Norm_16u32f_C3R_Ctx(src, srcStep, srcSize, tpl, tplStep, tplSize, dst,
                                                                   dstStep, context)
                        : nppiSqrDistanceValid_Norm_16u32f_C3R(src, srcStep, srcSize, tpl, tplStep, tplSize, dst,
                                                               dstStep);
    case ProximityLayout::C4:
      return useContext ? nppiSqrDistanceValid_Norm_16u32f_C4R_Ctx(src, srcStep, srcSize, tpl, tplStep, tplSize, dst,
                                                                   dstStep, context)
                        : nppiSqrDistanceValid_Norm_16u32f_C4R(src, srcStep, srcSize, tpl, tplStep, tplSize, dst,
                                                               dstStep);
    case ProximityLayout::AC4:
      return useContext ? nppiSqrDistanceValid_Norm_16u32f_AC4R_Ctx(src, srcStep, srcSize, tpl, tplStep, tplSize, dst,
                                                                    dstStep, context)
                        : nppiSqrDistanceValid_Norm_16u32f_AC4R(src, srcStep, srcSize, tpl, tplStep, tplSize, dst,
                                                                dstStep);
    }
    return NPP_BAD_ARGUMENT_ERROR;
  }
};

struct SqrDistance32fApi {
  using Src = Npp32f;
  using Dst = Npp32f;

  static NppStatus call(ProximityLayout layout, const Src *src, int srcStep, NppiSize srcSize, const Src *tpl,
                        int tplStep, NppiSize tplSize, Dst *dst, int dstStep, int, NppStreamContext context,
                        bool useContext) {
    switch (layout) {
    case ProximityLayout::C1:
      return useContext ? nppiSqrDistanceValid_Norm_32f_C1R_Ctx(src, srcStep, srcSize, tpl, tplStep, tplSize, dst,
                                                                dstStep, context)
                        : nppiSqrDistanceValid_Norm_32f_C1R(src, srcStep, srcSize, tpl, tplStep, tplSize, dst,
                                                            dstStep);
    case ProximityLayout::C3:
      return useContext ? nppiSqrDistanceValid_Norm_32f_C3R_Ctx(src, srcStep, srcSize, tpl, tplStep, tplSize, dst,
                                                                dstStep, context)
                        : nppiSqrDistanceValid_Norm_32f_C3R(src, srcStep, srcSize, tpl, tplStep, tplSize, dst,
                                                            dstStep);
    case ProximityLayout::C4:
      return useContext ? nppiSqrDistanceValid_Norm_32f_C4R_Ctx(src, srcStep, srcSize, tpl, tplStep, tplSize, dst,
                                                                dstStep, context)
                        : nppiSqrDistanceValid_Norm_32f_C4R(src, srcStep, srcSize, tpl, tplStep, tplSize, dst,
                                                            dstStep);
    case ProximityLayout::AC4:
      return useContext ? nppiSqrDistanceValid_Norm_32f_AC4R_Ctx(src, srcStep, srcSize, tpl, tplStep, tplSize, dst,
                                                                 dstStep, context)
                        : nppiSqrDistanceValid_Norm_32f_AC4R(src, srcStep, srcSize, tpl, tplStep, tplSize, dst,
                                                             dstStep);
    }
    return NPP_BAD_ARGUMENT_ERROR;
  }
};

template <typename T> T inputValue(int x, int y, int channel, bool isTemplate) {
  if constexpr (std::is_same_v<T, Npp8u>) {
    return static_cast<T>(isTemplate ? 7 + x * 5 + y * 9 + channel * 11 : 5 + x * 7 + y * 3 + channel * 13);
  } else if constexpr (std::is_same_v<T, Npp8s>) {
    return static_cast<T>(isTemplate ? -11 + x * 4 + y * 7 + channel * 3 : -9 + x * 5 - y * 2 + channel * 4);
  } else if constexpr (std::is_same_v<T, Npp16u>) {
    return static_cast<T>(isTemplate ? 113 + x * 17 + y * 23 + channel * 31
                                     : 97 + x * 29 + y * 19 + channel * 37);
  } else {
    return static_cast<T>(isTemplate ? -1.75 + x * 0.625 + y * 1.125 + channel * 0.875
                                     : -2.25 + x * 0.75 - y * 0.375 + channel * 1.25);
  }
}

template <typename T> double asDouble(T value) { return static_cast<double>(value); }

template <typename Dst> Dst convertExpected(double value, int scaleFactor) {
  if constexpr (std::is_same_v<Dst, Npp8u>) {
    const double scaled = std::ldexp(value, -scaleFactor);
    return static_cast<Npp8u>(std::clamp(std::floor(scaled + 0.5), 0.0, 255.0));
  } else {
    return static_cast<Dst>(value);
  }
}

template <typename Api> void runAccuracyCase(ProximityLayout layout, bool useContext, int scaleFactor) {
  using Src = typename Api::Src;
  using Dst = typename Api::Dst;
  constexpr int srcWidth = 6;
  constexpr int srcHeight = 5;
  constexpr int tplWidth = 3;
  constexpr int tplHeight = 2;
  constexpr int dstWidth = srcWidth - tplWidth + 1;
  constexpr int dstHeight = srcHeight - tplHeight + 1;
  const int channels = channelCount(layout);
  const int activeChannels = activeChannelCount(layout);
  const int srcStepElements = srcWidth * channels + 5;
  const int tplStepElements = tplWidth * channels + 7;
  const int dstStepElements = dstWidth * channels + 9;
  const int srcStep = srcStepElements * static_cast<int>(sizeof(Src));
  const int tplStep = tplStepElements * static_cast<int>(sizeof(Src));
  const int dstStep = dstStepElements * static_cast<int>(sizeof(Dst));
  const NppiSize srcSize{srcWidth, srcHeight};
  const NppiSize tplSize{tplWidth, tplHeight};

  std::vector<Src> source(static_cast<size_t>(srcStepElements) * srcHeight, Src{});
  std::vector<Src> templ(static_cast<size_t>(tplStepElements) * tplHeight, Src{});
  const Dst sentinel = static_cast<Dst>(73);
  std::vector<Dst> initialDestination(static_cast<size_t>(dstStepElements) * dstHeight, sentinel);
  for (int y = 0; y < srcHeight; ++y) {
    for (int x = 0; x < srcWidth; ++x) {
      for (int channel = 0; channel < channels; ++channel) {
        source[static_cast<size_t>(y) * srcStepElements + x * channels + channel] =
            channel == 3 && layout == ProximityLayout::AC4 ? static_cast<Src>(31 + x + y)
                                                            : inputValue<Src>(x, y, channel, false);
      }
    }
  }
  for (int y = 0; y < tplHeight; ++y) {
    for (int x = 0; x < tplWidth; ++x) {
      for (int channel = 0; channel < channels; ++channel) {
        templ[static_cast<size_t>(y) * tplStepElements + x * channels + channel] =
            channel == 3 && layout == ProximityLayout::AC4 ? static_cast<Src>(101 + x + y)
                                                            : inputValue<Src>(x, y, channel, true);
      }
    }
  }

  Src *deviceSource = nullptr;
  Src *deviceTemplate = nullptr;
  Dst *deviceDestination = nullptr;
  ASSERT_EQ(cudaMalloc(&deviceSource, static_cast<size_t>(srcStep) * srcHeight), cudaSuccess);
  ASSERT_EQ(cudaMalloc(&deviceTemplate, static_cast<size_t>(tplStep) * tplHeight), cudaSuccess);
  ASSERT_EQ(cudaMalloc(&deviceDestination, static_cast<size_t>(dstStep) * dstHeight), cudaSuccess);
  ASSERT_EQ(cudaMemcpy(deviceSource, source.data(), static_cast<size_t>(srcStep) * srcHeight, cudaMemcpyHostToDevice),
            cudaSuccess);
  ASSERT_EQ(cudaMemcpy(deviceTemplate, templ.data(), static_cast<size_t>(tplStep) * tplHeight,
                       cudaMemcpyHostToDevice),
            cudaSuccess);
  ASSERT_EQ(cudaMemcpy(deviceDestination, initialDestination.data(), static_cast<size_t>(dstStep) * dstHeight,
                       cudaMemcpyHostToDevice),
            cudaSuccess);

  const cudaStream_t originalStream = nppGetStream();
  cudaStream_t managedStream = nullptr;
  cudaStream_t contextStream = nullptr;
  ASSERT_EQ(cudaStreamCreate(&managedStream), cudaSuccess);
  ASSERT_EQ(cudaStreamCreate(&contextStream), cudaSuccess);
  ASSERT_EQ(nppSetStream(managedStream), NPP_SUCCESS);
  NppStreamContext context{};
  ASSERT_EQ(nppGetStreamContext(&context), NPP_SUCCESS);
  context.hStream = contextStream;

  ASSERT_EQ(Api::call(layout, deviceSource, srcStep, srcSize, deviceTemplate, tplStep, tplSize, deviceDestination,
                      dstStep, scaleFactor, context, useContext),
            NPP_SUCCESS);
  ASSERT_EQ(cudaStreamSynchronize(useContext ? contextStream : managedStream), cudaSuccess);

  std::vector<Dst> actual(initialDestination.size());
  ASSERT_EQ(cudaMemcpy(actual.data(), deviceDestination, static_cast<size_t>(dstStep) * dstHeight,
                       cudaMemcpyDeviceToHost),
            cudaSuccess);
  for (int y = 0; y < dstHeight; ++y) {
    for (int x = 0; x < dstWidth; ++x) {
      for (int channel = 0; channel < channels; ++channel) {
        const size_t destinationIndex = static_cast<size_t>(y) * dstStepElements + x * channels + channel;
        if (channel >= activeChannels) {
          EXPECT_EQ(actual[destinationIndex], sentinel);
          continue;
        }
        double squaredDistance = 0.0;
        double sourceEnergy = 0.0;
        double templateEnergy = 0.0;
        for (int templateY = 0; templateY < tplHeight; ++templateY) {
          for (int templateX = 0; templateX < tplWidth; ++templateX) {
            const double sourceValue = asDouble(source[static_cast<size_t>(y + templateY) * srcStepElements +
                                                        (x + templateX) * channels + channel]);
            const double templateValue = asDouble(templ[static_cast<size_t>(templateY) * tplStepElements +
                                                        templateX * channels + channel]);
            const double difference = templateValue - sourceValue;
            squaredDistance += difference * difference;
            sourceEnergy += sourceValue * sourceValue;
            templateEnergy += templateValue * templateValue;
          }
        }
        const double denominator = std::sqrt(sourceEnergy * templateEnergy);
        const double expectedValue = denominator == 0.0 ? 0.0 : squaredDistance / denominator;
        const Dst expected = convertExpected<Dst>(expectedValue, scaleFactor);
        if constexpr (std::is_same_v<Dst, Npp8u>) {
          EXPECT_EQ(actual[destinationIndex], expected) << "x=" << x << " y=" << y << " c=" << channel;
        } else {
          EXPECT_NEAR(actual[destinationIndex], expected, 2e-5f)
              << "x=" << x << " y=" << y << " c=" << channel;
        }
      }
    }
  }

  EXPECT_EQ(nppSetStream(originalStream), NPP_SUCCESS);
  EXPECT_EQ(cudaStreamDestroy(contextStream), cudaSuccess);
  EXPECT_EQ(cudaStreamDestroy(managedStream), cudaSuccess);
  EXPECT_EQ(cudaFree(deviceDestination), cudaSuccess);
  EXPECT_EQ(cudaFree(deviceTemplate), cudaSuccess);
  EXPECT_EQ(cudaFree(deviceSource), cudaSuccess);
}

template <typename Api> void runApiMatrix(int scaleFactor = 0) {
  for (const auto layout : {ProximityLayout::C1, ProximityLayout::C3, ProximityLayout::C4,
                            ProximityLayout::AC4}) {
    for (const bool useContext : {false, true}) {
      SCOPED_TRACE(::testing::Message() << "layout=" << static_cast<int>(layout)
                                       << " useContext=" << useContext);
      runAccuracyCase<Api>(layout, useContext, scaleFactor);
    }
  }
}

TEST(NppiSqrDistanceValidNormTest, EightBitAllLayoutsAndEntryPoints) { runApiMatrix<SqrDistance8uApi>(0); }

TEST(NppiSqrDistanceValidNormTest, EightBitToFloatAllLayoutsAndEntryPoints) {
  runApiMatrix<SqrDistance8u32fApi>();
}

TEST(NppiSqrDistanceValidNormTest, SignedEightBitToFloatAllLayoutsAndEntryPoints) {
  runApiMatrix<SqrDistance8s32fApi>();
}

TEST(NppiSqrDistanceValidNormTest, SixteenBitToFloatAllLayoutsAndEntryPoints) {
  runApiMatrix<SqrDistance16u32fApi>();
}

TEST(NppiSqrDistanceValidNormTest, FloatAllLayoutsAndEntryPoints) { runApiMatrix<SqrDistance32fApi>(); }

TEST(NppiSqrDistanceValidNormTest, EightBitScaleFactorAndSaturation) {
  const NppiSize srcSize{3, 2};
  const NppiSize tplSize{1, 1};
  const int srcStep = srcSize.width;
  const int tplStep = tplSize.width;
  const int dstStep = srcSize.width;
  const std::vector<Npp8u> source(static_cast<size_t>(srcSize.width) * srcSize.height, 1);
  const std::vector<Npp8u> templ(1, 3);
  Npp8u *deviceSource = nullptr;
  Npp8u *deviceTemplate = nullptr;
  Npp8u *deviceDestination = nullptr;
  ASSERT_EQ(cudaMalloc(&deviceSource, source.size()), cudaSuccess);
  ASSERT_EQ(cudaMalloc(&deviceTemplate, templ.size()), cudaSuccess);
  ASSERT_EQ(cudaMalloc(&deviceDestination, source.size()), cudaSuccess);
  ASSERT_EQ(cudaMemcpy(deviceSource, source.data(), source.size(), cudaMemcpyHostToDevice), cudaSuccess);
  ASSERT_EQ(cudaMemcpy(deviceTemplate, templ.data(), templ.size(), cudaMemcpyHostToDevice), cudaSuccess);

  for (const auto &scaleAndExpected : {std::pair<int, int>{-8, 255}, {-1, 3}, {0, 1}, {1, 1}, {2, 0}}) {
    ASSERT_EQ(nppiSqrDistanceValid_Norm_8u_C1RSfs(deviceSource, srcStep, srcSize, deviceTemplate, tplStep, tplSize,
                                                  deviceDestination, dstStep, scaleAndExpected.first),
              NPP_SUCCESS);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    std::vector<Npp8u> actual(source.size());
    ASSERT_EQ(cudaMemcpy(actual.data(), deviceDestination, actual.size(), cudaMemcpyDeviceToHost), cudaSuccess);
    for (const Npp8u value : actual) {
      EXPECT_EQ(value, scaleAndExpected.second) << "scaleFactor=" << scaleAndExpected.first;
    }
  }

  EXPECT_EQ(cudaFree(deviceDestination), cudaSuccess);
  EXPECT_EQ(cudaFree(deviceTemplate), cudaSuccess);
  EXPECT_EQ(cudaFree(deviceSource), cudaSuccess);
}

TEST(NppiSqrDistanceValidNormTest, ZeroEnergyProducesZero) {
  const NppiSize srcSize{4, 3};
  const NppiSize tplSize{2, 2};
  const int dstWidth = srcSize.width - tplSize.width + 1;
  const int dstHeight = srcSize.height - tplSize.height + 1;
  std::vector<Npp32f> source(static_cast<size_t>(srcSize.width) * srcSize.height, 0.0f);
  std::vector<Npp32f> templ(static_cast<size_t>(tplSize.width) * tplSize.height, 0.0f);
  std::vector<Npp32f> destination(static_cast<size_t>(dstWidth) * dstHeight, -1.0f);
  Npp32f *deviceSource = nullptr;
  Npp32f *deviceTemplate = nullptr;
  Npp32f *deviceDestination = nullptr;
  ASSERT_EQ(cudaMalloc(&deviceSource, source.size() * sizeof(Npp32f)), cudaSuccess);
  ASSERT_EQ(cudaMalloc(&deviceTemplate, templ.size() * sizeof(Npp32f)), cudaSuccess);
  ASSERT_EQ(cudaMalloc(&deviceDestination, destination.size() * sizeof(Npp32f)), cudaSuccess);
  ASSERT_EQ(cudaMemcpy(deviceSource, source.data(), source.size() * sizeof(Npp32f), cudaMemcpyHostToDevice),
            cudaSuccess);
  ASSERT_EQ(cudaMemcpy(deviceTemplate, templ.data(), templ.size() * sizeof(Npp32f), cudaMemcpyHostToDevice),
            cudaSuccess);

  NppStreamContext context{};
  ASSERT_EQ(nppGetStreamContext(&context), NPP_SUCCESS);
  for (const bool useContext : {false, true}) {
    ASSERT_EQ(cudaMemcpy(deviceDestination, destination.data(), destination.size() * sizeof(Npp32f),
                         cudaMemcpyHostToDevice),
              cudaSuccess);
    const NppStatus status =
        useContext ? nppiSqrDistanceValid_Norm_32f_C1R_Ctx(
                         deviceSource, srcSize.width * static_cast<int>(sizeof(Npp32f)), srcSize, deviceTemplate,
                         tplSize.width * static_cast<int>(sizeof(Npp32f)), tplSize, deviceDestination,
                         dstWidth * static_cast<int>(sizeof(Npp32f)), context)
                   : nppiSqrDistanceValid_Norm_32f_C1R(
                         deviceSource, srcSize.width * static_cast<int>(sizeof(Npp32f)), srcSize, deviceTemplate,
                         tplSize.width * static_cast<int>(sizeof(Npp32f)), tplSize, deviceDestination,
                         dstWidth * static_cast<int>(sizeof(Npp32f)));
    ASSERT_EQ(status, NPP_SUCCESS);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(destination.data(), deviceDestination, destination.size() * sizeof(Npp32f),
                         cudaMemcpyDeviceToHost),
              cudaSuccess);
    for (const Npp32f value : destination) {
      EXPECT_EQ(value, 0.0f);
    }
  }

  EXPECT_EQ(cudaFree(deviceDestination), cudaSuccess);
  EXPECT_EQ(cudaFree(deviceTemplate), cudaSuccess);
  EXPECT_EQ(cudaFree(deviceSource), cudaSuccess);
}

TEST(NppiSqrDistanceValidNormTest, ValidatesPointersStepsAndSizes) {
  const auto *source = reinterpret_cast<const Npp32f *>(0x1000);
  const auto *templ = reinterpret_cast<const Npp32f *>(0x2000);
  auto *destination = reinterpret_cast<Npp32f *>(0x3000);
  const NppiSize srcSize{5, 4};
  const NppiSize tplSize{2, 2};
  const int srcStep = srcSize.width * static_cast<int>(sizeof(Npp32f));
  const int tplStep = tplSize.width * static_cast<int>(sizeof(Npp32f));
  const int dstStep = (srcSize.width - tplSize.width + 1) * static_cast<int>(sizeof(Npp32f));
  NppStreamContext context{};

  EXPECT_EQ(nppiSqrDistanceValid_Norm_32f_C1R(nullptr, srcStep, srcSize, templ, tplStep, tplSize, destination,
                                              dstStep),
            NPP_NULL_POINTER_ERROR);
  EXPECT_EQ(nppiSqrDistanceValid_Norm_32f_C1R(source, srcStep, srcSize, nullptr, tplStep, tplSize, destination,
                                              dstStep),
            NPP_NULL_POINTER_ERROR);
  EXPECT_EQ(nppiSqrDistanceValid_Norm_32f_C1R(source, srcStep, srcSize, templ, tplStep, tplSize, nullptr, dstStep),
            NPP_NULL_POINTER_ERROR);
  EXPECT_EQ(nppiSqrDistanceValid_Norm_32f_C1R_Ctx(nullptr, srcStep, srcSize, templ, tplStep, tplSize, destination,
                                                  dstStep, context),
            NPP_NULL_POINTER_ERROR);
  EXPECT_EQ(nppiSqrDistanceValid_Norm_32f_C1R(source, 0, srcSize, templ, tplStep, tplSize, destination, dstStep),
            NPP_STEP_ERROR);
  EXPECT_EQ(nppiSqrDistanceValid_Norm_32f_C1R(source, srcStep, srcSize, templ, 0, tplSize, destination, dstStep),
            NPP_STEP_ERROR);
  EXPECT_EQ(nppiSqrDistanceValid_Norm_32f_C1R(source, srcStep, srcSize, templ, tplStep, tplSize, destination, 0),
            NPP_STEP_ERROR);
  EXPECT_EQ(nppiSqrDistanceValid_Norm_32f_C1R(source, srcStep - 1, srcSize, templ, tplStep, tplSize, destination,
                                              dstStep),
            NPP_STEP_ERROR);
  EXPECT_EQ(nppiSqrDistanceValid_Norm_32f_C1R(source, srcStep, NppiSize{0, srcSize.height}, templ, tplStep, tplSize,
                                              destination, dstStep),
            NPP_SIZE_ERROR);
  EXPECT_EQ(nppiSqrDistanceValid_Norm_32f_C1R(source, srcStep, srcSize, templ, tplStep,
                                              NppiSize{tplSize.width, 0}, destination, dstStep),
            NPP_SIZE_ERROR);
  EXPECT_EQ(nppiSqrDistanceValid_Norm_32f_C1R(source, srcStep, srcSize, templ, tplStep,
                                              NppiSize{srcSize.width + 1, tplSize.height}, destination, dstStep),
            NPP_SIZE_ERROR);
}

} // namespace
