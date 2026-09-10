#include "npp.h"

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <type_traits>
#include <vector>

namespace {

enum class CorrMode { Full, Same };
enum class CorrLayout { C1, C3, C4, AC4 };

int modeIndex(CorrMode mode) { return mode == CorrMode::Full ? 0 : 1; }
int layoutIndex(CorrLayout layout) { return static_cast<int>(layout); }
int channelCount(CorrLayout layout) { return layout == CorrLayout::C1 ? 1 : layout == CorrLayout::C3 ? 3 : 4; }
int correlatedChannelCount(CorrLayout layout) { return layout == CorrLayout::AC4 ? 3 : channelCount(layout); }

#define DEFINE_SCALED_API(NAME, FULL_C1, FULL_C1_CTX, FULL_C3, FULL_C3_CTX, FULL_C4, FULL_C4_CTX, FULL_AC4,       \
                          FULL_AC4_CTX, SAME_C1, SAME_C1_CTX, SAME_C3, SAME_C3_CTX, SAME_C4, SAME_C4_CTX,         \
                          SAME_AC4, SAME_AC4_CTX)                                                                 \
  struct NAME {                                                                                                  \
    using Source = Npp8u;                                                                                        \
    using Destination = Npp8u;                                                                                   \
    static constexpr bool kScaled = true;                                                                        \
    static NppStatus call(CorrMode mode, CorrLayout layout, const Source *src, int srcStep, NppiSize srcSize,    \
                          const Source *tpl, int tplStep, NppiSize tplSize, Destination *dst, int dstStep,        \
                          int scaleFactor, NppStreamContext context, bool useContext) {                            \
      using Fn = NppStatus (*)(const Source *, int, NppiSize, const Source *, int, NppiSize, Destination *, int, \
                               int);                                                                              \
      using CtxFn = NppStatus (*)(const Source *, int, NppiSize, const Source *, int, NppiSize, Destination *,   \
                                  int, int, NppStreamContext);                                                    \
      static const Fn functions[2][4] = {{FULL_C1, FULL_C3, FULL_C4, FULL_AC4},                                  \
                                          {SAME_C1, SAME_C3, SAME_C4, SAME_AC4}};                                 \
      static const CtxFn contextFunctions[2][4] = {{FULL_C1_CTX, FULL_C3_CTX, FULL_C4_CTX, FULL_AC4_CTX},        \
                                                    {SAME_C1_CTX, SAME_C3_CTX, SAME_C4_CTX, SAME_AC4_CTX}};       \
      const int m = modeIndex(mode);                                                                             \
      const int l = layoutIndex(layout);                                                                         \
      return useContext ? contextFunctions[m][l](src, srcStep, srcSize, tpl, tplStep, tplSize, dst, dstStep,     \
                                                  scaleFactor, context)                                           \
                        : functions[m][l](src, srcStep, srcSize, tpl, tplStep, tplSize, dst, dstStep,            \
                                          scaleFactor);                                                           \
    }                                                                                                             \
  }

#define DEFINE_UNSCALED_API(NAME, SOURCE_TYPE, DESTINATION_TYPE, FULL_C1, FULL_C1_CTX, FULL_C3, FULL_C3_CTX,      \
                            FULL_C4, FULL_C4_CTX, FULL_AC4, FULL_AC4_CTX, SAME_C1, SAME_C1_CTX, SAME_C3,          \
                            SAME_C3_CTX, SAME_C4, SAME_C4_CTX, SAME_AC4, SAME_AC4_CTX)                             \
  struct NAME {                                                                                                   \
    using Source = SOURCE_TYPE;                                                                                   \
    using Destination = DESTINATION_TYPE;                                                                         \
    static constexpr bool kScaled = false;                                                                         \
    static NppStatus call(CorrMode mode, CorrLayout layout, const Source *src, int srcStep, NppiSize srcSize,     \
                          const Source *tpl, int tplStep, NppiSize tplSize, Destination *dst, int dstStep, int,    \
                          NppStreamContext context, bool useContext) {                                              \
      using Fn = NppStatus (*)(const Source *, int, NppiSize, const Source *, int, NppiSize, Destination *, int); \
      using CtxFn = NppStatus (*)(const Source *, int, NppiSize, const Source *, int, NppiSize, Destination *,    \
                                  int, NppStreamContext);                                                          \
      static const Fn functions[2][4] = {{FULL_C1, FULL_C3, FULL_C4, FULL_AC4},                                   \
                                          {SAME_C1, SAME_C3, SAME_C4, SAME_AC4}};                                  \
      static const CtxFn contextFunctions[2][4] = {{FULL_C1_CTX, FULL_C3_CTX, FULL_C4_CTX, FULL_AC4_CTX},         \
                                                    {SAME_C1_CTX, SAME_C3_CTX, SAME_C4_CTX, SAME_AC4_CTX}};        \
      const int m = modeIndex(mode);                                                                              \
      const int l = layoutIndex(layout);                                                                          \
      return useContext ? contextFunctions[m][l](src, srcStep, srcSize, tpl, tplStep, tplSize, dst, dstStep,      \
                                                  context)                                                         \
                        : functions[m][l](src, srcStep, srcSize, tpl, tplStep, tplSize, dst, dstStep);             \
    }                                                                                                              \
  }

DEFINE_SCALED_API(CrossCorr8uApi, nppiCrossCorrFull_Norm_8u_C1RSfs, nppiCrossCorrFull_Norm_8u_C1RSfs_Ctx,
                  nppiCrossCorrFull_Norm_8u_C3RSfs, nppiCrossCorrFull_Norm_8u_C3RSfs_Ctx,
                  nppiCrossCorrFull_Norm_8u_C4RSfs, nppiCrossCorrFull_Norm_8u_C4RSfs_Ctx,
                  nppiCrossCorrFull_Norm_8u_AC4RSfs, nppiCrossCorrFull_Norm_8u_AC4RSfs_Ctx,
                  nppiCrossCorrSame_Norm_8u_C1RSfs, nppiCrossCorrSame_Norm_8u_C1RSfs_Ctx,
                  nppiCrossCorrSame_Norm_8u_C3RSfs, nppiCrossCorrSame_Norm_8u_C3RSfs_Ctx,
                  nppiCrossCorrSame_Norm_8u_C4RSfs, nppiCrossCorrSame_Norm_8u_C4RSfs_Ctx,
                  nppiCrossCorrSame_Norm_8u_AC4RSfs, nppiCrossCorrSame_Norm_8u_AC4RSfs_Ctx);

DEFINE_UNSCALED_API(CrossCorr8u32fApi, Npp8u, Npp32f, nppiCrossCorrFull_Norm_8u32f_C1R,
                    nppiCrossCorrFull_Norm_8u32f_C1R_Ctx, nppiCrossCorrFull_Norm_8u32f_C3R,
                    nppiCrossCorrFull_Norm_8u32f_C3R_Ctx, nppiCrossCorrFull_Norm_8u32f_C4R,
                    nppiCrossCorrFull_Norm_8u32f_C4R_Ctx, nppiCrossCorrFull_Norm_8u32f_AC4R,
                    nppiCrossCorrFull_Norm_8u32f_AC4R_Ctx, nppiCrossCorrSame_Norm_8u32f_C1R,
                    nppiCrossCorrSame_Norm_8u32f_C1R_Ctx, nppiCrossCorrSame_Norm_8u32f_C3R,
                    nppiCrossCorrSame_Norm_8u32f_C3R_Ctx, nppiCrossCorrSame_Norm_8u32f_C4R,
                    nppiCrossCorrSame_Norm_8u32f_C4R_Ctx, nppiCrossCorrSame_Norm_8u32f_AC4R,
                    nppiCrossCorrSame_Norm_8u32f_AC4R_Ctx);

DEFINE_UNSCALED_API(CrossCorr8s32fApi, Npp8s, Npp32f, nppiCrossCorrFull_Norm_8s32f_C1R,
                    nppiCrossCorrFull_Norm_8s32f_C1R_Ctx, nppiCrossCorrFull_Norm_8s32f_C3R,
                    nppiCrossCorrFull_Norm_8s32f_C3R_Ctx, nppiCrossCorrFull_Norm_8s32f_C4R,
                    nppiCrossCorrFull_Norm_8s32f_C4R_Ctx, nppiCrossCorrFull_Norm_8s32f_AC4R,
                    nppiCrossCorrFull_Norm_8s32f_AC4R_Ctx, nppiCrossCorrSame_Norm_8s32f_C1R,
                    nppiCrossCorrSame_Norm_8s32f_C1R_Ctx, nppiCrossCorrSame_Norm_8s32f_C3R,
                    nppiCrossCorrSame_Norm_8s32f_C3R_Ctx, nppiCrossCorrSame_Norm_8s32f_C4R,
                    nppiCrossCorrSame_Norm_8s32f_C4R_Ctx, nppiCrossCorrSame_Norm_8s32f_AC4R,
                    nppiCrossCorrSame_Norm_8s32f_AC4R_Ctx);

DEFINE_UNSCALED_API(CrossCorr16u32fApi, Npp16u, Npp32f, nppiCrossCorrFull_Norm_16u32f_C1R,
                    nppiCrossCorrFull_Norm_16u32f_C1R_Ctx, nppiCrossCorrFull_Norm_16u32f_C3R,
                    nppiCrossCorrFull_Norm_16u32f_C3R_Ctx, nppiCrossCorrFull_Norm_16u32f_C4R,
                    nppiCrossCorrFull_Norm_16u32f_C4R_Ctx, nppiCrossCorrFull_Norm_16u32f_AC4R,
                    nppiCrossCorrFull_Norm_16u32f_AC4R_Ctx, nppiCrossCorrSame_Norm_16u32f_C1R,
                    nppiCrossCorrSame_Norm_16u32f_C1R_Ctx, nppiCrossCorrSame_Norm_16u32f_C3R,
                    nppiCrossCorrSame_Norm_16u32f_C3R_Ctx, nppiCrossCorrSame_Norm_16u32f_C4R,
                    nppiCrossCorrSame_Norm_16u32f_C4R_Ctx, nppiCrossCorrSame_Norm_16u32f_AC4R,
                    nppiCrossCorrSame_Norm_16u32f_AC4R_Ctx);

DEFINE_UNSCALED_API(CrossCorr32fApi, Npp32f, Npp32f, nppiCrossCorrFull_Norm_32f_C1R,
                    nppiCrossCorrFull_Norm_32f_C1R_Ctx, nppiCrossCorrFull_Norm_32f_C3R,
                    nppiCrossCorrFull_Norm_32f_C3R_Ctx, nppiCrossCorrFull_Norm_32f_C4R,
                    nppiCrossCorrFull_Norm_32f_C4R_Ctx, nppiCrossCorrFull_Norm_32f_AC4R,
                    nppiCrossCorrFull_Norm_32f_AC4R_Ctx, nppiCrossCorrSame_Norm_32f_C1R,
                    nppiCrossCorrSame_Norm_32f_C1R_Ctx, nppiCrossCorrSame_Norm_32f_C3R,
                    nppiCrossCorrSame_Norm_32f_C3R_Ctx, nppiCrossCorrSame_Norm_32f_C4R,
                    nppiCrossCorrSame_Norm_32f_C4R_Ctx, nppiCrossCorrSame_Norm_32f_AC4R,
                    nppiCrossCorrSame_Norm_32f_AC4R_Ctx);

DEFINE_UNSCALED_API(CrossCorr64fApi, Npp64f, Npp64f, nppiCrossCorrFull_Norm_64f_C1R,
                    nppiCrossCorrFull_Norm_64f_C1R_Ctx, nppiCrossCorrFull_Norm_64f_C3R,
                    nppiCrossCorrFull_Norm_64f_C3R_Ctx, nppiCrossCorrFull_Norm_64f_C4R,
                    nppiCrossCorrFull_Norm_64f_C4R_Ctx, nppiCrossCorrFull_Norm_64f_AC4R,
                    nppiCrossCorrFull_Norm_64f_AC4R_Ctx, nppiCrossCorrSame_Norm_64f_C1R,
                    nppiCrossCorrSame_Norm_64f_C1R_Ctx, nppiCrossCorrSame_Norm_64f_C3R,
                    nppiCrossCorrSame_Norm_64f_C3R_Ctx, nppiCrossCorrSame_Norm_64f_C4R,
                    nppiCrossCorrSame_Norm_64f_C4R_Ctx, nppiCrossCorrSame_Norm_64f_AC4R,
                    nppiCrossCorrSame_Norm_64f_AC4R_Ctx);

#undef DEFINE_UNSCALED_API
#undef DEFINE_SCALED_API

template <typename T> T sourceValue(int x, int y, int channel) {
  if constexpr (std::is_same_v<T, Npp8u>) {
    return static_cast<T>(3 + ((x * 17 + y * 11 + channel * 7 + x * y * 5) % 41));
  } else if constexpr (std::is_same_v<T, Npp8s>) {
    return static_cast<T>(-19 + ((x * 13 + y * 23 + channel * 5 + x * y * 3) % 39));
  } else if constexpr (std::is_same_v<T, Npp16u>) {
    return static_cast<T>(101 + ((x * 733 + y * 881 + channel * 257 + x * y * 53) % 9000));
  } else {
    return static_cast<T>(-1.25 + x * 0.73 - y * 0.41 + channel * 0.29 + x * y * 0.17);
  }
}

template <typename T> T templateValue(int x, int y, int channel) {
  if constexpr (std::is_same_v<T, Npp8u>) {
    return static_cast<T>(2 + ((x * 29 + y * 7 + channel * 13 + x * y * 3) % 37));
  } else if constexpr (std::is_same_v<T, Npp8s>) {
    return static_cast<T>(-17 + ((x * 19 + y * 11 + channel * 7 + x * y * 5) % 35));
  } else if constexpr (std::is_same_v<T, Npp16u>) {
    return static_cast<T>(307 + ((x * 613 + y * 977 + channel * 193 + x * y * 47) % 8000));
  } else {
    return static_cast<T>(0.85 - x * 0.37 + y * 0.61 - channel * 0.43 + x * y * 0.12);
  }
}

template <typename T> T alphaSentinel() { return static_cast<T>(73); }
template <> Npp32f alphaSentinel<Npp32f>() { return 73.25F; }
template <> Npp64f alphaSentinel<Npp64f>() { return 73.25; }

Npp8u scaled8u(double value, int scaleFactor) {
  const double scaled = std::ldexp(value, -scaleFactor);
  return static_cast<Npp8u>(std::clamp(std::floor(scaled + 0.5), 0.0, 255.0));
}

template <typename Destination> Destination expectedValue(double value, int scaleFactor) {
  if constexpr (std::is_same_v<Destination, Npp8u>) {
    return scaled8u(value, scaleFactor);
  } else {
    return static_cast<Destination>(value);
  }
}

template <typename Destination> void expectNear(Destination actual, Destination expected) {
  if constexpr (std::is_same_v<Destination, Npp8u>) {
    EXPECT_EQ(actual, expected);
  } else if constexpr (std::is_same_v<Destination, Npp32f>) {
    EXPECT_NEAR(actual, expected, 2.0e-5F);
  } else {
    EXPECT_NEAR(actual, expected, 2.0e-12);
  }
}

template <typename Api> void runAccuracy(CorrMode mode, CorrLayout layout) {
  using Source = typename Api::Source;
  using Destination = typename Api::Destination;
  constexpr NppiSize srcSize{3, 4};
  constexpr NppiSize tplSize{4, 3};
  constexpr int scaleFactor = -7;
  const int channels = channelCount(layout);
  const int correlatedChannels = correlatedChannelCount(layout);
  const int dstWidth = mode == CorrMode::Full ? srcSize.width + tplSize.width - 1 : srcSize.width;
  const int dstHeight = mode == CorrMode::Full ? srcSize.height + tplSize.height - 1 : srcSize.height;
  const int srcPitch = srcSize.width * channels + 3;
  const int tplPitch = tplSize.width * channels + 2;
  const int dstPitch = dstWidth * channels + 5;

  std::vector<Source> source(static_cast<size_t>(srcPitch) * srcSize.height, Source{});
  std::vector<Source> tpl(static_cast<size_t>(tplPitch) * tplSize.height, Source{});
  for (int y = 0; y < srcSize.height; ++y) {
    for (int x = 0; x < srcSize.width; ++x) {
      for (int channel = 0; channel < channels; ++channel) {
        source[static_cast<size_t>(y) * srcPitch + x * channels + channel] = sourceValue<Source>(x, y, channel);
      }
    }
  }
  for (int y = 0; y < tplSize.height; ++y) {
    for (int x = 0; x < tplSize.width; ++x) {
      for (int channel = 0; channel < channels; ++channel) {
        tpl[static_cast<size_t>(y) * tplPitch + x * channels + channel] = templateValue<Source>(x, y, channel);
      }
    }
  }

  std::vector<Destination> initial(static_cast<size_t>(dstPitch) * dstHeight, alphaSentinel<Destination>());
  std::vector<Destination> expected = initial;
  for (int outY = 0; outY < dstHeight; ++outY) {
    for (int outX = 0; outX < dstWidth; ++outX) {
      const int sourceOriginX =
          mode == CorrMode::Full ? outX - (tplSize.width - 1) : outX - tplSize.width / 2;
      const int sourceOriginY =
          mode == CorrMode::Full ? outY - (tplSize.height - 1) : outY - tplSize.height / 2;
      for (int channel = 0; channel < correlatedChannels; ++channel) {
        double dot = 0.0;
        double sourceSquare = 0.0;
        double templateSquare = 0.0;
        for (int tplY = 0; tplY < tplSize.height; ++tplY) {
          for (int tplX = 0; tplX < tplSize.width; ++tplX) {
            const double templateSample =
                static_cast<double>(tpl[static_cast<size_t>(tplY) * tplPitch + tplX * channels + channel]);
            templateSquare += templateSample * templateSample;
            const int sourceX = sourceOriginX + tplX;
            const int sourceY = sourceOriginY + tplY;
            if (sourceX >= 0 && sourceX < srcSize.width && sourceY >= 0 && sourceY < srcSize.height) {
              const double sourceSample =
                  static_cast<double>(source[static_cast<size_t>(sourceY) * srcPitch + sourceX * channels + channel]);
              dot += sourceSample * templateSample;
              sourceSquare += sourceSample * sourceSample;
            }
          }
        }
        const double correlation =
            sourceSquare > 0.0 && templateSquare > 0.0 ? dot / std::sqrt(sourceSquare * templateSquare) : 0.0;
        expected[static_cast<size_t>(outY) * dstPitch + outX * channels + channel] =
            expectedValue<Destination>(correlation, scaleFactor);
      }
    }
  }

  Source *deviceSource = nullptr;
  Source *deviceTemplate = nullptr;
  Destination *deviceDestination = nullptr;
  ASSERT_EQ(cudaMalloc(reinterpret_cast<void **>(&deviceSource), source.size() * sizeof(Source)), cudaSuccess);
  ASSERT_EQ(cudaMalloc(reinterpret_cast<void **>(&deviceTemplate), tpl.size() * sizeof(Source)), cudaSuccess);
  ASSERT_EQ(cudaMalloc(reinterpret_cast<void **>(&deviceDestination), initial.size() * sizeof(Destination)),
            cudaSuccess);
  ASSERT_EQ(cudaMemcpy(deviceSource, source.data(), source.size() * sizeof(Source), cudaMemcpyHostToDevice),
            cudaSuccess);
  ASSERT_EQ(cudaMemcpy(deviceTemplate, tpl.data(), tpl.size() * sizeof(Source), cudaMemcpyHostToDevice), cudaSuccess);

  cudaStream_t stream = nullptr;
  ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);
  NppStreamContext context{};
  ASSERT_EQ(nppGetStreamContext(&context), NPP_SUCCESS);
  context.hStream = stream;
  const cudaStream_t originalStream = nppGetStream();

  for (bool useContext : {false, true}) {
    ASSERT_EQ(cudaMemcpy(deviceDestination, initial.data(), initial.size() * sizeof(Destination),
                         cudaMemcpyHostToDevice),
              cudaSuccess);
    if (!useContext) {
      ASSERT_EQ(nppSetStream(stream), NPP_SUCCESS);
    }
    const NppStatus status = Api::call(mode, layout, deviceSource, srcPitch * static_cast<int>(sizeof(Source)),
                                       srcSize, deviceTemplate, tplPitch * static_cast<int>(sizeof(Source)), tplSize,
                                       deviceDestination, dstPitch * static_cast<int>(sizeof(Destination)), scaleFactor,
                                       context, useContext);
    if (!useContext) {
      ASSERT_EQ(nppSetStream(originalStream), NPP_SUCCESS);
    }
    ASSERT_EQ(status, NPP_SUCCESS);
    ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
    std::vector<Destination> actual(initial.size());
    ASSERT_EQ(cudaMemcpy(actual.data(), deviceDestination, actual.size() * sizeof(Destination),
                         cudaMemcpyDeviceToHost),
              cudaSuccess);
    for (int y = 0; y < dstHeight; ++y) {
      for (int x = 0; x < dstWidth; ++x) {
        for (int channel = 0; channel < channels; ++channel) {
          const size_t index = static_cast<size_t>(y) * dstPitch + x * channels + channel;
          expectNear(actual[index], expected[index]);
        }
      }
    }
  }

  cudaStreamDestroy(stream);
  cudaFree(deviceDestination);
  cudaFree(deviceTemplate);
  cudaFree(deviceSource);
}

template <typename Api> void runZeroDenominator(CorrMode mode, CorrLayout layout) {
  using Source = typename Api::Source;
  using Destination = typename Api::Destination;
  constexpr NppiSize srcSize{2, 2};
  constexpr NppiSize tplSize{3, 2};
  const int channels = channelCount(layout);
  const int correlatedChannels = correlatedChannelCount(layout);
  const int dstWidth = mode == CorrMode::Full ? srcSize.width + tplSize.width - 1 : srcSize.width;
  const int dstHeight = mode == CorrMode::Full ? srcSize.height + tplSize.height - 1 : srcSize.height;
  const int srcStep = srcSize.width * channels * static_cast<int>(sizeof(Source));
  const int tplStep = tplSize.width * channels * static_cast<int>(sizeof(Source));
  const int dstStep = dstWidth * channels * static_cast<int>(sizeof(Destination));
  std::vector<Source> source(static_cast<size_t>(srcSize.width) * srcSize.height * channels, Source{});
  std::vector<Source> tpl(static_cast<size_t>(tplSize.width) * tplSize.height * channels, static_cast<Source>(3));
  std::vector<Destination> initial(static_cast<size_t>(dstWidth) * dstHeight * channels,
                                   alphaSentinel<Destination>());

  Source *deviceSource = nullptr;
  Source *deviceTemplate = nullptr;
  Destination *deviceDestination = nullptr;
  ASSERT_EQ(cudaMalloc(reinterpret_cast<void **>(&deviceSource), source.size() * sizeof(Source)), cudaSuccess);
  ASSERT_EQ(cudaMalloc(reinterpret_cast<void **>(&deviceTemplate), tpl.size() * sizeof(Source)), cudaSuccess);
  ASSERT_EQ(cudaMalloc(reinterpret_cast<void **>(&deviceDestination), initial.size() * sizeof(Destination)),
            cudaSuccess);
  ASSERT_EQ(cudaMemcpy(deviceSource, source.data(), source.size() * sizeof(Source), cudaMemcpyHostToDevice),
            cudaSuccess);
  ASSERT_EQ(cudaMemcpy(deviceTemplate, tpl.data(), tpl.size() * sizeof(Source), cudaMemcpyHostToDevice), cudaSuccess);
  ASSERT_EQ(cudaMemcpy(deviceDestination, initial.data(), initial.size() * sizeof(Destination), cudaMemcpyHostToDevice),
            cudaSuccess);
  NppStreamContext context{};
  ASSERT_EQ(nppGetStreamContext(&context), NPP_SUCCESS);
  ASSERT_EQ(Api::call(mode, layout, deviceSource, srcStep, srcSize, deviceTemplate, tplStep, tplSize,
                      deviceDestination, dstStep, -7, context, true),
            NPP_SUCCESS);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  std::vector<Destination> actual(initial.size());
  ASSERT_EQ(cudaMemcpy(actual.data(), deviceDestination, actual.size() * sizeof(Destination), cudaMemcpyDeviceToHost),
            cudaSuccess);
  for (int pixel = 0; pixel < dstWidth * dstHeight; ++pixel) {
    for (int channel = 0; channel < channels; ++channel) {
      const size_t index = static_cast<size_t>(pixel) * channels + channel;
      if (channel < correlatedChannels) {
        EXPECT_EQ(actual[index], static_cast<Destination>(0));
      } else {
        EXPECT_EQ(actual[index], initial[index]);
      }
    }
  }

  cudaFree(deviceDestination);
  cudaFree(deviceTemplate);
  cudaFree(deviceSource);
}

template <typename Api> void runErrors(CorrMode mode) {
  using Source = typename Api::Source;
  using Destination = typename Api::Destination;
  constexpr CorrLayout layout = CorrLayout::C3;
  constexpr int channels = 3;
  constexpr NppiSize srcSize{3, 4};
  constexpr NppiSize tplSize{4, 3};
  const int dstWidth = mode == CorrMode::Full ? srcSize.width + tplSize.width - 1 : srcSize.width;
  const int dstHeight = mode == CorrMode::Full ? srcSize.height + tplSize.height - 1 : srcSize.height;
  const int srcStep = srcSize.width * channels * static_cast<int>(sizeof(Source));
  const int tplStep = tplSize.width * channels * static_cast<int>(sizeof(Source));
  const int dstStep = dstWidth * channels * static_cast<int>(sizeof(Destination));
  Source *source = nullptr;
  Source *tpl = nullptr;
  Destination *destination = nullptr;
  ASSERT_EQ(cudaMalloc(reinterpret_cast<void **>(&source), static_cast<size_t>(srcStep) * srcSize.height),
            cudaSuccess);
  ASSERT_EQ(cudaMalloc(reinterpret_cast<void **>(&tpl), static_cast<size_t>(tplStep) * tplSize.height), cudaSuccess);
  ASSERT_EQ(cudaMalloc(reinterpret_cast<void **>(&destination), static_cast<size_t>(dstStep) * dstHeight), cudaSuccess);
  NppStreamContext context{};
  ASSERT_EQ(nppGetStreamContext(&context), NPP_SUCCESS);

  EXPECT_EQ(Api::call(mode, layout, nullptr, srcStep, srcSize, tpl, tplStep, tplSize, destination, dstStep, -7,
                      context, false),
            NPP_NULL_POINTER_ERROR);
  EXPECT_EQ(Api::call(mode, layout, source, srcStep, srcSize, nullptr, tplStep, tplSize, destination, dstStep, -7,
                      context, true),
            NPP_NULL_POINTER_ERROR);
  EXPECT_EQ(Api::call(mode, layout, source, srcStep, srcSize, tpl, tplStep, tplSize, nullptr, dstStep, -7, context,
                      false),
            NPP_NULL_POINTER_ERROR);
  EXPECT_EQ(Api::call(mode, layout, source, srcStep - static_cast<int>(sizeof(Source)), srcSize, tpl, tplStep, tplSize,
                      destination, dstStep, -7, context, false),
            NPP_STEP_ERROR);
  EXPECT_EQ(Api::call(mode, layout, source, srcStep, srcSize, tpl, tplStep - static_cast<int>(sizeof(Source)), tplSize,
                      destination, dstStep, -7, context, true),
            NPP_STEP_ERROR);
  EXPECT_EQ(Api::call(mode, layout, source, srcStep, srcSize, tpl, tplStep, tplSize, destination,
                      dstStep - static_cast<int>(sizeof(Destination)), -7, context, false),
            NPP_STEP_ERROR);
  EXPECT_EQ(Api::call(mode, layout, source, srcStep, {0, srcSize.height}, tpl, tplStep, tplSize, destination, dstStep,
                      -7, context, true),
            NPP_SIZE_ERROR);
  EXPECT_EQ(Api::call(mode, layout, source, srcStep, srcSize, tpl, tplStep, {tplSize.width, 0}, destination, dstStep,
                      -7, context, false),
            NPP_SIZE_ERROR);
  EXPECT_EQ(Api::call(mode, layout, source, srcStep, {-1, srcSize.height}, tpl, tplStep, tplSize, destination, dstStep,
                      -7, context, true),
            NPP_SIZE_ERROR);

  cudaFree(destination);
  cudaFree(tpl);
  cudaFree(source);
}

template <typename Api> void runFamily() {
  for (CorrMode mode : {CorrMode::Full, CorrMode::Same}) {
    for (CorrLayout layout : {CorrLayout::C1, CorrLayout::C3, CorrLayout::C4, CorrLayout::AC4}) {
      runAccuracy<Api>(mode, layout);
    }
    runZeroDenominator<Api>(mode, CorrLayout::C1);
    runZeroDenominator<Api>(mode, CorrLayout::AC4);
    runErrors<Api>(mode);
  }
}

TEST(NppiCrossCorrFullSameNormTest, CrossCorrFullSameNorm_8u_AllLayoutsAndParameters) {
  runFamily<CrossCorr8uApi>();
}
TEST(NppiCrossCorrFullSameNormTest, CrossCorrFullSameNorm_8u32f_AllLayoutsAndParameters) {
  runFamily<CrossCorr8u32fApi>();
}
TEST(NppiCrossCorrFullSameNormTest, CrossCorrFullSameNorm_8s32f_AllLayoutsAndParameters) {
  runFamily<CrossCorr8s32fApi>();
}
TEST(NppiCrossCorrFullSameNormTest, CrossCorrFullSameNorm_16u32f_AllLayoutsAndParameters) {
  runFamily<CrossCorr16u32fApi>();
}
TEST(NppiCrossCorrFullSameNormTest, CrossCorrFullSameNorm_32f_AllLayoutsAndParameters) {
  runFamily<CrossCorr32fApi>();
}
TEST(NppiCrossCorrFullSameNormTest, CrossCorrFullSameNorm_64f_AllLayoutsAndParameters) {
  runFamily<CrossCorr64fApi>();
}

} // namespace
