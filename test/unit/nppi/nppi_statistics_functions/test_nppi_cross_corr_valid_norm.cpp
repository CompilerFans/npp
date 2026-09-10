#include "npp.h"

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <string>
#include <type_traits>
#include <vector>

namespace {

enum class CorrLayout { C1, C3, C4, AC4 };

int channelCount(CorrLayout layout) { return layout == CorrLayout::C1 ? 1 : layout == CorrLayout::C3 ? 3 : 4; }

int correlatedChannelCount(CorrLayout layout) { return layout == CorrLayout::AC4 ? 3 : channelCount(layout); }

struct CrossCorr8uApi {
  using Source = Npp8u;
  using Destination = Npp8u;
  static constexpr bool kScaled = true;

  static NppStatus call(CorrLayout layout, const Source *src, int srcStep, NppiSize srcSize, const Source *tpl,
                        int tplStep, NppiSize tplSize, Destination *dst, int dstStep, int scaleFactor,
                        NppStreamContext context, bool useContext) {
    switch (layout) {
    case CorrLayout::C1:
      return useContext ? nppiCrossCorrValid_Norm_8u_C1RSfs_Ctx(src, srcStep, srcSize, tpl, tplStep, tplSize, dst,
                                                                 dstStep, scaleFactor, context)
                        : nppiCrossCorrValid_Norm_8u_C1RSfs(src, srcStep, srcSize, tpl, tplStep, tplSize, dst,
                                                           dstStep, scaleFactor);
    case CorrLayout::C3:
      return useContext ? nppiCrossCorrValid_Norm_8u_C3RSfs_Ctx(src, srcStep, srcSize, tpl, tplStep, tplSize, dst,
                                                                 dstStep, scaleFactor, context)
                        : nppiCrossCorrValid_Norm_8u_C3RSfs(src, srcStep, srcSize, tpl, tplStep, tplSize, dst,
                                                           dstStep, scaleFactor);
    case CorrLayout::C4:
      return useContext ? nppiCrossCorrValid_Norm_8u_C4RSfs_Ctx(src, srcStep, srcSize, tpl, tplStep, tplSize, dst,
                                                                 dstStep, scaleFactor, context)
                        : nppiCrossCorrValid_Norm_8u_C4RSfs(src, srcStep, srcSize, tpl, tplStep, tplSize, dst,
                                                           dstStep, scaleFactor);
    case CorrLayout::AC4:
      return useContext ? nppiCrossCorrValid_Norm_8u_AC4RSfs_Ctx(src, srcStep, srcSize, tpl, tplStep, tplSize, dst,
                                                                  dstStep, scaleFactor, context)
                        : nppiCrossCorrValid_Norm_8u_AC4RSfs(src, srcStep, srcSize, tpl, tplStep, tplSize, dst,
                                                            dstStep, scaleFactor);
    }
    return NPP_BAD_ARGUMENT_ERROR;
  }
};

struct CrossCorr8u32fApi {
  using Source = Npp8u;
  using Destination = Npp32f;
  static constexpr bool kScaled = false;

  static NppStatus call(CorrLayout layout, const Source *src, int srcStep, NppiSize srcSize, const Source *tpl,
                        int tplStep, NppiSize tplSize, Destination *dst, int dstStep, int, NppStreamContext context,
                        bool useContext) {
    switch (layout) {
    case CorrLayout::C1:
      return useContext ? nppiCrossCorrValid_Norm_8u32f_C1R_Ctx(src, srcStep, srcSize, tpl, tplStep, tplSize, dst,
                                                                 dstStep, context)
                        : nppiCrossCorrValid_Norm_8u32f_C1R(src, srcStep, srcSize, tpl, tplStep, tplSize, dst,
                                                           dstStep);
    case CorrLayout::C3:
      return useContext ? nppiCrossCorrValid_Norm_8u32f_C3R_Ctx(src, srcStep, srcSize, tpl, tplStep, tplSize, dst,
                                                                 dstStep, context)
                        : nppiCrossCorrValid_Norm_8u32f_C3R(src, srcStep, srcSize, tpl, tplStep, tplSize, dst,
                                                           dstStep);
    case CorrLayout::C4:
      return useContext ? nppiCrossCorrValid_Norm_8u32f_C4R_Ctx(src, srcStep, srcSize, tpl, tplStep, tplSize, dst,
                                                                 dstStep, context)
                        : nppiCrossCorrValid_Norm_8u32f_C4R(src, srcStep, srcSize, tpl, tplStep, tplSize, dst,
                                                           dstStep);
    case CorrLayout::AC4:
      return useContext ? nppiCrossCorrValid_Norm_8u32f_AC4R_Ctx(src, srcStep, srcSize, tpl, tplStep, tplSize, dst,
                                                                  dstStep, context)
                        : nppiCrossCorrValid_Norm_8u32f_AC4R(src, srcStep, srcSize, tpl, tplStep, tplSize, dst,
                                                            dstStep);
    }
    return NPP_BAD_ARGUMENT_ERROR;
  }
};

struct CrossCorr8s32fApi {
  using Source = Npp8s;
  using Destination = Npp32f;
  static constexpr bool kScaled = false;

  static NppStatus call(CorrLayout layout, const Source *src, int srcStep, NppiSize srcSize, const Source *tpl,
                        int tplStep, NppiSize tplSize, Destination *dst, int dstStep, int, NppStreamContext context,
                        bool useContext) {
    switch (layout) {
    case CorrLayout::C1:
      return useContext ? nppiCrossCorrValid_Norm_8s32f_C1R_Ctx(src, srcStep, srcSize, tpl, tplStep, tplSize, dst,
                                                                 dstStep, context)
                        : nppiCrossCorrValid_Norm_8s32f_C1R(src, srcStep, srcSize, tpl, tplStep, tplSize, dst,
                                                           dstStep);
    case CorrLayout::C3:
      return useContext ? nppiCrossCorrValid_Norm_8s32f_C3R_Ctx(src, srcStep, srcSize, tpl, tplStep, tplSize, dst,
                                                                 dstStep, context)
                        : nppiCrossCorrValid_Norm_8s32f_C3R(src, srcStep, srcSize, tpl, tplStep, tplSize, dst,
                                                           dstStep);
    case CorrLayout::C4:
      return useContext ? nppiCrossCorrValid_Norm_8s32f_C4R_Ctx(src, srcStep, srcSize, tpl, tplStep, tplSize, dst,
                                                                 dstStep, context)
                        : nppiCrossCorrValid_Norm_8s32f_C4R(src, srcStep, srcSize, tpl, tplStep, tplSize, dst,
                                                           dstStep);
    case CorrLayout::AC4:
      return useContext ? nppiCrossCorrValid_Norm_8s32f_AC4R_Ctx(src, srcStep, srcSize, tpl, tplStep, tplSize, dst,
                                                                  dstStep, context)
                        : nppiCrossCorrValid_Norm_8s32f_AC4R(src, srcStep, srcSize, tpl, tplStep, tplSize, dst,
                                                            dstStep);
    }
    return NPP_BAD_ARGUMENT_ERROR;
  }
};

struct CrossCorr16u32fApi {
  using Source = Npp16u;
  using Destination = Npp32f;
  static constexpr bool kScaled = false;

  static NppStatus call(CorrLayout layout, const Source *src, int srcStep, NppiSize srcSize, const Source *tpl,
                        int tplStep, NppiSize tplSize, Destination *dst, int dstStep, int, NppStreamContext context,
                        bool useContext) {
    switch (layout) {
    case CorrLayout::C1:
      return useContext ? nppiCrossCorrValid_Norm_16u32f_C1R_Ctx(src, srcStep, srcSize, tpl, tplStep, tplSize, dst,
                                                                  dstStep, context)
                        : nppiCrossCorrValid_Norm_16u32f_C1R(src, srcStep, srcSize, tpl, tplStep, tplSize, dst,
                                                            dstStep);
    case CorrLayout::C3:
      return useContext ? nppiCrossCorrValid_Norm_16u32f_C3R_Ctx(src, srcStep, srcSize, tpl, tplStep, tplSize, dst,
                                                                  dstStep, context)
                        : nppiCrossCorrValid_Norm_16u32f_C3R(src, srcStep, srcSize, tpl, tplStep, tplSize, dst,
                                                            dstStep);
    case CorrLayout::C4:
      return useContext ? nppiCrossCorrValid_Norm_16u32f_C4R_Ctx(src, srcStep, srcSize, tpl, tplStep, tplSize, dst,
                                                                  dstStep, context)
                        : nppiCrossCorrValid_Norm_16u32f_C4R(src, srcStep, srcSize, tpl, tplStep, tplSize, dst,
                                                            dstStep);
    case CorrLayout::AC4:
      return useContext ? nppiCrossCorrValid_Norm_16u32f_AC4R_Ctx(src, srcStep, srcSize, tpl, tplStep, tplSize, dst,
                                                                   dstStep, context)
                        : nppiCrossCorrValid_Norm_16u32f_AC4R(src, srcStep, srcSize, tpl, tplStep, tplSize, dst,
                                                             dstStep);
    }
    return NPP_BAD_ARGUMENT_ERROR;
  }
};

struct CrossCorr32fApi {
  using Source = Npp32f;
  using Destination = Npp32f;
  static constexpr bool kScaled = false;

  static NppStatus call(CorrLayout layout, const Source *src, int srcStep, NppiSize srcSize, const Source *tpl,
                        int tplStep, NppiSize tplSize, Destination *dst, int dstStep, int, NppStreamContext context,
                        bool useContext) {
    switch (layout) {
    case CorrLayout::C1:
      return useContext ? nppiCrossCorrValid_Norm_32f_C1R_Ctx(src, srcStep, srcSize, tpl, tplStep, tplSize, dst,
                                                               dstStep, context)
                        : nppiCrossCorrValid_Norm_32f_C1R(src, srcStep, srcSize, tpl, tplStep, tplSize, dst,
                                                         dstStep);
    case CorrLayout::C3:
      return useContext ? nppiCrossCorrValid_Norm_32f_C3R_Ctx(src, srcStep, srcSize, tpl, tplStep, tplSize, dst,
                                                               dstStep, context)
                        : nppiCrossCorrValid_Norm_32f_C3R(src, srcStep, srcSize, tpl, tplStep, tplSize, dst,
                                                         dstStep);
    case CorrLayout::C4:
      return useContext ? nppiCrossCorrValid_Norm_32f_C4R_Ctx(src, srcStep, srcSize, tpl, tplStep, tplSize, dst,
                                                               dstStep, context)
                        : nppiCrossCorrValid_Norm_32f_C4R(src, srcStep, srcSize, tpl, tplStep, tplSize, dst,
                                                         dstStep);
    case CorrLayout::AC4:
      return useContext ? nppiCrossCorrValid_Norm_32f_AC4R_Ctx(src, srcStep, srcSize, tpl, tplStep, tplSize, dst,
                                                                dstStep, context)
                        : nppiCrossCorrValid_Norm_32f_AC4R(src, srcStep, srcSize, tpl, tplStep, tplSize, dst,
                                                          dstStep);
    }
    return NPP_BAD_ARGUMENT_ERROR;
  }
};

struct CrossCorr64fApi {
  using Source = Npp64f;
  using Destination = Npp64f;
  static constexpr bool kScaled = false;

  static NppStatus call(CorrLayout layout, const Source *src, int srcStep, NppiSize srcSize, const Source *tpl,
                        int tplStep, NppiSize tplSize, Destination *dst, int dstStep, int, NppStreamContext context,
                        bool useContext) {
    switch (layout) {
    case CorrLayout::C1:
      return useContext ? nppiCrossCorrValid_Norm_64f_C1R_Ctx(src, srcStep, srcSize, tpl, tplStep, tplSize, dst,
                                                               dstStep, context)
                        : nppiCrossCorrValid_Norm_64f_C1R(src, srcStep, srcSize, tpl, tplStep, tplSize, dst,
                                                         dstStep);
    case CorrLayout::C3:
      return useContext ? nppiCrossCorrValid_Norm_64f_C3R_Ctx(src, srcStep, srcSize, tpl, tplStep, tplSize, dst,
                                                               dstStep, context)
                        : nppiCrossCorrValid_Norm_64f_C3R(src, srcStep, srcSize, tpl, tplStep, tplSize, dst,
                                                         dstStep);
    case CorrLayout::C4:
      return useContext ? nppiCrossCorrValid_Norm_64f_C4R_Ctx(src, srcStep, srcSize, tpl, tplStep, tplSize, dst,
                                                               dstStep, context)
                        : nppiCrossCorrValid_Norm_64f_C4R(src, srcStep, srcSize, tpl, tplStep, tplSize, dst,
                                                         dstStep);
    case CorrLayout::AC4:
      return useContext ? nppiCrossCorrValid_Norm_64f_AC4R_Ctx(src, srcStep, srcSize, tpl, tplStep, tplSize, dst,
                                                                dstStep, context)
                        : nppiCrossCorrValid_Norm_64f_AC4R(src, srcStep, srcSize, tpl, tplStep, tplSize, dst,
                                                          dstStep);
    }
    return NPP_BAD_ARGUMENT_ERROR;
  }
};

template <typename T> T sourceValue(int x, int y, int channel) {
  if constexpr (std::is_same_v<T, Npp8u>) {
    return static_cast<T>(11 + ((x * 29 + y * 17 + channel * 37 + x * y * 3) % 211));
  } else if constexpr (std::is_same_v<T, Npp8s>) {
    return static_cast<T>(((x * 11 + y * 7 + channel * 5 + x * y * 3) % 47) - 23);
  } else if constexpr (std::is_same_v<T, Npp16u>) {
    return static_cast<T>(503 + ((x * 997 + y * 619 + channel * 431 + x * y * 71) % 12000));
  } else {
    return static_cast<T>(-2.75 + x * 0.83 - y * 0.61 + channel * 1.17 + x * y * 0.19);
  }
}

template <typename T> T templateValue(int x, int y, int channel) {
  if constexpr (std::is_same_v<T, Npp8u>) {
    return static_cast<T>(7 + ((x * 41 + y * 23 + channel * 31 + x * y * 5) % 173));
  } else if constexpr (std::is_same_v<T, Npp8s>) {
    return static_cast<T>(((x * 13 + y * 17 + channel * 11 + x * y * 5) % 43) - 21);
  } else if constexpr (std::is_same_v<T, Npp16u>) {
    return static_cast<T>(307 + ((x * 733 + y * 881 + channel * 257 + x * y * 53) % 9000));
  } else {
    return static_cast<T>(1.25 - x * 0.47 + y * 0.91 - channel * 0.63 + x * y * 0.14);
  }
}

template <typename T> T alphaSentinel() { return static_cast<T>(73); }
template <> Npp32f alphaSentinel<Npp32f>() { return 73.25F; }
template <> Npp64f alphaSentinel<Npp64f>() { return 73.25; }

Npp8u scaled8u(double value, int scaleFactor) {
  const double scaled = std::ldexp(value, -scaleFactor);
  return static_cast<Npp8u>(std::clamp(std::floor(scaled + 0.5), 0.0, 255.0));
}

template <typename Api> void runAccuracy(CorrLayout layout) {
  using Source = typename Api::Source;
  using Destination = typename Api::Destination;
  constexpr int srcWidth = 6;
  constexpr int srcHeight = 5;
  constexpr int tplWidth = 3;
  constexpr int tplHeight = 2;
  constexpr int dstWidth = srcWidth - tplWidth + 1;
  constexpr int dstHeight = srcHeight - tplHeight + 1;
  constexpr int scaleFactor = -7;
  const int channels = channelCount(layout);
  const int correlatedChannels = correlatedChannelCount(layout);
  const NppiSize srcSize{srcWidth, srcHeight};
  const NppiSize tplSize{tplWidth, tplHeight};
  const int srcRowBytes = srcWidth * channels * static_cast<int>(sizeof(Source));
  const int tplRowBytes = tplWidth * channels * static_cast<int>(sizeof(Source));
  const int dstRowBytes = dstWidth * channels * static_cast<int>(sizeof(Destination));
  const int srcStep = srcRowBytes + 5 * static_cast<int>(sizeof(Source));
  const int tplStep = tplRowBytes + 7 * static_cast<int>(sizeof(Source));
  const int dstStep = dstRowBytes + 9 * static_cast<int>(sizeof(Destination));

  std::vector<Source> source(static_cast<size_t>(srcWidth) * srcHeight * channels);
  std::vector<Source> tpl(static_cast<size_t>(tplWidth) * tplHeight * channels);
  std::vector<Destination> initial(static_cast<size_t>(dstWidth) * dstHeight * channels, alphaSentinel<Destination>());
  for (int y = 0; y < srcHeight; ++y) {
    for (int x = 0; x < srcWidth; ++x) {
      for (int channel = 0; channel < channels; ++channel) {
        source[(static_cast<size_t>(y) * srcWidth + x) * channels + channel] = sourceValue<Source>(x, y, channel);
      }
    }
  }
  for (int y = 0; y < tplHeight; ++y) {
    for (int x = 0; x < tplWidth; ++x) {
      for (int channel = 0; channel < channels; ++channel) {
        tpl[(static_cast<size_t>(y) * tplWidth + x) * channels + channel] =
            channel == 3 && layout == CorrLayout::AC4 ? templateValue<Source>(x + 17, y + 11, channel)
                                                       : templateValue<Source>(x, y, channel);
      }
    }
  }

  Source *deviceSource = nullptr;
  Source *deviceTemplate = nullptr;
  Destination *deviceDestination = nullptr;
  ASSERT_EQ(cudaMalloc(reinterpret_cast<void **>(&deviceSource), static_cast<size_t>(srcStep) * srcHeight),
            cudaSuccess);
  ASSERT_EQ(cudaMalloc(reinterpret_cast<void **>(&deviceTemplate), static_cast<size_t>(tplStep) * tplHeight),
            cudaSuccess);
  ASSERT_EQ(cudaMalloc(reinterpret_cast<void **>(&deviceDestination), static_cast<size_t>(dstStep) * dstHeight),
            cudaSuccess);
  ASSERT_EQ(cudaMemcpy2D(deviceSource, srcStep, source.data(), srcRowBytes, srcRowBytes, srcHeight,
                         cudaMemcpyHostToDevice),
            cudaSuccess);
  ASSERT_EQ(cudaMemcpy2D(deviceTemplate, tplStep, tpl.data(), tplRowBytes, tplRowBytes, tplHeight,
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

  for (bool useContext : {false, true}) {
    SCOPED_TRACE(useContext ? "Ctx" : "default");
    const cudaStream_t stream = useContext ? contextStream : managedStream;
    ASSERT_EQ(cudaMemcpy2DAsync(deviceDestination, dstStep, initial.data(), dstRowBytes, dstRowBytes, dstHeight,
                                cudaMemcpyHostToDevice, stream),
              cudaSuccess);
    ASSERT_EQ(Api::call(layout, deviceSource, srcStep, srcSize, deviceTemplate, tplStep, tplSize, deviceDestination,
                        dstStep, scaleFactor, context, useContext),
              NPP_SUCCESS);
    ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
    std::vector<Destination> actual(initial.size());
    ASSERT_EQ(cudaMemcpy2D(actual.data(), dstRowBytes, deviceDestination, dstStep, dstRowBytes, dstHeight,
                           cudaMemcpyDeviceToHost),
              cudaSuccess);

    for (int y = 0; y < dstHeight; ++y) {
      for (int x = 0; x < dstWidth; ++x) {
        for (int channel = 0; channel < channels; ++channel) {
          const size_t dstIndex = (static_cast<size_t>(y) * dstWidth + x) * channels + channel;
          if (channel >= correlatedChannels) {
            EXPECT_EQ(actual[dstIndex], initial[dstIndex]) << "x=" << x << " y=" << y;
            continue;
          }
          double dot = 0.0;
          double sourceSquare = 0.0;
          double templateSquare = 0.0;
          for (int tplY = 0; tplY < tplHeight; ++tplY) {
            for (int tplX = 0; tplX < tplWidth; ++tplX) {
              const double sourceSample = static_cast<double>(
                  source[(static_cast<size_t>(y + tplY) * srcWidth + x + tplX) * channels + channel]);
              const double templateSample =
                  static_cast<double>(tpl[(static_cast<size_t>(tplY) * tplWidth + tplX) * channels + channel]);
              dot += sourceSample * templateSample;
              sourceSquare += sourceSample * sourceSample;
              templateSquare += templateSample * templateSample;
            }
          }
          const double denominator = std::sqrt(sourceSquare * templateSquare);
          const double expected = denominator == 0.0 ? 0.0 : dot / denominator;
          if constexpr (Api::kScaled) {
            EXPECT_NEAR(static_cast<double>(actual[dstIndex]), static_cast<double>(scaled8u(expected, scaleFactor)),
                        1.0)
                << "x=" << x << " y=" << y << " channel=" << channel;
          } else {
            const double tolerance = std::is_same_v<Destination, Npp64f> ? 1e-11 : 2e-5;
            EXPECT_NEAR(static_cast<double>(actual[dstIndex]), expected, tolerance)
                << "x=" << x << " y=" << y << " channel=" << channel;
          }
        }
      }
    }
  }

  EXPECT_EQ(nppSetStream(originalStream), NPP_SUCCESS);
  cudaStreamDestroy(contextStream);
  cudaStreamDestroy(managedStream);
  cudaFree(deviceDestination);
  cudaFree(deviceTemplate);
  cudaFree(deviceSource);
}

template <typename Api> void runZeroDenominator(CorrLayout layout) {
  using Source = typename Api::Source;
  using Destination = typename Api::Destination;
  const int channels = channelCount(layout);
  const int correlatedChannels = correlatedChannelCount(layout);
  const NppiSize srcSize{3, 2};
  const NppiSize tplSize{2, 1};
  const int dstWidth = 2;
  const int dstHeight = 2;
  const int srcStep = srcSize.width * channels * static_cast<int>(sizeof(Source));
  const int tplStep = tplSize.width * channels * static_cast<int>(sizeof(Source));
  const int dstStep = dstWidth * channels * static_cast<int>(sizeof(Destination));
  std::vector<Source> source(static_cast<size_t>(srcSize.width) * srcSize.height * channels, Source{});
  std::vector<Source> tpl(static_cast<size_t>(tplSize.width) * tplSize.height * channels, static_cast<Source>(3));
  std::vector<Destination> initial(static_cast<size_t>(dstWidth) * dstHeight * channels, alphaSentinel<Destination>());

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
  NppStreamContext context{};
  ASSERT_EQ(nppGetStreamContext(&context), NPP_SUCCESS);

  for (bool useContext : {false, true}) {
    ASSERT_EQ(cudaMemcpy(deviceDestination, initial.data(), initial.size() * sizeof(Destination),
                         cudaMemcpyHostToDevice),
              cudaSuccess);
    ASSERT_EQ(Api::call(layout, deviceSource, srcStep, srcSize, deviceTemplate, tplStep, tplSize, deviceDestination,
                        dstStep, -7, context, useContext),
              NPP_SUCCESS);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    std::vector<Destination> actual(initial.size());
    ASSERT_EQ(cudaMemcpy(actual.data(), deviceDestination, actual.size() * sizeof(Destination),
                         cudaMemcpyDeviceToHost),
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
  }

  cudaFree(deviceDestination);
  cudaFree(deviceTemplate);
  cudaFree(deviceSource);
}

template <typename Api> void runErrors() {
  using Source = typename Api::Source;
  using Destination = typename Api::Destination;
  constexpr CorrLayout layout = CorrLayout::C3;
  constexpr int channels = 3;
  const NppiSize srcSize{5, 4};
  const NppiSize tplSize{3, 2};
  const int dstWidth = srcSize.width - tplSize.width + 1;
  const int srcStep = srcSize.width * channels * static_cast<int>(sizeof(Source));
  const int tplStep = tplSize.width * channels * static_cast<int>(sizeof(Source));
  const int dstStep = dstWidth * channels * static_cast<int>(sizeof(Destination));
  Source *source = nullptr;
  Source *tpl = nullptr;
  Destination *destination = nullptr;
  ASSERT_EQ(cudaMalloc(reinterpret_cast<void **>(&source), static_cast<size_t>(srcStep) * srcSize.height),
            cudaSuccess);
  ASSERT_EQ(cudaMalloc(reinterpret_cast<void **>(&tpl), static_cast<size_t>(tplStep) * tplSize.height), cudaSuccess);
  ASSERT_EQ(cudaMalloc(reinterpret_cast<void **>(&destination), static_cast<size_t>(dstStep) * 3), cudaSuccess);
  NppStreamContext context{};
  ASSERT_EQ(nppGetStreamContext(&context), NPP_SUCCESS);

  EXPECT_EQ(Api::call(layout, nullptr, srcStep, srcSize, tpl, tplStep, tplSize, destination, dstStep, -7, context,
                      false),
            NPP_NULL_POINTER_ERROR);
  EXPECT_EQ(Api::call(layout, source, srcStep, srcSize, nullptr, tplStep, tplSize, destination, dstStep, -7, context,
                      true),
            NPP_NULL_POINTER_ERROR);
  EXPECT_EQ(Api::call(layout, source, srcStep, srcSize, tpl, tplStep, tplSize, nullptr, dstStep, -7, context, false),
            NPP_NULL_POINTER_ERROR);
  EXPECT_EQ(Api::call(layout, source, srcStep - static_cast<int>(sizeof(Source)), srcSize, tpl, tplStep, tplSize,
                      destination, dstStep, -7, context, false),
            NPP_STEP_ERROR);
  EXPECT_EQ(Api::call(layout, source, srcStep, srcSize, tpl, tplStep - static_cast<int>(sizeof(Source)), tplSize,
                      destination, dstStep, -7, context, true),
            NPP_STEP_ERROR);
  EXPECT_EQ(Api::call(layout, source, srcStep, srcSize, tpl, tplStep, tplSize, destination,
                      dstStep - static_cast<int>(sizeof(Destination)), -7, context, false),
            NPP_STEP_ERROR);
  EXPECT_EQ(Api::call(layout, source, srcStep, {0, srcSize.height}, tpl, tplStep, tplSize, destination, dstStep, -7,
                      context, true),
            NPP_SIZE_ERROR);
  EXPECT_EQ(Api::call(layout, source, srcStep, srcSize, tpl, tplStep, {tplSize.width, 0}, destination, dstStep, -7,
                      context, false),
            NPP_SIZE_ERROR);
  EXPECT_EQ(Api::call(layout, source, srcStep, srcSize, tpl, tplStep, {srcSize.width + 1, tplSize.height}, destination,
                      dstStep, -7, context, true),
            NPP_SIZE_ERROR);

  cudaFree(destination);
  cudaFree(tpl);
  cudaFree(source);
}

template <typename Api> void runFamily() {
  for (CorrLayout layout : {CorrLayout::C1, CorrLayout::C3, CorrLayout::C4, CorrLayout::AC4}) {
    runAccuracy<Api>(layout);
  }
  runZeroDenominator<Api>(CorrLayout::C1);
  runZeroDenominator<Api>(CorrLayout::AC4);
  runErrors<Api>();
}

TEST(NppiCrossCorrValidNormTest, CrossCorrValidNorm_8u_AllLayoutsAndParameters) { runFamily<CrossCorr8uApi>(); }
TEST(NppiCrossCorrValidNormTest, CrossCorrValidNorm_8u32f_AllLayoutsAndParameters) {
  runFamily<CrossCorr8u32fApi>();
}
TEST(NppiCrossCorrValidNormTest, CrossCorrValidNorm_8s32f_AllLayoutsAndParameters) {
  runFamily<CrossCorr8s32fApi>();
}
TEST(NppiCrossCorrValidNormTest, CrossCorrValidNorm_16u32f_AllLayoutsAndParameters) {
  runFamily<CrossCorr16u32fApi>();
}
TEST(NppiCrossCorrValidNormTest, CrossCorrValidNorm_32f_AllLayoutsAndParameters) { runFamily<CrossCorr32fApi>(); }
TEST(NppiCrossCorrValidNormTest, CrossCorrValidNorm_64f_AllLayoutsAndParameters) { runFamily<CrossCorr64fApi>(); }

TEST(NppiCrossCorrValidNormTest, CrossCorrValidNorm_8u_ScaleFactorsAndSaturation) {
  const NppiSize size{2, 2};
  const int step = size.width * static_cast<int>(sizeof(Npp8u));
  const std::vector<Npp8u> source{3, 7, 11, 19};
  Npp8u *deviceSource = nullptr;
  Npp8u *deviceTemplate = nullptr;
  Npp8u *deviceDestination = nullptr;
  ASSERT_EQ(cudaMalloc(reinterpret_cast<void **>(&deviceSource), source.size() * sizeof(Npp8u)), cudaSuccess);
  ASSERT_EQ(cudaMalloc(reinterpret_cast<void **>(&deviceTemplate), source.size() * sizeof(Npp8u)), cudaSuccess);
  ASSERT_EQ(cudaMalloc(reinterpret_cast<void **>(&deviceDestination), sizeof(Npp8u)), cudaSuccess);
  ASSERT_EQ(cudaMemcpy(deviceSource, source.data(), source.size() * sizeof(Npp8u), cudaMemcpyHostToDevice),
            cudaSuccess);
  ASSERT_EQ(cudaMemcpy(deviceTemplate, source.data(), source.size() * sizeof(Npp8u), cudaMemcpyHostToDevice),
            cudaSuccess);
  NppStreamContext context{};
  ASSERT_EQ(nppGetStreamContext(&context), NPP_SUCCESS);

  struct ScaleCase {
    int scaleFactor;
    Npp8u expected;
  };
  const std::vector<ScaleCase> cases{{-8, 255}, {-7, 128}, {-1, 2}, {0, 1}, {1, 1}, {2, 0}};
  for (size_t index = 0; index < cases.size(); ++index) {
    const ScaleCase testCase = cases[index];
    SCOPED_TRACE(testCase.scaleFactor);
    const bool useContext = index % 2 != 0;
    ASSERT_EQ(CrossCorr8uApi::call(CorrLayout::C1, deviceSource, step, size, deviceTemplate, step, size,
                                   deviceDestination, sizeof(Npp8u), testCase.scaleFactor, context, useContext),
              NPP_SUCCESS);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    Npp8u actual = 0;
    ASSERT_EQ(cudaMemcpy(&actual, deviceDestination, sizeof(actual), cudaMemcpyDeviceToHost), cudaSuccess);
    EXPECT_EQ(actual, testCase.expected);
  }

  cudaFree(deviceDestination);
  cudaFree(deviceTemplate);
  cudaFree(deviceSource);
}

} // namespace
