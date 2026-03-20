#include "npp_test_base.h"
#include <algorithm>
#include <cmath>
#include <type_traits>
#include <vector>

using namespace npp_functional_test;

class RemapFunctionalTest : public NppTestBase {
protected:
  void SetUp() override { NppTestBase::SetUp(); }

  void TearDown() override { NppTestBase::TearDown(); }
};

namespace {

template <typename T> std::vector<T> makePackedIdentityData(int width, int height, int channels) {
  std::vector<T> data(width * height * channels);
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      for (int c = 0; c < channels; ++c) {
        const int idx = (y * width + x) * channels + c;
        if constexpr (std::is_same_v<T, Npp8u>) {
          data[idx] = static_cast<Npp8u>((x * 11 + y * 7 + c * 31) & 0xFF);
        } else if constexpr (std::is_same_v<T, Npp16u>) {
          data[idx] = static_cast<Npp16u>((x * 211 + y * 137 + c * 1009) & 0xFFFF);
        } else if constexpr (std::is_same_v<T, Npp16s>) {
          data[idx] = static_cast<Npp16s>((x * 37 - y * 19 + c * 211) - 2048);
        } else if constexpr (std::is_same_v<T, Npp32f>) {
          data[idx] = static_cast<Npp32f>(x * 0.25f + y * 0.5f + c * 1.25f);
        } else {
          data[idx] = static_cast<Npp64f>(x * 0.25 + y * 0.5 + c * 1.25);
        }
      }
    }
  }
  return data;
}

template <typename T> std::vector<T> makePlanarIdentityData(int width, int height, int plane) {
  std::vector<T> data(width * height);
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      const int idx = y * width + x;
      if constexpr (std::is_same_v<T, Npp8u>) {
        data[idx] = static_cast<Npp8u>((x * 13 + y * 17 + plane * 23) & 0xFF);
      } else if constexpr (std::is_same_v<T, Npp16u>) {
        data[idx] = static_cast<Npp16u>((x * 211 + y * 131 + plane * 3001) & 0xFFFF);
      } else if constexpr (std::is_same_v<T, Npp16s>) {
        data[idx] = static_cast<Npp16s>((x * 37 - y * 41 + plane * 509) - 4096);
      } else if constexpr (std::is_same_v<T, Npp32f>) {
        data[idx] = static_cast<Npp32f>(x * 0.125f + y * 0.75f + plane * 2.0f);
      } else {
        data[idx] = static_cast<Npp64f>(x * 0.125 + y * 0.75 + plane * 2.0);
      }
    }
  }
  return data;
}

template <typename MapT> void fillIdentityMaps(std::vector<MapT> &xMapData, std::vector<MapT> &yMapData, int width, int height) {
  xMapData.resize(width * height);
  yMapData.resize(width * height);
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      const int idx = y * width + x;
      xMapData[idx] = static_cast<MapT>(x);
      yMapData[idx] = static_cast<MapT>(y);
    }
  }
}

template <typename T>
void expectVectorsEqual(const std::vector<T> &expected, const std::vector<T> &actual) {
  ASSERT_EQ(expected.size(), actual.size());
  for (size_t i = 0; i < expected.size(); ++i) {
    if constexpr (std::is_floating_point_v<T>) {
      EXPECT_NEAR(expected[i], actual[i], static_cast<T>(1e-6)) << "index " << i;
    } else {
      EXPECT_EQ(expected[i], actual[i]) << "index " << i;
    }
  }
}

template <typename T>
void copyImageToDevice(DeviceMemory<T> &device, int dstStepBytes, const std::vector<T> &hostData, int width, int height, int channels) {
  ASSERT_EQ(hostData.size(), static_cast<size_t>(width * height * channels));
  ASSERT_EQ(cudaMemcpy2D(device.get(), dstStepBytes, hostData.data(), width * channels * sizeof(T), width * channels * sizeof(T),
                         height, cudaMemcpyHostToDevice),
            cudaSuccess);
}

template <typename T>
std::vector<T> copyImageToHost(const DeviceMemory<T> &device, int srcStepBytes, int width, int height, int channels) {
  std::vector<T> hostData(width * height * channels);
  const cudaError_t err = cudaMemcpy2D(hostData.data(), width * channels * sizeof(T), device.get(), srcStepBytes,
                                       width * channels * sizeof(T), height, cudaMemcpyDeviceToHost);
  EXPECT_EQ(err, cudaSuccess);
  if (err != cudaSuccess) {
    return {};
  }
  return hostData;
}

template <typename T, typename MapT, typename CtxFn, typename PlainFn>
void runPackedIdentityCase(int width, int height, int channels, CtxFn ctxFn, PlainFn plainFn) {
  auto srcData = makePackedIdentityData<T>(width, height, channels);
  std::vector<MapT> xMapData;
  std::vector<MapT> yMapData;
  fillIdentityMaps(xMapData, yMapData, width, height);

  if constexpr (std::is_same_v<T, Npp64f> || std::is_same_v<MapT, Npp64f>) {
    const int srcStep = width * channels * sizeof(T);
    const int mapStep = width * sizeof(MapT);
    DeviceMemory<T> src;
    DeviceMemory<T> dstCtx;
    DeviceMemory<T> dstPlain;
    DeviceMemory<MapT> xMap;
    DeviceMemory<MapT> yMap;
    src.allocate(width * height * channels);
    dstCtx.allocate(width * height * channels);
    dstPlain.allocate(width * height * channels);
    xMap.allocate(width * height);
    yMap.allocate(width * height);

    copyImageToDevice(src, srcStep, srcData, width, height, channels);
    copyImageToDevice(xMap, mapStep, xMapData, width, height, 1);
    copyImageToDevice(yMap, mapStep, yMapData, width, height, 1);

    NppiSize srcSize = {width, height};
    NppiRect srcROI = {0, 0, width, height};
    NppiSize dstSizeROI = {width, height};
    NppStreamContext ctx;
    nppGetStreamContext(&ctx);

    ASSERT_EQ(ctxFn(src.get(), srcSize, srcStep, srcROI, xMap.get(), mapStep, yMap.get(), mapStep, dstCtx.get(), srcStep,
                    dstSizeROI, NPPI_INTER_NN, ctx),
              NPP_SUCCESS);
    ASSERT_EQ(plainFn(src.get(), srcSize, srcStep, srcROI, xMap.get(), mapStep, yMap.get(), mapStep, dstPlain.get(), srcStep,
                      dstSizeROI, NPPI_INTER_NN),
              NPP_SUCCESS);

    cudaStreamSynchronize(ctx.hStream);
    auto ctxResult = copyImageToHost(dstCtx, srcStep, width, height, channels);
    auto plainResult = copyImageToHost(dstPlain, srcStep, width, height, channels);
    expectVectorsEqual(srcData, ctxResult);
    expectVectorsEqual(srcData, plainResult);
  } else if (std::is_same_v<T, Npp16s> && channels != 1) {
    const int srcStep = width * channels * sizeof(T);
    const int mapStep = width * sizeof(MapT);
    DeviceMemory<T> src;
    DeviceMemory<T> dstCtx;
    DeviceMemory<T> dstPlain;
    DeviceMemory<MapT> xMap;
    DeviceMemory<MapT> yMap;
    src.allocate(width * height * channels);
    dstCtx.allocate(width * height * channels);
    dstPlain.allocate(width * height * channels);
    xMap.allocate(width * height);
    yMap.allocate(width * height);

    copyImageToDevice(src, srcStep, srcData, width, height, channels);
    copyImageToDevice(xMap, mapStep, xMapData, width, height, 1);
    copyImageToDevice(yMap, mapStep, yMapData, width, height, 1);

    NppiSize srcSize = {width, height};
    NppiRect srcROI = {0, 0, width, height};
    NppiSize dstSizeROI = {width, height};
    NppStreamContext ctx;
    nppGetStreamContext(&ctx);

    ASSERT_EQ(ctxFn(src.get(), srcSize, srcStep, srcROI, xMap.get(), mapStep, yMap.get(), mapStep, dstCtx.get(), srcStep,
                    dstSizeROI, NPPI_INTER_NN, ctx),
              NPP_SUCCESS);
    ASSERT_EQ(plainFn(src.get(), srcSize, srcStep, srcROI, xMap.get(), mapStep, yMap.get(), mapStep, dstPlain.get(), srcStep,
                      dstSizeROI, NPPI_INTER_NN),
              NPP_SUCCESS);

    cudaStreamSynchronize(ctx.hStream);
    auto ctxResult = copyImageToHost(dstCtx, srcStep, width, height, channels);
    auto plainResult = copyImageToHost(dstPlain, srcStep, width, height, channels);
    expectVectorsEqual(srcData, ctxResult);
    expectVectorsEqual(srcData, plainResult);
  } else {
    NppImageMemory<T> src(width, height, channels);
    NppImageMemory<T> dstCtx(width, height, channels);
    NppImageMemory<T> dstPlain(width, height, channels);
    NppImageMemory<MapT> xMap(width, height);
    NppImageMemory<MapT> yMap(width, height);

    src.copyFromHost(srcData);
    xMap.copyFromHost(xMapData);
    yMap.copyFromHost(yMapData);

    NppiSize srcSize = {width, height};
    NppiRect srcROI = {0, 0, width, height};
    NppiSize dstSizeROI = {width, height};
    NppStreamContext ctx;
    nppGetStreamContext(&ctx);

    ASSERT_EQ(ctxFn(src.get(), srcSize, src.step(), srcROI, xMap.get(), xMap.step(), yMap.get(), yMap.step(), dstCtx.get(),
                    dstCtx.step(), dstSizeROI, NPPI_INTER_NN, ctx),
              NPP_SUCCESS);
    ASSERT_EQ(
        plainFn(src.get(), srcSize, src.step(), srcROI, xMap.get(), xMap.step(), yMap.get(), yMap.step(), dstPlain.get(),
                dstPlain.step(), dstSizeROI, NPPI_INTER_NN),
        NPP_SUCCESS);

    cudaStreamSynchronize(ctx.hStream);
    std::vector<T> ctxResult(width * height * channels);
    std::vector<T> plainResult(width * height * channels);
    dstCtx.copyToHost(ctxResult);
    dstPlain.copyToHost(plainResult);
    expectVectorsEqual(srcData, ctxResult);
    expectVectorsEqual(srcData, plainResult);
  }
}

template <typename T, typename MapT, typename CtxFn, typename PlainFn>
void runAC4IdentityCase(int width, int height, CtxFn ctxFn, PlainFn plainFn) {
  auto srcData = makePackedIdentityData<T>(width, height, 4);
  auto dstInit = makePackedIdentityData<T>(width, height, 4);
  std::vector<MapT> xMapData;
  std::vector<MapT> yMapData;
  fillIdentityMaps(xMapData, yMapData, width, height);

  if constexpr (std::is_same_v<T, Npp64f> || std::is_same_v<MapT, Npp64f>) {
    const int step = width * 4 * sizeof(T);
    const int mapStep = width * sizeof(MapT);
    DeviceMemory<T> src;
    DeviceMemory<T> dstCtx;
    DeviceMemory<T> dstPlain;
    DeviceMemory<MapT> xMap;
    DeviceMemory<MapT> yMap;
    src.allocate(width * height * 4);
    dstCtx.allocate(width * height * 4);
    dstPlain.allocate(width * height * 4);
    xMap.allocate(width * height);
    yMap.allocate(width * height);

    copyImageToDevice(src, step, srcData, width, height, 4);
    copyImageToDevice(dstCtx, step, dstInit, width, height, 4);
    copyImageToDevice(dstPlain, step, dstInit, width, height, 4);
    copyImageToDevice(xMap, mapStep, xMapData, width, height, 1);
    copyImageToDevice(yMap, mapStep, yMapData, width, height, 1);

    NppiSize srcSize = {width, height};
    NppiRect srcROI = {0, 0, width, height};
    NppiSize dstSizeROI = {width, height};
    NppStreamContext ctx;
    nppGetStreamContext(&ctx);

    ASSERT_EQ(ctxFn(src.get(), srcSize, step, srcROI, xMap.get(), mapStep, yMap.get(), mapStep, dstCtx.get(), step, dstSizeROI,
                    NPPI_INTER_NN, ctx),
              NPP_SUCCESS);
    ASSERT_EQ(plainFn(src.get(), srcSize, step, srcROI, xMap.get(), mapStep, yMap.get(), mapStep, dstPlain.get(), step,
                      dstSizeROI, NPPI_INTER_NN),
              NPP_SUCCESS);

    cudaStreamSynchronize(ctx.hStream);
    auto ctxResult = copyImageToHost(dstCtx, step, width, height, 4);
    auto plainResult = copyImageToHost(dstPlain, step, width, height, 4);
    for (int i = 0; i < width * height; ++i) {
      const int idx = i * 4;
      for (int c = 0; c < 3; ++c) {
        EXPECT_NEAR(ctxResult[idx + c], srcData[idx + c], static_cast<T>(1e-9));
        EXPECT_NEAR(plainResult[idx + c], srcData[idx + c], static_cast<T>(1e-9));
      }
      EXPECT_NEAR(ctxResult[idx + 3], dstInit[idx + 3], static_cast<T>(1e-9));
      EXPECT_NEAR(plainResult[idx + 3], dstInit[idx + 3], static_cast<T>(1e-9));
    }
  } else if constexpr (std::is_same_v<T, Npp16s>) {
    const int step = width * 4 * sizeof(T);
    const int mapStep = width * sizeof(MapT);
    DeviceMemory<T> src;
    DeviceMemory<T> dstCtx;
    DeviceMemory<T> dstPlain;
    DeviceMemory<MapT> xMap;
    DeviceMemory<MapT> yMap;
    src.allocate(width * height * 4);
    dstCtx.allocate(width * height * 4);
    dstPlain.allocate(width * height * 4);
    xMap.allocate(width * height);
    yMap.allocate(width * height);

    copyImageToDevice(src, step, srcData, width, height, 4);
    copyImageToDevice(dstCtx, step, dstInit, width, height, 4);
    copyImageToDevice(dstPlain, step, dstInit, width, height, 4);
    copyImageToDevice(xMap, mapStep, xMapData, width, height, 1);
    copyImageToDevice(yMap, mapStep, yMapData, width, height, 1);

    NppiSize srcSize = {width, height};
    NppiRect srcROI = {0, 0, width, height};
    NppiSize dstSizeROI = {width, height};
    NppStreamContext ctx;
    nppGetStreamContext(&ctx);

    ASSERT_EQ(ctxFn(src.get(), srcSize, step, srcROI, xMap.get(), mapStep, yMap.get(), mapStep, dstCtx.get(), step, dstSizeROI,
                    NPPI_INTER_NN, ctx),
              NPP_SUCCESS);
    ASSERT_EQ(plainFn(src.get(), srcSize, step, srcROI, xMap.get(), mapStep, yMap.get(), mapStep, dstPlain.get(), step,
                      dstSizeROI, NPPI_INTER_NN),
              NPP_SUCCESS);

    cudaStreamSynchronize(ctx.hStream);
    auto ctxResult = copyImageToHost(dstCtx, step, width, height, 4);
    auto plainResult = copyImageToHost(dstPlain, step, width, height, 4);
    for (int i = 0; i < width * height; ++i) {
      const int idx = i * 4;
      for (int c = 0; c < 3; ++c) {
        EXPECT_EQ(ctxResult[idx + c], srcData[idx + c]);
        EXPECT_EQ(plainResult[idx + c], srcData[idx + c]);
      }
      EXPECT_EQ(ctxResult[idx + 3], dstInit[idx + 3]);
      EXPECT_EQ(plainResult[idx + 3], dstInit[idx + 3]);
    }
  } else {
    NppImageMemory<T> src(width, height, 4);
    NppImageMemory<T> dstCtx(width, height, 4);
    NppImageMemory<T> dstPlain(width, height, 4);
    NppImageMemory<MapT> xMap(width, height);
    NppImageMemory<MapT> yMap(width, height);

    src.copyFromHost(srcData);
    dstCtx.copyFromHost(dstInit);
    dstPlain.copyFromHost(dstInit);
    xMap.copyFromHost(xMapData);
    yMap.copyFromHost(yMapData);

    NppiSize srcSize = {width, height};
    NppiRect srcROI = {0, 0, width, height};
    NppiSize dstSizeROI = {width, height};
    NppStreamContext ctx;
    nppGetStreamContext(&ctx);

    ASSERT_EQ(ctxFn(src.get(), srcSize, src.step(), srcROI, xMap.get(), xMap.step(), yMap.get(), yMap.step(), dstCtx.get(),
                    dstCtx.step(), dstSizeROI, NPPI_INTER_NN, ctx),
              NPP_SUCCESS);
    ASSERT_EQ(
        plainFn(src.get(), srcSize, src.step(), srcROI, xMap.get(), xMap.step(), yMap.get(), yMap.step(), dstPlain.get(),
                dstPlain.step(), dstSizeROI, NPPI_INTER_NN),
        NPP_SUCCESS);

    cudaStreamSynchronize(ctx.hStream);
    std::vector<T> ctxResult(width * height * 4);
    std::vector<T> plainResult(width * height * 4);
    dstCtx.copyToHost(ctxResult);
    dstPlain.copyToHost(plainResult);

    for (int i = 0; i < width * height; ++i) {
      const int idx = i * 4;
      for (int c = 0; c < 3; ++c) {
        if constexpr (std::is_floating_point_v<T>) {
          EXPECT_NEAR(ctxResult[idx + c], srcData[idx + c], static_cast<T>(1e-6));
          EXPECT_NEAR(plainResult[idx + c], srcData[idx + c], static_cast<T>(1e-6));
        } else {
          EXPECT_EQ(ctxResult[idx + c], srcData[idx + c]);
          EXPECT_EQ(plainResult[idx + c], srcData[idx + c]);
        }
      }
      if constexpr (std::is_floating_point_v<T>) {
        EXPECT_NEAR(ctxResult[idx + 3], dstInit[idx + 3], static_cast<T>(1e-6));
        EXPECT_NEAR(plainResult[idx + 3], dstInit[idx + 3], static_cast<T>(1e-6));
      } else {
        EXPECT_EQ(ctxResult[idx + 3], dstInit[idx + 3]);
        EXPECT_EQ(plainResult[idx + 3], dstInit[idx + 3]);
      }
    }
  }
}

template <typename T, typename MapT, int Planes, typename CtxFn, typename PlainFn>
void runPlanarIdentityCase(int width, int height, CtxFn ctxFn, PlainFn plainFn) {
  std::vector<std::vector<T>> srcHost;
  std::vector<std::vector<T>> ctxHost;
  std::vector<std::vector<T>> plainHost;
  srcHost.reserve(Planes);
  ctxHost.resize(Planes);
  plainHost.resize(Planes);

  for (int i = 0; i < Planes; ++i) {
    srcHost.push_back(makePlanarIdentityData<T>(width, height, i));
  }

  std::vector<MapT> xMapData;
  std::vector<MapT> yMapData;
  fillIdentityMaps(xMapData, yMapData, width, height);

  if constexpr (std::is_same_v<T, Npp64f> || std::is_same_v<MapT, Npp64f>) {
    const int step = width * sizeof(T);
    const int mapStep = width * sizeof(MapT);
    std::vector<DeviceMemory<T>> srcPlanesRaw;
    std::vector<DeviceMemory<T>> dstCtxPlanesRaw;
    std::vector<DeviceMemory<T>> dstPlainPlanesRaw;
    srcPlanesRaw.reserve(Planes);
    dstCtxPlanesRaw.reserve(Planes);
    dstPlainPlanesRaw.reserve(Planes);
    for (int i = 0; i < Planes; ++i) {
      srcPlanesRaw.emplace_back();
      dstCtxPlanesRaw.emplace_back();
      dstPlainPlanesRaw.emplace_back();
      srcPlanesRaw.back().allocate(width * height);
      dstCtxPlanesRaw.back().allocate(width * height);
      dstPlainPlanesRaw.back().allocate(width * height);
      copyImageToDevice(srcPlanesRaw.back(), step, srcHost[i], width, height, 1);
    }

    DeviceMemory<MapT> xMap;
    DeviceMemory<MapT> yMap;
    xMap.allocate(width * height);
    yMap.allocate(width * height);
    copyImageToDevice(xMap, mapStep, xMapData, width, height, 1);
    copyImageToDevice(yMap, mapStep, yMapData, width, height, 1);

    const T *srcPtrs[Planes];
    T *dstCtxPtrs[Planes];
    T *dstPlainPtrs[Planes];
    for (int i = 0; i < Planes; ++i) {
      srcPtrs[i] = srcPlanesRaw[i].get();
      dstCtxPtrs[i] = dstCtxPlanesRaw[i].get();
      dstPlainPtrs[i] = dstPlainPlanesRaw[i].get();
    }

    NppiSize srcSize = {width, height};
    NppiRect srcROI = {0, 0, width, height};
    NppiSize dstSizeROI = {width, height};
    NppStreamContext ctx;
    nppGetStreamContext(&ctx);

    ASSERT_EQ(ctxFn(srcPtrs, srcSize, step, srcROI, xMap.get(), mapStep, yMap.get(), mapStep, dstCtxPtrs, step, dstSizeROI,
                    NPPI_INTER_NN, ctx),
              NPP_SUCCESS);
    ASSERT_EQ(plainFn(srcPtrs, srcSize, step, srcROI, xMap.get(), mapStep, yMap.get(), mapStep, dstPlainPtrs, step,
                      dstSizeROI, NPPI_INTER_NN),
              NPP_SUCCESS);

    cudaStreamSynchronize(ctx.hStream);
    for (int i = 0; i < Planes; ++i) {
      ctxHost[i] = copyImageToHost(dstCtxPlanesRaw[i], step, width, height, 1);
      plainHost[i] = copyImageToHost(dstPlainPlanesRaw[i], step, width, height, 1);
      if constexpr (Planes == 3) {
        expectVectorsEqual(ctxHost[i], plainHost[i]);
      } else {
        expectVectorsEqual(srcHost[i], ctxHost[i]);
        expectVectorsEqual(srcHost[i], plainHost[i]);
      }
    }
  } else {
    std::vector<NppImageMemory<T>> srcPlanes;
    std::vector<NppImageMemory<T>> dstCtxPlanes;
    std::vector<NppImageMemory<T>> dstPlainPlanes;
    srcPlanes.reserve(Planes);
    dstCtxPlanes.reserve(Planes);
    dstPlainPlanes.reserve(Planes);
    for (int i = 0; i < Planes; ++i) {
      srcPlanes.emplace_back(width, height);
      dstCtxPlanes.emplace_back(width, height);
      dstPlainPlanes.emplace_back(width, height);
      srcPlanes.back().copyFromHost(srcHost[i]);
    }

    NppImageMemory<MapT> xMap(width, height);
    NppImageMemory<MapT> yMap(width, height);
    xMap.copyFromHost(xMapData);
    yMap.copyFromHost(yMapData);

    const T *srcPtrs[Planes];
    T *dstCtxPtrs[Planes];
    T *dstPlainPtrs[Planes];
    for (int i = 0; i < Planes; ++i) {
      srcPtrs[i] = srcPlanes[i].get();
      dstCtxPtrs[i] = dstCtxPlanes[i].get();
      dstPlainPtrs[i] = dstPlainPlanes[i].get();
    }

    NppiSize srcSize = {width, height};
    NppiRect srcROI = {0, 0, width, height};
    NppiSize dstSizeROI = {width, height};
    NppStreamContext ctx;
    nppGetStreamContext(&ctx);

    ASSERT_EQ(ctxFn(srcPtrs, srcSize, srcPlanes[0].step(), srcROI, xMap.get(), xMap.step(), yMap.get(), yMap.step(),
                    dstCtxPtrs, dstCtxPlanes[0].step(), dstSizeROI, NPPI_INTER_NN, ctx),
              NPP_SUCCESS);
    ASSERT_EQ(plainFn(srcPtrs, srcSize, srcPlanes[0].step(), srcROI, xMap.get(), xMap.step(), yMap.get(), yMap.step(),
                      dstPlainPtrs, dstPlainPlanes[0].step(), dstSizeROI, NPPI_INTER_NN),
              NPP_SUCCESS);

    cudaStreamSynchronize(ctx.hStream);
    for (int i = 0; i < Planes; ++i) {
      dstCtxPlanes[i].copyToHost(ctxHost[i]);
      dstPlainPlanes[i].copyToHost(plainHost[i]);
      expectVectorsEqual(srcHost[i], ctxHost[i]);
      expectVectorsEqual(srcHost[i], plainHost[i]);
    }
  }
}

#define REMAP_PACKED_TEST(TEST_NAME, TYPE, MAPTYPE, CHANNELS, CTX_FN, FN)                                                      \
  TEST_F(RemapFunctionalTest, TEST_NAME) {                                                                                      \
    runPackedIdentityCase<TYPE, MAPTYPE>(17, 13, CHANNELS, CTX_FN, FN);                                                        \
  }

#define REMAP_AC4_TEST(TEST_NAME, TYPE, MAPTYPE, CTX_FN, FN)                                                                    \
  TEST_F(RemapFunctionalTest, TEST_NAME) {                                                                                      \
    runAC4IdentityCase<TYPE, MAPTYPE>(17, 13, CTX_FN, FN);                                                                     \
  }

#define REMAP_PLANAR_TEST(TEST_NAME, TYPE, MAPTYPE, PLANES, CTX_FN, FN)                                                         \
  TEST_F(RemapFunctionalTest, TEST_NAME) {                                                                                      \
    runPlanarIdentityCase<TYPE, MAPTYPE, PLANES>(19, 11, CTX_FN, FN);                                                          \
  }

} // namespace

// 测试8位单通道重映射
TEST_F(RemapFunctionalTest, Remap_8u_C1R_Ctx_Identity) {
  const int width = 64, height = 64;

  // prepare test data - 渐变图案
  std::vector<Npp8u> srcData(width * height);
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      srcData[y * width + x] = static_cast<Npp8u>((x + y) % 256);
    }
  }

  // 准备映射表 - 恒等映射
  std::vector<Npp32f> xMapData(width * height);
  std::vector<Npp32f> yMapData(width * height);
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      int idx = y * width + x;
      xMapData[idx] = static_cast<float>(x);
      yMapData[idx] = static_cast<float>(y);
    }
  }

  NppImageMemory<Npp8u> src(width, height);
  NppImageMemory<Npp8u> dst(width, height);
  NppImageMemory<Npp32f> xMap(width, height);
  NppImageMemory<Npp32f> yMap(width, height);

  src.copyFromHost(srcData);
  xMap.copyFromHost(xMapData);
  yMap.copyFromHost(yMapData);

  // 设置参数
  NppiSize srcSize = {width, height};
  NppiRect srcROI = {0, 0, width, height};
  NppiSize dstSizeROI = {width, height};

  // 获取流上下文
  NppStreamContext nppStreamCtx;
  nppGetStreamContext(&nppStreamCtx);

  // 执行重映射
  NppStatus status = nppiRemap_8u_C1R_Ctx(src.get(), srcSize, src.step(), srcROI, xMap.get(), xMap.step(), yMap.get(),
                                          yMap.step(), dst.get(), dst.step(), dstSizeROI, NPPI_INTER_NN, nppStreamCtx);

  ASSERT_EQ(status, NPP_SUCCESS);

  // 同步并获取结果
  cudaStreamSynchronize(nppStreamCtx.hStream);
  std::vector<Npp8u> dstData(width * height);
  dst.copyToHost(dstData);

  // Validate恒等映射 - 输出应该与输入相同
  for (int i = 0; i < width * height; i++) {
    ASSERT_EQ(srcData[i], dstData[i]);
  }
}

// 测试8位四通道重映射
TEST_F(RemapFunctionalTest, Remap_8u_C4R_Ctx_Identity) {
  const int width = 32, height = 32;

  // prepare test data - RGBA渐变
  std::vector<Npp8u> srcData(width * height * 4);
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      int idx = (y * width + x) * 4;
      srcData[idx + 0] = static_cast<Npp8u>(x);
      srcData[idx + 1] = static_cast<Npp8u>(y);
      srcData[idx + 2] = static_cast<Npp8u>((x + y) % 255);
      srcData[idx + 3] = static_cast<Npp8u>(200);
    }
  }

  // identity maps
  std::vector<Npp32f> xMapData(width * height);
  std::vector<Npp32f> yMapData(width * height);
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      int idx = y * width + x;
      xMapData[idx] = static_cast<float>(x);
      yMapData[idx] = static_cast<float>(y);
    }
  }

  NppImageMemory<Npp8u> src(width, height, 4);
  NppImageMemory<Npp8u> dst(width, height, 4);
  NppImageMemory<Npp32f> xMap(width, height);
  NppImageMemory<Npp32f> yMap(width, height);

  src.copyFromHost(srcData);
  xMap.copyFromHost(xMapData);
  yMap.copyFromHost(yMapData);

  NppiSize srcSize = {width, height};
  NppiRect srcROI = {0, 0, width, height};
  NppiSize dstSizeROI = {width, height};

  NppStreamContext nppStreamCtx;
  nppGetStreamContext(&nppStreamCtx);

  NppStatus status = nppiRemap_8u_C4R_Ctx(src.get(), srcSize, src.step(), srcROI, xMap.get(), xMap.step(), yMap.get(),
                                          yMap.step(), dst.get(), dst.step(), dstSizeROI, NPPI_INTER_NN, nppStreamCtx);
  ASSERT_EQ(status, NPP_SUCCESS);

  cudaStreamSynchronize(nppStreamCtx.hStream);
  std::vector<Npp8u> dstData(width * height * 4);
  dst.copyToHost(dstData);

  for (int i = 0; i < width * height * 4; i++) {
    ASSERT_EQ(srcData[i], dstData[i]);
  }
}

// 测试8位AC4重映射
TEST_F(RemapFunctionalTest, Remap_8u_AC4R_Ctx_Identity) {
  const int width = 16, height = 16;

  std::vector<Npp8u> srcData(width * height * 4);
  std::vector<Npp8u> dstInit(width * height * 4, 0);
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      int idx = (y * width + x) * 4;
      srcData[idx + 0] = static_cast<Npp8u>(x * 3);
      srcData[idx + 1] = static_cast<Npp8u>(y * 5);
      srcData[idx + 2] = static_cast<Npp8u>(x + y);
      srcData[idx + 3] = static_cast<Npp8u>(128);
      dstInit[idx + 3] = static_cast<Npp8u>(77);
    }
  }

  std::vector<Npp32f> xMapData(width * height);
  std::vector<Npp32f> yMapData(width * height);
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      int idx = y * width + x;
      xMapData[idx] = static_cast<float>(x);
      yMapData[idx] = static_cast<float>(y);
    }
  }

  NppImageMemory<Npp8u> src(width, height, 4);
  NppImageMemory<Npp8u> dst(width, height, 4);
  NppImageMemory<Npp32f> xMap(width, height);
  NppImageMemory<Npp32f> yMap(width, height);

  src.copyFromHost(srcData);
  dst.copyFromHost(dstInit);
  xMap.copyFromHost(xMapData);
  yMap.copyFromHost(yMapData);

  NppiSize srcSize = {width, height};
  NppiRect srcROI = {0, 0, width, height};
  NppiSize dstSizeROI = {width, height};

  NppStreamContext nppStreamCtx;
  nppGetStreamContext(&nppStreamCtx);

  NppStatus status = nppiRemap_8u_AC4R_Ctx(src.get(), srcSize, src.step(), srcROI, xMap.get(), xMap.step(), yMap.get(),
                                           yMap.step(), dst.get(), dst.step(), dstSizeROI, NPPI_INTER_NN, nppStreamCtx);
  ASSERT_EQ(status, NPP_SUCCESS);

  cudaStreamSynchronize(nppStreamCtx.hStream);
  std::vector<Npp8u> dstData(width * height * 4);
  dst.copyToHost(dstData);

  for (int i = 0; i < width * height; i++) {
    int idx = i * 4;
    ASSERT_EQ(srcData[idx + 0], dstData[idx + 0]);
    ASSERT_EQ(srcData[idx + 1], dstData[idx + 1]);
    ASSERT_EQ(srcData[idx + 2], dstData[idx + 2]);
    ASSERT_EQ(dstInit[idx + 3], dstData[idx + 3]);
  }
}

// 测试8位三通道平面重映射
TEST_F(RemapFunctionalTest, Remap_8u_P3R_Ctx_Identity) {
  const int width = 32, height = 32;

  std::vector<Npp8u> srcPlane0(width * height);
  std::vector<Npp8u> srcPlane1(width * height);
  std::vector<Npp8u> srcPlane2(width * height);
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      int idx = y * width + x;
      srcPlane0[idx] = static_cast<Npp8u>(x);
      srcPlane1[idx] = static_cast<Npp8u>(y);
      srcPlane2[idx] = static_cast<Npp8u>((x + y) % 255);
    }
  }

  std::vector<Npp32f> xMapData(width * height);
  std::vector<Npp32f> yMapData(width * height);
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      int idx = y * width + x;
      xMapData[idx] = static_cast<float>(x);
      yMapData[idx] = static_cast<float>(y);
    }
  }

  NppImageMemory<Npp8u> src0(width, height);
  NppImageMemory<Npp8u> src1(width, height);
  NppImageMemory<Npp8u> src2(width, height);
  NppImageMemory<Npp8u> dst0(width, height);
  NppImageMemory<Npp8u> dst1(width, height);
  NppImageMemory<Npp8u> dst2(width, height);
  NppImageMemory<Npp32f> xMap(width, height);
  NppImageMemory<Npp32f> yMap(width, height);

  src0.copyFromHost(srcPlane0);
  src1.copyFromHost(srcPlane1);
  src2.copyFromHost(srcPlane2);
  xMap.copyFromHost(xMapData);
  yMap.copyFromHost(yMapData);

  const Npp8u *srcPlanes[3] = {src0.get(), src1.get(), src2.get()};
  Npp8u *dstPlanes[3] = {dst0.get(), dst1.get(), dst2.get()};

  NppiSize srcSize = {width, height};
  NppiRect srcROI = {0, 0, width, height};
  NppiSize dstSizeROI = {width, height};

  NppStreamContext nppStreamCtx;
  nppGetStreamContext(&nppStreamCtx);

  NppStatus status = nppiRemap_8u_P3R_Ctx(srcPlanes, srcSize, src0.step(), srcROI, xMap.get(), xMap.step(), yMap.get(),
                                          yMap.step(), dstPlanes, dst0.step(), dstSizeROI, NPPI_INTER_NN, nppStreamCtx);
  ASSERT_EQ(status, NPP_SUCCESS);

  cudaStreamSynchronize(nppStreamCtx.hStream);
  std::vector<Npp8u> dstPlane0(width * height);
  std::vector<Npp8u> dstPlane1(width * height);
  std::vector<Npp8u> dstPlane2(width * height);
  dst0.copyToHost(dstPlane0);
  dst1.copyToHost(dstPlane1);
  dst2.copyToHost(dstPlane2);

  for (int i = 0; i < width * height; i++) {
    ASSERT_EQ(srcPlane0[i], dstPlane0[i]);
    ASSERT_EQ(srcPlane1[i], dstPlane1[i]);
    ASSERT_EQ(srcPlane2[i], dstPlane2[i]);
  }
}

// 测试16位无符号整数三通道重映射
TEST_F(RemapFunctionalTest, Remap_16u_C3R_Ctx_Mirror) {
  const int width = 32, height = 32;

  // prepare test data - RGB渐变
  std::vector<Npp16u> srcData(width * height * 3);
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      int idx = (y * width + x) * 3;
      srcData[idx + 0] = static_cast<Npp16u>(x * 2000); // R
      srcData[idx + 1] = static_cast<Npp16u>(y * 2000); // G
      srcData[idx + 2] = static_cast<Npp16u>(30000);    // B
    }
  }

  // 准备映射表 - 水平镜像
  std::vector<Npp32f> xMapData(width * height);
  std::vector<Npp32f> yMapData(width * height);
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      int idx = y * width + x;
      xMapData[idx] = static_cast<float>(width - 1 - x);
      yMapData[idx] = static_cast<float>(y);
    }
  }

  NppImageMemory<Npp16u> src(width, height, 3);
  NppImageMemory<Npp16u> dst(width, height, 3);
  NppImageMemory<Npp32f> xMap(width, height);
  NppImageMemory<Npp32f> yMap(width, height);

  src.copyFromHost(srcData);
  xMap.copyFromHost(xMapData);
  yMap.copyFromHost(yMapData);

  // 设置参数
  NppiSize srcSize = {width, height};
  NppiRect srcROI = {0, 0, width, height};
  NppiSize dstSizeROI = {width, height};

  // 获取流上下文
  NppStreamContext nppStreamCtx;
  nppGetStreamContext(&nppStreamCtx);

  // 执行重映射
  NppStatus status = nppiRemap_16u_C3R_Ctx(src.get(), srcSize, src.step(), srcROI, xMap.get(), xMap.step(), yMap.get(),
                                           yMap.step(), dst.get(), dst.step(), dstSizeROI, NPPI_INTER_NN, nppStreamCtx);

  ASSERT_EQ(status, NPP_SUCCESS);

  // 同步并获取结果
  cudaStreamSynchronize(nppStreamCtx.hStream);
  std::vector<Npp16u> dstData(width * height * 3);
  dst.copyToHost(dstData);

  //
  int firstIdx = 0;
  int lastIdx = (width - 1) * 3;
  ASSERT_EQ(srcData[firstIdx], dstData[lastIdx]);         // R通道
  ASSERT_EQ(srcData[firstIdx + 1], dstData[lastIdx + 1]); // G通道
}

// 测试32位浮点单通道重映射 - 旋转
TEST_F(RemapFunctionalTest, Remap_32f_C1R_Ctx_Rotation) {
  const int width = 64, height = 64;

  // prepare test data - 中心有一个正方形
  std::vector<Npp32f> srcData(width * height, 0.0f);
  for (int y = 20; y < 40; y++) {
    for (int x = 20; x < 40; x++) {
      srcData[y * width + x] = 1.0f;
    }
  }

  // 准备映射表 - 绕中心旋转90度
  std::vector<Npp32f> xMapData(width * height);
  std::vector<Npp32f> yMapData(width * height);
  float cx = width / 2.0f;
  float cy = height / 2.0f;

  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      // 90度逆时针旋转
      float dx = x - cx;
      float dy = y - cy;
      float newX = -dy + cx;
      float newY = dx + cy;

      int idx = y * width + x;
      xMapData[idx] = newX;
      yMapData[idx] = newY;
    }
  }

  NppImageMemory<Npp32f> src(width, height);
  NppImageMemory<Npp32f> dst(width, height);
  NppImageMemory<Npp32f> xMap(width, height);
  NppImageMemory<Npp32f> yMap(width, height);

  src.copyFromHost(srcData);
  xMap.copyFromHost(xMapData);
  yMap.copyFromHost(yMapData);

  // 设置参数
  NppiSize srcSize = {width, height};
  NppiRect srcROI = {0, 0, width, height};
  NppiSize dstSizeROI = {width, height};

  // 获取流上下文
  NppStreamContext nppStreamCtx;
  nppGetStreamContext(&nppStreamCtx);

  // 执行重映射
  NppStatus status =
      nppiRemap_32f_C1R_Ctx(src.get(), srcSize, src.step(), srcROI, xMap.get(), xMap.step(), yMap.get(), yMap.step(),
                            dst.get(), dst.step(), dstSizeROI, NPPI_INTER_LINEAR, nppStreamCtx);

  ASSERT_EQ(status, NPP_SUCCESS);

  // 同步并获取结果
  cudaStreamSynchronize(nppStreamCtx.hStream);
  std::vector<Npp32f> dstData(width * height);
  dst.copyToHost(dstData);

  // Validate旋转 - 检查是否有非零值
  float sum = 0.0f;
  for (const auto &val : dstData) {
    sum += val;
  }
  ASSERT_GT(sum, 100.0f); // 确保有数据被映射
}

REMAP_PACKED_TEST(Remap_8u_C1R_PlainAndCtx_Identity, Npp8u, Npp32f, 1, nppiRemap_8u_C1R_Ctx, nppiRemap_8u_C1R)
REMAP_PACKED_TEST(Remap_8u_C3R_PlainAndCtx_Identity, Npp8u, Npp32f, 3, nppiRemap_8u_C3R_Ctx, nppiRemap_8u_C3R)
REMAP_PACKED_TEST(Remap_8u_C4R_PlainAndCtx_Identity, Npp8u, Npp32f, 4, nppiRemap_8u_C4R_Ctx, nppiRemap_8u_C4R)
REMAP_AC4_TEST(Remap_8u_AC4R_PlainAndCtx_Identity, Npp8u, Npp32f, nppiRemap_8u_AC4R_Ctx, nppiRemap_8u_AC4R)
REMAP_PLANAR_TEST(Remap_8u_P3R_PlainAndCtx_Identity, Npp8u, Npp32f, 3, nppiRemap_8u_P3R_Ctx, nppiRemap_8u_P3R)
REMAP_PLANAR_TEST(Remap_8u_P4R_PlainAndCtx_Identity, Npp8u, Npp32f, 4, nppiRemap_8u_P4R_Ctx, nppiRemap_8u_P4R)

REMAP_PACKED_TEST(Remap_16u_C1R_PlainAndCtx_Identity, Npp16u, Npp32f, 1, nppiRemap_16u_C1R_Ctx, nppiRemap_16u_C1R)
REMAP_PACKED_TEST(Remap_16u_C3R_PlainAndCtx_Identity, Npp16u, Npp32f, 3, nppiRemap_16u_C3R_Ctx, nppiRemap_16u_C3R)
REMAP_PACKED_TEST(Remap_16u_C4R_PlainAndCtx_Identity, Npp16u, Npp32f, 4, nppiRemap_16u_C4R_Ctx, nppiRemap_16u_C4R)
REMAP_AC4_TEST(Remap_16u_AC4R_PlainAndCtx_Identity, Npp16u, Npp32f, nppiRemap_16u_AC4R_Ctx, nppiRemap_16u_AC4R)
REMAP_PLANAR_TEST(Remap_16u_P3R_PlainAndCtx_Identity, Npp16u, Npp32f, 3, nppiRemap_16u_P3R_Ctx, nppiRemap_16u_P3R)
REMAP_PLANAR_TEST(Remap_16u_P4R_PlainAndCtx_Identity, Npp16u, Npp32f, 4, nppiRemap_16u_P4R_Ctx, nppiRemap_16u_P4R)

REMAP_PACKED_TEST(Remap_16s_C1R_PlainAndCtx_Identity, Npp16s, Npp32f, 1, nppiRemap_16s_C1R_Ctx, nppiRemap_16s_C1R)
REMAP_PACKED_TEST(Remap_16s_C3R_PlainAndCtx_Identity, Npp16s, Npp32f, 3, nppiRemap_16s_C3R_Ctx, nppiRemap_16s_C3R)
REMAP_PACKED_TEST(Remap_16s_C4R_PlainAndCtx_Identity, Npp16s, Npp32f, 4, nppiRemap_16s_C4R_Ctx, nppiRemap_16s_C4R)
REMAP_AC4_TEST(Remap_16s_AC4R_PlainAndCtx_Identity, Npp16s, Npp32f, nppiRemap_16s_AC4R_Ctx, nppiRemap_16s_AC4R)
REMAP_PLANAR_TEST(Remap_16s_P3R_PlainAndCtx_Identity, Npp16s, Npp32f, 3, nppiRemap_16s_P3R_Ctx, nppiRemap_16s_P3R)
REMAP_PLANAR_TEST(Remap_16s_P4R_PlainAndCtx_Identity, Npp16s, Npp32f, 4, nppiRemap_16s_P4R_Ctx, nppiRemap_16s_P4R)

REMAP_PACKED_TEST(Remap_32f_C1R_PlainAndCtx_Identity, Npp32f, Npp32f, 1, nppiRemap_32f_C1R_Ctx, nppiRemap_32f_C1R)
REMAP_PACKED_TEST(Remap_32f_C3R_PlainAndCtx_Identity, Npp32f, Npp32f, 3, nppiRemap_32f_C3R_Ctx, nppiRemap_32f_C3R)
REMAP_PACKED_TEST(Remap_32f_C4R_PlainAndCtx_Identity, Npp32f, Npp32f, 4, nppiRemap_32f_C4R_Ctx, nppiRemap_32f_C4R)
REMAP_AC4_TEST(Remap_32f_AC4R_PlainAndCtx_Identity, Npp32f, Npp32f, nppiRemap_32f_AC4R_Ctx, nppiRemap_32f_AC4R)
REMAP_PLANAR_TEST(Remap_32f_P3R_PlainAndCtx_Identity, Npp32f, Npp32f, 3, nppiRemap_32f_P3R_Ctx, nppiRemap_32f_P3R)
REMAP_PLANAR_TEST(Remap_32f_P4R_PlainAndCtx_Identity, Npp32f, Npp32f, 4, nppiRemap_32f_P4R_Ctx, nppiRemap_32f_P4R)

REMAP_PACKED_TEST(Remap_64f_C1R_PlainAndCtx_Identity, Npp64f, Npp64f, 1, nppiRemap_64f_C1R_Ctx, nppiRemap_64f_C1R)
REMAP_PACKED_TEST(Remap_64f_C3R_PlainAndCtx_Identity, Npp64f, Npp64f, 3, nppiRemap_64f_C3R_Ctx, nppiRemap_64f_C3R)
REMAP_PACKED_TEST(Remap_64f_C4R_PlainAndCtx_Identity, Npp64f, Npp64f, 4, nppiRemap_64f_C4R_Ctx, nppiRemap_64f_C4R)
REMAP_AC4_TEST(Remap_64f_AC4R_PlainAndCtx_Identity, Npp64f, Npp64f, nppiRemap_64f_AC4R_Ctx, nppiRemap_64f_AC4R)
REMAP_PLANAR_TEST(Remap_64f_P3R_PlainAndCtx_Identity, Npp64f, Npp64f, 3, nppiRemap_64f_P3R_Ctx, nppiRemap_64f_P3R)
REMAP_PLANAR_TEST(Remap_64f_P4R_PlainAndCtx_Identity, Npp64f, Npp64f, 4, nppiRemap_64f_P4R_Ctx, nppiRemap_64f_P4R)
