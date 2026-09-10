#include "npp.h"

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <vector>

namespace {

class DeviceBuffer {
public:
  explicit DeviceBuffer(size_t bytes) {
    if (cudaMalloc(reinterpret_cast<void **>(&data_), bytes) != cudaSuccess) {
      data_ = nullptr;
    }
  }

  ~DeviceBuffer() {
    if (data_) {
      cudaFree(data_);
    }
  }

  DeviceBuffer(const DeviceBuffer &) = delete;
  DeviceBuffer &operator=(const DeviceBuffer &) = delete;

  Npp8u *get() const { return data_; }

private:
  Npp8u *data_ = nullptr;
};

class StreamGuard {
public:
  StreamGuard() {
    if (cudaStreamCreate(&stream_) != cudaSuccess) {
      stream_ = nullptr;
    }
  }

  ~StreamGuard() {
    if (stream_) {
      cudaStreamDestroy(stream_);
    }
  }

  cudaStream_t get() const { return stream_; }

private:
  cudaStream_t stream_ = nullptr;
};

class ManagedStreamGuard {
public:
  ManagedStreamGuard() : original_(nppGetStream()) {
    cudaStream_t stream = nullptr;
    if (cudaStreamCreate(&stream) != cudaSuccess) {
      return;
    }
    if (nppSetStream(stream) != NPP_SUCCESS) {
      cudaStreamDestroy(stream);
      return;
    }
    stream_ = stream;
  }

  ~ManagedStreamGuard() {
    nppSetStream(original_);
    if (stream_) {
      cudaStreamDestroy(stream_);
    }
  }

  cudaStream_t get() const { return stream_; }

private:
  cudaStream_t original_ = nullptr;
  cudaStream_t stream_ = nullptr;
};

struct Layout422Data {
  std::vector<Npp8u> y;
  std::vector<Npp8u> cb;
  std::vector<Npp8u> cr;
  std::vector<Npp8u> ycbcr;
  std::vector<Npp8u> ycrcb;
};

Layout422Data makeLayout422Data(int width, int height) {
  Layout422Data data;
  data.y.resize(static_cast<size_t>(width) * height);
  data.cb.resize(static_cast<size_t>(width / 2) * height);
  data.cr.resize(static_cast<size_t>(width / 2) * height);
  data.ycbcr.resize(static_cast<size_t>(width) * height * 2);
  data.ycrcb.resize(static_cast<size_t>(width) * height * 2);

  for (int row = 0; row < height; ++row) {
    for (int pair = 0; pair < width / 2; ++pair) {
      const Npp8u y0 = static_cast<Npp8u>(11 + row * 37 + pair * 7);
      const Npp8u y1 = static_cast<Npp8u>(91 + row * 29 + pair * 5);
      const Npp8u cb = static_cast<Npp8u>(31 + row * 13 + pair * 9);
      const Npp8u cr = static_cast<Npp8u>(181 + row * 11 + pair * 7);
      const size_t yIndex = static_cast<size_t>(row) * width + pair * 2;
      const size_t chromaIndex = static_cast<size_t>(row) * (width / 2) + pair;
      const size_t packedIndex = static_cast<size_t>(row) * width * 2 + pair * 4;

      data.y[yIndex] = y0;
      data.y[yIndex + 1] = y1;
      data.cb[chromaIndex] = cb;
      data.cr[chromaIndex] = cr;
      data.ycbcr[packedIndex] = y0;
      data.ycbcr[packedIndex + 1] = cb;
      data.ycbcr[packedIndex + 2] = y1;
      data.ycbcr[packedIndex + 3] = cr;
      data.ycrcb[packedIndex] = y0;
      data.ycrcb[packedIndex + 1] = cr;
      data.ycrcb[packedIndex + 2] = y1;
      data.ycrcb[packedIndex + 3] = cb;
    }
  }
  return data;
}

void copyToDevice2D(Npp8u *dst, int dstStep, const std::vector<Npp8u> &src, int rowBytes, int rows) {
  ASSERT_EQ(cudaMemcpy2D(dst, dstStep, src.data(), rowBytes, rowBytes, rows, cudaMemcpyHostToDevice), cudaSuccess);
}

std::vector<Npp8u> copyFromDevice2D(const Npp8u *src, int srcStep, int rowBytes, int rows) {
  std::vector<Npp8u> result(static_cast<size_t>(rowBytes) * rows);
  EXPECT_EQ(cudaMemcpy2D(result.data(), rowBytes, src, srcStep, rowBytes, rows, cudaMemcpyDeviceToHost), cudaSuccess);
  return result;
}

void expectImage(const Npp8u *src, int srcStep, int rowBytes, int rows, const std::vector<Npp8u> &expected) {
  EXPECT_EQ(copyFromDevice2D(src, srcStep, rowBytes, rows), expected);
}

void clearBuffer(DeviceBuffer &buffer, int step, int rows) {
  ASSERT_EQ(cudaMemset(buffer.get(), 0xa5, static_cast<size_t>(step) * rows), cudaSuccess);
}

NppStreamContext streamContext(cudaStream_t stream) {
  NppStreamContext context{};
  EXPECT_EQ(nppGetStreamContext(&context), NPP_SUCCESS);
  context.hStream = stream;
  return context;
}

class YCbCr422LayoutTest : public ::testing::Test {
protected:
  void SetUp() override { ASSERT_EQ(cudaSetDevice(0), cudaSuccess); }
  void TearDown() override { EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess); }
};

TEST_F(YCbCr422LayoutTest, PackedPlanarRoundTripUsesIndependentStepsAndStreams) {
  constexpr int width = 6;
  constexpr int height = 3;
  constexpr int srcStep = width * 2 + 5;
  constexpr int yStep = width + 3;
  constexpr int cbStep = width / 2 + 5;
  constexpr int crStep = width / 2 + 7;
  constexpr int packedStep = width * 2 + 9;
  const NppiSize roi{width, height};
  const Layout422Data expected = makeLayout422Data(width, height);

  DeviceBuffer src(static_cast<size_t>(srcStep) * height);
  DeviceBuffer y(static_cast<size_t>(yStep) * height);
  DeviceBuffer cb(static_cast<size_t>(cbStep) * height);
  DeviceBuffer cr(static_cast<size_t>(crStep) * height);
  DeviceBuffer packed(static_cast<size_t>(packedStep) * height);
  ASSERT_NE(src.get(), nullptr);
  ASSERT_NE(y.get(), nullptr);
  ASSERT_NE(cb.get(), nullptr);
  ASSERT_NE(cr.get(), nullptr);
  ASSERT_NE(packed.get(), nullptr);
  copyToDevice2D(src.get(), srcStep, expected.ycbcr, width * 2, height);

  Npp8u *dst[3] = {y.get(), cb.get(), cr.get()};
  int dstSteps[3] = {yStep, cbStep, crStep};
  ManagedStreamGuard managedStream;
  ASSERT_NE(managedStream.get(), nullptr);
  ASSERT_EQ(nppiYCbCr422_8u_C2P3R(src.get(), srcStep, dst, dstSteps, roi), NPP_SUCCESS);
  ASSERT_EQ(cudaStreamSynchronize(managedStream.get()), cudaSuccess);
  expectImage(y.get(), yStep, width, height, expected.y);
  expectImage(cb.get(), cbStep, width / 2, height, expected.cb);
  expectImage(cr.get(), crStep, width / 2, height, expected.cr);

  StreamGuard stream;
  ASSERT_NE(stream.get(), nullptr);
  const NppStreamContext context = streamContext(stream.get());
  clearBuffer(y, yStep, height);
  clearBuffer(cb, cbStep, height);
  clearBuffer(cr, crStep, height);
  ASSERT_EQ(nppiYCbCr422_8u_C2P3R_Ctx(src.get(), srcStep, dst, dstSteps, roi, context), NPP_SUCCESS);
  ASSERT_EQ(cudaStreamSynchronize(stream.get()), cudaSuccess);
  expectImage(y.get(), yStep, width, height, expected.y);
  expectImage(cb.get(), cbStep, width / 2, height, expected.cb);
  expectImage(cr.get(), crStep, width / 2, height, expected.cr);

  const Npp8u *planar[3] = {y.get(), cb.get(), cr.get()};
  int planarSteps[3] = {yStep, cbStep, crStep};
  ASSERT_EQ(nppiYCbCr422_8u_P3C2R(planar, planarSteps, packed.get(), packedStep, roi), NPP_SUCCESS);
  ASSERT_EQ(cudaStreamSynchronize(managedStream.get()), cudaSuccess);
  expectImage(packed.get(), packedStep, width * 2, height, expected.ycbcr);

  clearBuffer(packed, packedStep, height);
  ASSERT_EQ(nppiYCbCr422_8u_P3C2R_Ctx(planar, planarSteps, packed.get(), packedStep, roi, context), NPP_SUCCESS);
  ASSERT_EQ(cudaStreamSynchronize(stream.get()), cudaSuccess);
  expectImage(packed.get(), packedStep, width * 2, height, expected.ycbcr);
}

TEST_F(YCbCr422LayoutTest, YCbCrToYCrCbPreservesByteOrderForPackedAndPlanarInputs) {
  constexpr int width = 6;
  constexpr int height = 3;
  constexpr int srcStep = width * 2 + 5;
  constexpr int yStep = width + 3;
  constexpr int cbStep = width / 2 + 5;
  constexpr int crStep = width / 2 + 7;
  constexpr int dstStep = width * 2 + 9;
  const NppiSize roi{width, height};
  const Layout422Data expected = makeLayout422Data(width, height);

  DeviceBuffer src(static_cast<size_t>(srcStep) * height);
  DeviceBuffer y(static_cast<size_t>(yStep) * height);
  DeviceBuffer cb(static_cast<size_t>(cbStep) * height);
  DeviceBuffer cr(static_cast<size_t>(crStep) * height);
  DeviceBuffer dst(static_cast<size_t>(dstStep) * height);
  ASSERT_NE(src.get(), nullptr);
  ASSERT_NE(y.get(), nullptr);
  ASSERT_NE(cb.get(), nullptr);
  ASSERT_NE(cr.get(), nullptr);
  ASSERT_NE(dst.get(), nullptr);
  copyToDevice2D(src.get(), srcStep, expected.ycbcr, width * 2, height);
  copyToDevice2D(y.get(), yStep, expected.y, width, height);
  copyToDevice2D(cb.get(), cbStep, expected.cb, width / 2, height);
  copyToDevice2D(cr.get(), crStep, expected.cr, width / 2, height);

  ManagedStreamGuard managedStream;
  ASSERT_NE(managedStream.get(), nullptr);
  ASSERT_EQ(nppiYCbCr422ToYCrCb422_8u_C2R(src.get(), srcStep, dst.get(), dstStep, roi), NPP_SUCCESS);
  ASSERT_EQ(cudaStreamSynchronize(managedStream.get()), cudaSuccess);
  expectImage(dst.get(), dstStep, width * 2, height, expected.ycrcb);

  StreamGuard stream;
  ASSERT_NE(stream.get(), nullptr);
  const NppStreamContext context = streamContext(stream.get());
  clearBuffer(dst, dstStep, height);
  ASSERT_EQ(nppiYCbCr422ToYCrCb422_8u_C2R_Ctx(src.get(), srcStep, dst.get(), dstStep, roi, context), NPP_SUCCESS);
  ASSERT_EQ(cudaStreamSynchronize(stream.get()), cudaSuccess);
  expectImage(dst.get(), dstStep, width * 2, height, expected.ycrcb);

  const Npp8u *planar[3] = {y.get(), cb.get(), cr.get()};
  int planarSteps[3] = {yStep, cbStep, crStep};
  clearBuffer(dst, dstStep, height);
  ASSERT_EQ(nppiYCbCr422ToYCrCb422_8u_P3C2R(planar, planarSteps, dst.get(), dstStep, roi), NPP_SUCCESS);
  ASSERT_EQ(cudaStreamSynchronize(managedStream.get()), cudaSuccess);
  expectImage(dst.get(), dstStep, width * 2, height, expected.ycrcb);

  clearBuffer(dst, dstStep, height);
  ASSERT_EQ(nppiYCbCr422ToYCrCb422_8u_P3C2R_Ctx(planar, planarSteps, dst.get(), dstStep, roi, context), NPP_SUCCESS);
  ASSERT_EQ(cudaStreamSynchronize(stream.get()), cudaSuccess);
  expectImage(dst.get(), dstStep, width * 2, height, expected.ycrcb);
}

TEST_F(YCbCr422LayoutTest, YCrCbPackedToYCbCrPlanarPreservesByteOrder) {
  constexpr int width = 6;
  constexpr int height = 3;
  constexpr int srcStep = width * 2 + 5;
  constexpr int yStep = width + 3;
  constexpr int cbStep = width / 2 + 5;
  constexpr int crStep = width / 2 + 7;
  const NppiSize roi{width, height};
  const Layout422Data expected = makeLayout422Data(width, height);

  DeviceBuffer src(static_cast<size_t>(srcStep) * height);
  DeviceBuffer y(static_cast<size_t>(yStep) * height);
  DeviceBuffer cb(static_cast<size_t>(cbStep) * height);
  DeviceBuffer cr(static_cast<size_t>(crStep) * height);
  ASSERT_NE(src.get(), nullptr);
  ASSERT_NE(y.get(), nullptr);
  ASSERT_NE(cb.get(), nullptr);
  ASSERT_NE(cr.get(), nullptr);
  copyToDevice2D(src.get(), srcStep, expected.ycrcb, width * 2, height);

  Npp8u *dst[3] = {y.get(), cb.get(), cr.get()};
  int dstSteps[3] = {yStep, cbStep, crStep};
  ManagedStreamGuard managedStream;
  ASSERT_NE(managedStream.get(), nullptr);
  ASSERT_EQ(nppiYCrCb422ToYCbCr422_8u_C2P3R(src.get(), srcStep, dst, dstSteps, roi), NPP_SUCCESS);
  ASSERT_EQ(cudaStreamSynchronize(managedStream.get()), cudaSuccess);
  expectImage(y.get(), yStep, width, height, expected.y);
  expectImage(cb.get(), cbStep, width / 2, height, expected.cb);
  expectImage(cr.get(), crStep, width / 2, height, expected.cr);

  StreamGuard stream;
  ASSERT_NE(stream.get(), nullptr);
  const NppStreamContext context = streamContext(stream.get());
  clearBuffer(y, yStep, height);
  clearBuffer(cb, cbStep, height);
  clearBuffer(cr, crStep, height);
  ASSERT_EQ(nppiYCrCb422ToYCbCr422_8u_C2P3R_Ctx(src.get(), srcStep, dst, dstSteps, roi, context), NPP_SUCCESS);
  ASSERT_EQ(cudaStreamSynchronize(stream.get()), cudaSuccess);
  expectImage(y.get(), yStep, width, height, expected.y);
  expectImage(cb.get(), cbStep, width / 2, height, expected.cb);
  expectImage(cr.get(), crStep, width / 2, height, expected.cr);
}

TEST_F(YCbCr422LayoutTest, ValidatesEveryPointerStepAndRoi) {
  constexpr int width = 6;
  constexpr int height = 3;
  constexpr int packedStep = width * 2;
  constexpr int yStep = width;
  constexpr int cbStep = width / 2;
  constexpr int crStep = width / 2;
  const NppiSize roi{width, height};

  DeviceBuffer packed(static_cast<size_t>(packedStep) * height);
  DeviceBuffer packedDst(static_cast<size_t>(packedStep) * height);
  DeviceBuffer y(static_cast<size_t>(yStep) * height);
  DeviceBuffer cb(static_cast<size_t>(cbStep) * height);
  DeviceBuffer cr(static_cast<size_t>(crStep) * height);
  ASSERT_NE(packed.get(), nullptr);
  ASSERT_NE(packedDst.get(), nullptr);
  ASSERT_NE(y.get(), nullptr);
  ASSERT_NE(cb.get(), nullptr);
  ASSERT_NE(cr.get(), nullptr);

  Npp8u *dst[3] = {y.get(), cb.get(), cr.get()};
  int dstSteps[3] = {yStep, cbStep, crStep};
  const Npp8u *src[3] = {y.get(), cb.get(), cr.get()};
  int srcSteps[3] = {yStep, cbStep, crStep};
  NppStreamContext context{};

  EXPECT_EQ(nppiYCbCr422_8u_C2P3R(nullptr, packedStep, dst, dstSteps, roi), NPP_NULL_POINTER_ERROR);
  EXPECT_EQ(nppiYCbCr422_8u_C2P3R(packed.get(), packedStep, nullptr, dstSteps, roi), NPP_NULL_POINTER_ERROR);
  EXPECT_EQ(nppiYCbCr422_8u_C2P3R(packed.get(), packedStep, dst, nullptr, roi), NPP_NULL_POINTER_ERROR);
  for (int plane = 0; plane < 3; ++plane) {
    SCOPED_TRACE(::testing::Message() << "destination plane " << plane);
    Npp8u *badDst[3] = {y.get(), cb.get(), cr.get()};
    badDst[plane] = nullptr;
    EXPECT_EQ(nppiYCbCr422_8u_C2P3R_Ctx(packed.get(), packedStep, badDst, dstSteps, roi, context),
              NPP_NULL_POINTER_ERROR);

    int badSteps[3] = {yStep, cbStep, crStep};
    --badSteps[plane];
    EXPECT_EQ(nppiYCbCr422_8u_C2P3R(packed.get(), packedStep, dst, badSteps, roi), NPP_STEP_ERROR);
  }
  EXPECT_EQ(nppiYCbCr422_8u_C2P3R(packed.get(), packedStep - 1, dst, dstSteps, roi), NPP_STEP_ERROR);

  EXPECT_EQ(nppiYCbCr422_8u_P3C2R(nullptr, srcSteps, packedDst.get(), packedStep, roi), NPP_NULL_POINTER_ERROR);
  EXPECT_EQ(nppiYCbCr422_8u_P3C2R(src, nullptr, packedDst.get(), packedStep, roi), NPP_NULL_POINTER_ERROR);
  EXPECT_EQ(nppiYCbCr422_8u_P3C2R(src, srcSteps, nullptr, packedStep, roi), NPP_NULL_POINTER_ERROR);
  for (int plane = 0; plane < 3; ++plane) {
    SCOPED_TRACE(::testing::Message() << "source plane " << plane);
    const Npp8u *badSrc[3] = {y.get(), cb.get(), cr.get()};
    badSrc[plane] = nullptr;
    EXPECT_EQ(nppiYCbCr422_8u_P3C2R_Ctx(badSrc, srcSteps, packedDst.get(), packedStep, roi, context),
              NPP_NULL_POINTER_ERROR);

    int badSteps[3] = {yStep, cbStep, crStep};
    --badSteps[plane];
    EXPECT_EQ(nppiYCbCr422ToYCrCb422_8u_P3C2R(src, badSteps, packedDst.get(), packedStep, roi), NPP_STEP_ERROR);
  }
  EXPECT_EQ(nppiYCbCr422_8u_P3C2R(src, srcSteps, packedDst.get(), packedStep - 1, roi), NPP_STEP_ERROR);

  EXPECT_EQ(nppiYCbCr422ToYCrCb422_8u_C2R(nullptr, packedStep, packedDst.get(), packedStep, roi),
            NPP_NULL_POINTER_ERROR);
  EXPECT_EQ(nppiYCbCr422ToYCrCb422_8u_C2R(packed.get(), packedStep, nullptr, packedStep, roi),
            NPP_NULL_POINTER_ERROR);
  EXPECT_EQ(nppiYCbCr422ToYCrCb422_8u_C2R(packed.get(), packedStep - 1, packedDst.get(), packedStep, roi),
            NPP_STEP_ERROR);
  EXPECT_EQ(nppiYCbCr422ToYCrCb422_8u_C2R(packed.get(), packedStep, packedDst.get(), packedStep - 1, roi),
            NPP_STEP_ERROR);

  EXPECT_EQ(nppiYCrCb422ToYCbCr422_8u_C2P3R(packed.get(), packedStep, dst, nullptr, roi),
            NPP_NULL_POINTER_ERROR);
  EXPECT_EQ(nppiYCbCr422_8u_C2P3R(packed.get(), packedStep, dst, dstSteps, {width - 1, height}),
            NPP_WRONG_INTERSECTION_ROI_ERROR);
  EXPECT_EQ(nppiYCbCr422_8u_P3C2R(src, srcSteps, packedDst.get(), packedStep, {width - 1, height}),
            NPP_WRONG_INTERSECTION_ROI_ERROR);
  EXPECT_EQ(nppiYCbCr422ToYCrCb422_8u_C2R_Ctx(packed.get(), packedStep, packedDst.get(), packedStep,
                                              {width - 1, height}, context),
            NPP_WRONG_INTERSECTION_ROI_ERROR);

  EXPECT_EQ(nppiYCbCr422_8u_C2P3R(packed.get(), packedStep, dst, dstSteps, {0, height}), NPP_SIZE_ERROR);
  EXPECT_EQ(nppiYCbCr422_8u_C2P3R(packed.get(), packedStep, dst, dstSteps, {width, 0}), NPP_SIZE_ERROR);
  EXPECT_EQ(nppiYCbCr422_8u_P3C2R(src, srcSteps, packedDst.get(), packedStep, {-2, height}), NPP_SIZE_ERROR);
  EXPECT_EQ(nppiYCbCr422ToYCrCb422_8u_C2R(packed.get(), packedStep, packedDst.get(), packedStep, {width, -1}),
            NPP_SIZE_ERROR);
}

} // namespace
