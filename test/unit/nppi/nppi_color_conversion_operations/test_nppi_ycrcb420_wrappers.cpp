#include "npp.h"

#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <vector>

namespace {

class DeviceBuffer {
public:
  explicit DeviceBuffer(size_t bytes) : bytes_(bytes) {
    if (cudaMalloc(reinterpret_cast<void **>(&data_), bytes_) != cudaSuccess) {
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
  size_t bytes_ = 0;
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
    if (cudaStreamCreate(&stream_) != cudaSuccess || nppSetStream(stream_) != NPP_SUCCESS) {
      stream_ = nullptr;
    }
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

void copyToDevice2D(Npp8u *dst, int dstStep, const std::vector<Npp8u> &src, int rowBytes, int rows) {
  ASSERT_EQ(cudaMemcpy2D(dst, dstStep, src.data(), rowBytes, rowBytes, rows, cudaMemcpyHostToDevice), cudaSuccess);
}

std::vector<Npp8u> copyFromDevice2D(const Npp8u *src, int srcStep, int rowBytes, int rows) {
  std::vector<Npp8u> result(static_cast<size_t>(rowBytes) * rows);
  EXPECT_EQ(cudaMemcpy2D(result.data(), rowBytes, src, srcStep, rowBytes, rows, cudaMemcpyDeviceToHost), cudaSuccess);
  return result;
}

void expectPlane(const Npp8u *src, int srcStep, int rowBytes, int rows, const std::vector<Npp8u> &expected) {
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

class YCrCb420WrappersTest : public ::testing::Test {
protected:
  void SetUp() override { ASSERT_EQ(cudaSetDevice(0), cudaSuccess); }
  void TearDown() override { EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess); }
};

TEST_F(YCrCb420WrappersTest, BGRAndBGRAProduceYCrCbPlanesWithIndependentSteps) {
  constexpr int width = 4;
  constexpr int height = 4;
  constexpr int srcC3Step = width * 3 + 5;
  constexpr int srcAC4Step = width * 4 + 5;
  constexpr int yStep = width + 3;
  constexpr int crStep = width / 2 + 3;
  constexpr int cbStep = width / 2 + 5;
  const NppiSize roi{width, height};

  std::vector<Npp8u> bgr(static_cast<size_t>(width) * height * 3);
  std::vector<Npp8u> bgra(static_cast<size_t>(width) * height * 4);
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      const Npp8u r = static_cast<Npp8u>(10 + x * 30 + y * 7);
      const Npp8u g = static_cast<Npp8u>(20 + x * 12 + y * 11);
      const Npp8u b = static_cast<Npp8u>(40 + x * 5 + y * 13);
      const int c3 = (y * width + x) * 3;
      bgr[c3] = b;
      bgr[c3 + 1] = g;
      bgr[c3 + 2] = r;
      const int ac4 = (y * width + x) * 4;
      bgra[ac4] = b;
      bgra[ac4 + 1] = g;
      bgra[ac4 + 2] = r;
      bgra[ac4 + 3] = static_cast<Npp8u>(17 + x + y * width);
    }
  }

  const std::vector<Npp8u> expectedY = {32, 46, 61, 75, 41, 55, 69, 83,
                                        49, 64, 78, 92, 58, 72, 86, 101};
  const std::vector<Npp8u> expectedCr = {125, 141, 121, 138};
  const std::vector<Npp8u> expectedCb = {135, 124, 138, 127};

  DeviceBuffer srcC3(static_cast<size_t>(srcC3Step) * height);
  DeviceBuffer srcAC4(static_cast<size_t>(srcAC4Step) * height);
  DeviceBuffer dstY(static_cast<size_t>(yStep) * height);
  DeviceBuffer dstCr(static_cast<size_t>(crStep) * (height / 2));
  DeviceBuffer dstCb(static_cast<size_t>(cbStep) * (height / 2));
  ASSERT_NE(srcC3.get(), nullptr);
  ASSERT_NE(srcAC4.get(), nullptr);
  ASSERT_NE(dstY.get(), nullptr);
  ASSERT_NE(dstCr.get(), nullptr);
  ASSERT_NE(dstCb.get(), nullptr);
  copyToDevice2D(srcC3.get(), srcC3Step, bgr, width * 3, height);
  copyToDevice2D(srcAC4.get(), srcAC4Step, bgra, width * 4, height);

  ManagedStreamGuard managedStream;
  ASSERT_NE(managedStream.get(), nullptr);
  Npp8u *dst[3] = {dstY.get(), dstCr.get(), dstCb.get()};
  int dstSteps[3] = {yStep, crStep, cbStep};
  ASSERT_EQ(nppiBGRToYCrCb420_8u_C3P3R(srcC3.get(), srcC3Step, dst, dstSteps, roi), NPP_SUCCESS);
  ASSERT_EQ(cudaStreamSynchronize(managedStream.get()), cudaSuccess);
  expectPlane(dstY.get(), yStep, width, height, expectedY);
  expectPlane(dstCr.get(), crStep, width / 2, height / 2, expectedCr);
  expectPlane(dstCb.get(), cbStep, width / 2, height / 2, expectedCb);

  StreamGuard stream;
  ASSERT_NE(stream.get(), nullptr);
  NppStreamContext context = streamContext(stream.get());
  clearBuffer(dstY, yStep, height);
  clearBuffer(dstCr, crStep, height / 2);
  clearBuffer(dstCb, cbStep, height / 2);
  ASSERT_EQ(nppiBGRToYCrCb420_8u_C3P3R_Ctx(srcC3.get(), srcC3Step, dst, dstSteps, roi, context), NPP_SUCCESS);
  ASSERT_EQ(cudaStreamSynchronize(stream.get()), cudaSuccess);
  expectPlane(dstY.get(), yStep, width, height, expectedY);
  expectPlane(dstCr.get(), crStep, width / 2, height / 2, expectedCr);
  expectPlane(dstCb.get(), cbStep, width / 2, height / 2, expectedCb);

  clearBuffer(dstY, yStep, height);
  clearBuffer(dstCr, crStep, height / 2);
  clearBuffer(dstCb, cbStep, height / 2);
  ASSERT_EQ(nppiBGRToYCrCb420_8u_AC4P3R(srcAC4.get(), srcAC4Step, dst, dstSteps, roi), NPP_SUCCESS);
  ASSERT_EQ(cudaStreamSynchronize(managedStream.get()), cudaSuccess);
  expectPlane(dstY.get(), yStep, width, height, expectedY);
  expectPlane(dstCr.get(), crStep, width / 2, height / 2, expectedCr);
  expectPlane(dstCb.get(), cbStep, width / 2, height / 2, expectedCb);

  clearBuffer(dstY, yStep, height);
  clearBuffer(dstCr, crStep, height / 2);
  clearBuffer(dstCb, cbStep, height / 2);
  ASSERT_EQ(nppiBGRToYCrCb420_8u_AC4P3R_Ctx(srcAC4.get(), srcAC4Step, dst, dstSteps, roi, context), NPP_SUCCESS);
  ASSERT_EQ(cudaStreamSynchronize(stream.get()), cudaSuccess);
  expectPlane(dstY.get(), yStep, width, height, expectedY);
  expectPlane(dstCr.get(), crStep, width / 2, height / 2, expectedCr);
  expectPlane(dstCb.get(), cbStep, width / 2, height / 2, expectedCb);
}

TEST_F(YCrCb420WrappersTest, P2AndP3LayoutsRoundTripByteForByteWithIndependentSteps) {
  constexpr int width = 6;
  constexpr int height = 4;
  constexpr int srcYStep = width + 3;
  constexpr int srcCbCrStep = width + 5;
  constexpr int dstYStep = width + 7;
  constexpr int dstCrStep = width / 2 + 3;
  constexpr int dstCbStep = width / 2 + 5;
  constexpr int mergedYStep = width + 9;
  constexpr int mergedCbCrStep = width + 11;
  const NppiSize roi{width, height};

  std::vector<Npp8u> y(static_cast<size_t>(width) * height);
  std::vector<Npp8u> cbcr(static_cast<size_t>(width) * (height / 2));
  std::vector<Npp8u> expectedCr(static_cast<size_t>(width / 2) * (height / 2));
  std::vector<Npp8u> expectedCb(static_cast<size_t>(width / 2) * (height / 2));
  for (size_t i = 0; i < y.size(); ++i) {
    y[i] = static_cast<Npp8u>(19 + i * 3);
  }
  for (int row = 0; row < height / 2; ++row) {
    for (int x = 0; x < width / 2; ++x) {
      const Npp8u cb = static_cast<Npp8u>(31 + row * 17 + x * 3);
      const Npp8u cr = static_cast<Npp8u>(181 + row * 11 + x * 5);
      cbcr[static_cast<size_t>(row) * width + x * 2] = cb;
      cbcr[static_cast<size_t>(row) * width + x * 2 + 1] = cr;
      expectedCb[static_cast<size_t>(row) * (width / 2) + x] = cb;
      expectedCr[static_cast<size_t>(row) * (width / 2) + x] = cr;
    }
  }

  DeviceBuffer srcY(static_cast<size_t>(srcYStep) * height);
  DeviceBuffer srcCbCr(static_cast<size_t>(srcCbCrStep) * (height / 2));
  DeviceBuffer dstY(static_cast<size_t>(dstYStep) * height);
  DeviceBuffer dstCr(static_cast<size_t>(dstCrStep) * (height / 2));
  DeviceBuffer dstCb(static_cast<size_t>(dstCbStep) * (height / 2));
  DeviceBuffer mergedY(static_cast<size_t>(mergedYStep) * height);
  DeviceBuffer mergedCbCr(static_cast<size_t>(mergedCbCrStep) * (height / 2));
  ASSERT_NE(srcY.get(), nullptr);
  ASSERT_NE(srcCbCr.get(), nullptr);
  ASSERT_NE(dstY.get(), nullptr);
  ASSERT_NE(dstCr.get(), nullptr);
  ASSERT_NE(dstCb.get(), nullptr);
  ASSERT_NE(mergedY.get(), nullptr);
  ASSERT_NE(mergedCbCr.get(), nullptr);
  copyToDevice2D(srcY.get(), srcYStep, y, width, height);
  copyToDevice2D(srcCbCr.get(), srcCbCrStep, cbcr, width, height / 2);

  ManagedStreamGuard managedStream;
  ASSERT_NE(managedStream.get(), nullptr);
  Npp8u *dst[3] = {dstY.get(), dstCr.get(), dstCb.get()};
  int dstSteps[3] = {dstYStep, dstCrStep, dstCbStep};
  ASSERT_EQ(nppiYCbCr420ToYCrCb420_8u_P2P3R(srcY.get(), srcYStep, srcCbCr.get(), srcCbCrStep, dst, dstSteps,
                                            roi),
            NPP_SUCCESS);
  ASSERT_EQ(cudaStreamSynchronize(managedStream.get()), cudaSuccess);
  expectPlane(dstY.get(), dstYStep, width, height, y);
  expectPlane(dstCr.get(), dstCrStep, width / 2, height / 2, expectedCr);
  expectPlane(dstCb.get(), dstCbStep, width / 2, height / 2, expectedCb);

  StreamGuard stream;
  ASSERT_NE(stream.get(), nullptr);
  NppStreamContext context = streamContext(stream.get());
  clearBuffer(dstY, dstYStep, height);
  clearBuffer(dstCr, dstCrStep, height / 2);
  clearBuffer(dstCb, dstCbStep, height / 2);
  ASSERT_EQ(nppiYCbCr420ToYCrCb420_8u_P2P3R_Ctx(srcY.get(), srcYStep, srcCbCr.get(), srcCbCrStep, dst, dstSteps,
                                                roi, context),
            NPP_SUCCESS);
  ASSERT_EQ(cudaStreamSynchronize(stream.get()), cudaSuccess);
  expectPlane(dstY.get(), dstYStep, width, height, y);
  expectPlane(dstCr.get(), dstCrStep, width / 2, height / 2, expectedCr);
  expectPlane(dstCb.get(), dstCbStep, width / 2, height / 2, expectedCb);

  const Npp8u *src[3] = {dstY.get(), dstCr.get(), dstCb.get()};
  int srcSteps[3] = {dstYStep, dstCrStep, dstCbStep};
  ASSERT_EQ(nppiYCrCb420ToYCbCr420_8u_P3P2R(src, srcSteps, mergedY.get(), mergedYStep, mergedCbCr.get(),
                                            mergedCbCrStep, roi),
            NPP_SUCCESS);
  ASSERT_EQ(cudaStreamSynchronize(managedStream.get()), cudaSuccess);
  expectPlane(mergedY.get(), mergedYStep, width, height, y);
  expectPlane(mergedCbCr.get(), mergedCbCrStep, width, height / 2, cbcr);

  clearBuffer(mergedY, mergedYStep, height);
  clearBuffer(mergedCbCr, mergedCbCrStep, height / 2);
  ASSERT_EQ(nppiYCrCb420ToYCbCr420_8u_P3P2R_Ctx(src, srcSteps, mergedY.get(), mergedYStep, mergedCbCr.get(),
                                                mergedCbCrStep, roi, context),
            NPP_SUCCESS);
  ASSERT_EQ(cudaStreamSynchronize(stream.get()), cudaSuccess);
  expectPlane(mergedY.get(), mergedYStep, width, height, y);
  expectPlane(mergedCbCr.get(), mergedCbCrStep, width, height / 2, cbcr);
}

TEST_F(YCrCb420WrappersTest, ValidatesPointersStepsAndEvenRoi) {
  constexpr int width = 4;
  constexpr int height = 4;
  constexpr int srcC3Step = width * 3;
  constexpr int srcAC4Step = width * 4;
  constexpr int yStep = width;
  constexpr int chromaStep = width / 2;
  constexpr int cbcrStep = width;
  const NppiSize roi{width, height};

  DeviceBuffer srcC3(static_cast<size_t>(srcC3Step) * height);
  DeviceBuffer srcAC4(static_cast<size_t>(srcAC4Step) * height);
  DeviceBuffer y(static_cast<size_t>(yStep) * height);
  DeviceBuffer cr(static_cast<size_t>(chromaStep) * (height / 2));
  DeviceBuffer cb(static_cast<size_t>(chromaStep) * (height / 2));
  DeviceBuffer cbcr(static_cast<size_t>(cbcrStep) * (height / 2));
  ASSERT_NE(srcC3.get(), nullptr);
  ASSERT_NE(srcAC4.get(), nullptr);
  ASSERT_NE(y.get(), nullptr);
  ASSERT_NE(cr.get(), nullptr);
  ASSERT_NE(cb.get(), nullptr);
  ASSERT_NE(cbcr.get(), nullptr);

  Npp8u *dst[3] = {y.get(), cr.get(), cb.get()};
  int dstSteps[3] = {yStep, chromaStep, chromaStep};
  const Npp8u *src[3] = {y.get(), cr.get(), cb.get()};
  int srcSteps[3] = {yStep, chromaStep, chromaStep};
  NppStreamContext context{};

  EXPECT_EQ(nppiBGRToYCrCb420_8u_C3P3R(nullptr, srcC3Step, dst, dstSteps, roi), NPP_NULL_POINTER_ERROR);
  EXPECT_EQ(nppiBGRToYCrCb420_8u_C3P3R(srcC3.get(), srcC3Step, nullptr, dstSteps, roi),
            NPP_NULL_POINTER_ERROR);
  EXPECT_EQ(nppiBGRToYCrCb420_8u_C3P3R(srcC3.get(), srcC3Step, dst, nullptr, roi), NPP_STEP_ERROR);
  Npp8u *nullDst[3] = {y.get(), nullptr, cb.get()};
  EXPECT_EQ(nppiBGRToYCrCb420_8u_AC4P3R_Ctx(srcAC4.get(), srcAC4Step, nullDst, dstSteps, roi, context),
            NPP_NULL_POINTER_ERROR);
  EXPECT_EQ(nppiYCbCr420ToYCrCb420_8u_P2P3R(nullptr, yStep, cbcr.get(), cbcrStep, dst, dstSteps, roi),
            NPP_NULL_POINTER_ERROR);
  EXPECT_EQ(nppiYCbCr420ToYCrCb420_8u_P2P3R(y.get(), yStep, nullptr, cbcrStep, dst, dstSteps, roi),
            NPP_NULL_POINTER_ERROR);
  EXPECT_EQ(nppiYCbCr420ToYCrCb420_8u_P2P3R(y.get(), yStep, cbcr.get(), cbcrStep, nullptr, dstSteps, roi),
            NPP_NULL_POINTER_ERROR);
  EXPECT_EQ(nppiYCbCr420ToYCrCb420_8u_P2P3R(y.get(), yStep, cbcr.get(), cbcrStep, dst, nullptr, roi),
            NPP_NULL_POINTER_ERROR);
  const Npp8u *nullSrc[3] = {y.get(), cr.get(), nullptr};
  EXPECT_EQ(nppiYCrCb420ToYCbCr420_8u_P3P2R_Ctx(nullSrc, srcSteps, y.get(), yStep, cbcr.get(), cbcrStep, roi,
                                                context),
            NPP_NULL_POINTER_ERROR);
  EXPECT_EQ(nppiYCrCb420ToYCbCr420_8u_P3P2R(nullptr, srcSteps, y.get(), yStep, cbcr.get(), cbcrStep, roi),
            NPP_NULL_POINTER_ERROR);
  EXPECT_EQ(nppiYCrCb420ToYCbCr420_8u_P3P2R(src, nullptr, y.get(), yStep, cbcr.get(), cbcrStep, roi),
            NPP_NULL_POINTER_ERROR);
  EXPECT_EQ(nppiYCrCb420ToYCbCr420_8u_P3P2R(src, srcSteps, nullptr, yStep, cbcr.get(), cbcrStep, roi),
            NPP_NULL_POINTER_ERROR);
  EXPECT_EQ(nppiYCrCb420ToYCbCr420_8u_P3P2R(src, srcSteps, y.get(), yStep, nullptr, cbcrStep, roi),
            NPP_NULL_POINTER_ERROR);

  EXPECT_EQ(nppiBGRToYCrCb420_8u_C3P3R(srcC3.get(), srcC3Step - 1, dst, dstSteps, roi), NPP_STEP_ERROR);
  EXPECT_EQ(nppiBGRToYCrCb420_8u_AC4P3R(srcAC4.get(), srcAC4Step - 1, dst, dstSteps, roi), NPP_STEP_ERROR);
  int badDstSteps[3] = {yStep, chromaStep - 1, chromaStep};
  EXPECT_EQ(nppiYCbCr420ToYCrCb420_8u_P2P3R(srcC3.get(), yStep, cbcr.get(), cbcrStep, dst, badDstSteps, roi),
            NPP_STEP_ERROR);
  int badSrcSteps[3] = {yStep, chromaStep, chromaStep - 1};
  EXPECT_EQ(nppiYCrCb420ToYCbCr420_8u_P3P2R(src, badSrcSteps, y.get(), yStep, cbcr.get(), cbcrStep, roi),
            NPP_STEP_ERROR);
  EXPECT_EQ(nppiYCbCr420ToYCrCb420_8u_P2P3R(y.get(), yStep - 1, cbcr.get(), cbcrStep, dst, dstSteps, roi),
            NPP_STEP_ERROR);
  EXPECT_EQ(nppiYCbCr420ToYCrCb420_8u_P2P3R(y.get(), yStep, cbcr.get(), cbcrStep - 1, dst, dstSteps, roi),
            NPP_STEP_ERROR);
  EXPECT_EQ(nppiYCrCb420ToYCbCr420_8u_P3P2R(src, srcSteps, y.get(), yStep - 1, cbcr.get(), cbcrStep, roi),
            NPP_STEP_ERROR);
  EXPECT_EQ(nppiYCrCb420ToYCbCr420_8u_P3P2R(src, srcSteps, y.get(), yStep, cbcr.get(), cbcrStep - 1, roi),
            NPP_STEP_ERROR);

  EXPECT_EQ(nppiBGRToYCrCb420_8u_C3P3R(srcC3.get(), srcC3Step, dst, dstSteps, {0, height}), NPP_SIZE_ERROR);
  EXPECT_EQ(nppiBGRToYCrCb420_8u_AC4P3R(srcAC4.get(), srcAC4Step, dst, dstSteps, {width, -1}), NPP_SIZE_ERROR);
  EXPECT_EQ(nppiYCbCr420ToYCrCb420_8u_P2P3R(y.get(), yStep, cbcr.get(), cbcrStep, dst, dstSteps, {0, height}),
            NPP_SIZE_ERROR);
  EXPECT_EQ(nppiYCrCb420ToYCbCr420_8u_P3P2R(src, srcSteps, y.get(), yStep, cbcr.get(), cbcrStep, {width, -1}),
            NPP_SIZE_ERROR);

  EXPECT_EQ(nppiBGRToYCrCb420_8u_C3P3R(srcC3.get(), srcC3Step, dst, dstSteps, {width - 1, height}),
            NPP_WRONG_INTERSECTION_ROI_ERROR);
  EXPECT_EQ(nppiBGRToYCrCb420_8u_AC4P3R_Ctx(srcAC4.get(), srcAC4Step, dst, dstSteps, {width, height - 1}, context),
            NPP_WRONG_INTERSECTION_ROI_ERROR);
  EXPECT_EQ(nppiYCbCr420ToYCrCb420_8u_P2P3R(srcC3.get(), yStep, cbcr.get(), cbcrStep, dst, dstSteps,
                                            {width - 1, height}),
            NPP_WRONG_INTERSECTION_ROI_ERROR);
  EXPECT_EQ(nppiYCrCb420ToYCbCr420_8u_P3P2R_Ctx(src, srcSteps, y.get(), yStep, cbcr.get(), cbcrStep,
                                                {width, height - 1}, context),
            NPP_WRONG_INTERSECTION_ROI_ERROR);
}

} // namespace
