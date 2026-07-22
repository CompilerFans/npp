#include "npp.h"

#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <vector>

namespace {

class YCbCr420LayoutTest : public ::testing::Test {
protected:
  void SetUp() override {
    int deviceCount = 0;
    if (cudaGetDeviceCount(&deviceCount) != cudaSuccess || deviceCount <= 0) {
      GTEST_SKIP() << "CUDA device not available";
    }
    width = 8;
    height = 4;
  }

  void TearDown() override {
    for (Npp8u *ptr : allocations) {
      if (ptr) {
        nppiFree(ptr);
      }
    }
    allocations.clear();
  }

  Npp8u *allocatePlane(int planeWidth, int planeHeight, int &step) {
    Npp8u *ptr = nppiMalloc_8u_C1(planeWidth, planeHeight, &step);
    EXPECT_NE(ptr, nullptr);
    allocations.push_back(ptr);
    return ptr;
  }

  void copyHostToDevice2D(Npp8u *dst, int dstStep, const std::vector<Npp8u> &src, int rowBytes, int rows) const {
    ASSERT_EQ(cudaMemcpy2D(dst, dstStep, src.data(), rowBytes, rowBytes, rows, cudaMemcpyHostToDevice), cudaSuccess);
  }

  std::vector<Npp8u> copyDeviceToHost2D(const Npp8u *src, int srcStep, int rowBytes, int rows) const {
    std::vector<Npp8u> host(static_cast<size_t>(rowBytes) * rows);
    EXPECT_EQ(cudaMemcpy2D(host.data(), rowBytes, src, srcStep, rowBytes, rows, cudaMemcpyDeviceToHost), cudaSuccess);
    return host;
  }

  std::vector<Npp8u> makeYPlane() const {
    std::vector<Npp8u> plane(static_cast<size_t>(width) * height);
    for (int y = 0; y < height; ++y) {
      for (int x = 0; x < width; ++x) {
        plane[y * width + x] = static_cast<Npp8u>(16 + y * 11 + x * 3);
      }
    }
    return plane;
  }

  std::vector<Npp8u> makeCbPlane420() const {
    const int chromaWidth = width / 2;
    const int chromaHeight = height / 2;
    std::vector<Npp8u> plane(static_cast<size_t>(chromaWidth) * chromaHeight);
    for (int y = 0; y < chromaHeight; ++y) {
      for (int x = 0; x < chromaWidth; ++x) {
        plane[y * chromaWidth + x] = static_cast<Npp8u>(70 + y * 9 + x * 5);
      }
    }
    return plane;
  }

  std::vector<Npp8u> makeCrPlane420() const {
    const int chromaWidth = width / 2;
    const int chromaHeight = height / 2;
    std::vector<Npp8u> plane(static_cast<size_t>(chromaWidth) * chromaHeight);
    for (int y = 0; y < chromaHeight; ++y) {
      for (int x = 0; x < chromaWidth; ++x) {
        plane[y * chromaWidth + x] = static_cast<Npp8u>(130 + y * 7 + x * 4);
      }
    }
    return plane;
  }

  std::vector<Npp8u> interleaveChroma420(const std::vector<Npp8u> &chroma0, const std::vector<Npp8u> &chroma1) const {
    const int chromaWidth = width / 2;
    const int chromaHeight = height / 2;
    std::vector<Npp8u> plane(static_cast<size_t>(width) * chromaHeight);
    for (int y = 0; y < chromaHeight; ++y) {
      for (int x = 0; x < chromaWidth; ++x) {
        int srcIndex = y * chromaWidth + x;
        int dstIndex = y * width + (x << 1);
        plane[dstIndex] = chroma0[srcIndex];
        plane[dstIndex + 1] = chroma1[srcIndex];
      }
    }
    return plane;
  }

  std::vector<Npp8u> upsample420To422Chroma(const std::vector<Npp8u> &plane420) const {
    const int chromaWidth = width / 2;
    std::vector<Npp8u> plane422(static_cast<size_t>(chromaWidth) * height);
    for (int y = 0; y < height; ++y) {
      int srcY = y >> 1;
      for (int x = 0; x < chromaWidth; ++x) {
        plane422[y * chromaWidth + x] = plane420[srcY * chromaWidth + x];
      }
    }
    return plane422;
  }

  std::vector<Npp8u> packYCbCr422(const std::vector<Npp8u> &yPlane, const std::vector<Npp8u> &cbPlane422,
                                  const std::vector<Npp8u> &crPlane422) const {
    std::vector<Npp8u> packed(static_cast<size_t>(width) * height * 2);
    const int chromaWidth = width / 2;
    for (int y = 0; y < height; ++y) {
      for (int x = 0; x < chromaWidth; ++x) {
        int dstIndex = (y * width + (x << 1)) * 2;
        packed[dstIndex] = yPlane[y * width + (x << 1)];
        packed[dstIndex + 1] = cbPlane422[y * chromaWidth + x];
        packed[dstIndex + 2] = yPlane[y * width + (x << 1) + 1];
        packed[dstIndex + 3] = crPlane422[y * chromaWidth + x];
      }
    }
    return packed;
  }

  std::vector<Npp8u> packCbYCr422(const std::vector<Npp8u> &yPlane, const std::vector<Npp8u> &cbPlane422,
                                  const std::vector<Npp8u> &crPlane422) const {
    std::vector<Npp8u> packed(static_cast<size_t>(width) * height * 2);
    const int chromaWidth = width / 2;
    for (int y = 0; y < height; ++y) {
      for (int x = 0; x < chromaWidth; ++x) {
        int dstIndex = (y * width + (x << 1)) * 2;
        packed[dstIndex] = cbPlane422[y * chromaWidth + x];
        packed[dstIndex + 1] = yPlane[y * width + (x << 1)];
        packed[dstIndex + 2] = crPlane422[y * chromaWidth + x];
        packed[dstIndex + 3] = yPlane[y * width + (x << 1) + 1];
      }
    }
    return packed;
  }

  std::vector<Npp8u> downsample420To411Plane(const std::vector<Npp8u> &plane420) const {
    const int chromaWidth420 = width / 2;
    const int chromaWidth411 = width / 4;
    std::vector<Npp8u> plane411(static_cast<size_t>(chromaWidth411) * height);
    for (int y = 0; y < height; ++y) {
      int srcY = y >> 1;
      for (int x = 0; x < chromaWidth411; ++x) {
        const int srcIndex = srcY * chromaWidth420 + (x << 1) + (y & 1);
        plane411[y * chromaWidth411 + x] = plane420[srcIndex];
      }
    }
    return plane411;
  }

  std::vector<Npp8u> interleaveChroma411(const std::vector<Npp8u> &cbPlane411, const std::vector<Npp8u> &crPlane411) const {
    const int chromaWidth = width / 4;
    std::vector<Npp8u> plane(static_cast<size_t>(width / 2) * height);
    for (int y = 0; y < height; ++y) {
      for (int x = 0; x < chromaWidth; ++x) {
        int srcIndex = y * chromaWidth + x;
        int dstIndex = y * (width / 2) + (x << 1);
        plane[dstIndex] = cbPlane411[srcIndex];
        plane[dstIndex + 1] = crPlane411[srcIndex];
      }
    }
    return plane;
  }

  NppStreamContext streamContext() const {
    NppStreamContext ctx{};
    nppGetStreamContext(&ctx);
    ctx.hStream = 0;
    return ctx;
  }

  int width = 0;
  int height = 0;
  std::vector<Npp8u *> allocations;
};

} // namespace

TEST_F(YCbCr420LayoutTest, YCbCr420_8u_P3P2R_And_P2P3R_RoundTrip) {
  const std::vector<Npp8u> hostY = makeYPlane();
  const std::vector<Npp8u> hostCb = makeCbPlane420();
  const std::vector<Npp8u> hostCr = makeCrPlane420();
  const std::vector<Npp8u> expectedCbCr = interleaveChroma420(hostCb, hostCr);

  int srcYStep = 0, srcCbStep = 0, srcCrStep = 0;
  Npp8u *d_srcY = allocatePlane(width, height, srcYStep);
  Npp8u *d_srcCb = allocatePlane(width / 2, height / 2, srcCbStep);
  Npp8u *d_srcCr = allocatePlane(width / 2, height / 2, srcCrStep);
  copyHostToDevice2D(d_srcY, srcYStep, hostY, width, height);
  copyHostToDevice2D(d_srcCb, srcCbStep, hostCb, width / 2, height / 2);
  copyHostToDevice2D(d_srcCr, srcCrStep, hostCr, width / 2, height / 2);

  int dstYStep = 0, dstCbCrStep = 0;
  Npp8u *d_dstY = allocatePlane(width, height, dstYStep);
  Npp8u *d_dstCbCr = allocatePlane(width, height / 2, dstCbCrStep);

  const Npp8u *pSrc[3] = {d_srcY, d_srcCb, d_srcCr};
  int srcSteps[3] = {srcYStep, srcCbStep, srcCrStep};
  const NppiSize roi = {width, height};

  EXPECT_EQ(nppiYCbCr420_8u_P3P2R(pSrc, srcSteps, d_dstY, dstYStep, d_dstCbCr, dstCbCrStep, roi), NPP_NO_ERROR);
  EXPECT_EQ(copyDeviceToHost2D(d_dstY, dstYStep, width, height), hostY);
  EXPECT_EQ(copyDeviceToHost2D(d_dstCbCr, dstCbCrStep, width, height / 2), expectedCbCr);

  int roundTripYStep = 0, roundTripCbStep = 0, roundTripCrStep = 0;
  Npp8u *d_roundTripY = allocatePlane(width, height, roundTripYStep);
  Npp8u *d_roundTripCb = allocatePlane(width / 2, height / 2, roundTripCbStep);
  Npp8u *d_roundTripCr = allocatePlane(width / 2, height / 2, roundTripCrStep);
  Npp8u *pRoundTrip[3] = {d_roundTripY, d_roundTripCb, d_roundTripCr};
  int roundTripSteps[3] = {roundTripYStep, roundTripCbStep, roundTripCrStep};

  EXPECT_EQ(nppiYCbCr420_8u_P2P3R(d_dstY, dstYStep, d_dstCbCr, dstCbCrStep, pRoundTrip, roundTripSteps, roi),
            NPP_NO_ERROR);
  EXPECT_EQ(copyDeviceToHost2D(d_roundTripY, roundTripYStep, width, height), hostY);
  EXPECT_EQ(copyDeviceToHost2D(d_roundTripCb, roundTripCbStep, width / 2, height / 2), hostCb);
  EXPECT_EQ(copyDeviceToHost2D(d_roundTripCr, roundTripCrStep, width / 2, height / 2), hostCr);

  const NppStreamContext ctx = streamContext();
  EXPECT_EQ(nppiYCbCr420_8u_P3P2R_Ctx(pSrc, srcSteps, d_dstY, dstYStep, d_dstCbCr, dstCbCrStep, roi, ctx),
            NPP_NO_ERROR);
  EXPECT_EQ(nppiYCbCr420_8u_P2P3R_Ctx(d_dstY, dstYStep, d_dstCbCr, dstCbCrStep, pRoundTrip, roundTripSteps, roi, ctx),
            NPP_NO_ERROR);
}

TEST_F(YCbCr420LayoutTest, YCrCb420ToYCbCr420_And_YCbCr420ToYCrCb420_ReorderChroma) {
  const std::vector<Npp8u> hostY = makeYPlane();
  const std::vector<Npp8u> hostCb = makeCbPlane420();
  const std::vector<Npp8u> hostCr = makeCrPlane420();
  const std::vector<Npp8u> hostCbCr = interleaveChroma420(hostCb, hostCr);

  int srcYStep = 0, srcCrStep = 0, srcCbStep = 0;
  Npp8u *d_srcY = allocatePlane(width, height, srcYStep);
  Npp8u *d_srcCr = allocatePlane(width / 2, height / 2, srcCrStep);
  Npp8u *d_srcCb = allocatePlane(width / 2, height / 2, srcCbStep);
  copyHostToDevice2D(d_srcY, srcYStep, hostY, width, height);
  copyHostToDevice2D(d_srcCr, srcCrStep, hostCr, width / 2, height / 2);
  copyHostToDevice2D(d_srcCb, srcCbStep, hostCb, width / 2, height / 2);

  int dstYStep = 0, dstCbCrStep = 0;
  Npp8u *d_dstY = allocatePlane(width, height, dstYStep);
  Npp8u *d_dstCbCr = allocatePlane(width, height / 2, dstCbCrStep);

  const Npp8u *pYCrCb[3] = {d_srcY, d_srcCr, d_srcCb};
  int srcSteps[3] = {srcYStep, srcCrStep, srcCbStep};
  const NppStreamContext ctx = streamContext();
  const NppiSize roi = {width, height};

  EXPECT_EQ(nppiYCrCb420ToYCbCr420_8u_P3P2R(pYCrCb, srcSteps, d_dstY, dstYStep, d_dstCbCr, dstCbCrStep, roi),
            NPP_NO_ERROR);
  EXPECT_EQ(copyDeviceToHost2D(d_dstY, dstYStep, width, height), hostY);
  EXPECT_EQ(copyDeviceToHost2D(d_dstCbCr, dstCbCrStep, width, height / 2), hostCbCr);
  EXPECT_EQ(nppiYCrCb420ToYCbCr420_8u_P3P2R_Ctx(pYCrCb, srcSteps, d_dstY, dstYStep, d_dstCbCr, dstCbCrStep, roi, ctx),
            NPP_NO_ERROR);

  int dstCrStep = 0, dstCbStep = 0;
  Npp8u *d_outY = allocatePlane(width, height, dstYStep);
  Npp8u *d_outCr = allocatePlane(width / 2, height / 2, dstCrStep);
  Npp8u *d_outCb = allocatePlane(width / 2, height / 2, dstCbStep);
  Npp8u *pDst[3] = {d_outY, d_outCr, d_outCb};
  int dstSteps[3] = {dstYStep, dstCrStep, dstCbStep};

  EXPECT_EQ(nppiYCbCr420ToYCrCb420_8u_P2P3R(d_dstY, dstYStep, d_dstCbCr, dstCbCrStep, pDst, dstSteps, roi),
            NPP_NO_ERROR);
  EXPECT_EQ(copyDeviceToHost2D(d_outY, dstYStep, width, height), hostY);
  EXPECT_EQ(copyDeviceToHost2D(d_outCr, dstCrStep, width / 2, height / 2), hostCr);
  EXPECT_EQ(copyDeviceToHost2D(d_outCb, dstCbStep, width / 2, height / 2), hostCb);
  EXPECT_EQ(nppiYCbCr420ToYCrCb420_8u_P2P3R_Ctx(d_dstY, dstYStep, d_dstCbCr, dstCbCrStep, pDst, dstSteps, roi, ctx),
            NPP_NO_ERROR);
}

TEST_F(YCbCr420LayoutTest, YCbCr420ToYCbCr422_PlanarOutputs) {
  const std::vector<Npp8u> hostY = makeYPlane();
  const std::vector<Npp8u> hostCb420 = makeCbPlane420();
  const std::vector<Npp8u> hostCr420 = makeCrPlane420();
  const std::vector<Npp8u> hostCbCr420 = interleaveChroma420(hostCb420, hostCr420);
  const std::vector<Npp8u> expectedCb422 = upsample420To422Chroma(hostCb420);
  const std::vector<Npp8u> expectedCr422 = upsample420To422Chroma(hostCr420);

  int srcYStep = 0, srcCbStep = 0, srcCrStep = 0, srcCbCrStep = 0;
  Npp8u *d_srcY = allocatePlane(width, height, srcYStep);
  Npp8u *d_srcCb = allocatePlane(width / 2, height / 2, srcCbStep);
  Npp8u *d_srcCr = allocatePlane(width / 2, height / 2, srcCrStep);
  Npp8u *d_srcCbCr = allocatePlane(width, height / 2, srcCbCrStep);
  copyHostToDevice2D(d_srcY, srcYStep, hostY, width, height);
  copyHostToDevice2D(d_srcCb, srcCbStep, hostCb420, width / 2, height / 2);
  copyHostToDevice2D(d_srcCr, srcCrStep, hostCr420, width / 2, height / 2);
  copyHostToDevice2D(d_srcCbCr, srcCbCrStep, hostCbCr420, width, height / 2);

  int dstYStep = 0, dstCbStep = 0, dstCrStep = 0;
  Npp8u *d_dstY0 = allocatePlane(width, height, dstYStep);
  Npp8u *d_dstCb0 = allocatePlane(width / 2, height, dstCbStep);
  Npp8u *d_dstCr0 = allocatePlane(width / 2, height, dstCrStep);
  Npp8u *d_dstY1 = allocatePlane(width, height, dstYStep);
  Npp8u *d_dstCb1 = allocatePlane(width / 2, height, dstCbStep);
  Npp8u *d_dstCr1 = allocatePlane(width / 2, height, dstCrStep);
  const Npp8u *pSrc[3] = {d_srcY, d_srcCb, d_srcCr};
  Npp8u *pDst0[3] = {d_dstY0, d_dstCb0, d_dstCr0};
  Npp8u *pDst1[3] = {d_dstY1, d_dstCb1, d_dstCr1};
  int srcSteps[3] = {srcYStep, srcCbStep, srcCrStep};
  int dstSteps[3] = {dstYStep, dstCbStep, dstCrStep};
  const NppStreamContext ctx = streamContext();
  const NppiSize roi = {width, height};

  EXPECT_EQ(nppiYCbCr420ToYCbCr422_8u_P3R(pSrc, srcSteps, pDst0, dstSteps, roi), NPP_NO_ERROR);
  EXPECT_EQ(copyDeviceToHost2D(d_dstY0, dstYStep, width, height), hostY);
  EXPECT_EQ(copyDeviceToHost2D(d_dstCb0, dstCbStep, width / 2, height), expectedCb422);
  EXPECT_EQ(copyDeviceToHost2D(d_dstCr0, dstCrStep, width / 2, height), expectedCr422);

  EXPECT_EQ(nppiYCbCr420ToYCbCr422_8u_P2P3R(d_srcY, srcYStep, d_srcCbCr, srcCbCrStep, pDst1, dstSteps, roi),
            NPP_NO_ERROR);
  EXPECT_EQ(copyDeviceToHost2D(d_dstY1, dstYStep, width, height), hostY);
  EXPECT_EQ(copyDeviceToHost2D(d_dstCb1, dstCbStep, width / 2, height), expectedCb422);
  EXPECT_EQ(copyDeviceToHost2D(d_dstCr1, dstCrStep, width / 2, height), expectedCr422);

  EXPECT_EQ(nppiYCbCr420ToYCbCr422_8u_P3R_Ctx(pSrc, srcSteps, pDst0, dstSteps, roi, ctx), NPP_NO_ERROR);
  EXPECT_EQ(nppiYCbCr420ToYCbCr422_8u_P2P3R_Ctx(d_srcY, srcYStep, d_srcCbCr, srcCbCrStep, pDst1, dstSteps, roi, ctx),
            NPP_NO_ERROR);
}

TEST_F(YCbCr420LayoutTest, YCbCr420ToPacked422Outputs) {
  const std::vector<Npp8u> hostY = makeYPlane();
  const std::vector<Npp8u> hostCb420 = makeCbPlane420();
  const std::vector<Npp8u> hostCr420 = makeCrPlane420();
  const std::vector<Npp8u> hostCbCr420 = interleaveChroma420(hostCb420, hostCr420);
  const std::vector<Npp8u> hostCb422 = upsample420To422Chroma(hostCb420);
  const std::vector<Npp8u> hostCr422 = upsample420To422Chroma(hostCr420);
  const std::vector<Npp8u> expectedYCbCr422 = packYCbCr422(hostY, hostCb422, hostCr422);
  const std::vector<Npp8u> expectedCbYCr422 = packCbYCr422(hostY, hostCb422, hostCr422);

  int srcYStep = 0, srcCbCrStep = 0;
  Npp8u *d_srcY = allocatePlane(width, height, srcYStep);
  Npp8u *d_srcCbCr = allocatePlane(width, height / 2, srcCbCrStep);
  copyHostToDevice2D(d_srcY, srcYStep, hostY, width, height);
  copyHostToDevice2D(d_srcCbCr, srcCbCrStep, hostCbCr420, width, height / 2);

  int dstPackedStep = 0;
  Npp8u *d_dstYCbCr422 = allocatePlane(width * 2, height, dstPackedStep);
  Npp8u *d_dstCbYCr422 = allocatePlane(width * 2, height, dstPackedStep);
  const NppStreamContext ctx = streamContext();
  const NppiSize roi = {width, height};

  EXPECT_EQ(nppiYCbCr420ToYCbCr422_8u_P2C2R(d_srcY, srcYStep, d_srcCbCr, srcCbCrStep, d_dstYCbCr422, dstPackedStep, roi),
            NPP_NO_ERROR);
  EXPECT_EQ(copyDeviceToHost2D(d_dstYCbCr422, dstPackedStep, width * 2, height), expectedYCbCr422);
  EXPECT_EQ(nppiYCbCr420ToYCbCr422_8u_P2C2R_Ctx(d_srcY, srcYStep, d_srcCbCr, srcCbCrStep, d_dstYCbCr422, dstPackedStep,
                                                roi, ctx),
            NPP_NO_ERROR);

  EXPECT_EQ(nppiYCbCr420ToCbYCr422_8u_P2C2R(d_srcY, srcYStep, d_srcCbCr, srcCbCrStep, d_dstCbYCr422, dstPackedStep, roi),
            NPP_NO_ERROR);
  EXPECT_EQ(copyDeviceToHost2D(d_dstCbYCr422, dstPackedStep, width * 2, height), expectedCbYCr422);
  EXPECT_EQ(nppiYCbCr420ToCbYCr422_8u_P2C2R_Ctx(d_srcY, srcYStep, d_srcCbCr, srcCbCrStep, d_dstCbYCr422, dstPackedStep,
                                                roi, ctx),
            NPP_NO_ERROR);
}

TEST_F(YCbCr420LayoutTest, YCrCb420ToCbYCr422_And_YCbCr411Outputs) {
  const std::vector<Npp8u> hostY = makeYPlane();
  const std::vector<Npp8u> hostCb420 = makeCbPlane420();
  const std::vector<Npp8u> hostCr420 = makeCrPlane420();
  const std::vector<Npp8u> hostCb422 = upsample420To422Chroma(hostCb420);
  const std::vector<Npp8u> hostCr422 = upsample420To422Chroma(hostCr420);
  const std::vector<Npp8u> expectedCbYCr422 = packCbYCr422(hostY, hostCb422, hostCr422);
  const std::vector<Npp8u> expectedCb411 = downsample420To411Plane(hostCb420);
  const std::vector<Npp8u> expectedCr411 = downsample420To411Plane(hostCr420);
  const std::vector<Npp8u> expectedCbCr411 = interleaveChroma411(expectedCb411, expectedCr411);

  int srcYStep = 0, srcCrStep = 0, srcCbStep = 0;
  Npp8u *d_srcY = allocatePlane(width, height, srcYStep);
  Npp8u *d_srcCr = allocatePlane(width / 2, height / 2, srcCrStep);
  Npp8u *d_srcCb = allocatePlane(width / 2, height / 2, srcCbStep);
  copyHostToDevice2D(d_srcY, srcYStep, hostY, width, height);
  copyHostToDevice2D(d_srcCr, srcCrStep, hostCr420, width / 2, height / 2);
  copyHostToDevice2D(d_srcCb, srcCbStep, hostCb420, width / 2, height / 2);

  const Npp8u *pSrc[3] = {d_srcY, d_srcCr, d_srcCb};
  int srcSteps[3] = {srcYStep, srcCrStep, srcCbStep};
  const NppiSize roi = {width, height};
  const NppStreamContext ctx = streamContext();

  int dstPackedStep = 0;
  Npp8u *d_dstCbYCr422 = allocatePlane(width * 2, height, dstPackedStep);
  EXPECT_EQ(nppiYCrCb420ToCbYCr422_8u_P3C2R(pSrc, srcSteps, d_dstCbYCr422, dstPackedStep, roi), NPP_NO_ERROR);
  EXPECT_EQ(copyDeviceToHost2D(d_dstCbYCr422, dstPackedStep, width * 2, height), expectedCbYCr422);
  EXPECT_EQ(nppiYCrCb420ToCbYCr422_8u_P3C2R_Ctx(pSrc, srcSteps, d_dstCbYCr422, dstPackedStep, roi, ctx), NPP_NO_ERROR);

  int dstYStep = 0, dstCbCr411Step = 0;
  Npp8u *d_dstY = allocatePlane(width, height, dstYStep);
  Npp8u *d_dstCbCr411 = allocatePlane(width / 2, height, dstCbCr411Step);
  EXPECT_EQ(nppiYCrCb420ToYCbCr411_8u_P3P2R(pSrc, srcSteps, d_dstY, dstYStep, d_dstCbCr411, dstCbCr411Step, roi),
            NPP_NO_ERROR);
  EXPECT_EQ(copyDeviceToHost2D(d_dstY, dstYStep, width, height), hostY);
  EXPECT_EQ(copyDeviceToHost2D(d_dstCbCr411, dstCbCr411Step, width / 2, height), expectedCbCr411);
  EXPECT_EQ(nppiYCrCb420ToYCbCr411_8u_P3P2R_Ctx(pSrc, srcSteps, d_dstY, dstYStep, d_dstCbCr411, dstCbCr411Step, roi, ctx),
            NPP_NO_ERROR);
}

TEST_F(YCbCr420LayoutTest, YCbCr420ToYCbCr411_P3P2R_And_P2P3R_Outputs) {
  const std::vector<Npp8u> hostY = makeYPlane();
  const std::vector<Npp8u> hostCb420 = makeCbPlane420();
  const std::vector<Npp8u> hostCr420 = makeCrPlane420();
  const std::vector<Npp8u> hostCbCr420 = interleaveChroma420(hostCb420, hostCr420);
  const std::vector<Npp8u> expectedCb411 = downsample420To411Plane(hostCb420);
  const std::vector<Npp8u> expectedCr411 = downsample420To411Plane(hostCr420);
  const std::vector<Npp8u> expectedCbCr411 = interleaveChroma411(expectedCb411, expectedCr411);

  int srcYStep = 0, srcCbStep = 0, srcCrStep = 0, srcCbCrStep = 0;
  Npp8u *d_srcY = allocatePlane(width, height, srcYStep);
  Npp8u *d_srcCb = allocatePlane(width / 2, height / 2, srcCbStep);
  Npp8u *d_srcCr = allocatePlane(width / 2, height / 2, srcCrStep);
  Npp8u *d_srcCbCr = allocatePlane(width, height / 2, srcCbCrStep);
  copyHostToDevice2D(d_srcY, srcYStep, hostY, width, height);
  copyHostToDevice2D(d_srcCb, srcCbStep, hostCb420, width / 2, height / 2);
  copyHostToDevice2D(d_srcCr, srcCrStep, hostCr420, width / 2, height / 2);
  copyHostToDevice2D(d_srcCbCr, srcCbCrStep, hostCbCr420, width, height / 2);

  const Npp8u *pSrc[3] = {d_srcY, d_srcCb, d_srcCr};
  int srcSteps[3] = {srcYStep, srcCbStep, srcCrStep};
  const NppiSize roi = {width, height};
  const NppStreamContext ctx = streamContext();

  int dstYStep = 0, dstCbCr411Step = 0;
  Npp8u *d_dstY = allocatePlane(width, height, dstYStep);
  Npp8u *d_dstCbCr411 = allocatePlane(width / 2, height, dstCbCr411Step);
  EXPECT_EQ(nppiYCbCr420ToYCbCr411_8u_P3P2R(pSrc, srcSteps, d_dstY, dstYStep, d_dstCbCr411, dstCbCr411Step, roi),
            NPP_NO_ERROR);
  EXPECT_EQ(copyDeviceToHost2D(d_dstY, dstYStep, width, height), hostY);
  EXPECT_EQ(copyDeviceToHost2D(d_dstCbCr411, dstCbCr411Step, width / 2, height), expectedCbCr411);
  EXPECT_EQ(nppiYCbCr420ToYCbCr411_8u_P3P2R_Ctx(pSrc, srcSteps, d_dstY, dstYStep, d_dstCbCr411, dstCbCr411Step, roi, ctx),
            NPP_NO_ERROR);

  int dstY411Step = 0, dstCb411Step = 0, dstCr411Step = 0;
  Npp8u *d_dstY411 = allocatePlane(width, height, dstY411Step);
  Npp8u *d_dstCb411 = allocatePlane(width / 4, height, dstCb411Step);
  Npp8u *d_dstCr411 = allocatePlane(width / 4, height, dstCr411Step);
  Npp8u *pDst411[3] = {d_dstY411, d_dstCb411, d_dstCr411};
  int dst411Steps[3] = {dstY411Step, dstCb411Step, dstCr411Step};
  EXPECT_EQ(nppiYCbCr420ToYCbCr411_8u_P2P3R(d_srcY, srcYStep, d_srcCbCr, srcCbCrStep, pDst411, dst411Steps, roi),
            NPP_NO_ERROR);
  EXPECT_EQ(copyDeviceToHost2D(d_dstY411, dstY411Step, width, height), hostY);
  EXPECT_EQ(copyDeviceToHost2D(d_dstCb411, dstCb411Step, width / 4, height), expectedCb411);
  EXPECT_EQ(copyDeviceToHost2D(d_dstCr411, dstCr411Step, width / 4, height), expectedCr411);
  EXPECT_EQ(nppiYCbCr420ToYCbCr411_8u_P2P3R_Ctx(d_srcY, srcYStep, d_srcCbCr, srcCbCrStep, pDst411, dst411Steps, roi, ctx),
            NPP_NO_ERROR);
}

TEST_F(YCbCr420LayoutTest, Validation) {
#ifdef USE_NVIDIA_NPP_TESTS
  GTEST_SKIP() << "NVIDIA NPP validation-path behavior for these APIs is not stable in this environment";
#endif
  const std::vector<Npp8u> hostY = makeYPlane();
  const std::vector<Npp8u> hostCb = makeCbPlane420();
  const std::vector<Npp8u> hostCr = makeCrPlane420();
  const std::vector<Npp8u> hostCbCr = interleaveChroma420(hostCb, hostCr);

  int srcYStep = 0, srcCbStep = 0, srcCrStep = 0, srcCbCrStep = 0;
  Npp8u *d_srcY = allocatePlane(width, height, srcYStep);
  Npp8u *d_srcCb = allocatePlane(width / 2, height / 2, srcCbStep);
  Npp8u *d_srcCr = allocatePlane(width / 2, height / 2, srcCrStep);
  Npp8u *d_srcCbCr = allocatePlane(width, height / 2, srcCbCrStep);
  copyHostToDevice2D(d_srcY, srcYStep, hostY, width, height);
  copyHostToDevice2D(d_srcCb, srcCbStep, hostCb, width / 2, height / 2);
  copyHostToDevice2D(d_srcCr, srcCrStep, hostCr, width / 2, height / 2);
  copyHostToDevice2D(d_srcCbCr, srcCbCrStep, hostCbCr, width, height / 2);

  int dstYStep = 0, dstCbStep = 0, dstCrStep = 0, dstPackedStep = 0, dst411Step = 0;
  Npp8u *d_dstY = allocatePlane(width, height, dstYStep);
  Npp8u *d_dstCb = allocatePlane(width / 2, height, dstCbStep);
  Npp8u *d_dstCr = allocatePlane(width / 2, height, dstCrStep);
  Npp8u *d_dstPacked = allocatePlane(width * 2, height, dstPackedStep);
  Npp8u *d_dst411 = allocatePlane(width / 2, height, dst411Step);
  const Npp8u *pSrc[3] = {d_srcY, d_srcCb, d_srcCr};
  int srcSteps[3] = {srcYStep, srcCbStep, srcCrStep};
  Npp8u *pDst422[3] = {d_dstY, d_dstCb, d_dstCr};
  int dst422Steps[3] = {dstYStep, dstCbStep, dstCrStep};

  EXPECT_EQ(nppiYCbCr420_8u_P3P2R(nullptr, srcSteps, d_dstY, dstYStep, d_dst411, dst411Step, NppiSize{width, height}),
            NPP_NULL_POINTER_ERROR);
  EXPECT_EQ(nppiYCbCr420ToYCbCr422_8u_P3R(pSrc, srcSteps, pDst422, dst422Steps, NppiSize{width + 1, height}),
            NPP_WRONG_INTERSECTION_ROI_ERROR);
  EXPECT_EQ(nppiYCbCr420ToYCbCr422_8u_P2C2R(d_srcY, 0, d_srcCbCr, srcCbCrStep, d_dstPacked, dstPackedStep,
                                            NppiSize{width, height}),
            NPP_STEP_ERROR);
  EXPECT_EQ(nppiYCbCr420ToYCbCr411_8u_P3P2R(pSrc, srcSteps, d_dstY, dstYStep, d_dst411, dst411Step,
                                            NppiSize{width + 2, height}),
            NPP_WRONG_INTERSECTION_ROI_ERROR);
  Npp8u *pDst411[3] = {d_dstY, d_dstCb, d_dstCr};
  int dst411Steps[3] = {dstYStep, dstCbStep, dstCrStep};
  EXPECT_EQ(nppiYCbCr420ToYCbCr411_8u_P2P3R(d_srcY, srcYStep, d_srcCbCr, srcCbCrStep, pDst411, dst411Steps,
                                            NppiSize{width + 2, height}),
            NPP_WRONG_INTERSECTION_ROI_ERROR);
  EXPECT_EQ(nppiYCrCb420ToYCbCr411_8u_P3P2R(pSrc, srcSteps, d_dstY, dstYStep, d_dst411, dst411Step,
                                            NppiSize{width + 2, height}),
            NPP_WRONG_INTERSECTION_ROI_ERROR);
}
