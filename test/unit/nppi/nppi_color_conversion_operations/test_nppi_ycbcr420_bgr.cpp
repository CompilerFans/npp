#include "npp.h"
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <iostream>
#include <vector>

namespace {

class BgrToYCbCr420Test : public ::testing::Test {
protected:
  void SetUp() override {
    width = 4;
    height = 4;
    ASSERT_EQ(width % 2, 0);
    ASSERT_EQ(height % 2, 0);
  }

  void createPackedBGR(std::vector<Npp8u> &bgr) const {
    bgr.resize(width * height * 3);
    for (int y = 0; y < height; ++y) {
      for (int x = 0; x < width; ++x) {
        int idx = (y * width + x) * 3;
        Npp8u r = static_cast<Npp8u>(10 + x * 30 + y * 7);
        Npp8u g = static_cast<Npp8u>(20 + x * 12 + y * 11);
        Npp8u b = static_cast<Npp8u>(40 + x * 5 + y * 13);
        bgr[idx + 0] = b;
        bgr[idx + 1] = g;
        bgr[idx + 2] = r;
      }
    }
  }

  void createPackedBGRA(std::vector<Npp8u> &bgra) const {
    bgra.resize(width * height * 4);
    for (int y = 0; y < height; ++y) {
      for (int x = 0; x < width; ++x) {
        int idx = (y * width + x) * 4;
        Npp8u r = static_cast<Npp8u>(10 + x * 30 + y * 7);
        Npp8u g = static_cast<Npp8u>(20 + x * 12 + y * 11);
        Npp8u b = static_cast<Npp8u>(40 + x * 5 + y * 13);
        bgra[idx + 0] = b;
        bgra[idx + 1] = g;
        bgra[idx + 2] = r;
        bgra[idx + 3] = static_cast<Npp8u>(100 + x + y);
      }
    }
  }

  template <typename T>
  void dumpExpected(const char *label, const std::vector<T> &values, int count) const {
    std::cout << label << " = {";
    for (int i = 0; i < count; ++i) {
      if (i) {
        std::cout << ", ";
      }
      std::cout << static_cast<long long>(values[i]);
    }
    std::cout << "};\n";
  }

  int width = 0;
  int height = 0;
};

} // namespace

static void flattenPlanes(const std::vector<Npp8u> &hostY, int yStep, const std::vector<Npp8u> &hostCb, int cbStep,
                          const std::vector<Npp8u> &hostCr, int crStep, int width, int height,
                          std::vector<Npp8u> &flatY, std::vector<Npp8u> &flatCb, std::vector<Npp8u> &flatCr) {
  flatY.resize(width * height);
  flatCb.resize((width / 2) * (height / 2));
  flatCr.resize((width / 2) * (height / 2));

  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      flatY[y * width + x] = hostY[y * yStep + x];
    }
  }

  for (int y = 0; y < height / 2; ++y) {
    for (int x = 0; x < width / 2; ++x) {
      int idx = y * (width / 2) + x;
      flatCb[idx] = hostCb[y * cbStep + x];
      flatCr[idx] = hostCr[y * crStep + x];
    }
  }
}

TEST_F(BgrToYCbCr420Test, BGRToYCbCr420_8u_C3P3R_ExpectedValues) {
  std::vector<Npp8u> hostBGR;
  createPackedBGR(hostBGR);

  int srcStep = 0;
  Npp8u *d_src = nppiMalloc_8u_C3(width, height, &srcStep);
  ASSERT_NE(d_src, nullptr);

  int yStep = 0;
  int cbStep = 0;
  int crStep = 0;
  Npp8u *d_y = nppiMalloc_8u_C1(width, height, &yStep);
  Npp8u *d_cb = nppiMalloc_8u_C1(width / 2, height / 2, &cbStep);
  Npp8u *d_cr = nppiMalloc_8u_C1(width / 2, height / 2, &crStep);
  ASSERT_NE(d_y, nullptr);
  ASSERT_NE(d_cb, nullptr);
  ASSERT_NE(d_cr, nullptr);

  cudaMemcpy2D(d_src, srcStep, hostBGR.data(), width * 3, width * 3, height, cudaMemcpyHostToDevice);

  Npp8u *pDst[3] = {d_y, d_cb, d_cr};
  int dstSteps[3] = {yStep, cbStep, crStep};
  NppiSize roi = {width, height};
  NppStatus status = nppiBGRToYCbCr420_8u_C3P3R(d_src, srcStep, pDst, dstSteps, roi);
  EXPECT_EQ(status, NPP_NO_ERROR);

  std::vector<Npp8u> hostY(yStep * height);
  std::vector<Npp8u> hostCb(cbStep * (height / 2));
  std::vector<Npp8u> hostCr(crStep * (height / 2));

  cudaMemcpy(hostY.data(), d_y, hostY.size(), cudaMemcpyDeviceToHost);
  cudaMemcpy(hostCb.data(), d_cb, hostCb.size(), cudaMemcpyDeviceToHost);
  cudaMemcpy(hostCr.data(), d_cr, hostCr.size(), cudaMemcpyDeviceToHost);

  std::vector<Npp8u> flatY;
  std::vector<Npp8u> flatCb;
  std::vector<Npp8u> flatCr;
  flattenPlanes(hostY, yStep, hostCb, cbStep, hostCr, crStep, width, height, flatY, flatCb, flatCr);

  if (std::getenv("NPP_DUMP_EXPECTED")) {
    dumpExpected("kExpectedBGRToYCbCr420Y", flatY, width * height);
    dumpExpected("kExpectedBGRToYCbCr420Cb", flatCb, (width / 2) * (height / 2));
    dumpExpected("kExpectedBGRToYCbCr420Cr", flatCr, (width / 2) * (height / 2));
    GTEST_SKIP();
  }

  const Npp8u kExpectedBGRToYCbCr420Y[16] = {32, 46, 61, 75, 41, 55, 69, 83, 49, 64, 78, 92, 58, 72, 86, 101};
  const Npp8u kExpectedBGRToYCbCr420Cb[4] = {135, 124, 138, 127};
  const Npp8u kExpectedBGRToYCbCr420Cr[4] = {125, 141, 121, 138};

  for (int i = 0; i < width * height; ++i) {
    EXPECT_EQ(flatY[i], kExpectedBGRToYCbCr420Y[i]) << "Y mismatch at " << i;
  }
  for (int i = 0; i < (width / 2) * (height / 2); ++i) {
    EXPECT_EQ(flatCb[i], kExpectedBGRToYCbCr420Cb[i]) << "Cb mismatch at " << i;
    EXPECT_EQ(flatCr[i], kExpectedBGRToYCbCr420Cr[i]) << "Cr mismatch at " << i;
  }

  NppStreamContext ctx{};
  nppGetStreamContext(&ctx);
  ctx.hStream = 0;
  status = nppiBGRToYCbCr420_8u_C3P3R_Ctx(d_src, srcStep, pDst, dstSteps, roi, ctx);
  EXPECT_EQ(status, NPP_NO_ERROR);

  cudaMemcpy(hostY.data(), d_y, hostY.size(), cudaMemcpyDeviceToHost);
  cudaMemcpy(hostCb.data(), d_cb, hostCb.size(), cudaMemcpyDeviceToHost);
  cudaMemcpy(hostCr.data(), d_cr, hostCr.size(), cudaMemcpyDeviceToHost);

  flattenPlanes(hostY, yStep, hostCb, cbStep, hostCr, crStep, width, height, flatY, flatCb, flatCr);

  for (int i = 0; i < width * height; ++i) {
    EXPECT_EQ(flatY[i], kExpectedBGRToYCbCr420Y[i]) << "Ctx Y mismatch at " << i;
  }
  for (int i = 0; i < (width / 2) * (height / 2); ++i) {
    EXPECT_EQ(flatCb[i], kExpectedBGRToYCbCr420Cb[i]) << "Ctx Cb mismatch at " << i;
    EXPECT_EQ(flatCr[i], kExpectedBGRToYCbCr420Cr[i]) << "Ctx Cr mismatch at " << i;
  }

  nppiFree(d_src);
  nppiFree(d_y);
  nppiFree(d_cb);
  nppiFree(d_cr);
}

TEST_F(BgrToYCbCr420Test, BGRToYCbCr420_8u_AC4P3R_ExpectedValues) {
  std::vector<Npp8u> hostBGRA;
  createPackedBGRA(hostBGRA);

  int srcStep = 0;
  Npp8u *d_src = nppiMalloc_8u_C4(width, height, &srcStep);
  ASSERT_NE(d_src, nullptr);

  int yStep = 0;
  int cbStep = 0;
  int crStep = 0;
  Npp8u *d_y = nppiMalloc_8u_C1(width, height, &yStep);
  Npp8u *d_cb = nppiMalloc_8u_C1(width / 2, height / 2, &cbStep);
  Npp8u *d_cr = nppiMalloc_8u_C1(width / 2, height / 2, &crStep);
  ASSERT_NE(d_y, nullptr);
  ASSERT_NE(d_cb, nullptr);
  ASSERT_NE(d_cr, nullptr);

  cudaMemcpy2D(d_src, srcStep, hostBGRA.data(), width * 4, width * 4, height, cudaMemcpyHostToDevice);

  Npp8u *pDst[3] = {d_y, d_cb, d_cr};
  int dstSteps[3] = {yStep, cbStep, crStep};
  NppiSize roi = {width, height};
  NppStatus status = nppiBGRToYCbCr420_8u_AC4P3R(d_src, srcStep, pDst, dstSteps, roi);
  EXPECT_EQ(status, NPP_NO_ERROR);

  std::vector<Npp8u> hostY(yStep * height);
  std::vector<Npp8u> hostCb(cbStep * (height / 2));
  std::vector<Npp8u> hostCr(crStep * (height / 2));

  cudaMemcpy(hostY.data(), d_y, hostY.size(), cudaMemcpyDeviceToHost);
  cudaMemcpy(hostCb.data(), d_cb, hostCb.size(), cudaMemcpyDeviceToHost);
  cudaMemcpy(hostCr.data(), d_cr, hostCr.size(), cudaMemcpyDeviceToHost);

  std::vector<Npp8u> flatY;
  std::vector<Npp8u> flatCb;
  std::vector<Npp8u> flatCr;
  flattenPlanes(hostY, yStep, hostCb, cbStep, hostCr, crStep, width, height, flatY, flatCb, flatCr);

  if (std::getenv("NPP_DUMP_EXPECTED")) {
    dumpExpected("kExpectedBGRAToYCbCr420Y", flatY, width * height);
    dumpExpected("kExpectedBGRAToYCbCr420Cb", flatCb, (width / 2) * (height / 2));
    dumpExpected("kExpectedBGRAToYCbCr420Cr", flatCr, (width / 2) * (height / 2));
    GTEST_SKIP();
  }

  const Npp8u kExpectedBGRAToYCbCr420Y[16] = {32, 46, 61, 75, 41, 55, 69, 83, 49, 64, 78, 92, 58, 72, 86, 101};
  const Npp8u kExpectedBGRAToYCbCr420Cb[4] = {135, 124, 138, 127};
  const Npp8u kExpectedBGRAToYCbCr420Cr[4] = {125, 141, 121, 138};

  for (int i = 0; i < width * height; ++i) {
    EXPECT_EQ(flatY[i], kExpectedBGRAToYCbCr420Y[i]) << "Y mismatch at " << i;
  }
  for (int i = 0; i < (width / 2) * (height / 2); ++i) {
    EXPECT_EQ(flatCb[i], kExpectedBGRAToYCbCr420Cb[i]) << "Cb mismatch at " << i;
    EXPECT_EQ(flatCr[i], kExpectedBGRAToYCbCr420Cr[i]) << "Cr mismatch at " << i;
  }

  NppStreamContext ctx{};
  nppGetStreamContext(&ctx);
  ctx.hStream = 0;
  status = nppiBGRToYCbCr420_8u_AC4P3R_Ctx(d_src, srcStep, pDst, dstSteps, roi, ctx);
  EXPECT_EQ(status, NPP_NO_ERROR);

  cudaMemcpy(hostY.data(), d_y, hostY.size(), cudaMemcpyDeviceToHost);
  cudaMemcpy(hostCb.data(), d_cb, hostCb.size(), cudaMemcpyDeviceToHost);
  cudaMemcpy(hostCr.data(), d_cr, hostCr.size(), cudaMemcpyDeviceToHost);

  flattenPlanes(hostY, yStep, hostCb, cbStep, hostCr, crStep, width, height, flatY, flatCb, flatCr);

  for (int i = 0; i < width * height; ++i) {
    EXPECT_EQ(flatY[i], kExpectedBGRAToYCbCr420Y[i]) << "Ctx Y mismatch at " << i;
  }
  for (int i = 0; i < (width / 2) * (height / 2); ++i) {
    EXPECT_EQ(flatCb[i], kExpectedBGRAToYCbCr420Cb[i]) << "Ctx Cb mismatch at " << i;
    EXPECT_EQ(flatCr[i], kExpectedBGRAToYCbCr420Cr[i]) << "Ctx Cr mismatch at " << i;
  }

  nppiFree(d_src);
  nppiFree(d_y);
  nppiFree(d_cb);
  nppiFree(d_cr);
}

// ==================== YCbCr420ToBGR Tests ====================

TEST_F(BgrToYCbCr420Test, YCbCr420ToBGR_RoundTrip) {
  std::vector<Npp8u> hostBGR;
  createPackedBGR(hostBGR);

  int srcStep = 0;
  Npp8u *d_bgrSrc = nppiMalloc_8u_C3(width, height, &srcStep);
  ASSERT_NE(d_bgrSrc, nullptr);

  int yStep = 0, cbStep = 0, crStep = 0;
  Npp8u *d_y = nppiMalloc_8u_C1(width, height, &yStep);
  Npp8u *d_cb = nppiMalloc_8u_C1(width / 2, height / 2, &cbStep);
  Npp8u *d_cr = nppiMalloc_8u_C1(width / 2, height / 2, &crStep);
  ASSERT_NE(d_y, nullptr);
  ASSERT_NE(d_cb, nullptr);
  ASSERT_NE(d_cr, nullptr);

  int dstStep = 0;
  Npp8u *d_bgrDst = nppiMalloc_8u_C3(width, height, &dstStep);
  ASSERT_NE(d_bgrDst, nullptr);

  cudaMemcpy2D(d_bgrSrc, srcStep, hostBGR.data(), width * 3, width * 3, height, cudaMemcpyHostToDevice);

  Npp8u *pDst[3] = {d_y, d_cb, d_cr};
  int dstSteps[3] = {yStep, cbStep, crStep};
  NppiSize roi = {width, height};
  NppStatus status = nppiBGRToYCbCr420_8u_C3P3R(d_bgrSrc, srcStep, pDst, dstSteps, roi);
  ASSERT_EQ(status, NPP_NO_ERROR);

  const Npp8u *pSrc[3] = {d_y, d_cb, d_cr};
  int srcSteps[3] = {yStep, cbStep, crStep};
  status = nppiYCbCr420ToBGR_8u_P3C3R(pSrc, srcSteps, d_bgrDst, dstStep, roi);
  ASSERT_EQ(status, NPP_NO_ERROR);

  std::vector<Npp8u> resultBGR(width * height * 3);
  cudaMemcpy2D(resultBGR.data(), width * 3, d_bgrDst, dstStep, width * 3, height, cudaMemcpyDeviceToHost);

  for (int i = 0; i < width * height * 3; ++i) {
    int diff = std::abs(static_cast<int>(resultBGR[i]) - static_cast<int>(hostBGR[i]));
    EXPECT_LE(diff, 20) << "Roundtrip mismatch at byte " << i;
  }

  nppiFree(d_bgrSrc);
  nppiFree(d_y);
  nppiFree(d_cb);
  nppiFree(d_cr);
  nppiFree(d_bgrDst);
}

TEST_F(BgrToYCbCr420Test, YCbCr420ToBGR_RoundTrip_Ctx) {
  std::vector<Npp8u> hostBGR;
  createPackedBGR(hostBGR);

  int srcStep = 0;
  Npp8u *d_bgrSrc = nppiMalloc_8u_C3(width, height, &srcStep);
  ASSERT_NE(d_bgrSrc, nullptr);

  int yStep = 0, cbStep = 0, crStep = 0;
  Npp8u *d_y = nppiMalloc_8u_C1(width, height, &yStep);
  Npp8u *d_cb = nppiMalloc_8u_C1(width / 2, height / 2, &cbStep);
  Npp8u *d_cr = nppiMalloc_8u_C1(width / 2, height / 2, &crStep);
  ASSERT_NE(d_y, nullptr);
  ASSERT_NE(d_cb, nullptr);
  ASSERT_NE(d_cr, nullptr);

  int dstStep = 0;
  Npp8u *d_bgrDst = nppiMalloc_8u_C3(width, height, &dstStep);
  ASSERT_NE(d_bgrDst, nullptr);

  cudaMemcpy2D(d_bgrSrc, srcStep, hostBGR.data(), width * 3, width * 3, height, cudaMemcpyHostToDevice);

  NppStreamContext ctx{};
  nppGetStreamContext(&ctx);
  ctx.hStream = 0;

  Npp8u *pDst[3] = {d_y, d_cb, d_cr};
  int dstSteps[3] = {yStep, cbStep, crStep};
  NppiSize roi = {width, height};
  NppStatus status = nppiBGRToYCbCr420_8u_C3P3R_Ctx(d_bgrSrc, srcStep, pDst, dstSteps, roi, ctx);
  ASSERT_EQ(status, NPP_NO_ERROR);

  const Npp8u *pSrc[3] = {d_y, d_cb, d_cr};
  int srcSteps[3] = {yStep, cbStep, crStep};
  status = nppiYCbCr420ToBGR_8u_P3C3R_Ctx(pSrc, srcSteps, d_bgrDst, dstStep, roi, ctx);
  ASSERT_EQ(status, NPP_NO_ERROR);

  std::vector<Npp8u> resultBGR(width * height * 3);
  cudaMemcpy2D(resultBGR.data(), width * 3, d_bgrDst, dstStep, width * 3, height, cudaMemcpyDeviceToHost);

  for (int i = 0; i < width * height * 3; ++i) {
    int diff = std::abs(static_cast<int>(resultBGR[i]) - static_cast<int>(hostBGR[i]));
    EXPECT_LE(diff, 20) << "Ctx roundtrip mismatch at byte " << i;
  }

  nppiFree(d_bgrSrc);
  nppiFree(d_y);
  nppiFree(d_cb);
  nppiFree(d_cr);
  nppiFree(d_bgrDst);
}

TEST_F(BgrToYCbCr420Test, YCbCr420ToBGR_ChannelOrder) {
  std::vector<Npp8u> hostBGR(width * height * 3, 0);
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      int idx = (y * width + x) * 3;
      hostBGR[idx + 0] = 0;   // B
      hostBGR[idx + 1] = 0;   // G
      hostBGR[idx + 2] = 255; // R
    }
  }

  int srcStep = 0;
  Npp8u *d_bgrSrc = nppiMalloc_8u_C3(width, height, &srcStep);
  ASSERT_NE(d_bgrSrc, nullptr);

  int yStep = 0, cbStep = 0, crStep = 0;
  Npp8u *d_y = nppiMalloc_8u_C1(width, height, &yStep);
  Npp8u *d_cb = nppiMalloc_8u_C1(width / 2, height / 2, &cbStep);
  Npp8u *d_cr = nppiMalloc_8u_C1(width / 2, height / 2, &crStep);
  ASSERT_NE(d_y, nullptr);
  ASSERT_NE(d_cb, nullptr);
  ASSERT_NE(d_cr, nullptr);

  int dstStep = 0;
  Npp8u *d_bgrDst = nppiMalloc_8u_C3(width, height, &dstStep);
  ASSERT_NE(d_bgrDst, nullptr);

  cudaMemcpy2D(d_bgrSrc, srcStep, hostBGR.data(), width * 3, width * 3, height, cudaMemcpyHostToDevice);

  Npp8u *pDst[3] = {d_y, d_cb, d_cr};
  int dstSteps[3] = {yStep, cbStep, crStep};
  NppiSize roi = {width, height};
  NppStatus status = nppiBGRToYCbCr420_8u_C3P3R(d_bgrSrc, srcStep, pDst, dstSteps, roi);
  ASSERT_EQ(status, NPP_NO_ERROR);

  const Npp8u *pSrc[3] = {d_y, d_cb, d_cr};
  int srcSteps[3] = {yStep, cbStep, crStep};
  status = nppiYCbCr420ToBGR_8u_P3C3R(pSrc, srcSteps, d_bgrDst, dstStep, roi);
  ASSERT_EQ(status, NPP_NO_ERROR);

  std::vector<Npp8u> resultBGR(width * height * 3);
  cudaMemcpy2D(resultBGR.data(), width * 3, d_bgrDst, dstStep, width * 3, height, cudaMemcpyDeviceToHost);

  int cx = width / 2;
  int cy = height / 2;
  int idx = (cy * width + cx) * 3;
  EXPECT_LT(resultBGR[idx + 0], 30) << "B channel should be near 0 for red input";
  EXPECT_LT(resultBGR[idx + 1], 30) << "G channel should be near 0 for red input";
  EXPECT_GT(resultBGR[idx + 2], 220) << "R channel should be near 255 for red input";

  nppiFree(d_bgrSrc);
  nppiFree(d_y);
  nppiFree(d_cb);
  nppiFree(d_cr);
  nppiFree(d_bgrDst);
}

TEST_F(BgrToYCbCr420Test, YCbCr420ToBGR_DirectPixelValue_DarkImage) {
  // Test with a dark image (all pixels near 0)
  width = 8;
  height = 8;
  std::vector<Npp8u> hostBGR(width * height * 3);
  for (int i = 0; i < width * height * 3; i++) {
    hostBGR[i] = static_cast<Npp8u>(10 + (i % 5));
  }

  int srcStep = 0;
  Npp8u *d_bgrSrc = nppiMalloc_8u_C3(width, height, &srcStep);
  ASSERT_NE(d_bgrSrc, nullptr);

  int yStep = 0, cbStep = 0, crStep = 0;
  Npp8u *d_y = nppiMalloc_8u_C1(width, height, &yStep);
  Npp8u *d_cb = nppiMalloc_8u_C1(width / 2, height / 2, &cbStep);
  Npp8u *d_cr = nppiMalloc_8u_C1(width / 2, height / 2, &crStep);
  ASSERT_NE(d_y, nullptr);
  ASSERT_NE(d_cb, nullptr);
  ASSERT_NE(d_cr, nullptr);

  int dstStep = 0;
  Npp8u *d_bgrDst = nppiMalloc_8u_C3(width, height, &dstStep);
  ASSERT_NE(d_bgrDst, nullptr);

  cudaMemcpy2D(d_bgrSrc, srcStep, hostBGR.data(), width * 3, width * 3, height, cudaMemcpyHostToDevice);

  Npp8u *pDst[3] = {d_y, d_cb, d_cr};
  int dstSteps[3] = {yStep, cbStep, crStep};
  NppiSize roi = {width, height};
  NppStatus status = nppiBGRToYCbCr420_8u_C3P3R(d_bgrSrc, srcStep, pDst, dstSteps, roi);
  ASSERT_EQ(status, NPP_NO_ERROR);

  // Verify Y values are in valid range for dark image
  std::vector<Npp8u> hostY(yStep * height);
  cudaMemcpy(hostY.data(), d_y, hostY.size(), cudaMemcpyDeviceToHost);
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      Npp8u yVal = hostY[y * (yStep) + x];
      EXPECT_LT(yVal, 50) << "Y value too high for dark image at (" << x << "," << y << ")";
    }
  }

  // Convert back to BGR
  const Npp8u *pSrc[3] = {d_y, d_cb, d_cr};
  int srcSteps[3] = {yStep, cbStep, crStep};
  status = nppiYCbCr420ToBGR_8u_P3C3R(pSrc, srcSteps, d_bgrDst, dstStep, roi);
  ASSERT_EQ(status, NPP_NO_ERROR);

  std::vector<Npp8u> resultBGR(width * height * 3);
  cudaMemcpy2D(resultBGR.data(), width * 3, d_bgrDst, dstStep, width * 3, height, cudaMemcpyDeviceToHost);

  // Dark image output should also be dark
  for (int i = 0; i < width * height * 3; i++) {
    EXPECT_LT(resultBGR[i], 60) << "Dark image output too bright at byte " << i;
  }

  nppiFree(d_bgrSrc);
  nppiFree(d_y);
  nppiFree(d_cb);
  nppiFree(d_cr);
  nppiFree(d_bgrDst);
}

TEST_F(BgrToYCbCr420Test, YCbCr420ToBGR_DirectPixelValue_BrightImage) {
  // Test with a bright image (all pixels near 255)
  width = 8;
  height = 8;
  std::vector<Npp8u> hostBGR(width * height * 3);
  for (int i = 0; i < width * height * 3; i++) {
    hostBGR[i] = static_cast<Npp8u>(245 + (i % 5));
  }

  int srcStep = 0;
  Npp8u *d_bgrSrc = nppiMalloc_8u_C3(width, height, &srcStep);
  ASSERT_NE(d_bgrSrc, nullptr);

  int yStep = 0, cbStep = 0, crStep = 0;
  Npp8u *d_y = nppiMalloc_8u_C1(width, height, &yStep);
  Npp8u *d_cb = nppiMalloc_8u_C1(width / 2, height / 2, &cbStep);
  Npp8u *d_cr = nppiMalloc_8u_C1(width / 2, height / 2, &crStep);
  ASSERT_NE(d_y, nullptr);
  ASSERT_NE(d_cb, nullptr);
  ASSERT_NE(d_cr, nullptr);

  int dstStep = 0;
  Npp8u *d_bgrDst = nppiMalloc_8u_C3(width, height, &dstStep);
  ASSERT_NE(d_bgrDst, nullptr);

  cudaMemcpy2D(d_bgrSrc, srcStep, hostBGR.data(), width * 3, width * 3, height, cudaMemcpyHostToDevice);

  Npp8u *pDst[3] = {d_y, d_cb, d_cr};
  int dstSteps[3] = {yStep, cbStep, crStep};
  NppiSize roi = {width, height};
  NppStatus status = nppiBGRToYCbCr420_8u_C3P3R(d_bgrSrc, srcStep, pDst, dstSteps, roi);
  ASSERT_EQ(status, NPP_NO_ERROR);

  const Npp8u *pSrc[3] = {d_y, d_cb, d_cr};
  int srcSteps[3] = {yStep, cbStep, crStep};
  status = nppiYCbCr420ToBGR_8u_P3C3R(pSrc, srcSteps, d_bgrDst, dstStep, roi);
  ASSERT_EQ(status, NPP_NO_ERROR);

  std::vector<Npp8u> resultBGR(width * height * 3);
  cudaMemcpy2D(resultBGR.data(), width * 3, d_bgrDst, dstStep, width * 3, height, cudaMemcpyDeviceToHost);

  // Bright image output should also be bright
  for (int i = 0; i < width * height * 3; i++) {
    EXPECT_GT(resultBGR[i], 200) << "Bright image output too dark at byte " << i;
  }

  nppiFree(d_bgrSrc);
  nppiFree(d_y);
  nppiFree(d_cb);
  nppiFree(d_cr);
  nppiFree(d_bgrDst);
}

TEST_F(BgrToYCbCr420Test, YCbCr420ToBGR_DirectPixelValue_NearGray) {
  // Test with near-gray image (R=G=B)
  width = 8;
  height = 8;
  std::vector<Npp8u> hostBGR(width * height * 3);
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      int idx = (y * width + x) * 3;
      Npp8u gray = static_cast<Npp8u>(128 + (x + y) % 10);
      hostBGR[idx + 0] = gray; // B
      hostBGR[idx + 1] = gray; // G
      hostBGR[idx + 2] = gray; // R
    }
  }

  int srcStep = 0;
  Npp8u *d_bgrSrc = nppiMalloc_8u_C3(width, height, &srcStep);
  ASSERT_NE(d_bgrSrc, nullptr);

  int yStep = 0, cbStep = 0, crStep = 0;
  Npp8u *d_y = nppiMalloc_8u_C1(width, height, &yStep);
  Npp8u *d_cb = nppiMalloc_8u_C1(width / 2, height / 2, &cbStep);
  Npp8u *d_cr = nppiMalloc_8u_C1(width / 2, height / 2, &crStep);
  ASSERT_NE(d_y, nullptr);
  ASSERT_NE(d_cb, nullptr);
  ASSERT_NE(d_cr, nullptr);

  int dstStep = 0;
  Npp8u *d_bgrDst = nppiMalloc_8u_C3(width, height, &dstStep);
  ASSERT_NE(d_bgrDst, nullptr);

  cudaMemcpy2D(d_bgrSrc, srcStep, hostBGR.data(), width * 3, width * 3, height, cudaMemcpyHostToDevice);

  Npp8u *pDst[3] = {d_y, d_cb, d_cr};
  int dstSteps[3] = {yStep, cbStep, crStep};
  NppiSize roi = {width, height};
  NppStatus status = nppiBGRToYCbCr420_8u_C3P3R(d_bgrSrc, srcStep, pDst, dstSteps, roi);
  ASSERT_EQ(status, NPP_NO_ERROR);

  // For gray image, Cb and Cr should be near 128 (neutral)
  std::vector<Npp8u> hostCb(cbStep * (height / 2));
  std::vector<Npp8u> hostCr(crStep * (height / 2));
  cudaMemcpy(hostCb.data(), d_cb, hostCb.size(), cudaMemcpyDeviceToHost);
  cudaMemcpy(hostCr.data(), d_cr, hostCr.size(), cudaMemcpyDeviceToHost);

  for (int y = 0; y < height / 2; y++) {
    for (int x = 0; x < width / 2; x++) {
      Npp8u cb = hostCb[y * cbStep + x];
      Npp8u cr = hostCr[y * crStep + x];
      EXPECT_NEAR(cb, 128, 10) << "Cb should be near 128 for gray at (" << x << "," << y << ")";
      EXPECT_NEAR(cr, 128, 10) << "Cr should be near 128 for gray at (" << x << "," << y << ")";
    }
  }

  const Npp8u *pSrc[3] = {d_y, d_cb, d_cr};
  int srcSteps[3] = {yStep, cbStep, crStep};
  status = nppiYCbCr420ToBGR_8u_P3C3R(pSrc, srcSteps, d_bgrDst, dstStep, roi);
  ASSERT_EQ(status, NPP_NO_ERROR);

  std::vector<Npp8u> resultBGR(width * height * 3);
  cudaMemcpy2D(resultBGR.data(), width * 3, d_bgrDst, dstStep, width * 3, height, cudaMemcpyDeviceToHost);

  // For gray input, output BGR channels should be nearly equal
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      int idx = (y * width + x) * 3;
      int b = resultBGR[idx + 0];
      int g = resultBGR[idx + 1];
      int r = resultBGR[idx + 2];
      EXPECT_NEAR(b, g, 15) << "B!=G for gray at (" << x << "," << y << ")";
      EXPECT_NEAR(g, r, 15) << "G!=R for gray at (" << x << "," << y << ")";
    }
  }

  nppiFree(d_bgrSrc);
  nppiFree(d_y);
  nppiFree(d_cb);
  nppiFree(d_cr);
  nppiFree(d_bgrDst);
}

TEST_F(BgrToYCbCr420Test, YCbCr420ToBGR_MinimumSize_2x2) {
  // Test with minimum valid size (2x2)
  width = 2;
  height = 2;
  std::vector<Npp8u> hostBGR = {0, 0, 255, 0, 255, 0, 255, 0, 0, 128, 128, 128};

  int srcStep = 0;
  Npp8u *d_bgrSrc = nppiMalloc_8u_C3(width, height, &srcStep);
  ASSERT_NE(d_bgrSrc, nullptr);

  int yStep = 0, cbStep = 0, crStep = 0;
  Npp8u *d_y = nppiMalloc_8u_C1(width, height, &yStep);
  Npp8u *d_cb = nppiMalloc_8u_C1(width / 2, height / 2, &cbStep);
  Npp8u *d_cr = nppiMalloc_8u_C1(width / 2, height / 2, &crStep);
  ASSERT_NE(d_y, nullptr);
  ASSERT_NE(d_cb, nullptr);
  ASSERT_NE(d_cr, nullptr);

  int dstStep = 0;
  Npp8u *d_bgrDst = nppiMalloc_8u_C3(width, height, &dstStep);
  ASSERT_NE(d_bgrDst, nullptr);

  cudaMemcpy2D(d_bgrSrc, srcStep, hostBGR.data(), width * 3, width * 3, height, cudaMemcpyHostToDevice);

  Npp8u *pDst[3] = {d_y, d_cb, d_cr};
  int dstSteps[3] = {yStep, cbStep, crStep};
  NppiSize roi = {width, height};
  NppStatus status = nppiBGRToYCbCr420_8u_C3P3R(d_bgrSrc, srcStep, pDst, dstSteps, roi);
  ASSERT_EQ(status, NPP_NO_ERROR);

  const Npp8u *pSrc[3] = {d_y, d_cb, d_cr};
  int srcSteps[3] = {yStep, cbStep, crStep};
  status = nppiYCbCr420ToBGR_8u_P3C3R(pSrc, srcSteps, d_bgrDst, dstStep, roi);
  ASSERT_EQ(status, NPP_NO_ERROR);

  std::vector<Npp8u> resultBGR(width * height * 3);
  cudaMemcpy2D(resultBGR.data(), width * 3, d_bgrDst, dstStep, width * 3, height, cudaMemcpyDeviceToHost);

  // Should not crash and produce valid output
  for (int i = 0; i < width * height * 3; i++) {
    EXPECT_GE(resultBGR[i], 0);
    EXPECT_LE(resultBGR[i], 255);
  }

  nppiFree(d_bgrSrc);
  nppiFree(d_y);
  nppiFree(d_cb);
  nppiFree(d_cr);
  nppiFree(d_bgrDst);
}
