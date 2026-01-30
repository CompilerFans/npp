#include "npp.h"
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <iostream>
#include <vector>

namespace {

class YCbCr420Test : public ::testing::Test {
protected:
  void SetUp() override {
    width = 4;
    height = 4;
    ASSERT_EQ(width % 2, 0);
    ASSERT_EQ(height % 2, 0);
  }

  void createPackedRGB(std::vector<Npp8u> &rgb) const {
    rgb.resize(width * height * 3);
    for (int y = 0; y < height; ++y) {
      for (int x = 0; x < width; ++x) {
        int idx = (y * width + x) * 3;
        rgb[idx + 0] = static_cast<Npp8u>(10 + x * 30 + y * 7);
        rgb[idx + 1] = static_cast<Npp8u>(20 + x * 12 + y * 11);
        rgb[idx + 2] = static_cast<Npp8u>(40 + x * 5 + y * 13);
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

TEST_F(YCbCr420Test, RGBToYCbCr420_And_Back_ExpectedValues) {
  std::vector<Npp8u> hostRGB;
  createPackedRGB(hostRGB);

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

  cudaMemcpy2D(d_src, srcStep, hostRGB.data(), width * 3, width * 3, height, cudaMemcpyHostToDevice);

  Npp8u *pDst[3] = {d_y, d_cb, d_cr};
  int dstSteps[3] = {yStep, cbStep, crStep};
  NppiSize roi = {width, height};

  NppStatus status = nppiRGBToYCbCr420_8u_C3P3R(d_src, srcStep, pDst, dstSteps, roi);
  EXPECT_EQ(status, NPP_NO_ERROR);

  std::vector<Npp8u> hostY(yStep * height);
  std::vector<Npp8u> hostCb(cbStep * (height / 2));
  std::vector<Npp8u> hostCr(crStep * (height / 2));

  cudaMemcpy(hostY.data(), d_y, hostY.size(), cudaMemcpyDeviceToHost);
  cudaMemcpy(hostCb.data(), d_cb, hostCb.size(), cudaMemcpyDeviceToHost);
  cudaMemcpy(hostCr.data(), d_cr, hostCr.size(), cudaMemcpyDeviceToHost);

  int dstRgbStep = 0;
  Npp8u *d_dst = nppiMalloc_8u_C3(width, height, &dstRgbStep);
  ASSERT_NE(d_dst, nullptr);

  const Npp8u *pSrcPlanes[3] = {d_y, d_cb, d_cr};
  int srcSteps[3] = {yStep, cbStep, crStep};
  status = nppiYCbCr420ToRGB_8u_P3C3R(pSrcPlanes, srcSteps, d_dst, dstRgbStep, roi);
  EXPECT_EQ(status, NPP_NO_ERROR);

  std::vector<Npp8u> hostRGBOut(dstRgbStep * height);
  cudaMemcpy(hostRGBOut.data(), d_dst, hostRGBOut.size(), cudaMemcpyDeviceToHost);

  std::vector<Npp8u> flatY(width * height);
  std::vector<Npp8u> flatCb((width / 2) * (height / 2));
  std::vector<Npp8u> flatCr((width / 2) * (height / 2));
  std::vector<Npp8u> flatRGB(width * height * 3);

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

  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      int idx = (y * width + x) * 3;
      int srcIdx = y * dstRgbStep + x * 3;
      flatRGB[idx + 0] = hostRGBOut[srcIdx + 0];
      flatRGB[idx + 1] = hostRGBOut[srcIdx + 1];
      flatRGB[idx + 2] = hostRGBOut[srcIdx + 2];
    }
  }

  if (std::getenv("NPP_DUMP_EXPECTED")) {
    dumpExpected("kExpectedYCbCr420Y", flatY, width * height);
    dumpExpected("kExpectedYCbCr420Cb", flatCb, (width / 2) * (height / 2));
    dumpExpected("kExpectedYCbCr420Cr", flatCr, (width / 2) * (height / 2));
    dumpExpected("kExpectedYCbCr420ToRGB", flatRGB, width * height * 3);
    GTEST_SKIP();
  }

  const Npp8u kExpectedYCbCr420Y[16] = {32, 46, 61, 75, 41, 55, 69, 83, 49, 64, 78, 92, 58, 72, 86, 101};
  const Npp8u kExpectedYCbCr420Cb[4] = {135, 124, 138, 127};
  const Npp8u kExpectedYCbCr420Cr[4] = {125, 141, 121, 138};
  const Npp8u kExpectedYCbCr420ToRGB[48] = {13,  18,  32,  30,  34,  49,  73,  43,  44,  89,  59,  60,
                                           24,  28,  43,  40,  45,  59,  82,  52,  53,  98,  68,  69,
                                           27,  40,  58,  44,  57,  76,  88,  64,  70,  104, 80,  86,
                                           37,  50,  69,  54,  66,  85,  97,  73,  79,  114, 91,  96};

  ASSERT_EQ(16, static_cast<int>(sizeof(kExpectedYCbCr420Y) / sizeof(kExpectedYCbCr420Y[0])));
  ASSERT_EQ(4, static_cast<int>(sizeof(kExpectedYCbCr420Cb) / sizeof(kExpectedYCbCr420Cb[0])));
  ASSERT_EQ(4, static_cast<int>(sizeof(kExpectedYCbCr420Cr) / sizeof(kExpectedYCbCr420Cr[0])));
  ASSERT_EQ(48, static_cast<int>(sizeof(kExpectedYCbCr420ToRGB) / sizeof(kExpectedYCbCr420ToRGB[0])));

  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      int expIdx = y * width + x;
      EXPECT_EQ(flatY[expIdx], kExpectedYCbCr420Y[expIdx]) << "Y mismatch at " << expIdx;
    }
  }

  for (int y = 0; y < height / 2; ++y) {
    for (int x = 0; x < width / 2; ++x) {
      int expIdx = y * (width / 2) + x;
      EXPECT_EQ(flatCb[expIdx], kExpectedYCbCr420Cb[expIdx]) << "Cb mismatch at " << expIdx;
      EXPECT_EQ(flatCr[expIdx], kExpectedYCbCr420Cr[expIdx]) << "Cr mismatch at " << expIdx;
    }
  }

  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      int expIdx = (y * width + x) * 3;
      EXPECT_EQ(flatRGB[expIdx + 0], kExpectedYCbCr420ToRGB[expIdx + 0]) << "R mismatch at " << expIdx;
      EXPECT_EQ(flatRGB[expIdx + 1], kExpectedYCbCr420ToRGB[expIdx + 1]) << "G mismatch at " << expIdx;
      EXPECT_EQ(flatRGB[expIdx + 2], kExpectedYCbCr420ToRGB[expIdx + 2]) << "B mismatch at " << expIdx;
    }
  }

  nppiFree(d_src);
  nppiFree(d_y);
  nppiFree(d_cb);
  nppiFree(d_cr);
  nppiFree(d_dst);
}
