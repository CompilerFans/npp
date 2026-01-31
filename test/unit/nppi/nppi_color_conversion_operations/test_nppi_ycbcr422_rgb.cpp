#include "npp.h"
#include <cmath>
#include <cstring>
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <iostream>
#include <random>
#include <vector>

namespace {

inline Npp8u clamp_u8_double(double v) {
  if (v < 0.0) {
    v = 0.0;
  } else if (v > 255.0) {
    v = 255.0;
  }
  // NVIDIA NPP uses truncation (floor), not rounding
  return static_cast<Npp8u>(v);
}

inline void rgb_to_ycbcr_pixel(Npp8u r, Npp8u g, Npp8u b, Npp8u &y, Npp8u &cb, Npp8u &cr) {
  double R = r;
  double G = g;
  double B = b;
  double Y = 0.257 * R + 0.504 * G + 0.098 * B + 16.0;
  double Cb = -0.148 * R - 0.291 * G + 0.439 * B + 128.0;
  double Cr = 0.439 * R - 0.368 * G - 0.071 * B + 128.0;

  y = clamp_u8_double(Y);
  cb = clamp_u8_double(Cb);
  cr = clamp_u8_double(Cr);
}

inline void ycbcr_to_rgb_pixel(Npp8u y, Npp8u cb, Npp8u cr, Npp8u &r, Npp8u &g, Npp8u &b) {
  double Y = 1.164 * (static_cast<double>(y) - 16.0);
  double Cr = static_cast<double>(cr) - 128.0;
  double Cb = static_cast<double>(cb) - 128.0;

  double R = Y + 1.596 * Cr;
  double G = Y - 0.813 * Cr - 0.392 * Cb;
  double B = Y + 2.017 * Cb;

  r = clamp_u8_double(R);
  g = clamp_u8_double(G);
  b = clamp_u8_double(B);
}

class YCbCr422Test : public ::testing::Test {
protected:
  void SetUp() override {
    width = 4;
    height = 4;
    ASSERT_EQ(width % 2, 0);
  }

  void createPackedRGB(std::vector<Npp8u> &rgb) const {
    rgb.resize(width * height * 3);
    for (int y = 0; y < height; ++y) {
      for (int x = 0; x < width; ++x) {
        int idx = (y * width + x) * 3;
        rgb[idx + 0] = static_cast<Npp8u>(12 + x * 23 + y * 5);
        rgb[idx + 1] = static_cast<Npp8u>(34 + x * 11 + y * 9);
        rgb[idx + 2] = static_cast<Npp8u>(56 + x * 7 + y * 13);
      }
    }
  }

  void computeYCbCr422Planar(const std::vector<Npp8u> &rgb, std::vector<Npp8u> &y, std::vector<Npp8u> &cb,
                             std::vector<Npp8u> &cr) const {
    y.assign(width * height, 0);
    cb.assign((width / 2) * height, 0);
    cr.assign((width / 2) * height, 0);

    for (int row = 0; row < height; ++row) {
      for (int x = 0; x < width; x += 2) {
        int idx0 = (row * width + x) * 3;
        int idx1 = (row * width + x + 1) * 3;
        Npp8u y0, cb0, cr0;
        Npp8u y1, cb1, cr1;
        rgb_to_ycbcr_pixel(rgb[idx0 + 0], rgb[idx0 + 1], rgb[idx0 + 2], y0, cb0, cr0);
        rgb_to_ycbcr_pixel(rgb[idx1 + 0], rgb[idx1 + 1], rgb[idx1 + 2], y1, cb1, cr1);

        y[row * width + x] = y0;
        y[row * width + x + 1] = y1;
        cb[row * (width / 2) + (x / 2)] = static_cast<Npp8u>((cb0 + cb1) / 2);
        cr[row * (width / 2) + (x / 2)] = static_cast<Npp8u>((cr0 + cr1) / 2);
      }
    }
  }

  void computeRGBFromYCbCr422Planar(const std::vector<Npp8u> &y, const std::vector<Npp8u> &cb,
                                    const std::vector<Npp8u> &cr, std::vector<Npp8u> &rgb) const {
    rgb.assign(width * height * 3, 0);
    for (int row = 0; row < height; ++row) {
      for (int x = 0; x < width; ++x) {
        int cbx = x >> 1;
        Npp8u r, g, b;
        ycbcr_to_rgb_pixel(y[row * width + x], cb[row * (width / 2) + cbx], cr[row * (width / 2) + cbx], r, g, b);
        int idx = (row * width + x) * 3;
        rgb[idx + 0] = r;
        rgb[idx + 1] = g;
        rgb[idx + 2] = b;
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

struct YCbCr422Case {
  int width;
  int height;
  int seed;
  bool random;
};

class YCbCr422PlanarParamTest : public ::testing::TestWithParam<YCbCr422Case> {
protected:
  void SetUp() override {
    const auto &param = GetParam();
    width = param.width;
    height = param.height;
    seed = param.seed;
    use_random = param.random;
    ASSERT_GT(width, 0);
    ASSERT_GT(height, 0);
    ASSERT_EQ(width % 2, 0);
  }

  void createPackedRGB(std::vector<Npp8u> &rgb) const {
    rgb.resize(width * height * 3);
    if (use_random) {
      std::mt19937 rng(static_cast<unsigned int>(seed));
      std::uniform_int_distribution<int> dist(0, 255);
      for (int i = 0; i < width * height * 3; ++i) {
        rgb[i] = static_cast<Npp8u>(dist(rng));
      }
      return;
    }

    for (int y = 0; y < height; ++y) {
      for (int x = 0; x < width; ++x) {
        int idx = (y * width + x) * 3;
        rgb[idx + 0] = static_cast<Npp8u>(12 + x * 23 + y * 5);
        rgb[idx + 1] = static_cast<Npp8u>(34 + x * 11 + y * 9);
        rgb[idx + 2] = static_cast<Npp8u>(56 + x * 7 + y * 13);
      }
    }
  }

  void computeYCbCr422Planar(const std::vector<Npp8u> &rgb, std::vector<Npp8u> &y, std::vector<Npp8u> &cb,
                             std::vector<Npp8u> &cr) const {
    y.assign(width * height, 0);
    cb.assign((width / 2) * height, 0);
    cr.assign((width / 2) * height, 0);

    for (int row = 0; row < height; ++row) {
      for (int x = 0; x < width; x += 2) {
        int idx0 = (row * width + x) * 3;
        int idx1 = (row * width + x + 1) * 3;
        Npp8u y0, cb0, cr0;
        Npp8u y1, cb1, cr1;
        rgb_to_ycbcr_pixel(rgb[idx0 + 0], rgb[idx0 + 1], rgb[idx0 + 2], y0, cb0, cr0);
        rgb_to_ycbcr_pixel(rgb[idx1 + 0], rgb[idx1 + 1], rgb[idx1 + 2], y1, cb1, cr1);

        y[row * width + x] = y0;
        y[row * width + x + 1] = y1;
        cb[row * (width / 2) + (x / 2)] = static_cast<Npp8u>((cb0 + cb1) / 2);
        cr[row * (width / 2) + (x / 2)] = static_cast<Npp8u>((cr0 + cr1) / 2);
      }
    }
  }

  void computeRGBFromYCbCr422Planar(const std::vector<Npp8u> &y, const std::vector<Npp8u> &cb,
                                    const std::vector<Npp8u> &cr, std::vector<Npp8u> &rgb) const {
    rgb.assign(width * height * 3, 0);
    for (int row = 0; row < height; ++row) {
      for (int x = 0; x < width; ++x) {
        int cbx = x >> 1;
        Npp8u r, g, b;
        ycbcr_to_rgb_pixel(y[row * width + x], cb[row * (width / 2) + cbx], cr[row * (width / 2) + cbx], r, g, b);
        int idx = (row * width + x) * 3;
        rgb[idx + 0] = r;
        rgb[idx + 1] = g;
        rgb[idx + 2] = b;
      }
    }
  }

  int width = 0;
  int height = 0;
  int seed = 0;
  bool use_random = false;
};

} // namespace

TEST_F(YCbCr422Test, RGBToYCbCr422_And_Back_ExpectedValues) {
  int deviceCount = 0;
  cudaError_t devErr = cudaGetDeviceCount(&deviceCount);
  if (devErr != cudaSuccess || deviceCount == 0) {
    GTEST_FAIL() << "CUDA device unavailable for YCbCr422 tests";
  }
  std::vector<Npp8u> hostRGB;
  createPackedRGB(hostRGB);

  int srcStep = 0;
  Npp8u *d_src = nppiMalloc_8u_C3(width, height, &srcStep);
  if (!d_src) {
    GTEST_FAIL() << "nppiMalloc_8u_C3 failed for YCbCr422 tests";
  }

  int c2Step = 0;
  Npp8u *d_c2 = nppiMalloc_8u_C2(width, height, &c2Step);
  if (!d_c2) {
    nppiFree(d_src);
    GTEST_FAIL() << "nppiMalloc_8u_C2 failed for YCbCr422 tests";
  }

  cudaMemcpy2D(d_src, srcStep, hostRGB.data(), width * 3, width * 3, height, cudaMemcpyHostToDevice);

  NppiSize roi = {width, height};
  NppStatus status = nppiRGBToYCbCr422_8u_C3C2R(d_src, srcStep, d_c2, c2Step, roi);
  EXPECT_EQ(status, NPP_NO_ERROR);

  std::vector<Npp8u> hostC2(c2Step * height);
  cudaMemcpy(hostC2.data(), d_c2, hostC2.size(), cudaMemcpyDeviceToHost);

  int dstStep = 0;
  Npp8u *d_dst = nppiMalloc_8u_C3(width, height, &dstStep);
  if (!d_dst) {
    nppiFree(d_src);
    nppiFree(d_c2);
    GTEST_FAIL() << "nppiMalloc_8u_C3 failed for YCbCr422 tests";
  }

  status = nppiYCbCr422ToRGB_8u_C2C3R(d_c2, c2Step, d_dst, dstStep, roi);
  EXPECT_EQ(status, NPP_NO_ERROR);

  std::vector<Npp8u> hostRGBOut(dstStep * height);
  cudaMemcpy(hostRGBOut.data(), d_dst, hostRGBOut.size(), cudaMemcpyDeviceToHost);

  std::vector<Npp8u> flatC2(width * height * 2);
  std::vector<Npp8u> flatRGB(width * height * 3);

  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width * 2; ++x) {
      flatC2[y * width * 2 + x] = hostC2[y * c2Step + x];
    }
  }

  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      int idx = (y * width + x) * 3;
      int srcIdx = y * dstStep + x * 3;
      flatRGB[idx + 0] = hostRGBOut[srcIdx + 0];
      flatRGB[idx + 1] = hostRGBOut[srcIdx + 1];
      flatRGB[idx + 2] = hostRGBOut[srcIdx + 2];
    }
  }

  if (std::getenv("NPP_DUMP_EXPECTED")) {
    dumpExpected("kExpectedYCbCr422C2", flatC2, width * height * 2);
    dumpExpected("kExpectedYCbCr422ToRGB", flatRGB, width * height * 3);
    GTEST_SKIP();
  }

  const Npp8u kExpectedYCbCr422C2[32] = {41, 138, 53, 119, 65, 131, 78, 130, 48, 141, 60, 117, 73,
                                         134, 85, 128, 55, 143, 68, 115, 80, 136, 92, 126, 62, 145,
                                         75, 113, 87, 138, 99, 124};
  const Npp8u kExpectedYCbCr422ToRGB[48] = {14, 32, 49, 28, 46, 63, 60, 54, 63, 75, 69, 78,
                                            19, 41, 63, 33, 55, 77, 66, 63, 78, 80, 77, 92,
                                            24, 50, 75, 39, 65, 90, 71, 72, 90, 85, 86, 104,
                                            29, 59, 87, 44, 74, 102, 76, 81, 102, 90, 95, 116};

  for (int i = 0; i < width * height * 2; ++i) {
    EXPECT_EQ(flatC2[i], kExpectedYCbCr422C2[i]) << "C2 mismatch at " << i;
  }
  for (int i = 0; i < width * height * 3; ++i) {
    EXPECT_EQ(flatRGB[i], kExpectedYCbCr422ToRGB[i]) << "RGB mismatch at " << i;
  }

  nppiFree(d_src);
  nppiFree(d_c2);
  nppiFree(d_dst);
}

TEST_P(YCbCr422PlanarParamTest, RGBToYCbCr422_C3P3R_ExactMatch) {
  SCOPED_TRACE(use_random ? "precision" : "functional");
  std::vector<Npp8u> hostRGB;
  createPackedRGB(hostRGB);

  int srcStep = 0;
  Npp8u *d_src = nppiMalloc_8u_C3(width, height, &srcStep);
  if (!d_src) {
    GTEST_FAIL() << "nppiMalloc_8u_C3 failed for YCbCr422 tests";
  }

  int yStep = 0;
  int cbStep = 0;
  int crStep = 0;
  Npp8u *d_y = nppiMalloc_8u_C1(width, height, &yStep);
  Npp8u *d_cb = nppiMalloc_8u_C1(width / 2, height, &cbStep);
  Npp8u *d_cr = nppiMalloc_8u_C1(width / 2, height, &crStep);
  if (!d_y || !d_cb || !d_cr) {
    if (d_y)
      nppiFree(d_y);
    if (d_cb)
      nppiFree(d_cb);
    if (d_cr)
      nppiFree(d_cr);
    nppiFree(d_src);
    GTEST_FAIL() << "nppiMalloc_8u_C1 failed for YCbCr422 tests";
  }

  cudaMemcpy2D(d_src, srcStep, hostRGB.data(), width * 3, width * 3, height, cudaMemcpyHostToDevice);

  Npp8u *dst_planes[3] = {d_y, d_cb, d_cr};
  int dst_steps[3] = {yStep, cbStep, crStep};
  NppiSize roi = {width, height};
  NppStatus status = nppiRGBToYCbCr422_8u_C3P3R(d_src, srcStep, dst_planes, dst_steps, roi);
  EXPECT_EQ(status, NPP_NO_ERROR);

  std::vector<Npp8u> yPlane(yStep * height);
  std::vector<Npp8u> cbPlane(cbStep * height);
  std::vector<Npp8u> crPlane(crStep * height);
  cudaMemcpy(yPlane.data(), d_y, yPlane.size(), cudaMemcpyDeviceToHost);
  cudaMemcpy(cbPlane.data(), d_cb, cbPlane.size(), cudaMemcpyDeviceToHost);
  cudaMemcpy(crPlane.data(), d_cr, crPlane.size(), cudaMemcpyDeviceToHost);

  std::vector<Npp8u> flatY(width * height);
  std::vector<Npp8u> flatCb((width / 2) * height);
  std::vector<Npp8u> flatCr((width / 2) * height);
  for (int row = 0; row < height; ++row) {
    std::memcpy(&flatY[row * width], &yPlane[row * yStep], width);
    std::memcpy(&flatCb[row * (width / 2)], &cbPlane[row * cbStep], width / 2);
    std::memcpy(&flatCr[row * (width / 2)], &crPlane[row * crStep], width / 2);
  }

  std::vector<Npp8u> expectedY;
  std::vector<Npp8u> expectedCb;
  std::vector<Npp8u> expectedCr;
  computeYCbCr422Planar(hostRGB, expectedY, expectedCb, expectedCr);

  for (int i = 0; i < width * height; ++i) {
    EXPECT_EQ(flatY[i], expectedY[i]) << "Y mismatch at " << i;
  }
  for (int i = 0; i < (width / 2) * height; ++i) {
    EXPECT_EQ(flatCb[i], expectedCb[i]) << "Cb mismatch at " << i;
    EXPECT_EQ(flatCr[i], expectedCr[i]) << "Cr mismatch at " << i;
  }

  nppiFree(d_src);
  nppiFree(d_y);
  nppiFree(d_cb);
  nppiFree(d_cr);
}

TEST_P(YCbCr422PlanarParamTest, YCbCr422ToRGB_P3C3R_ExactMatch) {
  SCOPED_TRACE(use_random ? "precision" : "functional");
  std::vector<Npp8u> hostRGB;
  createPackedRGB(hostRGB);

  std::vector<Npp8u> yPlane;
  std::vector<Npp8u> cbPlane;
  std::vector<Npp8u> crPlane;
  computeYCbCr422Planar(hostRGB, yPlane, cbPlane, crPlane);

  int yStep = 0;
  int cbStep = 0;
  int crStep = 0;
  Npp8u *d_y = nppiMalloc_8u_C1(width, height, &yStep);
  Npp8u *d_cb = nppiMalloc_8u_C1(width / 2, height, &cbStep);
  Npp8u *d_cr = nppiMalloc_8u_C1(width / 2, height, &crStep);

  for (int row = 0; row < height; ++row) {
    cudaMemcpy((char *)d_y + row * yStep, &yPlane[row * width], width, cudaMemcpyHostToDevice);
    cudaMemcpy((char *)d_cb + row * cbStep, &cbPlane[row * (width / 2)], width / 2, cudaMemcpyHostToDevice);
    cudaMemcpy((char *)d_cr + row * crStep, &crPlane[row * (width / 2)], width / 2, cudaMemcpyHostToDevice);
  }

  int dstStep = 0;
  Npp8u *d_dst = nppiMalloc_8u_C3(width, height, &dstStep);
  if (!d_dst) {
    nppiFree(d_y);
    nppiFree(d_cb);
    nppiFree(d_cr);
    GTEST_FAIL() << "nppiMalloc_8u_C3 failed for YCbCr422 tests";
  }

  const Npp8u *src_planes[3] = {d_y, d_cb, d_cr};
  int src_steps[3] = {yStep, cbStep, crStep};
  NppiSize roi = {width, height};
  NppStatus status = nppiYCbCr422ToRGB_8u_P3C3R(src_planes, src_steps, d_dst, dstStep, roi);
  EXPECT_EQ(status, NPP_NO_ERROR);

  std::vector<Npp8u> rgbOut(dstStep * height);
  cudaMemcpy(rgbOut.data(), d_dst, rgbOut.size(), cudaMemcpyDeviceToHost);

  std::vector<Npp8u> flatRGB(width * height * 3);
  for (int row = 0; row < height; ++row) {
    for (int x = 0; x < width; ++x) {
      int idx = (row * width + x) * 3;
      int srcIdx = row * dstStep + x * 3;
      flatRGB[idx + 0] = rgbOut[srcIdx + 0];
      flatRGB[idx + 1] = rgbOut[srcIdx + 1];
      flatRGB[idx + 2] = rgbOut[srcIdx + 2];
    }
  }

  std::vector<Npp8u> expectedRGB;
  computeRGBFromYCbCr422Planar(yPlane, cbPlane, crPlane, expectedRGB);

  // For random data (precision tests), allow tolerance of 1 due to rounding errors
  // in the round-trip conversion (RGB->YCbCr->RGB)
  int tolerance = use_random ? 1 : 0;
  for (int i = 0; i < width * height * 3; ++i) {
    int diff = std::abs(static_cast<int>(flatRGB[i]) - static_cast<int>(expectedRGB[i]));
    EXPECT_LE(diff, tolerance) << "RGB mismatch at " << i
                               << " (got " << static_cast<int>(flatRGB[i])
                               << ", expected " << static_cast<int>(expectedRGB[i]) << ")";
  }

  nppiFree(d_y);
  nppiFree(d_cb);
  nppiFree(d_cr);
  nppiFree(d_dst);
}

INSTANTIATE_TEST_SUITE_P(FunctionalCases, YCbCr422PlanarParamTest,
                         ::testing::Values(YCbCr422Case{4, 4, 0, false}, YCbCr422Case{8, 2, 0, false},
                                           YCbCr422Case{16, 6, 0, false}));

INSTANTIATE_TEST_SUITE_P(PrecisionCases, YCbCr422PlanarParamTest,
                         ::testing::Values(YCbCr422Case{32, 16, 123, true}, YCbCr422Case{64, 8, 456, true}));
