#include "npp.h"
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <iostream>
#include <vector>

namespace {

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

TEST_F(YCbCr422Test, RGBToYCbCr422_And_Back_ExpectedValues) {
  std::vector<Npp8u> hostRGB;
  createPackedRGB(hostRGB);

  int srcStep = 0;
  Npp8u *d_src = nppiMalloc_8u_C3(width, height, &srcStep);
  ASSERT_NE(d_src, nullptr);

  int c2Step = 0;
  Npp8u *d_c2 = nppiMalloc_8u_C2(width, height, &c2Step);
  ASSERT_NE(d_c2, nullptr);

  cudaMemcpy2D(d_src, srcStep, hostRGB.data(), width * 3, width * 3, height, cudaMemcpyHostToDevice);

  NppiSize roi = {width, height};
  NppStatus status = nppiRGBToYCbCr422_8u_C3C2R(d_src, srcStep, d_c2, c2Step, roi);
  EXPECT_EQ(status, NPP_NO_ERROR);

  std::vector<Npp8u> hostC2(c2Step * height);
  cudaMemcpy(hostC2.data(), d_c2, hostC2.size(), cudaMemcpyDeviceToHost);

  int dstStep = 0;
  Npp8u *d_dst = nppiMalloc_8u_C3(width, height, &dstStep);
  ASSERT_NE(d_dst, nullptr);

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
