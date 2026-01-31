#include "npp.h"
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <iostream>
#include <vector>

namespace {

class BgrYCbCr422Test : public ::testing::Test {
protected:
  void SetUp() override {
    width = 4;
    height = 4;
    ASSERT_EQ(width % 2, 0);
  }

  void createPackedBGR(std::vector<Npp8u> &bgr) const {
    bgr.resize(width * height * 3);
    for (int y = 0; y < height; ++y) {
      for (int x = 0; x < width; ++x) {
        int idx = (y * width + x) * 3;
        Npp8u r = static_cast<Npp8u>(12 + x * 23 + y * 5);
        Npp8u g = static_cast<Npp8u>(34 + x * 11 + y * 9);
        Npp8u b = static_cast<Npp8u>(56 + x * 7 + y * 13);
        bgr[idx + 0] = b;
        bgr[idx + 1] = g;
        bgr[idx + 2] = r;
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

TEST_F(BgrYCbCr422Test, BGRToYCbCr422_And_Back_ExpectedValues) {
  std::vector<Npp8u> hostBGR;
  createPackedBGR(hostBGR);

  int srcStep = 0;
  Npp8u *d_src = nppiMalloc_8u_C3(width, height, &srcStep);
  ASSERT_NE(d_src, nullptr);

  int c2Step = 0;
  Npp8u *d_c2 = nppiMalloc_8u_C2(width, height, &c2Step);
  ASSERT_NE(d_c2, nullptr);

  cudaMemcpy2D(d_src, srcStep, hostBGR.data(), width * 3, width * 3, height, cudaMemcpyHostToDevice);

  NppiSize roi = {width, height};
  NppStatus status = nppiBGRToYCbCr422_8u_C3C2R(d_src, srcStep, d_c2, c2Step, roi);
  EXPECT_EQ(status, NPP_NO_ERROR);

  std::vector<Npp8u> hostC2(c2Step * height);
  cudaMemcpy(hostC2.data(), d_c2, hostC2.size(), cudaMemcpyDeviceToHost);

  int dstStep = 0;
  Npp8u *d_dst = nppiMalloc_8u_C3(width, height, &dstStep);
  ASSERT_NE(d_dst, nullptr);

  status = nppiYCbCr422ToBGR_8u_C2C3R(d_c2, c2Step, d_dst, dstStep, roi);
  EXPECT_EQ(status, NPP_NO_ERROR);

  std::vector<Npp8u> hostBGROut(dstStep * height);
  cudaMemcpy(hostBGROut.data(), d_dst, hostBGROut.size(), cudaMemcpyDeviceToHost);

  std::vector<Npp8u> flatC2(width * height * 2);
  std::vector<Npp8u> flatBGR(width * height * 3);

  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width * 2; ++x) {
      flatC2[y * width * 2 + x] = hostC2[y * c2Step + x];
    }
  }

  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      int idx = (y * width + x) * 3;
      int srcIdx = y * dstStep + x * 3;
      flatBGR[idx + 0] = hostBGROut[srcIdx + 0];
      flatBGR[idx + 1] = hostBGROut[srcIdx + 1];
      flatBGR[idx + 2] = hostBGROut[srcIdx + 2];
    }
  }

  if (std::getenv("NPP_DUMP_EXPECTED")) {
    dumpExpected("kExpectedBGRYCbCr422C2", flatC2, width * height * 2);
    dumpExpected("kExpectedYCbCr422ToBGR", flatBGR, width * height * 3);
    GTEST_SKIP();
  }

  const Npp8u kExpectedBGRYCbCr422C2[32] = {41, 138, 53, 119, 65, 131, 78, 130, 48, 141, 60, 117, 73,
                                            134, 85, 128, 55, 143, 68, 115, 80, 136, 92, 126, 62, 145,
                                            75, 113, 87, 138, 99, 124};
  const Npp8u kExpectedYCbCr422ToBGR[48] = {49, 32, 14, 63, 46, 28, 63, 54, 60, 78, 69, 75,
                                            63, 41, 19, 77, 55, 33, 78, 63, 66, 92, 77, 80,
                                            75, 50, 24, 90, 65, 39, 90, 72, 71, 104, 86, 85,
                                            87, 59, 29, 102, 74, 44, 102, 81, 76, 116, 95, 90};

  for (int i = 0; i < width * height * 2; ++i) {
    EXPECT_EQ(flatC2[i], kExpectedBGRYCbCr422C2[i]) << "C2 mismatch at " << i;
  }
  for (int i = 0; i < width * height * 3; ++i) {
    EXPECT_EQ(flatBGR[i], kExpectedYCbCr422ToBGR[i]) << "BGR mismatch at " << i;
  }

  NppStreamContext ctx{};
  nppGetStreamContext(&ctx);
  ctx.hStream = 0;
  status = nppiBGRToYCbCr422_8u_C3C2R_Ctx(d_src, srcStep, d_c2, c2Step, roi, ctx);
  EXPECT_EQ(status, NPP_NO_ERROR);

  cudaMemcpy(hostC2.data(), d_c2, hostC2.size(), cudaMemcpyDeviceToHost);
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width * 2; ++x) {
      flatC2[y * width * 2 + x] = hostC2[y * c2Step + x];
    }
  }
  for (int i = 0; i < width * height * 2; ++i) {
    EXPECT_EQ(flatC2[i], kExpectedBGRYCbCr422C2[i]) << "Ctx C2 mismatch at " << i;
  }

  status = nppiYCbCr422ToBGR_8u_C2C3R_Ctx(d_c2, c2Step, d_dst, dstStep, roi, ctx);
  EXPECT_EQ(status, NPP_NO_ERROR);

  cudaMemcpy(hostBGROut.data(), d_dst, hostBGROut.size(), cudaMemcpyDeviceToHost);
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      int idx = (y * width + x) * 3;
      int srcIdx = y * dstStep + x * 3;
      flatBGR[idx + 0] = hostBGROut[srcIdx + 0];
      flatBGR[idx + 1] = hostBGROut[srcIdx + 1];
      flatBGR[idx + 2] = hostBGROut[srcIdx + 2];
    }
  }
  for (int i = 0; i < width * height * 3; ++i) {
    EXPECT_EQ(flatBGR[i], kExpectedYCbCr422ToBGR[i]) << "Ctx BGR mismatch at " << i;
  }

  nppiFree(d_src);
  nppiFree(d_c2);
  nppiFree(d_dst);
}
