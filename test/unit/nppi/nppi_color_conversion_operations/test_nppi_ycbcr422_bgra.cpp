#include "npp.h"
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <iostream>
#include <vector>

namespace {

class BgrAtoYCbCr422Test : public ::testing::Test {
protected:
  void SetUp() override {
    width = 4;
    height = 4;
    ASSERT_EQ(width % 2, 0);
  }

  void createPackedBGRA(std::vector<Npp8u> &bgra) const {
    bgra.resize(width * height * 4);
    for (int y = 0; y < height; ++y) {
      for (int x = 0; x < width; ++x) {
        int idx = (y * width + x) * 4;
        Npp8u r = static_cast<Npp8u>(12 + x * 23 + y * 5);
        Npp8u g = static_cast<Npp8u>(34 + x * 11 + y * 9);
        Npp8u b = static_cast<Npp8u>(56 + x * 7 + y * 13);
        bgra[idx + 0] = b;
        bgra[idx + 1] = g;
        bgra[idx + 2] = r;
        bgra[idx + 3] = static_cast<Npp8u>(200 + x + y);
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

TEST_F(BgrAtoYCbCr422Test, BGRAToYCbCr422_And_Back_ExpectedValues) {
  std::vector<Npp8u> hostBGRA;
  createPackedBGRA(hostBGRA);

  int srcStep = 0;
  Npp8u *d_src = nppiMalloc_8u_C4(width, height, &srcStep);
  ASSERT_NE(d_src, nullptr);

  int c2Step = 0;
  Npp8u *d_c2 = nppiMalloc_8u_C2(width, height, &c2Step);
  ASSERT_NE(d_c2, nullptr);

  cudaMemcpy2D(d_src, srcStep, hostBGRA.data(), width * 4, width * 4, height, cudaMemcpyHostToDevice);

  NppiSize roi = {width, height};
  NppStatus status = nppiBGRToYCbCr422_8u_AC4C2R(d_src, srcStep, d_c2, c2Step, roi);
  EXPECT_EQ(status, NPP_NO_ERROR);

  std::vector<Npp8u> hostC2(c2Step * height);
  cudaMemcpy(hostC2.data(), d_c2, hostC2.size(), cudaMemcpyDeviceToHost);

  int dstStep = 0;
  Npp8u *d_dst = nppiMalloc_8u_C4(width, height, &dstStep);
  ASSERT_NE(d_dst, nullptr);

  const Npp8u alpha = 77;
  status = nppiYCbCr422ToBGR_8u_C2C4R(d_c2, c2Step, d_dst, dstStep, roi, alpha);
  EXPECT_EQ(status, NPP_NO_ERROR);

  std::vector<Npp8u> hostBGRAOut(dstStep * height);
  cudaMemcpy(hostBGRAOut.data(), d_dst, hostBGRAOut.size(), cudaMemcpyDeviceToHost);

  std::vector<Npp8u> flatC2(width * height * 2);
  std::vector<Npp8u> flatBGRA(width * height * 4);

  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width * 2; ++x) {
      flatC2[y * width * 2 + x] = hostC2[y * c2Step + x];
    }
  }

  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      int idx = (y * width + x) * 4;
      int srcIdx = y * dstStep + x * 4;
      flatBGRA[idx + 0] = hostBGRAOut[srcIdx + 0];
      flatBGRA[idx + 1] = hostBGRAOut[srcIdx + 1];
      flatBGRA[idx + 2] = hostBGRAOut[srcIdx + 2];
      flatBGRA[idx + 3] = hostBGRAOut[srcIdx + 3];
    }
  }

  if (std::getenv("NPP_DUMP_EXPECTED")) {
    dumpExpected("kExpectedBGRAToYCbCr422C2", flatC2, width * height * 2);
    dumpExpected("kExpectedYCbCr422ToBGRA", flatBGRA, width * height * 4);
    GTEST_SKIP();
  }

  const Npp8u kExpectedBGRAToYCbCr422C2[32] = {41, 138, 53, 119, 65, 131, 78, 130, 48, 141, 60, 117, 73,
                                               134, 85, 128, 55, 143, 68, 115, 80, 136, 92, 126, 62, 145,
                                               75, 113, 87, 138, 99, 124};
  const Npp8u kExpectedYCbCr422ToBGRA[64] = {49, 32, 14, 77, 63, 46, 28, 77, 63, 54, 60, 77, 78, 69, 75, 77,
                                             63, 41, 19, 77, 77, 55, 33, 77, 78, 63, 66, 77, 92, 77, 80, 77,
                                             75, 50, 24, 77, 90, 65, 39, 77, 90, 72, 71, 77, 104, 86, 85, 77,
                                             87, 59, 29, 77, 102, 74, 44, 77, 102, 81, 76, 77, 116, 95, 90, 77};

  for (int i = 0; i < width * height * 2; ++i) {
    EXPECT_EQ(flatC2[i], kExpectedBGRAToYCbCr422C2[i]) << "C2 mismatch at " << i;
  }
  for (int i = 0; i < width * height * 4; ++i) {
    EXPECT_EQ(flatBGRA[i], kExpectedYCbCr422ToBGRA[i]) << "BGRA mismatch at " << i;
  }

  nppiFree(d_src);
  nppiFree(d_c2);
  nppiFree(d_dst);
}
