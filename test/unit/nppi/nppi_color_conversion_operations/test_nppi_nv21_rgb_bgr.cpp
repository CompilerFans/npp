#include "npp.h"
#include "npp_test_base.h"

#include <algorithm>
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <vector>

namespace {

class NV21ToRGBBGRTest : public ::testing::Test {
protected:
  void SetUp() override {
    width = 32;
    height = 32;
    ASSERT_EQ(width % 2, 0);
    ASSERT_EQ(height % 2, 0);
  }

  void createTestNV21(std::vector<Npp8u> &yData, std::vector<Npp8u> &vuData) {
    yData.resize(width * height);
    vuData.resize(width * height / 2);

    for (int y = 0; y < height; ++y) {
      for (int x = 0; x < width; ++x) {
        yData[y * width + x] = static_cast<Npp8u>(16 + (x + y) * 239 / (width + height - 2));
      }
    }

    for (int y = 0; y < height / 2; ++y) {
      for (int x = 0; x < width; x += 2) {
        int idx = y * width + x;
        vuData[idx] = static_cast<Npp8u>(64 + y * 128 / (height / 2));
        vuData[idx + 1] = static_cast<Npp8u>(64 + x * 128 / width);
      }
    }
  }

  int shiftRightFloor(int value, int bits) const {
    if (value >= 0) {
      return value >> bits;
    }
    const int denom = 1 << bits;
    return -(((-value) + denom - 1) / denom);
  }

  void nv21ToRgbExpected(Npp8u y, Npp8u u, Npp8u v, Npp8u &r, Npp8u &g, Npp8u &b) const {
    const int yv = static_cast<int>(y);
    const int ud = static_cast<int>(u) - 128;
    const int vd = static_cast<int>(v) - 128;

    int r_val = yv + shiftRightFloor(455 * vd, 8);
    int g_val = yv + shiftRightFloor(181 * ud - 89 * vd, 8);
    int b_val = yv + shiftRightFloor(359 * ud, 8);

    r_val = std::max(0, std::min(255, r_val));
    g_val = std::max(0, std::min(255, g_val));
    b_val = std::max(0, std::min(255, b_val));

    r = static_cast<Npp8u>(r_val);
    g = static_cast<Npp8u>(g_val);
    b = static_cast<Npp8u>(b_val);
  }

  int width = 0;
  int height = 0;
};

} // namespace

TEST_F(NV21ToRGBBGRTest, NV21ToRGB_8u_P2C4R) {
  std::vector<Npp8u> hostYData, hostVUData;
  createTestNV21(hostYData, hostVUData);

  Npp8u *d_srcY = nppsMalloc_8u(width * height);
  Npp8u *d_srcVU = nppsMalloc_8u(width * height / 2);

  int rgbaStep = 0;
  Npp8u *d_rgba = nppiMalloc_8u_C4(width, height, &rgbaStep);

  ASSERT_NE(d_srcY, nullptr);
  ASSERT_NE(d_srcVU, nullptr);
  ASSERT_NE(d_rgba, nullptr);

  cudaMemcpy(d_srcY, hostYData.data(), hostYData.size(), cudaMemcpyHostToDevice);
  cudaMemcpy(d_srcVU, hostVUData.data(), hostVUData.size(), cudaMemcpyHostToDevice);

  const Npp8u *pSrc[2] = {d_srcY, d_srcVU};
  NppiSize roi = {width, height};

  NppStatus status = nppiNV21ToRGB_8u_P2C4R(pSrc, width, d_rgba, rgbaStep, roi);
  EXPECT_EQ(status, NPP_NO_ERROR);

  std::vector<Npp8u> hostRGBA(rgbaStep * height);
  cudaMemcpy(hostRGBA.data(), d_rgba, rgbaStep * height, cudaMemcpyDeviceToHost);
  std::vector<Npp8u> baseRGBA = hostRGBA;

  for (int y = 0; y < height; y += 4) {
    for (int x = 0; x < width; x += 4) {
      int idx = y * rgbaStep + x * 4;
      EXPECT_EQ(hostRGBA[idx + 3], 0xFF);
    }
  }

  for (int y = 0; y < height; y += 8) {
    for (int x = 0; x < width; x += 8) {
      int yIdx = y * width + x;
      int uvIdx = (y / 2) * width + (x & ~1);
      Npp8u Y = hostYData[yIdx];
      Npp8u V = hostVUData[uvIdx];
      Npp8u U = hostVUData[uvIdx + 1];

      Npp8u er, eg, eb;
      nv21ToRgbExpected(Y, U, V, er, eg, eb);

      int outIdx = y * rgbaStep + x * 4;
      Npp8u ar = hostRGBA[outIdx];
      Npp8u ag = hostRGBA[outIdx + 1];
      Npp8u ab = hostRGBA[outIdx + 2];

      EXPECT_EQ(ar, er);
      EXPECT_EQ(ag, eg);
      EXPECT_EQ(ab, eb);
    }
  }

  NppStreamContext ctx{};
  nppGetStreamContext(&ctx);
  ctx.hStream = 0;
  status = nppiNV21ToRGB_8u_P2C4R_Ctx(pSrc, width, d_rgba, rgbaStep, roi, ctx);
  EXPECT_EQ(status, NPP_NO_ERROR);

  std::vector<Npp8u> ctxRGBA(rgbaStep * height);
  cudaMemcpy(ctxRGBA.data(), d_rgba, rgbaStep * height, cudaMemcpyDeviceToHost);
  for (size_t i = 0; i < ctxRGBA.size(); ++i) {
    EXPECT_EQ(ctxRGBA[i], baseRGBA[i]) << "Ctx mismatch at " << i;
  }

  nppsFree(d_srcY);
  nppsFree(d_srcVU);
  nppiFree(d_rgba);
}

TEST_F(NV21ToRGBBGRTest, NV21ToBGR_8u_P2C4R) {
  std::vector<Npp8u> hostYData, hostVUData;
  createTestNV21(hostYData, hostVUData);

  Npp8u *d_srcY = nppsMalloc_8u(width * height);
  Npp8u *d_srcVU = nppsMalloc_8u(width * height / 2);

  int bgraStep = 0;
  Npp8u *d_bgra = nppiMalloc_8u_C4(width, height, &bgraStep);

  ASSERT_NE(d_srcY, nullptr);
  ASSERT_NE(d_srcVU, nullptr);
  ASSERT_NE(d_bgra, nullptr);

  cudaMemcpy(d_srcY, hostYData.data(), hostYData.size(), cudaMemcpyHostToDevice);
  cudaMemcpy(d_srcVU, hostVUData.data(), hostVUData.size(), cudaMemcpyHostToDevice);

  const Npp8u *pSrc[2] = {d_srcY, d_srcVU};
  NppiSize roi = {width, height};

  NppStatus status = nppiNV21ToBGR_8u_P2C4R(pSrc, width, d_bgra, bgraStep, roi);
  EXPECT_EQ(status, NPP_NO_ERROR);

  std::vector<Npp8u> hostBGRA(bgraStep * height);
  cudaMemcpy(hostBGRA.data(), d_bgra, bgraStep * height, cudaMemcpyDeviceToHost);
  std::vector<Npp8u> baseBGRA = hostBGRA;

  for (int y = 0; y < height; y += 4) {
    for (int x = 0; x < width; x += 4) {
      int idx = y * bgraStep + x * 4;
      EXPECT_EQ(hostBGRA[idx + 3], 0xFF);
    }
  }

  for (int y = 0; y < height; y += 8) {
    for (int x = 0; x < width; x += 8) {
      int yIdx = y * width + x;
      int uvIdx = (y / 2) * width + (x & ~1);
      Npp8u Y = hostYData[yIdx];
      Npp8u V = hostVUData[uvIdx];
      Npp8u U = hostVUData[uvIdx + 1];

      Npp8u er, eg, eb;
      nv21ToRgbExpected(Y, U, V, er, eg, eb);

      int outIdx = y * bgraStep + x * 4;
      Npp8u ab = hostBGRA[outIdx];
      Npp8u ag = hostBGRA[outIdx + 1];
      Npp8u ar = hostBGRA[outIdx + 2];

      EXPECT_EQ(ab, eb);
      EXPECT_EQ(ag, eg);
      EXPECT_EQ(ar, er);
    }
  }

  NppStreamContext ctx{};
  nppGetStreamContext(&ctx);
  ctx.hStream = 0;
  status = nppiNV21ToBGR_8u_P2C4R_Ctx(pSrc, width, d_bgra, bgraStep, roi, ctx);
  EXPECT_EQ(status, NPP_NO_ERROR);

  std::vector<Npp8u> ctxBGRA(bgraStep * height);
  cudaMemcpy(ctxBGRA.data(), d_bgra, bgraStep * height, cudaMemcpyDeviceToHost);
  for (size_t i = 0; i < ctxBGRA.size(); ++i) {
    EXPECT_EQ(ctxBGRA[i], baseBGRA[i]) << "Ctx mismatch at " << i;
  }

  nppsFree(d_srcY);
  nppsFree(d_srcVU);
  nppiFree(d_bgra);
}
