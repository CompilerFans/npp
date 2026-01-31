#include "npp_test_base.h"
#include <algorithm>
#include <vector>

using namespace npp_functional_test;

namespace {
static inline void yuv_to_rgb_bt601_ref(Npp8u y, Npp8u u, Npp8u v, Npp8u &r, Npp8u &g, Npp8u &b) {
  float Y = static_cast<float>(y);
  float U = static_cast<float>(u) - 128.0f;
  float V = static_cast<float>(v) - 128.0f;

  float rf = Y + 1.140f * V;
  float gf = Y - 0.395f * U - 0.581f * V;
  float bf = Y + 2.032f * U;

  int r_val = static_cast<int>(std::min(255.0f, std::max(0.0f, rf)));
  int g_val = static_cast<int>(std::min(255.0f, std::max(0.0f, gf)));
  int b_val = static_cast<int>(std::min(255.0f, std::max(0.0f, bf)));

  r = static_cast<Npp8u>(r_val);
  g = static_cast<Npp8u>(g_val);
  b = static_cast<Npp8u>(b_val);
}

static inline void yuv422_yuyv_pixel(const std::vector<Npp8u> &src, int srcStep, int x, int y, Npp8u &Y,
                                     Npp8u &U, Npp8u &V) {
  const Npp8u *row = src.data() + y * srcStep;
  int pair = (x / 2) * 4;
  Npp8u y0 = row[pair + 0];
  Npp8u u0 = row[pair + 1];
  Npp8u y1 = row[pair + 2];
  Npp8u v0 = row[pair + 3];
  if ((x & 1) == 0) {
    Y = y0;
    U = u0;
    V = v0;
  } else {
    Y = y1;
    U = u0;
    V = v0;
  }
}
} // namespace

class YUV422ToRGBTest : public NppTestBase {};

TEST_F(YUV422ToRGBTest, YUV422ToRGB_8u_P3C3R_Reference) {
  const int width = 2;
  const int height = 2;

  std::vector<Npp8u> yPlane = {16, 235, 81, 145};
  std::vector<Npp8u> uPlane((width / 2) * height, 64);
  std::vector<Npp8u> vPlane((width / 2) * height, 192);

  NppImageMemory<Npp8u> y(width, height);
  NppImageMemory<Npp8u> u(width / 2, height);
  NppImageMemory<Npp8u> v(width / 2, height);
  NppImageMemory<Npp8u> dst(width, height, 3);

  y.copyFromHost(yPlane);
  u.copyFromHost(uPlane);
  v.copyFromHost(vPlane);

  const Npp8u *srcPlanes[3] = {y.get(), u.get(), v.get()};
  int srcSteps[3] = {y.step(), u.step(), v.step()};

  NppiSize roi = {width, height};
  NppStatus status = nppiYUV422ToRGB_8u_P3C3R(srcPlanes, srcSteps, dst.get(), dst.step(), roi);
  ASSERT_EQ(status, NPP_NO_ERROR);

  std::vector<Npp8u> dstData(width * height * 3);
  dst.copyToHost(dstData);
  std::vector<Npp8u> baseData = dstData;

  for (int i = 0; i < width * height; i++) {
    int uvIndex = (i % width) / 2 + (i / width) * (width / 2);
    Npp8u r, g, b;
    yuv_to_rgb_bt601_ref(yPlane[i], uPlane[uvIndex], vPlane[uvIndex], r, g, b);
    int idx = i * 3;
    ASSERT_EQ(dstData[idx + 0], r);
    ASSERT_EQ(dstData[idx + 1], g);
    ASSERT_EQ(dstData[idx + 2], b);
  }

  NppStreamContext ctx{};
  nppGetStreamContext(&ctx);
  ctx.hStream = 0;
  status = nppiYUV422ToRGB_8u_P3C3R_Ctx(srcPlanes, srcSteps, dst.get(), dst.step(), roi, ctx);
  ASSERT_EQ(status, NPP_NO_ERROR);

  std::vector<Npp8u> ctxData(width * height * 3);
  dst.copyToHost(ctxData);
  for (size_t i = 0; i < ctxData.size(); ++i) {
    EXPECT_EQ(ctxData[i], baseData[i]) << "Ctx mismatch at " << i;
  }
}

TEST_F(YUV422ToRGBTest, YUV422ToRGB_8u_P3R_Reference) {
  const int width = 2;
  const int height = 2;

  std::vector<Npp8u> yPlane = {16, 235, 81, 145};
  std::vector<Npp8u> uPlane((width / 2) * height, 64);
  std::vector<Npp8u> vPlane((width / 2) * height, 192);

  NppImageMemory<Npp8u> y(width, height);
  NppImageMemory<Npp8u> u(width / 2, height);
  NppImageMemory<Npp8u> v(width / 2, height);
  NppImageMemory<Npp8u> r(width, height);
  NppImageMemory<Npp8u> g(width, height);
  NppImageMemory<Npp8u> b(width, height);

  y.copyFromHost(yPlane);
  u.copyFromHost(uPlane);
  v.copyFromHost(vPlane);

  const Npp8u *srcPlanes[3] = {y.get(), u.get(), v.get()};
  int srcSteps[3] = {y.step(), u.step(), v.step()};
  Npp8u *dstPlanes[3] = {r.get(), g.get(), b.get()};

  NppiSize roi = {width, height};
  NppStatus status = nppiYUV422ToRGB_8u_P3R(srcPlanes, srcSteps, dstPlanes, r.step(), roi);
  ASSERT_EQ(status, NPP_NO_ERROR);

  std::vector<Npp8u> rPlane, gPlane, bPlane;
  r.copyToHost(rPlane);
  g.copyToHost(gPlane);
  b.copyToHost(bPlane);
  std::vector<Npp8u> baseR = rPlane;
  std::vector<Npp8u> baseG = gPlane;
  std::vector<Npp8u> baseB = bPlane;

  for (int i = 0; i < width * height; i++) {
    int uvIndex = (i % width) / 2 + (i / width) * (width / 2);
    Npp8u rr, gg, bb;
    yuv_to_rgb_bt601_ref(yPlane[i], uPlane[uvIndex], vPlane[uvIndex], rr, gg, bb);
    ASSERT_EQ(rPlane[i], rr);
    ASSERT_EQ(gPlane[i], gg);
    ASSERT_EQ(bPlane[i], bb);
  }

  NppStreamContext ctx{};
  nppGetStreamContext(&ctx);
  ctx.hStream = 0;
  status = nppiYUV422ToRGB_8u_P3R_Ctx(srcPlanes, srcSteps, dstPlanes, r.step(), roi, ctx);
  ASSERT_EQ(status, NPP_NO_ERROR);

  r.copyToHost(rPlane);
  g.copyToHost(gPlane);
  b.copyToHost(bPlane);
  for (int i = 0; i < width * height; i++) {
    EXPECT_EQ(rPlane[i], baseR[i]) << "Ctx R mismatch at " << i;
    EXPECT_EQ(gPlane[i], baseG[i]) << "Ctx G mismatch at " << i;
    EXPECT_EQ(bPlane[i], baseB[i]) << "Ctx B mismatch at " << i;
  }
}

TEST_F(YUV422ToRGBTest, YUV422ToRGB_8u_P3AC4R_AlphaCleared) {
  const int width = 2;
  const int height = 2;

  std::vector<Npp8u> yPlane = {16, 235, 81, 145};
  std::vector<Npp8u> uPlane((width / 2) * height, 64);
  std::vector<Npp8u> vPlane((width / 2) * height, 192);

  NppImageMemory<Npp8u> y(width, height);
  NppImageMemory<Npp8u> u(width / 2, height);
  NppImageMemory<Npp8u> v(width / 2, height);
  NppImageMemory<Npp8u> dst(width, height, 4);

  y.copyFromHost(yPlane);
  u.copyFromHost(uPlane);
  v.copyFromHost(vPlane);

  const Npp8u *srcPlanes[3] = {y.get(), u.get(), v.get()};
  int srcSteps[3] = {y.step(), u.step(), v.step()};

  NppiSize roi = {width, height};
  NppStatus status = nppiYUV422ToRGB_8u_P3AC4R(srcPlanes, srcSteps, dst.get(), dst.step(), roi);
  ASSERT_EQ(status, NPP_NO_ERROR);

  std::vector<Npp8u> dstData(width * height * 4);
  dst.copyToHost(dstData);
  std::vector<Npp8u> baseData = dstData;
  for (int i = 0; i < width * height; i++) {
    ASSERT_EQ(dstData[i * 4 + 3], 0);
  }

  NppStreamContext ctx{};
  nppGetStreamContext(&ctx);
  ctx.hStream = 0;
  status = nppiYUV422ToRGB_8u_P3AC4R_Ctx(srcPlanes, srcSteps, dst.get(), dst.step(), roi, ctx);
  ASSERT_EQ(status, NPP_NO_ERROR);

  std::vector<Npp8u> ctxData(width * height * 4);
  dst.copyToHost(ctxData);
  for (size_t i = 0; i < ctxData.size(); ++i) {
    EXPECT_EQ(ctxData[i], baseData[i]) << "Ctx mismatch at " << i;
  }
}

TEST_F(YUV422ToRGBTest, YUV422ToRGB_8u_C2C3R_YUYV) {
  const int width = 2;
  const int height = 2;

  NppImageMemory<Npp8u> src_mem(width, height, 2);
  NppImageMemory<Npp8u> dst(width, height, 3);

  std::vector<Npp8u> src(src_mem.step() * height, 0);
  for (int y = 0; y < height; ++y) {
    int offset = y * src_mem.step();
    // Y0 U0 Y1 V0
    src[offset + 0] = (y == 0) ? 16 : 81;
    src[offset + 1] = 64;
    src[offset + 2] = (y == 0) ? 235 : 145;
    src[offset + 3] = 192;
  }

  cudaError_t err = cudaMemcpy2D(src_mem.get(), src_mem.step(), src.data(), src_mem.step(), width * 2, height,
                                 cudaMemcpyHostToDevice);
  ASSERT_EQ(err, cudaSuccess);

  NppiSize roi = {width, height};
  NppStatus status = nppiYUV422ToRGB_8u_C2C3R(src_mem.get(), src_mem.step(), dst.get(), dst.step(), roi);
  ASSERT_EQ(status, NPP_NO_ERROR);

  std::vector<Npp8u> dstData(width * height * 3);
  dst.copyToHost(dstData);
  std::vector<Npp8u> baseData = dstData;

  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      Npp8u Y, U, V;
      yuv422_yuyv_pixel(src, src_mem.step(), x, y, Y, U, V);
      Npp8u r, g, b;
      yuv_to_rgb_bt601_ref(Y, U, V, r, g, b);
      int idx = (y * width + x) * 3;
      ASSERT_EQ(dstData[idx + 0], r);
      ASSERT_EQ(dstData[idx + 1], g);
      ASSERT_EQ(dstData[idx + 2], b);
    }
  }

  NppStreamContext ctx{};
  nppGetStreamContext(&ctx);
  ctx.hStream = 0;
  status = nppiYUV422ToRGB_8u_C2C3R_Ctx(src_mem.get(), src_mem.step(), dst.get(), dst.step(), roi, ctx);
  ASSERT_EQ(status, NPP_NO_ERROR);

  std::vector<Npp8u> ctxData(width * height * 3);
  dst.copyToHost(ctxData);
  for (size_t i = 0; i < ctxData.size(); ++i) {
    EXPECT_EQ(ctxData[i], baseData[i]) << "Ctx mismatch at " << i;
  }
}
