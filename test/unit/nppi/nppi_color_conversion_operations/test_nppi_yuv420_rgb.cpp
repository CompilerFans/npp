#include "npp_test_base.h"
#include <algorithm>
#include <cmath>

using namespace npp_functional_test;

class YUV420ToRGBTest : public NppTestBase {};

TEST_F(YUV420ToRGBTest, YUV420ToRGB_8u_P3C3R_Gray) {
  const int width = 32;
  const int height = 32;

  std::vector<Npp8u> yPlane(width * height, 80);
  std::vector<Npp8u> uPlane((width / 2) * (height / 2), 128);
  std::vector<Npp8u> vPlane((width / 2) * (height / 2), 128);

  NppImageMemory<Npp8u> y(width, height);
  NppImageMemory<Npp8u> u(width / 2, height / 2);
  NppImageMemory<Npp8u> v(width / 2, height / 2);
  NppImageMemory<Npp8u> dst(width, height, 3);

  y.copyFromHost(yPlane);
  u.copyFromHost(uPlane);
  v.copyFromHost(vPlane);

  const Npp8u *srcPlanes[3] = {y.get(), u.get(), v.get()};
  int srcSteps[3] = {y.step(), u.step(), v.step()};

  NppiSize roi = {width, height};
  NppStatus status = nppiYUV420ToRGB_8u_P3C3R(srcPlanes, srcSteps, dst.get(), dst.step(), roi);
  ASSERT_EQ(status, NPP_NO_ERROR);

  std::vector<Npp8u> dstData(width * height * 3);
  dst.copyToHost(dstData);

  bool hasPixel = false;
  for (int i = 0; i < width * height; i++) {
    int idx = i * 3;
    if (dstData[idx] || dstData[idx + 1] || dstData[idx + 2]) {
      hasPixel = true;
      ASSERT_NEAR(dstData[idx], dstData[idx + 1], 1);
      ASSERT_NEAR(dstData[idx], dstData[idx + 2], 1);
      break;
    }
  }
  ASSERT_TRUE(hasPixel);
}

TEST_F(YUV420ToRGBTest, YUV420ToRGB_8u_P3R_Gray) {
  const int width = 16;
  const int height = 16;

  std::vector<Npp8u> yPlane(width * height, 90);
  std::vector<Npp8u> uPlane((width / 2) * (height / 2), 128);
  std::vector<Npp8u> vPlane((width / 2) * (height / 2), 128);

  NppImageMemory<Npp8u> y(width, height);
  NppImageMemory<Npp8u> u(width / 2, height / 2);
  NppImageMemory<Npp8u> v(width / 2, height / 2);
  NppImageMemory<Npp8u> dstR(width, height);
  NppImageMemory<Npp8u> dstG(width, height);
  NppImageMemory<Npp8u> dstB(width, height);

  y.copyFromHost(yPlane);
  u.copyFromHost(uPlane);
  v.copyFromHost(vPlane);

  const Npp8u *srcPlanes[3] = {y.get(), u.get(), v.get()};
  int srcSteps[3] = {y.step(), u.step(), v.step()};
  Npp8u *dstPlanes[3] = {dstR.get(), dstG.get(), dstB.get()};

  NppiSize roi = {width, height};
  NppStatus status = nppiYUV420ToRGB_8u_P3R(srcPlanes, srcSteps, dstPlanes, dstR.step(), roi);
  ASSERT_EQ(status, NPP_NO_ERROR);

  std::vector<Npp8u> r(width * height);
  std::vector<Npp8u> g(width * height);
  std::vector<Npp8u> b(width * height);
  dstR.copyToHost(r);
  dstG.copyToHost(g);
  dstB.copyToHost(b);

  for (int i = 0; i < width * height; i++) {
    ASSERT_NEAR(r[i], g[i], 1);
    ASSERT_NEAR(r[i], b[i], 1);
  }
}

TEST_F(YUV420ToRGBTest, YUV420ToRGB_8u_P3C4R_Alpha) {
  const int width = 16;
  const int height = 16;

  std::vector<Npp8u> yPlane(width * height, 100);
  std::vector<Npp8u> uPlane((width / 2) * (height / 2), 128);
  std::vector<Npp8u> vPlane((width / 2) * (height / 2), 128);

  NppImageMemory<Npp8u> y(width, height);
  NppImageMemory<Npp8u> u(width / 2, height / 2);
  NppImageMemory<Npp8u> v(width / 2, height / 2);
  NppImageMemory<Npp8u> dst(width, height, 4);

  y.copyFromHost(yPlane);
  u.copyFromHost(uPlane);
  v.copyFromHost(vPlane);

  const Npp8u *srcPlanes[3] = {y.get(), u.get(), v.get()};
  int srcSteps[3] = {y.step(), u.step(), v.step()};

  NppiSize roi = {width, height};
  NppStatus status = nppiYUV420ToRGB_8u_P3C4R(srcPlanes, srcSteps, dst.get(), dst.step(), roi);
  ASSERT_EQ(status, NPP_NO_ERROR);

  std::vector<Npp8u> dstData(width * height * 4);
  dst.copyToHost(dstData);
#ifndef USE_NVIDIA_NPP_TESTS
  for (int i = 0; i < width * height; i++) {
    ASSERT_EQ(dstData[i * 4 + 3], 0xFF);
  }
#endif
}

TEST_F(YUV420ToRGBTest, YUV420ToRGB_8u_P3AC4R_AlphaPreserve) {
  const int width = 16;
  const int height = 16;

  std::vector<Npp8u> yPlane(width * height, 110);
  std::vector<Npp8u> uPlane((width / 2) * (height / 2), 128);
  std::vector<Npp8u> vPlane((width / 2) * (height / 2), 128);
  std::vector<Npp8u> dstInit(width * height * 4, 0);
  for (int i = 0; i < width * height; i++) {
    dstInit[i * 4 + 3] = 37;
  }

  NppImageMemory<Npp8u> y(width, height);
  NppImageMemory<Npp8u> u(width / 2, height / 2);
  NppImageMemory<Npp8u> v(width / 2, height / 2);
  NppImageMemory<Npp8u> dst(width, height, 4);

  y.copyFromHost(yPlane);
  u.copyFromHost(uPlane);
  v.copyFromHost(vPlane);
  dst.copyFromHost(dstInit);

  const Npp8u *srcPlanes[3] = {y.get(), u.get(), v.get()};
  int srcSteps[3] = {y.step(), u.step(), v.step()};

  NppiSize roi = {width, height};
  NppStatus status = nppiYUV420ToRGB_8u_P3AC4R(srcPlanes, srcSteps, dst.get(), dst.step(), roi);
  ASSERT_EQ(status, NPP_NO_ERROR);

  std::vector<Npp8u> dstData(width * height * 4);
  dst.copyToHost(dstData);
#ifndef USE_NVIDIA_NPP_TESTS
  for (int i = 0; i < width * height; i++) {
    ASSERT_EQ(dstData[i * 4 + 3], 37);
  }
#endif
}
