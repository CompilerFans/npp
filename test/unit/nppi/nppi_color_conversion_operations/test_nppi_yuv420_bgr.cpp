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
} // namespace

class YUV420ToBGRTest : public NppTestBase {};

TEST_F(YUV420ToBGRTest, YUV420ToBGR_8u_P3C3R_Reference) {
  const int width = 2;
  const int height = 2;

  std::vector<Npp8u> yPlane = {16, 235, 81, 145};
  std::vector<Npp8u> uPlane((width / 2) * (height / 2), 64);
  std::vector<Npp8u> vPlane((width / 2) * (height / 2), 192);

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
  NppStatus status = nppiYUV420ToBGR_8u_P3C3R(srcPlanes, srcSteps, dst.get(), dst.step(), roi);
  ASSERT_EQ(status, NPP_NO_ERROR);

  std::vector<Npp8u> dstData(width * height * 3);
  dst.copyToHost(dstData);

  for (int i = 0; i < width * height; i++) {
    Npp8u r, g, b;
    yuv_to_rgb_bt601_ref(yPlane[i], 64, 192, r, g, b);
    int idx = i * 3;
    ASSERT_EQ(dstData[idx + 0], b);
    ASSERT_EQ(dstData[idx + 1], g);
    ASSERT_EQ(dstData[idx + 2], r);
  }
}

TEST_F(YUV420ToBGRTest, YUV420ToBGR_8u_P3C4R_AlphaConstant) {
  const int width = 4;
  const int height = 2;

  std::vector<Npp8u> yPlane(width * height);
  for (int i = 0; i < width * height; ++i) {
    yPlane[i] = static_cast<Npp8u>((i * 17) & 0xFF);
  }
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
  NppStatus status = nppiYUV420ToBGR_8u_P3C4R(srcPlanes, srcSteps, dst.get(), dst.step(), roi);
  ASSERT_EQ(status, NPP_NO_ERROR);

  std::vector<Npp8u> dstData(width * height * 4);
  dst.copyToHost(dstData);
  for (int i = 0; i < width * height; i++) {
    ASSERT_EQ(dstData[i * 4 + 3], 0xFF);
  }
}
