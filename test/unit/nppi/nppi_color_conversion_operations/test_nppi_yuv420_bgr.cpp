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
  std::vector<Npp8u> baseData = dstData;

  for (int i = 0; i < width * height; i++) {
    Npp8u r, g, b;
    yuv_to_rgb_bt601_ref(yPlane[i], 64, 192, r, g, b);
    int idx = i * 3;
    ASSERT_EQ(dstData[idx + 0], b);
    ASSERT_EQ(dstData[idx + 1], g);
    ASSERT_EQ(dstData[idx + 2], r);
  }

  NppStreamContext ctx{};
  nppGetStreamContext(&ctx);
  ctx.hStream = 0;
  status = nppiYUV420ToBGR_8u_P3C3R_Ctx(srcPlanes, srcSteps, dst.get(), dst.step(), roi, ctx);
  ASSERT_EQ(status, NPP_NO_ERROR);

  std::vector<Npp8u> ctxData(width * height * 3);
  dst.copyToHost(ctxData);
  for (size_t i = 0; i < ctxData.size(); ++i) {
    EXPECT_EQ(ctxData[i], baseData[i]) << "Ctx mismatch at " << i;
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
  std::vector<Npp8u> baseData = dstData;
  for (int i = 0; i < width * height; i++) {
    ASSERT_EQ(dstData[i * 4 + 3], 0xFF);
  }

  NppStreamContext ctx{};
  nppGetStreamContext(&ctx);
  ctx.hStream = 0;
  status = nppiYUV420ToBGR_8u_P3C4R_Ctx(srcPlanes, srcSteps, dst.get(), dst.step(), roi, ctx);
  ASSERT_EQ(status, NPP_NO_ERROR);

  std::vector<Npp8u> ctxData(width * height * 4);
  dst.copyToHost(ctxData);
  for (size_t i = 0; i < ctxData.size(); ++i) {
    EXPECT_EQ(ctxData[i], baseData[i]) << "Ctx mismatch at " << i;
  }
}

TEST_F(YUV420ToBGRTest, YUV420ToBGR_8u_P3C3R_CameraSize_640x480) {
  const int camWidth = 640;
  const int camHeight = 480;
  std::vector<Npp8u> yPlane(static_cast<size_t>(camWidth) * camHeight);
  std::vector<Npp8u> uPlane(static_cast<size_t>(camWidth / 2) * (camHeight / 2), 128);
  std::vector<Npp8u> vPlane(static_cast<size_t>(camWidth / 2) * (camHeight / 2), 128);
  for (size_t i = 0; i < yPlane.size(); ++i) {
    yPlane[i] = static_cast<Npp8u>((i * 31 + 7) % 256);
  }

  NppImageMemory<Npp8u> y(camWidth, camHeight);
  NppImageMemory<Npp8u> u(camWidth / 2, camHeight / 2);
  NppImageMemory<Npp8u> v(camWidth / 2, camHeight / 2);
  NppImageMemory<Npp8u> dst(camWidth, camHeight, 3);
  y.copyFromHost(yPlane);
  u.copyFromHost(uPlane);
  v.copyFromHost(vPlane);

  const Npp8u *srcPlanes[3] = {y.get(), u.get(), v.get()};
  int srcSteps[3] = {y.step(), u.step(), v.step()};
  const NppiSize roi{camWidth, camHeight};
  ASSERT_EQ(nppiYUV420ToBGR_8u_P3C3R(srcPlanes, srcSteps, dst.get(), dst.step(), roi), NPP_NO_ERROR);

  std::vector<Npp8u> dstData(static_cast<size_t>(camWidth) * camHeight * 3);
  dst.copyToHost(dstData);
  for (int i = 0; i < camWidth * camHeight; ++i) {
    Npp8u r, g, b;
    yuv_to_rgb_bt601_ref(yPlane[i], 128, 128, r, g, b);
    const int idx = i * 3;
    EXPECT_EQ(dstData[idx + 0], b) << "B mismatch at " << i;
    EXPECT_EQ(dstData[idx + 1], g) << "G mismatch at " << i;
    EXPECT_EQ(dstData[idx + 2], r) << "R mismatch at " << i;
  }
}

TEST_F(YUV420ToBGRTest, YUV420ToBGR_8u_P3C3R_PartialEvenROI) {
  const int width = 128;
  const int height = 128;
  const int roiX = 32;
  const int roiY = 16;
  const int roiW = 64;
  const int roiH = 48;

  std::vector<Npp8u> yPlane(static_cast<size_t>(width) * height);
  std::vector<Npp8u> uPlane(static_cast<size_t>(width / 2) * (height / 2));
  std::vector<Npp8u> vPlane(static_cast<size_t>(width / 2) * (height / 2));
  for (size_t i = 0; i < yPlane.size(); ++i) {
    yPlane[i] = static_cast<Npp8u>((i * 23 + 5) % 256);
  }
  for (size_t i = 0; i < uPlane.size(); ++i) {
    uPlane[i] = static_cast<Npp8u>(64 + (i * 7) % 127);
    vPlane[i] = static_cast<Npp8u>(64 + (i * 11) % 127);
  }

  NppImageMemory<Npp8u> y(width, height);
  NppImageMemory<Npp8u> u(width / 2, height / 2);
  NppImageMemory<Npp8u> v(width / 2, height / 2);
  NppImageMemory<Npp8u> fullDst(width, height, 3);
  NppImageMemory<Npp8u> roiDst(roiW, roiH, 3);
  y.copyFromHost(yPlane);
  u.copyFromHost(uPlane);
  v.copyFromHost(vPlane);

  // Reference: full-image conversion, then take the same ROI window.
  const Npp8u *fullPlanes[3] = {y.get(), u.get(), v.get()};
  int fullSteps[3] = {y.step(), u.step(), v.step()};
  const NppiSize fullRoi{width, height};
  ASSERT_EQ(nppiYUV420ToBGR_8u_P3C3R(fullPlanes, fullSteps, fullDst.get(), fullDst.step(), fullRoi), NPP_NO_ERROR);

  // Chroma pointer offsets are half the luma offsets (2x2 subsampling).
  Npp8u *dRoiY = y.get() + static_cast<size_t>(roiY) * y.step() + roiX;
  Npp8u *dRoiU = u.get() + static_cast<size_t>(roiY / 2) * u.step() + roiX / 2;
  Npp8u *dRoiV = v.get() + static_cast<size_t>(roiY / 2) * v.step() + roiX / 2;
  const Npp8u *roiPlanes[3] = {dRoiY, dRoiU, dRoiV};
  const NppiSize roi{roiW, roiH};
  ASSERT_EQ(nppiYUV420ToBGR_8u_P3C3R(roiPlanes, fullSteps, roiDst.get(), roiDst.step(), roi), NPP_NO_ERROR);

  std::vector<Npp8u> fullData(static_cast<size_t>(width) * height * 3);
  std::vector<Npp8u> roiData(static_cast<size_t>(roiW) * roiH * 3);
  fullDst.copyToHost(fullData);
  roiDst.copyToHost(roiData);
  for (int i = 0; i < roiW * roiH; ++i) {
    const int fullRow = roiY + i / roiW;
    const int fullCol = roiX + i % roiW;
    for (int c = 0; c < 3; ++c) {
      EXPECT_EQ(roiData[i * 3 + c], fullData[(static_cast<size_t>(fullRow) * width + fullCol) * 3 + c])
          << "ROI mismatch at pixel " << i << " channel " << c;
    }
  }
}
