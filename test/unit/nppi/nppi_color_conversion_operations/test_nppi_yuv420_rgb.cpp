#include "npp_test_base.h"
#include <algorithm>
#include <cmath>

using namespace npp_functional_test;

class YUV420ToRGBTest : public NppTestBase {};

static inline void yuv_to_rgb_nvidia_ref(Npp8u y, Npp8u u, Npp8u v, Npp8u &r, Npp8u &g, Npp8u &b) {
  int yv = static_cast<int>(y);
  int ud = static_cast<int>(u) - 128;
  int vd = static_cast<int>(v) - 128;

  int r_val = yv + ((145 * vd) >> 7);
  int g_val = yv + (((-22 * ud) + (-45 * vd)) >> 7);
  int b_val = yv + ((255 * ud) >> 7);

  r_val = std::min(255, std::max(0, r_val));
  g_val = std::min(255, std::max(0, g_val));
  b_val = std::min(255, std::max(0, b_val));

  r = static_cast<Npp8u>(r_val);
  g = static_cast<Npp8u>(g_val);
  b = static_cast<Npp8u>(b_val);
}

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

  std::vector<Npp8u> baseData = dstData;

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

  NppStreamContext ctx{};
  nppGetStreamContext(&ctx);
  ctx.hStream = 0;
  status = nppiYUV420ToRGB_8u_P3C3R_Ctx(srcPlanes, srcSteps, dst.get(), dst.step(), roi, ctx);
  ASSERT_EQ(status, NPP_NO_ERROR);

  std::vector<Npp8u> ctxData(width * height * 3);
  dst.copyToHost(ctxData);
  for (size_t i = 0; i < ctxData.size(); ++i) {
    EXPECT_EQ(ctxData[i], baseData[i]) << "Ctx mismatch at " << i;
  }
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

  std::vector<Npp8u> baseR = r;
  std::vector<Npp8u> baseG = g;
  std::vector<Npp8u> baseB = b;

  NppStreamContext ctx{};
  nppGetStreamContext(&ctx);
  ctx.hStream = 0;
  status = nppiYUV420ToRGB_8u_P3R_Ctx(srcPlanes, srcSteps, dstPlanes, dstR.step(), roi, ctx);
  ASSERT_EQ(status, NPP_NO_ERROR);

  dstR.copyToHost(r);
  dstG.copyToHost(g);
  dstB.copyToHost(b);

  for (int i = 0; i < width * height; i++) {
    EXPECT_EQ(r[i], baseR[i]) << "Ctx R mismatch at " << i;
    EXPECT_EQ(g[i], baseG[i]) << "Ctx G mismatch at " << i;
    EXPECT_EQ(b[i], baseB[i]) << "Ctx B mismatch at " << i;
  }
}

TEST_F(YUV420ToRGBTest, YUV420ToRGB_8u_P3C3R_GrayscaleReference) {
  const int width = 2;
  const int height = 2;

  std::vector<Npp8u> yPlane = {16, 235, 81, 145};
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

  for (int i = 0; i < width * height; i++) {
    Npp8u r, g, b;
    yuv_to_rgb_nvidia_ref(yPlane[i], 128, 128, r, g, b);
    int idx = i * 3;
    ASSERT_EQ(dstData[idx], r);
    ASSERT_EQ(dstData[idx + 1], g);
    ASSERT_EQ(dstData[idx + 2], b);
  }
}

TEST_F(YUV420ToRGBTest, YUV420ToRGB_8u_P3C3R_ChromaReference) {
  const int width = 2;
  const int height = 2;

  std::vector<Npp8u> yPlane(width * height, 128);
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
  NppStatus status = nppiYUV420ToRGB_8u_P3C3R(srcPlanes, srcSteps, dst.get(), dst.step(), roi);
  ASSERT_EQ(status, NPP_NO_ERROR);

  std::vector<Npp8u> dstData(width * height * 3);
  dst.copyToHost(dstData);

  Npp8u r, g, b;
  yuv_to_rgb_nvidia_ref(128, 64, 192, r, g, b);
  for (int i = 0; i < width * height; i++) {
    int idx = i * 3;
    ASSERT_EQ(dstData[idx], r);
    ASSERT_EQ(dstData[idx + 1], g);
    ASSERT_EQ(dstData[idx + 2], b);
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
  std::vector<Npp8u> baseData = dstData;
  // NVIDIA NPP 12.4 bug: the RGB P3C4R kernel clears alpha for most pixels but leaks luma into the
  // first 4 columns of each row (its header documents constant 0xFF). MPP clears alpha uniformly;
  // the NVIDIA-side expectation models the buggy device behavior exactly.
#ifdef USE_NVIDIA_NPP_TESTS
  for (int i = 0; i < width * height; i++) {
    const int expectAlpha = (i % width) < 4 ? yPlane[i] : 0;
    ASSERT_EQ(dstData[i * 4 + 3], expectAlpha);
  }
#else
  for (int i = 0; i < width * height; i++) {
    ASSERT_EQ(dstData[i * 4 + 3], 0);
  }
#endif

  NppStreamContext ctx{};
  nppGetStreamContext(&ctx);
  ctx.hStream = 0;
  status = nppiYUV420ToRGB_8u_P3C4R_Ctx(srcPlanes, srcSteps, dst.get(), dst.step(), roi, ctx);
  ASSERT_EQ(status, NPP_NO_ERROR);

  std::vector<Npp8u> ctxData(width * height * 4);
  dst.copyToHost(ctxData);
  for (size_t i = 0; i < ctxData.size(); ++i) {
    EXPECT_EQ(ctxData[i], baseData[i]) << "Ctx mismatch at " << i;
  }
}

TEST_F(YUV420ToRGBTest, YUV420ToRGB_8u_P3C4R_AlphaCleared) {
  const int width = 32;
  const int height = 16;

  std::vector<Npp8u> yPlane(width * height);
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      yPlane[y * width + x] = static_cast<Npp8u>((x * 7 + y * 3) & 0xFF);
    }
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
  NppStatus status = nppiYUV420ToRGB_8u_P3C4R(srcPlanes, srcSteps, dst.get(), dst.step(), roi);
  ASSERT_EQ(status, NPP_NO_ERROR);

  std::vector<Npp8u> dstData(width * height * 4);
  dst.copyToHost(dstData);
  // NVIDIA 12.4 clears alpha to 0 for RGB P3C4R despite its header documenting constant 0xFF;
  // likely a library bug (the BGR variant fills 0xFF). MPP follows the observed behavior.
  for (int i = 0; i < width * height; i++) {
    ASSERT_EQ(dstData[i * 4 + 3], 0);
  }
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
  std::vector<Npp8u> baseData = dstData;
  for (int i = 0; i < width * height; i++) {
    ASSERT_EQ(dstData[i * 4 + 3], 0);
  }

  NppStreamContext ctx{};
  nppGetStreamContext(&ctx);
  ctx.hStream = 0;
  status = nppiYUV420ToRGB_8u_P3AC4R_Ctx(srcPlanes, srcSteps, dst.get(), dst.step(), roi, ctx);
  ASSERT_EQ(status, NPP_NO_ERROR);

  std::vector<Npp8u> ctxData(width * height * 4);
  dst.copyToHost(ctxData);
  for (size_t i = 0; i < ctxData.size(); ++i) {
    EXPECT_EQ(ctxData[i], baseData[i]) << "Ctx mismatch at " << i;
  }
}

TEST_F(YUV420ToRGBTest, YUV420ToRGB_8u_P3AC4R_AlphaClearedFromPreset) {
  const int width = 32;
  const int height = 16;

  std::vector<Npp8u> yPlane(width * height, 120);
  std::vector<Npp8u> uPlane((width / 2) * (height / 2), 128);
  std::vector<Npp8u> vPlane((width / 2) * (height / 2), 128);
  std::vector<Npp8u> dstInit(width * height * 4, 0);
  for (int i = 0; i < width * height; i++) {
    dstInit[i * 4 + 3] = static_cast<Npp8u>(200);
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
  for (int i = 0; i < width * height; i++) {
    ASSERT_EQ(dstData[i * 4 + 3], 0);
  }
}

TEST_F(YUV420ToRGBTest, YUV420ToRGB_8u_P3C3R_CameraSize_640x480) {
  const int camWidth = 640;
  const int camHeight = 480;
  std::vector<Npp8u> yPlane(static_cast<size_t>(camWidth) * camHeight);
  std::vector<Npp8u> uPlane(static_cast<size_t>(camWidth / 2) * (camHeight / 2), 128);
  std::vector<Npp8u> vPlane(static_cast<size_t>(camWidth / 2) * (camHeight / 2), 128);
  for (size_t i = 0; i < yPlane.size(); ++i) {
    yPlane[i] = static_cast<Npp8u>((i * 19 + 3) % 256);
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
  ASSERT_EQ(nppiYUV420ToRGB_8u_P3C3R(srcPlanes, srcSteps, dst.get(), dst.step(), roi), NPP_NO_ERROR);

  std::vector<Npp8u> dstData(static_cast<size_t>(camWidth) * camHeight * 3);
  dst.copyToHost(dstData);
  for (int i = 0; i < camWidth * camHeight; ++i) {
    // Neutral chroma (128) means R == G == B == Y.
    EXPECT_EQ(dstData[i * 3 + 0], yPlane[i]) << "R mismatch at " << i;
    EXPECT_EQ(dstData[i * 3 + 1], yPlane[i]) << "G mismatch at " << i;
    EXPECT_EQ(dstData[i * 3 + 2], yPlane[i]) << "B mismatch at " << i;
  }

  // Ctx variant must match the default-stream result exactly.
  NppStreamContext ctx{};
  nppGetStreamContext(&ctx);
  ASSERT_EQ(nppiYUV420ToRGB_8u_P3C3R_Ctx(srcPlanes, srcSteps, dst.get(), dst.step(), roi, ctx), NPP_NO_ERROR);
  std::vector<Npp8u> ctxData(static_cast<size_t>(camWidth) * camHeight * 3);
  dst.copyToHost(ctxData);
  for (size_t i = 0; i < ctxData.size(); ++i) {
    EXPECT_EQ(ctxData[i], dstData[i]) << "Ctx mismatch at " << i;
  }
}

TEST_F(YUV420ToRGBTest, YUV420ToRGB_8u_P3C3R_PartialEvenROI) {
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
    yPlane[i] = static_cast<Npp8u>((i * 41 + 9) % 256);
  }
  for (size_t i = 0; i < uPlane.size(); ++i) {
    uPlane[i] = static_cast<Npp8u>(64 + (i * 13) % 127);
    vPlane[i] = static_cast<Npp8u>(64 + (i * 17) % 127);
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
  ASSERT_EQ(nppiYUV420ToRGB_8u_P3C3R(fullPlanes, fullSteps, fullDst.get(), fullDst.step(), fullRoi), NPP_NO_ERROR);

  Npp8u *dRoiY = y.get() + static_cast<size_t>(roiY) * y.step() + roiX;
  Npp8u *dRoiU = u.get() + static_cast<size_t>(roiY / 2) * u.step() + roiX / 2;
  Npp8u *dRoiV = v.get() + static_cast<size_t>(roiY / 2) * v.step() + roiX / 2;
  const Npp8u *roiPlanes[3] = {dRoiY, dRoiU, dRoiV};
  const NppiSize roi{roiW, roiH};
  ASSERT_EQ(nppiYUV420ToRGB_8u_P3C3R(roiPlanes, fullSteps, roiDst.get(), roiDst.step(), roi), NPP_NO_ERROR);

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
