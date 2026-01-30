#include "npp_test_base.h"
#include <cmath>
#include <vector>

using namespace npp_functional_test;

namespace {
struct RgbPixel {
  Npp8u r;
  Npp8u g;
  Npp8u b;
};

static inline Npp8u clamp_u8(float v) {
  if (v < 0.0f) {
    v = 0.0f;
  } else if (v > 255.0f) {
    v = 255.0f;
  }
  return static_cast<Npp8u>(v);
}

static inline void bgr_to_lab_ref(Npp8u b, Npp8u g, Npp8u r, Npp8u &l, Npp8u &a, Npp8u &bb) {
  const float nCIE_LAB_D65_xn = 0.950455f;
  const float nCIE_LAB_D65_zn = 1.088753f;

  float nNormalizedR = static_cast<float>(r) * 0.003921569f;
  float nNormalizedG = static_cast<float>(g) * 0.003921569f;
  float nNormalizedB = static_cast<float>(b) * 0.003921569f;

  float nX = 0.412453f * nNormalizedR + 0.35758f * nNormalizedG + 0.180423f * nNormalizedB;
  float nY = 0.212671f * nNormalizedR + 0.71516f * nNormalizedG + 0.072169f * nNormalizedB;
  float nZ = 0.019334f * nNormalizedR + 0.119193f * nNormalizedG + 0.950227f * nNormalizedB;

  float nL = cbrtf(nY);
  float nfX = nX * (1.0f / nCIE_LAB_D65_xn);
  float nfY = nY;
  float nfZ = nZ * (1.0f / nCIE_LAB_D65_zn);

  nfY = nL - 16.0f;
  nL = 116.0f * nL - 16.0f;
  float nA = cbrtf(nfX) - 16.0f;
  nA = 500.0f * (nA - nfY);
  float nB = cbrtf(nfZ) - 16.0f;
  nB = 200.0f * (nfY - nB);

  nL = nL * 255.0f * 0.01f;
  nA = nA + 128.0f;
  nB = nB + 128.0f;

  l = clamp_u8(nL);
  a = clamp_u8(nA);
  bb = clamp_u8(nB);
}

static inline void lab_to_bgr_ref(Npp8u l, Npp8u a, Npp8u bb, Npp8u &b, Npp8u &g, Npp8u &r) {
  const float nCIE_LAB_D65_xn = 0.950455f;
  const float nCIE_LAB_D65_zn = 1.088753f;

  float nL = static_cast<float>(l) * 100.0f * 0.003921569f;
  float nA = static_cast<float>(a) - 128.0f;
  float nB = static_cast<float>(bb) - 128.0f;

  float nP = (nL + 16.0f) * 0.008621f;
  float nNormalizedY = nP * nP * nP;
  float nNormalizedX = nCIE_LAB_D65_xn * powf((nP + nA * 0.002f), 3.0f);
  float nNormalizedZ = nCIE_LAB_D65_zn * powf((nP - nB * 0.005f), 3.0f);

  float nR = 3.240479f * nNormalizedX - 1.53715f * nNormalizedY - 0.498535f * nNormalizedZ;
  float nG = -0.969256f * nNormalizedX + 1.875991f * nNormalizedY + 0.041556f * nNormalizedZ;
  float nBB = 0.055648f * nNormalizedX - 0.204043f * nNormalizedY + 1.057311f * nNormalizedZ;

  if (nR > 1.0f) {
    nR = 1.0f;
  }
  if (nG > 1.0f) {
    nG = 1.0f;
  }
  if (nBB > 1.0f) {
    nBB = 1.0f;
  }
  if (nR < 0.0f) {
    nR = 0.0f;
  }
  if (nG < 0.0f) {
    nG = 0.0f;
  }
  if (nBB < 0.0f) {
    nBB = 0.0f;
  }

  r = clamp_u8(nR * 255.0f);
  g = clamp_u8(nG * 255.0f);
  b = clamp_u8(nBB * 255.0f);
}
} // namespace

class LabConversionTest : public NppTestBase {};

TEST_F(LabConversionTest, BGRToLab_8u_C3R_Accuracy) {
  const int width = 4;
  const int height = 1;
  std::vector<RgbPixel> pixels = {
      {255, 0, 0},
      {0, 255, 0},
      {0, 0, 255},
      {255, 255, 255},
  };

  std::vector<Npp8u> src(width * height * 3);
  for (int i = 0; i < width * height; ++i) {
    src[i * 3 + 0] = pixels[i].b;
    src[i * 3 + 1] = pixels[i].g;
    src[i * 3 + 2] = pixels[i].r;
  }

  NppImageMemory<Npp8u> src_mem(width, height, 3);
  NppImageMemory<Npp8u> dst_mem(width, height, 3);
  src_mem.copyFromHost(src);

  NppiSize roi = {width, height};
  NppStatus status = nppiBGRToLab_8u_C3R(src_mem.get(), src_mem.step(), dst_mem.get(), dst_mem.step(), roi);
  ASSERT_EQ(status, NPP_NO_ERROR);

  std::vector<Npp8u> dst;
  dst_mem.copyToHost(dst);
  for (int i = 0; i < width; ++i) {
    Npp8u l, a, bb;
    bgr_to_lab_ref(pixels[i].b, pixels[i].g, pixels[i].r, l, a, bb);
    EXPECT_NEAR(dst[i * 3 + 0], l, 2);
    EXPECT_NEAR(dst[i * 3 + 1], a, 2);
    EXPECT_NEAR(dst[i * 3 + 2], bb, 2);
  }
}

TEST_F(LabConversionTest, LabToBGR_8u_C3R_RoundTrip) {
  const int width = 4;
  const int height = 1;
  std::vector<RgbPixel> pixels = {
      {10, 30, 200},
      {120, 200, 40},
      {200, 50, 50},
      {0, 0, 0},
  };

  std::vector<Npp8u> src(width * height * 3);
  for (int i = 0; i < width * height; ++i) {
    Npp8u l, a, bb;
    bgr_to_lab_ref(pixels[i].b, pixels[i].g, pixels[i].r, l, a, bb);
    src[i * 3 + 0] = l;
    src[i * 3 + 1] = a;
    src[i * 3 + 2] = bb;
  }

  NppImageMemory<Npp8u> src_mem(width, height, 3);
  NppImageMemory<Npp8u> dst_mem(width, height, 3);
  src_mem.copyFromHost(src);

  NppiSize roi = {width, height};
  NppStatus status = nppiLabToBGR_8u_C3R(src_mem.get(), src_mem.step(), dst_mem.get(), dst_mem.step(), roi);
  ASSERT_EQ(status, NPP_NO_ERROR);

  std::vector<Npp8u> dst;
  dst_mem.copyToHost(dst);
  for (int i = 0; i < width; ++i) {
    EXPECT_NEAR(dst[i * 3 + 0], pixels[i].b, 3);
    EXPECT_NEAR(dst[i * 3 + 1], pixels[i].g, 3);
    EXPECT_NEAR(dst[i * 3 + 2], pixels[i].r, 3);
  }
}
