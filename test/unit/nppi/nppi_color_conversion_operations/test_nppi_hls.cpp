#include "npp_test_base.h"
#include <algorithm>
#include <cmath>
#include <vector>

using namespace npp_functional_test;

namespace {
struct RgbPixel {
  Npp8u r;
  Npp8u g;
  Npp8u b;
};

static inline Npp8u clamp_u8_round(float v) {
  if (v < 0.0f) {
    return 0;
  }
  if (v > 255.0f) {
    return 255;
  }
  return static_cast<Npp8u>(std::lround(v));
}

static inline void rgb_to_hls_ref(Npp8u r, Npp8u g, Npp8u b, Npp8u &h, Npp8u &l, Npp8u &s) {
  float rf = r / 255.0f;
  float gf = g / 255.0f;
  float bf = b / 255.0f;

  float maxv = std::max(rf, std::max(gf, bf));
  float minv = std::min(rf, std::min(gf, bf));
  float lval = 0.5f * (maxv + minv);

  float hval = 0.0f;
  float sval = 0.0f;
  float delta = maxv - minv;

  if (delta > 0.0f) {
    if (lval < 0.5f) {
      sval = delta / (maxv + minv);
    } else {
      sval = delta / (2.0f - maxv - minv);
    }

    if (maxv == rf) {
      hval = (gf - bf) / delta;
      if (gf < bf) {
        hval += 6.0f;
      }
    } else if (maxv == gf) {
      hval = 2.0f + (bf - rf) / delta;
    } else {
      hval = 4.0f + (rf - gf) / delta;
    }
    hval /= 6.0f;
  }

  h = clamp_u8_round(hval * 255.0f);
  l = clamp_u8_round(lval * 255.0f);
  s = clamp_u8_round(sval * 255.0f);
}

static inline float hue_to_rgb(float p, float q, float t) {
  if (t < 0.0f) {
    t += 1.0f;
  }
  if (t > 1.0f) {
    t -= 1.0f;
  }
  if (t < 1.0f / 6.0f) {
    return p + (q - p) * 6.0f * t;
  }
  if (t < 0.5f) {
    return q;
  }
  if (t < 2.0f / 3.0f) {
    return p + (q - p) * (2.0f / 3.0f - t) * 6.0f;
  }
  return p;
}

static inline void hls_to_rgb_ref(Npp8u h, Npp8u l, Npp8u s, Npp8u &r, Npp8u &g, Npp8u &b) {
  float hf = h / 255.0f;
  float lf = l / 255.0f;
  float sf = s / 255.0f;

  float rf, gf, bf;
  if (sf <= 0.0f) {
    rf = gf = bf = lf;
  } else {
    float q = (lf < 0.5f) ? (lf * (1.0f + sf)) : (lf + sf - lf * sf);
    float p = 2.0f * lf - q;
    rf = hue_to_rgb(p, q, hf + 1.0f / 3.0f);
    gf = hue_to_rgb(p, q, hf);
    bf = hue_to_rgb(p, q, hf - 1.0f / 3.0f);
  }

  r = clamp_u8_round(rf * 255.0f);
  g = clamp_u8_round(gf * 255.0f);
  b = clamp_u8_round(bf * 255.0f);
}

static void build_pixels(std::vector<Npp8u> &packed, const std::vector<RgbPixel> &pixels, bool bgr) {
  packed.resize(pixels.size() * 3);
  for (size_t i = 0; i < pixels.size(); ++i) {
    if (bgr) {
      packed[i * 3 + 0] = pixels[i].b;
      packed[i * 3 + 1] = pixels[i].g;
      packed[i * 3 + 2] = pixels[i].r;
    } else {
      packed[i * 3 + 0] = pixels[i].r;
      packed[i * 3 + 1] = pixels[i].g;
      packed[i * 3 + 2] = pixels[i].b;
    }
  }
}


static void build_planar_bgr(const std::vector<RgbPixel> &pixels, std::vector<Npp8u> &b_plane,
                             std::vector<Npp8u> &g_plane, std::vector<Npp8u> &r_plane) {
  b_plane.resize(pixels.size());
  g_plane.resize(pixels.size());
  r_plane.resize(pixels.size());
  for (size_t i = 0; i < pixels.size(); ++i) {
    b_plane[i] = pixels[i].b;
    g_plane[i] = pixels[i].g;
    r_plane[i] = pixels[i].r;
  }
}
} // namespace

class HLSConversionTest : public NppTestBase {};

TEST_F(HLSConversionTest, RGBToHLS_8u_C3R_BasicGray) {
  const int width = 4;
  const int height = 1;
  std::vector<RgbPixel> pixels = {
      {0, 0, 0},
      {64, 64, 64},
      {128, 128, 128},
      {255, 255, 255},
  };

  std::vector<Npp8u> src;
  build_pixels(src, pixels, false);

  NppImageMemory<Npp8u> src_mem(width, height, 3);
  NppImageMemory<Npp8u> dst_mem(width, height, 3);
  src_mem.copyFromHost(src);

  NppiSize roi = {width, height};
  NppStatus status = nppiRGBToHLS_8u_C3R(src_mem.get(), src_mem.step(), dst_mem.get(), dst_mem.step(), roi);
  ASSERT_EQ(status, NPP_NO_ERROR);

  std::vector<Npp8u> dst;
  dst_mem.copyToHost(dst);

  for (int i = 0; i < width; ++i) {
    Npp8u h, l, s;
    rgb_to_hls_ref(pixels[i].r, pixels[i].g, pixels[i].b, h, l, s);
    EXPECT_NEAR(dst[i * 3 + 0], h, 1);
    EXPECT_NEAR(dst[i * 3 + 1], l, 1);
    EXPECT_NEAR(dst[i * 3 + 2], s, 1);
  }
}

TEST_F(HLSConversionTest, RGBToHLS_8u_C3R_PrimaryColorsAccuracy) {
  const int width = 4;
  const int height = 1;
  std::vector<RgbPixel> pixels = {
      {255, 0, 0},
      {0, 255, 0},
      {0, 0, 255},
      {255, 255, 255},
  };

  std::vector<Npp8u> src;
  build_pixels(src, pixels, false);

  NppImageMemory<Npp8u> src_mem(width, height, 3);
  NppImageMemory<Npp8u> dst_mem(width, height, 3);
  src_mem.copyFromHost(src);

  NppiSize roi = {width, height};
  NppStatus status = nppiRGBToHLS_8u_C3R(src_mem.get(), src_mem.step(), dst_mem.get(), dst_mem.step(), roi);
  ASSERT_EQ(status, NPP_NO_ERROR);

  std::vector<Npp8u> dst;
  dst_mem.copyToHost(dst);

  for (int i = 0; i < width; ++i) {
    Npp8u h, l, s;
    rgb_to_hls_ref(pixels[i].r, pixels[i].g, pixels[i].b, h, l, s);
    EXPECT_NEAR(dst[i * 3 + 0], h, 2);
    EXPECT_NEAR(dst[i * 3 + 1], l, 2);
    EXPECT_NEAR(dst[i * 3 + 2], s, 2);
  }
}

TEST_F(HLSConversionTest, HLSToRGB_8u_C3R_RoundTripAccuracy) {
  const int width = 4;
  const int height = 1;
  std::vector<RgbPixel> pixels = {
      {10, 30, 200},
      {120, 200, 40},
      {200, 50, 50},
      {0, 0, 0},
  };

  std::vector<Npp8u> src;
  build_pixels(src, pixels, false);

  NppImageMemory<Npp8u> src_mem(width, height, 3);
  NppImageMemory<Npp8u> hls_mem(width, height, 3);
  NppImageMemory<Npp8u> dst_mem(width, height, 3);
  src_mem.copyFromHost(src);

  NppiSize roi = {width, height};
  NppStatus status = nppiRGBToHLS_8u_C3R(src_mem.get(), src_mem.step(), hls_mem.get(), hls_mem.step(), roi);
  ASSERT_EQ(status, NPP_NO_ERROR);

  status = nppiHLSToRGB_8u_C3R(hls_mem.get(), hls_mem.step(), dst_mem.get(), dst_mem.step(), roi);
  ASSERT_EQ(status, NPP_NO_ERROR);

  std::vector<Npp8u> dst;
  dst_mem.copyToHost(dst);
  for (int i = 0; i < width; ++i) {
    EXPECT_NEAR(dst[i * 3 + 0], pixels[i].r, 2);
    EXPECT_NEAR(dst[i * 3 + 1], pixels[i].g, 2);
    EXPECT_NEAR(dst[i * 3 + 2], pixels[i].b, 2);
  }
}


TEST_F(HLSConversionTest, RGBToHLS_8u_AC4R_AlphaBehavior) {
  const int width = 2;
  const int height = 2;
  std::vector<RgbPixel> pixels = {
      {255, 0, 0},
      {0, 255, 0},
      {0, 0, 255},
      {255, 255, 255},
  };

  std::vector<Npp8u> src(width * height * 4);
  std::vector<Npp8u> alpha(width * height);
  for (int i = 0; i < width * height; ++i) {
    src[i * 4 + 0] = pixels[i].r;
    src[i * 4 + 1] = pixels[i].g;
    src[i * 4 + 2] = pixels[i].b;
    src[i * 4 + 3] = static_cast<Npp8u>(10 + i * 20);
    alpha[i] = src[i * 4 + 3];
  }

  NppImageMemory<Npp8u> src_mem(width, height, 4);
  NppImageMemory<Npp8u> dst_mem(width, height, 4);
  src_mem.copyFromHost(src);

  NppiSize roi = {width, height};
  NppStatus status = nppiRGBToHLS_8u_AC4R(src_mem.get(), src_mem.step(), dst_mem.get(), dst_mem.step(), roi);
  ASSERT_EQ(status, NPP_NO_ERROR);

  std::vector<Npp8u> dst;
  dst_mem.copyToHost(dst);
  for (int i = 0; i < width * height; ++i) {
    Npp8u h, l, s;
    rgb_to_hls_ref(pixels[i].r, pixels[i].g, pixels[i].b, h, l, s);
    EXPECT_NEAR(dst[i * 4 + 0], h, 2);
    EXPECT_NEAR(dst[i * 4 + 1], l, 2);
    EXPECT_NEAR(dst[i * 4 + 2], s, 2);
    EXPECT_EQ(dst[i * 4 + 3], 0);
  }
}

TEST_F(HLSConversionTest, HLSToRGB_8u_AC4R_AlphaBehavior) {
  const int width = 2;
  const int height = 2;
  std::vector<RgbPixel> pixels = {
      {10, 30, 200},
      {120, 200, 40},
      {200, 50, 50},
      {0, 0, 0},
  };

  std::vector<Npp8u> src(width * height * 4);
  std::vector<Npp8u> alpha(width * height);
  for (int i = 0; i < width * height; ++i) {
    Npp8u h, l, s;
    rgb_to_hls_ref(pixels[i].r, pixels[i].g, pixels[i].b, h, l, s);
    src[i * 4 + 0] = h;
    src[i * 4 + 1] = l;
    src[i * 4 + 2] = s;
    src[i * 4 + 3] = static_cast<Npp8u>(15 + i * 15);
    alpha[i] = src[i * 4 + 3];
  }

  NppImageMemory<Npp8u> src_mem(width, height, 4);
  NppImageMemory<Npp8u> dst_mem(width, height, 4);
  src_mem.copyFromHost(src);

  NppiSize roi = {width, height};
  NppStatus status = nppiHLSToRGB_8u_AC4R(src_mem.get(), src_mem.step(), dst_mem.get(), dst_mem.step(), roi);
  ASSERT_EQ(status, NPP_NO_ERROR);

  std::vector<Npp8u> dst;
  dst_mem.copyToHost(dst);
  for (int i = 0; i < width * height; ++i) {
    EXPECT_NEAR(dst[i * 4 + 0], pixels[i].r, 2);
    EXPECT_NEAR(dst[i * 4 + 1], pixels[i].g, 3);
    EXPECT_NEAR(dst[i * 4 + 2], pixels[i].b, 3);
    EXPECT_EQ(dst[i * 4 + 3], 0);
  }
}

TEST_F(HLSConversionTest, BGRToHLS_8u_AC4R_AlphaCleared) {
  const int width = 2;
  const int height = 2;
  std::vector<RgbPixel> pixels = {
      {255, 0, 0},
      {0, 255, 0},
      {0, 0, 255},
      {255, 255, 255},
  };

  std::vector<Npp8u> src(width * height * 4);
  for (int i = 0; i < width * height; ++i) {
    src[i * 4 + 0] = pixels[i].b;
    src[i * 4 + 1] = pixels[i].g;
    src[i * 4 + 2] = pixels[i].r;
    src[i * 4 + 3] = static_cast<Npp8u>(10 + i * 30);
  }

  NppImageMemory<Npp8u> src_mem(width, height, 4);
  NppImageMemory<Npp8u> dst_mem(width, height, 4);
  src_mem.copyFromHost(src);

  NppiSize roi = {width, height};
  NppStatus status = nppiBGRToHLS_8u_AC4R(src_mem.get(), src_mem.step(), dst_mem.get(), dst_mem.step(), roi);
  ASSERT_EQ(status, NPP_NO_ERROR);

  std::vector<Npp8u> dst;
  dst_mem.copyToHost(dst);
  for (int i = 0; i < width * height; ++i) {
    Npp8u h, l, s;
    rgb_to_hls_ref(pixels[i].r, pixels[i].g, pixels[i].b, h, l, s);
    EXPECT_NEAR(dst[i * 4 + 0], h, 2);
    EXPECT_NEAR(dst[i * 4 + 1], l, 2);
    EXPECT_NEAR(dst[i * 4 + 2], s, 2);
    EXPECT_EQ(dst[i * 4 + 3], 0);
  }
}

TEST_F(HLSConversionTest, BGRToHLS_8u_C3P3R_RoundTripPacked) {
  const int width = 2;
  const int height = 2;
  std::vector<RgbPixel> pixels = {
      {10, 30, 200},
      {120, 200, 40},
      {200, 50, 50},
      {0, 0, 0},
  };

  std::vector<Npp8u> src;
  build_pixels(src, pixels, true);

  NppImageMemory<Npp8u> src_mem(width, height, 3);
  src_mem.copyFromHost(src);

  NppImageMemory<Npp8u> h_plane(width, height, 1);
  NppImageMemory<Npp8u> l_plane(width, height, 1);
  NppImageMemory<Npp8u> s_plane(width, height, 1);

  Npp8u *hls_planes[3] = {h_plane.get(), l_plane.get(), s_plane.get()};

  NppiSize roi = {width, height};
  NppStatus status = nppiBGRToHLS_8u_C3P3R(src_mem.get(), src_mem.step(), hls_planes, h_plane.step(), roi);
  ASSERT_EQ(status, NPP_NO_ERROR);

  NppImageMemory<Npp8u> dst_mem(width, height, 3);
  status = nppiHLSToBGR_8u_P3C3R((const Npp8u *const *)hls_planes, h_plane.step(), dst_mem.get(), dst_mem.step(), roi);
  ASSERT_EQ(status, NPP_NO_ERROR);

  std::vector<Npp8u> dst;
  dst_mem.copyToHost(dst);
  for (int i = 0; i < width * height; ++i) {
    EXPECT_NEAR(dst[i * 3 + 0], pixels[i].b, 3);
    EXPECT_NEAR(dst[i * 3 + 1], pixels[i].g, 3);
    EXPECT_NEAR(dst[i * 3 + 2], pixels[i].r, 3);
  }
}

TEST_F(HLSConversionTest, BGRToHLS_8u_P3R_RoundTripPlanar) {
  const int width = 2;
  const int height = 2;
  std::vector<RgbPixel> pixels = {
      {255, 0, 0},
      {0, 255, 0},
      {0, 0, 255},
      {255, 255, 255},
  };

  std::vector<Npp8u> b_plane, g_plane, r_plane;
  build_planar_bgr(pixels, b_plane, g_plane, r_plane);

  NppImageMemory<Npp8u> b_mem(width, height, 1);
  NppImageMemory<Npp8u> g_mem(width, height, 1);
  NppImageMemory<Npp8u> r_mem(width, height, 1);
  b_mem.copyFromHost(b_plane);
  g_mem.copyFromHost(g_plane);
  r_mem.copyFromHost(r_plane);

  const Npp8u *bgr_planes[3] = {b_mem.get(), g_mem.get(), r_mem.get()};

  NppImageMemory<Npp8u> h_plane(width, height, 1);
  NppImageMemory<Npp8u> l_plane(width, height, 1);
  NppImageMemory<Npp8u> s_plane(width, height, 1);
  Npp8u *hls_planes[3] = {h_plane.get(), l_plane.get(), s_plane.get()};

  NppiSize roi = {width, height};
  NppStatus status = nppiBGRToHLS_8u_P3R(bgr_planes, b_mem.step(), hls_planes, h_plane.step(), roi);
  ASSERT_EQ(status, NPP_NO_ERROR);

  NppImageMemory<Npp8u> out_b(width, height, 1);
  NppImageMemory<Npp8u> out_g(width, height, 1);
  NppImageMemory<Npp8u> out_r(width, height, 1);
  Npp8u *out_planes[3] = {out_b.get(), out_g.get(), out_r.get()};

  status = nppiHLSToBGR_8u_P3R((const Npp8u *const *)hls_planes, h_plane.step(), out_planes, out_b.step(), roi);
  ASSERT_EQ(status, NPP_NO_ERROR);

  std::vector<Npp8u> out_bh, out_gh, out_rh;
  out_b.copyToHost(out_bh);
  out_g.copyToHost(out_gh);
  out_r.copyToHost(out_rh);

  for (int i = 0; i < width * height; ++i) {
    EXPECT_NEAR(out_bh[i], b_plane[i], 3);
    EXPECT_NEAR(out_gh[i], g_plane[i], 3);
    EXPECT_NEAR(out_rh[i], r_plane[i], 3);
  }
}

