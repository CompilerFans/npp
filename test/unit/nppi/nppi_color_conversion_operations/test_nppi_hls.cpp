#include "npp_test_base.h"
#include <algorithm>
#include <cmath>
#include <random>
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

namespace {
struct HlsRoundTripCase {
  int width;
  int height;
  unsigned int seed;
};

static void fill_random_pixels(std::vector<RgbPixel> &pixels, int width, int height, unsigned int seed) {
  std::mt19937 rng(seed);
  std::uniform_int_distribution<int> dist(0, 255);
  pixels.resize(width * height);
  for (auto &p : pixels) {
    p.r = static_cast<Npp8u>(dist(rng));
    p.g = static_cast<Npp8u>(dist(rng));
    p.b = static_cast<Npp8u>(dist(rng));
  }
}

static void build_bgra(const std::vector<RgbPixel> &pixels, std::vector<Npp8u> &bgra, std::vector<Npp8u> &a_plane) {
  bgra.resize(pixels.size() * 4);
  a_plane.resize(pixels.size());
  for (size_t i = 0; i < pixels.size(); ++i) {
    bgra[i * 4 + 0] = pixels[i].b;
    bgra[i * 4 + 1] = pixels[i].g;
    bgra[i * 4 + 2] = pixels[i].r;
    bgra[i * 4 + 3] = static_cast<Npp8u>(50 + (i % 200));
    a_plane[i] = bgra[i * 4 + 3];
  }
}

static void split_bgr_planes(const std::vector<RgbPixel> &pixels, std::vector<Npp8u> &b, std::vector<Npp8u> &g,
                             std::vector<Npp8u> &r) {
  b.resize(pixels.size());
  g.resize(pixels.size());
  r.resize(pixels.size());
  for (size_t i = 0; i < pixels.size(); ++i) {
    b[i] = pixels[i].b;
    g[i] = pixels[i].g;
    r[i] = pixels[i].r;
  }
}

static void expect_plane_near(const std::vector<Npp8u> &got, const std::vector<Npp8u> &exp, int tol) {
  ASSERT_EQ(got.size(), exp.size());
  for (size_t i = 0; i < got.size(); ++i) {
    EXPECT_NEAR(got[i], exp[i], tol) << "Mismatch at " << i;
  }
}
} // namespace

class HLSMissingRoundTripTest : public NppTestBase, public ::testing::WithParamInterface<HlsRoundTripCase> {};

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

TEST_F(HLSConversionTest, RGBToHLS_8u_C3R_Ctx_BasicGray) {
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
  NppStreamContext nppStreamCtx{};
  nppGetStreamContext(&nppStreamCtx);
  nppStreamCtx.hStream = 0;

  NppStatus status =
      nppiRGBToHLS_8u_C3R_Ctx(src_mem.get(), src_mem.step(), dst_mem.get(), dst_mem.step(), roi, nppStreamCtx);
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

TEST_F(HLSConversionTest, HLSToRGB_8u_C3R_Ctx_RoundTripAccuracy) {
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
  NppStreamContext nppStreamCtx{};
  nppGetStreamContext(&nppStreamCtx);
  nppStreamCtx.hStream = 0;

  NppStatus status =
      nppiRGBToHLS_8u_C3R_Ctx(src_mem.get(), src_mem.step(), hls_mem.get(), hls_mem.step(), roi, nppStreamCtx);
  ASSERT_EQ(status, NPP_NO_ERROR);

  status = nppiHLSToRGB_8u_C3R_Ctx(hls_mem.get(), hls_mem.step(), dst_mem.get(), dst_mem.step(), roi, nppStreamCtx);
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

TEST_F(HLSConversionTest, RGBToHLS_8u_AC4R_Ctx_AlphaBehavior) {
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
  NppStreamContext nppStreamCtx{};
  nppGetStreamContext(&nppStreamCtx);
  nppStreamCtx.hStream = 0;

  NppStatus status =
      nppiRGBToHLS_8u_AC4R_Ctx(src_mem.get(), src_mem.step(), dst_mem.get(), dst_mem.step(), roi, nppStreamCtx);
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

TEST_F(HLSConversionTest, HLSToRGB_8u_AC4R_Ctx_AlphaBehavior) {
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
  NppStreamContext nppStreamCtx{};
  nppGetStreamContext(&nppStreamCtx);
  nppStreamCtx.hStream = 0;

  NppStatus status =
      nppiHLSToRGB_8u_AC4R_Ctx(src_mem.get(), src_mem.step(), dst_mem.get(), dst_mem.step(), roi, nppStreamCtx);
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

TEST_F(HLSConversionTest, BGRToHLS_8u_AC4R_Ctx_AlphaCleared) {
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
  NppStreamContext nppStreamCtx{};
  nppGetStreamContext(&nppStreamCtx);
  nppStreamCtx.hStream = 0;

  NppStatus status =
      nppiBGRToHLS_8u_AC4R_Ctx(src_mem.get(), src_mem.step(), dst_mem.get(), dst_mem.step(), roi, nppStreamCtx);
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

TEST_F(HLSConversionTest, BGRToHLS_8u_C3P3R_Ctx_RoundTripPacked) {
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
  NppStreamContext nppStreamCtx{};
  nppGetStreamContext(&nppStreamCtx);
  nppStreamCtx.hStream = 0;

  NppStatus status =
      nppiBGRToHLS_8u_C3P3R_Ctx(src_mem.get(), src_mem.step(), hls_planes, h_plane.step(), roi, nppStreamCtx);
  ASSERT_EQ(status, NPP_NO_ERROR);

  NppImageMemory<Npp8u> dst_mem(width, height, 3);
  status = nppiHLSToBGR_8u_P3C3R_Ctx((const Npp8u *const *)hls_planes, h_plane.step(), dst_mem.get(),
                                    dst_mem.step(), roi, nppStreamCtx);
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

TEST_F(HLSConversionTest, BGRToHLS_8u_P3R_Ctx_RoundTripPlanar) {
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
  NppStreamContext nppStreamCtx{};
  nppGetStreamContext(&nppStreamCtx);
  nppStreamCtx.hStream = 0;

  NppStatus status =
      nppiBGRToHLS_8u_P3R_Ctx(bgr_planes, b_mem.step(), hls_planes, h_plane.step(), roi, nppStreamCtx);
  ASSERT_EQ(status, NPP_NO_ERROR);

  NppImageMemory<Npp8u> out_b(width, height, 1);
  NppImageMemory<Npp8u> out_g(width, height, 1);
  NppImageMemory<Npp8u> out_r(width, height, 1);
  Npp8u *out_planes[3] = {out_b.get(), out_g.get(), out_r.get()};

  status = nppiHLSToBGR_8u_P3R_Ctx((const Npp8u *const *)hls_planes, h_plane.step(), out_planes, out_b.step(), roi,
                                  nppStreamCtx);
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

TEST_P(HLSMissingRoundTripTest, MissingBGRToHLSVariants_RoundTrip) {
  const auto param = GetParam();
  const int width = param.width;
  const int height = param.height;
  NppiSize roi = {width, height};

  std::vector<RgbPixel> pixels;
  fill_random_pixels(pixels, width, height, param.seed);

  std::vector<Npp8u> b_plane, g_plane, r_plane;
  split_bgr_planes(pixels, b_plane, g_plane, r_plane);

  std::vector<Npp8u> bgra;
  std::vector<Npp8u> a_plane;
  build_bgra(pixels, bgra, a_plane);

  NppImageMemory<Npp8u> b_mem(width, height);
  NppImageMemory<Npp8u> g_mem(width, height);
  NppImageMemory<Npp8u> r_mem(width, height);
  b_mem.copyFromHost(b_plane);
  g_mem.copyFromHost(g_plane);
  r_mem.copyFromHost(r_plane);

  NppImageMemory<Npp8u> src_bgra(width, height, 4);
  src_bgra.copyFromHost(bgra);

  NppStreamContext ctx{};
  nppGetStreamContext(&ctx);
  ctx.hStream = 0;

  // P3C3R: planar BGR -> packed HLS -> planar BGR
  {
    const Npp8u *src_planes[3] = {b_mem.get(), g_mem.get(), r_mem.get()};
    NppImageMemory<Npp8u> hls_packed(width, height, 3);
    NppStatus status = nppiBGRToHLS_8u_P3C3R(src_planes, b_mem.step(), hls_packed.get(), hls_packed.step(), roi);
    ASSERT_EQ(status, NPP_NO_ERROR);

    NppImageMemory<Npp8u> b_out(width, height);
    NppImageMemory<Npp8u> g_out(width, height);
    NppImageMemory<Npp8u> r_out(width, height);
    Npp8u *dst_planes[3] = {b_out.get(), g_out.get(), r_out.get()};
    status = nppiHLSToBGR_8u_C3P3R(hls_packed.get(), hls_packed.step(), dst_planes, b_out.step(), roi);
    ASSERT_EQ(status, NPP_NO_ERROR);

    std::vector<Npp8u> b_out_h, g_out_h, r_out_h;
    b_out.copyToHost(b_out_h);
    g_out.copyToHost(g_out_h);
    r_out.copyToHost(r_out_h);
    expect_plane_near(b_out_h, b_plane, 2);
    expect_plane_near(g_out_h, g_plane, 2);
    expect_plane_near(r_out_h, r_plane, 2);

    NppImageMemory<Npp8u> hls_packed_ctx(width, height, 3);
    status =
        nppiBGRToHLS_8u_P3C3R_Ctx(src_planes, b_mem.step(), hls_packed_ctx.get(), hls_packed_ctx.step(), roi, ctx);
    ASSERT_EQ(status, NPP_NO_ERROR);

    NppImageMemory<Npp8u> b_ctx(width, height);
    NppImageMemory<Npp8u> g_ctx(width, height);
    NppImageMemory<Npp8u> r_ctx(width, height);
    Npp8u *dst_ctx_planes[3] = {b_ctx.get(), g_ctx.get(), r_ctx.get()};
    status = nppiHLSToBGR_8u_C3P3R_Ctx(hls_packed_ctx.get(), hls_packed_ctx.step(), dst_ctx_planes, b_ctx.step(), roi,
                                       ctx);
    ASSERT_EQ(status, NPP_NO_ERROR);

    std::vector<Npp8u> b_ctx_h, g_ctx_h, r_ctx_h;
    b_ctx.copyToHost(b_ctx_h);
    g_ctx.copyToHost(g_ctx_h);
    r_ctx.copyToHost(r_ctx_h);
    expect_plane_near(b_ctx_h, b_out_h, 0);
    expect_plane_near(g_ctx_h, g_out_h, 0);
    expect_plane_near(r_ctx_h, r_out_h, 0);
  }

  // AC4P4R: packed BGRA -> planar HLSA -> planar BGRA
  {
    NppImageMemory<Npp8u> h_plane(width, height);
    NppImageMemory<Npp8u> l_plane(width, height);
    NppImageMemory<Npp8u> s_plane(width, height);
    NppImageMemory<Npp8u> a_out(width, height);
    Npp8u *dst_planes[4] = {h_plane.get(), l_plane.get(), s_plane.get(), a_out.get()};

    NppStatus status =
        nppiBGRToHLS_8u_AC4P4R(src_bgra.get(), src_bgra.step(), dst_planes, h_plane.step(), roi);
    ASSERT_EQ(status, NPP_NO_ERROR);

    NppImageMemory<Npp8u> b_out(width, height);
    NppImageMemory<Npp8u> g_out(width, height);
    NppImageMemory<Npp8u> r_out(width, height);
    NppImageMemory<Npp8u> a_back(width, height);
    Npp8u *bgra_planes[4] = {b_out.get(), g_out.get(), r_out.get(), a_back.get()};
    status = nppiHLSToBGR_8u_AP4R((const Npp8u *const *)dst_planes, h_plane.step(), bgra_planes, b_out.step(), roi);
    ASSERT_EQ(status, NPP_NO_ERROR);

    std::vector<Npp8u> b_out_h, g_out_h, r_out_h, a_back_h;
    b_out.copyToHost(b_out_h);
    g_out.copyToHost(g_out_h);
    r_out.copyToHost(r_out_h);
    a_back.copyToHost(a_back_h);
    expect_plane_near(b_out_h, b_plane, 2);
    expect_plane_near(g_out_h, g_plane, 2);
    expect_plane_near(r_out_h, r_plane, 2);
    expect_plane_near(a_back_h, a_plane, 0);

    NppImageMemory<Npp8u> h_plane_ctx(width, height);
    NppImageMemory<Npp8u> l_plane_ctx(width, height);
    NppImageMemory<Npp8u> s_plane_ctx(width, height);
    NppImageMemory<Npp8u> a_plane_ctx(width, height);
    Npp8u *dst_ctx_planes[4] = {h_plane_ctx.get(), l_plane_ctx.get(), s_plane_ctx.get(), a_plane_ctx.get()};
    status =
        nppiBGRToHLS_8u_AC4P4R_Ctx(src_bgra.get(), src_bgra.step(), dst_ctx_planes, h_plane_ctx.step(), roi, ctx);
    ASSERT_EQ(status, NPP_NO_ERROR);

    NppImageMemory<Npp8u> b_ctx(width, height);
    NppImageMemory<Npp8u> g_ctx(width, height);
    NppImageMemory<Npp8u> r_ctx(width, height);
    NppImageMemory<Npp8u> a_ctx(width, height);
    Npp8u *bgra_ctx_planes[4] = {b_ctx.get(), g_ctx.get(), r_ctx.get(), a_ctx.get()};
    status = nppiHLSToBGR_8u_AP4R_Ctx((const Npp8u *const *)dst_ctx_planes, h_plane_ctx.step(), bgra_ctx_planes,
                                      b_ctx.step(), roi, ctx);
    ASSERT_EQ(status, NPP_NO_ERROR);

    std::vector<Npp8u> b_ctx_h, g_ctx_h, r_ctx_h, a_ctx_h;
    b_ctx.copyToHost(b_ctx_h);
    g_ctx.copyToHost(g_ctx_h);
    r_ctx.copyToHost(r_ctx_h);
    a_ctx.copyToHost(a_ctx_h);
    expect_plane_near(b_ctx_h, b_out_h, 0);
    expect_plane_near(g_ctx_h, g_out_h, 0);
    expect_plane_near(r_ctx_h, r_out_h, 0);
    expect_plane_near(a_ctx_h, a_back_h, 0);
  }

  // AP4C4R: planar BGRA -> packed HLSA -> planar BGRA
  {
    const Npp8u *src_planes[4] = {b_mem.get(), g_mem.get(), r_mem.get(), nullptr};
    NppImageMemory<Npp8u> alpha_plane(width, height);
    alpha_plane.copyFromHost(a_plane);
    src_planes[3] = alpha_plane.get();

    NppImageMemory<Npp8u> hls_packed(width, height, 4);
    NppStatus status =
        nppiBGRToHLS_8u_AP4C4R(src_planes, b_mem.step(), hls_packed.get(), hls_packed.step(), roi);
    ASSERT_EQ(status, NPP_NO_ERROR);

    NppImageMemory<Npp8u> b_out(width, height);
    NppImageMemory<Npp8u> g_out(width, height);
    NppImageMemory<Npp8u> r_out(width, height);
    NppImageMemory<Npp8u> a_out(width, height);
    Npp8u *dst_planes[4] = {b_out.get(), g_out.get(), r_out.get(), a_out.get()};
    status = nppiHLSToBGR_8u_AC4P4R(hls_packed.get(), hls_packed.step(), dst_planes, b_out.step(), roi);
    ASSERT_EQ(status, NPP_NO_ERROR);

    std::vector<Npp8u> b_out_h, g_out_h, r_out_h, a_out_h;
    b_out.copyToHost(b_out_h);
    g_out.copyToHost(g_out_h);
    r_out.copyToHost(r_out_h);
    a_out.copyToHost(a_out_h);
    expect_plane_near(b_out_h, b_plane, 2);
    expect_plane_near(g_out_h, g_plane, 2);
    expect_plane_near(r_out_h, r_plane, 2);
    expect_plane_near(a_out_h, a_plane, 0);

    NppImageMemory<Npp8u> hls_packed_ctx(width, height, 4);
    status = nppiBGRToHLS_8u_AP4C4R_Ctx(src_planes, b_mem.step(), hls_packed_ctx.get(), hls_packed_ctx.step(), roi,
                                        ctx);
    ASSERT_EQ(status, NPP_NO_ERROR);

    NppImageMemory<Npp8u> b_ctx(width, height);
    NppImageMemory<Npp8u> g_ctx(width, height);
    NppImageMemory<Npp8u> r_ctx(width, height);
    NppImageMemory<Npp8u> a_ctx(width, height);
    Npp8u *dst_ctx_planes[4] = {b_ctx.get(), g_ctx.get(), r_ctx.get(), a_ctx.get()};
    status = nppiHLSToBGR_8u_AC4P4R_Ctx(hls_packed_ctx.get(), hls_packed_ctx.step(), dst_ctx_planes, b_ctx.step(),
                                        roi, ctx);
    ASSERT_EQ(status, NPP_NO_ERROR);

    std::vector<Npp8u> b_ctx_h, g_ctx_h, r_ctx_h, a_ctx_h;
    b_ctx.copyToHost(b_ctx_h);
    g_ctx.copyToHost(g_ctx_h);
    r_ctx.copyToHost(r_ctx_h);
    a_ctx.copyToHost(a_ctx_h);
    expect_plane_near(b_ctx_h, b_out_h, 0);
    expect_plane_near(g_ctx_h, g_out_h, 0);
    expect_plane_near(r_ctx_h, r_out_h, 0);
    expect_plane_near(a_ctx_h, a_out_h, 0);
  }

  // AP4R: planar BGRA -> planar HLSA -> planar BGRA
  {
    const Npp8u *src_planes[4] = {b_mem.get(), g_mem.get(), r_mem.get(), nullptr};
    NppImageMemory<Npp8u> alpha_plane(width, height);
    alpha_plane.copyFromHost(a_plane);
    src_planes[3] = alpha_plane.get();

    NppImageMemory<Npp8u> h_plane(width, height);
    NppImageMemory<Npp8u> l_plane(width, height);
    NppImageMemory<Npp8u> s_plane(width, height);
    NppImageMemory<Npp8u> a_plane_out(width, height);
    Npp8u *dst_planes[4] = {h_plane.get(), l_plane.get(), s_plane.get(), a_plane_out.get()};
    NppStatus status = nppiBGRToHLS_8u_AP4R(src_planes, b_mem.step(), dst_planes, h_plane.step(), roi);
    ASSERT_EQ(status, NPP_NO_ERROR);

    NppImageMemory<Npp8u> b_out(width, height);
    NppImageMemory<Npp8u> g_out(width, height);
    NppImageMemory<Npp8u> r_out(width, height);
    NppImageMemory<Npp8u> a_out(width, height);
    Npp8u *bgra_planes[4] = {b_out.get(), g_out.get(), r_out.get(), a_out.get()};
    status = nppiHLSToBGR_8u_AP4R((const Npp8u *const *)dst_planes, h_plane.step(), bgra_planes, b_out.step(), roi);
    ASSERT_EQ(status, NPP_NO_ERROR);

    std::vector<Npp8u> b_out_h, g_out_h, r_out_h, a_out_h;
    b_out.copyToHost(b_out_h);
    g_out.copyToHost(g_out_h);
    r_out.copyToHost(r_out_h);
    a_out.copyToHost(a_out_h);
    expect_plane_near(b_out_h, b_plane, 2);
    expect_plane_near(g_out_h, g_plane, 2);
    expect_plane_near(r_out_h, r_plane, 2);
    expect_plane_near(a_out_h, a_plane, 0);

    NppImageMemory<Npp8u> h_plane_ctx(width, height);
    NppImageMemory<Npp8u> l_plane_ctx(width, height);
    NppImageMemory<Npp8u> s_plane_ctx(width, height);
    NppImageMemory<Npp8u> a_plane_ctx(width, height);
    Npp8u *dst_ctx_planes[4] = {h_plane_ctx.get(), l_plane_ctx.get(), s_plane_ctx.get(), a_plane_ctx.get()};
    status = nppiBGRToHLS_8u_AP4R_Ctx(src_planes, b_mem.step(), dst_ctx_planes, h_plane_ctx.step(), roi, ctx);
    ASSERT_EQ(status, NPP_NO_ERROR);

    NppImageMemory<Npp8u> b_ctx(width, height);
    NppImageMemory<Npp8u> g_ctx(width, height);
    NppImageMemory<Npp8u> r_ctx(width, height);
    NppImageMemory<Npp8u> a_ctx(width, height);
    Npp8u *bgra_ctx_planes[4] = {b_ctx.get(), g_ctx.get(), r_ctx.get(), a_ctx.get()};
    status = nppiHLSToBGR_8u_AP4R_Ctx((const Npp8u *const *)dst_ctx_planes, h_plane_ctx.step(), bgra_ctx_planes,
                                      b_ctx.step(), roi, ctx);
    ASSERT_EQ(status, NPP_NO_ERROR);

    std::vector<Npp8u> b_ctx_h, g_ctx_h, r_ctx_h, a_ctx_h;
    b_ctx.copyToHost(b_ctx_h);
    g_ctx.copyToHost(g_ctx_h);
    r_ctx.copyToHost(r_ctx_h);
    a_ctx.copyToHost(a_ctx_h);
    expect_plane_near(b_ctx_h, b_out_h, 0);
    expect_plane_near(g_ctx_h, g_out_h, 0);
    expect_plane_near(r_ctx_h, r_out_h, 0);
    expect_plane_near(a_ctx_h, a_out_h, 0);
  }
}

INSTANTIATE_TEST_SUITE_P(FunctionalCases, HLSMissingRoundTripTest,
                         ::testing::Values(HlsRoundTripCase{8, 4, 123}, HlsRoundTripCase{10, 6, 321}));
INSTANTIATE_TEST_SUITE_P(PrecisionCases, HLSMissingRoundTripTest,
                         ::testing::Values(HlsRoundTripCase{32, 16, 456}, HlsRoundTripCase{64, 24, 789}));
