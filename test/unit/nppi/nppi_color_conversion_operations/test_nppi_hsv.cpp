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

static inline void rgb_to_hsv_ref(Npp8u r, Npp8u g, Npp8u b, Npp8u &h, Npp8u &s, Npp8u &v) {
  float rf = r / 255.0f;
  float gf = g / 255.0f;
  float bf = b / 255.0f;

  float maxv = std::max(rf, std::max(gf, bf));
  float minv = std::min(rf, std::min(gf, bf));
  float delta = maxv - minv;

  float hval = 0.0f;
  float sval = (maxv <= 0.0f) ? 0.0f : (delta / maxv);

  if (delta > 0.0f) {
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
  s = clamp_u8_round(sval * 255.0f);
  v = clamp_u8_round(maxv * 255.0f);
}

static inline void hsv_to_rgb_ref(Npp8u h, Npp8u s, Npp8u v, Npp8u &r, Npp8u &g, Npp8u &b) {
  float hf = h / 255.0f;
  float sf = s / 255.0f;
  float vf = v / 255.0f;

  if (sf <= 0.0f) {
    r = g = b = clamp_u8_round(vf * 255.0f);
    return;
  }

  float h6 = hf * 6.0f;
  int i = static_cast<int>(std::floor(h6));
  float f = h6 - static_cast<float>(i);
  float p = vf * (1.0f - sf);
  float q = vf * (1.0f - sf * f);
  float t = vf * (1.0f - sf * (1.0f - f));

  float rf = 0.0f, gf = 0.0f, bf = 0.0f;
  switch (i % 6) {
  case 0:
    rf = vf;
    gf = t;
    bf = p;
    break;
  case 1:
    rf = q;
    gf = vf;
    bf = p;
    break;
  case 2:
    rf = p;
    gf = vf;
    bf = t;
    break;
  case 3:
    rf = p;
    gf = q;
    bf = vf;
    break;
  case 4:
    rf = t;
    gf = p;
    bf = vf;
    break;
  default:
    rf = vf;
    gf = p;
    bf = q;
    break;
  }

  r = clamp_u8_round(rf * 255.0f);
  g = clamp_u8_round(gf * 255.0f);
  b = clamp_u8_round(bf * 255.0f);
}
} // namespace

class HSVConversionTest : public NppTestBase {};

namespace {
struct HsvCtxCase {
  int width;
  int height;
  unsigned int seed;
};

static void fill_random_u8(std::vector<Npp8u> &data, int count, unsigned int seed) {
  std::mt19937 rng(seed);
  std::uniform_int_distribution<int> dist(0, 255);
  data.resize(count);
  for (auto &v : data) {
    v = static_cast<Npp8u>(dist(rng));
  }
}

static void expect_equal_u8(const std::vector<Npp8u> &a, const std::vector<Npp8u> &b, int tol) {
  ASSERT_EQ(a.size(), b.size());
  for (size_t i = 0; i < a.size(); ++i) {
    EXPECT_NEAR(a[i], b[i], tol) << "Mismatch at " << i;
  }
}
} // namespace

class HSVConversionCtxParamTest : public NppTestBase, public ::testing::WithParamInterface<HsvCtxCase> {};

TEST_F(HSVConversionTest, RGBToHSV_8u_C3R_BasicColors) {
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
    src[i * 3 + 0] = pixels[i].r;
    src[i * 3 + 1] = pixels[i].g;
    src[i * 3 + 2] = pixels[i].b;
  }

  NppImageMemory<Npp8u> src_mem(width, height, 3);
  NppImageMemory<Npp8u> dst_mem(width, height, 3);
  src_mem.copyFromHost(src);

  NppiSize roi = {width, height};
  NppStatus status = nppiRGBToHSV_8u_C3R(src_mem.get(), src_mem.step(), dst_mem.get(), dst_mem.step(), roi);
  ASSERT_EQ(status, NPP_NO_ERROR);

  std::vector<Npp8u> dst;
  dst_mem.copyToHost(dst);
  for (int i = 0; i < width; ++i) {
    Npp8u h, s, v;
    rgb_to_hsv_ref(pixels[i].r, pixels[i].g, pixels[i].b, h, s, v);
    EXPECT_NEAR(dst[i * 3 + 0], h, 2);
    EXPECT_NEAR(dst[i * 3 + 1], s, 2);
    EXPECT_NEAR(dst[i * 3 + 2], v, 2);
  }
}

TEST_F(HSVConversionTest, HSVToRGB_8u_C3R_RoundTripAccuracy) {
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
    src[i * 3 + 0] = pixels[i].r;
    src[i * 3 + 1] = pixels[i].g;
    src[i * 3 + 2] = pixels[i].b;
  }

  NppImageMemory<Npp8u> src_mem(width, height, 3);
  NppImageMemory<Npp8u> hsv_mem(width, height, 3);
  NppImageMemory<Npp8u> dst_mem(width, height, 3);
  src_mem.copyFromHost(src);

  NppiSize roi = {width, height};
  NppStatus status = nppiRGBToHSV_8u_C3R(src_mem.get(), src_mem.step(), hsv_mem.get(), hsv_mem.step(), roi);
  ASSERT_EQ(status, NPP_NO_ERROR);

  status = nppiHSVToRGB_8u_C3R(hsv_mem.get(), hsv_mem.step(), dst_mem.get(), dst_mem.step(), roi);
  ASSERT_EQ(status, NPP_NO_ERROR);

  std::vector<Npp8u> dst;
  dst_mem.copyToHost(dst);
  for (int i = 0; i < width; ++i) {
    EXPECT_NEAR(dst[i * 3 + 0], pixels[i].r, 3);
    EXPECT_NEAR(dst[i * 3 + 1], pixels[i].g, 3);
    EXPECT_NEAR(dst[i * 3 + 2], pixels[i].b, 3);
  }
}

TEST_F(HSVConversionTest, RGBToHSV_8u_AC4R_AlphaCleared) {
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
    src[i * 4 + 0] = pixels[i].r;
    src[i * 4 + 1] = pixels[i].g;
    src[i * 4 + 2] = pixels[i].b;
    src[i * 4 + 3] = static_cast<Npp8u>(10 + i * 30);
  }

  NppImageMemory<Npp8u> src_mem(width, height, 4);
  NppImageMemory<Npp8u> dst_mem(width, height, 4);
  src_mem.copyFromHost(src);

  NppiSize roi = {width, height};
  NppStatus status = nppiRGBToHSV_8u_AC4R(src_mem.get(), src_mem.step(), dst_mem.get(), dst_mem.step(), roi);
  ASSERT_EQ(status, NPP_NO_ERROR);

  std::vector<Npp8u> dst;
  dst_mem.copyToHost(dst);
  for (int i = 0; i < width * height; ++i) {
    Npp8u h, s, v;
    rgb_to_hsv_ref(pixels[i].r, pixels[i].g, pixels[i].b, h, s, v);
    EXPECT_NEAR(dst[i * 4 + 0], h, 2);
    EXPECT_NEAR(dst[i * 4 + 1], s, 2);
    EXPECT_NEAR(dst[i * 4 + 2], v, 2);
    EXPECT_EQ(dst[i * 4 + 3], 0);
  }
}

TEST_F(HSVConversionTest, HSVToRGB_8u_AC4R_AlphaCleared) {
  const int width = 2;
  const int height = 2;
  std::vector<RgbPixel> pixels = {
      {10, 30, 200},
      {120, 200, 40},
      {200, 50, 50},
      {0, 0, 0},
  };

  std::vector<Npp8u> src(width * height * 4);
  for (int i = 0; i < width * height; ++i) {
    Npp8u h, s, v;
    rgb_to_hsv_ref(pixels[i].r, pixels[i].g, pixels[i].b, h, s, v);
    src[i * 4 + 0] = h;
    src[i * 4 + 1] = s;
    src[i * 4 + 2] = v;
    src[i * 4 + 3] = static_cast<Npp8u>(5 + i * 20);
  }

  NppImageMemory<Npp8u> src_mem(width, height, 4);
  NppImageMemory<Npp8u> dst_mem(width, height, 4);
  src_mem.copyFromHost(src);

  NppiSize roi = {width, height};
  NppStatus status = nppiHSVToRGB_8u_AC4R(src_mem.get(), src_mem.step(), dst_mem.get(), dst_mem.step(), roi);
  ASSERT_EQ(status, NPP_NO_ERROR);

  std::vector<Npp8u> dst;
  dst_mem.copyToHost(dst);
  for (int i = 0; i < width * height; ++i) {
    EXPECT_NEAR(dst[i * 4 + 0], pixels[i].r, 3);
    EXPECT_NEAR(dst[i * 4 + 1], pixels[i].g, 3);
    EXPECT_NEAR(dst[i * 4 + 2], pixels[i].b, 3);
    EXPECT_EQ(dst[i * 4 + 3], 0);
  }
}

TEST_P(HSVConversionCtxParamTest, RGBToHSV_C3R_CtxMatches) {
  const auto param = GetParam();
  std::vector<Npp8u> src;
  fill_random_u8(src, param.width * param.height * 3, param.seed);

  NppImageMemory<Npp8u> src_mem(param.width, param.height, 3);
  NppImageMemory<Npp8u> dst_mem(param.width, param.height, 3);
  NppImageMemory<Npp8u> ctx_mem(param.width, param.height, 3);
  src_mem.copyFromHost(src);

  NppiSize roi = {param.width, param.height};
  NppStatus status = nppiRGBToHSV_8u_C3R(src_mem.get(), src_mem.step(), dst_mem.get(), dst_mem.step(), roi);
  ASSERT_EQ(status, NPP_NO_ERROR);

  NppStreamContext ctx{};
  nppGetStreamContext(&ctx);
  ctx.hStream = 0;
  status = nppiRGBToHSV_8u_C3R_Ctx(src_mem.get(), src_mem.step(), ctx_mem.get(), ctx_mem.step(), roi, ctx);
  ASSERT_EQ(status, NPP_NO_ERROR);

  std::vector<Npp8u> dst;
  std::vector<Npp8u> ctx_out;
  dst_mem.copyToHost(dst);
  ctx_mem.copyToHost(ctx_out);
  expect_equal_u8(dst, ctx_out, 2);
}

TEST_P(HSVConversionCtxParamTest, HSVToRGB_C3R_CtxMatches) {
  const auto param = GetParam();
  std::vector<Npp8u> src;
  fill_random_u8(src, param.width * param.height * 3, param.seed + 17);

  NppImageMemory<Npp8u> src_mem(param.width, param.height, 3);
  NppImageMemory<Npp8u> dst_mem(param.width, param.height, 3);
  NppImageMemory<Npp8u> ctx_mem(param.width, param.height, 3);
  src_mem.copyFromHost(src);

  NppiSize roi = {param.width, param.height};
  NppStatus status = nppiHSVToRGB_8u_C3R(src_mem.get(), src_mem.step(), dst_mem.get(), dst_mem.step(), roi);
  ASSERT_EQ(status, NPP_NO_ERROR);

  NppStreamContext ctx{};
  nppGetStreamContext(&ctx);
  ctx.hStream = 0;
  status = nppiHSVToRGB_8u_C3R_Ctx(src_mem.get(), src_mem.step(), ctx_mem.get(), ctx_mem.step(), roi, ctx);
  ASSERT_EQ(status, NPP_NO_ERROR);

  std::vector<Npp8u> dst;
  std::vector<Npp8u> ctx_out;
  dst_mem.copyToHost(dst);
  ctx_mem.copyToHost(ctx_out);
  expect_equal_u8(dst, ctx_out, 2);
}

TEST_P(HSVConversionCtxParamTest, RGBToHSV_AC4R_CtxMatches) {
  const auto param = GetParam();
  std::vector<Npp8u> src;
  fill_random_u8(src, param.width * param.height * 4, param.seed + 33);

  NppImageMemory<Npp8u> src_mem(param.width, param.height, 4);
  NppImageMemory<Npp8u> dst_mem(param.width, param.height, 4);
  NppImageMemory<Npp8u> ctx_mem(param.width, param.height, 4);
  src_mem.copyFromHost(src);

  NppiSize roi = {param.width, param.height};
  NppStatus status = nppiRGBToHSV_8u_AC4R(src_mem.get(), src_mem.step(), dst_mem.get(), dst_mem.step(), roi);
  ASSERT_EQ(status, NPP_NO_ERROR);

  NppStreamContext ctx{};
  nppGetStreamContext(&ctx);
  ctx.hStream = 0;
  status = nppiRGBToHSV_8u_AC4R_Ctx(src_mem.get(), src_mem.step(), ctx_mem.get(), ctx_mem.step(), roi, ctx);
  ASSERT_EQ(status, NPP_NO_ERROR);

  std::vector<Npp8u> dst;
  std::vector<Npp8u> ctx_out;
  dst_mem.copyToHost(dst);
  ctx_mem.copyToHost(ctx_out);
  expect_equal_u8(dst, ctx_out, 2);
}

TEST_P(HSVConversionCtxParamTest, HSVToRGB_AC4R_CtxMatches) {
  const auto param = GetParam();
  std::vector<Npp8u> src;
  fill_random_u8(src, param.width * param.height * 4, param.seed + 49);

  NppImageMemory<Npp8u> src_mem(param.width, param.height, 4);
  NppImageMemory<Npp8u> dst_mem(param.width, param.height, 4);
  NppImageMemory<Npp8u> ctx_mem(param.width, param.height, 4);
  src_mem.copyFromHost(src);

  NppiSize roi = {param.width, param.height};
  NppStatus status = nppiHSVToRGB_8u_AC4R(src_mem.get(), src_mem.step(), dst_mem.get(), dst_mem.step(), roi);
  ASSERT_EQ(status, NPP_NO_ERROR);

  NppStreamContext ctx{};
  nppGetStreamContext(&ctx);
  ctx.hStream = 0;
  status = nppiHSVToRGB_8u_AC4R_Ctx(src_mem.get(), src_mem.step(), ctx_mem.get(), ctx_mem.step(), roi, ctx);
  ASSERT_EQ(status, NPP_NO_ERROR);

  std::vector<Npp8u> dst;
  std::vector<Npp8u> ctx_out;
  dst_mem.copyToHost(dst);
  ctx_mem.copyToHost(ctx_out);
  expect_equal_u8(dst, ctx_out, 2);
}

INSTANTIATE_TEST_SUITE_P(FunctionalCases, HSVConversionCtxParamTest,
                         ::testing::Values(HsvCtxCase{4, 3, 101}, HsvCtxCase{8, 5, 202}));
INSTANTIATE_TEST_SUITE_P(PrecisionCases, HSVConversionCtxParamTest,
                         ::testing::Values(HsvCtxCase{64, 32, 303}, HsvCtxCase{128, 16, 404}));
