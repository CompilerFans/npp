#include "npp_test_base.h"
#include <vector>

using namespace npp_functional_test;

namespace {
struct YCbCrPixel {
  Npp8u y;
  Npp8u cb;
  Npp8u cr;
};

static inline Npp8u clamp_to_u8(float v) {
  if (v < 0.0f) {
    v = 0.0f;
  } else if (v > 255.0f) {
    v = 255.0f;
  }
  return static_cast<Npp8u>(v);
}

static inline void ycbcr_to_rgb_ref(Npp8u y, Npp8u cb, Npp8u cr, Npp8u &r, Npp8u &g, Npp8u &b) {
  float yf = 1.164f * (static_cast<float>(y) - 16.0f);
  float crf = static_cast<float>(cr) - 128.0f;
  float cbf = static_cast<float>(cb) - 128.0f;

  float rf = yf + 1.596f * crf;
  float gf = yf - 0.813f * crf - 0.392f * cbf;
  float bf = yf + 2.017f * cbf;

  r = clamp_to_u8(rf);
  g = clamp_to_u8(gf);
  b = clamp_to_u8(bf);
}

static void fill_ycbcr_inputs(std::vector<Npp8u> &packed, std::vector<Npp8u> &y_plane,
                              std::vector<Npp8u> &cb_plane, std::vector<Npp8u> &cr_plane, int width, int height,
                              std::vector<YCbCrPixel> &pixels) {
  pixels = {
      {16, 128, 128},
      {200, 90, 240},
      {50, 200, 40},
      {128, 64, 192},
  };
  packed.resize(width * height * 3);
  y_plane.resize(width * height);
  cb_plane.resize(width * height);
  cr_plane.resize(width * height);
  for (int i = 0; i < width * height; ++i) {
    const YCbCrPixel &p = pixels[i % pixels.size()];
    packed[i * 3 + 0] = p.y;
    packed[i * 3 + 1] = p.cb;
    packed[i * 3 + 2] = p.cr;
    y_plane[i] = p.y;
    cb_plane[i] = p.cb;
    cr_plane[i] = p.cr;
  }
}
} // namespace

class YCbCrToRGBVariantsTest : public NppTestBase {};

TEST_F(YCbCrToRGBVariantsTest, YCbCrToRGB_8u_C3R_Reference) {
  const int width = 2;
  const int height = 2;

  std::vector<Npp8u> src;
  std::vector<Npp8u> y_plane, cb_plane, cr_plane;
  std::vector<YCbCrPixel> pixels;
  fill_ycbcr_inputs(src, y_plane, cb_plane, cr_plane, width, height, pixels);

  NppImageMemory<Npp8u> src_mem(width, height, 3);
  NppImageMemory<Npp8u> dst_mem(width, height, 3);
  src_mem.copyFromHost(src);

  NppiSize roi = {width, height};
  NppStatus status = nppiYCbCrToRGB_8u_C3R(src_mem.get(), src_mem.step(), dst_mem.get(), dst_mem.step(), roi);
  ASSERT_EQ(status, NPP_NO_ERROR);

  std::vector<Npp8u> dst;
  dst_mem.copyToHost(dst);
  for (int i = 0; i < width * height; ++i) {
    Npp8u r, g, b;
    ycbcr_to_rgb_ref(pixels[i].y, pixels[i].cb, pixels[i].cr, r, g, b);
    int idx = i * 3;
    ASSERT_EQ(dst[idx + 0], r);
    ASSERT_EQ(dst[idx + 1], g);
    ASSERT_EQ(dst[idx + 2], b);
  }

  NppStreamContext ctx{};
  nppGetStreamContext(&ctx);
  ctx.hStream = 0;
  NppImageMemory<Npp8u> ctx_mem(width, height, 3);
  status = nppiYCbCrToRGB_8u_C3R_Ctx(src_mem.get(), src_mem.step(), ctx_mem.get(), ctx_mem.step(), roi, ctx);
  ASSERT_EQ(status, NPP_NO_ERROR);
  std::vector<Npp8u> ctx_out;
  ctx_mem.copyToHost(ctx_out);
  EXPECT_EQ(ctx_out, dst);
}

TEST_F(YCbCrToRGBVariantsTest, YCbCrToRGB_8u_AC4R_AlphaBehavior) {
  const int width = 2;
  const int height = 2;

  std::vector<Npp8u> src;
  std::vector<Npp8u> y_plane, cb_plane, cr_plane;
  std::vector<YCbCrPixel> pixels;
  fill_ycbcr_inputs(src, y_plane, cb_plane, cr_plane, width, height, pixels);

  std::vector<Npp8u> src_ac4(width * height * 4);
  for (int i = 0; i < width * height; ++i) {
    src_ac4[i * 4 + 0] = pixels[i].y;
    src_ac4[i * 4 + 1] = pixels[i].cb;
    src_ac4[i * 4 + 2] = pixels[i].cr;
    src_ac4[i * 4 + 3] = static_cast<Npp8u>(21 + i * 13);
  }

  NppImageMemory<Npp8u> src_mem(width, height, 4);
  NppImageMemory<Npp8u> dst_mem(width, height, 4);
  src_mem.copyFromHost(src_ac4);

  NppiSize roi = {width, height};
  NppStatus status = nppiYCbCrToRGB_8u_AC4R(src_mem.get(), src_mem.step(), dst_mem.get(), dst_mem.step(), roi);
  ASSERT_EQ(status, NPP_NO_ERROR);

  std::vector<Npp8u> dst;
  dst_mem.copyToHost(dst);
  for (int i = 0; i < width * height; ++i) {
    Npp8u r, g, b;
    ycbcr_to_rgb_ref(pixels[i].y, pixels[i].cb, pixels[i].cr, r, g, b);
    int idx = i * 4;
    ASSERT_EQ(dst[idx + 0], r);
    ASSERT_EQ(dst[idx + 1], g);
    ASSERT_EQ(dst[idx + 2], b);
    ASSERT_EQ(dst[idx + 3], 0);
  }

  NppStreamContext ctx{};
  nppGetStreamContext(&ctx);
  ctx.hStream = 0;
  NppImageMemory<Npp8u> ctx_mem(width, height, 4);
  status = nppiYCbCrToRGB_8u_AC4R_Ctx(src_mem.get(), src_mem.step(), ctx_mem.get(), ctx_mem.step(), roi, ctx);
  ASSERT_EQ(status, NPP_NO_ERROR);

  std::vector<Npp8u> ctx_out;
  ctx_mem.copyToHost(ctx_out);
  EXPECT_EQ(ctx_out, dst);
}

TEST_F(YCbCrToRGBVariantsTest, YCbCrToRGB_8u_P3C3R_Reference) {
  const int width = 2;
  const int height = 2;

  std::vector<Npp8u> src;
  std::vector<Npp8u> y_plane, cb_plane, cr_plane;
  std::vector<YCbCrPixel> pixels;
  fill_ycbcr_inputs(src, y_plane, cb_plane, cr_plane, width, height, pixels);

  NppImageMemory<Npp8u> y_mem(width, height);
  NppImageMemory<Npp8u> cb_mem(width, height);
  NppImageMemory<Npp8u> cr_mem(width, height);
  NppImageMemory<Npp8u> dst_mem(width, height, 3);
  y_mem.copyFromHost(y_plane);
  cb_mem.copyFromHost(cb_plane);
  cr_mem.copyFromHost(cr_plane);

  const Npp8u *src_planes[3] = {y_mem.get(), cb_mem.get(), cr_mem.get()};

  NppiSize roi = {width, height};
  NppStatus status = nppiYCbCrToRGB_8u_P3C3R(src_planes, y_mem.step(), dst_mem.get(), dst_mem.step(), roi);
  ASSERT_EQ(status, NPP_NO_ERROR);

  std::vector<Npp8u> dst;
  dst_mem.copyToHost(dst);
  for (int i = 0; i < width * height; ++i) {
    Npp8u r, g, b;
    ycbcr_to_rgb_ref(pixels[i].y, pixels[i].cb, pixels[i].cr, r, g, b);
    int idx = i * 3;
    ASSERT_EQ(dst[idx + 0], r);
    ASSERT_EQ(dst[idx + 1], g);
    ASSERT_EQ(dst[idx + 2], b);
  }

  NppStreamContext ctx{};
  nppGetStreamContext(&ctx);
  ctx.hStream = 0;
  NppImageMemory<Npp8u> ctx_mem(width, height, 3);
  status = nppiYCbCrToRGB_8u_P3C3R_Ctx(src_planes, y_mem.step(), ctx_mem.get(), ctx_mem.step(), roi, ctx);
  ASSERT_EQ(status, NPP_NO_ERROR);

  std::vector<Npp8u> ctx_out;
  ctx_mem.copyToHost(ctx_out);
  EXPECT_EQ(ctx_out, dst);
}

TEST_F(YCbCrToRGBVariantsTest, YCbCrToRGB_8u_P3R_Reference) {
  const int width = 2;
  const int height = 2;

  std::vector<Npp8u> src;
  std::vector<Npp8u> y_plane, cb_plane, cr_plane;
  std::vector<YCbCrPixel> pixels;
  fill_ycbcr_inputs(src, y_plane, cb_plane, cr_plane, width, height, pixels);

  NppImageMemory<Npp8u> y_mem(width, height);
  NppImageMemory<Npp8u> cb_mem(width, height);
  NppImageMemory<Npp8u> cr_mem(width, height);
  NppImageMemory<Npp8u> r_mem(width, height);
  NppImageMemory<Npp8u> g_mem(width, height);
  NppImageMemory<Npp8u> b_mem(width, height);
  y_mem.copyFromHost(y_plane);
  cb_mem.copyFromHost(cb_plane);
  cr_mem.copyFromHost(cr_plane);

  const Npp8u *src_planes[3] = {y_mem.get(), cb_mem.get(), cr_mem.get()};
  Npp8u *dst_planes[3] = {r_mem.get(), g_mem.get(), b_mem.get()};

  NppiSize roi = {width, height};
  NppStatus status = nppiYCbCrToRGB_8u_P3R(src_planes, y_mem.step(), dst_planes, r_mem.step(), roi);
  ASSERT_EQ(status, NPP_NO_ERROR);

  std::vector<Npp8u> r_plane, g_plane, b_plane;
  r_mem.copyToHost(r_plane);
  g_mem.copyToHost(g_plane);
  b_mem.copyToHost(b_plane);

  for (int i = 0; i < width * height; ++i) {
    Npp8u r, g, b;
    ycbcr_to_rgb_ref(pixels[i].y, pixels[i].cb, pixels[i].cr, r, g, b);
    ASSERT_EQ(r_plane[i], r);
    ASSERT_EQ(g_plane[i], g);
    ASSERT_EQ(b_plane[i], b);
  }

  NppStreamContext ctx{};
  nppGetStreamContext(&ctx);
  ctx.hStream = 0;
  NppImageMemory<Npp8u> r_ctx(width, height);
  NppImageMemory<Npp8u> g_ctx(width, height);
  NppImageMemory<Npp8u> b_ctx(width, height);
  Npp8u *ctx_planes[3] = {r_ctx.get(), g_ctx.get(), b_ctx.get()};
  status = nppiYCbCrToRGB_8u_P3R_Ctx(src_planes, y_mem.step(), ctx_planes, r_ctx.step(), roi, ctx);
  ASSERT_EQ(status, NPP_NO_ERROR);

  std::vector<Npp8u> r_plane_ctx, g_plane_ctx, b_plane_ctx;
  r_ctx.copyToHost(r_plane_ctx);
  g_ctx.copyToHost(g_plane_ctx);
  b_ctx.copyToHost(b_plane_ctx);

  EXPECT_EQ(r_plane_ctx, r_plane);
  EXPECT_EQ(g_plane_ctx, g_plane);
  EXPECT_EQ(b_plane_ctx, b_plane);
}

TEST_F(YCbCrToRGBVariantsTest, YCbCrToRGB_8u_P3C4R_AlphaConstant) {
  const int width = 2;
  const int height = 2;
  const Npp8u alpha = 77;

  std::vector<Npp8u> src;
  std::vector<Npp8u> y_plane, cb_plane, cr_plane;
  std::vector<YCbCrPixel> pixels;
  fill_ycbcr_inputs(src, y_plane, cb_plane, cr_plane, width, height, pixels);

  NppImageMemory<Npp8u> y_mem(width, height);
  NppImageMemory<Npp8u> cb_mem(width, height);
  NppImageMemory<Npp8u> cr_mem(width, height);
  NppImageMemory<Npp8u> dst_mem(width, height, 4);
  y_mem.copyFromHost(y_plane);
  cb_mem.copyFromHost(cb_plane);
  cr_mem.copyFromHost(cr_plane);

  const Npp8u *src_planes[3] = {y_mem.get(), cb_mem.get(), cr_mem.get()};

  NppiSize roi = {width, height};
  NppStatus status = nppiYCbCrToRGB_8u_P3C4R(src_planes, y_mem.step(), dst_mem.get(), dst_mem.step(), roi, alpha);
  ASSERT_EQ(status, NPP_NO_ERROR);

  std::vector<Npp8u> dst;
  dst_mem.copyToHost(dst);
  for (int i = 0; i < width * height; ++i) {
    Npp8u r, g, b;
    ycbcr_to_rgb_ref(pixels[i].y, pixels[i].cb, pixels[i].cr, r, g, b);
    int idx = i * 4;
    ASSERT_EQ(dst[idx + 0], r);
    ASSERT_EQ(dst[idx + 1], g);
    ASSERT_EQ(dst[idx + 2], b);
    ASSERT_EQ(dst[idx + 3], alpha);
  }

  NppStreamContext ctx{};
  nppGetStreamContext(&ctx);
  ctx.hStream = 0;
  NppImageMemory<Npp8u> ctx_mem(width, height, 4);
  status = nppiYCbCrToRGB_8u_P3C4R_Ctx(src_planes, y_mem.step(), ctx_mem.get(), ctx_mem.step(), roi, alpha, ctx);
  ASSERT_EQ(status, NPP_NO_ERROR);

  std::vector<Npp8u> ctx_out;
  ctx_mem.copyToHost(ctx_out);
  EXPECT_EQ(ctx_out, dst);
}

TEST_F(YCbCrToRGBVariantsTest, YCbCrToBGR_8u_P3C3R_Reference) {
  const int width = 2;
  const int height = 2;

  std::vector<Npp8u> src;
  std::vector<Npp8u> y_plane, cb_plane, cr_plane;
  std::vector<YCbCrPixel> pixels;
  fill_ycbcr_inputs(src, y_plane, cb_plane, cr_plane, width, height, pixels);

  NppImageMemory<Npp8u> y_mem(width, height);
  NppImageMemory<Npp8u> cb_mem(width, height);
  NppImageMemory<Npp8u> cr_mem(width, height);
  NppImageMemory<Npp8u> dst_mem(width, height, 3);
  y_mem.copyFromHost(y_plane);
  cb_mem.copyFromHost(cb_plane);
  cr_mem.copyFromHost(cr_plane);

  const Npp8u *src_planes[3] = {y_mem.get(), cb_mem.get(), cr_mem.get()};

  NppiSize roi = {width, height};
  NppStatus status = nppiYCbCrToBGR_8u_P3C3R(src_planes, y_mem.step(), dst_mem.get(), dst_mem.step(), roi);
  ASSERT_EQ(status, NPP_NO_ERROR);

  std::vector<Npp8u> dst;
  dst_mem.copyToHost(dst);
  for (int i = 0; i < width * height; ++i) {
    Npp8u r, g, b;
    ycbcr_to_rgb_ref(pixels[i].y, pixels[i].cb, pixels[i].cr, r, g, b);
    int idx = i * 3;
    ASSERT_EQ(dst[idx + 0], b);
    ASSERT_EQ(dst[idx + 1], g);
    ASSERT_EQ(dst[idx + 2], r);
  }

  NppStreamContext ctx{};
  nppGetStreamContext(&ctx);
  ctx.hStream = 0;
  NppImageMemory<Npp8u> ctx_mem(width, height, 3);
  status = nppiYCbCrToBGR_8u_P3C3R_Ctx(src_planes, y_mem.step(), ctx_mem.get(), ctx_mem.step(), roi, ctx);
  ASSERT_EQ(status, NPP_NO_ERROR);

  std::vector<Npp8u> ctx_out;
  ctx_mem.copyToHost(ctx_out);
  EXPECT_EQ(ctx_out, dst);
}

TEST_F(YCbCrToRGBVariantsTest, YCbCrToBGR_8u_P3C4R_AlphaConstant) {
  const int width = 2;
  const int height = 2;
  const Npp8u alpha = 93;

  std::vector<Npp8u> src;
  std::vector<Npp8u> y_plane, cb_plane, cr_plane;
  std::vector<YCbCrPixel> pixels;
  fill_ycbcr_inputs(src, y_plane, cb_plane, cr_plane, width, height, pixels);

  NppImageMemory<Npp8u> y_mem(width, height);
  NppImageMemory<Npp8u> cb_mem(width, height);
  NppImageMemory<Npp8u> cr_mem(width, height);
  NppImageMemory<Npp8u> dst_mem(width, height, 4);
  y_mem.copyFromHost(y_plane);
  cb_mem.copyFromHost(cb_plane);
  cr_mem.copyFromHost(cr_plane);

  const Npp8u *src_planes[3] = {y_mem.get(), cb_mem.get(), cr_mem.get()};

  NppiSize roi = {width, height};
  NppStatus status = nppiYCbCrToBGR_8u_P3C4R(src_planes, y_mem.step(), dst_mem.get(), dst_mem.step(), roi, alpha);
  ASSERT_EQ(status, NPP_NO_ERROR);

  std::vector<Npp8u> dst;
  dst_mem.copyToHost(dst);
  for (int i = 0; i < width * height; ++i) {
    Npp8u r, g, b;
    ycbcr_to_rgb_ref(pixels[i].y, pixels[i].cb, pixels[i].cr, r, g, b);
    int idx = i * 4;
    ASSERT_EQ(dst[idx + 0], b);
    ASSERT_EQ(dst[idx + 1], g);
    ASSERT_EQ(dst[idx + 2], r);
    ASSERT_EQ(dst[idx + 3], alpha);
  }

  NppStreamContext ctx{};
  nppGetStreamContext(&ctx);
  ctx.hStream = 0;
  NppImageMemory<Npp8u> ctx_mem(width, height, 4);
  status = nppiYCbCrToBGR_8u_P3C4R_Ctx(src_planes, y_mem.step(), ctx_mem.get(), ctx_mem.step(), roi, alpha, ctx);
  ASSERT_EQ(status, NPP_NO_ERROR);

  std::vector<Npp8u> ctx_out;
  ctx_mem.copyToHost(ctx_out);
  EXPECT_EQ(ctx_out, dst);
}
