#include "npp_test_base.h"
#include <vector>

using namespace npp_functional_test;

namespace {
struct RgbPixel {
  Npp8u r;
  Npp8u g;
  Npp8u b;
};

static inline Npp8u clamp_to_u8(float v) {
  if (v < 0.0f) {
    v = 0.0f;
  } else if (v > 255.0f) {
    v = 255.0f;
  }
  return static_cast<Npp8u>(v);
}

static inline void rgb_to_ycbcr_ref(Npp8u r, Npp8u g, Npp8u b, Npp8u &y, Npp8u &cb, Npp8u &cr) {
  double yf = 0.257 * r + 0.504 * g + 0.098 * b + 16.0;
  double cbf = -0.148 * r - 0.291 * g + 0.439 * b + 128.0;
  double crf = 0.439 * r - 0.368 * g - 0.071 * b + 128.0;

  y = clamp_to_u8(static_cast<float>(yf));
  cb = clamp_to_u8(static_cast<float>(cbf));
  cr = clamp_to_u8(static_cast<float>(crf));
}

static inline void bgr_to_ycbcr_ref(Npp8u b, Npp8u g, Npp8u r, Npp8u &y, Npp8u &cb, Npp8u &cr) {
  rgb_to_ycbcr_ref(r, g, b, y, cb, cr);
}

static void build_rgb_inputs(std::vector<Npp8u> &packed, std::vector<Npp8u> &r, std::vector<Npp8u> &g,
                             std::vector<Npp8u> &b, int width, int height, std::vector<RgbPixel> &pixels) {
  pixels = {
      {0, 0, 0},
      {255, 255, 255},
      {255, 0, 0},
      {0, 0, 255},
  };
  packed.resize(width * height * 3);
  r.resize(width * height);
  g.resize(width * height);
  b.resize(width * height);
  for (int i = 0; i < width * height; ++i) {
    const RgbPixel &p = pixels[i % pixels.size()];
    packed[i * 3 + 0] = p.r;
    packed[i * 3 + 1] = p.g;
    packed[i * 3 + 2] = p.b;
    r[i] = p.r;
    g[i] = p.g;
    b[i] = p.b;
  }
}

static void build_bgr_inputs(std::vector<Npp8u> &packed, std::vector<Npp8u> &b, std::vector<Npp8u> &g,
                             std::vector<Npp8u> &r, int width, int height, std::vector<RgbPixel> &pixels) {
  pixels = {
      {0, 0, 0},
      {255, 255, 255},
      {255, 0, 0},
      {0, 0, 255},
  };
  packed.resize(width * height * 3);
  b.resize(width * height);
  g.resize(width * height);
  r.resize(width * height);
  for (int i = 0; i < width * height; ++i) {
    const RgbPixel &p = pixels[i % pixels.size()];
    packed[i * 3 + 0] = p.b;
    packed[i * 3 + 1] = p.g;
    packed[i * 3 + 2] = p.r;
    b[i] = p.b;
    g[i] = p.g;
    r[i] = p.r;
  }
}
} // namespace

class RGBToYCbCrVariantsTest : public NppTestBase {};

TEST_F(RGBToYCbCrVariantsTest, RGBToYCbCr_8u_C3R_Reference) {
  const int width = 2;
  const int height = 2;
  std::vector<Npp8u> src;
  std::vector<Npp8u> r_plane, g_plane, b_plane;
  std::vector<RgbPixel> pixels;
  build_rgb_inputs(src, r_plane, g_plane, b_plane, width, height, pixels);

  NppImageMemory<Npp8u> src_mem(width, height, 3);
  NppImageMemory<Npp8u> dst_mem(width, height, 3);
  src_mem.copyFromHost(src);

  NppiSize roi = {width, height};
  NppStatus status = nppiRGBToYCbCr_8u_C3R(src_mem.get(), src_mem.step(), dst_mem.get(), dst_mem.step(), roi);
  ASSERT_EQ(status, NPP_NO_ERROR);

  std::vector<Npp8u> dst;
  dst_mem.copyToHost(dst);
  for (int i = 0; i < width * height; ++i) {
    Npp8u y, cb, cr;
    rgb_to_ycbcr_ref(pixels[i].r, pixels[i].g, pixels[i].b, y, cb, cr);
    int idx = i * 3;
    ASSERT_EQ(dst[idx + 0], y);
    ASSERT_EQ(dst[idx + 1], cb);
    ASSERT_EQ(dst[idx + 2], cr);
  }

  NppStreamContext ctx{};
  nppGetStreamContext(&ctx);
  ctx.hStream = 0;
  NppImageMemory<Npp8u> ctx_mem(width, height, 3);
  status = nppiRGBToYCbCr_8u_C3R_Ctx(src_mem.get(), src_mem.step(), ctx_mem.get(), ctx_mem.step(), roi, ctx);
  ASSERT_EQ(status, NPP_NO_ERROR);

  std::vector<Npp8u> ctx_out;
  ctx_mem.copyToHost(ctx_out);
  EXPECT_EQ(ctx_out, dst);
}

TEST_F(RGBToYCbCrVariantsTest, RGBToYCbCr_8u_AC4R_AlphaBehavior) {
  const int width = 2;
  const int height = 2;
  std::vector<Npp8u> src(width * height * 4);
  std::vector<RgbPixel> pixels = {
      {12, 34, 56},
      {78, 90, 123},
      {200, 10, 40},
      {0, 255, 128},
  };
  for (int i = 0; i < width * height; ++i) {
    const RgbPixel &p = pixels[i];
    src[i * 4 + 0] = p.r;
    src[i * 4 + 1] = p.g;
    src[i * 4 + 2] = p.b;
    src[i * 4 + 3] = static_cast<Npp8u>(11 + i * 23);
  }

  NppImageMemory<Npp8u> src_mem(width, height, 4);
  NppImageMemory<Npp8u> dst_mem(width, height, 4);
  src_mem.copyFromHost(src);

  NppiSize roi = {width, height};
  NppStatus status = nppiRGBToYCbCr_8u_AC4R(src_mem.get(), src_mem.step(), dst_mem.get(), dst_mem.step(), roi);
  ASSERT_EQ(status, NPP_NO_ERROR);

  std::vector<Npp8u> dst;
  dst_mem.copyToHost(dst);
  for (int i = 0; i < width * height; ++i) {
    Npp8u y, cb, cr;
    rgb_to_ycbcr_ref(pixels[i].r, pixels[i].g, pixels[i].b, y, cb, cr);
    int idx = i * 4;
    ASSERT_EQ(dst[idx + 0], y);
    ASSERT_EQ(dst[idx + 1], cb);
    ASSERT_EQ(dst[idx + 2], cr);
    // Alpha behavior is validated against NVIDIA NPP.
    ASSERT_EQ(dst[idx + 3], 0);
  }

  NppStreamContext ctx{};
  nppGetStreamContext(&ctx);
  ctx.hStream = 0;
  NppImageMemory<Npp8u> ctx_mem(width, height, 4);
  status = nppiRGBToYCbCr_8u_AC4R_Ctx(src_mem.get(), src_mem.step(), ctx_mem.get(), ctx_mem.step(), roi, ctx);
  ASSERT_EQ(status, NPP_NO_ERROR);

  std::vector<Npp8u> ctx_out;
  ctx_mem.copyToHost(ctx_out);
  EXPECT_EQ(ctx_out, dst);
}

TEST_F(RGBToYCbCrVariantsTest, RGBToYCbCr_8u_C3P3R_Reference) {
  const int width = 2;
  const int height = 2;
  std::vector<Npp8u> src;
  std::vector<Npp8u> r_plane, g_plane, b_plane;
  std::vector<RgbPixel> pixels;
  build_rgb_inputs(src, r_plane, g_plane, b_plane, width, height, pixels);

  NppImageMemory<Npp8u> src_mem(width, height, 3);
  NppImageMemory<Npp8u> y_mem(width, height);
  NppImageMemory<Npp8u> cb_mem(width, height);
  NppImageMemory<Npp8u> cr_mem(width, height);
  src_mem.copyFromHost(src);

  Npp8u *dst_planes[3] = {y_mem.get(), cb_mem.get(), cr_mem.get()};
  NppiSize roi = {width, height};
  NppStatus status = nppiRGBToYCbCr_8u_C3P3R(src_mem.get(), src_mem.step(), dst_planes, y_mem.step(), roi);
  ASSERT_EQ(status, NPP_NO_ERROR);

  std::vector<Npp8u> y_plane, cb_plane, cr_plane;
  y_mem.copyToHost(y_plane);
  cb_mem.copyToHost(cb_plane);
  cr_mem.copyToHost(cr_plane);

  for (int i = 0; i < width * height; ++i) {
    Npp8u y, cb, cr;
    rgb_to_ycbcr_ref(pixels[i].r, pixels[i].g, pixels[i].b, y, cb, cr);
    ASSERT_EQ(y_plane[i], y);
    ASSERT_EQ(cb_plane[i], cb);
    ASSERT_EQ(cr_plane[i], cr);
  }

  NppStreamContext ctx{};
  nppGetStreamContext(&ctx);
  ctx.hStream = 0;
  NppImageMemory<Npp8u> y_ctx(width, height);
  NppImageMemory<Npp8u> cb_ctx(width, height);
  NppImageMemory<Npp8u> cr_ctx(width, height);
  Npp8u *ctx_planes[3] = {y_ctx.get(), cb_ctx.get(), cr_ctx.get()};
  status = nppiRGBToYCbCr_8u_C3P3R_Ctx(src_mem.get(), src_mem.step(), ctx_planes, y_ctx.step(), roi, ctx);
  ASSERT_EQ(status, NPP_NO_ERROR);

  std::vector<Npp8u> y_plane_ctx, cb_plane_ctx, cr_plane_ctx;
  y_ctx.copyToHost(y_plane_ctx);
  cb_ctx.copyToHost(cb_plane_ctx);
  cr_ctx.copyToHost(cr_plane_ctx);

  EXPECT_EQ(y_plane_ctx, y_plane);
  EXPECT_EQ(cb_plane_ctx, cb_plane);
  EXPECT_EQ(cr_plane_ctx, cr_plane);
}

TEST_F(RGBToYCbCrVariantsTest, RGBToYCbCr_8u_P3R_Reference) {
  const int width = 2;
  const int height = 2;
  std::vector<Npp8u> src;
  std::vector<Npp8u> r_plane, g_plane, b_plane;
  std::vector<RgbPixel> pixels;
  build_rgb_inputs(src, r_plane, g_plane, b_plane, width, height, pixels);

  NppImageMemory<Npp8u> r_mem(width, height);
  NppImageMemory<Npp8u> g_mem(width, height);
  NppImageMemory<Npp8u> b_mem(width, height);
  NppImageMemory<Npp8u> y_mem(width, height);
  NppImageMemory<Npp8u> cb_mem(width, height);
  NppImageMemory<Npp8u> cr_mem(width, height);
  r_mem.copyFromHost(r_plane);
  g_mem.copyFromHost(g_plane);
  b_mem.copyFromHost(b_plane);

  const Npp8u *src_planes[3] = {r_mem.get(), g_mem.get(), b_mem.get()};
  Npp8u *dst_planes[3] = {y_mem.get(), cb_mem.get(), cr_mem.get()};
  NppiSize roi = {width, height};
  NppStatus status = nppiRGBToYCbCr_8u_P3R(src_planes, r_mem.step(), dst_planes, y_mem.step(), roi);
  ASSERT_EQ(status, NPP_NO_ERROR);

  std::vector<Npp8u> y_plane, cb_plane, cr_plane;
  y_mem.copyToHost(y_plane);
  cb_mem.copyToHost(cb_plane);
  cr_mem.copyToHost(cr_plane);

  for (int i = 0; i < width * height; ++i) {
    Npp8u y, cb, cr;
    rgb_to_ycbcr_ref(pixels[i].r, pixels[i].g, pixels[i].b, y, cb, cr);
    ASSERT_EQ(y_plane[i], y);
    ASSERT_EQ(cb_plane[i], cb);
    ASSERT_EQ(cr_plane[i], cr);
  }

  NppStreamContext ctx{};
  nppGetStreamContext(&ctx);
  ctx.hStream = 0;
  NppImageMemory<Npp8u> y_ctx(width, height);
  NppImageMemory<Npp8u> cb_ctx(width, height);
  NppImageMemory<Npp8u> cr_ctx(width, height);
  Npp8u *ctx_planes[3] = {y_ctx.get(), cb_ctx.get(), cr_ctx.get()};
  status = nppiRGBToYCbCr_8u_P3R_Ctx(src_planes, r_mem.step(), ctx_planes, y_ctx.step(), roi, ctx);
  ASSERT_EQ(status, NPP_NO_ERROR);

  std::vector<Npp8u> y_plane_ctx, cb_plane_ctx, cr_plane_ctx;
  y_ctx.copyToHost(y_plane_ctx);
  cb_ctx.copyToHost(cb_plane_ctx);
  cr_ctx.copyToHost(cr_plane_ctx);

  EXPECT_EQ(y_plane_ctx, y_plane);
  EXPECT_EQ(cb_plane_ctx, cb_plane);
  EXPECT_EQ(cr_plane_ctx, cr_plane);
}

TEST_F(RGBToYCbCrVariantsTest, RGBToYCbCr_8u_AC4P3R_AlphaIgnored) {
  const int width = 2;
  const int height = 2;
  std::vector<RgbPixel> pixels = {
      {12, 34, 56},
      {78, 90, 123},
      {200, 10, 40},
      {0, 255, 128},
  };
  std::vector<Npp8u> src(width * height * 4);
  for (int i = 0; i < width * height; ++i) {
    const RgbPixel &p = pixels[i];
    src[i * 4 + 0] = p.r;
    src[i * 4 + 1] = p.g;
    src[i * 4 + 2] = p.b;
    src[i * 4 + 3] = static_cast<Npp8u>(5 + i * 19);
  }

  NppImageMemory<Npp8u> src_mem(width, height, 4);
  NppImageMemory<Npp8u> y_mem(width, height);
  NppImageMemory<Npp8u> cb_mem(width, height);
  NppImageMemory<Npp8u> cr_mem(width, height);
  src_mem.copyFromHost(src);

  Npp8u *dst_planes[3] = {y_mem.get(), cb_mem.get(), cr_mem.get()};
  NppiSize roi = {width, height};
  NppStatus status = nppiRGBToYCbCr_8u_AC4P3R(src_mem.get(), src_mem.step(), dst_planes, y_mem.step(), roi);
  ASSERT_EQ(status, NPP_NO_ERROR);

  std::vector<Npp8u> y_plane, cb_plane, cr_plane;
  y_mem.copyToHost(y_plane);
  cb_mem.copyToHost(cb_plane);
  cr_mem.copyToHost(cr_plane);

  for (int i = 0; i < width * height; ++i) {
    Npp8u y, cb, cr;
    rgb_to_ycbcr_ref(pixels[i].r, pixels[i].g, pixels[i].b, y, cb, cr);
    ASSERT_EQ(y_plane[i], y);
    ASSERT_EQ(cb_plane[i], cb);
    ASSERT_EQ(cr_plane[i], cr);
  }

  NppStreamContext ctx{};
  nppGetStreamContext(&ctx);
  ctx.hStream = 0;
  NppImageMemory<Npp8u> y_ctx(width, height);
  NppImageMemory<Npp8u> cb_ctx(width, height);
  NppImageMemory<Npp8u> cr_ctx(width, height);
  Npp8u *ctx_planes[3] = {y_ctx.get(), cb_ctx.get(), cr_ctx.get()};
  status = nppiRGBToYCbCr_8u_AC4P3R_Ctx(src_mem.get(), src_mem.step(), ctx_planes, y_ctx.step(), roi, ctx);
  ASSERT_EQ(status, NPP_NO_ERROR);

  std::vector<Npp8u> y_plane_ctx, cb_plane_ctx, cr_plane_ctx;
  y_ctx.copyToHost(y_plane_ctx);
  cb_ctx.copyToHost(cb_plane_ctx);
  cr_ctx.copyToHost(cr_plane_ctx);

  EXPECT_EQ(y_plane_ctx, y_plane);
  EXPECT_EQ(cb_plane_ctx, cb_plane);
  EXPECT_EQ(cr_plane_ctx, cr_plane);
}

TEST_F(RGBToYCbCrVariantsTest, BGRToYCbCr_8u_C3P3R_Reference) {
  const int width = 2;
  const int height = 2;
  std::vector<Npp8u> src;
  std::vector<Npp8u> b_plane, g_plane, r_plane;
  std::vector<RgbPixel> pixels;
  build_bgr_inputs(src, b_plane, g_plane, r_plane, width, height, pixels);

  NppImageMemory<Npp8u> src_mem(width, height, 3);
  NppImageMemory<Npp8u> y_mem(width, height);
  NppImageMemory<Npp8u> cb_mem(width, height);
  NppImageMemory<Npp8u> cr_mem(width, height);
  src_mem.copyFromHost(src);

  Npp8u *dst_planes[3] = {y_mem.get(), cb_mem.get(), cr_mem.get()};
  NppiSize roi = {width, height};
  NppStatus status = nppiBGRToYCbCr_8u_C3P3R(src_mem.get(), src_mem.step(), dst_planes, y_mem.step(), roi);
  ASSERT_EQ(status, NPP_NO_ERROR);

  std::vector<Npp8u> y_plane, cb_plane, cr_plane;
  y_mem.copyToHost(y_plane);
  cb_mem.copyToHost(cb_plane);
  cr_mem.copyToHost(cr_plane);

  for (int i = 0; i < width * height; ++i) {
    Npp8u y, cb, cr;
    bgr_to_ycbcr_ref(pixels[i].b, pixels[i].g, pixels[i].r, y, cb, cr);
    ASSERT_EQ(y_plane[i], y);
    ASSERT_EQ(cb_plane[i], cb);
    ASSERT_EQ(cr_plane[i], cr);
  }

  NppStreamContext ctx{};
  nppGetStreamContext(&ctx);
  ctx.hStream = 0;
  NppImageMemory<Npp8u> y_ctx(width, height);
  NppImageMemory<Npp8u> cb_ctx(width, height);
  NppImageMemory<Npp8u> cr_ctx(width, height);
  Npp8u *ctx_planes[3] = {y_ctx.get(), cb_ctx.get(), cr_ctx.get()};
  status = nppiBGRToYCbCr_8u_C3P3R_Ctx(src_mem.get(), src_mem.step(), ctx_planes, y_ctx.step(), roi, ctx);
  ASSERT_EQ(status, NPP_NO_ERROR);

  std::vector<Npp8u> y_plane_ctx, cb_plane_ctx, cr_plane_ctx;
  y_ctx.copyToHost(y_plane_ctx);
  cb_ctx.copyToHost(cb_plane_ctx);
  cr_ctx.copyToHost(cr_plane_ctx);

  EXPECT_EQ(y_plane_ctx, y_plane);
  EXPECT_EQ(cb_plane_ctx, cb_plane);
  EXPECT_EQ(cr_plane_ctx, cr_plane);
}

TEST_F(RGBToYCbCrVariantsTest, BGRToYCbCr_8u_AC4P3R_Reference) {
  const int width = 2;
  const int height = 2;
  std::vector<RgbPixel> pixels = {
      {12, 34, 56},
      {78, 90, 123},
      {200, 10, 40},
      {0, 255, 128},
  };
  std::vector<Npp8u> src(width * height * 4);
  for (int i = 0; i < width * height; ++i) {
    const RgbPixel &p = pixels[i];
    src[i * 4 + 0] = p.b;
    src[i * 4 + 1] = p.g;
    src[i * 4 + 2] = p.r;
    src[i * 4 + 3] = static_cast<Npp8u>(9 + i * 17);
  }

  NppImageMemory<Npp8u> src_mem(width, height, 4);
  NppImageMemory<Npp8u> y_mem(width, height);
  NppImageMemory<Npp8u> cb_mem(width, height);
  NppImageMemory<Npp8u> cr_mem(width, height);
  src_mem.copyFromHost(src);

  Npp8u *dst_planes[3] = {y_mem.get(), cb_mem.get(), cr_mem.get()};
  NppiSize roi = {width, height};
  NppStatus status = nppiBGRToYCbCr_8u_AC4P3R(src_mem.get(), src_mem.step(), dst_planes, y_mem.step(), roi);
  ASSERT_EQ(status, NPP_NO_ERROR);

  std::vector<Npp8u> y_plane, cb_plane, cr_plane;
  y_mem.copyToHost(y_plane);
  cb_mem.copyToHost(cb_plane);
  cr_mem.copyToHost(cr_plane);

  for (int i = 0; i < width * height; ++i) {
    Npp8u y, cb, cr;
    bgr_to_ycbcr_ref(pixels[i].b, pixels[i].g, pixels[i].r, y, cb, cr);
    ASSERT_EQ(y_plane[i], y);
    ASSERT_EQ(cb_plane[i], cb);
    ASSERT_EQ(cr_plane[i], cr);
  }

  NppStreamContext ctx{};
  nppGetStreamContext(&ctx);
  ctx.hStream = 0;
  NppImageMemory<Npp8u> y_ctx(width, height);
  NppImageMemory<Npp8u> cb_ctx(width, height);
  NppImageMemory<Npp8u> cr_ctx(width, height);
  Npp8u *ctx_planes[3] = {y_ctx.get(), cb_ctx.get(), cr_ctx.get()};
  status = nppiBGRToYCbCr_8u_AC4P3R_Ctx(src_mem.get(), src_mem.step(), ctx_planes, y_ctx.step(), roi, ctx);
  ASSERT_EQ(status, NPP_NO_ERROR);

  std::vector<Npp8u> y_plane_ctx, cb_plane_ctx, cr_plane_ctx;
  y_ctx.copyToHost(y_plane_ctx);
  cb_ctx.copyToHost(cb_plane_ctx);
  cr_ctx.copyToHost(cr_plane_ctx);

  EXPECT_EQ(y_plane_ctx, y_plane);
  EXPECT_EQ(cb_plane_ctx, cb_plane);
  EXPECT_EQ(cr_plane_ctx, cr_plane);
}

TEST_F(RGBToYCbCrVariantsTest, BGRToYCbCr_8u_AC4P4R_AlphaPreserved) {
  const int width = 2;
  const int height = 2;
  std::vector<RgbPixel> pixels = {
      {12, 34, 56},
      {78, 90, 123},
      {200, 10, 40},
      {0, 255, 128},
  };
  std::vector<Npp8u> src(width * height * 4);
  std::vector<Npp8u> alpha(width * height);
  for (int i = 0; i < width * height; ++i) {
    const RgbPixel &p = pixels[i];
    src[i * 4 + 0] = p.b;
    src[i * 4 + 1] = p.g;
    src[i * 4 + 2] = p.r;
    src[i * 4 + 3] = static_cast<Npp8u>(3 + i * 29);
    alpha[i] = src[i * 4 + 3];
  }

  NppImageMemory<Npp8u> src_mem(width, height, 4);
  NppImageMemory<Npp8u> y_mem(width, height);
  NppImageMemory<Npp8u> cb_mem(width, height);
  NppImageMemory<Npp8u> cr_mem(width, height);
  NppImageMemory<Npp8u> a_mem(width, height);
  src_mem.copyFromHost(src);

  Npp8u *dst_planes[4] = {y_mem.get(), cb_mem.get(), cr_mem.get(), a_mem.get()};
  NppiSize roi = {width, height};
  NppStatus status = nppiBGRToYCbCr_8u_AC4P4R(src_mem.get(), src_mem.step(), dst_planes, y_mem.step(), roi);
  ASSERT_EQ(status, NPP_NO_ERROR);

  std::vector<Npp8u> y_plane, cb_plane, cr_plane, a_plane;
  y_mem.copyToHost(y_plane);
  cb_mem.copyToHost(cb_plane);
  cr_mem.copyToHost(cr_plane);
  a_mem.copyToHost(a_plane);

  for (int i = 0; i < width * height; ++i) {
    Npp8u y, cb, cr;
    bgr_to_ycbcr_ref(pixels[i].b, pixels[i].g, pixels[i].r, y, cb, cr);
    ASSERT_EQ(y_plane[i], y);
    ASSERT_EQ(cb_plane[i], cb);
    ASSERT_EQ(cr_plane[i], cr);
    ASSERT_EQ(a_plane[i], alpha[i]);
  }

  NppStreamContext ctx{};
  nppGetStreamContext(&ctx);
  ctx.hStream = 0;
  NppImageMemory<Npp8u> y_ctx(width, height);
  NppImageMemory<Npp8u> cb_ctx(width, height);
  NppImageMemory<Npp8u> cr_ctx(width, height);
  NppImageMemory<Npp8u> a_ctx(width, height);
  Npp8u *ctx_planes[4] = {y_ctx.get(), cb_ctx.get(), cr_ctx.get(), a_ctx.get()};
  status = nppiBGRToYCbCr_8u_AC4P4R_Ctx(src_mem.get(), src_mem.step(), ctx_planes, y_ctx.step(), roi, ctx);
  ASSERT_EQ(status, NPP_NO_ERROR);

  std::vector<Npp8u> y_plane_ctx, cb_plane_ctx, cr_plane_ctx, a_plane_ctx;
  y_ctx.copyToHost(y_plane_ctx);
  cb_ctx.copyToHost(cb_plane_ctx);
  cr_ctx.copyToHost(cr_plane_ctx);
  a_ctx.copyToHost(a_plane_ctx);

  EXPECT_EQ(y_plane_ctx, y_plane);
  EXPECT_EQ(cb_plane_ctx, cb_plane);
  EXPECT_EQ(cr_plane_ctx, cr_plane);
  EXPECT_EQ(a_plane_ctx, a_plane);
}
