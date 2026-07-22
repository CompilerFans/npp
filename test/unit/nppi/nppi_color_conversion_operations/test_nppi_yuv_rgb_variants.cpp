#include "npp_test_base.h"
#include <vector>

using namespace npp_functional_test;

namespace {
struct YuvPixel {
  Npp8u y;
  Npp8u u;
  Npp8u v;
};

static inline Npp8u clamp_to_u8(float v) {
  if (v < 0.0f) {
    v = 0.0f;
  } else if (v > 255.0f) {
    v = 255.0f;
  }
  return static_cast<Npp8u>(v);
}

static inline void yuv_to_rgb_ref(Npp8u y, Npp8u u, Npp8u v, Npp8u &r, Npp8u &g, Npp8u &b) {
  float yf = y;
  float uf = static_cast<float>(u) - 128.0f;
  float vf = static_cast<float>(v) - 128.0f;

  float rf = yf + 1.140f * vf;
  float gf = yf - 0.395f * uf - 0.581f * vf;
  float bf = yf + 2.032f * uf;

  r = clamp_to_u8(rf);
  g = clamp_to_u8(gf);
  b = clamp_to_u8(bf);
}

static void fill_yuv_inputs(std::vector<Npp8u> &packed, std::vector<Npp8u> &y_plane, std::vector<Npp8u> &u_plane,
                            std::vector<Npp8u> &v_plane, int width, int height, std::vector<YuvPixel> &pixels) {
  pixels = {
      {16, 128, 128},
      {200, 90, 240},
      {50, 200, 40},
      {128, 64, 192},
  };
  packed.resize(width * height * 3);
  y_plane.resize(width * height);
  u_plane.resize(width * height);
  v_plane.resize(width * height);
  for (int i = 0; i < width * height; ++i) {
    const YuvPixel &p = pixels[i % pixels.size()];
    packed[i * 3 + 0] = p.y;
    packed[i * 3 + 1] = p.u;
    packed[i * 3 + 2] = p.v;
    y_plane[i] = p.y;
    u_plane[i] = p.u;
    v_plane[i] = p.v;
  }
}
} // namespace

class YUVToRGBVariantsTest : public NppTestBase {};

TEST_F(YUVToRGBVariantsTest, YUVToRGB_8u_C3R_Reference) {
  const int width = 2;
  const int height = 2;

  std::vector<Npp8u> src;
  std::vector<Npp8u> y_plane, u_plane, v_plane;
  std::vector<YuvPixel> pixels;
  fill_yuv_inputs(src, y_plane, u_plane, v_plane, width, height, pixels);

  NppImageMemory<Npp8u> src_mem(width, height, 3);
  NppImageMemory<Npp8u> dst_mem(width, height, 3);
  src_mem.copyFromHost(src);

  NppiSize roi = {width, height};
  NppStatus status = nppiYUVToRGB_8u_C3R(src_mem.get(), src_mem.step(), dst_mem.get(), dst_mem.step(), roi);
  ASSERT_EQ(status, NPP_NO_ERROR);

  std::vector<Npp8u> dst;
  dst_mem.copyToHost(dst);

  for (int i = 0; i < width * height; ++i) {
    Npp8u r, g, b;
    yuv_to_rgb_ref(pixels[i].y, pixels[i].u, pixels[i].v, r, g, b);
    int idx = i * 3;
    ASSERT_EQ(dst[idx + 0], r);
    ASSERT_EQ(dst[idx + 1], g);
    ASSERT_EQ(dst[idx + 2], b);
  }

  NppStreamContext ctx{};
  nppGetStreamContext(&ctx);
  ctx.hStream = 0;
  NppImageMemory<Npp8u> ctx_mem(width, height, 3);
  status = nppiYUVToRGB_8u_C3R_Ctx(src_mem.get(), src_mem.step(), ctx_mem.get(), ctx_mem.step(), roi, ctx);
  ASSERT_EQ(status, NPP_NO_ERROR);
  std::vector<Npp8u> ctx_out;
  ctx_mem.copyToHost(ctx_out);
  EXPECT_EQ(ctx_out, dst);
}

TEST_F(YUVToRGBVariantsTest, YUVToRGB_8u_AC4R_AlphaCleared) {
  const int width = 2;
  const int height = 2;

  std::vector<Npp8u> src;
  std::vector<Npp8u> alpha(width * height);
  std::vector<Npp8u> y_plane, u_plane, v_plane;
  std::vector<YuvPixel> pixels;
  fill_yuv_inputs(src, y_plane, u_plane, v_plane, width, height, pixels);

  std::vector<Npp8u> src_ac4(width * height * 4);
  for (int i = 0; i < width * height; ++i) {
    src_ac4[i * 4 + 0] = pixels[i].y;
    src_ac4[i * 4 + 1] = pixels[i].u;
    src_ac4[i * 4 + 2] = pixels[i].v;
    src_ac4[i * 4 + 3] = static_cast<Npp8u>(17 + i * 31);
    alpha[i] = src_ac4[i * 4 + 3];
  }

  NppImageMemory<Npp8u> src_mem(width, height, 4);
  NppImageMemory<Npp8u> dst_mem(width, height, 4);
  src_mem.copyFromHost(src_ac4);

  NppiSize roi = {width, height};
  NppStatus status = nppiYUVToRGB_8u_AC4R(src_mem.get(), src_mem.step(), dst_mem.get(), dst_mem.step(), roi);
  ASSERT_EQ(status, NPP_NO_ERROR);

  std::vector<Npp8u> dst;
  dst_mem.copyToHost(dst);

  for (int i = 0; i < width * height; ++i) {
    Npp8u r, g, b;
    yuv_to_rgb_ref(pixels[i].y, pixels[i].u, pixels[i].v, r, g, b);
    int idx = i * 4;
    ASSERT_EQ(dst[idx + 0], r);
    ASSERT_EQ(dst[idx + 1], g);
    ASSERT_EQ(dst[idx + 2], b);
    // NVIDIA NPP clears alpha for AC4 packed output.
    ASSERT_EQ(dst[idx + 3], 0);
  }

  NppStreamContext ctx{};
  nppGetStreamContext(&ctx);
  ctx.hStream = 0;
  NppImageMemory<Npp8u> ctx_mem(width, height, 4);
  status = nppiYUVToRGB_8u_AC4R_Ctx(src_mem.get(), src_mem.step(), ctx_mem.get(), ctx_mem.step(), roi, ctx);
  ASSERT_EQ(status, NPP_NO_ERROR);

  std::vector<Npp8u> ctx_out;
  ctx_mem.copyToHost(ctx_out);
  EXPECT_EQ(ctx_out, dst);
}

TEST_F(YUVToRGBVariantsTest, YUVToRGB_8u_P3C3R_Reference) {
  const int width = 2;
  const int height = 2;

  std::vector<Npp8u> src;
  std::vector<Npp8u> y_plane, u_plane, v_plane;
  std::vector<YuvPixel> pixels;
  fill_yuv_inputs(src, y_plane, u_plane, v_plane, width, height, pixels);

  NppImageMemory<Npp8u> y_mem(width, height);
  NppImageMemory<Npp8u> u_mem(width, height);
  NppImageMemory<Npp8u> v_mem(width, height);
  NppImageMemory<Npp8u> dst_mem(width, height, 3);
  y_mem.copyFromHost(y_plane);
  u_mem.copyFromHost(u_plane);
  v_mem.copyFromHost(v_plane);

  const Npp8u *src_planes[3] = {y_mem.get(), u_mem.get(), v_mem.get()};

  NppiSize roi = {width, height};
  NppStatus status = nppiYUVToRGB_8u_P3C3R(src_planes, y_mem.step(), dst_mem.get(), dst_mem.step(), roi);
  ASSERT_EQ(status, NPP_NO_ERROR);

  std::vector<Npp8u> dst;
  dst_mem.copyToHost(dst);

  for (int i = 0; i < width * height; ++i) {
    Npp8u r, g, b;
    yuv_to_rgb_ref(pixels[i].y, pixels[i].u, pixels[i].v, r, g, b);
    int idx = i * 3;
    ASSERT_EQ(dst[idx + 0], r);
    ASSERT_EQ(dst[idx + 1], g);
    ASSERT_EQ(dst[idx + 2], b);
  }

  NppStreamContext ctx{};
  nppGetStreamContext(&ctx);
  ctx.hStream = 0;
  NppImageMemory<Npp8u> ctx_mem(width, height, 3);
  status = nppiYUVToRGB_8u_P3C3R_Ctx(src_planes, y_mem.step(), ctx_mem.get(), ctx_mem.step(), roi, ctx);
  ASSERT_EQ(status, NPP_NO_ERROR);

  std::vector<Npp8u> ctx_out;
  ctx_mem.copyToHost(ctx_out);
  EXPECT_EQ(ctx_out, dst);
}

TEST_F(YUVToRGBVariantsTest, YUVToRGB_8u_P3R_Reference) {
  const int width = 2;
  const int height = 2;

  std::vector<Npp8u> src;
  std::vector<Npp8u> y_plane, u_plane, v_plane;
  std::vector<YuvPixel> pixels;
  fill_yuv_inputs(src, y_plane, u_plane, v_plane, width, height, pixels);

  NppImageMemory<Npp8u> y_mem(width, height);
  NppImageMemory<Npp8u> u_mem(width, height);
  NppImageMemory<Npp8u> v_mem(width, height);
  NppImageMemory<Npp8u> r_mem(width, height);
  NppImageMemory<Npp8u> g_mem(width, height);
  NppImageMemory<Npp8u> b_mem(width, height);
  y_mem.copyFromHost(y_plane);
  u_mem.copyFromHost(u_plane);
  v_mem.copyFromHost(v_plane);

  const Npp8u *src_planes[3] = {y_mem.get(), u_mem.get(), v_mem.get()};
  Npp8u *dst_planes[3] = {r_mem.get(), g_mem.get(), b_mem.get()};

  NppiSize roi = {width, height};
  NppStatus status = nppiYUVToRGB_8u_P3R(src_planes, y_mem.step(), dst_planes, r_mem.step(), roi);
  ASSERT_EQ(status, NPP_NO_ERROR);

  std::vector<Npp8u> r_plane, g_plane, b_plane;
  r_mem.copyToHost(r_plane);
  g_mem.copyToHost(g_plane);
  b_mem.copyToHost(b_plane);

  for (int i = 0; i < width * height; ++i) {
    Npp8u r, g, b;
    yuv_to_rgb_ref(pixels[i].y, pixels[i].u, pixels[i].v, r, g, b);
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
  status = nppiYUVToRGB_8u_P3R_Ctx(src_planes, y_mem.step(), ctx_planes, r_ctx.step(), roi, ctx);
  ASSERT_EQ(status, NPP_NO_ERROR);

  std::vector<Npp8u> r_plane_ctx, g_plane_ctx, b_plane_ctx;
  r_ctx.copyToHost(r_plane_ctx);
  g_ctx.copyToHost(g_plane_ctx);
  b_ctx.copyToHost(b_plane_ctx);

  EXPECT_EQ(r_plane_ctx, r_plane);
  EXPECT_EQ(g_plane_ctx, g_plane);
  EXPECT_EQ(b_plane_ctx, b_plane);
}

TEST_F(YUVToRGBVariantsTest, YUVToBGR_8u_C3R_Reference) {
  const int width = 2;
  const int height = 2;

  std::vector<Npp8u> src;
  std::vector<Npp8u> y_plane, u_plane, v_plane;
  std::vector<YuvPixel> pixels;
  fill_yuv_inputs(src, y_plane, u_plane, v_plane, width, height, pixels);

  NppImageMemory<Npp8u> src_mem(width, height, 3);
  NppImageMemory<Npp8u> dst_mem(width, height, 3);
  src_mem.copyFromHost(src);

  NppiSize roi = {width, height};
  NppStatus status = nppiYUVToBGR_8u_C3R(src_mem.get(), src_mem.step(), dst_mem.get(), dst_mem.step(), roi);
  ASSERT_EQ(status, NPP_NO_ERROR);

  std::vector<Npp8u> dst;
  dst_mem.copyToHost(dst);

  for (int i = 0; i < width * height; ++i) {
    Npp8u r, g, b;
    yuv_to_rgb_ref(pixels[i].y, pixels[i].u, pixels[i].v, r, g, b);
    int idx = i * 3;
    ASSERT_EQ(dst[idx + 0], b);
    ASSERT_EQ(dst[idx + 1], g);
    ASSERT_EQ(dst[idx + 2], r);
  }

  NppStreamContext ctx{};
  nppGetStreamContext(&ctx);
  ctx.hStream = 0;
  NppImageMemory<Npp8u> ctx_mem(width, height, 3);
  status = nppiYUVToBGR_8u_C3R_Ctx(src_mem.get(), src_mem.step(), ctx_mem.get(), ctx_mem.step(), roi, ctx);
  ASSERT_EQ(status, NPP_NO_ERROR);
  std::vector<Npp8u> ctx_out;
  ctx_mem.copyToHost(ctx_out);
  EXPECT_EQ(ctx_out, dst);
}

TEST_F(YUVToRGBVariantsTest, YUVToBGR_8u_AC4R_AlphaCleared) {
  const int width = 2;
  const int height = 2;

  std::vector<Npp8u> src;
  std::vector<Npp8u> alpha(width * height);
  std::vector<Npp8u> y_plane, u_plane, v_plane;
  std::vector<YuvPixel> pixels;
  fill_yuv_inputs(src, y_plane, u_plane, v_plane, width, height, pixels);

  std::vector<Npp8u> src_ac4(width * height * 4);
  for (int i = 0; i < width * height; ++i) {
    src_ac4[i * 4 + 0] = pixels[i].y;
    src_ac4[i * 4 + 1] = pixels[i].u;
    src_ac4[i * 4 + 2] = pixels[i].v;
    src_ac4[i * 4 + 3] = static_cast<Npp8u>(23 + i * 29);
    alpha[i] = src_ac4[i * 4 + 3];
  }

  NppImageMemory<Npp8u> src_mem(width, height, 4);
  NppImageMemory<Npp8u> dst_mem(width, height, 4);
  src_mem.copyFromHost(src_ac4);

  NppiSize roi = {width, height};
  NppStatus status = nppiYUVToBGR_8u_AC4R(src_mem.get(), src_mem.step(), dst_mem.get(), dst_mem.step(), roi);
  ASSERT_EQ(status, NPP_NO_ERROR);

  std::vector<Npp8u> dst;
  dst_mem.copyToHost(dst);

  for (int i = 0; i < width * height; ++i) {
    Npp8u r, g, b;
    yuv_to_rgb_ref(pixels[i].y, pixels[i].u, pixels[i].v, r, g, b);
    int idx = i * 4;
    ASSERT_EQ(dst[idx + 0], b);
    ASSERT_EQ(dst[idx + 1], g);
    ASSERT_EQ(dst[idx + 2], r);
    // NVIDIA NPP clears alpha for AC4 packed output.
    ASSERT_EQ(dst[idx + 3], 0);
  }

  NppStreamContext ctx{};
  nppGetStreamContext(&ctx);
  ctx.hStream = 0;
  NppImageMemory<Npp8u> ctx_mem(width, height, 4);
  status = nppiYUVToBGR_8u_AC4R_Ctx(src_mem.get(), src_mem.step(), ctx_mem.get(), ctx_mem.step(), roi, ctx);
  ASSERT_EQ(status, NPP_NO_ERROR);

  std::vector<Npp8u> ctx_out;
  ctx_mem.copyToHost(ctx_out);
  EXPECT_EQ(ctx_out, dst);
}

TEST_F(YUVToRGBVariantsTest, YUVToBGR_8u_P3C3R_Reference) {
  const int width = 2;
  const int height = 2;

  std::vector<Npp8u> src;
  std::vector<Npp8u> y_plane, u_plane, v_plane;
  std::vector<YuvPixel> pixels;
  fill_yuv_inputs(src, y_plane, u_plane, v_plane, width, height, pixels);

  NppImageMemory<Npp8u> y_mem(width, height);
  NppImageMemory<Npp8u> u_mem(width, height);
  NppImageMemory<Npp8u> v_mem(width, height);
  NppImageMemory<Npp8u> dst_mem(width, height, 3);
  y_mem.copyFromHost(y_plane);
  u_mem.copyFromHost(u_plane);
  v_mem.copyFromHost(v_plane);

  const Npp8u *src_planes[3] = {y_mem.get(), u_mem.get(), v_mem.get()};

  NppiSize roi = {width, height};
  NppStatus status = nppiYUVToBGR_8u_P3C3R(src_planes, y_mem.step(), dst_mem.get(), dst_mem.step(), roi);
  ASSERT_EQ(status, NPP_NO_ERROR);

  std::vector<Npp8u> dst;
  dst_mem.copyToHost(dst);

  for (int i = 0; i < width * height; ++i) {
    Npp8u r, g, b;
    yuv_to_rgb_ref(pixels[i].y, pixels[i].u, pixels[i].v, r, g, b);
    int idx = i * 3;
    ASSERT_EQ(dst[idx + 0], b);
    ASSERT_EQ(dst[idx + 1], g);
    ASSERT_EQ(dst[idx + 2], r);
  }

  NppStreamContext ctx{};
  nppGetStreamContext(&ctx);
  ctx.hStream = 0;
  NppImageMemory<Npp8u> ctx_mem(width, height, 3);
  status = nppiYUVToBGR_8u_P3C3R_Ctx(src_planes, y_mem.step(), ctx_mem.get(), ctx_mem.step(), roi, ctx);
  ASSERT_EQ(status, NPP_NO_ERROR);

  std::vector<Npp8u> ctx_out;
  ctx_mem.copyToHost(ctx_out);
  EXPECT_EQ(ctx_out, dst);
}

TEST_F(YUVToRGBVariantsTest, YUVToBGR_8u_P3R_Reference) {
  const int width = 2;
  const int height = 2;

  std::vector<Npp8u> src;
  std::vector<Npp8u> y_plane, u_plane, v_plane;
  std::vector<YuvPixel> pixels;
  fill_yuv_inputs(src, y_plane, u_plane, v_plane, width, height, pixels);

  NppImageMemory<Npp8u> y_mem(width, height);
  NppImageMemory<Npp8u> u_mem(width, height);
  NppImageMemory<Npp8u> v_mem(width, height);
  NppImageMemory<Npp8u> b_mem(width, height);
  NppImageMemory<Npp8u> g_mem(width, height);
  NppImageMemory<Npp8u> r_mem(width, height);
  y_mem.copyFromHost(y_plane);
  u_mem.copyFromHost(u_plane);
  v_mem.copyFromHost(v_plane);

  const Npp8u *src_planes[3] = {y_mem.get(), u_mem.get(), v_mem.get()};
  Npp8u *dst_planes[3] = {b_mem.get(), g_mem.get(), r_mem.get()};

  NppiSize roi = {width, height};
  NppStatus status = nppiYUVToBGR_8u_P3R(src_planes, y_mem.step(), dst_planes, b_mem.step(), roi);
  ASSERT_EQ(status, NPP_NO_ERROR);

  std::vector<Npp8u> b_plane, g_plane, r_plane;
  b_mem.copyToHost(b_plane);
  g_mem.copyToHost(g_plane);
  r_mem.copyToHost(r_plane);

  for (int i = 0; i < width * height; ++i) {
    Npp8u r, g, b;
    yuv_to_rgb_ref(pixels[i].y, pixels[i].u, pixels[i].v, r, g, b);
    ASSERT_EQ(b_plane[i], b);
    ASSERT_EQ(g_plane[i], g);
    ASSERT_EQ(r_plane[i], r);
  }

  NppStreamContext ctx{};
  nppGetStreamContext(&ctx);
  ctx.hStream = 0;
  NppImageMemory<Npp8u> b_ctx(width, height);
  NppImageMemory<Npp8u> g_ctx(width, height);
  NppImageMemory<Npp8u> r_ctx(width, height);
  Npp8u *ctx_planes[3] = {b_ctx.get(), g_ctx.get(), r_ctx.get()};
  status = nppiYUVToBGR_8u_P3R_Ctx(src_planes, y_mem.step(), ctx_planes, b_ctx.step(), roi, ctx);
  ASSERT_EQ(status, NPP_NO_ERROR);

  std::vector<Npp8u> b_plane_ctx, g_plane_ctx, r_plane_ctx;
  b_ctx.copyToHost(b_plane_ctx);
  g_ctx.copyToHost(g_plane_ctx);
  r_ctx.copyToHost(r_plane_ctx);

  EXPECT_EQ(b_plane_ctx, b_plane);
  EXPECT_EQ(g_plane_ctx, g_plane);
  EXPECT_EQ(r_plane_ctx, r_plane);
}
