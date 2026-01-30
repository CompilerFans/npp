#include "npp_test_base.h"
#include <algorithm>
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

static inline void rgb_to_yuv_ref(Npp8u r, Npp8u g, Npp8u b, Npp8u &y, Npp8u &u, Npp8u &v) {
  float yf = 0.299f * r + 0.587f * g + 0.114f * b;
  float uf = 0.492f * (b - yf) + 128.0f;
  float vf = 0.877f * (r - yf) + 128.0f;
  y = clamp_to_u8(yf);
  u = clamp_to_u8(uf);
  v = clamp_to_u8(vf);
}

static inline void bgr_to_yuv_ref(Npp8u b, Npp8u g, Npp8u r, Npp8u &y, Npp8u &u, Npp8u &v) {
  rgb_to_yuv_ref(r, g, b, y, u, v);
}

static void build_rgb_inputs(std::vector<Npp8u> &packed, std::vector<Npp8u> &r, std::vector<Npp8u> &g,
                             std::vector<Npp8u> &b, int width, int height) {
  const std::vector<RgbPixel> pixels = {
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
                             std::vector<Npp8u> &r, int width, int height) {
  const std::vector<RgbPixel> pixels = {
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

static void check_packed_yuv(const std::vector<Npp8u> &dst, const std::vector<RgbPixel> &pixels) {
  for (size_t i = 0; i < pixels.size(); ++i) {
    Npp8u y, u, v;
    rgb_to_yuv_ref(pixels[i].r, pixels[i].g, pixels[i].b, y, u, v);
    size_t idx = i * 3;
    ASSERT_EQ(dst[idx], y);
    ASSERT_EQ(dst[idx + 1], u);
    ASSERT_EQ(dst[idx + 2], v);
  }
}

static void check_packed_yuv_from_bgr(const std::vector<Npp8u> &dst, const std::vector<RgbPixel> &pixels) {
  for (size_t i = 0; i < pixels.size(); ++i) {
    Npp8u y, u, v;
    bgr_to_yuv_ref(pixels[i].b, pixels[i].g, pixels[i].r, y, u, v);
    size_t idx = i * 3;
    ASSERT_EQ(dst[idx], y);
    ASSERT_EQ(dst[idx + 1], u);
    ASSERT_EQ(dst[idx + 2], v);
  }
}

static void check_planar_yuv(const std::vector<Npp8u> &y_plane, const std::vector<Npp8u> &u_plane,
                             const std::vector<Npp8u> &v_plane, const std::vector<RgbPixel> &pixels) {
  for (size_t i = 0; i < pixels.size(); ++i) {
    Npp8u y, u, v;
    rgb_to_yuv_ref(pixels[i].r, pixels[i].g, pixels[i].b, y, u, v);
    ASSERT_EQ(y_plane[i], y);
    ASSERT_EQ(u_plane[i], u);
    ASSERT_EQ(v_plane[i], v);
  }
}

static void check_planar_yuv_from_bgr(const std::vector<Npp8u> &y_plane, const std::vector<Npp8u> &u_plane,
                                      const std::vector<Npp8u> &v_plane, const std::vector<RgbPixel> &pixels) {
  for (size_t i = 0; i < pixels.size(); ++i) {
    Npp8u y, u, v;
    bgr_to_yuv_ref(pixels[i].b, pixels[i].g, pixels[i].r, y, u, v);
    ASSERT_EQ(y_plane[i], y);
    ASSERT_EQ(u_plane[i], u);
    ASSERT_EQ(v_plane[i], v);
  }
}
} // namespace

class RGBToYUVVariantsTest : public NppTestBase {};

TEST_F(RGBToYUVVariantsTest, RGBToYUV_8u_C3R_Reference) {
  const int width = 2;
  const int height = 2;
  const std::vector<RgbPixel> pixels = {
      {0, 0, 0},
      {255, 255, 255},
      {255, 0, 0},
      {0, 0, 255},
  };

  std::vector<Npp8u> src;
  std::vector<Npp8u> r_plane, g_plane, b_plane;
  build_rgb_inputs(src, r_plane, g_plane, b_plane, width, height);

  NppImageMemory<Npp8u> src_mem(width, height, 3);
  NppImageMemory<Npp8u> dst_mem(width, height, 3);

  src_mem.copyFromHost(src);

  NppiSize roi = {width, height};
  NppStatus status = nppiRGBToYUV_8u_C3R(src_mem.get(), src_mem.step(), dst_mem.get(), dst_mem.step(), roi);
  ASSERT_EQ(status, NPP_NO_ERROR);

  std::vector<Npp8u> dst;
  dst_mem.copyToHost(dst);
  check_packed_yuv(dst, pixels);
}

TEST_F(RGBToYUVVariantsTest, RGBToYUV_8u_AC4R_AlphaCleared) {
  const int width = 2;
  const int height = 2;
  const std::vector<RgbPixel> pixels = {
      {12, 34, 56},
      {78, 90, 123},
      {200, 10, 40},
      {0, 255, 128},
  };

  std::vector<Npp8u> src(width * height * 4);
  std::vector<Npp8u> alpha(width * height);
  for (int i = 0; i < width * height; ++i) {
    const RgbPixel &p = pixels[i];
    src[i * 4 + 0] = p.r;
    src[i * 4 + 1] = p.g;
    src[i * 4 + 2] = p.b;
    src[i * 4 + 3] = static_cast<Npp8u>(10 + i * 20);
    alpha[i] = src[i * 4 + 3];
  }

  NppImageMemory<Npp8u> src_mem(width, height, 4);
  NppImageMemory<Npp8u> dst_mem(width, height, 4);
  src_mem.copyFromHost(src);

  NppiSize roi = {width, height};
  NppStatus status = nppiRGBToYUV_8u_AC4R(src_mem.get(), src_mem.step(), dst_mem.get(), dst_mem.step(), roi);
  ASSERT_EQ(status, NPP_NO_ERROR);

  std::vector<Npp8u> dst;
  dst_mem.copyToHost(dst);
  for (int i = 0; i < width * height; ++i) {
    Npp8u y, u, v;
    rgb_to_yuv_ref(pixels[i].r, pixels[i].g, pixels[i].b, y, u, v);
    ASSERT_EQ(dst[i * 4 + 0], y);
    ASSERT_EQ(dst[i * 4 + 1], u);
    ASSERT_EQ(dst[i * 4 + 2], v);
    // NVIDIA NPP clears alpha for AC4 packed output.
    ASSERT_EQ(dst[i * 4 + 3], 0);
  }
}

TEST_F(RGBToYUVVariantsTest, RGBToYUV_8u_C3P3R_Reference) {
  const int width = 2;
  const int height = 2;
  const std::vector<RgbPixel> pixels = {
      {0, 0, 0},
      {255, 255, 255},
      {255, 0, 0},
      {0, 0, 255},
  };

  std::vector<Npp8u> src;
  std::vector<Npp8u> r_plane, g_plane, b_plane;
  build_rgb_inputs(src, r_plane, g_plane, b_plane, width, height);

  NppImageMemory<Npp8u> src_mem(width, height, 3);
  NppImageMemory<Npp8u> y_mem(width, height);
  NppImageMemory<Npp8u> u_mem(width, height);
  NppImageMemory<Npp8u> v_mem(width, height);

  src_mem.copyFromHost(src);

  Npp8u *dst_planes[3] = {y_mem.get(), u_mem.get(), v_mem.get()};

  NppiSize roi = {width, height};
  NppStatus status = nppiRGBToYUV_8u_C3P3R(src_mem.get(), src_mem.step(), dst_planes, y_mem.step(), roi);
  ASSERT_EQ(status, NPP_NO_ERROR);

  std::vector<Npp8u> y_plane, u_plane, v_plane;
  y_mem.copyToHost(y_plane);
  u_mem.copyToHost(u_plane);
  v_mem.copyToHost(v_plane);

  check_planar_yuv(y_plane, u_plane, v_plane, pixels);
}

TEST_F(RGBToYUVVariantsTest, RGBToYUV_8u_P3R_Reference) {
  const int width = 2;
  const int height = 2;
  const std::vector<RgbPixel> pixels = {
      {0, 0, 0},
      {255, 255, 255},
      {255, 0, 0},
      {0, 0, 255},
  };

  std::vector<Npp8u> src;
  std::vector<Npp8u> r_plane, g_plane, b_plane;
  build_rgb_inputs(src, r_plane, g_plane, b_plane, width, height);

  NppImageMemory<Npp8u> r_mem(width, height);
  NppImageMemory<Npp8u> g_mem(width, height);
  NppImageMemory<Npp8u> b_mem(width, height);
  NppImageMemory<Npp8u> y_mem(width, height);
  NppImageMemory<Npp8u> u_mem(width, height);
  NppImageMemory<Npp8u> v_mem(width, height);

  r_mem.copyFromHost(r_plane);
  g_mem.copyFromHost(g_plane);
  b_mem.copyFromHost(b_plane);

  const Npp8u *src_planes[3] = {r_mem.get(), g_mem.get(), b_mem.get()};
  Npp8u *dst_planes[3] = {y_mem.get(), u_mem.get(), v_mem.get()};

  NppiSize roi = {width, height};
  NppStatus status = nppiRGBToYUV_8u_P3R(src_planes, r_mem.step(), dst_planes, y_mem.step(), roi);
  ASSERT_EQ(status, NPP_NO_ERROR);

  std::vector<Npp8u> y_plane, u_plane, v_plane;
  y_mem.copyToHost(y_plane);
  u_mem.copyToHost(u_plane);
  v_mem.copyToHost(v_plane);

  check_planar_yuv(y_plane, u_plane, v_plane, pixels);
}

TEST_F(RGBToYUVVariantsTest, RGBToYUV_8u_AC4P4R_AlphaPreserve) {
  const int width = 2;
  const int height = 2;
  const std::vector<RgbPixel> pixels = {
      {12, 34, 56},
      {78, 90, 123},
      {200, 10, 40},
      {0, 255, 128},
  };

  std::vector<Npp8u> src(width * height * 4);
  std::vector<Npp8u> alpha(width * height);
  for (int i = 0; i < width * height; ++i) {
    const RgbPixel &p = pixels[i];
    src[i * 4 + 0] = p.r;
    src[i * 4 + 1] = p.g;
    src[i * 4 + 2] = p.b;
    src[i * 4 + 3] = static_cast<Npp8u>(15 + i * 17);
    alpha[i] = src[i * 4 + 3];
  }

  NppImageMemory<Npp8u> src_mem(width, height, 4);
  NppImageMemory<Npp8u> y_mem(width, height);
  NppImageMemory<Npp8u> u_mem(width, height);
  NppImageMemory<Npp8u> v_mem(width, height);
  NppImageMemory<Npp8u> a_mem(width, height);

  src_mem.copyFromHost(src);

  Npp8u *dst_planes[4] = {y_mem.get(), u_mem.get(), v_mem.get(), a_mem.get()};

  NppiSize roi = {width, height};
  NppStatus status = nppiRGBToYUV_8u_AC4P4R(src_mem.get(), src_mem.step(), dst_planes, y_mem.step(), roi);
  ASSERT_EQ(status, NPP_NO_ERROR);

  std::vector<Npp8u> y_plane, u_plane, v_plane, a_plane;
  y_mem.copyToHost(y_plane);
  u_mem.copyToHost(u_plane);
  v_mem.copyToHost(v_plane);
  a_mem.copyToHost(a_plane);

  for (int i = 0; i < width * height; ++i) {
    Npp8u y, u, v;
    rgb_to_yuv_ref(pixels[i].r, pixels[i].g, pixels[i].b, y, u, v);
    ASSERT_EQ(y_plane[i], y);
    ASSERT_EQ(u_plane[i], u);
    ASSERT_EQ(v_plane[i], v);
    ASSERT_EQ(a_plane[i], alpha[i]);
  }
}

TEST_F(RGBToYUVVariantsTest, BGRToYUV_8u_C3R_Reference) {
  const int width = 2;
  const int height = 2;
  const std::vector<RgbPixel> pixels = {
      {0, 0, 0},
      {255, 255, 255},
      {255, 0, 0},
      {0, 0, 255},
  };

  std::vector<Npp8u> src;
  std::vector<Npp8u> b_plane, g_plane, r_plane;
  build_bgr_inputs(src, b_plane, g_plane, r_plane, width, height);

  NppImageMemory<Npp8u> src_mem(width, height, 3);
  NppImageMemory<Npp8u> dst_mem(width, height, 3);

  src_mem.copyFromHost(src);

  NppiSize roi = {width, height};
  NppStatus status = nppiBGRToYUV_8u_C3R(src_mem.get(), src_mem.step(), dst_mem.get(), dst_mem.step(), roi);
  ASSERT_EQ(status, NPP_NO_ERROR);

  std::vector<Npp8u> dst;
  dst_mem.copyToHost(dst);
  check_packed_yuv_from_bgr(dst, pixels);
}

TEST_F(RGBToYUVVariantsTest, BGRToYUV_8u_AC4R_AlphaCleared) {
  const int width = 2;
  const int height = 2;
  const std::vector<RgbPixel> pixels = {
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
    src[i * 4 + 3] = static_cast<Npp8u>(11 + i * 19);
    alpha[i] = src[i * 4 + 3];
  }

  NppImageMemory<Npp8u> src_mem(width, height, 4);
  NppImageMemory<Npp8u> dst_mem(width, height, 4);
  src_mem.copyFromHost(src);

  NppiSize roi = {width, height};
  NppStatus status = nppiBGRToYUV_8u_AC4R(src_mem.get(), src_mem.step(), dst_mem.get(), dst_mem.step(), roi);
  ASSERT_EQ(status, NPP_NO_ERROR);

  std::vector<Npp8u> dst;
  dst_mem.copyToHost(dst);
  for (int i = 0; i < width * height; ++i) {
    Npp8u y, u, v;
    bgr_to_yuv_ref(pixels[i].b, pixels[i].g, pixels[i].r, y, u, v);
    ASSERT_EQ(dst[i * 4 + 0], y);
    ASSERT_EQ(dst[i * 4 + 1], u);
    ASSERT_EQ(dst[i * 4 + 2], v);
    // NVIDIA NPP clears alpha for AC4 packed output.
    ASSERT_EQ(dst[i * 4 + 3], 0);
  }
}

TEST_F(RGBToYUVVariantsTest, BGRToYUV_8u_C3P3R_Reference) {
  const int width = 2;
  const int height = 2;
  const std::vector<RgbPixel> pixels = {
      {0, 0, 0},
      {255, 255, 255},
      {255, 0, 0},
      {0, 0, 255},
  };

  std::vector<Npp8u> src;
  std::vector<Npp8u> b_plane, g_plane, r_plane;
  build_bgr_inputs(src, b_plane, g_plane, r_plane, width, height);

  NppImageMemory<Npp8u> src_mem(width, height, 3);
  NppImageMemory<Npp8u> y_mem(width, height);
  NppImageMemory<Npp8u> u_mem(width, height);
  NppImageMemory<Npp8u> v_mem(width, height);

  src_mem.copyFromHost(src);

  Npp8u *dst_planes[3] = {y_mem.get(), u_mem.get(), v_mem.get()};

  NppiSize roi = {width, height};
  NppStatus status = nppiBGRToYUV_8u_C3P3R(src_mem.get(), src_mem.step(), dst_planes, y_mem.step(), roi);
  ASSERT_EQ(status, NPP_NO_ERROR);

  std::vector<Npp8u> y_plane, u_plane, v_plane;
  y_mem.copyToHost(y_plane);
  u_mem.copyToHost(u_plane);
  v_mem.copyToHost(v_plane);

  check_planar_yuv_from_bgr(y_plane, u_plane, v_plane, pixels);
}

TEST_F(RGBToYUVVariantsTest, BGRToYUV_8u_P3R_Reference) {
  const int width = 2;
  const int height = 2;
  const std::vector<RgbPixel> pixels = {
      {0, 0, 0},
      {255, 255, 255},
      {255, 0, 0},
      {0, 0, 255},
  };

  std::vector<Npp8u> src;
  std::vector<Npp8u> b_plane, g_plane, r_plane;
  build_bgr_inputs(src, b_plane, g_plane, r_plane, width, height);

  NppImageMemory<Npp8u> b_mem(width, height);
  NppImageMemory<Npp8u> g_mem(width, height);
  NppImageMemory<Npp8u> r_mem(width, height);
  NppImageMemory<Npp8u> y_mem(width, height);
  NppImageMemory<Npp8u> u_mem(width, height);
  NppImageMemory<Npp8u> v_mem(width, height);

  b_mem.copyFromHost(b_plane);
  g_mem.copyFromHost(g_plane);
  r_mem.copyFromHost(r_plane);

  const Npp8u *src_planes[3] = {b_mem.get(), g_mem.get(), r_mem.get()};
  Npp8u *dst_planes[3] = {y_mem.get(), u_mem.get(), v_mem.get()};

  NppiSize roi = {width, height};
  NppStatus status = nppiBGRToYUV_8u_P3R(src_planes, b_mem.step(), dst_planes, y_mem.step(), roi);
  ASSERT_EQ(status, NPP_NO_ERROR);

  std::vector<Npp8u> y_plane, u_plane, v_plane;
  y_mem.copyToHost(y_plane);
  u_mem.copyToHost(u_plane);
  v_mem.copyToHost(v_plane);

  check_planar_yuv_from_bgr(y_plane, u_plane, v_plane, pixels);
}

TEST_F(RGBToYUVVariantsTest, BGRToYUV_8u_AC4P4R_AlphaPreserve) {
  const int width = 2;
  const int height = 2;
  const std::vector<RgbPixel> pixels = {
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
    src[i * 4 + 3] = static_cast<Npp8u>(9 + i * 23);
    alpha[i] = src[i * 4 + 3];
  }

  NppImageMemory<Npp8u> src_mem(width, height, 4);
  NppImageMemory<Npp8u> y_mem(width, height);
  NppImageMemory<Npp8u> u_mem(width, height);
  NppImageMemory<Npp8u> v_mem(width, height);
  NppImageMemory<Npp8u> a_mem(width, height);

  src_mem.copyFromHost(src);

  Npp8u *dst_planes[4] = {y_mem.get(), u_mem.get(), v_mem.get(), a_mem.get()};

  NppiSize roi = {width, height};
  NppStatus status = nppiBGRToYUV_8u_AC4P4R(src_mem.get(), src_mem.step(), dst_planes, y_mem.step(), roi);
  ASSERT_EQ(status, NPP_NO_ERROR);

  std::vector<Npp8u> y_plane, u_plane, v_plane, a_plane;
  y_mem.copyToHost(y_plane);
  u_mem.copyToHost(u_plane);
  v_mem.copyToHost(v_plane);
  a_mem.copyToHost(a_plane);

  for (int i = 0; i < width * height; ++i) {
    Npp8u y, u, v;
    bgr_to_yuv_ref(pixels[i].b, pixels[i].g, pixels[i].r, y, u, v);
    ASSERT_EQ(y_plane[i], y);
    ASSERT_EQ(u_plane[i], u);
    ASSERT_EQ(v_plane[i], v);
    ASSERT_EQ(a_plane[i], alpha[i]);
  }
}
