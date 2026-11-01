#include "npp_test_base.h"
#include <algorithm>
#include <random>
#include <vector>

using namespace npp_functional_test;

namespace {
struct RgbPixel {
  Npp8u r;
  Npp8u g;
  Npp8u b;
};
} // namespace

class YCbCrSubsampledTest : public NppTestBase {};

TEST_F(YCbCrSubsampledTest, RGBToYCbCr411_P3C3R_RoundTripUniformBlocks) {
  const int width = 8;
  const int height = 2;
  std::vector<RgbPixel> pixels(width * height);
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      int idx = y * width + x;
      if ((x / 4) % 2 == 0) {
        pixels[idx] = {255, 0, 0};
      } else {
        pixels[idx] = {0, 255, 0};
      }
    }
  }

  std::vector<Npp8u> src(width * height * 3);
  for (int i = 0; i < width * height; ++i) {
    src[i * 3 + 0] = pixels[i].r;
    src[i * 3 + 1] = pixels[i].g;
    src[i * 3 + 2] = pixels[i].b;
  }

  NppImageMemory<Npp8u> src_mem(width, height, 3);
  src_mem.copyFromHost(src);

  NppImageMemory<Npp8u> y_plane(width, height, 1);
  NppImageMemory<Npp8u> cb_plane(width / 4, height, 1);
  NppImageMemory<Npp8u> cr_plane(width / 4, height, 1);
  Npp8u *dst_planes[3] = {y_plane.get(), cb_plane.get(), cr_plane.get()};
  int dst_steps[3] = {y_plane.step(), cb_plane.step(), cr_plane.step()};

  NppiSize roi = {width, height};
  NppStatus status = nppiRGBToYCbCr411_8u_C3P3R(src_mem.get(), src_mem.step(), dst_planes, dst_steps, roi);
  ASSERT_EQ(status, NPP_NO_ERROR);

  NppImageMemory<Npp8u> dst_mem(width, height, 3);
  status = nppiYCbCr411ToRGB_8u_P3C3R((const Npp8u *const *)dst_planes, dst_steps, dst_mem.get(), dst_mem.step(), roi);
  ASSERT_EQ(status, NPP_NO_ERROR);

  std::vector<Npp8u> dst;
  dst_mem.copyToHost(dst);
  for (int i = 0; i < width * height; ++i) {
    EXPECT_NEAR(dst[i * 3 + 0], pixels[i].r, 4);
    EXPECT_NEAR(dst[i * 3 + 1], pixels[i].g, 4);
    EXPECT_NEAR(dst[i * 3 + 2], pixels[i].b, 4);
  }
}

TEST_F(YCbCrSubsampledTest, BGRToYCbCr422_P3C3R_RoundTripUniformBlocks) {
  const int width = 6;
  const int height = 2;
  std::vector<RgbPixel> pixels(width * height);
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      int idx = y * width + x;
      if ((x / 2) % 2 == 0) {
        pixels[idx] = {0, 0, 255};
      } else {
        pixels[idx] = {255, 255, 0};
      }
    }
  }

  std::vector<Npp8u> src(width * height * 3);
  for (int i = 0; i < width * height; ++i) {
    src[i * 3 + 0] = pixels[i].b;
    src[i * 3 + 1] = pixels[i].g;
    src[i * 3 + 2] = pixels[i].r;
  }

  NppImageMemory<Npp8u> src_mem(width, height, 3);
  src_mem.copyFromHost(src);

  NppImageMemory<Npp8u> y_plane(width, height, 1);
  NppImageMemory<Npp8u> cb_plane(width / 2, height, 1);
  NppImageMemory<Npp8u> cr_plane(width / 2, height, 1);
  Npp8u *dst_planes[3] = {y_plane.get(), cb_plane.get(), cr_plane.get()};
  int dst_steps[3] = {y_plane.step(), cb_plane.step(), cr_plane.step()};

  NppiSize roi = {width, height};
  NppStatus status = nppiBGRToYCbCr422_8u_C3P3R(src_mem.get(), src_mem.step(), dst_planes, dst_steps, roi);
  ASSERT_EQ(status, NPP_NO_ERROR);

  NppImageMemory<Npp8u> dst_mem(width, height, 3);
  status = nppiYCbCr422ToBGR_8u_P3C3R((const Npp8u *const *)dst_planes, dst_steps, dst_mem.get(), dst_mem.step(), roi);
  ASSERT_EQ(status, NPP_NO_ERROR);

  std::vector<Npp8u> dst;
  dst_mem.copyToHost(dst);
  for (int i = 0; i < width * height; ++i) {
    EXPECT_NEAR(dst[i * 3 + 0], pixels[i].b, 4);
    EXPECT_NEAR(dst[i * 3 + 1], pixels[i].g, 4);
    EXPECT_NEAR(dst[i * 3 + 2], pixels[i].r, 4);
  }
}

namespace {
void fillPlanar411(NppImageMemory<Npp8u> &y, NppImageMemory<Npp8u> &cb, NppImageMemory<Npp8u> &cr, int width,
                   int height, Npp8u yv, Npp8u cbv, Npp8u crv) {
  std::vector<Npp8u> yh(static_cast<size_t>(width) * height, yv);
  std::vector<Npp8u> ch(static_cast<size_t>(width / 4) * height, cbv);
  std::vector<Npp8u> crh(static_cast<size_t>(width / 4) * height, crv);
  y.copyFromHost(yh);
  cb.copyFromHost(ch);
  cr.copyFromHost(crh);
}
} // namespace

// JPEG variants use the full-range JFIF curves; the normal variants use the
// video-range curves.
TEST_F(YCbCrSubsampledTest, YCbCr411JPEGDecodeUsesFullRange) {
  const int width = 8;
  const int height = 4;
  NppImageMemory<Npp8u> y(width, height, 1);
  NppImageMemory<Npp8u> cb(width / 4, height, 1);
  NppImageMemory<Npp8u> cr(width / 4, height, 1);
  const Npp8u *planes[3] = {y.get(), cb.get(), cr.get()};
  int steps[3] = {y.step(), cb.step(), cr.step()};
  NppiSize roi = {width, height};
  NppImageMemory<Npp8u> jpeg_dst(width, height, 3);
  NppImageMemory<Npp8u> video_dst(width, height, 3);

  struct Case {
    Npp8u y, cb, cr;
    Npp8u jpeg[3];
    Npp8u video[3];
  } cases[] = {
      {128, 128, 128, {128, 128, 128}, {130, 130, 130}},
      {16, 128, 128, {16, 16, 16}, {0, 0, 0}},
      {81, 90, 240, {238, 14, 13}, {254, 0, 0}},
  };
  for (const auto &c : cases) {
    fillPlanar411(y, cb, cr, width, height, c.y, c.cb, c.cr);
    ASSERT_EQ(nppiYCbCr411ToRGB_JPEG_8u_P3C3R(planes, steps, jpeg_dst.get(), jpeg_dst.step(), roi), NPP_NO_ERROR);
    ASSERT_EQ(nppiYCbCr411ToRGB_8u_P3C3R(planes, steps, video_dst.get(), video_dst.step(), roi), NPP_NO_ERROR);
    std::vector<Npp8u> jpeg;
    std::vector<Npp8u> video;
    jpeg_dst.copyToHost(jpeg);
    video_dst.copyToHost(video);
    for (int ch = 0; ch < 3; ++ch) {
      EXPECT_NEAR(jpeg[ch], c.jpeg[ch], 1) << "Y=" << (int)c.y << " ch=" << ch;
      EXPECT_NEAR(video[ch], c.video[ch], 1) << "Y=" << (int)c.y << " ch=" << ch;
    }
  }
}

TEST_F(YCbCrSubsampledTest, YCbCr411JPEGEncodeUsesFullRange) {
  const int width = 8;
  const int height = 4;
  NppImageMemory<Npp8u> src(width, height, 3);
  NppImageMemory<Npp8u> y(width, height, 1);
  NppImageMemory<Npp8u> cb(width / 4, height, 1);
  NppImageMemory<Npp8u> cr(width / 4, height, 1);
  Npp8u *planes[3] = {y.get(), cb.get(), cr.get()};
  int steps[3] = {y.step(), cb.step(), cr.step()};
  NppiSize roi = {width, height};

  struct Case {
    Npp8u rgb[3];
    Npp8u jpeg_y, jpeg_cb, jpeg_cr;
    Npp8u video_y;
  } cases[] = {
      {{128, 128, 128}, 128, 128, 128, 125},
      {{255, 0, 0}, 76, 84, 255, 81},
      {{0, 0, 255}, 29, 255, 107, 40},
  };
  for (const auto &c : cases) {
    std::vector<Npp8u> host(static_cast<size_t>(width) * height * 3);
    for (size_t i = 0; i < host.size(); i += 3) {
      host[i] = c.rgb[0];
      host[i + 1] = c.rgb[1];
      host[i + 2] = c.rgb[2];
    }
    src.copyFromHost(host);
    ASSERT_EQ(nppiRGBToYCbCr411_JPEG_8u_C3P3R(src.get(), src.step(), planes, steps, roi), NPP_NO_ERROR);
    std::vector<Npp8u> yj;
    std::vector<Npp8u> cbj;
    std::vector<Npp8u> crj;
    y.copyToHost(yj);
    cb.copyToHost(cbj);
    cr.copyToHost(crj);
    EXPECT_NEAR(yj[0], c.jpeg_y, 1) << "RGB=" << (int)c.rgb[0] << "," << (int)c.rgb[1] << "," << (int)c.rgb[2];
    EXPECT_NEAR(cbj[0], c.jpeg_cb, 1);
    EXPECT_NEAR(crj[0], c.jpeg_cr, 1);
    ASSERT_EQ(nppiRGBToYCbCr411_8u_C3P3R(src.get(), src.step(), planes, steps, roi), NPP_NO_ERROR);
    y.copyToHost(yj);
    EXPECT_NEAR(yj[0], c.video_y, 1);
  }
}
