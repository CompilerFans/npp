#include "npp_test_base.h"
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
