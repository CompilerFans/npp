// Differential tests for nppiHistogramOfGradientsBorder (HOG).
//
// Record on the RTX 5090 against NVIDIA NPP 13.4:
//   NPP_RECORD_DIR=... ./diff_tests --gtest_filter='NppiHogDiff.*'
// Verify against mcMPP:
//   NPP_VERIFY_DIR=artifacts/5090/npp_records ./diff_tests \
//     --gtest_filter='NppiHogDiff.*'
//
// The images are structured (ramps, a delta, steps and a checkerboard) so the
// reference's inlined atan approximation agrees with atan2f in every bin.
#include "npp_diff_harness.h"

#include <cuda_runtime.h>

#include <cstdio>
#include <string>
#include <vector>

using namespace npp_diff;

namespace {

NppStreamContext current_ctx() {
  NppStreamContext ctx{};
  EXPECT_EQ(nppGetStreamContext(&ctx), NPP_NO_ERROR);
  return ctx;
}

int descriptorCount(const NppiHOGConfig &config) {
  const int blocksX = 2 * (config.detectionWindowSize.width / config.histogramBlockSize) - 1;
  const int blocksY = 2 * (config.detectionWindowSize.height / config.histogramBlockSize) - 1;
  const int cells = config.histogramBlockSize / config.cellSize;
  return blocksX * blocksY * cells * cells * config.nHistogramBins;
}

template <typename Src>
std::vector<Src> makeStructured(int width, int height, int channels, int pattern) {
  std::vector<Src> img(static_cast<size_t>(width) * height * channels, Src(0));
  const auto set = [&](int x, int y, int c, float value) {
    img[(static_cast<size_t>(y) * width + x) * channels + c] = static_cast<Src>(value);
  };
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      for (int c = 0; c < channels; ++c) {
        float value = 0.0f;
        switch (pattern) {
        case 0:  // horizontal ramp
          value = static_cast<float>((x * 8) & 0xFF);
          break;
        case 1:  // vertical ramp
          value = static_cast<float>((y * 8) & 0xFF);
          break;
        case 2:  // diagonal ramp
          value = static_cast<float>(((x + y) * 4) & 0xFF);
          break;
        case 3:  // single delta
          value = (x == width / 2 && y == height / 2) ? 255.0f : 0.0f;
          break;
        case 4:  // vertical step
          value = (x >= width / 2) ? 255.0f : 0.0f;
          break;
        case 5:  // checkerboard of 4x4 blocks
          value = (((x / 4) + (y / 4)) & 1) ? 255.0f : 0.0f;
          break;
        case 6:  // C3: channel 0 strong horizontal ramp, channel 1 weak vertical ramp
          if (c == 0) value = static_cast<float>((x * 8) & 0xFF);
          else if (c == 1) value = static_cast<float>((y * 2) & 0xFF);
          break;
        case 7:  // C3: channel 0 vertical ramp only
          if (c == 0) value = static_cast<float>((y * 8) & 0xFF);
          break;
        case 8:  // C3: channel 1 strong vertical ramp, channel 0 weak horizontal ramp
          if (c == 0) value = static_cast<float>((x * 2) & 0xFF);
          else if (c == 1) value = static_cast<float>((y * 8) & 0xFF);
          break;
        default:
          break;
        }
        set(x, y, c, value);
      }
    }
  }
  return img;
}

template <typename Src, int Channels, typename Fn>
void runHog(const char *fnName, const std::string &key, Fn fn, const std::vector<Src> &image, int width,
            int height, NppiSize roi, NppiHOGConfig config, const std::vector<NppiPoint> &locations) {
  const int channels = Channels;
  const int rowBytes = width * channels * static_cast<int>(sizeof(Src));
  int step = (rowBytes + 3) & ~3;
  if (step < 128 * static_cast<int>(locations.size())) {
    step = 128 * static_cast<int>(locations.size());
  }
  Npp8u *dSrc = nullptr;
  Npp8u *dScratch = nullptr;
  Npp32f *dDst = nullptr;
  const int descCount = descriptorCount(config) * static_cast<int>(locations.size());
  int scratchBytes = 0;
  NppStatus getStatus =
      nppiHistogramOfGradientsBorderGetBufferSize(config, locations.data(), static_cast<int>(locations.size()),
                                                  roi, &scratchBytes);
  EXPECT_EQ(getStatus, NPP_NO_ERROR);
  ASSERT_EQ(cudaMalloc(&dSrc, static_cast<size_t>(step) * height), cudaSuccess);
  ASSERT_EQ(cudaMemset(dSrc, 0, static_cast<size_t>(step) * height), cudaSuccess);
  ASSERT_EQ(cudaMemcpy2D(dSrc, step, image.data(), rowBytes, static_cast<size_t>(rowBytes), height,
                         cudaMemcpyHostToDevice),
            cudaSuccess);
  ASSERT_EQ(cudaMalloc(&dScratch, scratchBytes > 0 ? scratchBytes : 1), cudaSuccess);
  ASSERT_EQ(cudaMalloc(&dDst, static_cast<size_t>(descCount) * sizeof(Npp32f)), cudaSuccess);
  ASSERT_EQ(cudaMemset(dDst, 0xEE, static_cast<size_t>(descCount) * sizeof(Npp32f)), cudaSuccess);

  clear_cuda_error();
  NppStatus status = fn(reinterpret_cast<const Src *>(dSrc), step, {width, height}, {0, 0}, locations.data(),
                        static_cast<int>(locations.size()), dDst, roi, config, dScratch,
                        NPP_BORDER_REPLICATE, current_ctx());
  Case c(make_id({"hog", key}), fnName);
  c.input(image.data(), image.size() * sizeof(Src));
  c.status(status);
  c.device_output<Npp32f>(dDst, descCount, ValueKind::F32, 1e-4, 1e-4);
  c.finish();

  cudaFree(dSrc);
  cudaFree(dScratch);
  cudaFree(dDst);
}

void runHogStatusOnly(const char *fnName, const std::string &key, int width, int height, NppiSize roi,
                      NppiHOGConfig config, std::vector<NppiPoint> locations, int step,
                      NppiBorderType border) {
  std::vector<Npp8u> image(static_cast<size_t>(width) * height, 0);
  Npp8u *dSrc = nullptr;
  Npp8u *dScratch = nullptr;
  Npp32f *dDst = nullptr;
  const int descCount = descriptorCount(config) * static_cast<int>(locations.size() > 0 ? locations.size() : 1);
  int scratchBytes = 0;
  nppiHistogramOfGradientsBorderGetBufferSize(config, locations.data(), static_cast<int>(locations.size()), roi,
                                              &scratchBytes);
  ASSERT_EQ(cudaMalloc(&dSrc, static_cast<size_t>(step) * height), cudaSuccess);
  ASSERT_EQ(cudaMemset(dSrc, 0, static_cast<size_t>(step) * height), cudaSuccess);
  ASSERT_EQ(cudaMalloc(&dScratch, scratchBytes > 0 ? scratchBytes : 1), cudaSuccess);
  ASSERT_EQ(cudaMalloc(&dDst, static_cast<size_t>(descCount) * sizeof(Npp32f)), cudaSuccess);
  clear_cuda_error();
  NppStatus status = nppiHistogramOfGradientsBorder_8u32f_C1R_Ctx(
      dSrc, step, {width, height}, {0, 0}, locations.data(), static_cast<int>(locations.size()), dDst, roi,
      config, dScratch, border, current_ctx());
  Case c(make_id({"hog", key}), fnName);
  c.status(status);
  c.finish();
  cudaFree(dSrc);
  cudaFree(dScratch);
  cudaFree(dDst);
}

}  // namespace

TEST(NppiHogDiff, Structured8u) {
  struct Config {
    int cell, block, bins, window;
    const char *name;
  };
  const Config configs[] = {{8, 16, 9, 32, "8/16/9"}, {8, 16, 4, 32, "8/16/4"},
                            {4, 8, 9, 16, "4/8/9"},   {2, 4, 9, 8, "2/4/9"}};
  for (const Config &cfg : configs) {
    for (int pattern = 0; pattern <= 4; ++pattern) {
      const int size = cfg.window;
      const std::vector<Npp8u> image = makeStructured<Npp8u>(size, size, 1, pattern);
      NppiHOGConfig config{cfg.cell, cfg.block, cfg.bins, {size, size}};
      const std::vector<NppiPoint> locations = {{0, 0}};
      runHog<Npp8u, 1>("nppiHistogramOfGradientsBorder_8u32f_C1R_Ctx",
                       std::string(cfg.name) + "/pattern" + std::to_string(pattern),
                       nppiHistogramOfGradientsBorder_8u32f_C1R_Ctx, image, size, size, {size, size}, config,
                       locations);
    }
  }
}

TEST(NppiHogDiff, OtherSources) {
  const int size = 32;
  NppiHOGConfig config{8, 16, 9, {size, size}};
  const std::vector<NppiPoint> locations = {{0, 0}};
  for (int pattern = 0; pattern <= 3; ++pattern) {
    const std::vector<Npp16u> image16u = makeStructured<Npp16u>(size, size, 1, pattern);
    runHog<Npp16u, 1>("nppiHistogramOfGradientsBorder_16u32f_C1R_Ctx",
                      "16u/pattern" + std::to_string(pattern), nppiHistogramOfGradientsBorder_16u32f_C1R_Ctx,
                      image16u, size, size, {size, size}, config, locations);
    const std::vector<Npp16s> image16s = makeStructured<Npp16s>(size, size, 1, pattern);
    runHog<Npp16s, 1>("nppiHistogramOfGradientsBorder_16s32f_C1R_Ctx",
                      "16s/pattern" + std::to_string(pattern), nppiHistogramOfGradientsBorder_16s32f_C1R_Ctx,
                      image16s, size, size, {size, size}, config, locations);
    const std::vector<Npp32f> image32f = makeStructured<Npp32f>(size, size, 1, pattern);
    runHog<Npp32f, 1>("nppiHistogramOfGradientsBorder_32f_C1R_Ctx",
                      "32f/pattern" + std::to_string(pattern), nppiHistogramOfGradientsBorder_32f_C1R_Ctx,
                      image32f, size, size, {size, size}, config, locations);
  }
}

TEST(NppiHogDiff, Color) {
  const int size = 32;
  NppiHOGConfig config{8, 16, 9, {size, size}};
  const std::vector<NppiPoint> locations = {{0, 0}};
  for (int pattern = 6; pattern <= 8; ++pattern) {
    const std::vector<Npp8u> image = makeStructured<Npp8u>(size, size, 3, pattern);
    runHog<Npp8u, 3>("nppiHistogramOfGradientsBorder_8u32f_C3R_Ctx",
                     "8uC3/pattern" + std::to_string(pattern),
                     nppiHistogramOfGradientsBorder_8u32f_C3R_Ctx, image, size, size, {size, size}, config,
                     locations);
  }
}

TEST(NppiHogDiff, MultipleLocations) {
  const int size = 64;
  NppiHOGConfig config{8, 16, 9, {32, 32}};
  const std::vector<NppiPoint> locations = {{0, 0}, {16, 16}, {32, 32}};
  for (int pattern = 0; pattern <= 1; ++pattern) {
    const std::vector<Npp8u> image = makeStructured<Npp8u>(size, size, 1, pattern);
    runHog<Npp8u, 1>("nppiHistogramOfGradientsBorder_8u32f_C1R_Ctx",
                     "multi/pattern" + std::to_string(pattern),
                     nppiHistogramOfGradientsBorder_8u32f_C1R_Ctx, image, size, size, {size, size}, config,
                     locations);
  }
}

TEST(NppiHogDiff, RoiSmallerThanImage) {
  const int size = 32;
  const std::vector<Npp8u> image = makeStructured<Npp8u>(size, size, 1, 0);
  NppiHOGConfig config{4, 8, 4, {8, 8}};
  const std::vector<NppiPoint> locations = {{8, 0}};
  runHog<Npp8u, 1>("nppiHistogramOfGradientsBorder_8u32f_C1R_Ctx", "roi/edge",
                   nppiHistogramOfGradientsBorder_8u32f_C1R_Ctx, image, size, size, {16, 16}, config,
                   locations);
}

TEST(NppiHogDiff, Errors) {
  NppiHOGConfig good{8, 16, 9, {16, 16}};
  NppiHOGConfig badCell{1, 2, 9, {16, 16}};
  runHogStatusOnly("nppiHistogramOfGradientsBorder_8u32f_C1R_Ctx", "error/bad_cell", 16, 16, {16, 16},
                   badCell, {{0, 0}}, 128, NPP_BORDER_REPLICATE);
  runHogStatusOnly("nppiHistogramOfGradientsBorder_8u32f_C1R_Ctx", "error/bad_border", 16, 16, {16, 16},
                   good, {{0, 0}}, 128, NPP_BORDER_MIRROR);
  runHogStatusOnly("nppiHistogramOfGradientsBorder_8u32f_C1R_Ctx", "error/location_outside", 16, 16,
                   {16, 16}, good, {{8, 0}}, 128, NPP_BORDER_REPLICATE);
  runHogStatusOnly("nppiHistogramOfGradientsBorder_8u32f_C1R_Ctx", "error/roi_zero", 16, 16, {0, 16},
                   good, {{0, 0}}, 128, NPP_BORDER_REPLICATE);
  runHogStatusOnly("nppiHistogramOfGradientsBorder_8u32f_C1R_Ctx", "error/step_short", 16, 16,
                   {16, 16}, good, {{0, 0}, {0, 0}}, 32, NPP_BORDER_REPLICATE);
  runHogStatusOnly("nppiHistogramOfGradientsBorder_8u32f_C1R_Ctx", "error/step_odd", 16, 16, {16, 16},
                   good, {{0, 0}}, 129, NPP_BORDER_REPLICATE);
  runHogStatusOnly("nppiHistogramOfGradientsBorder_8u32f_C1R_Ctx", "error/no_locations", 16, 16,
                   {16, 16}, good, {}, 128, NPP_BORDER_REPLICATE);
}
