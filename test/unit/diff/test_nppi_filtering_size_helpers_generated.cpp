// Differential tests for the host-only filtering scratch-size helpers.
//
// Record on the RTX 5090 against NVIDIA NPP 13.4:
//   NPP_RECORD_DIR=... ./diff_tests --gtest_filter='NppiFilteringSizeDiff.*'
// Verify against mcMPP:
//   NPP_VERIFY_DIR=artifacts/5090/npp_records ./diff_tests \
//     --gtest_filter='NppiFilteringSizeDiff.*'
//
// Null output pointers are not covered: NPP dereferences several of them.
#include "npp_diff_harness.h"

#include <cstdio>
#include <string>

using namespace npp_diff;

namespace {

std::string tag(int value) { return std::to_string(value); }

std::string ftag(float value) {
  char buf[32];
  std::snprintf(buf, sizeof(buf), "%.9g", value);
  return buf;
}

std::string id(const char *helper, const std::string &a = "", const std::string &b = "",
               const std::string &c = "", const std::string &d = "", const std::string &e = "") {
  std::string out = std::string("filterSize/") + helper;
  for (const std::string *part : {&a, &b, &c, &d, &e}) {
    if (!part->empty()) {
      out += "/";
      out += *part;
    }
  }
  return out;
}

void runBox(const char *fn, bool sixtyFour, int w, int h, int channels) {
  int size = -12345;
  clear_cuda_error();
  NppStatus st = sixtyFour ? nppiFilterBoxBorderAdvancedGetDeviceBufferSize_64({w, h}, channels, &size)
                           : nppiFilterBoxBorderAdvancedGetDeviceBufferSize({w, h}, channels, &size);
  Case c(id(fn, tag(w), tag(h), tag(channels)), fn);
  c.status(st);
  c.host_output<int>(&size, 1, ValueKind::I32);
  c.finish();
}

void runHarris(int w, int h) {
  int size = -12345;
  clear_cuda_error();
  NppStatus st = nppiFilterHarrisCornersBorderGetBufferSize({w, h}, &size);
  Case c(id("nppiFilterHarrisCornersBorderGetBufferSize", tag(w), tag(h)),
         "nppiFilterHarrisCornersBorderGetBufferSize");
  c.status(st);
  c.host_output<int>(&size, 1, ValueKind::I32);
  c.finish();
}

void runHough(int w, int h, float rho, float theta, int maxLines) {
  int size = -12345;
  clear_cuda_error();
  NppStatus st = nppiFilterHoughLineGetBufferSize({w, h}, {rho, theta}, maxLines, &size);
  Case c(id("nppiFilterHoughLineGetBufferSize", tag(w), tag(h), ftag(rho), ftag(theta), tag(maxLines)),
         "nppiFilterHoughLineGetBufferSize");
  c.status(st);
  c.host_output<int>(&size, 1, ValueKind::I32);
  c.finish();
}

void runHogDescriptors(int cell, int block, int bins, int windowW, int windowH, int locations) {
  NppiHOGConfig config{cell, block, bins, {windowW, windowH}};
  int size = -12345;
  clear_cuda_error();
  NppStatus st = nppiHistogramOfGradientsBorderGetDescriptorsSize(config, locations, &size);
  Case c(id("nppiHistogramOfGradientsBorderGetDescriptorsSize", tag(cell), tag(block), tag(bins),
            tag(windowW) + "x" + tag(windowH), tag(locations)),
         "nppiHistogramOfGradientsBorderGetDescriptorsSize");
  c.status(st);
  c.host_output<int>(&size, 1, ValueKind::I32);
  c.finish();
}

void runHogBuffer(const char *key, int cell, int block, int bins, int windowW, int windowH,
                  const NppiPoint *locations, int count, int roiW, int roiH) {
  NppiHOGConfig config{cell, block, bins, {windowW, windowH}};
  int size = -12345;
  clear_cuda_error();
  NppStatus st = nppiHistogramOfGradientsBorderGetBufferSize(config, locations, count,
                                                             {roiW, roiH}, &size);
  Case c(id("nppiHistogramOfGradientsBorderGetBufferSize", key, tag(cell), tag(block), tag(bins),
            tag(windowW) + "x" + tag(windowH)),
         "nppiHistogramOfGradientsBorderGetBufferSize");
  c.status(st);
  c.host_output<int>(&size, 1, ValueKind::I32);
  c.finish();
}

void runWatershed(int w, int h) {
  size_t size = static_cast<size_t>(-12345);
  clear_cuda_error();
  NppStatus st = nppiSegmentWatershedGetBufferSize_16u_C1R({w, h}, &size);
  Case c(id("nppiSegmentWatershedGetBufferSize_16u_C1R", tag(w), tag(h)),
         "nppiSegmentWatershedGetBufferSize_16u_C1R");
  c.status(st);
  c.host_output<size_t>(&size, 1, ValueKind::U64);
  c.finish();
}

void runPbaAa(int w, int h) {
  size_t size = static_cast<size_t>(-12345);
  clear_cuda_error();
  NppStatus st = nppiSignedDistanceTransformPBAGetAntialiasingBufferSize({w, h}, &size);
  Case c(id("nppiSignedDistanceTransformPBAGetAntialiasingBufferSize", tag(w), tag(h)),
         "nppiSignedDistanceTransformPBAGetAntialiasingBufferSize");
  c.status(st);
  c.host_output<size_t>(&size, 1, ValueKind::U64);
  c.finish();
}

}  // namespace

TEST(NppiFilteringSizeDiff, Box) {
  const int sizes[][2] = {{1, 1}, {2, 3}, {7, 8}, {17, 13}, {64, 40}, {0, 5}, {5, 0}, {-1, 5}};
  const int channels[] = {0, 1, 2, 3, 4, 5};
  for (const auto &size : sizes) {
    for (int channelsValue : channels) {
      runBox("nppiFilterBoxBorderAdvancedGetDeviceBufferSize", false, size[0], size[1],
             channelsValue);
      runBox("nppiFilterBoxBorderAdvancedGetDeviceBufferSize_64", true, size[0], size[1],
             channelsValue);
    }
  }
}

TEST(NppiFilteringSizeDiff, Harris) {
  const int values[] = {-2, -1, 0, 1, 2, 7, 17, 64, 1000};
  for (int w : values) {
    for (int h : values) {
      runHarris(w, h);
    }
  }
}

TEST(NppiFilteringSizeDiff, Hough) {
  const int sizes[] = {1, 8, 17, 32};
  const float deltas[] = {0.0f, 0.25f, 0.5f, 1.0f, 2.0f, 3.0f, 3.01f};
  const int counts[] = {0, 1, 10};
  for (int w : sizes) {
    for (int h : sizes) {
      for (float rho : deltas) {
        for (float theta : deltas) {
          for (int count : counts) {
            runHough(w, h, rho, theta, count);
          }
        }
      }
    }
  }
}

TEST(NppiFilteringSizeDiff, HogDescriptors) {
  struct Config {
    int cell, block, bins, w, h;
  };
  const Config configs[] = {{8, 16, 9, 64, 128}, {4, 8, 9, 32, 64},  {2, 4, 9, 16, 32},
                            {8, 16, 12, 128, 128}, {1, 2, 9, 16, 32}, {8, 24, 9, 64, 128},
                            {8, 16, 1, 64, 128},   {8, 16, 9, 65, 128}, {0, 16, 9, 64, 128}};
  const int locationCounts[] = {0, 1, 3, 128};
  for (const Config &config : configs) {
    for (int locations : locationCounts) {
      runHogDescriptors(config.cell, config.block, config.bins, config.w, config.h, locations);
    }
  }
}

TEST(NppiFilteringSizeDiff, HogBuffer) {
  const NppiPoint single[] = {{0, 0}};
  const NppiPoint two[] = {{0, 0}, {1, 1}};
  const NppiPoint three[] = {{0, 0}, {1, 1}, {100, 100}};
  const NppiPoint negative[] = {{-1, 0}};
  const NppiPoint outside[] = {{1, 1}};
  const NppiPoint one[] = {{0, 0}};
  const NppiPoint none[] = {{0, 0}};

  runHogBuffer("fits", 8, 16, 9, 64, 128, single, 1, 64, 128);
  runHogBuffer("fits2", 8, 16, 9, 64, 128, two, 2, 65, 129);
  runHogBuffer("fits3", 8, 16, 9, 64, 128, three, 3, 164, 228);
  runHogBuffer("negative", 8, 16, 9, 64, 128, negative, 1, 64, 128);
  runHogBuffer("outside", 8, 16, 9, 64, 128, outside, 1, 64, 128);
  runHogBuffer("zero_locations", 8, 16, 9, 64, 128, none, 0, 64, 128);
  runHogBuffer("too_many_locations", 8, 16, 9, 64, 128, one, 1025, 64, 128);
  runHogBuffer("roi_zero", 8, 16, 9, 64, 128, single, 1, 0, 128);
  runHogBuffer("roi_negative", 8, 16, 9, 64, 128, single, 1, -1, 128);
  runHogBuffer("small_cfg", 4, 8, 9, 32, 64, single, 1, 32, 64);
  runHogBuffer("tiny_cfg", 2, 4, 9, 16, 32, single, 1, 16, 32);
  runHogBuffer("bins12", 8, 16, 12, 128, 128, single, 1, 128, 128);
  runHogBuffer("bad_ratio", 8, 32, 9, 64, 128, single, 1, 64, 128);
  runHogBuffer("bad_cell", 1, 2, 9, 16, 32, single, 1, 16, 32);
  runHogBuffer("bad_bins", 8, 16, 1, 64, 128, single, 1, 64, 128);
  runHogBuffer("bad_window", 8, 16, 9, 65, 128, single, 1, 65, 128);
  runHogBuffer("too_many_cells", 16, 32, 9, 512, 512, single, 1, 512, 512);
}

TEST(NppiFilteringSizeDiff, Watershed) {
  const int values[] = {-2, -1, 0, 1, 2, 3, 7, 8, 9, 17, 31, 32, 33, 64, 100};
  for (int w : values) {
    for (int h : values) {
      runWatershed(w, h);
    }
  }
}

TEST(NppiFilteringSizeDiff, PbaAntialiasing) {
  const int values[] = {-2, -1, 0, 1, 2, 3, 7, 8, 9, 17, 31, 32, 33, 64, 100};
  for (int w : values) {
    for (int h : values) {
      runPbaAa(w, h);
    }
  }
}
