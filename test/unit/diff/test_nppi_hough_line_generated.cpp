// Differential tests for the Hough line transforms.
//
// Record on the RTX 5090 against NVIDIA NPP 13.4:
//   tools/remote5090.sh record 'NppiHoughLineDiff.*'
//
// Accumulator model decoded from 5090 dumps: theta-major int[], only source
// pixels equal to 255 vote, rhoBin = floor(rho / delta.rho + R) with
// R = floor(max(w,h)*sqrt(2)/2) and rho relative to the ROI centre, and the
// emitted rho/theta are bin * delta.  The tests use delta = (1,1) clean lines
// with a threshold that leaves exactly one candidate: NPP's multi-candidate
// ordering comes from unordered parallel writes, so it is not reproducible.
#include "npp_diff_harness.h"
#include "npp_test_base.h"

#include <cstdint>
#include <string>
#include <vector>

using namespace npp_diff;
using namespace npp_functional_test;

namespace {

NppStreamContext current_context() {
  NppStreamContext ctx{};
  NppStatus st = nppGetStreamContext(&ctx);
  EXPECT_EQ(st, NPP_NO_ERROR) << "nppGetStreamContext failed";
  return ctx;
}

void status_case(const std::string &id, const char *fname, NppStatus st) {
  Case c(id, fname);
  c.status(st);
  c.finish();
}

struct LineImage {
  int w, h;
  std::vector<Npp8u> data;
};

LineImage makeLine(int w, int h, int kind) {
  LineImage img{w, h, std::vector<Npp8u>((size_t)w * h, 0)};
  if (kind == 0) {  // horizontal at y = h/4
    for (int x = 0; x < w; ++x) img.data[(size_t)(h / 4) * w + x] = 255;
  } else if (kind == 1) {  // vertical at x = w/4
    for (int y = 0; y < h; ++y) img.data[(size_t)y * w + w / 4] = 255;
  } else if (kind == 2) {  // main diagonal
    for (int i = 0; i < w && i < h; ++i) img.data[(size_t)i * w + i] = 255;
  } else {  // anti-diagonal
    for (int i = 0; i < w && i < h; ++i) img.data[(size_t)i * w + (w - 1 - i)] = 255;
  }
  return img;
}

void line_case(const std::string &id, const char *fname, const LineImage &img, int threshold, int maxLines,
               bool useRegion, NppPointPolar lo, NppPointPolar hi) {
  int bufferSize = 0;
  EXPECT_EQ(nppiFilterHoughLineGetBufferSize({img.w, img.h}, {1.0f, 1.0f}, maxLines, &bufferSize), NPP_NO_ERROR);

  DeviceMemory<Npp8u> src(img.data.size());
  src.copyFromHost(img.data);
  DeviceMemory<Npp8u> buffer(bufferSize);
  DeviceMemory<NppPointPolar> lines((size_t)maxLines);
  DeviceMemory<int> count(1);
  std::vector<NppPointPolar> lineInit((size_t)maxLines, NppPointPolar{-1.0f, -1.0f});
  std::vector<int> countInit(1, -12345);
  lines.copyFromHost(lineInit);
  count.copyFromHost(countInit);

  NppStreamContext ctx = current_context();
  NppStatus st;
  if (useRegion) {
    NppPointPolar bounds[2] = {lo, hi};
    st = nppiFilterHoughLineRegion_8u32f_C1R_Ctx(src.get(), img.w, {img.w, img.h}, {1.0f, 1.0f}, threshold,
                                                 lines.get(), bounds, maxLines, count.get(), buffer.get(), ctx);
  } else {
    st = nppiFilterHoughLine_8u32f_C1R_Ctx(src.get(), img.w, {img.w, img.h}, {1.0f, 1.0f}, threshold, lines.get(),
                                           maxLines, count.get(), buffer.get(), ctx);
  }
  clear_cuda_error();

  Case c(id, fname);
  c.status(st);
  c.input(img.data.data(), img.data.size());
  if (st == NPP_NO_ERROR) {
    std::vector<int> hostCountVec(1);
    count.copyToHost(hostCountVec);
    int hostCount = hostCountVec[0];
    std::vector<NppPointPolar> hostLines((size_t)maxLines);
    lines.copyToHost(hostLines);
    c.host_output(&hostCount, 1, ValueKind::I32);
    if (hostCount > 0) {
      const int n = hostCount < maxLines ? hostCount : maxLines;
      c.host_output(hostLines.data(), (size_t)n * sizeof(NppPointPolar), ValueKind::Bytes);
    }
  }
  c.finish();
}

}  // namespace

TEST(NppiHoughLineDiff, CleanLines) {
  const char *fname = "nppiFilterHoughLine_8u32f_C1R_Ctx";
  line_case("s2/Hough/horiz_64", fname, makeLine(64, 64, 0), 50, 8, false, {0, 0}, {0, 0});
  line_case("s2/Hough/vert_64", fname, makeLine(64, 64, 1), 50, 8, false, {0, 0}, {0, 0});
  line_case("s2/Hough/diag_64", fname, makeLine(64, 64, 2), 50, 8, false, {0, 0}, {0, 0});
  line_case("s2/Hough/antidiag_64", fname, makeLine(64, 64, 3), 50, 8, false, {0, 0}, {0, 0});
  line_case("s2/Hough/horiz_96x80", fname, makeLine(96, 80, 0), 62, 8, false, {0, 0}, {0, 0});
  line_case("s2/Hough/vert_96x80", fname, makeLine(96, 80, 1), 62, 8, false, {0, 0}, {0, 0});
}

TEST(NppiHoughLineDiff, RegionFilter) {
  const char *fname = "nppiFilterHoughLineRegion_8u32f_C1R_Ctx";
  line_case("s2/Hough/region_include", fname, makeLine(64, 64, 0), 50, 8, true, {0.0f, 0.0f}, {64.0f, 180.0f});
  line_case("s2/Hough/region_exclude", fname, makeLine(64, 64, 0), 50, 8, true, {0.0f, 100.0f}, {64.0f, 180.0f});
  line_case("s2/Hough/region_theta_hi", fname, makeLine(64, 64, 1), 50, 8, true, {0.0f, 0.0f}, {64.0f, 10.0f});
}

TEST(NppiHoughLineDiff, Contract) {
  const int w = 64, h = 64;
  LineImage img = makeLine(w, h, 0);
  DeviceMemory<Npp8u> src(img.data.size());
  src.copyFromHost(img.data);
  int bufferSize = 0;
  nppiFilterHoughLineGetBufferSize({w, h}, {1.0f, 1.0f}, 8, &bufferSize);
  DeviceMemory<Npp8u> buffer(bufferSize);
  DeviceMemory<NppPointPolar> lines(8);
  DeviceMemory<int> count(1);
  NppStreamContext ctx = current_context();
  const char *fname = "nppiFilterHoughLine_8u32f_C1R_Ctx";
  const char *regionName = "nppiFilterHoughLineRegion_8u32f_C1R_Ctx";
  NppPointPolar good[2] = {{0.0f, 0.0f}, {64.0f, 180.0f}};
  NppPointPolar inverted[2] = {{100.0f, 100.0f}, {10.0f, 10.0f}};

  status_case("s2/Hough/null_src", fname,
              nppiFilterHoughLine_8u32f_C1R_Ctx(nullptr, w, {w, h}, {1, 1}, 50, lines.get(), 8, count.get(),
                                                buffer.get(), ctx));
  status_case("s2/Hough/null_lines", fname,
              nppiFilterHoughLine_8u32f_C1R_Ctx(src.get(), w, {w, h}, {1, 1}, 50, nullptr, 8, count.get(),
                                                buffer.get(), ctx));
  status_case("s2/Hough/null_count", fname,
              nppiFilterHoughLine_8u32f_C1R_Ctx(src.get(), w, {w, h}, {1, 1}, 50, lines.get(), 8, nullptr,
                                                buffer.get(), ctx));
  status_case("s2/Hough/null_buffer", fname,
              nppiFilterHoughLine_8u32f_C1R_Ctx(src.get(), w, {w, h}, {1, 1}, 50, lines.get(), 8, count.get(),
                                                nullptr, ctx));
  status_case("s2/Hough/roi_zero", fname,
              nppiFilterHoughLine_8u32f_C1R_Ctx(src.get(), w, {0, h}, {1, 1}, 50, lines.get(), 8, count.get(),
                                                buffer.get(), ctx));
  status_case("s2/Hough/roi_neg", fname,
              nppiFilterHoughLine_8u32f_C1R_Ctx(src.get(), w, {-1, h}, {1, 1}, 50, lines.get(), 8, count.get(),
                                                buffer.get(), ctx));
  status_case("s2/Hough/drho_zero", fname,
              nppiFilterHoughLine_8u32f_C1R_Ctx(src.get(), w, {w, h}, {0, 1}, 50, lines.get(), 8, count.get(),
                                                buffer.get(), ctx));
  status_case("s2/Hough/drho_neg", fname,
              nppiFilterHoughLine_8u32f_C1R_Ctx(src.get(), w, {w, h}, {-1, 1}, 50, lines.get(), 8, count.get(),
                                                buffer.get(), ctx));
  status_case("s2/Hough/drho_3", fname,
              nppiFilterHoughLine_8u32f_C1R_Ctx(src.get(), w, {w, h}, {3, 1}, 50, lines.get(), 8, count.get(),
                                                buffer.get(), ctx));
  status_case("s2/Hough/drho_3p1", fname,
              nppiFilterHoughLine_8u32f_C1R_Ctx(src.get(), w, {w, h}, {3.1f, 1}, 50, lines.get(), 8, count.get(),
                                                buffer.get(), ctx));
  status_case("s2/Hough/dtheta_zero", fname,
              nppiFilterHoughLine_8u32f_C1R_Ctx(src.get(), w, {w, h}, {1, 0}, 50, lines.get(), 8, count.get(),
                                                buffer.get(), ctx));
  status_case("s2/Hough/dtheta_0p2", fname,
              nppiFilterHoughLine_8u32f_C1R_Ctx(src.get(), w, {w, h}, {1, 0.2f}, 50, lines.get(), 8, count.get(),
                                                buffer.get(), ctx));
  status_case("s2/Hough/dtheta_3", fname,
              nppiFilterHoughLine_8u32f_C1R_Ctx(src.get(), w, {w, h}, {1, 3}, 50, lines.get(), 8, count.get(),
                                                buffer.get(), ctx));
  status_case("s2/Hough/dtheta_3p1", fname,
              nppiFilterHoughLine_8u32f_C1R_Ctx(src.get(), w, {w, h}, {1, 3.1f}, 50, lines.get(), 8, count.get(),
                                                buffer.get(), ctx));
  status_case("s2/Hough/threshold_zero", fname,
              nppiFilterHoughLine_8u32f_C1R_Ctx(src.get(), w, {w, h}, {1, 1}, 0, lines.get(), 8, count.get(),
                                                buffer.get(), ctx));
  status_case("s2/Hough/threshold_neg", fname,
              nppiFilterHoughLine_8u32f_C1R_Ctx(src.get(), w, {w, h}, {1, 1}, -1, lines.get(), 8, count.get(),
                                                buffer.get(), ctx));
  status_case("s2/Hough/max_zero", fname,
              nppiFilterHoughLine_8u32f_C1R_Ctx(src.get(), w, {w, h}, {1, 1}, 50, lines.get(), 0, count.get(),
                                                buffer.get(), ctx));
  status_case("s2/Hough/max_neg", fname,
              nppiFilterHoughLine_8u32f_C1R_Ctx(src.get(), w, {w, h}, {1, 1}, 50, lines.get(), -1, count.get(),
                                                buffer.get(), ctx));
  status_case("s2/Hough/region_inverted", regionName,
              nppiFilterHoughLineRegion_8u32f_C1R_Ctx(src.get(), w, {w, h}, {1, 1}, 50, lines.get(), inverted, 8,
                                                      count.get(), buffer.get(), ctx));
  status_case("s2/Hough/region_valid", regionName,
              nppiFilterHoughLineRegion_8u32f_C1R_Ctx(src.get(), w, {w, h}, {1, 1}, 50, lines.get(), good, 8,
                                                      count.get(), buffer.get(), ctx));
  clear_cuda_error();
}
