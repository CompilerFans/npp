// CPU-golden coverage for the Hough line detectors
// (nppi_filtering_functions.h):
//   nppiFilterHoughLine_8u32f_C1R and nppiFilterHoughLineRegion_8u32f_C1R.
//
// Measured NPP 13.4 semantics (RTX 5090), decoded from accumulator probes:
//   * only source pixels equal to 255 vote;
//   * theta is sampled at one degree per bin over [0,180) (the nDelta.theta
//     step only scales the reported value), rho bins are one unit apart with
//     theta-major layout accumulator[theta + 180*rhoBin];
//   * a vote lands in rhoBin = floor((x-w/2)cos(theta) + (y-h/2)sin(theta) + R)
//     with R = floor(max(w,h)*sqrt(2)/2);
//   * candidates are strict 3x3 local maxima of the accumulator (theta wraps,
//     rho clamps) whose count is >= nThreshold; the reported values are
//     rhoBin*nDelta.rho and thetaBin*nDelta.theta;
//   * the region flavour keeps only lines inside the inclusive oDstROI bounds.
// The emission order of equally valid peaks is an implementation detail of the
// reference (parallel accumulation), so the tests compare the sorted set of
// (rho, theta) pairs together with the reported line count.  Only
// nDelta = (1,1) is modelled: other steps show reference binning quirks.
#pragma once

#include "num_nppi_geom.h"

#include <algorithm>

namespace num {
namespace hough {

struct HLine {
  int rho, theta;
  bool operator<(const HLine &o) const { return theta != o.theta ? theta < o.theta : rho < o.rho; }
  bool operator==(const HLine &o) const { return rho == o.rho && theta == o.theta; }
};

// Golden accumulator over a binary image (see the file comment).
inline std::vector<HLine> golden(const std::vector<Npp8u> &img, int w, int h, int threshold, int rhoLo,
                                 int rhoHi, int thetaLo, int thetaHi) {
  const int thetaBins = 180;
  const int maxDim = w > h ? w : h;
  const int rhoBins = (int)std::ceil((double)maxDim * std::sqrt(2.0));
  const int R = (int)std::floor((double)maxDim * std::sqrt(2.0) * 0.5);
  std::vector<int> acc((size_t)thetaBins * rhoBins, 0);
  for (int y = 0; y < h; ++y)
    for (int x = 0; x < w; ++x) {
      if (img[(size_t)y * w + x] != 255) continue;
      const double dx = (double)x - (double)(w / 2);
      const double dy = (double)y - (double)(h / 2);
      for (int t = 0; t < thetaBins; ++t) {
        const double rad = (double)t * (M_PI / 180.0);
        const double rho = dx * std::cos(rad) + dy * std::sin(rad);
        const int bin = (int)std::floor(rho + (double)R);
        if (bin >= 0 && bin < rhoBins) ++acc[(size_t)t + (size_t)thetaBins * bin];
      }
    }
  std::vector<HLine> out;
  for (int r = 0; r < rhoBins; ++r)
    for (int t = 0; t < thetaBins; ++t) {
      const int v = acc[(size_t)t + (size_t)thetaBins * r];
      if (v < threshold) continue;
      bool isMax = true;
      for (int dr = -1; dr <= 1 && isMax; ++dr) {
        const int rr = r + dr;
        if (rr < 0 || rr >= rhoBins) continue;
        for (int dt = -1; dt <= 1; ++dt) {
          if (dr == 0 && dt == 0) continue;
          int tt = t + dt;
          if (tt < 0) tt += thetaBins;
          if (tt >= thetaBins) tt -= thetaBins;
          if (acc[(size_t)tt + (size_t)thetaBins * rr] > v) {
            isMax = false;
            break;
          }
        }
      }
      if (!isMax) continue;
      if (r < rhoLo || r > rhoHi || t < thetaLo || t > thetaHi) continue;
      out.push_back({r, t});
    }
  return out;
}

inline bool check_lines(const char *id, const GenCase &c, const std::vector<HLine> &got,
                        const std::vector<HLine> &want, RunCounts &cnt, std::string &first) {
  auto g = got, w = want;
  std::sort(g.begin(), g.end());
  std::sort(w.begin(), w.end());
  ++cnt.compared;
  if (g.size() != w.size()) {
    char b[320];
    std::snprintf(b, sizeof(b), "[%s] %s line count got=%zu want=%zu", id, c.id().c_str(), g.size(), w.size());
    if (first.empty()) first = b;
    EXPECT_TRUE(false) << b;
    ++cnt.mismatched;
    return false;
  }
  bool ok = true;
  for (size_t i = 0; i < g.size(); ++i) {
    ++cnt.compared;
    if (g[i] == w[i]) continue;
    char b[320];
    std::snprintf(b, sizeof(b), "[%s] %s line[%zu] got=(%d,%d) want=(%d,%d)", id, c.id().c_str(), i, g[i].rho,
                  g[i].theta, w[i].rho, w[i].theta);
    if (first.empty()) first = b;
    EXPECT_TRUE(false) << b;
    ++cnt.mismatched;
    ok = false;
    if (cnt.mismatched >= 8) return false;
  }
  return ok;
}

// Binary line patterns (all pixels 0 or 255).  The lines pass through the
// image centre, where the reference's accumulator bins are unambiguous.
inline std::vector<Npp8u> make_pattern(int w, int h, int pat) {
  std::vector<Npp8u> img((size_t)w * h, 0);
  auto set = [&](int x, int y) {
    if (x >= 0 && x < w && y >= 0 && y < h) img[(size_t)y * w + x] = 255;
  };
  const int m = w < h ? w : h;
  if (pat == 0) {
    for (int x = 4; x < w - 4; ++x) set(x, h / 2);           // horizontal
  } else if (pat == 1) {
    for (int y = 4; y < h - 4; ++y) set(w / 2, y);           // vertical
  } else if (pat == 2) {
    for (int t = 6; t < m - 6; ++t) set(t, t);               // 45-degree diagonal
  } else if (pat == 3) {
    for (int x = 0; x < w; ++x) set(x, h / 2);               // centred cross
    for (int y = 0; y < h; ++y) set(w / 2, y);
  } else {
    for (int t = 6; t < m - 6; ++t) set(t, h - 1 - t);       // anti-diagonal
  }
  return img;
}

struct HCase {
  int w, h, pat, threshold;
};

// Thresholds sit above the strongest spurious bin of these images (measured),
// so the reference reports exactly the full-count peak of each drawn line.
inline std::vector<HCase> cases() {
  return {{64, 64, 0, 45}, {64, 64, 1, 45}, {64, 64, 2, 45}, {64, 64, 3, 50},
          {64, 64, 4, 45}, {96, 64, 0, 60}, {96, 64, 1, 45}, {96, 64, 3, 50},
          {48, 48, 0, 30}, {48, 48, 2, 30}};
}

template <class Call>
void run_hough(const char *id, Call call) {
  NppStreamContext ctx{};
  if (nppGetStreamContext(&ctx) != NPP_NO_ERROR) {
    GTEST_SKIP() << "[" << id << "] no stream context";
    return;
  }
  RunCounts cnt;
  std::string first;
  bool failed = false;
  for (const HCase &c : cases()) {
    if (failed) break;
    const int W = c.w, H = c.h;
    const std::vector<Npp8u> img = make_pattern(W, H, c.pat);
    Buf<Npp8u> s(W, H, 1, 0, 0);
    for (int y = 0; y < H; ++y)
      std::memcpy(s.host.data() + (size_t)y * s.step, img.data() + (size_t)y * W, (size_t)W);
    s.dev.copyFromHost(s.host);
    const int maxLines = 200;
    npp_functional_test::DeviceMemory<NppPointPolar> lines(maxLines);
    npp_functional_test::DeviceMemory<int> lineCount(1);
    npp_functional_test::DeviceMemory<Npp8u> buf(1);
    int need = 0;
    nppiFilterHoughLineGetBufferSize(NppiSize{W, H}, NppPointPolar{1.0f, 1.0f}, maxLines, &need);
    buf.allocate((size_t)(need > 0 ? need : 1));
    std::vector<NppPointPolar> zero(maxLines);
    lines.copyFromHost(zero);
    std::vector<int> zc(1, 0);
    lineCount.copyFromHost(zc);
    npp_diff::clear_cuda_error();
    const NppStatus st = call(reinterpret_cast<const Npp8u *>(s.dptr()), s.step, NppiSize{W, H},
                              NppPointPolar{1.0f, 1.0f}, c.threshold, lines.get(), maxLines, lineCount.get(),
                              buf.get(), ctx);
    if (st != NPP_NO_ERROR) {
      ++cnt.skipped;
      if (st == NPP_NOT_IMPLEMENTED_ERROR) ++cnt.notimpl;
      continue;
    }
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess) << "[" << id << "] sync";
    ++cnt.executed;
    int n = 0;
    cudaMemcpy(&n, lineCount.get(), sizeof(int), cudaMemcpyDeviceToHost);
    std::vector<NppPointPolar> host(maxLines);
    cudaMemcpy(host.data(), lines.get(), host.size() * sizeof(NppPointPolar), cudaMemcpyDeviceToHost);
    std::vector<HLine> got;
    for (int i = 0; i < n && i < maxLines; ++i)
      got.push_back({(int)std::lround(host[i].rho), (int)std::lround(host[i].theta)});
    const std::vector<HLine> want = golden(img, W, H, c.threshold, 0, 1000, 0, 1000);
    const GenCase dc{W, H, c.pat, 0, 0, kSmall, c.threshold};
    if (!check_lines(id, dc, got, want, cnt, first)) failed = true;
    if (cnt.mismatched >= 8) failed = true;
  }
  finish(id, cnt, first);
}

template <class Call>
void run_hough_region(const char *id, Call call) {
  NppStreamContext ctx{};
  if (nppGetStreamContext(&ctx) != NPP_NO_ERROR) {
    GTEST_SKIP() << "[" << id << "] no stream context";
    return;
  }
  RunCounts cnt;
  std::string first;
  // Windows: everything, just the horizontal peak, just the vertical peak, and
  // an empty window.
  struct RCase {
    int w, h, pat, threshold, rhoLo, rhoHi, thetaLo, thetaHi;
  };
  static const RCase rc[] = {{64, 64, 3, 50, 0, 200, 0, 200},
                             {64, 64, 3, 50, 40, 50, 85, 95},
                             {64, 64, 3, 50, 40, 50, 0, 0},
                             {64, 64, 3, 50, 100, 120, 0, 200}};
  bool failed = false;
  for (const RCase &c : rc) {
    if (failed) break;
    const int W = c.w, H = c.h;
    const std::vector<Npp8u> img = make_pattern(W, H, c.pat);
    Buf<Npp8u> s(W, H, 1, 0, 0);
    for (int y = 0; y < H; ++y)
      std::memcpy(s.host.data() + (size_t)y * s.step, img.data() + (size_t)y * W, (size_t)W);
    s.dev.copyFromHost(s.host);
    const int maxLines = 200;
    npp_functional_test::DeviceMemory<NppPointPolar> lines(maxLines);
    npp_functional_test::DeviceMemory<int> lineCount(1);
    npp_functional_test::DeviceMemory<Npp8u> buf(1);
    int need = 0;
    nppiFilterHoughLineGetBufferSize(NppiSize{W, H}, NppPointPolar{1.0f, 1.0f}, maxLines, &need);
    buf.allocate((size_t)(need > 0 ? need : 1));
    std::vector<NppPointPolar> zero(maxLines);
    lines.copyFromHost(zero);
    std::vector<int> zc(1, 0);
    lineCount.copyFromHost(zc);
    NppPointPolar roi[2] = {{(float)c.rhoLo, (float)c.thetaLo}, {(float)c.rhoHi, (float)c.thetaHi}};
    npp_diff::clear_cuda_error();
    const NppStatus st = call(reinterpret_cast<const Npp8u *>(s.dptr()), s.step, NppiSize{W, H},
                              NppPointPolar{1.0f, 1.0f}, c.threshold, lines.get(), roi, maxLines,
                              lineCount.get(), buf.get(), ctx);
    if (st != NPP_NO_ERROR) {
      ++cnt.skipped;
      if (st == NPP_NOT_IMPLEMENTED_ERROR) ++cnt.notimpl;
      continue;
    }
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess) << "[" << id << "] sync";
    ++cnt.executed;
    int n = 0;
    cudaMemcpy(&n, lineCount.get(), sizeof(int), cudaMemcpyDeviceToHost);
    std::vector<NppPointPolar> host(maxLines);
    cudaMemcpy(host.data(), lines.get(), host.size() * sizeof(NppPointPolar), cudaMemcpyDeviceToHost);
    std::vector<HLine> got;
    for (int i = 0; i < n && i < maxLines; ++i)
      got.push_back({(int)std::lround(host[i].rho), (int)std::lround(host[i].theta)});
    const std::vector<HLine> want =
        golden(img, W, H, c.threshold, c.rhoLo, c.rhoHi, c.thetaLo, c.thetaHi);
    const GenCase dc{W, H, c.pat, 0, 0, kSmall, c.rhoLo * 100000 + c.rhoHi};
    if (!check_lines(id, dc, got, want, cnt, first)) failed = true;
    if (cnt.mismatched >= 8) failed = true;
  }
  finish(id, cnt, first);
}

}  // namespace hough
}  // namespace num
