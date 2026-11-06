// Coverage for the compressed-marker-label contour pipeline
// (nppi_filtering_functions.h):
//   nppiCompressedMarkerLabelsUFInfo_32u_C1R,
//   nppiCompressedMarkerLabelsUFContoursGenerateGeometryLists_C1R,
//   nppiContoursImageMarchingSquaresInterpolation_{32f,64f}_C1R.
//
// Measured NPP 13.4 conventions (RTX 5090):
//   * a pixel is a contour pixel of its label when any of its eight
//     neighbours carries a different label; every label (including label 0)
//     is traced, and the per-label contour counts are returned in
//     pContoursPixelCountsListHost with pContoursPixelStartingOffsetHost as
//     the running prefix sum;
//   * the contours image stores 0 at contour pixels and 255 elsewhere, so
//     the number of 0 pixels equals nTotalImagePixelContourCount;
//   * nLongestImageContourPixelCount is the largest per-label contour count.
//
// The information structs use layouts that differ between the installed
// reference ABI and the unified public header, so the struct arrays are only
// exercised for status and bounds; the plain integer lists, the contours
// image and the totals are compared against the CPU golden.
#pragma once

#include "num_nppi_geom.h"

#include <cmath>
#include <cstring>

namespace num {
namespace contours {

struct Setup {
  int w = 0, h = 0;
  std::vector<Npp32u> labels;
  std::vector<Npp32u> counts;
  std::vector<Npp32u> offsets;
  Npp32u total = 0;
  Npp32u longest = 0;
  std::vector<Npp8u> contourImage;
};

inline void buildSetup(Setup &s, int w, int h, int variant) {
  s.w = w;
  s.h = h;
  s.labels.assign((size_t)w * h, 0);
  for (int y = 6; y < 20; ++y)
    for (int x = 6; x < 24; ++x) s.labels[(size_t)y * w + x] = 1;
  for (int y = 8; y < 16; ++y)
    for (int x = 34; x < 46; ++x) s.labels[(size_t)y * w + x] = 2;
  if (variant == 0) {
    for (int y = 26; y < 40; ++y)
      for (int x = 10; x < 18; ++x) s.labels[(size_t)y * w + x] = 3;
    for (int y = 34; y < 40; ++y)
      for (int x = 18; x < 34; ++x) s.labels[(size_t)y * w + x] = 3;
  } else {
    // single-pixel-wide diagonal island (exercises touching corners)
    for (int t = 8; t < 20; ++t) s.labels[(size_t)(30 + t - 8) * w + (50 + (t - 8))] = 4;
  }
  const Npp32u maxLabel = variant == 0 ? 3 : 4;
  s.counts.assign(maxLabel + 1, 0);
  s.contourImage.assign((size_t)w * h, 255);
  for (int y = 0; y < h; ++y)
    for (int x = 0; x < w; ++x) {
      const Npp32u lab = s.labels[(size_t)y * w + x];
      if (lab > maxLabel) continue;
      bool contour = false;
      for (int dy = -1; dy <= 1 && !contour; ++dy)
        for (int dx = -1; dx <= 1; ++dx) {
          if (dx == 0 && dy == 0) continue;
          const int nx = x + dx, ny = y + dy;
          if (nx < 0 || nx >= w || ny < 0 || ny >= h) continue;
          if (s.labels[(size_t)ny * w + nx] != lab) {
            contour = true;
            break;
          }
        }
      if (contour) {
        s.contourImage[(size_t)y * w + x] = 0;
        ++s.counts[lab];
      }
    }
  s.offsets.assign(maxLabel + 1, 0);
  s.total = 0;
  s.longest = 0;
  for (Npp32u i = 0; i <= maxLabel; ++i) {
    s.offsets[i] = s.total;
    s.total += s.counts[i];
    if (s.counts[i] > s.longest) s.longest = s.counts[i];
  }
}

inline bool check_info(const char *id, const GenCase &c, const Setup &s, const std::vector<Npp32u> &counts,
                       const std::vector<Npp32u> &offsets, Npp32u total, Npp32u longest,
                       const std::vector<Npp8u> &contours, RunCounts &cnt, std::string &first) {
  bool ok = true;
  auto fail = [&](const char *what, size_t i, double got, double want) {
    char b[360];
    std::snprintf(b, sizeof(b), "[%s] %s %s[%zu] got=%g want=%g", id, c.id().c_str(), what, i, got, want);
    if (first.empty()) first = b;
    EXPECT_TRUE(false) << b;
    ++cnt.mismatched;
    ok = false;
  };
  for (size_t i = 0; i < s.counts.size(); ++i) {
    ++cnt.compared;
    if (counts[i] != s.counts[i]) fail("count", i, counts[i], s.counts[i]);
    if (offsets[i] != s.offsets[i]) fail("offset", i, offsets[i], s.offsets[i]);
  }
  ++cnt.compared;
  if (total != s.total) fail("totalContours", 0, total, s.total);
  ++cnt.compared;
  if (longest != s.longest) fail("longest", 0, longest, s.longest);
  size_t zeroPixels = 0;
  for (int y = 0; y < s.h; ++y)
    for (int x = 0; x < s.w; ++x) {
      const size_t i = (size_t)y * s.w + x;
      ++cnt.compared;
      if (contours[i] != s.contourImage[i]) fail("contoursImage", i, contours[i], s.contourImage[i]);
      zeroPixels += contours[i] == 0 ? 1 : 0;
    }
  ++cnt.compared;
  if ((Npp32u)zeroPixels != s.total) fail("zeroPixels", 0, (double)zeroPixels, (double)s.total);
  return ok;
}

template <class Call>
void run_info(const char *id, Call call) {
  NppStreamContext ctx{};
  if (nppGetStreamContext(&ctx) != NPP_NO_ERROR) {
    GTEST_SKIP() << "[" << id << "] no stream context";
    return;
  }
  RunCounts cnt;
  std::string first;
  bool failed = false;
  static const int cfgs[][3] = {{64, 64, 0}, {96, 64, 0}, {48, 48, 1}};
  for (const auto &cf : cfgs) {
    if (failed) break;
    const int W = cf[0], H = cf[1], variant = cf[2];
    Setup s;
    buildSetup(s, W, H, variant);
    const unsigned maxId = variant == 0 ? 3 : 4;
    const unsigned maxLabel = maxId + 1;  // label count passed to the API
    const size_t count = s.labels.size();
    npp_functional_test::DeviceMemory<Npp32u> dLabels(count);
    dLabels.copyFromHost(s.labels);
    unsigned isize = 0;
    nppiCompressedMarkerLabelsUFGetInfoListSize_32u_C1R(maxLabel, &isize);
    npp_functional_test::DeviceMemory<Npp8u> info((size_t)(isize > 0 ? isize : 1));
    npp_functional_test::DeviceMemory<Npp8u> contourImg(count);
    const size_t dirBytes = count * sizeof(NppiContourPixelDirectionInfo);
    npp_functional_test::DeviceMemory<Npp8u> dir(dirBytes);
    npp_functional_test::DeviceMemory<Npp32u> countsDev(maxLabel + 2), offsDev(maxLabel + 2);
    std::vector<Npp32u> countsHost(maxLabel + 2, 0x5A5A5A5A), offsHost(maxLabel + 2, 0x5A5A5A5A);
    NppiContourTotalsInfo totals;
    std::memset(&totals, 0x5A, sizeof(totals));
    std::vector<Npp8u> cinit(count, 0x5A), dinit(dirBytes, 0);
    contourImg.copyFromHost(cinit);
    dir.copyFromHost(dinit);
    npp_diff::clear_cuda_error();
    const NppStatus st = call(dLabels.get(), W * 4, NppiSize{W, H}, maxLabel, info.get(), contourImg.get(), W,
                              dir.get(), W * (int)sizeof(NppiContourPixelDirectionInfo), &totals,
                              countsDev.get(), countsHost.data(), offsDev.get(), offsHost.data(), ctx);
    if (st != NPP_NO_ERROR) {
      ++cnt.skipped;
      if (st == NPP_NOT_IMPLEMENTED_ERROR) ++cnt.notimpl;
      continue;
    }
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess) << "[" << id << "] info sync";
    ++cnt.executed;
    std::vector<Npp8u> contours((size_t)W * H);
    cudaMemcpy(contours.data(), contourImg.get(), contours.size(), cudaMemcpyDeviceToHost);
    const GenCase dc{W, H, variant, 0, 0, kSmall, 0};
    if (!check_info(id, dc, s, countsHost, offsHost, totals.nTotalImagePixelContourCount,
                    totals.nLongestImageContourPixelCount, contours, cnt, first))
      failed = true;
    if (cnt.mismatched >= 8) failed = true;
  }
  finish(id, cnt, first);
}

// ---------------------------------------------------------------- geometry
inline void run_geometry(const char *id) {
  NppStreamContext ctx{};
  if (nppGetStreamContext(&ctx) != NPP_NO_ERROR) {
    GTEST_SKIP() << "[" << id << "] no stream context";
    return;
  }
  RunCounts cnt;
  std::string first;
  static const int cfgs[][3] = {{64, 64, 0}, {96, 64, 0}, {48, 48, 1}};
  for (const auto &cf : cfgs) {
    const int W = cf[0], H = cf[1], variant = cf[2];
    Setup s;
    buildSetup(s, W, H, variant);
    const unsigned maxId = variant == 0 ? 3 : 4;
    const unsigned maxLabel = maxId + 1;  // label count passed to the API
    const size_t count = s.labels.size();
    npp_functional_test::DeviceMemory<Npp32u> dLabels(count);
    dLabels.copyFromHost(s.labels);
    unsigned isize = 0;
    nppiCompressedMarkerLabelsUFGetInfoListSize_32u_C1R(maxLabel, &isize);
    npp_functional_test::DeviceMemory<Npp8u> info((size_t)(isize > 0 ? isize : 1));
    npp_functional_test::DeviceMemory<Npp8u> contourImg(count);
    const size_t dirBytes = count * sizeof(NppiContourPixelDirectionInfo);
    npp_functional_test::DeviceMemory<Npp8u> dir(dirBytes);
    npp_functional_test::DeviceMemory<Npp32u> countsDev(maxLabel + 2), offsDev(maxLabel + 2);
    std::vector<Npp32u> countsHost(maxLabel + 2), offsHost(maxLabel + 2);
    NppiContourTotalsInfo totals;
    std::memset(&totals, 0, sizeof(totals));
    std::vector<Npp8u> z(count, 0), zd(dirBytes, 0);
    contourImg.copyFromHost(z);
    dir.copyFromHost(zd);
    NppiContourTotalsInfo totalsInit = totals;
    npp_diff::clear_cuda_error();
    NppStatus st = nppiCompressedMarkerLabelsUFInfo_32u_C1R_Ctx(
        dLabels.get(), W * 4, NppiSize{W, H}, maxLabel, (NppiCompressedMarkerLabelsInfo *)info.get(),
        contourImg.get(), W, (NppiContourPixelDirectionInfo *)dir.get(), W * (int)sizeof(NppiContourPixelDirectionInfo),
        &totals, countsDev.get(), countsHost.data(), offsDev.get(), offsHost.data(), ctx);
    if (st != NPP_NO_ERROR) {
      ++cnt.skipped;
      continue;
    }
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess) << "[" << id << "] info sync";
    std::vector<Npp8u> infoHost((size_t)(isize > 0 ? isize : 1), 0);
    cudaMemcpy(infoHost.data(), info.get(), infoHost.size(), cudaMemcpyDeviceToHost);
    // Geometry lists sized from the total contour count.
    const Npp32u total = totals.nTotalImagePixelContourCount;
    unsigned gsize = 0;
    nppiCompressedMarkerLabelsUFGetGeometryListsSize_C1R(total, &gsize);
    npp_functional_test::DeviceMemory<Npp8u> geomDev((size_t)(gsize > 0 ? gsize : 1));
    std::vector<Npp8u> geomHost((size_t)(gsize > 0 ? gsize : 1), 0);
    npp_functional_test::DeviceMemory<Npp32u> foundDev(maxLabel + 2);
    std::vector<Npp32u> foundHost(maxLabel + 2, 0), geomOffsHost(maxLabel + 2, 0);
    npp_functional_test::DeviceMemory<Npp32u> geomOffsDev(maxLabel + 2);
    unsigned bsize = 0;
    nppiCompressedMarkerLabelsUFGetContoursBlockSegmentListSize_C1R(countsHost.data(), total, maxLabel, 0,
                                                                     maxLabel, &bsize);
    npp_functional_test::DeviceMemory<Npp8u> segDev((size_t)(bsize > 0 ? bsize : 1));
    std::vector<Npp8u> segHost((size_t)(bsize > 0 ? bsize : 1), 0);
    npp_diff::clear_cuda_error();
    st = nppiCompressedMarkerLabelsUFContoursGenerateGeometryLists_C1R_Ctx(
        (NppiCompressedMarkerLabelsInfo *)info.get(), (NppiCompressedMarkerLabelsInfo *)infoHost.data(),
        (NppiContourPixelDirectionInfo *)dir.get(),
        W * (int)sizeof(NppiContourPixelDirectionInfo), (NppiContourPixelGeometryInfo *)geomDev.get(),
        (NppiContourPixelGeometryInfo *)geomHost.data(), nullptr, 0, countsDev.get(), foundDev.get(),
        foundHost.data(), geomOffsDev.get(), geomOffsHost.data(), total, maxLabel, 0, maxLabel,
        (NppiContourBlockSegment *)segDev.get(), (NppiContourBlockSegment *)segHost.data(), 1, NppiSize{W, H}, ctx);
    if (st != NPP_NO_ERROR) {
      ++cnt.skipped;
      if (st == NPP_NOT_IMPLEMENTED_ERROR) ++cnt.notimpl;
      continue;
    }
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess) << "[" << id << "] geometry sync";
    ++cnt.executed;
    const GenCase dc{W, H, variant, 0, 0, kSmall, 0};
    // The returned starting offsets must be a non-decreasing partition of the
    // contour pixel list.
    ++cnt.compared;
    bool ordered = true;
    for (unsigned i = 0; i + 1 < maxLabel; ++i)
      if (geomOffsHost[i] > geomOffsHost[i + 1]) ordered = false;
    if (!ordered) {
      ++cnt.mismatched;
      EXPECT_TRUE(ordered) << "[" << id << "] " << dc.id() << " starting offsets not ordered";
    }
    ++cnt.compared;
    if (geomOffsHost[maxLabel - 1] > total) {
      ++cnt.mismatched;
      EXPECT_LE(geomOffsHost[maxLabel - 1], total) << "[" << id << "] " << dc.id() << " offset beyond contour count";
    }
    (void)totalsInit;
  }
  finish(id, cnt, first);
}

// ---------------------------------------------------------------- marching
template <typename P, bool Is64>
inline void run_marching_pair(const char *id) {
  NppStreamContext ctx{};
  if (nppGetStreamContext(&ctx) != NPP_NO_ERROR) {
    GTEST_SKIP() << "[" << id << "] no stream context";
    return;
  }
  RunCounts cnt;
  std::string first;
  static const int cfgs[][3] = {{64, 64, 0}, {96, 64, 0}};
  for (const auto &cf : cfgs) {
    const int W = cf[0], H = cf[1], variant = cf[2];
    Setup s;
    buildSetup(s, W, H, variant);
    const unsigned maxId = variant == 0 ? 3 : 4;
    const unsigned maxLabel = maxId + 1;  // label count passed to the API
    const size_t count = s.labels.size();
    npp_functional_test::DeviceMemory<Npp32u> dLabels(count);
    dLabels.copyFromHost(s.labels);
    unsigned isize = 0;
    nppiCompressedMarkerLabelsUFGetInfoListSize_32u_C1R(maxLabel, &isize);
    npp_functional_test::DeviceMemory<Npp8u> info((size_t)(isize > 0 ? isize : 1));
    npp_functional_test::DeviceMemory<Npp8u> contourImg(count);
    const size_t dirBytes = count * sizeof(NppiContourPixelDirectionInfo);
    npp_functional_test::DeviceMemory<Npp8u> dir(dirBytes);
    npp_functional_test::DeviceMemory<Npp32u> countsDev(maxLabel + 2), offsDev(maxLabel + 2);
    std::vector<Npp32u> countsHost(maxLabel + 2), offsHost(maxLabel + 2);
    NppiContourTotalsInfo totals;
    std::memset(&totals, 0, sizeof(totals));
    std::vector<Npp8u> z(count, 0), zd(dirBytes, 0);
    contourImg.copyFromHost(z);
    dir.copyFromHost(zd);
    npp_diff::clear_cuda_error();
    NppStatus st = nppiCompressedMarkerLabelsUFInfo_32u_C1R_Ctx(
        dLabels.get(), W * 4, NppiSize{W, H}, maxLabel, (NppiCompressedMarkerLabelsInfo *)info.get(),
        contourImg.get(), W, (NppiContourPixelDirectionInfo *)dir.get(), W * (int)sizeof(NppiContourPixelDirectionInfo),
        &totals, countsDev.get(), countsHost.data(), offsDev.get(), offsHost.data(), ctx);
    if (st != NPP_NO_ERROR) {
      ++cnt.skipped;
      continue;
    }
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess) << "[" << id << "] info sync";
    std::vector<Npp8u> infoHost((size_t)(isize > 0 ? isize : 1), 0);
    cudaMemcpy(infoHost.data(), info.get(), infoHost.size(), cudaMemcpyDeviceToHost);
    const Npp32u total = totals.nTotalImagePixelContourCount;
    unsigned gsize = 0;
    nppiCompressedMarkerLabelsUFGetGeometryListsSize_C1R(total, &gsize);
    npp_functional_test::DeviceMemory<Npp8u> geomDev((size_t)(gsize > 0 ? gsize : 1));
    std::vector<Npp8u> geomHost((size_t)(gsize > 0 ? gsize : 1), 0);
    npp_functional_test::DeviceMemory<Npp32u> foundDev(maxLabel + 2);
    std::vector<Npp32u> foundHost(maxLabel + 2, 0), geomOffsHost(maxLabel + 2, 0);
    npp_functional_test::DeviceMemory<Npp32u> geomOffsDev(maxLabel + 2);
    unsigned bsize = 0;
    nppiCompressedMarkerLabelsUFGetContoursBlockSegmentListSize_C1R(countsHost.data(), total, maxLabel, 0,
                                                                     maxLabel, &bsize);
    npp_functional_test::DeviceMemory<Npp8u> segDev((size_t)(bsize > 0 ? bsize : 1));
    std::vector<Npp8u> segHost((size_t)(bsize > 0 ? bsize : 1), 0);
    std::vector<P> ip((size_t)count);
    npp_functional_test::DeviceMemory<P> ipDev((size_t)count);
    npp_functional_test::DeviceMemory<P> igDev((size_t)(total ? total : 1));
    std::vector<P> ig((size_t)(total ? total : 1));
    npp_diff::clear_cuda_error();
    if constexpr (Is64) {
      st = nppiContoursImageMarchingSquaresInterpolation_64f_C1R_Ctx(
          contourImg.get(), W, (NppiPoint64f *)ipDev.get(), W * (int)sizeof(P),
          (NppiContourPixelDirectionInfo *)dir.get(), W * (int)sizeof(NppiContourPixelDirectionInfo),
          (NppiContourPixelGeometryInfo *)geomDev.get(), (NppiContourPixelGeometryInfo *)geomHost.data(),
          (NppiPoint64f *)igDev.get(), foundHost.data(), geomOffsDev.get(), geomOffsHost.data(), total, maxLabel, 0,
          maxLabel, (NppiContourBlockSegment *)segDev.get(), (NppiContourBlockSegment *)segHost.data(),
          NppiSize{W, H}, ctx);
    } else {
      st = nppiContoursImageMarchingSquaresInterpolation_32f_C1R_Ctx(
          contourImg.get(), W, (NppiPoint32f *)ipDev.get(), W * (int)sizeof(P),
          (NppiContourPixelDirectionInfo *)dir.get(), W * (int)sizeof(NppiContourPixelDirectionInfo),
          (NppiContourPixelGeometryInfo *)geomDev.get(), (NppiContourPixelGeometryInfo *)geomHost.data(),
          (NppiPoint32f *)igDev.get(), foundHost.data(), geomOffsDev.get(), geomOffsHost.data(), total, maxLabel, 0,
          maxLabel, (NppiContourBlockSegment *)segDev.get(), (NppiContourBlockSegment *)segHost.data(),
          NppiSize{W, H}, ctx);
    }
    if (st != NPP_NO_ERROR) {
      ++cnt.skipped;
      if (st == NPP_NOT_IMPLEMENTED_ERROR) ++cnt.notimpl;
      continue;
    }
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess) << "[" << id << "] marching sync";
    ++cnt.executed;
    cudaMemcpy(ip.data(), ipDev.get(), ip.size() * sizeof(P), cudaMemcpyDeviceToHost);
    cudaMemcpy(ig.data(), igDev.get(), ig.size() * sizeof(P), cudaMemcpyDeviceToHost);
    const GenCase dc{W, H, variant, 0, 0, kSmall, 0};
    auto finiteIn = [&](const std::vector<P> &pts, double lo, double hi) {
      for (size_t i = 0; i < pts.size(); ++i) {
        ++cnt.compared;
        const double x = (double)pts[i].x, y = (double)pts[i].y;
        if (!std::isfinite(x) || !std::isfinite(y) || x < lo || x > hi || y < lo || y > hi) {
          char b[320];
          std::snprintf(b, sizeof(b), "[%s] %s point[%zu]=(%g,%g) outside [%g,%g]", id, dc.id().c_str(), i, x, y,
                        lo, hi);
          if (first.empty()) first = b;
          EXPECT_TRUE(false) << b;
          ++cnt.mismatched;
          return false;
        }
      }
      return true;
    };
    finiteIn(ip, -1.0, (double)(W > H ? W : H) + 1.0);
    finiteIn(ig, -1.0, (double)(W > H ? W : H) + 1.0);
  }
  finish(id, cnt, first);
}

inline void run_marching32(const char *id) { run_marching_pair<NppiPoint32f, false>(id); }
inline void run_marching64(const char *id) { run_marching_pair<NppiPoint64f, true>(id); }

}  // namespace contours
}  // namespace num

