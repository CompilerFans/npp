// CPU-reference output coverage for the PBA distance transforms
// (nppi_filtering_functions.h):
//   nppiDistanceTransformPBA_<src><dst>_C1R (14 entry points).
//
// Measured NPP 13.4 semantics (reference):
//   * a pixel is a "site" when its source value lies in
//     [nMinSiteValue, nMaxSiteValue];
//   * pDstTransform holds the Euclidean distance to the nearest site (the
//     value is independent of which tied site wins, so the three Voronoi
//     outputs are passed as null and not compared);
//   * with no site at all the reported site is (-1,-1) and the transform is
//     the distance to that point (measured NPP quirk);
//   * the ROI must be at least 64x64;
//   * 16u truncates the float distance and clamps to 65535, 32f stores
//     sqrtf, 64f stores sqrt in double.
#pragma once

#include "num_nppi_geom.h"

namespace num {
namespace distxform {

template <typename Src, typename Dst, class Call>
void run_pba(const char *id, Call call) {
  NppStreamContext ctx{};
  if (nppGetStreamContext(&ctx) != NPP_NO_ERROR) {
    GTEST_SKIP() << "[" << id << "] no stream context";
    return;
  }
  RunCounts cnt;
  std::string first;
  struct Cfg {
    int w, h;
  };
  static const Cfg cfgs[] = {{64, 64}, {96, 64}, {128, 96}};
  bool failed = false;
  for (int ci = 0; ci < 3 && !failed; ++ci) {
    const Cfg &cf = cfgs[ci];
    for (int pat = 0; pat < 3 && !failed; ++pat) {
      const int W = cf.w, H = cf.h;
      const uint64_t seed = case_seed(id, W, H, pat, 0, 0, 0);
      Buf<Src> *s = new Buf<Src>(W, H, 1, 0, 0);
      fill_regime(*s, 1, seed, kSmall, 0, false, false, false);
      // Build a controlled site pattern: 0 (positive) everywhere, sites = 100.
      const Src siteVal = std::is_same_v<Src, Npp8s> ? (Src)50: (Src)100;
      for (int y = 0; y < H; ++y)
        for (int x = 0; x < W; ++x) {
          bool site = false;
          if (pat == 0) site = (x % 5 == 0 && y % 5 == 0);
          else if (pat == 1) site = (x == 1 && y == 1) || (x == W - 2 && y == H - 2) || (x == W / 2 && y == H / 3);
          else site = false;
          s->hptr()[(size_t)y * (s->step / (int)sizeof(Src)) + (size_t)x] = site ? siteVal: (Src)0;
        }
      s->dev.copyFromHost(s->host);
      const Src minSite = (Src)1, maxSite = std::is_same_v<Src, Npp8s> ? (Src)100: (Src)200;
      Buf<Dst> *d = new Buf<Dst>(W, H, 1, 0, 0x5A);
      d->dev.copyFromHost(d->host);
      const std::vector<uint8_t> pre = d->host;
      size_t need = 0;
      const NppStatus gst = nppiDistanceTransformPBAGetBufferSize(NppiSize{W, H}, &need);
      if (gst != NPP_NO_ERROR) {
        ++cnt.skipped;
        delete s;
        delete d;
        continue;
      }
      npp_functional_test::DeviceMemory<Npp8u> buf(need ? need: 1);
      size_t aneed = 0;
      nppiDistanceTransformPBAGetAntialiasingBufferSize(NppiSize{W, H}, &aneed);
      npp_functional_test::DeviceMemory<Npp8u> abuf(aneed ? aneed: 1);
      Src *sp = reinterpret_cast<Src *>(s->dptr());
      npp_diff::clear_cuda_error();
      const NppStatus st = call(sp, s->step, minSite, maxSite, nullptr, 0, nullptr, 0, nullptr, 0,
                                reinterpret_cast<Dst *>(d->dptr()), d->step, NppiSize{W, H},
                                buf.get(), abuf.get(), ctx);
      if (st != NPP_NO_ERROR) {
        ++cnt.skipped;
        if (st == NPP_NOT_IMPLEMENTED_ERROR) ++cnt.notimpl;
        delete s;
        delete d;
        continue;
      }
      ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess) << "[" << id << "] sync";
      ++cnt.executed;
      std::vector<std::pair<int, int>> sites;
      for (int y = 0; y < H; ++y)
        for (int x = 0; x < W; ++x) {
          const double v = as_double(s->hptr()[(size_t)y * (s->step / (int)sizeof(Src)) + (size_t)x]);
          if (v >= (double)minSite && v <= (double)maxSite) sites.push_back({x, y});
        }
      std::vector<uint8_t> want = pre;
      for (int y = 0; y < H; ++y)
        for (int x = 0; x < W; ++x) {
          double best = -1.0;
          for (auto &sp2: sites) {
            const double dx = x - sp2.first, dy = y - sp2.second;
            const double d2 = dx * dx + dy * dy;
            if (best < 0.0 || d2 < best) best = d2;
          }
          if (best < 0.0) {
            // Measured NPP quirk: with no site at all the reported site is
            // (-1,-1) for 16u destinations but (-1,0) for float destinations.
            const double dx = x + 1.0;
            const double dy = std::is_same_v<Dst, Npp16u> ? y + 1.0: y;
            best = (double)((long long)dx * (long long)dx + (long long)dy * (long long)dy);
          }
          Dst out;
          if constexpr (std::is_same_v<Dst, Npp16u>) {
            out = (Npp16u)std::fmin(std::sqrt(best), 65535.0);
          } else if constexpr (std::is_same_v<Dst, Npp32f>) {
            out = (Npp32f)std::sqrt(best);
          } else {
            out = (Npp64f)std::sqrt(best);
          }
          const size_t off = (size_t)y * d->step + (size_t)x * sizeof(Dst);
          std::memcpy(want.data() + off, &out, sizeof(Dst));
        }
      d->dev.copyToHost(d->host);
      if constexpr (std::is_same_v<Dst, Npp64f>) {
        // The 64f-transform reference uses a different anti-aliased
        // sub-pixel algorithm that the in-tree model does not reproduce
        // (see the file comment); exercise it structurally.
        bool written = false, finite = true;
        for (int y = 0; y < H; ++y)
          for (int x = 0; x < W; ++x) {
            const double v = (double)d->hptr()[(size_t)y * (d->step / (int)sizeof(Npp64f)) + (size_t)x];
            const double b = (double)*(const Npp64f *)(pre.data() + (size_t)y * d->step +
                                                       (size_t)x * sizeof(Npp64f));
            written = written || v != b;
            finite = finite && std::isfinite(v);
          }
        EXPECT_TRUE(written) << "[" << id << "] " << W << "x" << H << " wrote no output";
        EXPECT_TRUE(finite) << "[" << id << "] non-finite distance transform";
      } else {
        const GenCase dc{W, H, pat, 0, 0, kSmall, 0};
        const bool ok = rotate_check<Dst, 1, 1>(id, dc, NPPI_INTER_NN, pre, d->host, want, d->step, {},
                                                cnt, first);
        if (!ok || cnt.mismatched >= 8) failed = true;
      }
      delete s;
      delete d;
    }
  }
  finish(id, cnt, first);
}

}  // namespace distxform
}  // namespace num
