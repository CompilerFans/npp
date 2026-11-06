// CPU-golden coverage for the JPEG CMYK/YCCK conversions:
//   nppiCMYKOrYCCKToRGB/BGR_JPEG_8u_{P4C3R,P4P3R} and
//   nppiYCCKToCMYK_JPEG_601_8u_P4R.
//
// Measured NPP 13.4 semantics:
//   * CMYK -> RGB/BGR: each of the first three planes is multiplied by the
//     K plane as trunc((255-c)*(255-k)/255) (the reference has an integer
//     path quirk that returns one less for a few hundred exact-multiple
//     operand pairs; the conformance tolerance absorbs the one-LSB cases);
//   * YCCK -> CMYK: R = Y + 1.402*dCr, G = Y - 0.34414*dCb - 0.71414*dCr,
//     B = Y + 1.772*dCb, then C = 255-R, M = 255-G, Y' = 255-B (all
//     truncated and saturated), K is copied.
#pragma once

#include "num_nppi_geom.h"

namespace num {
namespace cmyk {

inline Npp8u sat(double v) {
  const int t = (int)v;
  return (Npp8u)(t < 0 ? 0 : (t > 255 ? 255 : t));
}

enum class Out { C3, C4, P3 };

// srcP4: four planar source planes with a single step.  bgr applies to the
// packed RGB/BGR destinations.
template <Out O, bool Bgr, class Call>
void run_cmyk_to_rgb(const char *id, Call call) {
  NppStreamContext ctx{};
  if (nppGetStreamContext(&ctx) != NPP_NO_ERROR) {
    GTEST_SKIP() << "[" << id << "] no stream context";
    return;
  }
  RunCounts cnt;
  std::string first;
  struct Cfg { int w, h; };
  static const Cfg cfgs[] = {{4, 4}, {8, 6}, {16, 12}, {32, 24}, {64, 48}, {5, 4}, {17, 13}};
  for (const Cfg &cf : cfgs) {
    const int w = cf.w, h = cf.h;
    const uint64_t seed = case_seed(id, w, h, 0, 0, 0, 0);
    std::vector<Buf<Npp8u> *> srcs;
    planar_fill_src(srcs, 4, w, h, 0, 0, 0, seed, kWide);
    std::vector<Buf<Npp8u> *> dsts;
    std::vector<std::vector<uint8_t>> pre;
    if (O == Out::P3) {
      for (int p = 0; p < 3; ++p) {
        Buf<Npp8u> *d = new Buf<Npp8u>(w, h, 1, 0, 0x5A);
        d->dev.copyFromHost(d->host);
        dsts.push_back(d);
        pre.push_back(d->host);
      }
    } else {
      Buf<Npp8u> *d = new Buf<Npp8u>(w, h, 3, 0, 0x5A);
      d->dev.copyFromHost(d->host);
      dsts.push_back(d);
      pre.push_back(d->host);
    }
    const Npp8u *sp[4] = {srcs[0]->dptr(), srcs[1]->dptr(), srcs[2]->dptr(), srcs[3]->dptr()};
    Npp8u *dp[4] = {nullptr, nullptr, nullptr, nullptr};
    int dsa[4] = {0, 0, 0, 0};
    Npp8u *pd = nullptr;
    if (O == Out::P3) {
      for (int p = 0; p < 3; ++p) {
        dp[p] = dsts[(size_t)p]->dptr();
        dsa[p] = dsts[(size_t)p]->step;
      }
    } else {
      pd = dsts[0]->dptr();
    }
    const int ss = srcs[0]->step;
    const int ds = dsts[0]->step;
    npp_diff::clear_cuda_error();
    const NppStatus st = call(sp, ss, dp, pd, ds, dsa, NppiSize{w, h}, ctx);
    if (st != NPP_NO_ERROR) {
      ++cnt.skipped;
      if (st == NPP_NOT_IMPLEMENTED_ERROR) ++cnt.notimpl;
      for (auto *p : srcs) delete p;
      for (auto *p : dsts) delete p;
      continue;
    }
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess) << "[" << id << "] sync";
    ++cnt.executed;
    std::vector<std::vector<uint8_t>> want = pre;
    for (int y = 0; y < h; ++y)
      for (int x = 0; x < w; ++x) {
        int v[3];
        for (int c = 0; c < 3; ++c) {
          const int cv = srcs[(size_t)c]->hptr()[(size_t)y * srcs[(size_t)c]->step + (size_t)x];
          const int kv = srcs[3]->hptr()[(size_t)y * srcs[3]->step + (size_t)x];
          v[c] = (255 - cv) * (255 - kv) / 255;
        }
        if (O == Out::P3) {
          for (int c = 0; c < 3; ++c) {
            const int p = Bgr ? 2 - c : c;
            want[(size_t)p][(size_t)y * dsts[(size_t)p]->step + (size_t)x] = (uint8_t)v[c];
          }
        } else {
          const size_t base = (size_t)y * dsts[0]->step + (size_t)x * 3;
          want[0][base + (Bgr ? 2 : 0)] = (uint8_t)v[0];
          want[0][base + 1] = (uint8_t)v[1];
          want[0][base + (Bgr ? 0 : 2)] = (uint8_t)v[2];
        }
      }
    bool failed = false;
    for (size_t p = 0; p < dsts.size() && !failed; ++p) {
      dsts[p]->dev.copyToHost(dsts[p]->host);
      const GenCase dc{w, h, 0, 0, 0, kSmall, 0};
      const bool ok = (O == Out::P3)
                          ? rotate_check<Npp8u, 1, 1>(id, dc, NPPI_INTER_NN, pre[p], dsts[p]->host, want[p],
                                                      dsts[p]->step, {}, cnt, first)
                          : rotate_check<Npp8u, 3, 3>(id, dc, NPPI_INTER_NN, pre[p], dsts[p]->host, want[p],
                                                      dsts[p]->step, {}, cnt, first);
      if (!ok) failed = true;
    }
    for (auto *p : srcs) delete p;
    for (auto *p : dsts) delete p;
    if (failed || cnt.mismatched >= 8) break;
  }
  finish(id, cnt, first);
}

// YCCK -> CMYK, four planar planes with a single step on each side.
template <class Call>
void run_ycck_to_cmyk(const char *id, Call call) {
  NppStreamContext ctx{};
  if (nppGetStreamContext(&ctx) != NPP_NO_ERROR) {
    GTEST_SKIP() << "[" << id << "] no stream context";
    return;
  }
  RunCounts cnt;
  std::string first;
  struct Cfg { int w, h; };
  static const Cfg cfgs[] = {{4, 4}, {8, 6}, {16, 12}, {32, 24}, {64, 48}, {5, 4}, {17, 13}};
  for (const Cfg &cf : cfgs) {
    const int w = cf.w, h = cf.h;
    const uint64_t seed = case_seed(id, w, h, 0, 0, 0, 0);
    std::vector<Buf<Npp8u> *> srcs;
    planar_fill_src(srcs, 4, w, h, 0, 0, 0, seed, kWide);
    std::vector<Buf<Npp8u> *> dsts;
    std::vector<std::vector<uint8_t>> pre;
    for (int p = 0; p < 4; ++p) {
      Buf<Npp8u> *d = new Buf<Npp8u>(w, h, 1, 0, 0x5A);
      d->dev.copyFromHost(d->host);
      dsts.push_back(d);
      pre.push_back(d->host);
    }
    const Npp8u *sp[4] = {srcs[0]->dptr(), srcs[1]->dptr(), srcs[2]->dptr(), srcs[3]->dptr()};
    Npp8u *dp[4] = {dsts[0]->dptr(), dsts[1]->dptr(), dsts[2]->dptr(), dsts[3]->dptr()};
    const int ss = srcs[0]->step;
    const int ds = dsts[0]->step;
    npp_diff::clear_cuda_error();
    const NppStatus st = call(sp, ss, dp, ds, NppiSize{w, h}, ctx);
    if (st != NPP_NO_ERROR) {
      ++cnt.skipped;
      if (st == NPP_NOT_IMPLEMENTED_ERROR) ++cnt.notimpl;
      for (auto *p : srcs) delete p;
      for (auto *p : dsts) delete p;
      continue;
    }
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess) << "[" << id << "] sync";
    ++cnt.executed;
    std::vector<std::vector<uint8_t>> want = pre;
    for (int y = 0; y < h; ++y)
      for (int x = 0; x < w; ++x) {
        const int fy = srcs[0]->hptr()[(size_t)y * srcs[0]->step + (size_t)x];
        const int fcb = srcs[1]->hptr()[(size_t)y * srcs[1]->step + (size_t)x];
        const int fcr = srcs[2]->hptr()[(size_t)y * srcs[2]->step + (size_t)x];
        const double R = fy + 1.402 * (fcr - 128.0);
        const double G = fy - 0.34414 * (fcb - 128.0) - 0.71414 * (fcr - 128.0);
        const double B = fy + 1.772 * (fcb - 128.0);
        const Npp8u c[4] = {sat(255.0 - R), sat(255.0 - G), sat(255.0 - B),
                            srcs[3]->hptr()[(size_t)y * srcs[3]->step + (size_t)x]};
        for (int p = 0; p < 4; ++p)
          want[(size_t)p][(size_t)y * dsts[(size_t)p]->step + (size_t)x] = c[p];
      }
    bool failed = false;
    for (size_t p = 0; p < dsts.size() && !failed; ++p) {
      dsts[p]->dev.copyToHost(dsts[p]->host);
      const GenCase dc{w, h, 0, 0, 0, kSmall, 0};
      if (!rotate_check<Npp8u, 1, 1>(id, dc, NPPI_INTER_NN, pre[p], dsts[p]->host, want[p], dsts[p]->step, {}, cnt,
                                     first))
        failed = true;
    }
    for (auto *p : srcs) delete p;
    for (auto *p : dsts) delete p;
    if (failed || cnt.mismatched >= 8) break;
  }
  finish(id, cnt, first);
}

}  // namespace cmyk
}  // namespace num
