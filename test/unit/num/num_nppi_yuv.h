// CPU-reference output coverage for the pointwise 8u YUV / YCbCr conversion families:
// nppiRGBToYUV / nppiBGRToYUV / nppiYUVToRGB / nppiYUVToBGR /
// nppiRGBToYCbCr / nppiBGRToYCbCr / nppiYCbCrToRGB / nppiYCbCrToBGR.
//
// Formulas follow the measured NPP 13.4 pipelines:
//   YUV:  Y =.299 R +.587 G +.114 B
//         U = -.147 R -.289 G +.436 B + 128
//         V =.615 R -.515 G -.100 B + 128          (float, truncate)
//   YCbCr (studio range):
//         Y =.257 R +.504 G +.098 B + 16
//         Cb = -.148 R -.291 G +.439 B + 128
//         Cr =.439 R -.368 G -.071 B + 128         (double, truncate)
//   YUV->RGB: R = Y + 1.140 V', G = Y -.395 U' -.581 V', B = Y + 2.032 U'
//   YCbCr->RGB: R = 1.164(Y-16) + 1.596 Cr',
//               G = 1.164(Y-16) -.392 Cb' -.813 Cr', B = 1.164(Y-16)+2.017 Cb'
//               (single-precision with explicit fma ordering)
// Alpha rules (measured on the reference): packed AC4 destinations keep their
// previous alpha, AC4->planar copies the source alpha into the fourth plane,
// and P3C4R writes the caller's constant alpha.
#pragma once

#include "num_nppi_geom.h"

namespace num {
namespace yuv {

enum class Conv { RGB2YUV, YUV2RGB, RGB2YCbCr, YCbCr2RGB };
enum class SL { C3, AC4, P3 };
enum class DL { C3, AC4, P3, P4, C4A };

inline Npp8u q8(double v) {
  if (v < 0.0) return 0;
  if (v > 255.0) return 255;
  return (Npp8u)v;
}

inline void pixel_conv(Conv k, bool bgr, Npp8u a0, Npp8u a1, Npp8u a2, Npp8u &o0, Npp8u &o1, Npp8u &o2) {
  if (k == Conv::RGB2YUV) {
    const float r = bgr ? a2: a0, g = a1, b = bgr ? a0: a2;
    o0 = q8(0.299f * r + 0.587f * g + 0.114f * b);
    o1 = q8(-0.147f * r - 0.289f * g + 0.436f * b + 128.0f);
    o2 = q8(0.615f * r - 0.515f * g - 0.100f * b + 128.0f);
  } else if (k == Conv::RGB2YCbCr) {
    const double r = bgr ? a2: a0, g = a1, b = bgr ? a0: a2;
    o0 = q8(0.257 * r + 0.504 * g + 0.098 * b + 16.0);
    o1 = q8(-0.148 * r - 0.291 * g + 0.439 * b + 128.0);
    o2 = q8(0.439 * r - 0.368 * g - 0.071 * b + 128.0);
  } else if (k == Conv::YUV2RGB) {
    const float y = a0, u = (float)a1 - 128.0f, v = (float)a2 - 128.0f;
    const float r = y + 1.140f * v;
    const float g = y - 0.395f * u - 0.581f * v;
    const float b = y + 2.032f * u;
    o0 = q8(bgr ? b: r);
    o1 = q8(g);
    o2 = q8(bgr ? r: b);
  } else {
    const float yv = 1.164f * ((float)a0 - 16.0f);
    const float cb = (float)a1 - 128.0f, cr = (float)a2 - 128.0f;
    const float r = std::fma(1.596f, cr, yv);
    const float g = std::fma(-0.813f, cr, std::fma(-0.392f, cb, yv));
    const float b = std::fma(2.017f, cb, yv);
    o0 = q8(bgr ? b: r);
    o1 = q8(g);
    o2 = q8(bgr ? r: b);
  }
}

template <Conv K, SL SrcL, DL DstL, bool Bgr, class Call>
void run_yuv(const char *id, Call call) {
  NppStreamContext ctx{};
  if (nppGetStreamContext(&ctx) != NPP_NO_ERROR) {
    GTEST_SKIP() << "[" << id << "] no stream context";
    return;
  }
  RunCounts cnt;
  std::string first;
  constexpr int Sc = SrcL == SL::C3 ? 3: 4;
  constexpr bool SrcPlanar = SrcL == SL::P3;
  constexpr bool DstPlanar = DstL == DL::P3 || DstL == DL::P4;
  constexpr int Dc = DstPlanar ? 1: (DstL == DL::C3 ? 3: 4);
  RunCounts dummy;
  (void)dummy;
  for (const GenCase &c: make_gen_cases(0)) {
    if (c.w < 2 || c.h < 2) continue;
    std::vector<Buf<Npp8u> *> srcs, dsts;
    std::vector<std::vector<uint8_t>> pre, orig;
    const uint64_t seed = case_seed(id, c.w, c.h, c.pad, 0, (int)c.regime, 0);
    if constexpr (SrcPlanar) {
      planar_fill_src(srcs, 3, c.w, c.h, c.ox, c.oy, c.pad, seed, c.regime);
    } else {
      Buf<Npp8u> *a = new Buf<Npp8u>(c.w + c.ox, c.h + c.oy, Sc, c.pad, 0);
      fill_regime(*a, Sc, seed, c.regime, 0, false, false, false);
      a->dev.copyFromHost(a->host);
      srcs.push_back(a);
    }
    const int dn = DstPlanar ? (DstL == DL::P4 ? 4: 3): 1;
    for (int p = 0; p < dn; ++p) {
      Buf<Npp8u> *d = new Buf<Npp8u>(c.w, c.h, DstPlanar ? 1: Dc, 0, 0x5A);
      d->dev.copyFromHost(d->host);
      dsts.push_back(d);
      pre.push_back(d->host);
    }
    const Npp8u *hsrcp[4] = {nullptr, nullptr, nullptr, nullptr};
    const Npp8u *dsrcp[4] = {nullptr, nullptr, nullptr, nullptr};
    Npp8u *ddstp[4] = {nullptr, nullptr, nullptr, nullptr};
    const Npp8u *hpacked = nullptr;
    const Npp8u *dpacked = nullptr;
    Npp8u *dpackedOut = nullptr;
    if constexpr (SrcPlanar) {
      for (int p = 0; p < 3; ++p) {
        hsrcp[p] = srcs[(size_t)p]->hptr() + off_bytes<Npp8u>(c.oy, c.ox, srcs[(size_t)p]->step, 1);
        dsrcp[p] = srcs[(size_t)p]->dptr() + off_bytes<Npp8u>(c.oy, c.ox, srcs[(size_t)p]->step, 1);
      }
    } else {
      hpacked = srcs[0]->hptr() + off_bytes<Npp8u>(c.oy, c.ox, srcs[0]->step, Sc);
      dpacked = srcs[0]->dptr() + off_bytes<Npp8u>(c.oy, c.ox, srcs[0]->step, Sc);
    }
    if constexpr (DstPlanar) {
      for (int p = 0; p < dn; ++p) ddstp[p] = dsts[(size_t)p]->dptr();
    } else {
      dpackedOut = dsts[0]->dptr();
    }
    const int sstep = srcs[0]->step;
    const int dstep = dsts[0]->step;
    npp_diff::clear_cuda_error();
    const NppStatus st = call(dsrcp, dpacked, sstep, ddstp, dpackedOut, dstep, NppiSize{c.w, c.h}, 200, ctx);
    if (st != NPP_NO_ERROR) {
      ++cnt.skipped;
      if (st == NPP_NOT_IMPLEMENTED_ERROR) ++cnt.notimpl;
      for (auto *p: srcs) delete p;
      for (auto *p: dsts) delete p;
      continue;
    }
    const cudaError_t se = cudaDeviceSynchronize();
    ASSERT_EQ(se, cudaSuccess) << "[" << id << "] " << c.id() << " sync: " << cudaGetErrorString(se);
    ++cnt.executed;
    bool failed = false;
    if constexpr (DstPlanar) {
      for (int p = 0; p < dn && !failed; ++p) {
        dsts[(size_t)p]->dev.copyToHost(dsts[(size_t)p]->host);
        std::vector<uint8_t> want = pre[(size_t)p];
        for (int y = 0; y < c.h; ++y)
          for (int x = 0; x < c.w; ++x) {
            Npp8u a0, a1, a2, o0, o1, o2;
            if constexpr (SrcPlanar) {
              const size_t so = (size_t)y * sstep + (size_t)x;
              a0 = hsrcp[0][so];
              a1 = hsrcp[1][so];
              a2 = hsrcp[2][so];
            } else {
              const size_t so = (size_t)y * sstep + (size_t)x * Sc;
              a0 = hpacked[so];
              a1 = hpacked[so + 1];
              a2 = hpacked[so + 2];
            }
            pixel_conv(K, Bgr, a0, a1, a2, o0, o1, o2);
            const size_t db = (size_t)y * dstep + (size_t)x;
            if (p == 0) want[db] = o0;
            else if (p == 1) want[db] = o1;
            else if (p == 2) want[db] = o2;
            else if constexpr (SrcL == SL::AC4) want[db] = hpacked[(size_t)y * sstep + (size_t)x * 4 + 3];
          }
        const GenCase dc{c.w, c.h, 0, 0, 0, kSmall, 0};
        if (!rotate_check<Npp8u, 1, 1>(id, dc, NPPI_INTER_NN, pre[(size_t)p], dsts[(size_t)p]->host, want,
                                       dsts[(size_t)p]->step, std::vector<uint8_t>(), cnt, first))
          failed = true;
      }
    } else {
      dsts[0]->dev.copyToHost(dsts[0]->host);
      std::vector<uint8_t> want = pre[0];
      for (int y = 0; y < c.h; ++y)
        for (int x = 0; x < c.w; ++x) {
          Npp8u a0, a1, a2, o0, o1, o2;
          if constexpr (SrcPlanar) {
            const size_t so = (size_t)y * sstep + (size_t)x;
            a0 = hsrcp[0][so];
            a1 = hsrcp[1][so];
            a2 = hsrcp[2][so];
          } else {
            const size_t so = (size_t)y * sstep + (size_t)x * Sc;
            a0 = hpacked[so];
            a1 = hpacked[so + 1];
            a2 = hpacked[so + 2];
          }
          pixel_conv(K, Bgr, a0, a1, a2, o0, o1, o2);
          const size_t db = off_bytes<Npp8u>(y, x, dstep, Dc);
          want[db] = o0;
          want[db + 1] = o1;
          want[db + 2] = o2;
          if constexpr (DstL == DL::C4A) want[db + 3] = 200;
        }
      const GenCase dc{c.w, c.h, 0, 0, 0, kSmall, 0};
      const int at = DstL == DL::AC4 ? 3: Dc;
      if (!rotate_check<Npp8u, Dc, at>(id, dc, NPPI_INTER_NN, pre[0], dsts[0]->host, want, dstep,
                                       std::vector<uint8_t>(), cnt, first))
        failed = true;
    }
    for (auto *p: srcs) delete p;
    for (auto *p: dsts) delete p;
    if (failed || cnt.mismatched >= 8) break;
  }
  finish(id, cnt, first);
}

}  // namespace yuv
}  // namespace num
