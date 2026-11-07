// CPU-reference output coverage for the batch color-conversion family
// (nppi_color_conversion.h):
//   nppiYCbCrTo{BGR,RGB}Batch_8u_{C3R,P3C3R}
//   nppiYUVTo{BGR,RGB}Batch_8u_{C3R,P3C3R}
//   nppi{YCbCr,YUV}{420,422}To{BGR,RGB}Batch_8u_P3C3R
// plus the _Advanced variants.
//
// The conversions are the measured NPP 13.4 inverse transforms evaluated in
// single precision with fused multiply-adds and truncation after saturation:
//   YUV   (BT.601 full range): R = Y + 1.13983 V,
//         G = Y - 0.39465 U - 0.58060 V, B = Y + 2.03211 U;
//   YCbCr (BT.601 video range): Y' = 1.164 (Y - 16), R = Y' + 1.596 Cr,
//         G = Y' - 0.392 Cb - 0.813 Cr, B = Y' + 2.017 Cb.
// 4:2:2 sources sample chroma at (x/2, y), 4:2:0 at (x/2, y/2).
//
// Batch semantics: every descriptor carries its own data pointer and row
// step; the plain variants apply the single oSizeROI to every pair, the
// _Advanced variants use pDstBatchList[i].oSize as that pair's ROI (the
// oMaxSizeROI argument is not consulted by the reference).  An odd ROI is
// converted up to the largest even rectangle (4:2:2 shrinks the width, 4:2:0
// the width and height) and reported as NPP_DOUBLE_SIZE_WARNING.
#pragma once

#include "num_nppi_geom.h"

namespace num {
namespace colbatch {

enum K {
  kYCbCr444C3 = 0,
  kYCbCr444P3,
  kYCbCr422P3,
  kYCbCr420P3,
  kYUV444C3,
  kYUV444P3,
  kYUV422P3,
  kYUV420P3,
};

constexpr bool isYuvK(int k) { return k >= kYUV444C3; }
constexpr bool planarK(int k) {
  return k == kYCbCr444P3 || k == kYCbCr422P3 || k == kYCbCr420P3 || k == kYUV444P3 ||
         k == kYUV422P3 || k == kYUV420P3;
}
constexpr int subXK(int k) {
  return (k == kYCbCr422P3 || k == kYCbCr420P3 || k == kYUV422P3 || k == kYUV420P3) ? 2: 1;
}
constexpr int subYK(int k) { return (k == kYCbCr420P3 || k == kYUV420P3) ? 2: 1; }

inline Npp8u sat8(float v) { return (Npp8u)std::fminf(std::fmaxf(v, 0.0f), 255.0f); }

template <bool Yuv> inline void inv_px(int y, int cb, int cr, Npp8u &r, Npp8u &g, Npp8u &b) {
  if constexpr (Yuv) {
    const float Y = (float)y;
    const float U = (float)(cb - 128);
    const float V = (float)(cr - 128);
    r = sat8(warp_fma(1.13983f, V, Y));
    g = sat8(warp_fma(-0.58060f, V, warp_fma(-0.39465f, U, Y)));
    b = sat8(warp_fma(2.03211f, U, Y));
  } else {
    const float Y = warp_fmul(1.164f, (float)(y - 16));
    const float Cb = (float)(cb - 128);
    const float Cr = (float)(cr - 128);
    r = sat8(warp_fma(1.596f, Cr, Y));
    g = sat8(warp_fma(-0.813f, Cr, warp_fma(-0.392f, Cb, Y)));
    b = sat8(warp_fma(2.017f, Cb, Y));
  }
}

// pad rotation so every image/plane exercises a distinct row pitch.
inline int padFor(int i, int p) { return 16 * ((i + 2 * p) % 3); }

// Adv=false: plain batch (one ROI for all pairs); Adv=true: _Advanced
// (per-pair ROI from the destination descriptor).
template <int K, bool Bgr, bool Adv, class Call>
void run_batch(const char *id, Call call) {
  NppStreamContext ctx{};
  if (nppGetStreamContext(&ctx) != NPP_NO_ERROR) {
    GTEST_SKIP() << "[" << id << "] no stream context";
    return;
  }
  constexpr int SX = subXK(K), SY = subYK(K);
  constexpr bool Planar = planarK(K);
  constexpr bool Yuv = isYuvK(K);
  constexpr int B = 3;
  RunCounts cnt;
  std::string first;
  struct Cfg {
    int w, h;
  };
  static const Cfg cfgs[] = {{16, 12}, {64, 48}, {128, 96}, {17, 13}};
  for (const Cfg &cf: cfgs) {
    int ws[B], hs[B];
    if constexpr (Adv) {
      ws[0] = cf.w;
      hs[0] = cf.h;
      ws[1] = std::max(4, cf.w - 2);
      hs[1] = cf.h;
      ws[2] = std::max(4, cf.w - 3);
      hs[2] = std::max(4, cf.h - 2);
    } else {
      for (int i = 0; i < B; ++i) {
        ws[i] = cf.w;
        hs[i] = cf.h;
      }
    }
    const uint64_t seed = case_seed(id, cf.w, cf.h, Adv ? 1: 0, 0, 0, 0);
    std::vector<Buf<Npp8u> *> plane[3];
    std::vector<Buf<Npp8u> *> psrc(B), dsts(B);
    std::vector<std::vector<uint8_t>> pre(B);
    for (int i = 0; i < B; ++i) {
      const int w = ws[i], h = hs[i];
      if constexpr (Planar) {
        for (int p = 0; p < 3; ++p) {
          Buf<Npp8u> *s = new Buf<Npp8u>(w, h, 1, padFor(i, p), 0);
          fill_regime(*s, 1, seed ^ mix64((uint64_t)(i * 3 + p + 1)), kWide, p, false, false, false);
          s->dev.copyFromHost(s->host);
          plane[p].push_back(s);
        }
      } else {
        Buf<Npp8u> *s = new Buf<Npp8u>(w, h, 3, padFor(i, 0), 0);
        fill_regime(*s, 3, seed ^ mix64((uint64_t)i + 1), kWide, 0, false, false, false);
        s->dev.copyFromHost(s->host);
        psrc[(size_t)i] = s;
      }
      Buf<Npp8u> *d = new Buf<Npp8u>(w, h, 3, padFor(i, 1), 0x5A);
      d->dev.copyFromHost(d->host);
      dsts[(size_t)i] = d;
      pre[(size_t)i] = d->host;
    }

    // Every pair of this batch shares one ROI (plain) or uses its own size
    // (advanced); two plain variants per case exercise an even and (for
    // subsampled formats) an odd ROI.
    const int nVariant = Adv ? 1: 2;
    for (int v = 0; v < nVariant; ++v) {
      int ew[B], eh[B];  // effective converted extent of each pair
      int maxW = 0, maxH = 0;
      for (int i = 0; i < B; ++i) {
        const int rw = Adv ? ws[i]: (v == 0 ? cf.w: std::max(2, cf.w - 1));
        const int rh = Adv ? hs[i]: (v == 0 ? cf.h: std::max(2, cf.h - 1));
        ew[i] = rw;
        eh[i] = rh;
        if constexpr (SX == 2) ew[i] &= ~1;
        if constexpr (SY == 2) eh[i] &= ~1;
        maxW = std::max(maxW, ws[i]);
        maxH = std::max(maxH, hs[i]);
      }
      std::vector<NppiImageDescriptor> hs0(B), hs1(B), hs2(B), hd(B);
      for (int i = 0; i < B; ++i) {
        const int rw = Adv ? ws[i]: cf.w;
        const int rh = Adv ? hs[i]: cf.h;
        // Descriptor sizes describe the whole logical image on every plane;
        // the reference binds the processed ROI to them (measured: a
        // chroma-sized descriptor makes it convert only that many pixels).
        const int cw = rw, ch = rh;
        if constexpr (Planar) {
          hs0[(size_t)i] = {(void *)plane[0][(size_t)i]->dptr(), plane[0][(size_t)i]->step,
                            NppiSize{rw, rh}};
          hs1[(size_t)i] = {(void *)plane[1][(size_t)i]->dptr(), plane[1][(size_t)i]->step,
                            NppiSize{cw, ch}};
          hs2[(size_t)i] = {(void *)plane[2][(size_t)i]->dptr(), plane[2][(size_t)i]->step,
                            NppiSize{cw, ch}};
        } else {
          hs0[(size_t)i] = {(void *)psrc[(size_t)i]->dptr(), psrc[(size_t)i]->step, NppiSize{rw, rh}};
        }
        hd[(size_t)i] = {(void *)dsts[(size_t)i]->dptr(), dsts[(size_t)i]->step,
                         NppiSize{Adv ? ew[i]: rw, Adv ? eh[i]: rh}};
      }
      Buf<NppiImageDescriptor> ds0(B, 1, 1, 0, 0), ds1(B, 1, 1, 0, 0), ds2(B, 1, 1, 0, 0),
          dd(B, 1, 1, 0, 0);
      std::memcpy(ds0.host.data(), hs0.data(), sizeof(NppiImageDescriptor) * B);
      std::memcpy(ds1.host.data(), hs1.data(), sizeof(NppiImageDescriptor) * B);
      std::memcpy(ds2.host.data(), hs2.data(), sizeof(NppiImageDescriptor) * B);
      std::memcpy(dd.host.data(), hd.data(), sizeof(NppiImageDescriptor) * B);
      ds0.dev.copyFromHost(ds0.host);
      ds1.dev.copyFromHost(ds1.host);
      ds2.dev.copyFromHost(ds2.host);
      dd.dev.copyFromHost(dd.host);
      const NppiImageDescriptor *sp[3] = {
          (const NppiImageDescriptor *)ds0.dptr(), (const NppiImageDescriptor *)ds1.dptr(),
          (const NppiImageDescriptor *)ds2.dptr()};
      NppiImageDescriptor *pd = (NppiImageDescriptor *)dd.dptr();
      const NppiSize roi = Adv ? NppiSize{maxW, maxH}: NppiSize{v == 0 ? cf.w: std::max(2, cf.w - 1),
                                          v == 0 ? cf.h: std::max(2, cf.h - 1)};
      npp_diff::clear_cuda_error();
      const NppStatus st = call(sp, pd, B, roi, ctx);
      if (st != NPP_NO_ERROR && st != NPP_DOUBLE_SIZE_WARNING) {
        ++cnt.skipped;
        if (st == NPP_NOT_IMPLEMENTED_ERROR) ++cnt.notimpl;
        continue;
      }
      const cudaError_t se = cudaDeviceSynchronize();
      ASSERT_EQ(se, cudaSuccess) << "[" << id << "] " << cf.w << "x" << cf.h << " sync: "
                                 << cudaGetErrorString(se);
      ++cnt.executed;

      bool failed = false;
      for (int i = 0; i < B && !failed; ++i) {
        auto srcPx = [&](int p, int x, int y) -> int {
          const Buf<Npp8u> *b = plane[p][(size_t)i];
          return (int)b->hptr()[(size_t)y * b->step + (size_t)x];
        };
        std::vector<uint8_t> want = pre[(size_t)i];
        for (int y = 0; y < eh[i]; ++y)
          for (int x = 0; x < ew[i]; ++x) {
            int yv, cbv, crv;
            if constexpr (Planar) {
              yv = srcPx(0, x, y);
              cbv = srcPx(1, x / SX, y / SY);
              crv = srcPx(2, x / SX, y / SY);
            } else {
              const Buf<Npp8u> *b = psrc[(size_t)i];
              const Npp8u *px = b->hptr() + (size_t)y * b->step + (size_t)x * 3;
              yv = px[0];
              cbv = px[1];
              crv = px[2];
            }
            Npp8u r, g, b;
            inv_px<Yuv>(yv, cbv, crv, r, g, b);
            const size_t base = (size_t)y * dsts[(size_t)i]->step + (size_t)x * 3;
            want[base + (Bgr ? 2: 0)] = r;
            want[base + 1] = g;
            want[base + (Bgr ? 0: 2)] = b;
          }
        Buf<Npp8u> *d = dsts[(size_t)i];
        d->dev.copyToHost(d->host);
        const GenCase dc{ew[i], eh[i], 0, 0, 0, kSmall, 0};
        if (!rotate_check<Npp8u, 3, 3>(id, dc, NPPI_INTER_NN, pre[(size_t)i], d->host, want,
                                       d->step, {}, cnt, first))
          failed = true;
        if (cnt.mismatched >= 8) break;
      }
      if (failed) break;
    }
    for (int p = 0; p < 3; ++p)
      for (auto *b: plane[p]) delete b;
    for (auto *b: psrc)
      if (b) delete b;
    for (auto *b: dsts) delete b;
    if (cnt.mismatched >= 8) break;
  }
  finish(id, cnt, first);
}

}  // namespace colbatch
}  // namespace num
