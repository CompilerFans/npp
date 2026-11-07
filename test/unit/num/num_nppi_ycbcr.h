// CPU-reference output coverage for the JPEG YCbCr conversion families
// (nppiRGBToYCbCr{444,420,422,411}_JPEG / BGR variants and the inverse
// nppiYCbCr*_JPEG -> RGB/BGR), packed C3 and planar P3 layouts.
//
// Measured NPP 13.4 semantics (single-precision pipeline, fitted on the reference):
//   forward: Y = fma(.114,B, fma(.299,R,.587*G)) and Cb/Cr stepwise
//            -.168736R-.331264G+.5B+128 /.5R-.418688G-.081312B+128 (every
//            product/add rounded to float), truncated to u8; verified
//            exhaustively over all 2^24 RGB triples;
//   inverse: R = fma(1.402,dCr,Y); G = fma(-.714136,dCr, fma(-.344136,dCb,Y));
//            B = fma(1.772,dCb,Y), truncating to u8 (R/B exact over 2^24
//            triples, G differs from the reference by one LSB on a few
//            knife-edge inputs, absorbed by the integer check tolerance);
//   420: forward writes only the (w&~1)x(h&~1) luma extent and chroma is the
//        sum of four truncated values >> 2; inverse converts the full width
//        but only h&~1 rows;
//   422: forward writes width&~1 columns on every row, chroma pair >> 1;
//        inverse full width, h&~1 rows;
//   411: width must be a multiple of 4; luma per pixel, chroma per 4-column
//        group sum/4 (truncating); inverse replicas chroma x/4, full extent;
//   420/422 planar-source forwards are measured NO-OPs; odd dimensions warn
//   (NPP_DOUBLE_SIZE_WARNING) but still convert the even part.
#pragma once

#include "num_nppi_geom.h"

namespace num {
namespace ycbcr {

enum class Sub { S444, S420, S422, S411 };

inline Npp8u q8(double v) {
  int t = (int)v;
  return (Npp8u)(t < 0 ? 0: (t > 255 ? 255: t));
}

inline void fwd_pixel(int r, int g, int b, int &y, int &cb, int &cr) {
  const float fr = (float)r, fg = (float)g, fb = (float)b;
  const float fy = warp_fma(0.114f, fb, warp_fma(0.299f, fr, warp_fmul(0.587f, fg)));
  const float fcb = warp_fadd(warp_fadd(warp_fsub(warp_fmul(-0.168736f, fr), warp_fmul(0.331264f, fg)),
                                       warp_fmul(0.5f, fb)), 128.0f);
  const float fcr = warp_fadd(warp_fsub(warp_fsub(warp_fmul(0.5f, fr), warp_fmul(0.418688f, fg)),
                                        warp_fmul(0.081312f, fb)), 128.0f);
  y = q8(fy);
  cb = q8(fcb);
  cr = q8(fcr);
}

inline void inv_pixel(int y, int cb, int cr, int &r, int &g, int &b) {
  const float fy = (float)y;
  const float dcb = warp_fsub((float)cb, 128.0f);
  const float dcr = warp_fsub((float)cr, 128.0f);
  r = q8(warp_fma(1.402f, dcr, fy));
  g = q8(warp_fma(-0.714136f, dcr, warp_fma(-0.344136f, dcb, fy)));
  b = q8(warp_fma(1.772f, dcb, fy));
}

// SrcPacked: packed C3 source (else three planes); DstPacked: packed C3
// destination (else three planes); Bgr: memory order B,G,R instead of R,G,B.
template <Sub S, bool Fwd, bool SrcPacked, bool DstPacked, bool Bgr, class Call>
void run_ycbcr_jpeg(const char *id, Call call) {
  NppStreamContext ctx{};
  if (nppGetStreamContext(&ctx) != NPP_NO_ERROR) {
    GTEST_SKIP() << "[" << id << "] no stream context";
    return;
  }
  RunCounts cnt;
  std::string first;
  struct Cfg { int w, h; };
  static const Cfg cfgs[] = {{4, 4},   {8, 6},   {16, 12}, {32, 24}, {64, 48},
                             {128, 96}, {17, 13}, {16, 13}, {17, 12}, {48, 32}};
  for (const Cfg &cf: cfgs) {
    if (S == Sub::S411 && (cf.w % 4) != 0) continue;
    std::vector<Buf<Npp8u> *> srcs;
    const uint64_t seed = case_seed(id, cf.w, cf.h, 0, 0, 0, 0);
    if constexpr (SrcPacked) {
      Buf<Npp8u> *a = new Buf<Npp8u>(cf.w, cf.h, 3, 0, 0);
      fill_regime(*a, 3, seed, kWide, 0, false, false, false);
      a->dev.copyFromHost(a->host);
      srcs.push_back(a);
    } else {
      planar_fill_src(srcs, 3, cf.w, cf.h, 0, 0, 0, seed, kWide);
    }
    // Only forward destinations are subsampled; inverse destinations are RGB.
    const int cw = (Fwd && S == Sub::S411) ? (cf.w / 4): cf.w;
    std::vector<Buf<Npp8u> *> dsts;
    std::vector<std::vector<uint8_t>> pre;
    if constexpr (DstPacked) {
      Buf<Npp8u> *d = new Buf<Npp8u>(cf.w, cf.h, 3, 0, 0x5A);
      d->dev.copyFromHost(d->host);
      dsts.push_back(d);
      pre.push_back(d->host);
    } else {
      for (int p = 0; p < 3; ++p) {
        const int dw = (p == 0) ? cf.w: cw;
        Buf<Npp8u> *d = new Buf<Npp8u>(dw, cf.h, 1, 0, 0x5A);
        d->dev.copyFromHost(d->host);
        dsts.push_back(d);
        pre.push_back(d->host);
      }
    }
    const Npp8u *sp[4] = {nullptr, nullptr, nullptr, nullptr};
    const Npp8u *ps = nullptr;
    Npp8u *dp[4] = {nullptr, nullptr, nullptr, nullptr};
    Npp8u *pd = nullptr;
    int ssa[4] = {0, 0, 0, 0};
    int dsa[4] = {0, 0, 0, 0};
    if constexpr (SrcPacked) {
      ps = srcs[0]->dptr();
    } else {
      for (int p = 0; p < 3; ++p) {
        sp[p] = srcs[(size_t)p]->dptr();
        ssa[p] = srcs[(size_t)p]->step;
      }
    }
    if constexpr (DstPacked) {
      pd = dsts[0]->dptr();
    } else {
      for (int p = 0; p < 3; ++p) {
        dp[p] = dsts[(size_t)p]->dptr();
        dsa[p] = dsts[(size_t)p]->step;
      }
    }
    const int ss = srcs[0]->step;
    const int ds = dsts[0]->step;
    npp_diff::clear_cuda_error();
    NppStatus st = call(sp, ps, ss, ssa, dp, pd, ds, dsa, NppiSize{cf.w, cf.h}, ctx);
    if (st != NPP_NO_ERROR && st != NPP_DOUBLE_SIZE_WARNING) {
      ++cnt.skipped;
      if (st == NPP_NOT_IMPLEMENTED_ERROR) ++cnt.notimpl;
      for (auto *p: srcs) delete p;
      for (auto *p: dsts) delete p;
      continue;
    }
    const cudaError_t se = cudaDeviceSynchronize();
    ASSERT_EQ(se, cudaSuccess) << "[" << id << "] " << cf.w << "x" << cf.h << " sync: " << cudaGetErrorString(se);
    ++cnt.executed;

    // Converted extents (measured).
    int ew = cf.w, eh = cf.h;
    if constexpr (Fwd) {
      if constexpr (S == Sub::S420) {
        ew = cf.w & ~1;
        eh = cf.h & ~1;
      } else if constexpr (S == Sub::S422) {
        ew = cf.w & ~1;
      }
    } else {
      if constexpr (S == Sub::S420 || S == Sub::S422) eh = cf.h & ~1;
    }
    const bool planarSrcNoop = (!SrcPacked) && Fwd && (S == Sub::S420 || S == Sub::S422);
    auto planeOf = [&](int c) -> int { return Bgr ? 2 - c: c; };
    // Forward sources are RGB/BGR images; inverse sources are always Y,Cb,Cr.
    auto srcAt = [&](int c, int x, int y) -> int {
      const int p = Fwd ? planeOf(c): c;
      if constexpr (SrcPacked) return srcs[0]->hptr()[(size_t)y * srcs[0]->step + (size_t)x * 3 + (size_t)p];
      else return srcs[(size_t)p]->hptr()[(size_t)y * srcs[(size_t)p]->step + (size_t)x];
    };
    auto chromaXY = [&](int x, int y, int &cx, int &cy) {
      if constexpr (S == Sub::S420) { cx = x >> 1; cy = y >> 1; }
      else if constexpr (S == Sub::S422) { cx = x >> 1; cy = y; }
      else if constexpr (S == Sub::S411) { cx = x >> 2; cy = y; }
      else { cx = x; cy = y; }
    };
    std::vector<std::vector<uint8_t>> want = pre;
    if (planarSrcNoop) {
      // Measured: planar-source 420/422 forward never writes.
    } else if constexpr (Fwd) {
      for (int y = 0; y < eh; ++y)
        for (int x = 0; x < ew; ++x) {
          int Yv, cb, cr;
          fwd_pixel(srcAt(0, x, y), srcAt(1, x, y), srcAt(2, x, y), Yv, cb, cr);
          if constexpr (!DstPacked) {
            want[0][(size_t)y * dsts[0]->step + (size_t)x] = (uint8_t)Yv;
          }
          if constexpr (!DstPacked && S == Sub::S444) {
            want[1][(size_t)y * dsts[1]->step + (size_t)x] = (uint8_t)cb;
            want[2][(size_t)y * dsts[2]->step + (size_t)x] = (uint8_t)cr;
          }
        }
      if constexpr (!DstPacked && (S == Sub::S420 || S == Sub::S422 || S == Sub::S411)) {
        const int cw2 = (S == Sub::S411) ? (cf.w / 4): (cf.w / 2);
        const int ch2 = (S == Sub::S420) ? (cf.h / 2): cf.h;
        // Measured: the JPEG 4:1:1 planar-source forward takes the first
        // sample of each 4-column group; the packed-source forward averages.
        const bool firstSampleOnly = (!SrcPacked) && (S == Sub::S411);
        for (int cy = 0; cy < ch2; ++cy)
          for (int cx = 0; cx < cw2; ++cx) {
            if (firstSampleOnly) {
              int Yv, cb, cr;
              fwd_pixel(srcAt(0, cx * 4, cy), srcAt(1, cx * 4, cy), srcAt(2, cx * 4, cy), Yv, cb, cr);
              want[1][(size_t)cy * dsts[1]->step + (size_t)cx] = (uint8_t)cb;
              want[2][(size_t)cy * dsts[2]->step + (size_t)cx] = (uint8_t)cr;
              continue;
            }
            int sumCb = 0, sumCr = 0, n = 0;
            const int spanX = (S == Sub::S411) ? 4: 2;
            const int spanY = (S == Sub::S420) ? 2: 1;
            for (int dy = 0; dy < spanY; ++dy)
              for (int dx = 0; dx < spanX; ++dx) {
                int Yv, cb, cr;
                fwd_pixel(srcAt(0, cx * spanX + dx, cy * spanY + dy), srcAt(1, cx * spanX + dx, cy * spanY + dy),
                          srcAt(2, cx * spanX + dx, cy * spanY + dy), Yv, cb, cr);
                sumCb += cb;
                sumCr += cr;
                ++n;
              }
            want[1][(size_t)cy * dsts[1]->step + (size_t)cx] = (uint8_t)(sumCb / n);
            want[2][(size_t)cy * dsts[2]->step + (size_t)cx] = (uint8_t)(sumCr / n);
          }
      }
    } else {
      for (int y = 0; y < eh; ++y)
        for (int x = 0; x < ew; ++x) {
          int cx, cy;
          chromaXY(x, y, cx, cy);
          const int Yv = srcAt(0, x, y);
          const int cb = srcAt(1, cx, cy);
          const int cr = srcAt(2, cx, cy);
          int r, g, b;
          inv_pixel(Yv, cb, cr, r, g, b);
          if constexpr (DstPacked) {
            const size_t db = (size_t)y * dsts[0]->step + (size_t)x * 3;
            want[0][db + (size_t)planeOf(0)] = (uint8_t)r;
            want[0][db + (size_t)planeOf(1)] = (uint8_t)g;
            want[0][db + (size_t)planeOf(2)] = (uint8_t)b;
          } else {
            want[planeOf(0)][(size_t)y * dsts[(size_t)planeOf(0)]->step + (size_t)x] = (uint8_t)r;
            want[planeOf(1)][(size_t)y * dsts[(size_t)planeOf(1)]->step + (size_t)x] = (uint8_t)g;
            want[planeOf(2)][(size_t)y * dsts[(size_t)planeOf(2)]->step + (size_t)x] = (uint8_t)b;
          }
        }
    }
    bool failed = false;
    for (size_t p = 0; p < dsts.size() && !failed; ++p) {
      dsts[p]->dev.copyToHost(dsts[p]->host);
      const GenCase dc{(int)dsts[p]->bw, (int)dsts[p]->bh, 0, 0, 0, kSmall, 0};
      const bool ok = DstPacked
                          ? rotate_check<Npp8u, 3, 3>(id, dc, NPPI_INTER_NN, pre[p], dsts[p]->host, want[p],
                                                      dsts[p]->step, std::vector<uint8_t>(), cnt, first): rotate_check<Npp8u, 1, 1>(id, dc, NPPI_INTER_NN, pre[p], dsts[p]->host, want[p],
                                                      dsts[p]->step, std::vector<uint8_t>(), cnt, first);
      if (!ok) failed = true;
    }
    for (auto *p: srcs) delete p;
    for (auto *p: dsts) delete p;
    if (failed || cnt.mismatched >= 8) break;
  }
  finish(id, cnt, first);
}

}  // namespace ycbcr
}  // namespace num
