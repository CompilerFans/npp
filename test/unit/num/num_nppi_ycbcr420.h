// CPU-golden coverage for the video-range YCbCr 4:2:0 / 4:1:1 colour
// conversions (packed C3/AC4 or planar sources, packed C3/C4 or planar
// destinations, including the YCrCb chroma-swapped variants).
//
// Measured NPP 13.4 semantics:
//   * 601 video-range matrices: forward sat(0.257 R + 0.504 G + 0.098 B + 16)
//     and the -0.148/-0.291/0.439 and 0.439/-0.368/-0.071 chroma rows; the
//     inverse is FMA-contracted 1.164*(Y-16) with 1.596/0.813/0.392/2.017;
//   * 4:2:0 luma extent (w&~1)x(h&~1) with (w/2)x(h/2) chroma cells that
//     average the four truncated samples (>>2); odd width or height warns;
//     the 4:2:0 inverse converts the full width but only height & ~1 rows and
//     warns on an odd height;
//   * 4:1:1 luma extent (w&~3)xh with (w/4)xh chroma cells averaging four
//     samples; the forward warns when width % 4 != 0; the inverse converts
//     the full extent and warns when width % 4 != 0;
//   * AC4 source alpha is ignored; the packed C4 destinations write
//     [R,G,B,aAval] or [B,G,R,aAval] (both with the alpha last);
//   * YCrCb variants swap the two chroma planes / destination channels.
#pragma once

#include "num_nppi_geom.h"

namespace num {
namespace ycbcr420 {

inline Npp8u sat(double v) {
  const int t = (int)v;
  return (Npp8u)(t < 0 ? 0 : (t > 255 ? 255 : t));
}

inline void fwd_601(int r, int g, int b, int &y, int &cb, int &cr) {
  const double R = r, G = g, B = b;
  y = sat(0.257 * R + 0.504 * G + 0.098 * B + 16.0);
  cb = sat(-0.148 * R - 0.291 * G + 0.439 * B + 128.0);
  cr = sat(0.439 * R - 0.368 * G - 0.071 * B + 128.0);
}

// 0 = BT.601 video, 1 = BT.709 video (CSC), 2 = BT.709 full range (HDTV).
inline void fwd_m(int m, int r, int g, int b, int &y, int &cb, int &cr) {
  if (m == 0) { fwd_601(r, g, b, y, cb, cr); return; }
  const double R = r, G = g, B = b;
  if (m == 1) {
    y = sat(0.183 * R + 0.614 * G + 0.062 * B + 16.0);
    cb = sat(-0.101 * R - 0.338 * G + 0.439 * B + 128.0);
    cr = sat(0.439 * R - 0.399 * G - 0.040 * B + 128.0);
  } else {
    y = sat(0.213 * R + 0.715 * G + 0.072 * B);
    cb = sat(-0.117 * R - 0.394 * G + 0.511 * B + 128.0);
    cr = sat(0.511 * R - 0.464 * G - 0.047 * B + 128.0);
  }
}



inline void inv_601(int y, int cb, int cr, int &r, int &g, int &b) {
  const float Y = (float)y;
  const float dcb = (float)cb - 128.0f;
  const float dcr = (float)cr - 128.0f;
  const float yt = warp_fmul(1.164f, warp_fsub(Y, 16.0f));
  r = sat((double)warp_fma(1.596f, dcr, yt));
  g = sat((double)warp_fma(-0.392f, dcb, warp_fma(-0.813f, dcr, yt)));
  b = sat((double)warp_fma(2.017f, dcb, yt));
}

inline void inv_m(int m, int y, int cb, int cr, int &r, int &g, int &b) {
  if (m == 0) { inv_601(y, cb, cr, r, g, b); return; }
  const float fy = (float)y;
  const float dcb = (float)cb - 128.0f;
  const float dcr = (float)cr - 128.0f;
  if (m == 1) {
    const float yt = warp_fmul(1.164f, warp_fsub(fy, 16.0f));
    r = sat((double)warp_fma(1.793f, dcr, yt));
    g = sat((double)warp_fsub(warp_fsub(yt, warp_fmul(0.213f, dcb)), warp_fmul(0.534f, dcr)));
    b = sat((double)warp_fma(2.115f, dcb, yt));
  } else {
    r = sat((double)warp_fma(1.54f, dcr, fy));
    g = sat((double)warp_fsub(warp_fsub(fy, warp_fmul(0.183f, dcb)), warp_fmul(0.459f, dcr)));
    b = sat((double)warp_fma(1.816f, dcb, fy));
  }
}
enum class Kind { Fwd420, Fwd411, Inv420, Inv411 };

// srcPp: packed source pitch (3 or 4) for the forward; dstPp: packed
// destination pitch (3 or 4) for the inverse.  bgr: packed RGB order is BGR
// (forward source or inverse destination).  swap: YCrCb (chroma planes or
// channels swapped).  aval: constant alpha for a C4 destination.
template <class Call>
void run_ycbcr420(const char *id, Kind kind, int srcPp, int dstPp, int bgr, int swap, int aval, int matId, Call call) {
  NppStreamContext ctx{};
  if (nppGetStreamContext(&ctx) != NPP_NO_ERROR) {
    GTEST_SKIP() << "[" << id << "] no stream context";
    return;
  }
  RunCounts cnt;
  std::string first;
  struct Cfg { int w, h; };
  static const Cfg cfgs[] = {{8, 6}, {16, 12}, {32, 24}, {64, 48}, {128, 96}, {6, 4}, {10, 6}, {33, 17}, {48, 32}};
  for (const Cfg &cf : cfgs) {
    const int w = cf.w, h = cf.h;
    const bool fwd = kind == Kind::Fwd420 || kind == Kind::Fwd411;
    const uint64_t seed = case_seed(id, w, h, 0, 0, 0, 0);
    std::vector<Buf<Npp8u> *> srcs;
    std::vector<Buf<Npp8u> *> dsts;
    std::vector<std::vector<uint8_t>> pre;
    if (fwd) {
      const int pp = srcPp;
      Buf<Npp8u> *s = new Buf<Npp8u>(w, h, pp, 0, 0);
      fill_regime(*s, pp, seed, kWide, 0, false, false, false);
      s->dev.copyFromHost(s->host);
      srcs.push_back(s);
      for (int p = 0; p < 3; ++p) {
        Buf<Npp8u> *d = new Buf<Npp8u>(w, h, 1, 0, 0x5A);
        d->dev.copyFromHost(d->host);
        dsts.push_back(d);
        pre.push_back(d->host);
      }
    } else {
      planar_fill_src(srcs, 3, w, h, 0, 0, 0, seed, kWide);
      Buf<Npp8u> *d = new Buf<Npp8u>(w, h, dstPp, 0, 0x5A);
      d->dev.copyFromHost(d->host);
      dsts.push_back(d);
      pre.push_back(d->host);
    }
    const Npp8u *sp[4] = {nullptr, nullptr, nullptr, nullptr};
    const Npp8u *ps = nullptr;
    Npp8u *dp[4] = {nullptr, nullptr, nullptr, nullptr};
    Npp8u *pd = nullptr;
    int ssa[4] = {0, 0, 0, 0};
    int dsa[4] = {0, 0, 0, 0};
    if (fwd) {
      ps = srcs[0]->dptr();
      for (int p = 0; p < 3; ++p) {
        dp[p] = dsts[(size_t)p]->dptr();
        dsa[p] = dsts[(size_t)p]->step;
      }
    } else {
      for (int p = 0; p < 3; ++p) {
        sp[p] = srcs[(size_t)p]->dptr();
        ssa[p] = srcs[(size_t)p]->step;
      }
      pd = dsts[0]->dptr();
    }
    const int ss = srcs[0]->step;
    const int ds = dsts[0]->step;
    npp_diff::clear_cuda_error();
    const NppStatus st = call(sp, ps, ss, ssa, dp, pd, ds, dsa, NppiSize{w, h}, ctx);
    if (st != NPP_NO_ERROR && st != NPP_DOUBLE_SIZE_WARNING) {
      ++cnt.skipped;
      if (st == NPP_NOT_IMPLEMENTED_ERROR) ++cnt.notimpl;
      for (auto *p : srcs) delete p;
      for (auto *p : dsts) delete p;
      continue;
    }
    const cudaError_t se = cudaDeviceSynchronize();
    ASSERT_EQ(se, cudaSuccess) << "[" << id << "] " << w << "x" << h << " sync: " << cudaGetErrorString(se);
    ++cnt.executed;

    std::vector<std::vector<uint8_t>> want = pre;
    if (kind == Kind::Fwd420 || kind == Kind::Fwd411) {
      const int spanX = (kind == Kind::Fwd411) ? 4 : 2;
      const int spanY = (kind == Kind::Fwd420) ? 2 : 1;
      const int ew = w & ~(spanX - 1);
      const int eh = (kind == Kind::Fwd420) ? (h & ~1) : h;
      auto px = [&](int x, int y, int &r, int &g, int &b) {
        const Npp8u *s = srcs[0]->hptr() + (size_t)y * srcs[0]->step + (size_t)x * srcPp;
        if (bgr) {
          b = s[0];
          g = s[1];
          r = s[2];
        } else {
          r = s[0];
          g = s[1];
          b = s[2];
        }
      };
      for (int y = 0; y < eh; ++y)
        for (int x = 0; x < ew; ++x) {
          int r, g, b, Y, Cb, Cr;
          px(x, y, r, g, b);
          fwd_m(matId, r, g, b, Y, Cb, Cr);
          want[0][(size_t)y * dsts[0]->step + (size_t)x] = (uint8_t)Y;
        }
      const int cw = w / spanX, ch = h / spanY;
      for (int cy = 0; cy < ch; ++cy)
        for (int cx = 0; cx < cw; ++cx) {
          int sum0 = 0, sum1 = 0;
          for (int dy = 0; dy < spanY; ++dy)
            for (int dx = 0; dx < spanX; ++dx) {
              int r, g, b, Y, Cb, Cr;
              px(cx * spanX + dx, cy * spanY + dy, r, g, b);
              fwd_m(matId, r, g, b, Y, Cb, Cr);
              sum0 += swap ? Cr : Cb;
              sum1 += swap ? Cb : Cr;
            }
          want[1][(size_t)cy * dsts[1]->step + (size_t)cx] = (uint8_t)(sum0 / (spanX * spanY));
          want[2][(size_t)cy * dsts[2]->step + (size_t)cx] = (uint8_t)(sum1 / (spanX * spanY));
        }
    } else {
      // Measured: the 4:2:0 C3 inverse clips to height & ~1; every C4
      // destination and the whole 4:1:1 family convert the full extent.
      const int eh = (kind == Kind::Inv420 && dstPp == 3) ? (h & ~1) : h;
      const int ew = (kind == Kind::Inv411 && dstPp == 3) ? (w & ~3) : w;
      const int spanX = (kind == Kind::Inv411) ? 4 : 2;
      for (int y = 0; y < eh; ++y)
        for (int x = 0; x < ew; ++x) {
          const int cx = x / spanX;
          const int cy = (kind == Kind::Inv420) ? (y >> 1) : y;
          const int Y = srcs[0]->hptr()[(size_t)y * srcs[0]->step + (size_t)x];
          const int c0 = srcs[1]->hptr()[(size_t)cy * srcs[1]->step + (size_t)cx];
          const int c1 = srcs[2]->hptr()[(size_t)cy * srcs[2]->step + (size_t)cx];
          const int cb = swap ? c1 : c0;
          const int cr = swap ? c0 : c1;
          int r, g, b;
          inv_m(matId, Y, cb, cr, r, g, b);
          const size_t base = (size_t)y * dsts[0]->step + (size_t)x * dstPp;
          want[0][base + 0] = (uint8_t)(bgr ? b : r);
          want[0][base + 1] = (uint8_t)g;
          want[0][base + 2] = (uint8_t)(bgr ? r : b);
          if (dstPp == 4) want[0][base + 3] = (uint8_t)aval;
        }
    }
    bool failed = false;
    for (size_t p = 0; p < dsts.size() && !failed; ++p) {
      dsts[p]->dev.copyToHost(dsts[p]->host);
      const GenCase dc{(int)dsts[p]->bw, (int)dsts[p]->bh, 0, 0, 0, kSmall, 0};
      bool ok;
      if (fwd)
        ok = rotate_check<Npp8u, 1, 1>(id, dc, NPPI_INTER_NN, pre[p], dsts[p]->host, want[p], dsts[p]->step,
                                        std::vector<uint8_t>(), cnt, first);
      else if (dstPp == 4)
        ok = rotate_check<Npp8u, 4, 4>(id, dc, NPPI_INTER_NN, pre[p], dsts[p]->host, want[p], dsts[p]->step,
                                        std::vector<uint8_t>(), cnt, first);
      else
        ok = rotate_check<Npp8u, 3, 3>(id, dc, NPPI_INTER_NN, pre[p], dsts[p]->host, want[p], dsts[p]->step,
                                        std::vector<uint8_t>(), cnt, first);
      if (!ok) failed = true;
    }
    for (auto *p : srcs) delete p;
    for (auto *p : dsts) delete p;
    if (failed || cnt.mismatched >= 8) break;
  }
  finish(id, cnt, first);
}

}  // namespace ycbcr420
}  // namespace num
