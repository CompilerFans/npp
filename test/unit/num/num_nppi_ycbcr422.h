// CPU-golden coverage for the video-range 4:2:2 YCbCr/CbYCr/YCrCb colour
// conversions (nppi_c3_conversion_executor "video-range" family and the
// 709HDTV c3color families).
//
// Measured NPP 13.4 semantics:
//   * forward matrices are applied in single precision as
//     sat(yr*R + yg*G + yb*B + yo), chroma pair-averaged with (c0+c1)>>1 over
//     the truncated samples;
//   * inverse: 601 video Yt = 1.164*(Y-16), R = Yt+1.596*dCr,
//     G = Yt-(0.813*dCr+0.392*dCb), B = Yt+2.017*dCb; the 709 variants use
//     the 1.793/-0.213/-0.534/2.115 (video) and 1.54/-0.183/-0.459/1.816
//     (full-range) sets; all truncate to u8;
//   * C2 layouts: cbycr=0 -> [Y0 Cb Y1 Cr] (or [Y0 Cr Y1 Cb] for YCrCb),
//     cbycr=1 -> [Cb Y0 Cr Y1]; the packed C4 destination is [B,G,R,aAval];
//   * extents: packed-source -> C2 keeps the full width (odd trailing pixel
//     uses its own chroma, its Cr slot is skipped when it would cross the
//     row); packed-source -> P3 and planar -> C2 use the even width and warn
//     on odd widths; the C2 inverse converts the full width and reuses the
//     previous pair's Cr for a trailing solo pixel; the planar-source inverse
//     converts the full width but only height & ~1 rows;
//   * nppiRGBToCbYCr422Gamma always returns NPP_STEP_ERROR and never writes.
#pragma once

#include "num_nppi_geom.h"

namespace num {
namespace ycbcr422 {

struct Mat {
  double yr, yg, yb, yo;
  double cbr, cbg, cbb, cbo;
  double crr, crg, crb, cro;
};

inline const Mat &mat601v() {
  static const Mat m{0.257, 0.504, 0.098, 16.0, -0.148, -0.291, 0.439, 128.0, 0.439, -0.368, -0.071, 128.0};
  return m;
}
inline const Mat &mat709v() {
  static const Mat m{0.183, 0.614, 0.062, 16.0, -0.101, -0.338, 0.439, 128.0, 0.439, -0.399, -0.040, 128.0};
  return m;
}
inline const Mat &mat709f() {
  static const Mat m{0.213, 0.715, 0.072, 0.0, -0.117, -0.394, 0.511, 128.0, 0.511, -0.464, -0.047, 128.0};
  return m;
}

inline Npp8u sat(double v) {
  const int t = (int)v;
  return (Npp8u)(t < 0 ? 0 : (t > 255 ? 255 : t));
}
inline void fwd_px(const Mat &m, int r, int g, int b, int &y, int &cb, int &cr) {
  y = sat(m.yr * r + m.yg * g + m.yb * b + m.yo);
  cb = sat(m.cbr * r + m.cbg * g + m.cbb * b + m.cbo);
  cr = sat(m.crr * r + m.crg * g + m.crb * b + m.cro);
}

// invStyle 0: 601 video (chroma terms summed before subtraction, c3conv)
// invStyle 1: generic sequential (c3color: ny - kCb*dCb - kCr*dCr)
// invStyle 2: full-range 709 (no luma offset)
inline void inv_px(int invStyle, int y, int cb, int cr, int &r, int &g, int &b) {
  const double fy = y, dcb = cb - 128.0, dcr = cr - 128.0;
  if (invStyle == 2) {
    r = sat(fy + 1.54 * dcr);
    g = sat(fy - 0.183 * dcb - 0.459 * dcr);
    b = sat(fy + 1.816 * dcb);
  } else if (invStyle == 1) {
    const double ny = 1.164 * (fy - 16.0);
    r = sat(ny + 1.793 * dcr);
    g = sat(ny - 0.213 * dcb - 0.534 * dcr);
    b = sat(ny + 2.115 * dcb);
  } else {
    const double ny = 1.164 * (fy - 16.0);
    r = sat(ny + 1.596 * dcr);
    g = sat(ny - (0.813 * dcr + 0.392 * dcb));
    b = sat(ny + 2.017 * dcb);
  }
}

enum class Kind {
  FwdC3C2,    // packed source -> packed C2 (full width)
  FwdC3P3,    // packed source -> planar (even width, warns)
  FwdP3C2,    // planar source -> packed C2 (even width, warns)
  InvC2C,     // packed C2 source -> packed C3/C4 (full width)
  InvC2P3,    // packed C2 source -> planar (even width, warns)
  InvP3C3,    // planar source -> packed (full width, even height)
  Reorder,    // packed C2 -> packed C2 component reorder
  GammaErr,   // reference returns NPP_STEP_ERROR and writes nothing
};

// a0..a3: Kind::FwdC3C2/InvC2C: packed pitch, bgr, layout, ctx flags;
//          FwdC3P3/InvC2P3: pitch, bgr, layout; FwdP3C2: -; InvP3C3: bgr;
//          FwdC2Mat: matrix id, bgr, layout.
template <class Call>
void run_ycbcr422(const char *id, Kind kind, const Mat *mat, int invStyle, int a0, int a1, int a2, int a3,
                  Call call) {
  NppStreamContext ctx{};
  if (nppGetStreamContext(&ctx) != NPP_NO_ERROR) {
    GTEST_SKIP() << "[" << id << "] no stream context";
    return;
  }
  RunCounts cnt;
  std::string first;
  struct Cfg { int w, h; };
  static const Cfg cfgs[] = {{4, 4}, {8, 6}, {16, 12}, {32, 24}, {64, 48}, {128, 96}, {5, 4}, {17, 13}};
  for (const Cfg &cf : cfgs) {
    const int w = cf.w, h = cf.h;
    const int we = w & ~1;
    const bool srcPacked = kind == Kind::FwdC3C2 || kind == Kind::FwdC3P3 || kind == Kind::GammaErr;
    const bool srcPlanar = kind == Kind::FwdP3C2 || kind == Kind::InvP3C3;
    const bool srcC2 = kind == Kind::InvC2C || kind == Kind::InvC2P3 || kind == Kind::Reorder;
    const bool dstPackedC2 = kind == Kind::FwdC3C2 || kind == Kind::FwdP3C2 || kind == Kind::Reorder;
    const bool dstPlanar = kind == Kind::FwdC3P3 || kind == Kind::InvC2P3;
    const bool dstC3C4 = kind == Kind::InvC2C || kind == Kind::InvP3C3;
    const int srcPp = kind == Kind::GammaErr ? 3 : ((kind == Kind::FwdC3C2 || kind == Kind::FwdC3P3) ? a0 : 1);
    const uint64_t seed = case_seed(id, w, h, 0, 0, 0, 0);
    std::vector<Buf<Npp8u> *> srcs;
    if (srcPacked) {
      Buf<Npp8u> *s = new Buf<Npp8u>(w, h, srcPp, 0, 0);
      fill_regime(*s, srcPp, seed, kWide, 0, false, false, false);
      s->dev.copyFromHost(s->host);
      srcs.push_back(s);
    } else if (srcPlanar) {
      planar_fill_src(srcs, 3, w, h, 0, 0, 0, seed, kWide);
    } else if (srcC2) {
      Buf<Npp8u> *s = new Buf<Npp8u>(w, h, 2, 0, 0);
      fill_regime(*s, 2, seed, kWide, 0, false, false, false);
      s->dev.copyFromHost(s->host);
      srcs.push_back(s);
    }
    std::vector<Buf<Npp8u> *> dsts;
    std::vector<std::vector<uint8_t>> pre;
    if (dstPackedC2) {
      Buf<Npp8u> *d = new Buf<Npp8u>(w, h, 2, 0, 0x5A);
      d->dev.copyFromHost(d->host);
      dsts.push_back(d);
      pre.push_back(d->host);
    } else if (dstPlanar) {
      for (int p = 0; p < 3; ++p) {
        Buf<Npp8u> *d = new Buf<Npp8u>(w, h, 1, 0, 0x5A);
        d->dev.copyFromHost(d->host);
        dsts.push_back(d);
        pre.push_back(d->host);
      }
    } else if (kind == Kind::GammaErr) {
      Buf<Npp8u> *d = new Buf<Npp8u>(w, h, 3, 0, 0x5A);
      d->dev.copyFromHost(d->host);
      dsts.push_back(d);
      pre.push_back(d->host);
    } else if (dstC3C4) {
      const int pp = (kind == Kind::InvC2C) ? std::max(3, a0) : 3;
      Buf<Npp8u> *d = new Buf<Npp8u>(w, h, pp, 0, 0x5A);
      d->dev.copyFromHost(d->host);
      dsts.push_back(d);
      pre.push_back(d->host);
    } else {
      Buf<Npp8u> *d = new Buf<Npp8u>(w, h, 1, 0, 0x5A);
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
    if (srcPacked || srcC2) {
      ps = srcs[0]->dptr();
    } else {
      for (int p = 0; p < 3; ++p) {
        sp[p] = srcs[(size_t)p]->dptr();
        ssa[p] = srcs[(size_t)p]->step;
      }
    }
    if (dstPackedC2 || dstC3C4 || kind == Kind::GammaErr) {
      pd = dsts[0]->dptr();
    } else {
      for (size_t p = 0; p < dsts.size(); ++p) {
        dp[p] = dsts[p]->dptr();
        dsa[p] = dsts[p]->step;
      }
    }
    const int ss = srcs[0]->step;
    const int ds = dsts[0]->step;
    npp_diff::clear_cuda_error();
    const NppStatus st = call(sp, ps, ss, ssa, dp, pd, ds, dsa, NppiSize{w, h}, ctx);
    if (kind == Kind::GammaErr) {
      // Measured: the reference writes the destination and returns NO_ERROR
      // when the destination step is at least 3*width, and returns
      // NPP_STEP_ERROR without writing for a smaller step.
      ++cnt.executed;
      EXPECT_EQ(st, NPP_NO_ERROR) << "[" << id << "] " << w << "x" << h;
      ++cnt.compared;
      dsts[0]->dev.copyToHost(dsts[0]->host);
      const std::vector<uint8_t> after1 = dsts[0]->host;
      npp_diff::clear_cuda_error();
      const NppStatus st2 = call(sp, ps, ss, ssa, dp, pd, 2 * w, dsa, NppiSize{w, h}, ctx);
      cudaDeviceSynchronize();
      ++cnt.compared;
      dsts[0]->dev.copyToHost(dsts[0]->host);
      if (st2 != NPP_STEP_ERROR || dsts[0]->host != after1) {
        ++cnt.mismatched;
        EXPECT_EQ(st2, NPP_STEP_ERROR) << "[" << id << "] " << w << "x" << h << " small-step call";
        EXPECT_EQ(dsts[0]->host, after1) << "[" << id << "] " << w << "x" << h
                                         << " small-step call wrote the destination";
      }
      for (auto *p : srcs) delete p;
      for (auto *p : dsts) delete p;
      continue;
    }
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

    auto srcPx = [&](int x, int y, int &r, int &g, int &b) {
      const Npp8u *s = srcs[0]->hptr() + (size_t)y * srcs[0]->step + (size_t)x * srcPp;
      if (a1) {  // bgr
        b = s[0];
        g = s[1];
        r = s[2];
      } else {
        r = s[0];
        g = s[1];
        b = s[2];
      }
    };
    std::vector<std::vector<uint8_t>> want = pre;
    if (kind == Kind::FwdC3C2) {
      const int cbycr = a2;
      for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) {
          int r, g, b, Y, Cb, Cr;
          srcPx(x, y, r, g, b);
          fwd_px(*mat, r, g, b, Y, Cb, Cr);
          const size_t row = (size_t)y * dsts[0]->step;
          want[0][row + (size_t)(cbycr ? 2 * x + 1 : 2 * x)] = (uint8_t)Y;
          if ((x & 1) == 0) {
            int cb = Cb, cr = Cr;
            if (x + 1 < w) {
              int r1, g1, b1, Y1, Cb1, Cr1;
              srcPx(x + 1, y, r1, g1, b1);
              fwd_px(*mat, r1, g1, b1, Y1, Cb1, Cr1);
              cb = (Cb + Cb1) >> 1;
              cr = (Cr + Cr1) >> 1;
            }
            want[0][row + (size_t)(cbycr ? 2 * x : 2 * x + 1)] = (uint8_t)cb;
            const int crOff = cbycr ? 2 * x + 2 : 2 * x + 3;
            if (crOff < 2 * w) want[0][row + (size_t)crOff] = (uint8_t)cr;
          }
        }
    } else if (kind == Kind::FwdC3P3) {
      for (int y = 0; y < h; ++y)
        for (int x = 0; x < we; ++x) {
          int r, g, b, Y, Cb, Cr;
          srcPx(x, y, r, g, b);
          fwd_px(*mat, r, g, b, Y, Cb, Cr);
          want[0][(size_t)y * dsts[0]->step + (size_t)x] = (uint8_t)Y;
          if ((x & 1) == 0) {
            int r1, g1, b1, Y1, Cb1, Cr1;
            srcPx(x + 1, y, r1, g1, b1);
            fwd_px(*mat, r1, g1, b1, Y1, Cb1, Cr1);
            want[1][(size_t)y * dsts[1]->step + (size_t)(x >> 1)] = (uint8_t)((Cb + Cb1) >> 1);
            want[2][(size_t)y * dsts[2]->step + (size_t)(x >> 1)] = (uint8_t)((Cr + Cr1) >> 1);
          }
        }
    } else if (kind == Kind::FwdP3C2) {
      for (int y = 0; y < h; ++y)
        for (int p = 0; p < (we >> 1); ++p) {
          int Y0, Cb0, Cr0, Y1, Cb1, Cr1;
          const int x = 2 * p;
          fwd_px(*mat, srcs[0]->hptr()[(size_t)y * srcs[0]->step + x], srcs[1]->hptr()[(size_t)y * srcs[1]->step + x],
                 srcs[2]->hptr()[(size_t)y * srcs[2]->step + x], Y0, Cb0, Cr0);
          fwd_px(*mat, srcs[0]->hptr()[(size_t)y * srcs[0]->step + x + 1],
                 srcs[1]->hptr()[(size_t)y * srcs[1]->step + x + 1],
                 srcs[2]->hptr()[(size_t)y * srcs[2]->step + x + 1], Y1, Cb1, Cr1);
          const size_t base = (size_t)y * dsts[0]->step + (size_t)p * 4;
          want[0][base + 0] = (uint8_t)Y0;
          want[0][base + 1] = (uint8_t)((Cb0 + Cb1) >> 1);
          want[0][base + 2] = (uint8_t)Y1;
          want[0][base + 3] = (uint8_t)((Cr0 + Cr1) >> 1);
        }
        } else if (kind == Kind::InvC2C) {
      const int dstPp = a0, bgr = a1, cbycr = a2;
      const Npp8u aval = (Npp8u)a3;
      for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) {
          const Npp8u *row = srcs[0]->hptr() + (size_t)y * srcs[0]->step;
          const int yv = row[cbycr ? 2 * x + 1 : 2 * x];
          const int p = x >> 1;
          const int cb = row[cbycr ? 4 * p : 4 * p + 1];
          const int pp = ((cbycr ? 4 * p + 2 : 4 * p + 3) < 2 * w) ? p : (p > 0 ? p - 1 : 0);
          const int cr = row[cbycr ? 4 * pp + 2 : 4 * pp + 3];
          int r, g, b;
          inv_px(invStyle, yv, cb, cr, r, g, b);
          const size_t base = (size_t)y * dsts[0]->step + (size_t)x * dstPp;
          if (dstPp == 4) {
            want[0][base + 0] = (uint8_t)b;
            want[0][base + 1] = (uint8_t)g;
            want[0][base + 2] = (uint8_t)r;
            want[0][base + 3] = aval;
          } else if (bgr) {
            want[0][base + 0] = (uint8_t)b;
            want[0][base + 1] = (uint8_t)g;
            want[0][base + 2] = (uint8_t)r;
          } else {
            want[0][base + 0] = (uint8_t)r;
            want[0][base + 1] = (uint8_t)g;
            want[0][base + 2] = (uint8_t)b;
          }
        }
    } else if (kind == Kind::InvC2P3) {
      const int cbycr = a0;
      for (int y = 0; y < h; ++y)
        for (int x = 0; x < we; ++x) {
          const Npp8u *row = srcs[0]->hptr() + (size_t)y * srcs[0]->step;
          const int yv = row[cbycr ? 2 * x + 1 : 2 * x];
          const int p = x >> 1;
          const int cb = row[cbycr ? 4 * p : 4 * p + 1];
          const int cr = row[cbycr ? 4 * p + 2 : 4 * p + 3];
          int r, g, b;
          inv_px(invStyle, yv, cb, cr, r, g, b);
          want[0][(size_t)y * dsts[0]->step + (size_t)x] = (uint8_t)r;
          want[1][(size_t)y * dsts[1]->step + (size_t)x] = (uint8_t)g;
          want[2][(size_t)y * dsts[2]->step + (size_t)x] = (uint8_t)b;
        }
    } else if (kind == Kind::InvP3C3) {
      const int bgr = a0;
      const int hh = h & ~1;
      for (int y = 0; y < hh; ++y)
        for (int x = 0; x < w; ++x) {
          const int yv = srcs[0]->hptr()[(size_t)y * srcs[0]->step + (size_t)x];
          const int cb = srcs[1]->hptr()[(size_t)y * srcs[1]->step + (size_t)(x >> 1)];
          const int cr = srcs[2]->hptr()[(size_t)y * srcs[2]->step + (size_t)(x >> 1)];
          int r, g, b;
          inv_px(invStyle, yv, cb, cr, r, g, b);
          const size_t base = (size_t)y * dsts[0]->step + (size_t)x * 3;
          want[0][base + 0] = (uint8_t)(bgr ? b : r);
          want[0][base + 1] = (uint8_t)g;
          want[0][base + 2] = (uint8_t)(bgr ? r : b);
        }
    } else {  // Reorder [Y0 Cb Y1 Cr] -> [Cb Y0 Cr Y1]
      for (int y = 0; y < h; ++y)
        for (int p = 0; p < (w >> 1); ++p) {
          const Npp8u *s = srcs[0]->hptr() + (size_t)y * srcs[0]->step + (size_t)p * 4;
          const size_t base = (size_t)y * dsts[0]->step + (size_t)p * 4;
          want[0][base + 0] = s[1];
          want[0][base + 1] = s[0];
          want[0][base + 2] = s[3];
          want[0][base + 3] = s[2];
        }
    }
    bool failed = false;
    for (size_t p = 0; p < dsts.size() && !failed; ++p) {
      dsts[p]->dev.copyToHost(dsts[p]->host);
      const GenCase dc{(int)dsts[p]->bw, (int)dsts[p]->bh, 0, 0, 0, kSmall, 0};
      bool ok;
      if (dstPlanar || (!dstPackedC2 && !dstC3C4))
        ok = rotate_check<Npp8u, 1, 1>(id, dc, NPPI_INTER_NN, pre[p], dsts[p]->host, want[p], dsts[p]->step,
                                        std::vector<uint8_t>(), cnt, first);
      else if (dstPackedC2)
        ok = rotate_check<Npp8u, 2, 2>(id, dc, NPPI_INTER_NN, pre[p], dsts[p]->host, want[p], dsts[p]->step,
                                        std::vector<uint8_t>(), cnt, first);
      else if (dsts[0]->bw >= 0 && a0 == 4)
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

}  // namespace ycbcr422
}  // namespace num
