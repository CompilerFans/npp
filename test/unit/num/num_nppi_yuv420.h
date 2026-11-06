// CPU-golden coverage for the YUV420/YUV422 conversion families (8u).
//
// Measured NPP 13.4 semantics on the 5090:
//   inverse (YUV -> RGB/BGR): R = Y + 1.13983*dV; G = Y - 0.39465*dU -
//         0.58060*dV; B = Y + 2.03211*dU (all products rounded to float),
//         truncated to u8; verified over all 2^24 (Y,U,V) triples;
//         4:2:0 reads chroma at (x/2,y/2), 4:2:2 at (x/2,y).  The converted
//         extent is the full width but only height & ~1 rows, and an odd
//         height returns NPP_DOUBLE_SIZE_WARNING.
//   forward (RGB -> YUV): Y = fma(.114,B, fma(.299,R, .587*G));
//         U = .492*(B-Y)+128, V = .877*(R-Y)+128 (float Y); chroma cells are
//         the integer average of the truncated per-pixel U/V.  The 4:2:0
//         luma extent is (w&~1)x(h&~1) with (w/2)x(h/2) chroma; 4:2:2 is
//         (w&~1)xh with (w/2)xh chroma.  Odd widths (and odd heights for
//         4:2:0) warn with NPP_DOUBLE_SIZE_WARNING.
//   alpha: P3AC4R writes RGBA with alpha 0.  The P3C4R "constant alpha"
//         reference kernels address the luma plane at 4x the horizontal
//         pixel index (out-of-bounds reads), so only their constant-alpha
//         byte is deterministic: byte 0 of each pixel for the RGB variant
//         (ARGB) and byte 3 for the BGR variant (BGRA); those are checked
//         structurally by run_yuv_c4_alpha.
#pragma once

#include "num_nppi_geom.h"

namespace num {
namespace yuvplanar {

enum class Sub { S420, S422 };
enum class Out { P3, C3, AC4 };  // P3 = three planes, C3 = packed RGB, AC4 = packed RGBA
enum class In { P3, C2 };        // C2 = interleaved Y0 U Y1 V source

inline Npp8u q8(double v) {
  int t = (int)v;
  return (Npp8u)(t < 0 ? 0 : (t > 255 ? 255 : t));
}

inline void jfif_inv(int y, int u, int v, int &r, int &g, int &b) {
  const float fy = (float)y;
  const float du = warp_fsub((float)u, 128.0f);
  const float dv = warp_fsub((float)v, 128.0f);
  r = q8(warp_fadd(fy, warp_fmul(1.13983f, dv)));
  g = q8(warp_fsub(warp_fsub(fy, warp_fmul(0.39465f, du)), warp_fmul(0.58060f, dv)));
  b = q8(warp_fadd(fy, warp_fmul(2.03211f, du)));
}

inline void yuv_fwd(int r, int g, int b, int &y, int &u, int &v) {
  const float fr = (float)r, fg = (float)g, fb = (float)b;
  const float fy = warp_fma(0.114f, fb, warp_fma(0.299f, fr, warp_fmul(0.587f, fg)));
  y = q8(fy);
  u = q8(warp_fadd(warp_fmul(0.492f, warp_fsub(fb, fy)), 128.0f));
  v = q8(warp_fadd(warp_fmul(0.877f, warp_fsub(fr, fy)), 128.0f));
}

template <Sub S> inline int chromaW(int w) { return w / 2; }
template <Sub S> inline int chromaH(int h) { return S == Sub::S420 ? h / 2 : h; }

// SrcPacked: packed C3 source (else three planes); DstC2: interleaved
// Y0 U Y1 V destination (else three planes).
template <Sub S, bool SrcPacked, bool DstC2, bool Bgr = false, int SrcPp = 3, class Call>
void run_yuv_fwd(const char *id, Call call) {
  NppStreamContext ctx{};
  if (nppGetStreamContext(&ctx) != NPP_NO_ERROR) {
    GTEST_SKIP() << "[" << id << "] no stream context";
    return;
  }
  RunCounts cnt;
  std::string first;
  struct Cfg { int w, h; };
  static const Cfg cfgs[] = {{4, 4},   {8, 6},   {16, 12}, {32, 24},  {64, 48},
                             {128, 96}, {5, 4},   {4, 3},   {6, 8},    {33, 17}};
  for (const Cfg &cf : cfgs) {
    if (DstC2 && (cf.w & 1)) continue;  // interleaved pairs need even width
    const uint64_t seed = case_seed(id, cf.w, cf.h, 0, 0, 0, 0);
    std::vector<Buf<Npp8u> *> srcs;
    if constexpr (SrcPacked) {
      Buf<Npp8u> *a = new Buf<Npp8u>(cf.w, cf.h, SrcPp, 0, 0);
      fill_regime(*a, SrcPp, seed, kWide, 0, false, false, false);
      a->dev.copyFromHost(a->host);
      srcs.push_back(a);
    } else {
      planar_fill_src(srcs, 3, cf.w, cf.h, 0, 0, 0, seed, kWide);
    }
    const int cw = chromaW<S>(cf.w), ch = chromaH<S>(cf.h);
    std::vector<Buf<Npp8u> *> dsts;
    std::vector<std::vector<uint8_t>> pre;
    if constexpr (DstC2) {
      Buf<Npp8u> *d = new Buf<Npp8u>(cf.w, cf.h, 2, 0, 0x5A);
      d->dev.copyFromHost(d->host);
      dsts.push_back(d);
      pre.push_back(d->host);
    } else {
      for (int p = 0; p < 3; ++p) {
        const int dw = (p == 0) ? cf.w : cw;
        const int dh = (p == 0) ? cf.h : ch;
        Buf<Npp8u> *d = new Buf<Npp8u>(std::max(1, dw), std::max(1, dh), 1, 0, 0x5A);
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
    if constexpr (DstC2) {
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
      for (auto *p : srcs) delete p;
      for (auto *p : dsts) delete p;
      continue;
    }
    const cudaError_t se = cudaDeviceSynchronize();
    ASSERT_EQ(se, cudaSuccess) << "[" << id << "] " << cf.w << "x" << cf.h << " sync: " << cudaGetErrorString(se);
    ++cnt.executed;

    const int ew = cf.w & ~1;
    const int eh = (S == Sub::S420) ? (cf.h & ~1) : cf.h;
    auto srcAt = [&](int c, int x, int y) -> int {
      const int p = Bgr ? 2 - c : c;
      if constexpr (SrcPacked) return srcs[0]->hptr()[(size_t)y * srcs[0]->step + (size_t)x * SrcPp + (size_t)p];
      else return srcs[(size_t)p]->hptr()[(size_t)y * srcs[(size_t)p]->step + (size_t)x];
    };
    std::vector<std::vector<uint8_t>> want = pre;
    for (int y = 0; y < eh; ++y)
      for (int x = 0; x < ew; ++x) {
        int Y, U, V;
        yuv_fwd(srcAt(0, x, y), srcAt(1, x, y), srcAt(2, x, y), Y, U, V);
        if constexpr (DstC2) {
          // [Y0 U Y1 V]; the pair chroma is the average of the two pixels.
          const size_t p = (size_t)y * dsts[0]->step + (size_t)(x & ~1) * 2;
          want[0][p + (size_t)((x & 1) * 2)] = (uint8_t)Y;
          if ((x & 1) == 0) {
            int Y2, U2, V2;
            yuv_fwd(srcAt(0, x + 1, y), srcAt(1, x + 1, y), srcAt(2, x + 1, y), Y2, U2, V2);
            want[0][p + 1] = (uint8_t)((U + U2) / 2);
            want[0][p + 3] = (uint8_t)((V + V2) / 2);
          }
        } else {
          want[0][(size_t)y * dsts[0]->step + (size_t)x] = (uint8_t)Y;
        }
      }
    if constexpr (!DstC2) {
      for (int cy = 0; cy < ch; ++cy)
        for (int cx = 0; cx < cw; ++cx) {
          int sumU = 0, sumV = 0, n = 0;
          const int spanX = 2, spanY = (S == Sub::S420) ? 2 : 1;
          for (int dy = 0; dy < spanY; ++dy)
            for (int dx = 0; dx < spanX; ++dx) {
              int Y, U, V;
              yuv_fwd(srcAt(0, cx * spanX + dx, cy * spanY + dy), srcAt(1, cx * spanX + dx, cy * spanY + dy),
                      srcAt(2, cx * spanX + dx, cy * spanY + dy), Y, U, V);
              sumU += U;
              sumV += V;
              ++n;
            }
          want[1][(size_t)cy * dsts[1]->step + (size_t)cx] = (uint8_t)(sumU / n);
          want[2][(size_t)cy * dsts[2]->step + (size_t)cx] = (uint8_t)(sumV / n);
        }
    }
    bool failed = false;
    for (size_t p = 0; p < dsts.size() && !failed; ++p) {
      dsts[p]->dev.copyToHost(dsts[p]->host);
      const GenCase dc{(int)dsts[p]->bw, (int)dsts[p]->bh, 0, 0, 0, kSmall, 0};
      const bool ok = DstC2
                          ? rotate_check<Npp8u, 2, 2>(id, dc, NPPI_INTER_NN, pre[p], dsts[p]->host, want[p],
                                                      dsts[p]->step, std::vector<uint8_t>(), cnt, first)
                          : rotate_check<Npp8u, 1, 1>(id, dc, NPPI_INTER_NN, pre[p], dsts[p]->host, want[p],
                                                      dsts[p]->step, std::vector<uint8_t>(), cnt, first);
      if (!ok) failed = true;
    }
    for (auto *p : srcs) delete p;
    for (auto *p : dsts) delete p;
    if (failed || cnt.mismatched >= 8) break;
  }
  finish(id, cnt, first);
}

// Src: P3 planes or interleaved C2; Out: three planes, packed RGB or packed
// RGBA; Bgr swaps the destination colour channels.
template <Sub S, In Src, Out O, bool Bgr, class Call>
void run_yuv_inv(const char *id, Call call) {
  NppStreamContext ctx{};
  if (nppGetStreamContext(&ctx) != NPP_NO_ERROR) {
    GTEST_SKIP() << "[" << id << "] no stream context";
    return;
  }
  RunCounts cnt;
  std::string first;
  struct Cfg { int w, h; };
  static const Cfg cfgs[] = {{4, 4},   {8, 6},   {16, 12}, {32, 24}, {64, 48},
                             {128, 96}, {5, 4},   {4, 3},   {6, 8},   {33, 17}};
  for (const Cfg &cf : cfgs) {
    if (Src == In::C2 && (cf.w & 1)) continue;
    const int cw = chromaW<S>(cf.w), ch = chromaH<S>(cf.h);
    std::vector<Buf<Npp8u> *> srcs;
    const uint64_t seed = case_seed(id, cf.w, cf.h, 0, 0, 0, 0);
    if constexpr (Src == In::C2) {
      Buf<Npp8u> *a = new Buf<Npp8u>(cf.w, cf.h, 2, 0, 0);
      fill_regime(*a, 2, seed, kWide, 0, false, false, false);
      a->dev.copyFromHost(a->host);
      srcs.push_back(a);
    } else {
      planar_fill_src(srcs, 3, cf.w, cf.h, 0, 0, 0, seed, kWide);
      // chroma planes only need half width/height; the extra rows/cols are
      // simply never read.
    }
    std::vector<Buf<Npp8u> *> dsts;
    std::vector<std::vector<uint8_t>> pre;
    if constexpr (O == Out::P3) {
      for (int p = 0; p < 3; ++p) {
        Buf<Npp8u> *d = new Buf<Npp8u>(cf.w, cf.h, 1, 0, 0x5A);
        d->dev.copyFromHost(d->host);
        dsts.push_back(d);
        pre.push_back(d->host);
      }
    } else {
      const int c = (O == Out::AC4) ? 4 : 3;
      Buf<Npp8u> *d = new Buf<Npp8u>(cf.w, cf.h, c, 0, 0x5A);
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
    if constexpr (Src == In::C2) {
      ps = srcs[0]->dptr();
    } else {
      for (int p = 0; p < 3; ++p) {
        sp[p] = srcs[(size_t)p]->dptr();
        ssa[p] = srcs[(size_t)p]->step;
      }
    }
    if constexpr (O == Out::P3) {
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
    NppStatus st = call(sp, ps, ss, ssa, dp, pd, ds, dsa, NppiSize{cf.w, cf.h}, ctx);
    if (st != NPP_NO_ERROR && st != NPP_DOUBLE_SIZE_WARNING) {
      ++cnt.skipped;
      if (st == NPP_NOT_IMPLEMENTED_ERROR) ++cnt.notimpl;
      for (auto *p : srcs) delete p;
      for (auto *p : dsts) delete p;
      continue;
    }
    const cudaError_t se = cudaDeviceSynchronize();
    ASSERT_EQ(se, cudaSuccess) << "[" << id << "] " << cf.w << "x" << cf.h << " sync: " << cudaGetErrorString(se);
    ++cnt.executed;

    // Measured: the planar-source inverse converts the full width but only
    // height & ~1 rows; the interleaved C2 variant converts every row.
    const int eh = (Src == In::C2) ? cf.h : (cf.h & ~1);
    auto srcAt = [&](int c, int x, int y) -> int {
      if constexpr (Src == In::C2) {
        const size_t base = (size_t)y * srcs[0]->step + (size_t)(x & ~1) * 2;
        const int h[4] = {0, 1, 2, 3};
        return srcs[0]->hptr()[base + (size_t)h[c]];
      } else {
        return srcs[(size_t)c]->hptr()[(size_t)y * srcs[(size_t)c]->step + (size_t)x];
      }
    };
    auto planeOf = [&](int c) -> int { return Bgr ? 2 - c : c; };
    std::vector<std::vector<uint8_t>> want = pre;
    for (int y = 0; y < eh; ++y)
      for (int x = 0; x < cf.w; ++x) {
        const int cx = x >> 1, cy = (S == Sub::S420) ? (y >> 1) : y;
        int Y, U, V;
        if constexpr (Src == In::C2) {
          const size_t base = (size_t)y * srcs[0]->step + (size_t)(x & ~1) * 2;
          Y = srcs[0]->hptr()[base + (size_t)((x & 1) * 2)];
          U = srcs[0]->hptr()[base + 1];
          V = srcs[0]->hptr()[base + 3];
        } else {
          Y = srcAt(0, x, y);
          U = srcAt(1, cx, cy);
          V = srcAt(2, cx, cy);
        }
        int r, g, b;
        jfif_inv(Y, U, V, r, g, b);
        if constexpr (O == Out::P3) {
          want[planeOf(0)][(size_t)y * dsts[(size_t)planeOf(0)]->step + (size_t)x] = (uint8_t)r;
          want[planeOf(1)][(size_t)y * dsts[(size_t)planeOf(1)]->step + (size_t)x] = (uint8_t)g;
          want[planeOf(2)][(size_t)y * dsts[(size_t)planeOf(2)]->step + (size_t)x] = (uint8_t)b;
        } else {
          const size_t p = (size_t)y * dsts[0]->step + (size_t)x * (O == Out::AC4 ? 4 : 3);
          want[0][p + (size_t)planeOf(0)] = (uint8_t)r;
          want[0][p + (size_t)planeOf(1)] = (uint8_t)g;
          want[0][p + (size_t)planeOf(2)] = (uint8_t)b;
          if constexpr (O == Out::AC4) want[0][p + 3] = 0;
        }
      }
    bool failed = false;
    for (size_t p = 0; p < dsts.size() && !failed; ++p) {
      dsts[p]->dev.copyToHost(dsts[p]->host);
      const GenCase dc{(int)dsts[p]->bw, (int)dsts[p]->bh, 0, 0, 0, kSmall, 0};
      const bool ok =
          (O == Out::P3)
              ? rotate_check<Npp8u, 1, 1>(id, dc, NPPI_INTER_NN, pre[p], dsts[p]->host, want[p], dsts[p]->step,
                                          std::vector<uint8_t>(), cnt, first)
              : (O == Out::AC4 ? rotate_check<Npp8u, 4, 4>(id, dc, NPPI_INTER_NN, pre[p], dsts[p]->host, want[p],
                                                            dsts[p]->step, std::vector<uint8_t>(), cnt, first)
                               : rotate_check<Npp8u, 3, 3>(id, dc, NPPI_INTER_NN, pre[p], dsts[p]->host, want[p],
                                                           dsts[p]->step, std::vector<uint8_t>(), cnt, first));
      if (!ok) failed = true;
    }
    for (auto *p : srcs) delete p;
    for (auto *p : dsts) delete p;
    if (failed || cnt.mismatched >= 8) break;
  }
  finish(id, cnt, first);
}

// Structural check for the P3C4R "constant alpha" references, whose RGB
// addressing is broken upstream (luma read at 4x the horizontal index, with
// out-of-bounds reads).  Only the constant alpha byte is deterministic:
// byte 0 of each pixel for the RGB/ARGB variant, byte 3 for BGR/BGRA.
template <Sub S, bool Bgr, class Call>
void run_yuv_c4_alpha(const char *id, Call call) {
  NppStreamContext ctx{};
  if (nppGetStreamContext(&ctx) != NPP_NO_ERROR) {
    GTEST_SKIP() << "[" << id << "] no stream context";
    return;
  }
  RunCounts cnt;
  std::string first;
  struct Cfg { int w, h; };
  static const Cfg cfgs[] = {{4, 4}, {8, 6}, {16, 12}, {32, 24}, {64, 48}, {128, 96}, {5, 4}, {4, 3}};
  for (const Cfg &cf : cfgs) {
    const int cw = chromaW<S>(cf.w), ch = chromaH<S>(cf.h);
    std::vector<Buf<Npp8u> *> srcs;
    const uint64_t seed = case_seed(id, cf.w, cf.h, 0, 0, 0, 0);
    planar_fill_src(srcs, 3, cf.w, cf.h, 0, 0, 0, seed, kWide);
    Buf<Npp8u> *d = new Buf<Npp8u>(cf.w, cf.h, 4, 0, 0x5A);
    d->dev.copyFromHost(d->host);
    const Npp8u *sp[4] = {srcs[0]->dptr(), srcs[1]->dptr(), srcs[2]->dptr(), nullptr};
    int ssa[4] = {srcs[0]->step, srcs[1]->step, srcs[2]->step, 0};
    const int ss = srcs[0]->step;
    npp_diff::clear_cuda_error();
    NppStatus st = call(sp, nullptr, ss, ssa, nullptr, d->dptr(), d->step, nullptr, NppiSize{cf.w, cf.h}, ctx);
    if (st != NPP_NO_ERROR && st != NPP_DOUBLE_SIZE_WARNING) {
      ++cnt.skipped;
      if (st == NPP_NOT_IMPLEMENTED_ERROR) ++cnt.notimpl;
      for (auto *p : srcs) delete p;
      delete d;
      continue;
    }
    const cudaError_t se = cudaDeviceSynchronize();
    ASSERT_EQ(se, cudaSuccess) << "[" << id << "] " << cf.w << "x" << cf.h << " sync: " << cudaGetErrorString(se);
    ++cnt.executed;
    d->dev.copyToHost(d->host);
    const int alpha_off = Bgr ? 3 : 0;
    int bad = 0;
    for (int y = 0; y < cf.h; ++y)
      for (int x = 0; x < cf.w; ++x) {
        ++cnt.compared;
        const uint8_t a = d->host[(size_t)y * d->step + (size_t)x * 4 + (size_t)alpha_off];
        if (a != 0xFF) {
          ++bad;
          if (first.empty()) {
            char b[200];
            std::snprintf(b, sizeof(b), "[%s] %dx%d alpha(%d,%d)=%u", id, cf.w, cf.h, x, y, a);
            first = b;
          }
        }
      }
    if (bad) {
      cnt.mismatched += (uint64_t)bad;
      EXPECT_TRUE(false) << first;
    }
    (void)cw;
    (void)ch;
    for (auto *p : srcs) delete p;
    delete d;
    if (bad) break;
  }
  finish(id, cnt, first);
}

}  // namespace yuvplanar
}  // namespace num
