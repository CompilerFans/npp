// CPU-golden coverage for the derivative NPP filters
// (nppi_filtering_functions.h):
//   nppiFilter{PrewittHoriz,PrewittVert,RobertsDown,RobertsUp,
//              SobelHoriz,SobelVert,ScharrHoriz,ScharrVert,
//              SobelCross,SobelHorizSecond,SobelVertSecond}
// plus the *Border and *Mask variants.
//
// Taps measured on NVIDIA NPP 13.4 (RTX 5090); note that PrewittHoriz /
// SobelHoriz are negated w.r.t. the published documentation (top row -1):
//   PrewittHoriz [[-1,-1,-1],[0,0,0],[1,1,1]]
//   PrewittVert  [[-1,0,1],[-1,0,1],[-1,0,1]]
//   RobertsDown  center - top-left,  RobertsUp center - top-right
//   SobelHoriz   [[-1,-2,-1],[0,0,0],[1,2,1]]
//   SobelVert    [[-1,0,1],[-2,0,2],[-1,0,1]]
//   ScharrHoriz  [[-3,-10,-3],[0,0,0],[3,10,3]]
//   ScharrVert   [[-3,0,3],[-10,0,10],[-3,0,3]]
//   SobelCross / *Second, 5x5 Sobel taps: see the coefficient tables below.
// Integer sources accumulate exactly (int) and the result is saturated to the
// destination type after a float conversion; float sources accumulate in
// float.  AC4 leaves alpha untouched.
//
// Plain variants read raw neighbours around pSrc (caller-provided margin);
// Border variants accept NPP_BORDER_REPLICATE only and clamp reads to the
// oSrcSize image relative to pSrc.  The driver allocates a two-pixel margin
// and sets pSrc to (2,2) inside the image.
#pragma once

#include "num_nppi_geom.h"

namespace num {
namespace filtderiv {

enum Op {
  kPrewittHoriz = 0,
  kPrewittVert,
  kRobertsDown,
  kRobertsUp,
  kSobelHoriz,
  kSobelVert,
  kScharrHoriz,
  kScharrVert,
  kSobelCross,
  kSobelHorizSecond,
  kSobelVertSecond,
};

struct Coef {
  int size;
  int k[5][5];
};

inline Coef mk3(const int *k9) {
  Coef c{};
  c.size = 3;
  for (int a = 0; a < 3; ++a)
    for (int b = 0; b < 3; ++b) c.k[a][b] = k9[a * 3 + b];
  return c;
}

inline Coef mk5(const int *k25) {
  Coef c{};
  c.size = 5;
  for (int a = 0; a < 5; ++a)
    for (int b = 0; b < 5; ++b) c.k[a][b] = k25[a * 5 + b];
  return c;
}

inline Coef coefOf(int op, int maskSize) {
  static const int PrewittHoriz3[9] = {-1, -1, -1, 0, 0, 0, 1, 1, 1};
  static const int PrewittVert3[9] = {-1, 0, 1, -1, 0, 1, -1, 0, 1};
  static const int RobertsDown3[9] = {-1, 0, 0, 0, 1, 0, 0, 0, 0};
  static const int RobertsUp3[9] = {0, 0, -1, 0, 1, 0, 0, 0, 0};
  static const int SobelHoriz3[9] = {-1, -2, -1, 0, 0, 0, 1, 2, 1};
  static const int SobelVert3[9] = {-1, 0, 1, -2, 0, 2, -1, 0, 1};
  static const int SobelHoriz5[25] = {-1, -4, -6, -4, -1, -2, -8, -12, -8, -2, 0, 0, 0, 0, 0,
                                      2, 8,  12, 8,  2,  1,  4,  6,   4,  1};
  static const int SobelVert5[25] = {-1, -2, 0,  2,  1,  -4, -8, 0,  8,  4,  -6, -12, 0,  12, 6,
                                     -4, -8, 0,  8,  4,  -1, -2, 0,  2,  1};
  static const int ScharrHoriz3[9] = {-3, -10, -3, 0, 0, 0, 3, 10, 3};
  static const int ScharrVert3[9] = {-3, 0, 3, -10, 0, 10, -3, 0, 3};
  static const int Cross3[9] = {-1, 0, 1, 0, 0, 0, 1, 0, -1};
  static const int Cross5[25] = {-1, -2, 0,  2,  1,  -2, -4, 0,  4,  2, 0, 0,  0,  0,  0,
                                 2,  4,  0,  -4, -2, 1,   2,  0,  -2, -1};
  static const int HorizSecond3[9] = {1, 2, 1, -2, -4, -2, 1, 2, 1};
  static const int HorizSecond5[25] = {1, 4, 6,  4,  1, 0, 0, 0,  0,  0, -2, -8, -12, -8, -2,
                                       0, 0, 0,  0,  0, 1, 4, 6,  4,  1};
  static const int VertSecond3[9] = {1, -2, 1, 2, -4, 2, 1, -2, 1};
  static const int VertSecond5[25] = {1, 0, -2,  0, 1, 4, 0, -8,  0, 4, 6, 0, -12, 0, 6,
                                      4, 0, -8,  0, 4, 1, 0, -2,  0, 1};
  const bool five = maskSize == 5;
  switch (op) {
    case kPrewittHoriz: return mk3(PrewittHoriz3);
    case kPrewittVert: return mk3(PrewittVert3);
    case kRobertsDown: return mk3(RobertsDown3);
    case kRobertsUp: return mk3(RobertsUp3);
    case kSobelHoriz: return five ? mk5(SobelHoriz5) : mk3(SobelHoriz3);
    case kSobelVert: return five ? mk5(SobelVert5) : mk3(SobelVert3);
    case kScharrHoriz: return mk3(ScharrHoriz3);
    case kScharrVert: return mk3(ScharrVert3);
    case kSobelCross: return five ? mk5(Cross5) : mk3(Cross3);
    case kSobelHorizSecond: return five ? mk5(HorizSecond5) : mk3(HorizSecond3);
    case kSobelVertSecond: return five ? mk5(VertSecond5) : mk3(VertSecond3);
  }
  return mk3(SobelHoriz3);
}

template <typename Dst> inline Dst emitFloat(float v) {
  if constexpr (std::is_same_v<Dst, Npp8u>) {
    return v <= 0.0f ? (Npp8u)0 : (v >= 255.0f ? (Npp8u)255 : (Npp8u)v);
  } else if constexpr (std::is_same_v<Dst, Npp16s>) {
    return v <= -32768.0f ? (Npp16s)-32768 : (v >= 32767.0f ? (Npp16s)32767 : (Npp16s)v);
  } else {
    return (Dst)v;
  }
}

// HasMask: the entry point takes an NppiMaskSize argument (3x3 and 5x5 are
// swept); otherwise the family is 3x3 only.
template <typename Src, typename Dst, int C, int A, int Op, bool HasMask, bool Border, class Call>
void run_deriv(const char *id, Call call) {
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
  static const Cfg cfgs[] = {{5, 4}, {8, 6}, {33, 25}, {96, 64}};
  const int masks[] = {(int)NPP_MASK_SIZE_3_X_3, (int)NPP_MASK_SIZE_5_X_5};
  const int nMask = HasMask ? 2 : 1;
  const int m = 2;
  bool failed = false;
  for (const Cfg &cf : cfgs) {
    for (int mi = 0; mi < nMask && !failed; ++mi) {
      const int maskSize = masks[mi] == (int)NPP_MASK_SIZE_5_X_5 ? 5 : 3;
      const Coef coef = coefOf(Op, maskSize);
      const int r = coef.size / 2;
      const int W = cf.w + 2 * m, H = cf.h + 2 * m;
      const uint64_t seed = case_seed(id, cf.w, cf.h, maskSize, Border ? 1 : 0, 0, 0);
      Buf<Src> *s = new Buf<Src>(W, H, C, 0, 0);
      fill_regime(*s, C, seed, kWide, 0, false, false, false);
      s->dev.copyFromHost(s->host);
      Buf<Dst> *d = new Buf<Dst>(cf.w, cf.h, C, 0, 0x5A);
      d->dev.copyFromHost(d->host);
      const std::vector<uint8_t> pre = d->host;
      const int srow = s->step / (int)sizeof(Src);
      const Src *sp = reinterpret_cast<const Src *>(s->dptr() + (size_t)m * s->step +
                                                    (size_t)m * C * sizeof(Src));
      npp_diff::clear_cuda_error();
      const NppStatus st = call(sp, s->step, reinterpret_cast<Dst *>(d->dptr()), d->step, NppiSize{cf.w, cf.h},
                                NppiSize{W, H}, NppiPoint{m, m}, (int)masks[mi], ctx);
      if (st != NPP_NO_ERROR) {
        ++cnt.skipped;
        if (st == NPP_NOT_IMPLEMENTED_ERROR) ++cnt.notimpl;
        delete s;
        delete d;
        continue;
      }
      ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess) << "[" << id << "] sync";
      ++cnt.executed;
      std::vector<uint8_t> want = pre;
      for (int y = 0; y < cf.h; ++y)
        for (int x = 0; x < cf.w; ++x)
          for (int c = 0; c < A; ++c) {
            int iacc = 0;
            float facc = 0.0f;
            for (int a = 0; a < coef.size; ++a) {
              int yy = m + y + a - r;
              if constexpr (Border) yy = yy < 0 ? 0 : (yy > H - 1 ? H - 1 : yy);
              for (int b = 0; b < coef.size; ++b) {
                int xx = m + x + b - r;
                if constexpr (Border) xx = xx < 0 ? 0 : (xx > W - 1 ? W - 1 : xx);
                const Src v = s->hptr()[(size_t)yy * srow + (size_t)xx * C + c];
                const int kv = coef.k[a][b];
                if constexpr (std::is_floating_point_v<Src>)
                  facc += (float)kv * (float)v;
                else
                  iacc += kv * (int)v;
              }
            }
            const float fv = std::is_floating_point_v<Src> ? facc : (float)iacc;
            const Dst out = emitFloat<Dst>(fv);
            const size_t off = (size_t)y * d->step + ((size_t)x * C + (size_t)c) * sizeof(Dst);
            std::memcpy(want.data() + off, &out, sizeof(Dst));
          }
      d->dev.copyToHost(d->host);
      const GenCase dc{cf.w, cf.h, 0, 0, 0, kSmall, 0};
      const bool ok = rotate_check<Dst, C, A>(id, dc, NPPI_INTER_NN, pre, d->host, want, d->step, {}, cnt,
                                              first);
      delete s;
      delete d;
      if (!ok || cnt.mismatched >= 8) failed = true;
    }
    if (failed) break;
  }
  finish(id, cnt, first);
}

}  // namespace filtderiv
}  // namespace num
