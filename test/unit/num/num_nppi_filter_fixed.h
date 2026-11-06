// CPU-golden coverage for the fixed-kernel NPP filters
// (nppi_filtering_functions.h):
//   nppiFilter{HighPass,LowPass,Sharpen,Laplace} and their *Border variants.
//
// Masks measured on NVIDIA NPP 13.4 (RTX 5090):
//   HighPass 3x3: 9*center - sum(3x3)   5x5: 25*center - sum(5x5)
//   LowPass  3x3: sum(3x3)/9            5x5: sum(5x5)/25
//   Sharpen  3x3: (16*center - sum(8 neighbours))/8
//   Laplace  3x3: [[-1,-1,-1],[-1,8,-1],[-1,-1,-1]]
//            5x5: [[-1,-3,-4,-3,-1],[-3,0,6,0,-3],[-4,6,20,6,-4],
//                  [-3,0,6,0,-3],[-1,-3,-4,-3,-1]]
// Integer destinations emit trunc((sum + bias) / denom) with C truncation
// (bias 4/12 for the LowPass boxes, 3 for Sharpen, 0 elsewhere) and saturate
// to the destination range; 32f stores the float accumulation unchanged.
// AC4 filters channels 0..2 and leaves alpha untouched.
//
// Plain variants read raw neighbour pixels around pSrc (the caller provides
// the margin); Border variants accept NPP_BORDER_REPLICATE only and clamp the
// read coordinates to the oSrcSize image centred on pSrc, whose origin sits at
// oSrcOffset inside that image.  The driver allocates a two-pixel margin and
// places pSrc at (2,2) of the image so both paths read defined values.
#pragma once

#include "num_nppi_geom.h"

namespace num {
namespace filtfixed {

enum Op { kHighPass = 0, kLowPass, kSharpen, kLaplace };

struct Coef {
  int size;
  int k[5][5];
  int bias;
  int denom;
};

inline Coef coefOf(int op, int maskSize) {
  Coef c{};
  const int size = op == kSharpen ? 3 : maskSize;
  c.size = size;
  c.bias = 0;
  c.denom = 1;
  const int r = size / 2;
  for (int a = 0; a < size; ++a)
    for (int b = 0; b < size; ++b) c.k[a][b] = -1;
  if (op == kHighPass) {
    c.k[r][r] = size * size - 1;
  } else if (op == kLowPass) {
    c.bias = size == 5 ? 12 : 4;
    c.denom = size * size;
    for (int a = 0; a < size; ++a)
      for (int b = 0; b < size; ++b) c.k[a][b] = 1;
  } else if (op == kSharpen) {
    c.k[1][1] = 16;
    c.bias = 3;
    c.denom = 8;
  } else {  // Laplace
    if (size == 3) {
      for (int a = 0; a < 3; ++a)
        for (int b = 0; b < 3; ++b) c.k[a][b] = -1;
      c.k[1][1] = 8;
    } else {
      static const int L5[25] = {-1, -3, -4, -3, -1, -3, 0, 6,  0,  -3, -4, 6,  20, 6,  -4,
                                 -3, 0,  6,  0,  -3, -1, -3, -4, -3, -1};
      for (int a = 0; a < 5; ++a)
        for (int b = 0; b < 5; ++b) c.k[a][b] = L5[a * 5 + b];
    }
  }
  return c;
}

template <typename Dst> inline Dst satInt(long long v) {
  if constexpr (std::is_same_v<Dst, Npp8u>) {
    return (Npp8u)(v <= 0 ? 0 : (v >= 255 ? 255 : v));
  } else if constexpr (std::is_same_v<Dst, Npp16u>) {
    return (Npp16u)(v <= 0 ? 0 : (v >= 65535 ? 65535 : v));
  } else {
    return (Dst)(v <= -32768 ? -32768 : (v >= 32767 ? 32767 : v));
  }
}

// Src/Dst element types (equal except for the Laplace 8u16s/8s16s forms);
// C channels in memory, the first A written by the filter.
template <typename Src, typename Dst, int C, int A, int Op, bool Border, class Call>
void run_fixed(const char *id, Call call) {
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
  const int nMask = Op == kSharpen ? 1 : 2;
  const int m = 2;  // margin large enough for the 5x5 taps
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
      const NppStatus st = call(sp, s->step, reinterpret_cast<Dst *>(d->dptr()), d->step, NppiSize{cf.w, cf.h}, NppiSize{W, H},
                                NppiPoint{m, m}, (int)masks[mi], ctx);
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
                if constexpr (std::is_floating_point_v<Dst>)
                  facc += (float)v * (float)kv;
                else
                  iacc += (int)v * kv;
              }
            }
            Dst out;
            if constexpr (std::is_floating_point_v<Dst>) {
              // The float path divides by the denominator but never adds the
              // integer rounding bias (measured on NPP 13.4).
              out = (Dst)(facc / (float)coef.denom);
            } else {
              out = satInt<Dst>(((long long)iacc + coef.bias) / coef.denom);
            }
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

}  // namespace filtfixed
}  // namespace num
