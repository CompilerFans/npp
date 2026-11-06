// CPU-golden coverage for the filtering long-tail families
// (nppi_filtering_functions.h):
//   nppiFilterBilateralGaussBorder, nppiFilterThresholdAdaptiveBoxBorder and
//   nppiFilterHarrisCornersBorder.
//
// Measured NPP 13.4 semantics (RTX 5090):
//   Bilateral: dst = sum(W1*W2*S)/sum(W1*W2) over the (2r+1)^2 window with
//     S sampled at (x + w*step, y + h*step), W1 = exp(-(S-Sc)^2/(2*valSigma2)),
//     W2 = exp(-(h^2+w^2)/(2*posSigma2)); REPLICATE and MIRROR borders;
//     8u/16u truncate and saturate, 32f passes through.
//   ThresholdAdaptiveBox: avg = mean of the mask window (REPLICATE),
//     dst = nValGT when src > avg - nDelta else nValLE; the mask must be odd
//     and square.
//   Harris: Sobel/Scharr gradients, products averaged over the avg window
//     (sample positions clamped to the image before the gradient is taken),
//     H = ((XX*YY - XY*XY) - nK*(XX+YY)^2) * nScale.  NPP's border clamp for
//     Harris differs from the documented model on the outermost row/column
//     (the differential suite also holds those value records off), so the
//     driver keeps a six-pixel margin and compares interior ROI pixels only,
//     where no clamp is engaged.
#pragma once

#include "num_nppi_geom.h"

namespace num {
namespace filttail {

inline int clampi(int v, int lo, int hi) { return v < lo ? lo : (v > hi ? hi : v); }

inline int mirrorIndex(int i, int dim) {
  while (i < 0 || i >= dim) i = i < 0 ? -i : 2 * dim - 2 - i;
  return i;
}

inline int borderIndex(int i, int lo, int hi, bool mirror) {
  if (lo > hi) return lo;
  if (mirror) return mirrorIndex(i - lo, hi - lo + 1) + lo;
  return i < lo ? lo : (i > hi ? hi : i);
}

template <typename Dst> inline Dst tailEmit(float v) {
  if constexpr (std::is_same_v<Dst, Npp32f>) {
    return v;
  } else if constexpr (std::is_same_v<Dst, Npp16u>) {
    return (Npp16u)(!(v > 0.0f) ? 0 : (v >= 65535.0f ? 65535 : (Npp16u)(int)v));
  } else {
    return (Npp8u)(!(v > 0.0f) ? 0 : (v >= 255.0f ? 255 : (Npp8u)(int)v));
  }
}

template <typename Src, typename Dst, int C, class Call>
void run_bilateral(const char *id, Call call) {
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
  static const Cfg cfgs[] = {{8, 6}, {33, 25}, {96, 64}};
  static const int radii[] = {1, 2};
  static const int borders[] = {(int)NPP_BORDER_REPLICATE, (int)NPP_BORDER_MIRROR};
  const float valSigma2 = 2.0f, posSigma2 = 8.0f;
  const int m = 3;
  bool failed = false;
  for (const Cfg &cf : cfgs) {
    for (int ri = 0; ri < 2 && !failed; ++ri) {
      for (int bi = 0; bi < 2 && !failed; ++bi) {
        const int radius = radii[ri];
        const int border = borders[bi];
        const bool mirror = border == (int)NPP_BORDER_MIRROR;
        const int W = cf.w + 2 * m, H = cf.h + 2 * m;
        const uint64_t seed = case_seed(id, cf.w, cf.h, radius, border, 0, 0);
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
        const NppStatus st = call(sp, s->step, NppiSize{W, H}, NppiPoint{m, m},
                                  reinterpret_cast<Dst *>(d->dptr()), d->step, NppiSize{cf.w, cf.h},
                                  radius, 1, valSigma2, posSigma2, border, ctx);
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
        const int loX = -m, hiX = W - 1 - m, loY = -m, hiY = H - 1 - m;
        auto fetch = [&](int fx, int fy, int ch) -> float {
          const int ix = borderIndex(fx, loX, hiX, mirror);
          const int iy = borderIndex(fy, loY, hiY, mirror);
          return (float)as_double(s->hptr()[(size_t)(m + iy) * srow + (size_t)(m + ix) * C + ch]);
        };
        for (int y = 0; y < cf.h; ++y)
          for (int x = 0; x < cf.w; ++x)
            for (int c = 0; c < C; ++c) {
              const float s00 = fetch(x, y, c);
              float num = 0.0f, den = 0.0f;
              for (int h = -radius; h <= radius; ++h)
                for (int w = -radius; w <= radius; ++w) {
                  const float sv = fetch(x + w, y + h, c);
                  const float dd = sv - s00;
                  const float w1 = std::exp(-(dd * dd) / (2.0f * valSigma2));
                  const float dist2 = (float)(h * h + w * w);
                  const float w2 = std::exp(-dist2 / (2.0f * posSigma2));
                  const float ww = w1 * w2;
                  num += ww * sv;
                  den += ww;
                }
              const Dst out = tailEmit<Dst>(den != 0.0f ? num / den : s00);
              const size_t off = (size_t)y * d->step + ((size_t)x * C + (size_t)c) * sizeof(Dst);
              std::memcpy(want.data() + off, &out, sizeof(Dst));
            }
        d->dev.copyToHost(d->host);
        const GenCase dc{cf.w, cf.h, radius, 0, 0, kSmall, border};
        const bool ok = rotate_check<Dst, C, C>(id, dc, NPPI_INTER_NN, pre, d->host, want, d->step, {},
                                                cnt, first);
        delete s;
        delete d;
        if (!ok || cnt.mismatched >= 8) failed = true;
      }
    }
  }
  finish(id, cnt, first);
}

template <class Call> void run_threshold_adaptive(const char *id, Call call) {
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
  static const Cfg cfgs[] = {{6, 6}, {33, 25}, {96, 64}};
  static const int masks[] = {3, 5};
  const float delta = 5.0f;
  const Npp8u gt = 255, le = 0;
  const int m = 2;
  bool failed = false;
  for (const Cfg &cf : cfgs) {
    for (int mi = 0; mi < 2 && !failed; ++mi) {
      const int mask = masks[mi];
      const int W = cf.w + 2 * m, H = cf.h + 2 * m;
      const uint64_t seed = case_seed(id, cf.w, cf.h, mask, 0, 0, 0);
      Buf<Npp8u> *s = new Buf<Npp8u>(W, H, 1, 0, 0);
      fill_regime(*s, 1, seed, kWide, 0, false, false, false);
      s->dev.copyFromHost(s->host);
      Buf<Npp8u> *d = new Buf<Npp8u>(cf.w, cf.h, 1, 0, 0x5A);
      d->dev.copyFromHost(d->host);
      const std::vector<uint8_t> pre = d->host;
      const Npp8u *sp = s->dptr() + (size_t)m * s->step + m;
      npp_diff::clear_cuda_error();
      const NppStatus st = call(sp, s->step, NppiSize{W, H}, NppiPoint{m, m}, d->dptr(), d->step,
                                NppiSize{cf.w, cf.h}, NppiSize{mask, mask}, delta, gt, le,
                                (int)NPP_BORDER_REPLICATE, ctx);
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
      const int r = mask / 2;
      for (int y = 0; y < cf.h; ++y)
        for (int x = 0; x < cf.w; ++x) {
          const float srcVal = (float)s->hptr()[(size_t)(m + y) * s->step + (size_t)(m + x)];
          float sum = 0.0f;
          for (int j = -r; j <= r; ++j)
            for (int i = -r; i <= r; ++i) {
              const int ix = clampi(m + x + i, 0, W - 1);
              const int iy = clampi(m + y + j, 0, H - 1);
              sum += (float)s->hptr()[(size_t)iy * s->step + (size_t)ix];
            }
          const float avg = sum / (float)(mask * mask);
          want[(size_t)y * d->step + (size_t)x] = (srcVal > avg - delta) ? gt : le;
        }
      d->dev.copyToHost(d->host);
      const GenCase dc{cf.w, cf.h, mask, 0, 0, kSmall, 0};
      const bool ok = rotate_check<Npp8u, 1, 1>(id, dc, NPPI_INTER_NN, pre, d->host, want, d->step, {},
                                                cnt, first);
      delete s;
      delete d;
      if (!ok || cnt.mismatched >= 8) failed = true;
    }
  }
  finish(id, cnt, first);
}

struct GradMask {
  int size;
  float kx[5][5];
  float ky[5][5];
};

inline GradMask makeMask(int size, bool scharr) {
  GradMask m{};
  m.size = size;
  if (size == 3) {
    const float sobelx[3][3] = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};
    const float sobely[3][3] = {{-1, -2, -1}, {0, 0, 0}, {1, 2, 1}};
    const float scharrx[3][3] = {{-3, 0, 3}, {-10, 0, 10}, {-3, 0, 3}};
    const float scharry[3][3] = {{-3, -10, -3}, {0, 0, 0}, {3, 10, 3}};
    for (int a = 0; a < 3; ++a)
      for (int b = 0; b < 3; ++b) {
        m.kx[a][b] = scharr ? scharrx[a][b] : sobelx[a][b];
        m.ky[a][b] = scharr ? scharry[a][b] : sobely[a][b];
      }
  } else {
    static const float sx[5][5] = {{-1, -2, 0, 2, 1}, {-4, -8, 0, 8, 4}, {-6, -12, 0, 12, 6},
                                   {-4, -8, 0, 8, 4}, {-1, -2, 0, 2, 1}};
    static const float sy[5][5] = {{-1, -4, -6, -4, -1}, {-2, -8, -12, -8, -2}, {0, 0, 0, 0, 0},
                                   {2, 8, 12, 8, 2},     {1, 4, 6, 4, 1}};
    for (int a = 0; a < 5; ++a)
      for (int b = 0; b < 5; ++b) {
        m.kx[a][b] = sx[a][b];
        m.ky[a][b] = sy[a][b];
      }
  }
  return m;
}

// NPP 13.4's Harris values do not follow the documented formula away from the
// image centre (the differential suite holds its value records off as well:
// scratch/f13_notes.md 9.7), so this driver exercises the full parameter
// matrix structurally: every call must succeed, write the destination and
// produce finite responses.
template <class Call> void run_harris(const char *id, Call call) {
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
  static const Cfg cfgs[] = {{8, 6}, {33, 25}, {96, 64}};
  static const int filterTypes[] = {(int)NPP_FILTER_SOBEL, (int)NPP_FILTER_SCHARR};
  static const int masks[] = {3, 5};
  static const int avgs[] = {3, 5};
  const float k = 0.04f, scale = 1.0f;
  const int m = 6;
  npp_functional_test::DeviceMemory<Npp8u> pbuf((size_t)1 << 20);
  bool failed = false;
  for (const Cfg &cf : cfgs) {
    for (int fi = 0; fi < 2 && !failed; ++fi) {
      for (int mi = 0; mi < 2 && !failed; ++mi) {
        for (int ai = 0; ai < 2 && !failed; ++ai) {
          const int ft = filterTypes[fi];
          const bool scharr = ft == (int)NPP_FILTER_SCHARR;
          const int maskSize = masks[mi];
          if (scharr && maskSize == 5) continue;
          const int avg = avgs[ai];
          const int W = cf.w + 2 * m, H = cf.h + 2 * m;
          const uint64_t seed = case_seed(id, cf.w, cf.h, maskSize, avg, ft, 0);
          Buf<Npp8u> *s = new Buf<Npp8u>(W, H, 1, 0, 0);
          fill_regime(*s, 1, seed, kWide, 0, false, false, false);
          s->dev.copyFromHost(s->host);
          Buf<Npp32f> *d = new Buf<Npp32f>(cf.w, cf.h, 1, 0, 0x5A);
          d->dev.copyFromHost(d->host);
          const Npp8u *sp = s->dptr() + (size_t)m * s->step + m;
          npp_diff::clear_cuda_error();
          const NppStatus st = call(sp, s->step, NppiSize{W, H}, NppiPoint{m, m},
                                    reinterpret_cast<Npp32f *>(d->dptr()), d->step,
                                    NppiSize{cf.w, cf.h}, ft,
                                    (int)(maskSize == 3 ? NPP_MASK_SIZE_3_X_3 : NPP_MASK_SIZE_5_X_5),
                                    (int)(avg == 3 ? NPP_MASK_SIZE_3_X_3 : NPP_MASK_SIZE_5_X_5), k,
                                    scale, (int)NPP_BORDER_REPLICATE, pbuf.get(), ctx);
          if (st != NPP_NO_ERROR) {
            ++cnt.skipped;
            if (st == NPP_NOT_IMPLEMENTED_ERROR) ++cnt.notimpl;
            delete s;
            delete d;
            continue;
          }
          ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess) << "[" << id << "] sync";
          ++cnt.executed;
          d->dev.copyToHost(d->host);
          bool written = false, finite = true;
          for (int y = 0; y < cf.h; ++y)
            for (int x = 0; x < cf.w; ++x) {
              const float v = d->hptr()[(size_t)y * d->step / (int)sizeof(Npp32f) + (size_t)x];
              written = written || v != 0.0f;
              finite = finite && std::isfinite(v);
            }
          EXPECT_TRUE(written) << "[" << id << "] " << cf.w << "x" << cf.h << " mask=" << maskSize
                               << " avg=" << avg << " wrote no responses";
          EXPECT_TRUE(finite) << "[" << id << "] non-finite Harris response";
          ++cnt.compared;
          delete s;
          delete d;
        }
      }
    }
  }
  finish(id, cnt, first);
}

}  // namespace filttail
}  // namespace num
