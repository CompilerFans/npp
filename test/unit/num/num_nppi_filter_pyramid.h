// CPU-golden coverage for the Gauss pyramid filters
// (nppi_filtering_functions.h):
//   nppiFilterGaussPyramidLayer{Down,Up}Border.
//
// Measured NPP 13.4 semantics (RTX 5090):
//   Down: destination pixel j samples the source at j*rate + (t - center) for
//     every tap t (center = (taps-1)/2); non-integer positions are bilinear
//     (the four neighbours are border-mapped individually), integer rates
//     reduce to subsampling.  Kernel taps are used verbatim (no
//     renormalisation).  Border mapping is REPLICATE or MIRROR (reflect-101)
//     on the fractional tap coordinates.
//   Up: the reference's exact Gaussian spreading was not decoded (see
//     scratch/f13_notes.md); the documented model is a nearest-neighbour
//     upsample src = floor(dst/rate) followed by the user taps.  Only delta
//     kernels reproduce the reference exactly, so Gaussian up cases are
//     exercised structurally (call succeeds, output written and finite).
//   Integer destinations accumulate in float, truncate toward zero and
//     saturate; 32f passes through.  Coordinates are pSrc-relative with the
//     image spanning [-oSrcOffset, oSrcSize-1-oSrcOffset].
#pragma once

#include "num_nppi_geom.h"

namespace num {
namespace filtpyr {

inline int borderIndex(int i, int lo, int hi, bool mirror) {
  if (lo > hi) return lo;
  if (!mirror) return i < lo ? lo : (i > hi ? hi : i);
  const int dim = hi - lo + 1;
  i -= lo;
  while (i < 0 || i >= dim) i = i < 0 ? -i : 2 * dim - 2 - i;
  return i + lo;
}

template <typename Dst> inline Dst pyrEmit(float v) {
  if constexpr (std::is_same_v<Dst, Npp32f>) {
    return v;
  } else if constexpr (std::is_same_v<Dst, Npp16u>) {
    return (Npp16u)(!(v > 0.0f) ? 0 : (v >= 65535.0f ? 65535 : (Npp16u)(int)v));
  } else {
    return (Npp8u)(!(v > 0.0f) ? 0 : (v >= 255.0f ? 255 : (Npp8u)(int)v));
  }
}

// Down: value comparisons for both Gaussian and delta kernels.
// Up:   value comparisons for delta kernels only; Gaussian kernels are run
//       structurally.
template <typename Src, typename Dst, int C, bool Down, class Call>
void run_pyramid(const char *id, Call call) {
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
  struct Ker {
    int taps;
    float k[5];
    bool delta;
  };
  static const Ker kers[] = {{3, {0.25f, 0.5f, 0.25f, 0, 0}, false},
                             {5, {1.0f / 16, 4.0f / 16, 6.0f / 16, 4.0f / 16, 1.0f / 16}, false},
                             {3, {0.0f, 1.0f, 0.0f, 0, 0}, true}};
  static const int borders[] = {(int)NPP_BORDER_REPLICATE, (int)NPP_BORDER_MIRROR};
  const float rate = 2.0f;
  const int m = 8;
  bool failed = false;
  for (const Cfg &cf : cfgs) {
    for (int ki = 0; ki < 3 && !failed; ++ki) {
      for (int bi = 0; bi < 2 && !failed; ++bi) {
        const Ker &ker = kers[ki];
        const bool compareValues = Down || ker.delta;
        const int border = borders[bi];
        const bool mirror = border == (int)NPP_BORDER_MIRROR;
        const int W = cf.w + 2 * m, H = cf.h + 2 * m;
        const uint64_t seed = case_seed(id, cf.w, cf.h, ker.taps, bi, ki, 0);
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
                                  rate, ker.taps, ker.k, border, ctx);
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
        auto fetchv = [&](int ix, int iy, int c) -> float {
          ix = borderIndex(ix, loX, hiX, mirror);
          iy = borderIndex(iy, loY, hiY, mirror);
          return (float)as_double(
              s->hptr()[(size_t)(m + iy) * srow + (size_t)(m + ix) * C + c]);
        };
        auto sample2d = [&](float fx, float fy, int c) -> float {
          const float fxf = std::floor(fx), fyf = std::floor(fy);
          const int x0 = (int)fxf, y0 = (int)fyf;
          const float ax = fx - fxf, ay = fy - fyf;
          if (ax == 0.0f && ay == 0.0f) return fetchv(x0, y0, c);
          const float v00 = fetchv(x0, y0, c), v10 = fetchv(x0 + 1, y0, c);
          const float v01 = fetchv(x0, y0 + 1, c), v11 = fetchv(x0 + 1, y0 + 1, c);
          const float top = v00 + (v10 - v00) * ax;
          const float bot = v01 + (v11 - v01) * ax;
          return top + (bot - top) * ay;
        };
        const int center = (ker.taps - 1) / 2;
        const float invRate = 1.0f / rate;
        if (compareValues) {
          for (int y = 0; y < cf.h; ++y)
            for (int x = 0; x < cf.w; ++x)
              for (int c = 0; c < C; ++c) {
                float acc = 0.0f;
                for (int sy = 0; sy < ker.taps; ++sy) {
                  float hsum = 0.0f;
                  for (int sx = 0; sx < ker.taps; ++sx) {
                    if constexpr (Down) {
                      const float fy = (float)y * rate + (float)(sy - center);
                      const float fx = (float)x * rate + (float)(sx - center);
                      hsum += ker.k[sx] * sample2d(fx, fy, c);
                    } else {
                      const int yy = (int)std::floor((float)(y + sy - center) * invRate);
                      const int xx = (int)std::floor((float)(x + sx - center) * invRate);
                      hsum += ker.k[sx] * fetchv(xx, yy, c);
                    }
                  }
                  acc += ker.k[sy] * hsum;
                }
                const Dst o = pyrEmit<Dst>(acc);
                const size_t off = (size_t)y * d->step + ((size_t)x * C + (size_t)c) * sizeof(Dst);
                std::memcpy(want.data() + off, &o, sizeof(Dst));
              }
        }
        d->dev.copyToHost(d->host);
        if (compareValues) {
          const GenCase dc{cf.w, cf.h, ker.taps, 0, 0, kSmall, bi};
          const bool ok = rotate_check<Dst, C, C>(id, dc, NPPI_INTER_NN, pre, d->host, want, d->step, {},
                                                  cnt, first);
          if (!ok || cnt.mismatched >= 8) failed = true;
        } else {
          bool written = false, finite = true;
          for (int y = 0; y < cf.h; ++y)
            for (int x = 0; x < cf.w; ++x)
              for (int c = 0; c < C; ++c) {
                const size_t off = (size_t)y * d->step + ((size_t)x * C + (size_t)c) * sizeof(Dst);
                const double v = (double)*(const Dst *)(d->host.data() + off);
                const double before = (double)*(const Dst *)(pre.data() + off);
                written = written || v != before;
                finite = finite && std::isfinite(v);
              }
          EXPECT_TRUE(written) << "[" << id << "] " << cf.w << "x" << cf.h
                               << " taps=" << ker.taps << " wrote no output";
          EXPECT_TRUE(finite) << "[" << id << "] non-finite pyramid output";
        }
        delete s;
        delete d;
      }
    }
  }
  finish(id, cnt, first);
}

}  // namespace filtpyr
}  // namespace num
