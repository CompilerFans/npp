// CPU-golden coverage for the general filter families
// (nppi_filtering_functions.h):
//   nppiFilter / nppiFilterBorder                 (32-bit integer kernel)
//   nppiFilter32f / nppiFilterBorder32f           (32f kernel)
//   nppiFilter{Row,Column} / ...Border            (1D integer kernel)
//   nppiFilter{Row,Column}32f / ...Border32f      (1D 32f kernel)
//   nppiFilter_64f_C1R, Row/Column_64f_C1R        (double kernel)
//
// Measured NPP 13.4 semantics (RTX 5090):
//   dst(x,y)[c] = sum_{j,i} src(x + ax - i, y + ay - j)[c] * k[j*kw + i]
//   (row: src(x + ax - i, y); column: src(x, y + ay - i)).  Tap coordinates
//   are clamped to the source image by the border variants (REPLICATE only)
//   and read raw by the plain variants (the driver allocates a margin).
//   Integer kernels accumulate in int32 ring arithmetic, divide by nDivisor
//   with C truncation and saturate to the destination range.  The 8u C4/AC4
//   row/column path additionally uses unsigned ring arithmetic and unsigned
//   division (a negative sum wraps high and saturates to 255).  Float kernels
//   accumulate in float32; integer destinations truncate and saturate, 16f
//   destinations round to half, 64f accumulates in double.
//   AC4 filters channels 0..2 and leaves the destination alpha untouched.
//
// Kernel arrays are uploaded to device memory (NPP 13.4 reads them on the
// device; the in-tree executor accepts either through cudaMemcpyDefault).
#pragma once

#include "num_nppi_geom.h"

namespace num {
namespace filtgen {

inline int clampi(int v, int lo, int hi) { return v < lo ? lo : (v > hi ? hi : v); }

template <typename Dst> inline Dst satI32(int v) {
  if constexpr (std::is_same_v<Dst, Npp8u>) {
    return (Npp8u)(v <= 0 ? 0 : (v >= 255 ? 255 : v));
  } else if constexpr (std::is_same_v<Dst, Npp16u>) {
    return (Npp16u)(v <= 0 ? 0 : (v >= 65535 ? 65535 : v));
  } else if constexpr (std::is_same_v<Dst, Npp16s>) {
    return (Npp16s)(v <= -32768 ? -32768 : (v >= 32767 ? 32767 : v));
  } else if constexpr (std::is_same_v<Dst, Npp8s>) {
    return (Npp8s)(v <= -128 ? -128 : (v >= 127 ? 127 : v));
  } else if constexpr (std::is_same_v<Dst, Npp32s>) {
    return (Npp32s)v;
  } else {
    return (Dst)v;
  }
}

template <typename Dst> inline Dst emitFloat(float acc) {
  if constexpr (std::is_same_v<Dst, Npp32f>) {
    return acc;
  } else if constexpr (std::is_same_v<Dst, Npp64f>) {
    return (Npp64f)acc;
  } else if constexpr (std::is_same_v<Dst, Npp16f>) {
    return half_round(acc);
  } else {
    const float t = std::trunc(acc);
    if constexpr (std::is_same_v<Dst, Npp32s>) {
      if (t >= 2147483648.0f) return 2147483647;
      if (t <= -2147483648.0f) return -2147483647 - 1;
      return (Npp32s)t;
    } else if constexpr (std::is_same_v<Dst, Npp16s>) {
      return (Npp16s)(t < -32768.0f ? -32768 : (t > 32767.0f ? 32767 : t));
    } else if constexpr (std::is_same_v<Dst, Npp16u>) {
      return (Npp16u)(t < 0.0f ? 0 : (t > 65535.0f ? 65535 : t));
    } else if constexpr (std::is_same_v<Dst, Npp8s>) {
      return (Npp8s)(t < -128.0f ? -128 : (t > 127.0f ? 127 : t));
    } else {  // Npp8u
      return (Npp8u)(t < 0.0f ? 0 : (t > 255.0f ? 255 : t));
    }
  }
}

// G: 0 = 2D, 1 = row, 2 = column.
// K: 0 = int32 kernel + divisor, 1 = Npp32f kernel, 2 = Npp64f kernel.
// Clamp: the measured plain-32f-C1 path that clamps taps into the ROI.
template <typename Src, typename Dst, int C, int A, int G, int K, bool Border, bool Clamp, class Call>
void run_generic(const char *id, Call call) {
  using KT = std::conditional_t<K == 0, Npp32s, std::conditional_t<K == 1, Npp32f, Npp64f>>;
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
  static const Cfg cfgs[] = {{6, 6}, {8, 6}, {33, 25}, {96, 64}};
  static const int masks2d[][2] = {{3, 3}, {5, 3}, {3, 5}, {1, 1}};
  static const int masks1d[] = {3, 5, 7};
  const int m = G == 0 ? 4 : 8;
  bool failed = false;
  for (const Cfg &cf : cfgs) {
    const int nGeo = G == 0 ? 4 : 3;
    for (int gi = 0; gi < nGeo && !failed; ++gi) {
      const int kw = G == 0 ? masks2d[gi][0] : masks1d[gi];
      const int kh = G == 0 ? masks2d[gi][1] : 1;
      const int maskSize = G == 0 ? 0 : masks1d[gi];
      const int nTaps = G == 0 ? kw * kh : maskSize;
      const int centerA[2] = {G == 2 ? 0 : (G == 0 ? (kw - 1) / 2 : maskSize / 2),
                              G == 1 ? 0 : (G == 0 ? (kh - 1) / 2 : maskSize / 2)};
      // Measured: NPP's 1D kernels mishandle non-centred anchors for some
      // layouts (the 8u C3 row kernel maps anchor 0 to shifted taps and even
      // duplicate responses), so 1D value cases use the centred anchor only -
      // the same geometry as the differential value records.  2D kernels
      // agree for both centred and zero anchors.
      const int nAnchor = (G != 0) ? 1 : ((kw == 1 && kh == 1) ? 1 : 2);
      const int divisors[2] = {1, 3};
      const int nDiv = K == 0 ? 2 : 1;
      for (int ai = 0; ai < nAnchor && !failed; ++ai) {
        const int ax = ai == 0 ? centerA[0] : 0;
        const int ay = ai == 0 ? centerA[1] : 0;
        for (int di = 0; di < nDiv && !failed; ++di) {
          const int divisor = divisors[di];
          std::vector<KT> khost((size_t)nTaps);
          static const float kf[7] = {0.11f, -0.23f, 0.37f, -0.41f, 0.53f, -0.61f, 0.71f};
          for (int t = 0; t < nTaps; ++t) {
            if constexpr (K == 0) khost[(size_t)t] = (KT)((t * 7 + 3) % 11 - 5);
            else if constexpr (K == 1) khost[(size_t)t] = kf[t % 7];
            else khost[(size_t)t] = (KT)kf[t % 7];
          }
          npp_functional_test::DeviceMemory<KT> kdev((size_t)nTaps);
          kdev.copyFromHost(khost);
          const int W = cf.w + 2 * m, H = cf.h + 2 * m;
          const uint64_t seed = case_seed(id, cf.w, cf.h, kw * 16 + kh, ai * 8 + gi + 32 * di,
                                          Border ? 1 : 0, 0);
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
          const NppStatus st = call(sp, s->step, reinterpret_cast<Dst *>(d->dptr()), d->step,
                                    NppiSize{cf.w, cf.h}, NppiSize{W, H}, NppiPoint{m, m},
                                    kdev.get(), NppiSize{kw, kh}, NppiPoint{ax, ay}, maskSize,
                                    divisor, Border ? (int)NPP_BORDER_REPLICATE : 0, ctx);
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
                long long iacc = 0;
                unsigned uacc = 0;
                float facc = 0.0f;
                double dacc = 0.0;
                for (int t = 0; t < nTaps; ++t) {
                  int dx = 0, dy = 0;
                  if (G == 0) {
                    dx = ax - t % kw;
                    dy = ay - t / kw;
                  } else if (G == 1) {
                    dx = ax - t;
                  } else {
                    dy = ay - t;
                  }
                  int px, py;
                  const bool useClamp =
                      Clamp && ((G == 0) ? (kw == kh && kw >= 3 && (kw & 1) != 0)
                                         : (maskSize >= 3 && (maskSize & 1) != 0));
                  if (useClamp) {
                    // NPP's plain 32f C1R path clamps every tap into the ROI
                    // for odd square kernels / odd masks of size >= 3.
                    int rx = x + dx, ry = y + dy;
                    if (G != 2) rx = clampi(rx, 0, cf.w - 1);
                    if (G != 1) ry = clampi(ry, 0, cf.h - 1);
                    px = m + rx;
                    py = m + ry;
                  } else {
                    px = m + x + dx;
                    py = m + y + dy;
                    if constexpr (Border) {
                      if (G != 2) px = clampi(px, 0, W - 1);
                      if (G != 1) py = clampi(py, 0, H - 1);
                    }
                  }
                  const double v = as_double(s->hptr()[(size_t)py * srow + (size_t)px * C + c]);
                  if constexpr (K == 0) {
                    const long long prod = (long long)khost[(size_t)t] * (long long)(int)v;
                    iacc += prod;
                    const unsigned pu = (unsigned)(unsigned long long)prod;
                    uacc = pu > 0xFFFFFFFFu - uacc ? 0xFFFFFFFFu : uacc + pu;
                  } else if constexpr (K == 1) {
                    facc += (float)v * (float)khost[(size_t)t];
                  } else {
                    // 64f sources are also converted through float first
                    // (to_float is a static_cast<float> in the executor).
                    dacc += (double)(float)v * (double)khost[(size_t)t];
                  }
                }
                Dst out;
                if constexpr (K == 0) {
                  constexpr bool kUnsignedU8x4 = std::is_same_v<Src, Npp8u> && std::is_same_v<Dst, Npp8u> && C == 4 && G != 0;
                  if constexpr (kUnsignedU8x4) {
                    // Measured: the reference accumulates these 1D unsigned
                    // sums with 32-bit SATURATING adds (not ring wrap): eight
                    // taps of 2^29 divided by 2^31-1 emit 2, and a mixed-sign
                    // window whose wrapped sum is 100 still emits 255.
                    const unsigned q = uacc / (unsigned)divisor;
                    out = (Npp8u)(q > 255u ? 255u : q);
                  } else {
                    const int wrapped = (int)(unsigned)iacc;
                    const int q = (wrapped == INT32_MIN && divisor == -1) ? INT32_MIN : wrapped / divisor;
                    out = satI32<Dst>(q);
                  }
                } else if constexpr (K == 1) {
                  out = emitFloat<Dst>(facc);
                } else {
                  out = (Dst)dacc;
                }
                const size_t off = (size_t)y * d->step + ((size_t)x * C + (size_t)c) * sizeof(Dst);
                std::memcpy(want.data() + off, &out, sizeof(Dst));
              }
          d->dev.copyToHost(d->host);
          const GenCase dc{cf.w, cf.h, kw * 16 + kh, 0, 0, kSmall, divisor};
          const bool ok = rotate_check<Dst, C, A>(id, dc, NPPI_INTER_NN, pre, d->host, want, d->step, {},
                                                  cnt, first);
          delete s;
          delete d;
          if (!ok || cnt.mismatched >= 8) failed = true;
        }
      }
    }
  }
  finish(id, cnt, first);
}

}  // namespace filtgen
}  // namespace num
