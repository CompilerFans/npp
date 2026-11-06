// CPU-golden coverage for nppiFilterUnsharpBorder
// (nppi_filtering_functions.h).
//
// Measured NPP 13.4 semantics (RTX 5090, probe-verified):
//   * taps: normalised 2-D Gaussian exp(-(dy^2+dx^2)/(2 sigma^2)) / sum,
//     evaluated in double and stored as float; k = 2*ceil(radius)+1 for an
//     explicit radius (auto radius is not exercised here).
//   * blur: double accumulation of tap * source over the k x k window; source
//     reads clamp to [-offX, ROI-1-offX] x [-offY, ROI-1-offY] relative to
//     pSrc (the API assumes the image is exactly the size of the ROI).
//   * dst = src + weight * (src - blurred); with nThreshold > 0 the source is
//     copied when |src - blurred| <= nThreshold * scale, where scale is
//     128 / 32768 / 16384 / 2^32 for 8u / 16u / 16s / 32f.
//   * integer destinations truncate toward zero and saturate; AC4 leaves the
//     destination alpha untouched.
//   * pDeviceBuffer must hold k*k*4 bytes (the taps are written there; any
//     other size makes the reference fail the transfer).
#pragma once

#include "num_nppi_geom.h"

namespace num {
namespace filtunsharp {

inline int clampi(int v, int lo, int hi) { return v < lo ? lo : (v > hi ? hi : v); }

inline double tailEps(int dtype) {
  switch (dtype) {
    case 0: return 1.1813e-2;  // 8u
    case 1: return 5.9274e-5;  // 16u
    case 2: return 1.1490e-4;  // 16s
    default: return 1.9930e-12;  // 32f
  }
}

inline int autoRadius(double sigma, double eps) {
  const double s2 = 2.0 * sigma * sigma;
  double total = 1.0;
  for (int i = 1; i <= 4096 + 64; ++i) {
    const double v = std::exp(-(double)i * i / s2);
    total += 2.0 * v;
    if (v < 1e-18 && i > 64) break;
  }
  double window = 1.0;
  for (int r = 1; r <= 4096; ++r) {
    window += 2.0 * std::exp(-(double)r * r / s2);
    const double ratio = window / total;
    if (1.0 - ratio * ratio <= eps) return r;
  }
  return 4096;
}

inline int kernelRadius(float radius, float sigma, int dtype) {
  if (radius > 0.0f) return (int)std::ceil((double)radius);
  return autoRadius((double)sigma, tailEps(dtype));
}

inline void buildTaps(std::vector<float> &taps, int r, float sigma) {
  const int k = 2 * r + 1;
  taps.resize((size_t)k * (size_t)k);
  const double s2 = 2.0 * (double)sigma * (double)sigma;
  double sum = 0.0;
  for (int a = 0; a < k; ++a)
    for (int b = 0; b < k; ++b) {
      const double dy = a - r, dx = b - r;
      const double v = std::exp(-(dy * dy + dx * dx) / s2);
      taps[(size_t)a * (size_t)k + (size_t)b] = (float)v;
      sum += v;
    }
  const double inv = 1.0 / sum;
  for (auto &t : taps) t = (float)((double)t * inv);
}

template <typename Dst> inline Dst unsharpEmit(double v) {
  if constexpr (std::is_same_v<Dst, Npp32f>) {
    return (Npp32f)v;
  } else if constexpr (std::is_same_v<Dst, Npp8u>) {
    return (Npp8u)(v <= 0.0 ? 0 : (v >= 255.0 ? 255 : (Npp8u)(int)v));
  } else if constexpr (std::is_same_v<Dst, Npp16u>) {
    return (Npp16u)(v <= 0.0 ? 0 : (v >= 65535.0 ? 65535 : (Npp16u)(int)v));
  } else {
    return (Npp16s)(v <= -32768.0 ? -32768 : (v >= 32767.0 ? 32767 : (Npp16s)(int)v));
  }
}

template <typename Src, typename Dst, int C, int A, int DType, class Call>
void run_unsharp(const char *id, Call call) {
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
  static const float radii[] = {1.0f, 2.0f};
  const float sigma = 1.0f, weight = 1.5f;
  const double scale = DType == 0 ? 128.0 : DType == 1 ? 32768.0 : DType == 2 ? 16384.0 : 4294967296.0;
  // The reference taps differ from the double-precision model by far less
  // than one LSB, but a threshold placed inside the data's |src-blurred|
  // distribution flips the copy branch on those tiny differences.  The copy
  // subcase therefore uses a threshold above the whole dynamic range, which
  // pins the branch itself; the sharpening path (threshold 0) pins the blur.
  const float copyThreshold = DType == 0 ? 2.0f : DType == 1 ? 2.0f : DType == 2 ? 4.0f : 1e-6f;
  const float thresholds[2] = {0.0f, copyThreshold};
  const int m = 1;  // clamp box is the ROI itself, so any margin works
  bool failed = false;
  for (const Cfg &cf : cfgs) {
    for (int ri = 0; ri < 2 && !failed; ++ri) {
      for (int ti = 0; ti < 2 && !failed; ++ti) {
        const float radius = radii[ri];
        const float threshold = thresholds[ti];
        const int k = 2 * kernelRadius(radius, sigma, DType) + 1;
        std::vector<float> taps;
        buildTaps(taps, k / 2, sigma);
        const int W = cf.w + 2 * m, H = cf.h + 2 * m;
        const uint64_t seed = case_seed(id, cf.w, cf.h, k, ti, 0, 0);
        Buf<Src> *s = new Buf<Src>(W, H, C, 0, 0);
        fill_regime(*s, C, seed, kWide, 0, false, false, false);
        s->dev.copyFromHost(s->host);
        Buf<Dst> *d = new Buf<Dst>(cf.w, cf.h, C, 0, 0x5A);
        d->dev.copyFromHost(d->host);
        npp_functional_test::DeviceMemory<Npp8u> pbuf((size_t)taps.size() * sizeof(float));
        const std::vector<uint8_t> pre = d->host;
        const int srow = s->step / (int)sizeof(Src);
        const Src *sp = reinterpret_cast<const Src *>(s->dptr() + (size_t)m * s->step +
                                                      (size_t)m * C * sizeof(Src));
        npp_diff::clear_cuda_error();
        const NppStatus st = call(sp, s->step, NppiPoint{m, m}, reinterpret_cast<Dst *>(d->dptr()),
                                  d->step, NppiSize{cf.w, cf.h}, radius, sigma, weight, threshold,
                                  (int)NPP_BORDER_REPLICATE, pbuf.get(), ctx);
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
        const int r = k / 2;
        const int loX = -m, hiX = cf.w - 1 - m, loY = -m, hiY = cf.h - 1 - m;
        auto load = [&](int xx, int yy, int c) -> double {
          return as_double(s->hptr()[(size_t)(m + clampi(yy, loY, hiY)) * srow +
                                     (size_t)(m + clampi(xx, loX, hiX)) * C + c]);
        };
        for (int y = 0; y < cf.h; ++y)
          for (int x = 0; x < cf.w; ++x) {
            double acc[4] = {0, 0, 0, 0};
            for (int a = 0; a < k; ++a)
              for (int b = 0; b < k; ++b) {
                const float t = taps[(size_t)a * (size_t)k + (size_t)b];
                if (t == 0.0f) continue;
                for (int c = 0; c < A; ++c) acc[c] += (double)t * load(x + b - r, y + a - r, c);
              }
            const bool useThreshold = threshold > 0.0f;
            for (int c = 0; c < A; ++c) {
              const double sv = load(x, y, c);
              const double hp = sv - acc[c];
              const double out = (useThreshold && std::fabs(hp) <= (double)threshold * scale)
                                     ? sv
                                     : sv + (double)weight * hp;
              const Dst o = unsharpEmit<Dst>(out);
              const size_t off = (size_t)y * d->step + ((size_t)x * C + (size_t)c) * sizeof(Dst);
              std::memcpy(want.data() + off, &o, sizeof(Dst));
            }
          }
        d->dev.copyToHost(d->host);
        const GenCase dc{cf.w, cf.h, k, 0, 0, kSmall, ti};
        const bool ok = rotate_check<Dst, C, A>(id, dc, NPPI_INTER_NN, pre, d->host, want, d->step, {},
                                                cnt, first);
        delete s;
        delete d;
        if (!ok || cnt.mismatched >= 8) failed = true;
      }
    }
  }
  finish(id, cnt, first);
}

}  // namespace filtunsharp
}  // namespace num
