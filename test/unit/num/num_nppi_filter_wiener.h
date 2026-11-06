// CPU-golden coverage for nppiFilterWienerBorder
// (nppi_filtering_functions.h).
//
// Measured NPP 13.4 semantics (RTX 5090; SASS-derived, probe-verified):
//   mean = (sum over NPP's statistics window) * 1/N, N = mw*mh;
//   var  = (sum of fma(v, v, -mean*mean) over the variance window) * 1/N,
//   where the window is the quirky loop measured on the reference:
//     * AC4 layouts use the plain rectangle ax-(mw-1)..ax by ay-(mh-1)..ay;
//     * other layouts visit mh-1 rows starting at y+ay and going down, the
//       first row with all mw columns, later rows with mw-1 columns
//       (x from ax-(mw-1)+1); mh == 1 is a single row whose variance uses
//       row y+ay-1.
//   Duplicated cells from the REPLICATE clamp are counted every time.
//   dst = var when aNoise[ch] == 0, otherwise mean when var <= noise and
//   mean + (var-noise)/var * (src-mean) otherwise, with noise scaled by
//   255^2 (8u) / 32767^2 (16s) and used directly for 32f.  Integer outputs
//   truncate and saturate; AC4 leaves alpha untouched.
#pragma once

#include "num_nppi_geom.h"

namespace num {
namespace filtwien {

inline int clampi(int v, int lo, int hi) { return v < lo ? lo : (v > hi ? hi : v); }

template <typename T> inline T wienerEmit(float v) {
  if constexpr (std::is_floating_point_v<T>) {
    return (T)v;
  } else {
    const float hi = std::is_same_v<T, Npp8u> ? 255.0f : 32767.0f;
    const float lo = std::is_same_v<T, Npp8u> ? 0.0f : -32768.0f;
    if (v > hi) v = hi;
    if (v < lo) v = lo;
    return (T)v;
  }
}

template <typename T, int C, int A, class Call>
void run_wiener(const char *id, Call call) {
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
  const bool rectangle = (C == 4 && A == 3);  // AC4 layout only
  const float noiseScale = std::is_same_v<T, Npp8u> ? 255.0f * 255.0f
                           : std::is_same_v<T, Npp16s> ? 32767.0f * 32767.0f
                                                       : 1.0f;
  static const float noiseSets[2][4] = {{0.0f, 0.0f, 0.0f, 0.0f}, {0.001f, 0.002f, 0.003f, 0.0005f}};
  const int m = 2;
  bool failed = false;
  for (const Cfg &cf : cfgs) {
    for (int mi = 0; mi < 2 && !failed; ++mi) {
      for (int ni = 0; ni < 2 && !failed; ++ni) {
        const int mask = masks[mi];
        const int ax = mask / 2, ay = mask / 2;
        const int W = cf.w + 2 * m, H = cf.h + 2 * m;
        const uint64_t seed = case_seed(id, cf.w, cf.h, mask, ni, 0, 0);
        Buf<T> *s = new Buf<T>(W, H, C, 0, 0);
        fill_regime(*s, C, seed, kWide, 0, false, false, false);
        s->dev.copyFromHost(s->host);
        Buf<T> *d = new Buf<T>(cf.w, cf.h, C, 0, 0x5A);
        d->dev.copyFromHost(d->host);
        const std::vector<uint8_t> pre = d->host;
        const int srow = s->step / (int)sizeof(T);
        const T *sp = reinterpret_cast<const T *>(s->dptr() + (size_t)m * s->step +
                                                  (size_t)m * C * sizeof(T));
        const Npp32f *noise = noiseSets[ni];
        npp_diff::clear_cuda_error();
        const NppStatus st = call(sp, s->step, NppiSize{W, H}, NppiPoint{m, m},
                                  reinterpret_cast<T *>(d->dptr()), d->step, NppiSize{cf.w, cf.h},
                                  NppiSize{mask, mask}, NppiPoint{ax, ay}, noise,
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
        const int loX = -m, hiX = W - 1 - m, loY = -m, hiY = H - 1 - m;
        auto load = [&](int xx, int yy, int c) -> float {
          return (float)as_double(
              s->hptr()[(size_t)(m + clampi(yy, loY, hiY)) * srow +
                        (size_t)(m + clampi(xx, loX, hiX)) * C + c]);
        };
        auto stats = [&](int x, int y, int c) -> float {
          float sum = 0.0f;
          if (rectangle) {
            for (int dy = ay - (mask - 1); dy <= ay; ++dy)
              for (int dx = ax - (mask - 1); dx <= ax; ++dx) sum += load(x + dx, y + dy, c);
            return sum;
          }
          if (mask == 1) {
            for (int dx = ax - (mask - 1); dx <= ax; ++dx) sum += load(x + dx, y + ay, c);
            return sum;
          }
          const int rows = mask - 1;
          for (int r = 0; r < rows; ++r) {
            const int x0 = ax - (mask - 1) + (r == 0 ? 0 : 1);
            for (int dx = x0; dx <= ax; ++dx) sum += load(x + dx, y + ay - r, c);
          }
          return sum;
        };
        auto vsumOf = [&](int x, int y, int c, float mean) -> float {
          float acc = 0.0f;
          auto add = [&](int xx, int yy) {
            const float v = load(xx, yy, c);
            acc += warp_fma(v, v, -mean * mean);
          };
          if (rectangle) {
            for (int dy = ay - (mask - 1); dy <= ay; ++dy)
              for (int dx = ax - (mask - 1); dx <= ax; ++dx) add(x + dx, y + dy);
            return acc;
          }
          if (mask == 1) {
            for (int dx = ax - (mask - 1); dx <= ax; ++dx) add(x + dx, y + ay - 1);
            return acc;
          }
          const int rows = mask - 1;
          for (int r = 0; r < rows; ++r) {
            const int x0 = ax - (mask - 1) + (r == 0 ? 0 : 1);
            for (int dx = x0; dx <= ax; ++dx) add(x + dx, y + ay - r);
          }
          return acc;
        };
        const float invN = 1.0f / (float)(mask * mask);
        for (int y = 0; y < cf.h; ++y)
          for (int x = 0; x < cf.w; ++x)
            for (int c = 0; c < A; ++c) {
              const float sum = stats(x, y, c);
              const float mean = sum * invN;
              const float var = vsumOf(x, y, c, mean) * invN;
              const float srcv = load(x, y, c);
              const float ne = noise[c] * noiseScale;
              float o;
              if (noise[c] == 0.0f) {
                o = var;
              } else if (var > ne) {
                o = mean + (var - ne) / var * (srcv - mean);
              } else {
                o = mean;
              }
              const T out = wienerEmit<T>(o);
              const size_t off = (size_t)y * d->step + ((size_t)x * C + (size_t)c) * sizeof(T);
              std::memcpy(want.data() + off, &out, sizeof(T));
            }
        d->dev.copyToHost(d->host);
        const GenCase dc{cf.w, cf.h, mask, 0, 0, kSmall, ni};
        const bool ok = rotate_check<T, C, A>(id, dc, NPPI_INTER_NN, pre, d->host, want, d->step, {},
                                              cnt, first);
        delete s;
        delete d;
        if (!ok || cnt.mismatched >= 8) failed = true;
      }
    }
  }
  finish(id, cnt, first);
}

}  // namespace filtwien
}  // namespace num
