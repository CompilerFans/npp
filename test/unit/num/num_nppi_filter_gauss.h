// CPU-reference output coverage for the Gauss filter families
// (nppi_filtering_functions.h):
//   nppiFilterGauss{,_Border} and nppiFilterGaussAdvanced{,_Border}.
//
// Measured NPP 13.4 semantics (reference):
//   * nppiFilterGauss accepts the square masks 3,5,7,9,11,13,15.  The 1D
//     kernel is exp(-d^2 / (2 sigma^2)) with sigma = 0.6 + 0.4 * (taps/2),
//     normalised so the taps sum to one; the 2D kernel is the outer product
//     k[j]*k[i] applied as a correlation (origin (taps-1)/2).
//   * nppiFilterGaussAdvanced takes an odd tap count >= 3 and the 1D user
//     kernel (host pointer), again used as an unnormalised outer product.
//   * Accumulation is float32 with fused multiply-add in row/column order;
//     integer destinations clamp to their range and truncate toward zero,
//     32f stores the accumulator unchanged.  AC4 leaves alpha untouched.
//   * Plain variants read raw neighbours around pSrc (caller margin); Border
//     variants accept NPP_BORDER_REPLICATE and clamp reads to the oSrcSize
//     image relative to pSrc.  The driver allocates a margin of seven pixels
//     and sets pSrc to (7,7), which covers the largest fixed kernel.
#pragma once

#include "num_nppi_geom.h"

namespace num {
namespace filtgauss {

inline void gauss1d(int taps, float *k) {
  const int r = taps / 2;
  const float sigma = 0.6f + 0.4f * (float)r;
  const float denom = 2.0f * sigma * sigma;
  float s = 0.0f;
  for (int i = 0; i < taps; ++i) {
    const float d = (float)(i - r);
    k[i] = std::exp(-(d * d) / denom);
    s += k[i];
  }
  for (int i = 0; i < taps; ++i) k[i] = k[i] / s;
}

template <typename Dst> inline Dst gaussEmit(float acc) {
  if constexpr (std::is_floating_point_v<Dst>) {
    return (Dst)acc;
  } else {
    constexpr float kLo = std::is_same_v<Dst, Npp16s> ? -32768.0f: 0.0f;
    constexpr float kHi = std::is_same_v<Dst, Npp8u> ? 255.0f: std::is_same_v<Dst, Npp16u> ? 65535.0f: 32767.0f;
    const float lo = acc < kLo ? kLo: acc;
    const float v = lo > kHi ? kHi: lo;
    return (Dst)(int)v;
  }
}

// Advanced=false: fixed Gaussian masks 3..15; Advanced=true: user tap counts
// 3 and 5 with the built-in normalised kernels below.
template <typename Src, typename Dst, int C, int A, bool Advanced, bool Border, class Call>
void run_gauss(const char *id, Call call) {
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
  static const int fixedMasks[] = {3, 5, 7, 9, 11, 13, 15};
  static const int advTaps[] = {3, 5};
  static const float kAdv3[3] = {0.25f, 0.5f, 0.25f};
  static const float kAdv5[5] = {0.0625f, 0.25f, 0.375f, 0.25f, 0.0625f};
  const int nCase = Advanced ? 2: 7;
  const int m = Advanced ? 2: 7;
  bool failed = false;
  for (const Cfg &cf: cfgs) {
    for (int ci = 0; ci < nCase && !failed; ++ci) {
      const int taps = Advanced ? advTaps[ci]: fixedMasks[ci];
      const int maskEnum = Advanced ? (int)NPP_MASK_SIZE_3_X_3: (taps == 3 ? (int)NPP_MASK_SIZE_3_X_3: taps == 5 ? (int)NPP_MASK_SIZE_5_X_5: taps == 7 ? (int)NPP_MASK_SIZE_7_X_7: taps == 9 ? (int)NPP_MASK_SIZE_9_X_9: taps == 11 ? (int)NPP_MASK_SIZE_11_X_11: taps == 13 ? (int)NPP_MASK_SIZE_13_X_13: (int)NPP_MASK_SIZE_15_X_15);
      // pKernel is a DEVICE pointer for the GaussAdvanced families (the
      // reference and the in-tree kernel read it from global memory).
      std::vector<float> khost((size_t)taps);
      if (Advanced) {
        const float *src = taps == 3 ? kAdv3: kAdv5;
        for (int i = 0; i < taps; ++i) khost[(size_t)i] = src[i];
      } else {
        gauss1d(taps, khost.data());
      }
      npp_functional_test::DeviceMemory<float> kdev((size_t)taps);
      kdev.copyFromHost(khost);
      const Npp32f *ker = kdev.get();
      const int r = taps / 2;
      const int W = cf.w + 2 * m, H = cf.h + 2 * m;
      const uint64_t seed = case_seed(id, cf.w, cf.h, taps, Border ? 1: 0, 0, 0);
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
                                NppiSize{W, H}, NppiPoint{m, m}, maskEnum, taps, ker, ctx);
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
            float acc = 0.0f;
            for (int j = 0; j < taps; ++j) {
              int yy = m + y + j - r;
              if constexpr (Border) yy = yy < 0 ? 0: (yy > H - 1 ? H - 1: yy);
              const float kj = khost[(size_t)j];
              for (int i = 0; i < taps; ++i) {
                int xx = m + x + i - r;
                if constexpr (Border) xx = xx < 0 ? 0: (xx > W - 1 ? W - 1: xx);
                const float kv = kj * khost[(size_t)i];
                const Src v = s->hptr()[(size_t)yy * srow + (size_t)xx * C + c];
                acc = warp_fma(kv, (float)v, acc);
              }
            }
            const Dst out = gaussEmit<Dst>(acc);
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

}  // namespace filtgauss
}  // namespace num
