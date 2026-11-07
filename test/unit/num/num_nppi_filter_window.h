// CPU-reference output coverage for the window/order-statistics filter families
// (nppi_filtering_functions.h):
//   nppiFilterBox{,_Border,_BorderAdvanced} and later Min/Max/Median.
//
// Measured NPP 13.4 semantics (reference):
//   * window geometry: dst(x,y)[c] reads src(x + i - ax, y + j - ay)[c] for
//     i in [0,mw), j in [0,mh);
//   * box: sum of the taps divided by mw*mh, integer destinations truncate
//     toward zero and saturate, 32f accumulates in float, 64f in double;
//   * plain variants read raw memory around pSrc (caller margin); Border
//     accepts REPLICATE (2), MIRROR (4, reflect without repeating the edge)
//     and the unnamed value 5 (reflect repeating the edge); the mirrored
//     kernels exist only for the square 3x3 and 5x5 masks;
//   * BoxBorderAdvanced accepts REPLICATE only and takes an unused device
//     buffer.  The reference kernel only reproduces the documented result when
//     oSizeROI.width == oSrcSize.width and oSrcOffset.x == 0, so this driver
//     uses exactly that geometry for the advanced family (full-width
//     destination, no horizontal margin) and exercises the y offset instead.
//   * AC4 filters channels 0..2 and leaves the destination alpha untouched.
#pragma once

#include "num_nppi_geom.h"

#include <algorithm>

namespace num {
namespace filtwin {

enum WOp { kBox = 0, kMin, kMax, kMedian };

struct MaskSize {
  int w, h;
};

inline int mapBorder(int v, int n, int mode) {
  if (n <= 0) return 0;
  if (mode == NPP_BORDER_REPLICATE) return v < 0 ? 0: (v >= n ? n - 1: v);
  if (n == 1) return 0;
  if (mode == NPP_BORDER_MIRROR) {  // reflect without repeating the edge
    const int p = 2 * (n - 1);
    v %= p;
    if (v < 0) v += p;
    return v < n ? v: p - v;
  }
  // value 5: reflect repeating the edge
  const int p = 2 * n;
  v %= p;
  if (v < 0) v += p;
  return v < n ? v: p - 1 - v;
}

template <typename T> inline T clampInt(long long v) {
  if constexpr (std::is_same_v<T, Npp8u>) {
    return (Npp8u)(v <= 0 ? 0: (v >= 255 ? 255: v));
  } else if constexpr (std::is_same_v<T, Npp16u>) {
    return (Npp16u)(v <= 0 ? 0: (v >= 65535 ? 65535: v));
  } else {
    return (T)(v <= -32768 ? -32768: (v >= 32767 ? 32767: v));
  }
}

// Border=false: raw reads.  Advanced=true implies Border semantics
// (REPLICATE only) and the full-width geometry described above.
template <typename T, int C, int A, int Op, bool Border, bool Advanced, class Call>
void run_window(const char *id, Call call) {
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
  static const Cfg cfgsPlain[] = {{6, 6}, {8, 6}, {33, 25}, {96, 64}};
  // Advanced geometry needs a full-width destination whose row size is also
  // the natural (16-byte aligned) pitch for every layout.
  static const Cfg cfgsAdv[] = {{16, 12}, {32, 24}, {64, 48}, {128, 96}};
  const Cfg *cfgs = Advanced ? cfgsAdv: cfgsPlain;
  const int nCfg = 4;
  static const MaskSize masks[] = {{3, 3}, {5, 5}, {3, 5}, {5, 2}, {1, 1}};
  const int nMasks = 5;
  const int m = 4;  // covers masks up to 5x5 under every anchor
  npp_functional_test::DeviceMemory<Npp8u> pbuf((size_t)1 << 22);  // Advanced scratch
  bool failed = false;
  for (int fi = 0; fi < nCfg && !failed; ++fi) {
    const Cfg &cf = cfgs[fi];
    for (int mi = 0; mi < nMasks && !failed; ++mi) {
      const MaskSize ms = masks[mi];
      // The reference's window kernels behave differently once the mask no
      // longer fits the ROI; keep every compared case inside the mask.
      if (ms.w > cf.w || ms.h > cf.h) continue;
      // Measured: NPP 13.4's 5x5 MedianBorder kernels do not follow the
      // image-clamped window model at all (the produced median is not even an
      // order statistic of the documented 5x5 window); the reference comparisons
      // also restricts median-border value records to the smaller masks.
      if (Op == kMedian && Border && ms.w * ms.h > 15) continue;
      // Even sample counts: the reference's small-mask median kernels pick
      // shape-dependent tie conventions (truncate vs floor vs away-from-zero
      // for the same operand pair), so even windows are not a reproducible
      if (Op == kMedian && (ms.w * ms.h) % 2 == 0) continue;
      // Measured: for larger ROIs NPP's median-border kernels only reproduce
      // the documented window for square masks; non-square masks are covered
      // on the small ROIs, where they agree.
      if (Op == kMedian && Border && ms.w != ms.h && cf.w * cf.h > 64) continue;
      int modes[3];
      int nMode = 0;
      if constexpr (Advanced) {
        modes[nMode++] = NPP_BORDER_REPLICATE;
      } else if constexpr (Border) {
        modes[nMode++] = NPP_BORDER_REPLICATE;
        const bool square = (ms.w == 3 && ms.h == 3) || (ms.w == 5 && ms.h == 5);
        if constexpr (Op == kBox) {
          if (square) {
            modes[nMode++] = NPP_BORDER_MIRROR;
            modes[nMode++] = 5;
          }
        } else if (Op == kMedian) {
          if (square) modes[nMode++] = NPP_BORDER_MIRROR;
        }
      } else {
        modes[nMode++] = -1;  // no border argument
      }
      const int anchors[2][2] = {{(ms.w - 1) / 2, (ms.h - 1) / 2}, {0, 0}};
      const int nOff = Advanced ? 2: 1;
      for (int ai = 0; ai < 2 && !failed; ++ai) {
        for (int bi = 0; bi < nMode && !failed; ++bi) {
          for (int oi = 0; oi < nOff && !failed; ++oi) {
          const int mode = modes[bi];
          // The advanced reference only reproduces the documented contract
          // when the ROI is the whole source image horizontally; its own
          // verified geometry puts the image at an allocation offset and runs
          // the ROI over the full image with an oSrcOffset.y of 0..2.
          const int W = cf.w + 2 * m, H = cf.h + 2 * m;  // allocation
          const int srcW = Advanced ? cf.w: W;
          const int srcH = Advanced ? cf.h: H;
          const int offX = Advanced ? 0: m;
          const int offY = Advanced ? oi * 2: m;
          const int base = Advanced ? m: 0;  // allocation offset of image (0,0)
          const uint64_t seed = case_seed(id, cf.w, cf.h, ms.w * 16 + ms.h,
                                          ai * 8 + (mode < 0 ? 0: mode) + 32 * oi, Advanced ? 1: 0, 0);
          Buf<T> *s = new Buf<T>(W, H, C, 0, 0);
          fill_regime(*s, C, seed, kWide, 0, false, false, false);
          s->dev.copyFromHost(s->host);
          Buf<T> *d = new Buf<T>(cf.w, cf.h, C, 0, 0x5A);
          d->dev.copyFromHost(d->host);
          const std::vector<uint8_t> pre = d->host;
          const int srow = s->step / (int)sizeof(T);
          const T *sp = reinterpret_cast<const T *>(s->dptr() + (size_t)(base + offY) * s->step +
                                                    (size_t)(base + offX) * C * sizeof(T));
          npp_diff::clear_cuda_error();
          const NppStatus st = call(sp, s->step, reinterpret_cast<T *>(d->dptr()), d->step,
                                    NppiSize{cf.w, cf.h}, NppiSize{srcW, srcH}, NppiPoint{offX, offY},
                                    NppiSize{ms.w, ms.h}, NppiPoint{anchors[ai][0], anchors[ai][1]}, mode,
                                    pbuf.get(), ctx);
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
          const int area = ms.w * ms.h;
          for (int y = 0; y < cf.h; ++y)
            for (int x = 0; x < cf.w; ++x)
              for (int c = 0; c < A; ++c) {
                std::vector<double> smp;
                smp.reserve((size_t)area);
                for (int j = 0; j < ms.h; ++j)
                  for (int i = 0; i < ms.w; ++i) {
                    int ix = offX + x + i - anchors[ai][0];
                    int iy = offY + y + j - anchors[ai][1];
                    if constexpr (Border || Advanced) {
                      ix = mapBorder(ix, srcW, mode);
                      iy = mapBorder(iy, srcH, mode);
                    }
                    smp.push_back(
                        (double)s->hptr()[(size_t)(base + iy) * srow + (size_t)(base + ix) * C + c]);
                  }
                T out;
                if (Op == kBox) {
                  if constexpr (std::is_same_v<T, Npp64f>) {
                    double sum = 0.0;
                    for (double v: smp) sum += v;
                    out = (T)(sum / (double)area);
                  } else if constexpr (std::is_floating_point_v<T>) {
                    float sum = 0.0f;
                    for (double v: smp) sum += (float)v;
                    out = (T)(sum / (float)area);
                  } else {
                    long long sum = 0;
                    for (double v: smp) sum += (long long)v;
                    out = clampInt<T>(sum / area);
                  }
                } else if (Op == kMin || Op == kMax) {
                  double best = smp[0];
                  for (double v: smp) best = Op == kMin ? (v < best ? v: best): (v > best ? v: best);
                  out = (T)best;
                } else {
                  std::sort(smp.begin(), smp.end());
                  const int k1 = (area - 1) / 2, k2 = area / 2;
                  if (k1 == k2) {
                    out = (T)smp[(size_t)k1];
                  } else if constexpr (std::is_floating_point_v<T>) {
                    out = (T)(((float)smp[(size_t)k1] + (float)smp[(size_t)k2]) * 0.5f);
                  } else {
                    out = clampInt<T>(((long long)smp[(size_t)k1] + (long long)smp[(size_t)k2]) / 2);
                  }
                }
                const size_t off = (size_t)y * d->step + ((size_t)x * C + (size_t)c) * sizeof(T);
                std::memcpy(want.data() + off, &out, sizeof(T));
              }
          d->dev.copyToHost(d->host);
          const GenCase dc{cf.w, cf.h, 0, 0, 0, kSmall, 0};
          const bool ok = rotate_check<T, C, A>(id, dc, NPPI_INTER_NN, pre, d->host, want, d->step, {}, cnt,
                                                first);
          delete s;
          delete d;
          if (!ok || cnt.mismatched >= 8) failed = true;
          }
        }
      }
    }
  }
  finish(id, cnt, first);
}

}  // namespace filtwin
}  // namespace num
