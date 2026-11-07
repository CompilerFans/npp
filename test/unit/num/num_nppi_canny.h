// CPU-golden coverage for nppiFilterCannyBorder_8u_C1R (nppi_filtering_functions.h).
//
// Measured NPP 13.4 semantics (RTX 5090), decoded stage by stage:
//   * phase 1 is the GradientVectorBorder filter (Sobel or Scharr, mask 3x3 or
//     5x5): integer taps over the oSrcSize image with replicate border
//     expressed relative to pSrc, magnitude truncated to a saturated Npp16s,
//     angle = atan2(Y, X) in radians;
//   * non-maximum suppression uses the 4 direction bins of
//     [0,180) degrees: <22.5 or >=157.5 horizontal, <67.5 anti-diagonal
//     (1,-1)/(-1,1), <112.5 vertical, else main diagonal (1,1)/(-1,-1);
//     a pixel survives when its magnitude is positive and >= both directional
//     neighbours (which are clamped to the ROI edge);
//   * hysteresis is a single pass on the suppressed magnitudes: accepted when
//     m >= high, or when m >= low and any of the eight neighbours in the
//     *unsuppressed* magnitude image is >= low (measured; the documented
//     "two neighbours in the same direction" check does not reproduce the
//     reference at diagonal corner pixels);
//   * the output is 0x00 / 0xFF, and only NPP_BORDER_REPLICATE is supported.
#pragma once

#include "num_nppi_filter_gradient.h"
#include "num_nppi_geom.h"

namespace num {
namespace canny {

struct Dirs {
  int dx1, dy1, dx2, dy2;
};

inline Dirs dirsOf(double deg) {
  while (deg < 0) deg += 180.0;
  if (deg >= 180.0) deg -= 180.0;
  if (deg < 22.5 || deg >= 157.5) return {1, 0, -1, 0};
  if (deg < 67.5) return {1, -1, -1, 1};
  if (deg < 112.5) return {0, 1, 0, -1};
  return {1, 1, -1, -1};
}

// Source patterns (all deterministic, covering the four direction bins).
inline std::vector<Npp8u> make_pattern(int w, int h, int pat) {
  std::vector<Npp8u> img((size_t)w * h, 0);
  for (int y = 0; y < h; ++y)
    for (int x = 0; x < w; ++x) {
      Npp8u v = 0;
      if (pat == 0) {
        v = x >= w / 2 ? 200 : 0;                                  // vertical step
      } else if (pat == 1) {
        v = (x >= w / 4 && x < 3 * w / 4 && y >= h / 4 && y < 3 * h / 4) ? 200 : 20;  // rectangle
      } else if (pat == 2) {
        v = ((x / 4 + y / 4) % 2) ? 200 : 30;                      // checkerboard
      } else if (pat == 3) {
        v = (Npp8u)(x + y);                                        // diagonal ramp
      } else {
        v = (x + y >= w) ? 200 : 20;                               // 45-degree edge
      }
      img[(size_t)y * w + x] = v;
    }
  return img;
}

inline Npp16s emit16(double v) {
  if (v <= -32768.0) return (Npp16s)-32768;
  if (v >= 32767.0) return (Npp16s)32767;
  return (Npp16s)v;  // truncation toward zero
}

template <class Call>
void run_canny(const char *id, Call call) {
  NppStreamContext ctx{};
  if (nppGetStreamContext(&ctx) != NPP_NO_ERROR) {
    GTEST_SKIP() << "[" << id << "] no stream context";
    return;
  }
  RunCounts cnt;
  std::string first;
  struct Cfg {
    int w, h, margin;
  };
  // oSrcSize == oSizeROI and oSrcOffset == (0,0): the reference's gradient
  // border handling around a larger source image does not follow the
  // documented GradientVectorBorder convention, so that offset mode is not
  // modelled here.
  static const Cfg cfgs[] = {{64, 64, 0}, {96, 64, 0}, {48, 48, 0}};
  struct Filter {
    int type;  // NPP_FILTER_SOBEL / NPP_FILTER_SCHARR
    int mask;  // 3 or 5
  };
  static const Filter filters[] = {{0, 3}, {0, 5}, {1, 3}};
  static const int norms[] = {(int)nppiNormL1, (int)nppiNormL2};
  static const int thresh[][2] = {{10, 30}, {200, 400}, {10, 1000}, {100, 100}};
  bool failed = false;
  for (const Cfg &cf : cfgs) {
    const int W = cf.w, H = cf.h, M = cf.margin;
    const int SW = W + 2 * M, SH = H + 2 * M;
    for (int pat = 0; pat < 5 && !failed; ++pat) {
      const std::vector<Npp8u> img = make_pattern(W, H, pat);
      Buf<Npp8u> s(SW, SH, 1, 0, 0);
      for (int y = 0; y < H; ++y)
        std::memcpy(s.host.data() + (size_t)(y + M) * s.step + M, img.data() + (size_t)y * W, (size_t)W);
      s.dev.copyFromHost(s.host);
      const int srow = s.step;
      for (const Filter &f : filters) {
        num::filtgrad::Coef cx, cy;
        num::filtgrad::tapsOf(f.type == 0 ? num::filtgrad::kSobel : num::filtgrad::kScharr, f.mask, cx, cy);
        const int maskSize = cx.size;
        const int r = maskSize / 2;
        for (int ni = 0; ni < 2 && !failed; ++ni) {
          const int norm = norms[ni];
          // Phase 1 + 2 + 3 on the host.
          std::vector<Npp16s> mag((size_t)W * H, 0);
          std::vector<double> ang((size_t)W * H, 0.0);
          for (int y = 0; y < H; ++y)
            for (int x = 0; x < W; ++x) {
              int accX = 0, accY = 0;
              for (int a = 0; a < maskSize; ++a) {
                const int sy = num::filtgrad::clampi(y + M + a - r, 0, SH - 1);
                for (int b = 0; b < maskSize; ++b) {
                  const int sx = num::filtgrad::clampi(x + M + b - r, 0, SW - 1);
                  const int v = (int)s.host[(size_t)sy * srow + (size_t)sx];
                  accX += cx.k[a][b] * v;
                  accY += cy.k[a][b] * v;
                }
              }
              const double magv = norm == (int)nppiNormL1
                                      ? std::fabs((double)accX) + std::fabs((double)accY)
                                      : std::sqrt((double)accX * accX + (double)accY * accY);
              mag[(size_t)y * W + x] = emit16(magv);
              ang[(size_t)y * W + x] = std::atan2((double)accY, (double)accX);
            }
          std::vector<Npp16s> nms((size_t)W * H, 0);
          std::vector<Dirs> dirs((size_t)W * H);
          auto rawAt = [&](int x, int y) -> int {
            x = num::filtgrad::clampi(x, 0, W - 1);
            y = num::filtgrad::clampi(y, 0, H - 1);
            return (int)mag[(size_t)y * W + x];
          };
          for (int y = 0; y < H; ++y)
            for (int x = 0; x < W; ++x) {
              const size_t i = (size_t)y * W + x;
              dirs[i] = dirsOf(ang[i] * 180.0 / M_PI);
              const int m = (int)mag[i];
              if (m > 0 && m >= rawAt(x + dirs[i].dx1, y + dirs[i].dy1) &&
                  m >= rawAt(x + dirs[i].dx2, y + dirs[i].dy2)) {
                nms[i] = (Npp16s)m;
              }
            }
          for (const auto &th : thresh) {
            const int lo = th[0], hi = th[1];
            Buf<Npp8u> d(W, H, 1, 0, 0x5A);
            d.dev.copyFromHost(d.host);
            const std::vector<uint8_t> pre = d.host;
            int need = 0;
            nppiFilterCannyBorderGetBufferSize(NppiSize{W, H}, &need);
            npp_functional_test::DeviceMemory<Npp8u> buf((size_t)(need > 0 ? need : 1));
            npp_diff::clear_cuda_error();
            const NppStatus st = call(reinterpret_cast<const Npp8u *>(s.dptr()), s.step, NppiSize{SW, SH},
                                      NppiPoint{M, M}, reinterpret_cast<Npp8u *>(d.dptr()), d.step,
                                      NppiSize{W, H}, f.type, f.mask == 3 ? (int)NPP_MASK_SIZE_3_X_3
                                                                         : (int)NPP_MASK_SIZE_5_X_5,
                                      (Npp16s)lo, (Npp16s)hi, norm, (int)NPP_BORDER_REPLICATE, buf.get(), ctx);
            if (st != NPP_NO_ERROR) {
              ++cnt.skipped;
              if (st == NPP_NOT_IMPLEMENTED_ERROR) ++cnt.notimpl;
              continue;
            }
            ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess) << "[" << id << "] sync";
            ++cnt.executed;
            std::vector<uint8_t> want = pre;
            for (int y = 0; y < H; ++y)
              for (int x = 0; x < W; ++x) {
                const size_t i = (size_t)y * W + x;
                const int m = (int)nms[i];
                int rmax = 0;
                for (int dy = -1; dy <= 1; ++dy)
                  for (int dx = -1; dx <= 1; ++dx)
                    if (dx != 0 || dy != 0) rmax = std::max(rmax, rawAt(x + dx, y + dy));
                const bool acc = (m >= hi) || (m >= lo && rmax >= lo);
                const Npp8u outv = acc ? (Npp8u)0xFF : (Npp8u)0x00;
                want[(size_t)y * d.step + (size_t)x] = outv;
              }
            d.dev.copyToHost(d.host);
            const GenCase dc{W, H, pat, M, M, kSmall, f.type * 100 + f.mask * 10 + ni};
            (void)dc;
            const GenCase gc{W, H, pat, 0, 0, kSmall, f.type * 100 + f.mask * 10 + ni};
            const bool ok =
                rotate_check<Npp8u, 1, 1>(id, gc, NPPI_INTER_NN, pre, d.host, want, d.step, {}, cnt, first);
            if (!ok) failed = true;
            if (cnt.mismatched >= 8) failed = true;
          }
        }
      }
    }
  }
  finish(id, cnt, first);
}


// ---------------------------------------------------------------------------
// Three-channel source variant (nppiFilterCannyBorder_8u_C3C1R, NPP 13.4).
// Measured on the reference:
//   * the channel is chosen per pixel by the largest X^2 + Y^2 (always the L2
//     measure, like the GradientVector C3C1R family), independently of eNorm;
//   * the emitted magnitude is that channel's magnitude under eNorm and the
//     direction is that channel's atan2;
//   * one NMS + hysteresis pass runs on the combined map (the neighbour
//     comparisons use the combined magnitudes, not the per-channel ones);
//   * a weak rectangle in any single channel produces the full edge map while
//     the luma image produces none, so luma is not used.
// ---------------------------------------------------------------------------
inline std::vector<Npp8u> make_pattern_c3_channel(int w, int h, int ch) {
  // ch0: vertical step; ch1: rectangle; ch2: 45-degree edge.
  const int pat = ch == 0 ? 0 : (ch == 1 ? 1 : 4);
  return make_pattern(w, h, pat);
}

template <class Call>
void run_canny_c3(const char *id, Call call) {
  NppStreamContext ctx{};
  if (nppGetStreamContext(&ctx) != NPP_NO_ERROR) {
    GTEST_SKIP() << "[" << id << "] no stream context";
    return;
  }
  RunCounts cnt;
  std::string first;
  struct Cfg { int w, h; };
  static const Cfg cfgs[] = {{64, 64}, {96, 64}, {48, 48}};
  struct Filter { int type; int mask; };
  static const Filter filters[] = {{0, 3}, {0, 5}, {1, 3}};
  static const int norms[] = {(int)nppiNormL1, (int)nppiNormL2};
  static const int thresh[][2] = {{10, 30}, {200, 400}, {10, 1000}, {100, 100}};
  bool failed = false;
  for (const Cfg &cf : cfgs) {
    const int W = cf.w, H = cf.h;
    for (int variant = 0; variant < 3 && !failed; ++variant) {
      // variant 0: the three-pattern combination; 1: rectangle in channel 0
      // only; 2: the same rectangle in every channel.
      std::vector<Npp8u> img((size_t)W * H * 3, 0);
      std::vector<std::vector<Npp8u>> pats(3);
      for (int c = 0; c < 3; ++c) pats[(size_t)c] = make_pattern_c3_channel(W, H, c);
      for (int y = 0; y < H; ++y)
        for (int x = 0; x < W; ++x)
          for (int c = 0; c < 3; ++c) {
            Npp8u v = 0;
            if (variant == 0) v = pats[(size_t)c][(size_t)y * W + x];
            else if (variant == 1) v = c == 0 ? pats[1][(size_t)y * W + x] : 0;
            else v = pats[1][(size_t)y * W + x];
            img[((size_t)y * W + x) * 3 + c] = v;
          }
      Buf<Npp8u> s(W, H, 3, 0, 0);
      for (int y = 0; y < H; ++y)
        std::memcpy(s.host.data() + (size_t)y * s.step, img.data() + (size_t)y * W * 3, (size_t)W * 3);
      s.dev.copyFromHost(s.host);
      for (const Filter &f : filters) {
        num::filtgrad::Coef cx, cy;
        num::filtgrad::tapsOf(f.type == 0 ? num::filtgrad::kSobel : num::filtgrad::kScharr, f.mask, cx, cy);
        const int maskSize = cx.size;
        const int r = maskSize / 2;
        for (int ni = 0; ni < 2 && !failed; ++ni) {
          const int norm = norms[ni];
          std::vector<Npp16s> mag((size_t)W * H, 0);
          std::vector<double> ang((size_t)W * H, 0.0);
          for (int y = 0; y < H; ++y)
            for (int x = 0; x < W; ++x) {
              double bestSq = -1.0, bestMag = 0.0, bestAng = 0.0;
              for (int c = 0; c < 3; ++c) {
                int accX = 0, accY = 0;
                for (int a = 0; a < maskSize; ++a) {
                  const int sy = num::filtgrad::clampi(y + a - r, 0, H - 1);
                  for (int b = 0; b < maskSize; ++b) {
                    const int sx = num::filtgrad::clampi(x + b - r, 0, W - 1);
                    const int v = (int)s.host[(size_t)sy * s.step + (size_t)sx * 3 + c];
                    accX += cx.k[a][b] * v;
                    accY += cy.k[a][b] * v;
                  }
                }
                const double sq = (double)accX * accX + (double)accY * accY;
                if (sq > bestSq) {
                  bestSq = sq;
                  bestMag = norm == (int)nppiNormL1
                                ? std::fabs((double)accX) + std::fabs((double)accY)
                                : std::sqrt(sq);
                  bestAng = std::atan2((double)accY, (double)accX);
                }
              }
              mag[(size_t)y * W + x] = emit16(bestMag);
              ang[(size_t)y * W + x] = bestAng;
            }
          auto rawAt = [&](int x, int y) -> int {
            x = num::filtgrad::clampi(x, 0, W - 1);
            y = num::filtgrad::clampi(y, 0, H - 1);
            return (int)mag[(size_t)y * W + x];
          };
          std::vector<Npp16s> nms((size_t)W * H, 0);
          for (int y = 0; y < H; ++y)
            for (int x = 0; x < W; ++x) {
              const size_t i = (size_t)y * W + x;
              const Dirs d = dirsOf(ang[i] * 180.0 / M_PI);
              const int m = (int)mag[i];
              if (m > 0 && m >= rawAt(x + d.dx1, y + d.dy1) && m >= rawAt(x + d.dx2, y + d.dy2))
                nms[i] = (Npp16s)m;
            }
          for (const auto &th : thresh) {
            const int lo = th[0], hi = th[1];
            Buf<Npp8u> d(W, H, 1, 0, 0x5A);
            d.dev.copyFromHost(d.host);
            const std::vector<uint8_t> pre = d.host;
            int need = 0;
            nppiFilterCannyBorderGetBufferSize(NppiSize{W, H}, &need);
            npp_functional_test::DeviceMemory<Npp8u> buf((size_t)(need > 0 ? need : 1));
            npp_diff::clear_cuda_error();
            const NppStatus st = call(reinterpret_cast<const Npp8u *>(s.dptr()), s.step, NppiSize{W, H},
                                      NppiPoint{0, 0}, reinterpret_cast<Npp8u *>(d.dptr()), d.step,
                                      NppiSize{W, H}, f.type, f.mask == 3 ? (int)NPP_MASK_SIZE_3_X_3
                                                                         : (int)NPP_MASK_SIZE_5_X_5,
                                      (Npp16s)lo, (Npp16s)hi, norm, (int)NPP_BORDER_REPLICATE, buf.get(), ctx);
            if (st != NPP_NO_ERROR) {
              ++cnt.skipped;
              if (st == NPP_NOT_IMPLEMENTED_ERROR) ++cnt.notimpl;
              continue;
            }
            ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess) << "[" << id << "] sync";
            ++cnt.executed;
            std::vector<uint8_t> want = pre;
            for (int y = 0; y < H; ++y)
              for (int x = 0; x < W; ++x) {
                const size_t i = (size_t)y * W + x;
                const int m = (int)nms[i];
                int rmax = 0;
                for (int dy = -1; dy <= 1; ++dy)
                  for (int dx = -1; dx <= 1; ++dx)
                    if (dx != 0 || dy != 0) rmax = std::max(rmax, rawAt(x + dx, y + dy));
                const bool acc = (m >= hi) || (m >= lo && rmax >= lo);
                want[(size_t)y * d.step + (size_t)x] = acc ? (Npp8u)0xFF : (Npp8u)0x00;
              }
            d.dev.copyToHost(d.host);
            const GenCase gc{W, H, variant, 0, 0, kSmall, f.type * 100 + f.mask * 10 + ni};
            const bool ok = rotate_check<Npp8u, 1, 1>(id, gc, NPPI_INTER_NN, pre, d.host, want, d.step, {}, cnt, first);
            if (!ok) failed = true;
            if (cnt.mismatched >= 8) failed = true;
          }
        }
      }
    }
  }
  finish(id, cnt, first);
}

}  // namespace canny
}  // namespace num
