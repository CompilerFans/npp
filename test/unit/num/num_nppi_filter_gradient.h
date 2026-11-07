// CPU-reference output coverage for the gradient-vector filters
// (nppi_filtering_functions.h):
//   nppiGradientVector{Sobel,Prewitt,Scharr}Border.
//
// Measured NPP 13.4 semantics (reference):
//   * X = -(vertical response) for Sobel/Prewitt, +(vertical response) for
//     Scharr; Y = +(horizontal response) for all three, with the same 3x3
//     taps as the derivative-filter executor.  Only Sobel accepts the 5x5
//     mask.
//   * Magnitude: L1 = |X|+|Y|, L2 = sqrt(X^2+Y^2); 16s destinations truncate
//     and saturate, 32f keep the float value.  Angle = atan2(Y, X) radians
//     stored as 32f.
//   * C3C1R computes the response per source channel and the channel with
//     the largest X^2+Y^2 (first wins ties) provides all outputs.
//   * Border reads clamp to the oSrcSize image (REPLICATE only), expressed
//     relative to pSrc: taps (x + b - r, y + a - r) with r = (mask-1)/2.
#pragma once

#include "num_nppi_geom.h"

namespace num {
namespace filtgrad {

inline int clampi(int v, int lo, int hi) { return v < lo ? lo: (v > hi ? hi: v); }

enum Op { kSobel = 0, kPrewitt, kScharr };

struct Coef {
  int size;
  int k[5][5];
};

inline Coef mk(const int *k, int n) {
  Coef c{};
  c.size = n;
  for (int a = 0; a < n; ++a)
    for (int b = 0; b < n; ++b) c.k[a][b] = k[a * n + b];
  return c;
}

inline void tapsOf(int op, int maskSize, Coef &x, Coef &y) {
  static const int SobelHoriz3[9] = {-1, -2, -1, 0, 0, 0, 1, 2, 1};
  static const int SobelVert3[9] = {-1, 0, 1, -2, 0, 2, -1, 0, 1};
  static const int SobelHoriz5[25] = {-1, -4, -6, -4, -1, -2, -8, -12, -8, -2, 0, 0, 0, 0, 0,
                                      2, 8,  12, 8,  2,  1,  4,  6,   4,  1};
  static const int SobelVert5[25] = {-1, -2, 0,  2,  1,  -4, -8, 0,  8,  4,  -6, -12, 0,  12, 6,
                                     -4, -8, 0,  8,  4,  -1, -2, 0,  2,  1};
  static const int PrewittHoriz3[9] = {-1, -1, -1, 0, 0, 0, 1, 1, 1};
  static const int PrewittVert3[9] = {-1, 0, 1, -1, 0, 1, -1, 0, 1};
  static const int ScharrHoriz3[9] = {-3, -10, -3, 0, 0, 0, 3, 10, 3};
  static const int ScharrVert3[9] = {-3, 0, 3, -10, 0, 10, -3, 0, 3};
  if (op == kSobel && maskSize == 5) {
    x = mk(SobelVert5, 5);
    y = mk(SobelHoriz5, 5);
  } else {
    const int *h = op == kSobel ? SobelHoriz3: (op == kPrewitt ? PrewittHoriz3: ScharrHoriz3);
    const int *v = op == kSobel ? SobelVert3: (op == kPrewitt ? PrewittVert3: ScharrVert3);
    x = mk(v, 3);
    y = mk(h, 3);
  }
  if (op != kScharr)
    for (int a = 0; a < x.size; ++a)
      for (int b = 0; b < x.size; ++b) x.k[a][b] = -x.k[a][b];
}

template <typename Dst> inline Dst gvEmit(float v) {
  if constexpr (std::is_same_v<Dst, Npp16s>) {
    return (Npp16s)(v <= -32768.0f ? -32768: (v >= 32767.0f ? 32767: v));
  } else {
    return (Dst)v;
  }
}

// SrcC: 1 or 3 (C3C1R).  Can5: the family accepts a 5x5 mask.
template <typename Src, typename Dst, int SrcC, int Op, bool Can5, class Call>
void run_gradient(const char *id, Call call) {
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
  static const int norms[] = {(int)nppiNormL1, (int)nppiNormL2};
  const int m = 2;
  bool failed = false;
  for (const Cfg &cf: cfgs) {
    for (int mi = 0; mi < (Can5 ? 2: 1) && !failed; ++mi) {
      const int maskSize = mi == 0 ? 3: 5;
      Coef cx, cy;
      tapsOf(Op, maskSize, cx, cy);
      const int r = maskSize / 2;
      for (int ni = 0; ni < 2 && !failed; ++ni) {
        const int norm = norms[ni];
        const int W = cf.w + 2 * m, H = cf.h + 2 * m;
        const uint64_t seed = case_seed(id, cf.w, cf.h, maskSize, SrcC, ni, Op);
        Buf<Src> *s = new Buf<Src>(W, H, SrcC, 0, 0);
        fill_regime(*s, SrcC, seed, kWide, 0, false, false, false);
        s->dev.copyFromHost(s->host);
        Buf<Dst> *dx = new Buf<Dst>(cf.w, cf.h, 1, 0, 0x5A);
        Buf<Dst> *dy = new Buf<Dst>(cf.w, cf.h, 1, 0, 0x5A);
        Buf<Dst> *dm = new Buf<Dst>(cf.w, cf.h, 1, 0, 0x5A);
        Buf<Npp32f> *da = new Buf<Npp32f>(cf.w, cf.h, 1, 0, 0x5A);
        for (Buf<Dst> *b: {dx, dy, dm}) b->dev.copyFromHost(b->host);
        da->dev.copyFromHost(da->host);
        const std::vector<uint8_t> preX = dx->host, preY = dy->host, preM = dm->host, preA = da->host;
        const int srow = s->step / (int)sizeof(Src);
        const Src *sp = reinterpret_cast<const Src *>(s->dptr() + (size_t)m * s->step +
                                                      (size_t)m * SrcC * sizeof(Src));
        npp_diff::clear_cuda_error();
        const NppStatus st =
            call(sp, s->step, NppiSize{W, H}, NppiPoint{m, m}, reinterpret_cast<Dst *>(dx->dptr()),
                 dx->step, reinterpret_cast<Dst *>(dy->dptr()), dy->step,
                 reinterpret_cast<Dst *>(dm->dptr()), dm->step, reinterpret_cast<Npp32f *>(da->dptr()), da->step,
                 NppiSize{cf.w, cf.h}, (int)(maskSize == 3 ? NPP_MASK_SIZE_3_X_3: NPP_MASK_SIZE_5_X_5),
                 (NppiNorm)norm, NPP_BORDER_REPLICATE, ctx);
        if (st != NPP_NO_ERROR) {
          ++cnt.skipped;
          if (st == NPP_NOT_IMPLEMENTED_ERROR) ++cnt.notimpl;
          delete s;
          for (Buf<Dst> *b: {dx, dy, dm}) delete b;
          delete da;
          continue;
        }
        ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess) << "[" << id << "] sync";
        ++cnt.executed;
        std::vector<uint8_t> wX = preX, wY = preY, wM = preM, wA = preA;
        for (int y = 0; y < cf.h; ++y)
          for (int x = 0; x < cf.w; ++x) {
            int accX[3] = {0, 0, 0}, accY[3] = {0, 0, 0};
            float fX[3] = {0, 0, 0}, fY[3] = {0, 0, 0};
            for (int a = 0; a < maskSize; ++a) {
              const int ry = clampi(y + a - r + m, 0, H - 1);
              for (int b = 0; b < maskSize; ++b) {
                const int rx = clampi(x + b - r + m, 0, W - 1);
                for (int c = 0; c < SrcC; ++c) {
                  const float v =
                      (float)as_double(s->hptr()[(size_t)ry * srow + (size_t)rx * SrcC + c]);
                  if constexpr (std::is_floating_point_v<Src>) {
                    fX[c] += (float)cx.k[a][b] * v;
                    fY[c] += (float)cy.k[a][b] * v;
                  } else {
                    accX[c] += cx.k[a][b] * (int)v;
                    accY[c] += cy.k[a][b] * (int)v;
                  }
                }
              }
            }
            int sel = 0;
            if constexpr (SrcC == 3) {
              double best = -1.0;
              for (int c = 0; c < 3; ++c) {
                const double gx = std::is_floating_point_v<Src> ? (double)fX[c]: (double)accX[c];
                const double gy = std::is_floating_point_v<Src> ? (double)fY[c]: (double)accY[c];
                const double l2 = gx * gx + gy * gy;
                if (l2 > best) {
                  best = l2;
                  sel = c;
                }
              }
            }
            const float gx = std::is_floating_point_v<Src> ? fX[sel]: (float)accX[sel];
            const float gy = std::is_floating_point_v<Src> ? fY[sel]: (float)accY[sel];
            const double mag = norm == (int)nppiNormL1
                                   ? (std::fabs((double)gx) + std::fabs((double)gy)): std::sqrt((double)gx * gx + (double)gy * gy);
            const Dst ox = gvEmit<Dst>(gx), oy = gvEmit<Dst>(gy);
            const Dst om = gvEmit<Dst>((float)mag);
            const Npp32f oa = (Npp32f)std::atan2((double)gy, (double)gx);
            const size_t off = (size_t)y * dx->step + (size_t)x * sizeof(Dst);
            const size_t offA = (size_t)y * da->step + (size_t)x * sizeof(Npp32f);
            std::memcpy(wX.data() + off, &ox, sizeof(Dst));
            std::memcpy(wY.data() + off, &oy, sizeof(Dst));
            std::memcpy(wM.data() + off, &om, sizeof(Dst));
            std::memcpy(wA.data() + offA, &oa, sizeof(Npp32f));
          }
        dx->dev.copyToHost(dx->host);
        dy->dev.copyToHost(dy->host);
        dm->dev.copyToHost(dm->host);
        da->dev.copyToHost(da->host);
        const GenCase dc{cf.w, cf.h, maskSize, 0, 0, kSmall, norm};
        bool ok = rotate_check<Dst, 1, 1>(id, dc, NPPI_INTER_NN, preX, dx->host, wX, dx->step, {}, cnt, first);
        ok = rotate_check<Dst, 1, 1>(id, dc, NPPI_INTER_NN, preY, dy->host, wY, dy->step, {}, cnt, first) && ok;
        ok = rotate_check<Dst, 1, 1>(id, dc, NPPI_INTER_NN, preM, dm->host, wM, dm->step, {}, cnt, first) && ok;
        ok = rotate_check<Npp32f, 1, 1>(id, dc, NPPI_INTER_NN, preA, da->host, wA, da->step, {}, cnt,
                                        first) && ok;
        delete s;
        for (Buf<Dst> *b: {dx, dy, dm}) delete b;
        delete da;
        if (!ok || cnt.mismatched >= 8) failed = true;
      }
    }
  }
  finish(id, cnt, first);
}

}  // namespace filtgrad
}  // namespace num
