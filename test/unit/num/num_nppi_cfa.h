// CPU-golden coverage for the Bayer CFA demosaic families
// (nppiCFAToRGB / nppiCFAToRGBA for 8u/16u/32u).
//
// Bilinear demosaic with chroma correlation, reproduced from the measured
// NPP 13.4 integer pipeline: base pixels copied, green at R/B bases chosen by
// the |2*center - (n1+n2)| horizontal/vertical comparison (an exact tie
// averages the four cardinal G samples), missing R/B from the parity-ordered
// diagonal averages, every stage floored, and border samples mirrored.
// The NVIDIA ROI quirk is honored: the produced extent is
// (width - x) x (height - y).
#pragma once

#include "num_nppi_geom.h"

namespace num {
namespace cfa {

inline int color_at(int grid, int x, int y) {
  const int p = ((y & 1) << 1) | (x & 1);
  switch (grid) {
    case 0: return (p == 0) ? 2 : ((p == 3) ? 0 : 1);  // BGGR
    case 1: return (p == 0) ? 0 : ((p == 3) ? 2 : 1);  // RGGB
    case 2: return (p == 0 || p == 3) ? 1 : ((p == 1) ? 2 : 0);  // GBRG
    default: return (p == 0 || p == 3) ? 1 : ((p == 1) ? 0 : 2);  // GRBG
  }
}

inline int mirror_index(int i, int n) {
  if (i < 0) return -i;
  if (i >= n) return 2 * n - 2 - i;
  return i;
}

template <typename T>
inline T at(const T *p, int stepElems, int w, int h, int x, int y) {
  return p[(size_t)mirror_index(y, h) * stepElems + (size_t)mirror_index(x, w)];
}

template <typename T>
inline T avg2(T a, T b) {
  return (T)(((unsigned long long)a + (unsigned long long)b) >> 1);
}

template <typename T>
inline T avg4_diag(int y, T lt, T rt, T lb, T rb) {
  if ((y & 1) == 0) return avg2(avg2(lt, lb), avg2(rt, rb));
  return avg2(avg2(lt, rt), avg2(lb, rb));
}

template <typename T, int OutC, class Call>
void run_cfa(const char *id, Call call) {
  NppStreamContext ctx{};
  if (nppGetStreamContext(&ctx) != NPP_NO_ERROR) {
    GTEST_SKIP() << "[" << id << "] no stream context";
    return;
  }
  RunCounts cnt;
  std::string first;
  struct Cfg { int w, h, ox, oy; };
  static const Cfg cfgs[] = {{4, 4, 0, 0}, {8, 6, 0, 0}, {32, 24, 0, 0},
                             {64, 48, 0, 0}, {128, 96, 0, 0}, {256, 192, 0, 0},
                             {16, 12, 2, 2}, {32, 24, 4, 2}};
  for (const Cfg &cf : cfgs) {
    Buf<T> s(cf.w, cf.h, 1, 0, 0);
    fill_regime(s, 1, case_seed(id, cf.w, cf.h, 0, 0, 0, 0), (cf.w & 2) ? kWide : kSmall, 0, false, false,
                false);
    s.dev.copyFromHost(s.host);
    const T *hsrc = s.hptr();
    const T *dsrc = reinterpret_cast<const T *>(s.dptr());
    const int sstep = s.step / (int)sizeof(T);
    const NppiSize size{cf.w, cf.h};
    const NppiRect roi{cf.ox, cf.oy, cf.w, cf.h};
    const int pw = cf.w - cf.ox, ph = cf.h - cf.oy;
    Buf<T> d(cf.w, cf.h, OutC, 0, 0x5A);
    d.dev.copyFromHost(d.host);
    const int dstep = d.step / (int)sizeof(T);
    for (int grid = 0; grid < 4; ++grid) {
      const T alpha = (T)(sizeof(T) == 1 ? 200 : (sizeof(T) == 2 ? 40000 : 0xABCDEFu));
      const NppStatus st = call(dsrc, s.step, size, roi, reinterpret_cast<T *>(d.dptr()), d.step, grid,
                                (unsigned)alpha, ctx);
      if (st != NPP_NO_ERROR) {
        ++cnt.skipped;
        if (st == NPP_NOT_IMPLEMENTED_ERROR) ++cnt.notimpl;
        continue;
      }
      const cudaError_t se = cudaDeviceSynchronize();
      ASSERT_EQ(se, cudaSuccess) << "[" << id << "] sync: " << cudaGetErrorString(se);
      d.dev.copyToHost(d.host);
      ++cnt.executed;
      std::vector<uint8_t> want = d.host;
      for (int oy = 0; oy < ph; ++oy)
        for (int ox = 0; ox < pw; ++ox) {
          const int ax = roi.x + ox, ay = roi.y + oy;
          const int color = color_at(grid, ox, oy);
          const T base = at(hsrc, sstep, cf.w, cf.h, ax, ay);
          T r = 0, g = 0, b = 0;
          if (color == 1) {
            g = base;
            const bool rHorizontal = (color_at(grid, ox - 1, oy) == 0) || (color_at(grid, ox + 1, oy) == 0);
            r = rHorizontal ? avg2(at(hsrc, sstep, cf.w, cf.h, ax - 1, ay), at(hsrc, sstep, cf.w, cf.h, ax + 1, ay))
                            : avg2(at(hsrc, sstep, cf.w, cf.h, ax, ay - 1), at(hsrc, sstep, cf.w, cf.h, ax, ay + 1));
            const bool bHorizontal = (color_at(grid, ox - 1, oy) == 2) || (color_at(grid, ox + 1, oy) == 2);
            b = bHorizontal ? avg2(at(hsrc, sstep, cf.w, cf.h, ax - 1, ay), at(hsrc, sstep, cf.w, cf.h, ax + 1, ay))
                            : avg2(at(hsrc, sstep, cf.w, cf.h, ax, ay - 1), at(hsrc, sstep, cf.w, cf.h, ax, ay + 1));
          } else {
            const long long c2 = 2LL * (long long)base;
            const long long dH =
                c2 - ((long long)at(hsrc, sstep, cf.w, cf.h, ax - 2, ay) + (long long)at(hsrc, sstep, cf.w, cf.h, ax + 2, ay));
            const long long dV =
                c2 - ((long long)at(hsrc, sstep, cf.w, cf.h, ax, ay - 2) + (long long)at(hsrc, sstep, cf.w, cf.h, ax, ay + 2));
            const long long aH = dH < 0 ? -dH : dH, aV = dV < 0 ? -dV : dV;
            const T gl = at(hsrc, sstep, cf.w, cf.h, ax - 1, ay), gr = at(hsrc, sstep, cf.w, cf.h, ax + 1, ay);
            const T gu = at(hsrc, sstep, cf.w, cf.h, ax, ay - 1), gd = at(hsrc, sstep, cf.w, cf.h, ax, ay + 1);
            if (aH < aV) g = avg2(gl, gr);
            else if (aV < aH) g = avg2(gu, gd);
            else g = avg2(avg2(gl, gr), avg2(gu, gd));
            const T other = avg4_diag(
                oy, at(hsrc, sstep, cf.w, cf.h, ax - 1, ay - 1), at(hsrc, sstep, cf.w, cf.h, ax + 1, ay - 1),
                at(hsrc, sstep, cf.w, cf.h, ax - 1, ay + 1), at(hsrc, sstep, cf.w, cf.h, ax + 1, ay + 1));
            if (color == 0) { r = base; b = other; }
            else { b = base; r = other; }
          }
          const size_t db = (size_t)oy * dstep + (size_t)ox * OutC;
          std::memcpy(want.data() + db * sizeof(T), &r, sizeof(T));
          std::memcpy(want.data() + db * sizeof(T) + sizeof(T), &g, sizeof(T));
          std::memcpy(want.data() + db * sizeof(T) + 2 * sizeof(T), &b, sizeof(T));
          if constexpr (OutC == 4) std::memcpy(want.data() + db * sizeof(T) + 3 * sizeof(T), &alpha, sizeof(T));
        }
      const GenCase dc{pw, ph, 0, 0, 0, kSmall, 0};
      if (!rotate_check<T, OutC, OutC>(id, dc, NPPI_INTER_NN, d.host, d.host, want, d.step,
                                       std::vector<uint8_t>(), cnt, first)) {
        if (cnt.mismatched >= 8) break;
      }
      if (cnt.mismatched >= 8) break;
    }
    if (cnt.mismatched >= 8) break;
  }
  finish(id, cnt, first);
}

}  // namespace cfa
}  // namespace num
