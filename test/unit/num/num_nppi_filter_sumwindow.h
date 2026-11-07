// CPU-reference output coverage for the running-sum filters
// (nppi_filtering_functions.h):
//   nppiSumWindowRow / Column and their *Border variants (8u,16u,16s -> 32f).
//
// Measured NPP 13.4 semantics (reference):
//   dst(x,y)[c] = sum_{i=0}^{mask-1} src[x - ax + i][y][c]   (row)
//                 sum_{i=0}^{mask-1} src[x][y - ay + i][c]   (column)
//   i.e. the window starts at the destination coordinate minus the anchor.
//   Plain variants read raw memory, except that the row kernel clamps its
//   rightmost tap coordinate to W + mask - 1 (the column kernel has no such
//   clamp); the driver keeps a wide margin so the clamped reads stay inside
//   the allocation.  Border variants accept NPP_BORDER_REPLICATE only, clamp
//   every tap into [0, oSrcSize-1] and replicate-clamp the cross axis too.
//   The accumulator and the destination are always 32f.
#pragma once

#include "num_nppi_geom.h"

namespace num {
namespace filtsum {

inline int clampi(int v, int lo, int hi) { return v < lo ? lo: (v > hi ? hi: v); }

// Row=false selects the column kernel.
template <typename Src, int C, bool Row, bool Border, class Call>
void run_sumwindow(const char *id, Call call) {
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
  static const int masks[] = {3, 5, 7};
  const int m = 6;
  bool failed = false;
  for (const Cfg &cf: cfgs) {
    for (int mi = 0; mi < 3 && !failed; ++mi) {
      const int mask = masks[mi];
      const int anchor = mask / 2;
      const int W = cf.w + 2 * m, H = cf.h + 2 * m;
      const uint64_t seed = case_seed(id, cf.w, cf.h, mask, Border ? 1: 0, 0, 0);
      Buf<Src> *s = new Buf<Src>(W, H, C, 0, 0);
      fill_regime(*s, C, seed, kWide, 0, false, false, false);
      s->dev.copyFromHost(s->host);
      Buf<Npp32f> *d = new Buf<Npp32f>(cf.w, cf.h, C, 0, 0);
      d->dev.copyFromHost(d->host);
      const std::vector<uint8_t> pre = d->host;
      const int srow = s->step / (int)sizeof(Src);
      const Src *sp = reinterpret_cast<const Src *>(s->dptr() + (size_t)m * s->step +
                                                    (size_t)m * C * sizeof(Src));
      npp_diff::clear_cuda_error();
      const NppStatus st = call(sp, s->step, reinterpret_cast<Npp32f *>(d->dptr()), d->step, NppiSize{cf.w, cf.h}, NppiSize{W, H},
                                NppiPoint{m, m}, mask, anchor, Border ? (int)NPP_BORDER_REPLICATE: 0,
                                ctx);
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
          for (int c = 0; c < C; ++c) {
            float acc = 0.0f;
            for (int i = 0; i < mask; ++i) {
              int px, py;
              if constexpr (Row) {
                int k = x - anchor + i;
                if constexpr (Border) {
                  px = clampi(k + m, 0, W - 1);
                  py = clampi(y + m, 0, H - 1);
                } else {
                  if (k > cf.w + mask - 1) k = cf.w + mask - 1;
                  px = m + k;
                  py = m + y;
                }
              } else {
                int k = y - anchor + i;
                if constexpr (Border) {
                  py = clampi(k + m, 0, H - 1);
                  px = clampi(x + m, 0, W - 1);
                } else {
                  py = m + k;
                  px = m + x;
                }
              }
              acc += (float)as_double(s->hptr()[(size_t)py * srow + (size_t)px * C + c]);
            }
            const size_t off = (size_t)y * d->step + ((size_t)x * C + (size_t)c) * sizeof(Npp32f);
            std::memcpy(want.data() + off, &acc, sizeof(Npp32f));
          }
      d->dev.copyToHost(d->host);
      const GenCase dc{cf.w, cf.h, mask, 0, 0, kSmall, 0};
      const bool ok = rotate_check<Npp32f, C, C>(id, dc, NPPI_INTER_NN, pre, d->host, want, d->step, {},
                                                 cnt, first);
      delete s;
      delete d;
      if (!ok || cnt.mismatched >= 8) failed = true;
    }
  }
  finish(id, cnt, first);
}

}  // namespace filtsum
}  // namespace num
