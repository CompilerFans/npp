// CPU-reference output coverage for the signed PBA distance transforms
// (nppi_filtering_functions.h):
//   nppiSignedDistanceTransformPBA_{32f,32f64f,64f}_C1R and the matching
//   nppiSignedDistanceTransformAbsPBA entry points.
//
// Measured NPP 13.4 semantics (reference; decoded from libnppif):
//   * inside pixels are those with src >= 1; outside sites are src <= 0 or
//     src < nCutoffValue;
//   * with the nearest opposite-class site (or the nearest inside pixel for
//     the Abs flavour) at (sx,sy): a = |x-sx| + subX, b = |y-sy| + subY,
//     C = 30 + cutoff, t = sqrt((max(a,b)-C)^2 + min(a,b)^2);
//   * signed 32f / 64f and Abs 64f store +t inside and -t outside (the
//     reference's Abs 64f entry point ignores the antialiasing buffer and
//     emits the same signed transform);
//   * Abs 32f stores +t with the site being the nearest inside pixel;
//   * the 32f64f entry points switch to an antialiasing path when the AA
//     buffer is supplied.  The Abs 32f64f entry point requires that buffer and
//     emits sqrt((d+subX)^2 + (d+subY)^2) where d is the horizontal distance
//     to the inside column span (the reference uses the horizontal distance
//     for both axes on binary masks).  The signed 32f64f AA path is not a
//     distance field for interior pixels (it collapses to a constant), so this
//     suite exercises its non-AA mode (AA buffer NULL), which is a documented
//     NPP usage and produces the folded signed transform;
//   * no inside pixels: the signed transform zeroes the output and the Abs
//     paths reject the call; the reference faults on all-inside / all-outside
//     masks, so every case below keeps both classes present.
// Only closed rectangular regions are used: the reference's antialiased path
// produces row-rotated values for open or fragmented masks (and faults for
// some of them).
#pragma once

#include "num_nppi_geom.h"

namespace num {
namespace signeddt {

enum Model {
  kSignedFolded = 0,   // signed folded transform (inside +t, outside -t)
  kAbsFolded = 1,      // folded magnitude to the nearest inside pixel
  kEuclidOpposite = 2, // antialiased signed 32f64f: |distance| to the opposite class
  kAaAbs = 3,          // antialiased Abs 32f64f: horizontal-distance rule
};

template <typename Src, typename OutT, int Model, class Call>
void run_signed_dt(const char *id, Call call) {
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
  static const Cfg cfgs[] = {{64, 64}, {128, 96}};
  const Src cutoff = (Src)0.5;
  const OutT shifts[2][2] = {{(OutT)0, (OutT)0}, {(OutT)0.25, (OutT)0.5}};
  // Closed regions only: a rectangle on the thirds and a centred island.
  const int nPat = 2;
  // The reference's antialiased Abs 32f64f interior values are not a distance
  // field once the sub-pixel shifts are non-zero (measured: they collapse to
  // the one-pixel band value), so that flavour is exercised with zero shifts.
  const int nShift = Model == kAaAbs ? 1: 2;
  bool failed = false;
  for (const Cfg &cf: cfgs) {
    for (int pat = 0; pat < nPat && !failed; ++pat) {
      for (int si = 0; si < nShift && !failed; ++si) {
        const int W = cf.w, H = cf.h;
        Buf<Src> *s = new Buf<Src>(W, H, 1, 0, 0);
        for (int y = 0; y < H; ++y)
          for (int x = 0; x < W; ++x) {
            Src v = (Src)0;
            if (pat == 0)
              v = (y >= H / 3 && y < 2 * H / 3 && x >= W / 3 && x < 2 * W / 3) ? (Src)1: (Src)0;
            else
              v = (x >= W / 2 - 8 && x < W / 2 + 8 && y >= H / 2 - 8 && y < H / 2 + 8) ? (Src)1: (Src)0;
            s->hptr()[(size_t)y * (s->step / (int)sizeof(Src)) + (size_t)x] = v;
          }
        s->dev.copyFromHost(s->host);
        Buf<OutT> *d = new Buf<OutT>(W, H, 1, 0, 0x5A);
        d->dev.copyFromHost(d->host);
        const std::vector<uint8_t> pre = d->host;
        size_t need = 0;
        nppiSignedDistanceTransformPBAGetBufferSize(NppiSize{W, H}, &need);
        size_t need64 = 0;
        nppiSignedDistanceTransformPBAGet64fBufferSize(NppiSize{W, H}, &need64);
        size_t aneed = 0, aneed2 = 0;
        nppiSignedDistanceTransformPBAGetAntialiasingBufferSize(NppiSize{W, H}, &aneed);
        nppiDistanceTransformPBAGetAntialiasingBufferSize(NppiSize{W, H}, &aneed2);
        if (aneed2 > aneed) aneed = aneed2;
        npp_functional_test::DeviceMemory<Npp8u> buf(need > need64 ? need: need64);
        npp_functional_test::DeviceMemory<Npp8u> abuf(aneed ? aneed: 1);
        // NPP's signed kernels write every output unconditionally, so the
        // three Voronoi buffers must be real device memory even though their
        // contents are not compared.
        const int vstep = (int)align_step((size_t)(2 * W) * sizeof(Npp16s), 16);
        npp_functional_test::DeviceMemory<Npp16s> vor((size_t)vstep * H);
        npp_functional_test::DeviceMemory<Npp16s> idx((size_t)vstep * H);
        npp_functional_test::DeviceMemory<Npp16s> man((size_t)vstep * H);
        const OutT sx = shifts[si][0], sy = shifts[si][1];
        // The antialiasing workspace is mandatory only for the Abs 32f64f
        // entry point; the other flavours are exercised in their non-AA mode
        // (passing the workspace to them has no effect on the reference).
        Npp8u *aa = Model == kAaAbs ? abuf.get(): nullptr;
        npp_diff::clear_cuda_error();
        const NppStatus st = call(reinterpret_cast<Src *>(s->dptr()), s->step, cutoff, sx, sy,
                                  vor.get(), vstep, idx.get(), vstep, man.get(), vstep,
                                  reinterpret_cast<OutT *>(d->dptr()), d->step, NppiSize{W, H},
                                  buf.get(), aa, ctx);
        if (st != NPP_NO_ERROR) {
          ++cnt.skipped;
          if (st == NPP_NOT_IMPLEMENTED_ERROR) ++cnt.notimpl;
          delete s;
          delete d;
          continue;
        }
        ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess) << "[" << id << "] sync";
        ++cnt.executed;
        const int srow = s->step / (int)sizeof(Src);
        auto srcv = [&](int x, int y) { return as_double(s->hptr()[(size_t)y * srow + (size_t)x]); };
        auto isInside = [&](int x, int y) { return srcv(x, y) >= 1.0; };
        // Site classification (outside sites are src <= 0 or src < cutoff).
        std::vector<int> in, out;
        int minCol = INT_MAX, maxCol = -1;
        for (int y = 0; y < H; ++y)
          for (int x = 0; x < W; ++x) {
            const double v = srcv(x, y);
            if (v >= 1.0) {
              in.push_back(y * W + x);
              minCol = std::min(minCol, x);
              maxCol = std::max(maxCol, x);
            } else if (v <= 0.0 || v < (double)cutoff) {
              out.push_back(y * W + x);
            }
          }
        // Nearest site with the reference's PBA tie-break (smaller x, then
        // larger y).  An empty list yields the (-1,0) dummy site used by the
        // reference when one class is absent.
        auto nearest = [&](const std::vector<int> &list, int x, int y, int &bx, int &by) {
          long long best = -1;
          bx = -1;
          by = 0;
          for (int idx: list) {
            const int sx2 = idx % W, sy2 = idx / W;
            const long long dx = x - sx2, dy = y - sy2;
            const long long d2 = dx * dx + dy * dy;
            const bool better = best < 0 || d2 < best || (d2 == best && (sx2 < bx || (sx2 == bx && sy2 > by)));
            if (better) {
              best = d2;
              bx = sx2;
              by = sy2;
            }
          }
        };
        auto folded = [&](int x, int y, int sx2, int sy2) -> double {
          const double a = std::fabs((double)(x - sx2)) + (double)sx;
          const double b = std::fabs((double)(y - sy2)) + (double)sy;
          const double major = a >= b ? a: b;
          const double minor = a >= b ? b: a;
          const double C = 30.0 + (double)cutoff;
          const double dd = major - C;
          return std::sqrt(dd * dd + minor * minor);
        };
        auto euclid = [&](int x, int y, int sx2, int sy2) -> double {
          const double a = std::fabs((double)(x - sx2)) + (double)sx;
          const double b = std::fabs((double)(y - sy2)) + (double)sy;
          return std::sqrt(a * a + b * b);
        };
        std::vector<uint8_t> want = pre;
        const bool noInside = in.empty();
        const bool noOutside = out.empty();
        for (int y = 0; y < H; ++y) {
          for (int x = 0; x < W; ++x) {
            int nx, ny;
            double value = 0.0;
            if (Model == kSignedFolded) {
              if (isInside(x, y)) {
                if (noOutside) {
                  value = folded(x, y, -1, 0);
                } else {
                  nearest(out, x, y, nx, ny);
                  value = folded(x, y, nx, ny);
                }
              } else if (!noInside) {
                nearest(in, x, y, nx, ny);
                value = -folded(x, y, nx, ny);
              }
            } else if (Model == kAbsFolded) {
              if (noInside) {
                value = 0.0;
              } else if (noOutside) {
                // Measured: the Abs 32f reference falls back to the dummy
                // site when there are no outside pixels.
                value = folded(x, y, -1, 0);
              } else {
                nearest(in, x, y, nx, ny);
                value = folded(x, y, nx, ny);
              }
            } else if (Model == kEuclidOpposite) {
              if (noInside) {
                value = 0.0;
              } else if (isInside(x, y)) {
                if (noOutside) {
                  value = 1.0;  // measured constant on an all-inside image
                } else {
                  nearest(out, x, y, nx, ny);
                  value = euclid(x, y, nx, ny);
                }
              } else {
                nearest(in, x, y, nx, ny);
                value = euclid(x, y, nx, ny);
              }
            } else {
              // kAaAbs: horizontal distance to the inside column span, used
              // for both axes.
              int d = 0;
              if (!noInside) d = x < minCol ? minCol - x: (x > maxCol ? x - maxCol: 0);
              const double a = (double)d + (double)sx;
              const double b = (double)d + (double)sy;
              value = std::sqrt(a * a + b * b);
            }
            const OutT outv = (OutT)value;
            const size_t off = (size_t)y * d->step + (size_t)x * sizeof(OutT);
            std::memcpy(want.data() + off, &outv, sizeof(OutT));
          }
        }
        d->dev.copyToHost(d->host);
        const GenCase dc{W, H, pat, 0, 0, kSmall, si};
        const bool ok = rotate_check<OutT, 1, 1>(id, dc, NPPI_INTER_NN, pre, d->host, want, d->step, {},
                                                 cnt, first);
        delete s;
        delete d;
        if (!ok || cnt.mismatched >= 8) failed = true;
      }
    }
  }
  finish(id, cnt, first);
}

}  // namespace signeddt
}  // namespace num
