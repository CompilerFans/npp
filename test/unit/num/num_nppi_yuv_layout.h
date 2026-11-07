// CPU-reference output coverage for the YUV/YCbCr sampling-format and layout
// conversions (pure byte permutation / decimation, no colour arithmetic).
//
// Semantics measured on the reference (NVIDIA NPP 13.4), matching the in-tree
// c3yuv executor:
//   * packed C2 rows: YCbCr422 = [Y0 Cb Y1 Cr], YCrCb422 = [Y0 Cr Y1 Cb],
//     CbYCr422 = [Cb Y0 Cr Y1];
//   * P3 = three planes (Y, Cb, Cr), P2 = Y plane + interleaved (Cb,Cr);
//   * subsampling is decimation, never averaging; 4:2:0 keeps even source
//     rows, 4:1:1 keeps every fourth luma column;
//   * the 4:1:1 -> 4:2:0 destination quirk interleaves the two source chroma
//     rows into one destination row at byte column 2*x with x = 2c + (y&1);
//   * luma extents: groups of 4 pixels for 4:1:1 destinations, 2 otherwise;
//     4:2:0 destinations round the height down to even.
#pragma once

#include "num_nppi_geom.h"

namespace num {
namespace yuvlayout {

// Ordering follows the executor switch tables.
enum class Kind {
  C2P3,      // packed 2-channel source -> three planes
  C2P2,      // packed source -> Y + interleaved CbCr
  P3C2,      // three planes -> packed source
  P2C2_411,  // Y + interleaved 4:1:1 CbCr -> packed 4:2:2
  P2P3_411,  // Y + interleaved 4:1:1 CbCr -> planar 4:1:1
  P2P3_422,  // Y + interleaved 4:1:1 CbCr -> planar 4:2:2 (replicated)
  C2REORDER, // packed source -> packed source
  P3RESAMPLE,// planar chroma resampling
  P3P2,      // planar -> Y + interleaved CbCr
  P2P3_420,  // Y + interleaved 4:1:1 CbCr -> planar 4:2:0 (row quirk)
};

inline int yOff(int x, int layout) { return ((x & 1) << 1) + (layout == 2 ? 1: 0); }
inline int cbOff(int layout) { return layout == 0 ? 1: (layout == 1 ? 3: 0); }
inline int crOff(int layout) { return layout == 0 ? 3: (layout == 1 ? 1: 2); }

// a0..a2 carry the executor's integer arguments (see the emitter):
//   C2P3: layoutSrc, px, py, swapDst
//   C2P2: layoutSrc, px, py
//   P3C2: swapSrc, srcMode, swapDst
//   P2C2_411: swapDst
//   P2P3_*: (unused) / P2P3_420: swapDst
//   C2REORDER: layoutSrc, layoutDst
//   P3RESAMPLE: swapSrc, swapDst, mode
//   P3P2: swapSrc, mode
template <class Call>
void run_layout(const char *id, Kind kind, int a0, int a1, int a2, int a3, Call call) {
  NppStreamContext ctx{};
  if (nppGetStreamContext(&ctx) != NPP_NO_ERROR) {
    GTEST_SKIP() << "[" << id << "] no stream context";
    return;
  }
  RunCounts cnt;
  std::string first;
  struct Cfg { int w, h; };
  static const Cfg cfgs[] = {{4, 4},   {8, 6},   {16, 12}, {32, 24},  {64, 48},
                             {128, 96}, {6, 4},   {10, 6},  {33, 17},  {48, 32}};
  for (const Cfg &cf: cfgs) {
    const int w = cf.w, h = cf.h;
    const uint64_t seed = case_seed(id, w, h, 0, 0, 0, 0);
    const bool srcC2 = kind == Kind::C2P3 || kind == Kind::C2P2 || kind == Kind::C2REORDER;
    const bool srcP2 = kind == Kind::P2C2_411 || kind == Kind::P2P3_411 || kind == Kind::P2P3_422 ||
                       kind == Kind::P2P3_420;
    const bool dstC2 = kind == Kind::P3C2 || kind == Kind::P2C2_411 || kind == Kind::C2REORDER;
    const bool dstP2 = kind == Kind::C2P2 || kind == Kind::P3P2;
    // Sources.
    std::vector<Buf<Npp8u> *> srcs;
    if (srcC2) {
      Buf<Npp8u> *s = new Buf<Npp8u>(w, h, 2, 0, 0);
      fill_regime(*s, 2, seed, kWide, 0, false, false, false);
      s->dev.copyFromHost(s->host);
      srcs.push_back(s);
    } else if (srcP2) {
      Buf<Npp8u> *y = new Buf<Npp8u>(w, h, 1, 0, 0);
      fill_regime(*y, 1, seed, kWide, 0, false, false, false);
      y->dev.copyFromHost(y->host);
      srcs.push_back(y);
      Buf<Npp8u> *c = new Buf<Npp8u>(std::max(1, w / 2), h, 2, 0, 0);
      fill_regime(*c, 2, seed ^ 0x9E3779B9ull, kWide, 0, false, false, false);
      c->dev.copyFromHost(c->host);
      srcs.push_back(c);
    } else {
      planar_fill_src(srcs, 3, w, h, 0, 0, 0, seed, kWide);
    }
    // Destinations (allocated at full size; the reference only writes the
    // mapped region and everything else must keep the prefill).
    std::vector<Buf<Npp8u> *> dsts;
    std::vector<std::vector<uint8_t>> pre;
    if (dstC2) {
      Buf<Npp8u> *d = new Buf<Npp8u>(w, h, 2, 0, 0x5A);
      d->dev.copyFromHost(d->host);
      dsts.push_back(d);
      pre.push_back(d->host);
    } else if (dstP2) {
      Buf<Npp8u> *dy = new Buf<Npp8u>(w, h, 1, 0, 0x5A);
      dy->dev.copyFromHost(dy->host);
      dsts.push_back(dy);
      pre.push_back(dy->host);
      Buf<Npp8u> *dc = new Buf<Npp8u>(std::max(1, w / 2), h, 2, 0, 0x5A);
      dc->dev.copyFromHost(dc->host);
      dsts.push_back(dc);
      pre.push_back(dc->host);
    } else {
      for (int p = 0; p < 3; ++p) {
        Buf<Npp8u> *d = new Buf<Npp8u>(w, h, 1, 0, 0x5A);
        d->dev.copyFromHost(d->host);
        dsts.push_back(d);
        pre.push_back(d->host);
      }
    }
    const Npp8u *sp[4] = {nullptr, nullptr, nullptr, nullptr};
    const Npp8u *ps = nullptr;
    Npp8u *dp[4] = {nullptr, nullptr, nullptr, nullptr};
    Npp8u *pd = nullptr;
    int ssa[4] = {0, 0, 0, 0};
    int dsa[4] = {0, 0, 0, 0};
    if (srcC2) {
      ps = srcs[0]->dptr();
    } else {
      for (size_t p = 0; p < srcs.size(); ++p) {
        sp[p] = srcs[p]->dptr();
        ssa[p] = srcs[p]->step;
      }
    }
    if (dstC2) {
      pd = dsts[0]->dptr();
    } else {
      for (size_t p = 0; p < dsts.size(); ++p) {
        dp[p] = dsts[p]->dptr();
        dsa[p] = dsts[p]->step;
      }
    }
    const int ss = srcs[0]->step;
    const int ds = dsts[0]->step;
    npp_diff::clear_cuda_error();
    NppStatus st = call(sp, ps, ss, ssa, dp, pd, ds, dsa, NppiSize{w, h}, ctx);
    if (st != NPP_NO_ERROR) {
      ++cnt.skipped;
      if (st == NPP_NOT_IMPLEMENTED_ERROR) ++cnt.notimpl;
      for (auto *p: srcs) delete p;
      for (auto *p: dsts) delete p;
      continue;
    }
    const cudaError_t se = cudaDeviceSynchronize();
    ASSERT_EQ(se, cudaSuccess) << "[" << id << "] " << w << "x" << h << " sync: " << cudaGetErrorString(se);
    ++cnt.executed;

    auto srcAt = [&](int p, int x, int y) -> int {
      const Buf<Npp8u> *b = srcs[(size_t)p];
      return b->hptr()[(size_t)y * b->step + (size_t)x];
    };
    auto c2At = [&](int x, int y, int off) -> int {  // x is a pixel column
      return srcs[0]->hptr()[(size_t)y * srcs[0]->step + (size_t)(x >> 1) * 4 + (size_t)off];
    };
    auto c2Pair = [&](int pair, int y, int off) -> int {  // pair index
      return srcs[0]->hptr()[(size_t)y * srcs[0]->step + (size_t)pair * 4 + (size_t)off];
    };
    std::vector<std::vector<uint8_t>> want = pre;
    auto wr = [&](int plane, int x, int y, int v) {
      const Buf<Npp8u> *d = dsts[(size_t)plane];
      want[(size_t)plane][(size_t)y * d->step + (size_t)x] = (uint8_t)v;
    };
    if (kind == Kind::C2P3 || kind == Kind::C2P2) {
      const int layoutSrc = a0, px = a1, py = a2, swapDst = a3;
      const int yh = (py == 2) ? (h & ~1): h;
      const int yw = w & ~((px == 2) ? 3: 1);
      for (int y = 0; y < yh; ++y)
        for (int x = 0; x < yw; ++x) wr(0, x, y, c2At(x, y, yOff(x, layoutSrc)));
      const int cw = (px == 2) ? (w >> 2): (w >> 1);
      const int ch = (py == 2) ? (h >> 1): h;
      for (int y = 0; y < ch; ++y)
        for (int x = 0; x < cw; ++x) {
          const int sx = x * px, sy = y * py;  // packed pair indices
          const int cb = c2Pair(sx, sy, cbOff(layoutSrc));
          const int cr = c2Pair(sx, sy, crOff(layoutSrc));
          if (kind == Kind::C2P3) {
            wr(swapDst ? 2: 1, x, y, cb);
            wr(swapDst ? 1: 2, x, y, cr);
          } else {
            const size_t off = (size_t)y * dsts[1]->step + (size_t)x * 2;
            want[1][off] = (uint8_t)cb;
            want[1][off + 1] = (uint8_t)cr;
          }
        }
    } else if (kind == Kind::P3C2 || kind == Kind::P2C2_411) {
      const bool fromP2 = kind == Kind::P2C2_411;
      int swapSrc = 0, srcMode = 0, swapDst = 0;
      if (fromP2) swapDst = a0;
      else { swapSrc = a0; srcMode = a1; swapDst = a2; }
      const int hh = (!fromP2 && srcMode == 2) ? (h & ~1): h;
      const int pairs = w >> 1;
      const int sCb = swapSrc ? 2: 1, sCr = swapSrc ? 1: 2;
      for (int y = 0; y < hh; ++y)
        for (int p = 0; p < pairs; ++p) {
          const int col = fromP2 ? (p >> 1): (srcMode == 1 ? (p >> 1): p);
          const int row = fromP2 ? y: (srcMode == 2 ? (y >> 1): y);
          int cb, cr;
          if (fromP2) {
            cb = srcAt(1, 2 * col, y);
            cr = srcAt(1, 2 * col + 1, y);
          } else {
            cb = srcAt(sCb, col, row);
            cr = srcAt(sCr, col, row);
          }
          const size_t base = (size_t)y * dsts[0]->step + (size_t)p * 4;
          want[0][base + (size_t)yOff(2 * p, swapDst)] = (uint8_t)srcAt(0, 2 * p, y);
          want[0][base + (size_t)yOff(2 * p + 1, swapDst)] = (uint8_t)srcAt(0, 2 * p + 1, y);
          want[0][base + (size_t)cbOff(swapDst)] = (uint8_t)cb;
          want[0][base + (size_t)crOff(swapDst)] = (uint8_t)cr;
        }
    } else if (kind == Kind::P2P3_411 || kind == Kind::P2P3_422 || kind == Kind::P2P3_420) {
      int yw = w, yh = h;
      if (kind == Kind::P2P3_411) yw = w & ~3;
      else if (kind == Kind::P2P3_422) yw = w & ~1;
      else { yw = w & ~1; yh = h & ~1; }
      for (int y = 0; y < yh; ++y)
        for (int x = 0; x < yw; ++x) wr(0, x, y, srcAt(0, x, y));
      if (kind == Kind::P2P3_411) {
        for (int y = 0; y < h; ++y)
          for (int x = 0; x < (w >> 2); ++x) {
            wr(1, x, y, srcAt(1, 2 * x, y));
            wr(2, x, y, srcAt(1, 2 * x + 1, y));
          }
      } else if (kind == Kind::P2P3_422) {
        for (int y = 0; y < h; ++y)
          for (int x = 0; x < (w >> 1); ++x) {
            wr(1, x, y, srcAt(1, 2 * (x >> 1), y));
            wr(2, x, y, srcAt(1, 2 * (x >> 1) + 1, y));
          }
      } else {
        const int swapDst = a0;
        for (int j = 0; j < (h & ~1) / 2; ++j)
          for (int x = 0; x < (w >> 1); ++x) {
            const int sy = 2 * j + (x & 1);
            const int c = x >> 1;
            const int cb = srcAt(1, 2 * c, sy);
            const int cr = srcAt(1, 2 * c + 1, sy);
            wr(swapDst ? 2: 1, x, j, cb);
            wr(swapDst ? 1: 2, x, j, cr);
          }
      }
    } else if (kind == Kind::C2REORDER) {
      const int layoutSrc = a0, layoutDst = a1;
      for (int y = 0; y < h; ++y)
        for (int p = 0; p < (w >> 1); ++p) {
          const size_t base = (size_t)y * dsts[0]->step + (size_t)p * 4;
          want[0][base + (size_t)yOff(2 * p, layoutDst)] = (uint8_t)c2At(2 * p, y, yOff(2 * p, layoutSrc));
          want[0][base + (size_t)yOff(2 * p + 1, layoutDst)] = (uint8_t)c2At(2 * p + 1, y, yOff(2 * p + 1, layoutSrc));
          want[0][base + (size_t)cbOff(layoutDst)] = (uint8_t)c2At(2 * p, y, cbOff(layoutSrc));
          want[0][base + (size_t)crOff(layoutDst)] = (uint8_t)c2At(2 * p, y, crOff(layoutSrc));
        }
    } else if (kind == Kind::P3RESAMPLE) {
      const int swapSrc = a0, swapDst = a1, mode = a2;
      const int yh = (mode == 2 || mode == 3 || mode == 4) ? (h & ~1): h;
      const int yw = w & ~((mode == 1) ? 3: 1);
      for (int y = 0; y < yh; ++y)
        for (int x = 0; x < yw; ++x) wr(0, x, y, srcAt(0, x, y));
      int cw = w >> 1;
      int ch = h;
      if (mode == 1) cw = w >> 2;
      else if (mode == 2 || mode == 4) ch = (h & ~1) >> 1;
      else if (mode == 3) ch = h & ~1;
      const int sCb = swapSrc ? 2: 1, sCr = swapSrc ? 1: 2;
      const int dCb = swapDst ? 2: 1, dCr = swapDst ? 1: 2;
      for (int y = 0; y < ch; ++y)
        for (int x = 0; x < cw; ++x) {
          int sx = x, sy = y;
          if (mode == 0) sx = x >> 1;
          else if (mode == 1) sx = 2 * x;
          else if (mode == 2) sy = 2 * y;
          else if (mode == 3) sy = y >> 1;
          else if (mode == 4) { sx = x >> 1; sy = 2 * y + (x & 1); }
          wr(dCb, x, y, srcAt(sCb, sx, sy));
          wr(dCr, x, y, srcAt(sCr, sx, sy));
        }
    } else {  // P3P2
      const int swapSrc = a0, mode = a1;
      const int yh = (mode == 1 || mode == 3) ? (h & ~1): h;
      const int yw = w & ~((mode == 0 || mode == 2) ? 3: 1);
      for (int y = 0; y < yh; ++y)
        for (int x = 0; x < yw; ++x) wr(0, x, y, srcAt(0, x, y));
      int cw = w >> 2;
      int ch = h;
      if (mode == 1 || mode == 3) { cw = w >> 1; ch = (h & ~1) >> 1; }
      const int sCb = swapSrc ? 2: 1, sCr = swapSrc ? 1: 2;
      for (int y = 0; y < ch; ++y)
        for (int x = 0; x < cw; ++x) {
          if (mode == 3) {
            if (x >= (w >> 1) || y >= ((h & ~1) >> 1)) continue;
            const int c = x >> 1;
            const int sy = 2 * y + (x & 1);
            const size_t off = (size_t)y * dsts[1]->step + (size_t)(2 * x);
            want[1][off] = (uint8_t)srcAt(sCb, c, sy);
            want[1][off + 1] = (uint8_t)srcAt(sCr, c, sy);
            continue;
          }
          int sx = x, sy = y;
          if (mode == 0) sx = 2 * x;
          else if (mode == 1) sy = 2 * y;
          const size_t off = (size_t)y * dsts[1]->step + (size_t)x * 2;
          want[1][off] = (uint8_t)srcAt(sCb, sx, sy);
          want[1][off + 1] = (uint8_t)srcAt(sCr, sx, sy);
        }
    }
    bool failed = false;
    for (size_t p = 0; p < dsts.size() && !failed; ++p) {
      dsts[p]->dev.copyToHost(dsts[p]->host);
      const GenCase dc{(int)dsts[p]->bw, (int)dsts[p]->bh, 0, 0, 0, kSmall, 0};
      const int chans = (dstC2 ? (p == 0): (dstP2 && p == 1)) ? 2: 1;
      const bool ok = (chans == 2)
                          ? rotate_check<Npp8u, 2, 2>(id, dc, NPPI_INTER_NN, pre[p], dsts[p]->host, want[p],
                                                      dsts[p]->step, std::vector<uint8_t>(), cnt, first): rotate_check<Npp8u, 1, 1>(id, dc, NPPI_INTER_NN, pre[p], dsts[p]->host, want[p],
                                                      dsts[p]->step, std::vector<uint8_t>(), cnt, first);
      if (!ok) failed = true;
    }
    for (auto *p: srcs) delete p;
    for (auto *p: dsts) delete p;
    if (failed || cnt.mismatched >= 8) break;
  }
  finish(id, cnt, first);
}

}  // namespace yuvlayout
}  // namespace num
