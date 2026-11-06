// CPU-golden coverage for the 4:2:0 layout/sampling conversions
// (nppi_ycbcr420_layout.*): planar <-> semi-planar 4:2:0, 4:2:0 planar ->
// 4:2:2 planar (row replication), semi-planar 4:2:0 -> packed 4:2:2, and the
// 4:2:0 -> 4:1:1 chroma decimation with the measured row-interleaving quirk
// (destination row y reads source chroma row y/2 at column 2x + (y & 1)).
//
// All conversions are pure byte moves: the luma plane is copied verbatim and
// the chroma planes are interleaved/split/replicated.  These formats only
// make sense for even dimensions, so the cases use even width and height.
#pragma once

#include "num_nppi_geom.h"

namespace num {
namespace ycbcr420layout {

enum class Kind {
  P3P2_420,      // planar Y/Cb/Cr -> planar Y + interleaved CbCr
  P2P3_420,      // planar Y + interleaved CbCr -> planar
  P3P3_422,      // planar 4:2:0 -> planar 4:2:2 (chroma rows replicated)
  P2P3_422,      // semi-planar 4:2:0 -> planar 4:2:2
  P2C2_422,      // semi-planar 4:2:0 -> packed 4:2:2 (packedMode in a0)
  YCrCb2CbYCr,   // planar YCrCb 4:2:0 -> packed CbYCr 4:2:2
  P3P2_411,      // planar YCrCb 4:2:0 -> semi-planar 4:1:1
  P2P3_411,      // semi-planar 4:2:0 -> planar 4:1:1
};

template <class Call>
void run_layout420(const char *id, Kind kind, int a0, Call call) {
  NppStreamContext ctx{};
  if (nppGetStreamContext(&ctx) != NPP_NO_ERROR) {
    GTEST_SKIP() << "[" << id << "] no stream context";
    return;
  }
  RunCounts cnt;
  std::string first;
  struct Cfg { int w, h; };
  static const Cfg cfgs[] = {{4, 4}, {8, 6}, {16, 12}, {32, 24}, {64, 48}, {128, 96}};
  for (const Cfg &cf : cfgs) {
    const int w = cf.w, h = cf.h;
    const bool srcP3 = kind == Kind::P3P2_420 || kind == Kind::P3P3_422 || kind == Kind::YCrCb2CbYCr ||
                       kind == Kind::P3P2_411;
    const bool srcP2 = kind == Kind::P2P3_420 || kind == Kind::P2P3_422 || kind == Kind::P2C2_422 ||
                       kind == Kind::P2P3_411;
    const bool dstP2 = kind == Kind::P3P2_420 || kind == Kind::P3P2_411;
    const bool dstC2 = kind == Kind::P2C2_422 || kind == Kind::YCrCb2CbYCr;
    const uint64_t seed = case_seed(id, w, h, 0, 0, 0, 0);
    std::vector<Buf<Npp8u> *> srcs;
    const int pw = w + 16, ph = h + 4;  // guard margins for the reference kernels
    if (srcP3) {
      planar_fill_src(srcs, 3, pw, ph, 0, 0, 0, seed, kWide);
    } else if (srcP2) {
      Buf<Npp8u> *y = new Buf<Npp8u>(pw, ph, 1, 0, 0);
      fill_regime(*y, 1, seed, kWide, 0, false, false, false);
      y->dev.copyFromHost(y->host);
      srcs.push_back(y);
      Buf<Npp8u> *c = new Buf<Npp8u>(pw / 2, ph, 2, 0, 0);
      fill_regime(*c, 2, seed ^ 0x9E3779B9ull, kWide, 0, false, false, false);
      c->dev.copyFromHost(c->host);
      srcs.push_back(c);
    }
    std::vector<Buf<Npp8u> *> dsts;
    std::vector<std::vector<uint8_t>> pre;
    if (dstP2) {
      Buf<Npp8u> *dy = new Buf<Npp8u>(pw, ph, 1, 0, 0x5A);
      dy->dev.copyFromHost(dy->host);
      dsts.push_back(dy);
      pre.push_back(dy->host);
      Buf<Npp8u> *dc = new Buf<Npp8u>(pw / 2, ph, 2, 0, 0x5A);
      dc->dev.copyFromHost(dc->host);
      dsts.push_back(dc);
      pre.push_back(dc->host);
    } else if (dstC2) {
      Buf<Npp8u> *d = new Buf<Npp8u>(pw, ph, 2, 0, 0x5A);
      d->dev.copyFromHost(d->host);
      dsts.push_back(d);
      pre.push_back(d->host);
    } else {
      for (int p = 0; p < 3; ++p) {
        Buf<Npp8u> *d = new Buf<Npp8u>(pw, ph, 1, 0, 0x5A);
        d->dev.copyFromHost(d->host);
        dsts.push_back(d);
        pre.push_back(d->host);
      }
    }
    const Npp8u *sp[4] = {nullptr, nullptr, nullptr, nullptr};
    Npp8u *dp[4] = {nullptr, nullptr, nullptr, nullptr};
    int ssa[4] = {0, 0, 0, 0};
    int dsa[4] = {0, 0, 0, 0};
    for (size_t p = 0; p < srcs.size(); ++p) {
      sp[p] = srcs[p]->dptr();
      ssa[p] = srcs[p]->step;
    }
    for (size_t p = 0; p < dsts.size(); ++p) {
      dp[p] = dsts[p]->dptr();
      dsa[p] = dsts[p]->step;
    }
    const int ss = srcs[0]->step;
    const int ds = dsts[0]->step;
    npp_diff::clear_cuda_error();
    const NppStatus st = call(sp, ss, ssa, dp, ds, dsa, NppiSize{w, h}, ctx);
    if (st != NPP_NO_ERROR && st != NPP_DOUBLE_SIZE_WARNING) {
      if (getenv("NUM_LAYOUT_DEBUG")) std::fprintf(stderr, "[%s] %dx%d status=%d\n", id, w, h, (int)st);
      ++cnt.skipped;
      if (st == NPP_NOT_IMPLEMENTED_ERROR) ++cnt.notimpl;
      for (auto *p : srcs) delete p;
      for (auto *p : dsts) delete p;
      continue;
    }
    const cudaError_t se = cudaDeviceSynchronize();
    ASSERT_EQ(se, cudaSuccess) << "[" << id << "] " << w << "x" << h << " sync: " << cudaGetErrorString(se);
    ++cnt.executed;

    auto srcY = [&](int x, int y) { return (int)srcs[0]->hptr()[(size_t)y * srcs[0]->step + (size_t)x]; };
    auto atP3 = [&](int p, int x, int y) {
      return (int)srcs[(size_t)p]->hptr()[(size_t)y * srcs[(size_t)p]->step + (size_t)x];
    };
    auto atP2 = [&](int c, int pairX, int y) {
      return (int)srcs[1]->hptr()[(size_t)y * srcs[1]->step + (size_t)pairX * 2 + (size_t)c];
    };
    std::vector<std::vector<uint8_t>> want = pre;
    auto wrP = [&](int plane, int x, int y, int v) {
      want[(size_t)plane][(size_t)y * dsts[(size_t)plane]->step + (size_t)x] = (uint8_t)v;
    };
    if (kind == Kind::P3P2_420 || kind == Kind::P2P3_420) {
      for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) wrP(0, x, y, srcY(x, y));
      const int cw = w / 2, ch = h / 2;
      for (int y = 0; y < ch; ++y)
        for (int x = 0; x < cw; ++x) {
          if (kind == Kind::P3P2_420) {
            want[1][(size_t)y * dsts[1]->step + (size_t)x * 2] = (uint8_t)atP3(a0 ? 2 : 1, x, y);
            want[1][(size_t)y * dsts[1]->step + (size_t)x * 2 + 1] = (uint8_t)atP3(a0 ? 1 : 2, x, y);
          } else {
            wrP(1, x, y, atP2(a0 ? 1 : 0, x, y));
            wrP(2, x, y, atP2(a0 ? 0 : 1, x, y));
          }
        }
    } else if (kind == Kind::P3P3_422 || kind == Kind::P2P3_422) {
      for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) wrP(0, x, y, srcY(x, y));
      const int cw = w / 2;
      for (int y = 0; y < h; ++y)
        for (int x = 0; x < cw; ++x) {
          if (kind == Kind::P3P3_422) {
            wrP(1, x, y, atP3(1, x, y >> 1));
            wrP(2, x, y, atP3(2, x, y >> 1));
          } else {
            wrP(1, x, y, atP2(0, x, y >> 1));
            wrP(2, x, y, atP2(1, x, y >> 1));
          }
        }
    } else if (kind == Kind::P2C2_422 || kind == Kind::YCrCb2CbYCr) {
      const int mode = a0;
      for (int y = 0; y < h; ++y)
        for (int p = 0; p < (w >> 1); ++p) {
          const size_t base = (size_t)y * dsts[0]->step + (size_t)p * 4;
          const int y0 = srcY(2 * p, y), y1 = srcY(2 * p + 1, y);
          int cb, cr;
          if (kind == Kind::P2C2_422) {
            cb = atP2(0, p, y >> 1);
            cr = atP2(1, p, y >> 1);
            if (mode == 0) {
              want[0][base + 0] = (uint8_t)y0;
              want[0][base + 1] = (uint8_t)cb;
              want[0][base + 2] = (uint8_t)y1;
              want[0][base + 3] = (uint8_t)cr;
            } else {
              want[0][base + 0] = (uint8_t)cb;
              want[0][base + 1] = (uint8_t)y0;
              want[0][base + 2] = (uint8_t)cr;
              want[0][base + 3] = (uint8_t)y1;
            }
          } else {
            // source planes are (Y, Cr, Cb)
            cb = atP3(2, p, y >> 1);
            cr = atP3(1, p, y >> 1);
            want[0][base + 0] = (uint8_t)cb;
            want[0][base + 1] = (uint8_t)y0;
            want[0][base + 2] = (uint8_t)cr;
            want[0][base + 3] = (uint8_t)y1;
          }
        }
    } else {  // P3P2_411 / P2P3_411
      for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) wrP(0, x, y, srcY(x, y));
      const int cw = w / 4;
      for (int y = 0; y < h; ++y)
        for (int x = 0; x < cw; ++x) {
          const int sy = y >> 1;
          const int sx = 2 * x + (y & 1);
          if (kind == Kind::P3P2_411) {
            // a0=1: the source planes are (Y, Cr, Cb) (YCrCb family)
            want[1][(size_t)y * dsts[1]->step + (size_t)x * 2] = (uint8_t)atP3(a0 ? 2 : 1, sx, sy);
            want[1][(size_t)y * dsts[1]->step + (size_t)x * 2 + 1] = (uint8_t)atP3(a0 ? 1 : 2, sx, sy);
          } else {
            wrP(1, x, y, atP2(0, sx, sy));
            wrP(2, x, y, atP2(1, sx, sy));
          }
        }
    }
    bool failed = false;
    for (size_t p = 0; p < dsts.size() && !failed; ++p) {
      dsts[p]->dev.copyToHost(dsts[p]->host);
      const GenCase dc{w, h, 0, 0, 0, kSmall, 0};
      const bool twoCh = (dstP2 && p == 1) || (dstC2 && p == 0);
      const bool ok = twoCh ? rotate_check<Npp8u, 2, 2>(id, dc, NPPI_INTER_NN, pre[p], dsts[p]->host, want[p],
                                                         dsts[p]->step, std::vector<uint8_t>(), cnt, first)
                            : rotate_check<Npp8u, 1, 1>(id, dc, NPPI_INTER_NN, pre[p], dsts[p]->host, want[p],
                                                        dsts[p]->step, std::vector<uint8_t>(), cnt, first);
      if (!ok) failed = true;
    }
    for (auto *p : srcs) delete p;
    for (auto *p : dsts) delete p;
    if (failed || cnt.mismatched >= 8) break;
  }
  finish(id, cnt, first);
}

}  // namespace ycbcr420layout
}  // namespace num
