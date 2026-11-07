// CPU-reference output coverage for the ColorTwist variants of the YUV <-> RGB
// conversions (8u and 16u).  The user 3x4 matrix is applied directly to the
// source triple; every result is rounded half up and clamped to the sample
// range, and 4:2:0 / 4:2:2 chroma is the rounded average of the twisted
// float channels of the covered samples (not an average of truncated
// samples).  These extents mirror the measured reference kernels:
//   * 4:2:0 forward: luma over (w&~1)x(h&~1), chroma (w/2)x(h/2), no warning;
//   * 4:2:0 inverse: full extent;
//   * 4:2:2 forward to packed C2 [U Y0 V Y1]: complete pairs only;
//   * 4:2:2 forward to planar and the inverse: full extent / even width.
#pragma once

#include "num_nppi_geom.h"

namespace num {
namespace yuvtwist {

template <typename T> inline int maxv() { return std::is_same_v<T, Npp8u> ? 255: 65535; }

template <typename T> inline int round_clamp(float v, int maxv) {
  const int t = (int)(v + 0.5f);
  return t < 0 ? 0: (t > maxv ? maxv: t);
}

struct Mat {
  float m[3][4];
};

inline Mat make_mat(const float *tw) {
  Mat t;
  for (int r = 0; r < 3; ++r)
    for (int c = 0; c < 4; ++c) t.m[r][c] = tw[r * 4 + c];
  return t;
}

inline float row(const Mat &t, int r, float a, float b, float c) {
  return t.m[r][0] * a + t.m[r][1] * b + t.m[r][2] * c + t.m[r][3];
}

enum class Kind {
  Fwd420P3,   // packed C3/AC4 source -> planar YUV420
  Fwd420P3P3, // planar RGB source -> planar YUV420
  Fwd422C2,   // packed source -> packed C2 [U Y0 V Y1]
  Fwd422P3,   // packed source -> planar YUV422
  Inv420C,    // planar YUV420 -> packed RGB/BGR (C3/C4)
  Inv420P3,   // planar YUV420 -> planar RGB
  Inv422C,    // planar YUV422 -> packed
  Inv422C2,   // packed YUV422 [U Y0 V Y1] -> packed RGB
  Inv422P3,   // planar YUV422 -> planar
  Fwd422P3P3, // planar RGB -> planar YUV422
};

// srcPp: packed source pitch (3/4) or 0 for a planar source.
// dstPp: packed destination pitch (3/4) for the inverse; alpha applies.
template <typename T, class Call>
void run_twist(const char *id, Kind kind, int srcPp, int dstPp, T alpha, const float *tw, Call call) {
  NppStreamContext ctx{};
  if (nppGetStreamContext(&ctx) != NPP_NO_ERROR) {
    GTEST_SKIP() << "[" << id << "] no stream context";
    return;
  }
  RunCounts cnt;
  std::string first;
  const Mat t = make_mat(tw);
  const int MAXV = maxv<T>();
  struct Cfg { int w, h; };
  static const Cfg cfgs[] = {{8, 6}, {16, 12}, {32, 24}, {64, 48}, {128, 96}, {6, 4}, {10, 6}, {48, 32}};
  for (const Cfg &cf: cfgs) {
    const int w = cf.w, h = cf.h;
    const bool fwd = kind == Kind::Fwd420P3 || kind == Kind::Fwd420P3P3 || kind == Kind::Fwd422C2 ||
                     kind == Kind::Fwd422P3 || kind == Kind::Fwd422P3P3;
    const bool srcPlanar = kind == Kind::Fwd420P3P3 || kind == Kind::Inv420C || kind == Kind::Inv420P3 ||
                           kind == Kind::Inv422C || kind == Kind::Inv422P3 || kind == Kind::Fwd422P3P3;
    const bool dstC2 = kind == Kind::Fwd422C2;
    const bool dstP3 = kind == Kind::Fwd420P3 || kind == Kind::Fwd420P3P3 || kind == Kind::Fwd422P3 ||
                       kind == Kind::Fwd422P3P3 || kind == Kind::Inv420P3 || kind == Kind::Inv422P3;
    const uint64_t seed = case_seed(id, w, h, 0, 0, 0, 0);
    std::vector<Buf<T> *> srcs;
    if (srcPlanar) {
      for (int p = 0; p < 3; ++p) {
        Buf<T> *a = new Buf<T>(w, h, 1, 0, 0);
        fill_regime(*a, 1, seed ^ (uint64_t)(p + 1), kWide, p, false, false, std::is_floating_point_v<T>);
        a->dev.copyFromHost(a->host);
        srcs.push_back(a);
      }
    } else {
      const int pp = srcPp;
      Buf<T> *a = new Buf<T>(w, h, pp, 0, 0);
      fill_regime(*a, pp, seed, kWide, 0, false, false, false);
      a->dev.copyFromHost(a->host);
      srcs.push_back(a);
    }
    std::vector<Buf<T> *> dsts;
    std::vector<std::vector<uint8_t>> pre;
    if (dstC2) {
      Buf<T> *d = new Buf<T>(w, h, 2, 0, 0x5A);
      d->dev.copyFromHost(d->host);
      dsts.push_back(d);
      pre.push_back(d->host);
    } else if (dstP3) {
      for (int p = 0; p < 3; ++p) {
        Buf<T> *d = new Buf<T>(w, h, 1, 0, 0x5A);
        d->dev.copyFromHost(d->host);
        dsts.push_back(d);
        pre.push_back(d->host);
      }
    } else {
      Buf<T> *d = new Buf<T>(w, h, dstPp, 0, 0x5A);
      d->dev.copyFromHost(d->host);
      dsts.push_back(d);
      pre.push_back(d->host);
    }
    const T *sp[4] = {nullptr, nullptr, nullptr, nullptr};
    const T *ps = nullptr;
    T *dp[4] = {nullptr, nullptr, nullptr, nullptr};
    T *pd = nullptr;
    int ssa[4] = {0, 0, 0, 0};
    int dsa[4] = {0, 0, 0, 0};
    if (!srcPlanar) {
      ps = reinterpret_cast<const T *>(srcs[0]->dptr());
    } else {
      for (int p = 0; p < 3; ++p) {
        sp[p] = reinterpret_cast<const T *>(srcs[(size_t)p]->dptr());
        ssa[p] = srcs[(size_t)p]->step;
      }
    }
    if (dstC2 || (!dstP3 && !dstC2)) {
      pd = reinterpret_cast<T *>(dsts[0]->dptr());
    }
    if (dstP3) {
      for (int p = 0; p < 3; ++p) {
        dp[p] = reinterpret_cast<T *>(dsts[(size_t)p]->dptr());
        dsa[p] = dsts[(size_t)p]->step;
      }
    }
    const int ss = srcs[0]->step;
    const int ds = dsts[0]->step;
    npp_diff::clear_cuda_error();
    const NppStatus st = call(sp, ps, ss, ssa, dp, pd, ds, dsa, NppiSize{w, h}, tw, ctx);
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

    // hptr is typed: row offsets must be in elements, not bytes.
    auto srcPx = [&](int x, int y, float &a, float &b, float &c) {
      const T *s = srcs[0]->hptr() + (size_t)y * (srcs[0]->step / (int)sizeof(T)) + (size_t)x * srcPp;
      a = (float)s[0];
      b = (float)s[1];
      c = (float)s[2];
    };
    auto srcPl = [&](int p, int x, int y) {
      return (float)srcs[(size_t)p]->hptr()[(size_t)y * (srcs[(size_t)p]->step / (int)sizeof(T)) + (size_t)x];
    };
    std::vector<std::vector<uint8_t>> want = pre;
    auto wr = [&](int plane, int x, int y, int v) {
      want[(size_t)plane][(size_t)y * dsts[(size_t)plane]->step + (size_t)x * sizeof(T)] = (uint8_t)v;
      // T is 8 or 16 bits; for 16u the value must be stored little-endian.
      if constexpr (sizeof(T) == 2) {
        want[(size_t)plane][(size_t)y * dsts[(size_t)plane]->step + (size_t)x * 2 + 1] = (uint8_t)(v >> 8);
      }
    };
    if (kind == Kind::Fwd420P3 || kind == Kind::Fwd420P3P3) {
      auto px = [&](int x, int y, float &a, float &b, float &c) {
        if (kind == Kind::Fwd420P3) srcPx(x, y, a, b, c);
        else {
          a = srcPl(0, x, y);
          b = srcPl(1, x, y);
          c = srcPl(2, x, y);
        }
      };
      const int ew = w & ~1, eh = h & ~1;
      for (int y = 0; y < eh; ++y)
        for (int x = 0; x < ew; ++x) {
          float a, b, c;
          px(x, y, a, b, c);
          wr(0, x, y, round_clamp<T>(row(t, 0, a, b, c), MAXV));
        }
      for (int cy = 0; cy < h / 2; ++cy)
        for (int cx = 0; cx < w / 2; ++cx) {
          float a[4], b[4], c[4];
          px(2 * cx, 2 * cy, a[0], b[0], c[0]);
          px(2 * cx + 1, 2 * cy, a[1], b[1], c[1]);
          px(2 * cx, 2 * cy + 1, a[2], b[2], c[2]);
          px(2 * cx + 1, 2 * cy + 1, a[3], b[3], c[3]);
          for (int r = 1; r < 3; ++r) {
            const float avg = (row(t, r, a[0], b[0], c[0]) + row(t, r, a[1], b[1], c[1]) +
                               row(t, r, a[2], b[2], c[2]) + row(t, r, a[3], b[3], c[3])) * 0.25f;
            wr(r, cx, cy, round_clamp<T>(avg, MAXV));
          }
        }
    } else if (kind == Kind::Fwd422C2 || kind == Kind::Fwd422P3 || kind == Kind::Fwd422P3P3) {
      const bool planarSrc = kind == Kind::Fwd422P3P3;
      auto px422 = [&](int x, int y, float &a, float &b, float &c) {
        if (planarSrc) {
          a = srcPl(0, x, y);
          b = srcPl(1, x, y);
          c = srcPl(2, x, y);
        } else {
          srcPx(x, y, a, b, c);
        }
      };
      for (int y = 0; y < h; ++y)
        for (int p = 0; p < (w >> 1); ++p) {
          float a0, b0, c0, a1, b1, c1;
          px422(2 * p, y, a0, b0, c0);
          px422(2 * p + 1, y, a1, b1, c1);
          const int y0 = round_clamp<T>(row(t, 0, a0, b0, c0), MAXV);
          const int y1 = round_clamp<T>(row(t, 0, a1, b1, c1), MAXV);
          const int u = round_clamp<T>((row(t, 1, a0, b0, c0) + row(t, 1, a1, b1, c1)) * 0.5f, MAXV);
          const int v = round_clamp<T>((row(t, 2, a0, b0, c0) + row(t, 2, a1, b1, c1)) * 0.5f, MAXV);
          if (kind == Kind::Fwd422C2) {
            const size_t base = (size_t)y * dsts[0]->step + (size_t)p * 4 * sizeof(T);
            if constexpr (sizeof(T) == 1) {
              want[0][base + 0] = (uint8_t)u;
              want[0][base + 1] = (uint8_t)y0;
              want[0][base + 2] = (uint8_t)v;
              want[0][base + 3] = (uint8_t)y1;
            } else {
              const uint16_t vals[4] = {(uint16_t)u, (uint16_t)y0, (uint16_t)v, (uint16_t)y1};
              for (int k = 0; k < 4; ++k) {
                want[0][base + (size_t)k * 2] = (uint8_t)(vals[k] & 0xFF);
                want[0][base + (size_t)k * 2 + 1] = (uint8_t)(vals[k] >> 8);
              }
            }
          } else {
            wr(0, 2 * p, y, y0);
            wr(0, 2 * p + 1, y, y1);
            wr(1, p, y, u);
            wr(2, p, y, v);
          }
        }
    } else if (kind == Kind::Inv422C2) {
      const int lastPair = (w >> 1) - 1;
      for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) {
          int p = x >> 1;
          if (p > lastPair) p = lastPair;
          const T *rowp = srcs[0]->hptr() + (size_t)y * (srcs[0]->step / (int)sizeof(T)) + (size_t)p * 4;
          const float fy = (float)rowp[(x & 1) ? 3: 1];
          const float fu = (float)rowp[0];
          const float fv = (float)rowp[2];
          const int vals[3] = {round_clamp<T>(row(t, 0, fy, fu, fv), MAXV),
                               round_clamp<T>(row(t, 1, fy, fu, fv), MAXV),
                               round_clamp<T>(row(t, 2, fy, fu, fv), MAXV)};
          const size_t base = (size_t)y * dsts[0]->step + (size_t)x * 3 * sizeof(T);
          for (int k = 0; k < 3; ++k) {
            const size_t o = base + (size_t)k * sizeof(T);
            want[0][o] = (uint8_t)(vals[k] & 0xFF);
            if constexpr (sizeof(T) == 2) want[0][o + 1] = (uint8_t)((vals[k] >> 8) & 0xFF);
          }
        }
    } else {  // inverse kinds
      const int spanY = (kind == Kind::Inv420C || kind == Kind::Inv420P3) ? 2: 1;
      for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) {
          const float fy = srcPl(0, x, y);
          const float fu = srcPl(1, x >> 1, y / spanY);
          const float fv = srcPl(2, x >> 1, y / spanY);
          const int o0 = round_clamp<T>(row(t, 0, fy, fu, fv), MAXV);
          const int o1 = round_clamp<T>(row(t, 1, fy, fu, fv), MAXV);
          const int o2 = round_clamp<T>(row(t, 2, fy, fu, fv), MAXV);
          if (getenv("NUM_TWIST_DEBUG") && x == 0 && y == 1)
            std::fprintf(stderr, "[%s] %dx%d src=(%.0f,%.0f,%.0f) o=(%d,%d,%d) t=%.3f,%.3f,%.3f,%.3f\n", id, w, h,
                         fy, fu, fv, o0, o1, o2, t.m[0][0], t.m[0][1], t.m[0][2], t.m[0][3]);
          if (dstP3) {
            wr(0, x, y, o0);
            wr(1, x, y, o1);
            wr(2, x, y, o2);
          } else {
            const size_t base = (size_t)y * dsts[0]->step + (size_t)x * dstPp * sizeof(T);
            const int vals[4] = {o0, o1, o2, (int)alpha};
            for (int k = 0; k < dstPp; ++k) {
              const size_t o = base + (size_t)k * sizeof(T);
              want[0][o] = (uint8_t)(vals[k] & 0xFF);
              if constexpr (sizeof(T) == 2) want[0][o + 1] = (uint8_t)((vals[k] >> 8) & 0xFF);
            }
          }
        }
    }
    bool failed = false;
    for (size_t p = 0; p < dsts.size() && !failed; ++p) {
      dsts[p]->dev.copyToHost(dsts[p]->host);
      const GenCase dc{(int)dsts[p]->bw, (int)dsts[p]->bh, 0, 0, 0, kSmall, 0};
      const int chans = dstC2 ? 2: (dstP3 ? 1: (fwd ? 1: dstPp));
      bool ok;
      if (chans == 1)
        ok = rotate_check<T, 1, 1>(id, dc, NPPI_INTER_NN, pre[p], dsts[p]->host, want[p], dsts[p]->step,
                                   std::vector<uint8_t>(), cnt, first);
      else if (chans == 2)
        ok = rotate_check<T, 2, 2>(id, dc, NPPI_INTER_NN, pre[p], dsts[p]->host, want[p], dsts[p]->step,
                                   std::vector<uint8_t>(), cnt, first);
      else if (chans == 4)
        ok = rotate_check<T, 4, 4>(id, dc, NPPI_INTER_NN, pre[p], dsts[p]->host, want[p], dsts[p]->step,
                                   std::vector<uint8_t>(), cnt, first);
      else
        ok = rotate_check<T, 3, 3>(id, dc, NPPI_INTER_NN, pre[p], dsts[p]->host, want[p], dsts[p]->step,
                                   std::vector<uint8_t>(), cnt, first);
      if (!ok) failed = true;
    }
    for (auto *p: srcs) delete p;
    for (auto *p: dsts) delete p;
    if (failed || cnt.mismatched >= 8) break;
  }
  finish(id, cnt, first);
}

}  // namespace yuvtwist
}  // namespace num
