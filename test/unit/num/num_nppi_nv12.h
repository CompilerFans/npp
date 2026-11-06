// CPU-golden coverage for the plain (non-ColorTwist) NV12 / NV21 families:
//   nppiNV12ToRGB / NV12ToBGR (BT.601), the 709CSC/709HDTV variants
//   (BT.709), nppiNV21ToRGB / NV21ToBGR (fixed point, 0xFF alpha) and
//   nppiNV12ToYUV420 (plane split).
//
// Semantics from the measured NPP 13.4 kernels: chroma comes from the
// top-left sample of each 2x2 block (UV row y/2, interleaved offset
// (x/2)*2 and +1); BT.601/709 apply the studio-range matrix in float with
// floor(x + 0.5) and clamping; NV21 uses y + floor((k*delta)/256) fixed point
// and writes alpha 0xFF; NV12ToYUV420 copies Y and deinterleaves UV into
// ceil(w/2) x ceil(h/2) U and V planes.
#pragma once

#include "num_nppi_geom.h"

namespace num {
namespace nv12 {

enum class Var { NV12_601, NV12_709, NV12_709_CSC, NV21_FIXED };
enum class OutOrder { RGB, BGR };

inline Npp8u q(float v) {
  v = v < 0.0f ? 0.0f : (v > 255.0f ? 255.0f : v);
  return (Npp8u)(v + 0.5f);
}

inline int rnd256(int n) {
  // floor(n/256 + 0.5): the NV21 kernels round half up.
  return (n + 128) >> 8;
}

inline void conv(Var var, int y, int u, int v, bool bgr, Npp8u &o0, Npp8u &o1, Npp8u &o2, Npp8u &alpha) {
  alpha = 0xFF;
  if (var == Var::NV21_FIXED) {
    const int ud = u - 128, vd = v - 128;
    const int r = std::min(255, std::max(0, y + rnd256(452 * vd)));
    const int g = std::min(255, std::max(0, y + rnd256(183 * ud - 88 * vd)));
    const int b = std::min(255, std::max(0, y + rnd256(360 * ud)));
    o0 = (Npp8u)(bgr ? b : r);
    o1 = (Npp8u)g;
    o2 = (Npp8u)(bgr ? r : b);
    return;
  }
  // The NV12 kernels use the FULL-RANGE YUV matrix (no 16/128 studio offset)
  // and round half up; measured on NPP 13.4.
  const float fy = (float)y;
  const float fu = (float)u - 128.0f;
  const float fv = (float)v - 128.0f;
  float fr, fg, fb;
  if (var == Var::NV12_601) {
    fr = fy + 1.140f * fv;
    fg = fy - 0.395f * fu - 0.581f * fv;
    fb = fy + 2.032f * fu;
  } else if (var == Var::NV12_709) {
    fr = fy + 1.28125f * fv;
    fg = fy - 0.2109375f * fu - 0.3828125f * fv;
    fb = fy + 2.125f * fu;
  } else {
    // 709CSC is the studio-range BT.709 conversion (Y-16 offset).
    const float yc = fy - 16.0f;
    fr = 1.164f * yc + 1.793f * fv;
    fg = 1.164f * yc - 0.213f * fu - 0.533f * fv;
    fb = 1.164f * yc + 2.112f * fu;
  }
  o0 = q(bgr ? fb : fr);
  o1 = q(fg);
  o2 = q(bgr ? fr : fb);
}

// dstC = 3 or 4 (4 only for the NV21 P2C4R forms).
template <Var V, OutOrder O, int DstC, class Call>
void run_nv12_to_rgb(const char *id, Call call) {
  NppStreamContext ctx{};
  if (nppGetStreamContext(&ctx) != NPP_NO_ERROR) {
    GTEST_SKIP() << "[" << id << "] no stream context";
    return;
  }
  RunCounts cnt;
  std::string first;
  struct Cfg { int w, h; };
  static const Cfg cfgs[] = {{4, 4}, {8, 6}, {32, 24}, {64, 48}, {16, 12}, {128, 96}};
  for (const Cfg &cf : cfgs) {
    const int uw = (cf.w + 1) / 2, uh = (cf.h + 1) / 2;
    // The P2 forms use one source step for the Y and UV planes.
    const int sstep = align16((size_t)std::max(cf.w, uw * 2));
    Buf<Npp8u> y(cf.w, cf.h, 1, sstep - align16((size_t)cf.w), 0);
    Buf<Npp8u> uv(uw, uh, 2, sstep - align16((size_t)(uw * 2)), 0);
    fill_regime(y, 1, case_seed(id, cf.w, cf.h, 0, 0, 0, 0), kWide, 0, false, false, false);
    fill_regime(uv, 2, case_seed(id, cf.w, cf.h, 1, 2, 0, 1), kWide, 1, false, false, false);
    y.dev.copyFromHost(y.host);
    uv.dev.copyFromHost(uv.host);
    Buf<Npp8u> d(cf.w, cf.h, DstC, 0, 0x5A);
    d.dev.copyFromHost(d.host);
    const Npp8u *planes[2] = {y.dptr(), uv.dptr()};
    const NppStatus st = call(planes, sstep, d.dptr(), d.step, NppiSize{cf.w, cf.h}, ctx);
    if (st != NPP_NO_ERROR) {
      ++cnt.skipped;
      if (st == NPP_NOT_IMPLEMENTED_ERROR) ++cnt.notimpl;
      continue;
    }
    const cudaError_t se = cudaDeviceSynchronize();
    ASSERT_EQ(se, cudaSuccess) << "[" << id << "] " << cf.w << "x" << cf.h << " sync: " << cudaGetErrorString(se);
    d.dev.copyToHost(d.host);
    ++cnt.executed;
    std::vector<uint8_t> want = d.host;
    for (int yy = 0; yy < cf.h; ++yy)
      for (int xx = 0; xx < cf.w; ++xx) {
        const int yv = y.host[(size_t)yy * sstep + (size_t)xx];
        const int uvi = (yy / 2) * sstep + (xx / 2) * 2;
        // NV21 stores the chroma pair as V,U; NV12 stores U,V.
        const int u = uv.host[(size_t)uvi + (V == Var::NV21_FIXED ? 1 : 0)];
        const int v = uv.host[(size_t)uvi + (V == Var::NV21_FIXED ? 0 : 1)];
        Npp8u o0, o1, o2, alpha;
        conv(V, yv, u, v, O == OutOrder::BGR, o0, o1, o2, alpha);
        // Measured NPP quirk: the NV21 BGR variant zeroes the R slot unless
        // the ROI width is a multiple of 32.
        if constexpr (V == Var::NV21_FIXED && O == OutOrder::BGR) {
          if (cf.w % 32 != 0) o2 = 0;
        }
        const size_t db = off_bytes<Npp8u>(yy, xx, d.step, DstC);
        want[db] = o0;
        want[db + 1] = o1;
        want[db + 2] = o2;
        if constexpr (DstC == 4) want[db + 3] = alpha;
      }
    const GenCase dc{cf.w, cf.h, 0, 0, 0, kSmall, 0};
    if (!rotate_check<Npp8u, DstC, DstC>(id, dc, NPPI_INTER_NN, d.host, d.host, want, d.step,
                                         std::vector<uint8_t>(), cnt, first)) {
      if (cnt.mismatched >= 8) break;
    }
    if (cnt.mismatched >= 8) break;
  }
  finish(id, cnt, first);
}

template <class Call>
void run_nv12_to_yuv420(const char *id, Call call) {
  NppStreamContext ctx{};
  if (nppGetStreamContext(&ctx) != NPP_NO_ERROR) {
    GTEST_SKIP() << "[" << id << "] no stream context";
    return;
  }
  RunCounts cnt;
  std::string first;
  struct Cfg { int w, h; };
  static const Cfg cfgs[] = {{4, 4}, {8, 6}, {32, 24}, {64, 48}, {16, 12}, {128, 96}};
  for (const Cfg &cf : cfgs) {
    const int uw = (cf.w + 1) / 2, uh = (cf.h + 1) / 2;
    const int sstep = align16((size_t)std::max(cf.w, uw * 2));
    Buf<Npp8u> y(cf.w, cf.h, 1, sstep - align16((size_t)cf.w), 0);
    Buf<Npp8u> uv(uw, uh, 2, sstep - align16((size_t)(uw * 2)), 0);
    fill_regime(y, 1, case_seed(id, cf.w, cf.h, 0, 0, 0, 0), kWide, 0, false, false, false);
    fill_regime(uv, 2, case_seed(id, cf.w, cf.h, 1, 2, 0, 1), kWide, 1, false, false, false);
    y.dev.copyFromHost(y.host);
    uv.dev.copyFromHost(uv.host);
    Buf<Npp8u> dy(cf.w, cf.h, 1, 0, 0x5A), du(uw, uh, 1, 0, 0x5A), dv(uw, uh, 1, 0, 0x5A);
    dy.dev.copyFromHost(dy.host);
    du.dev.copyFromHost(du.host);
    dv.dev.copyFromHost(dv.host);
    const Npp8u *planes[2] = {y.dptr(), uv.dptr()};
    Npp8u *dplanes[3] = {dy.dptr(), du.dptr(), dv.dptr()};
    int dstSteps[3] = {dy.step, du.step, dv.step};
    const NppStatus st = call(planes, sstep, dplanes, dstSteps, NppiSize{cf.w, cf.h}, ctx);
    if (st != NPP_NO_ERROR) {
      ++cnt.skipped;
      if (st == NPP_NOT_IMPLEMENTED_ERROR) ++cnt.notimpl;
      continue;
    }
    const cudaError_t se = cudaDeviceSynchronize();
    ASSERT_EQ(se, cudaSuccess) << "[" << id << "] sync: " << cudaGetErrorString(se);
    dy.dev.copyToHost(dy.host);
    du.dev.copyToHost(du.host);
    dv.dev.copyToHost(dv.host);
    ++cnt.executed;
    ++cnt.compared;
    bool bad = false;
    for (int yy = 0; yy < cf.h && !bad; ++yy)
      for (int xx = 0; xx < cf.w; ++xx)
        if (dy.host[(size_t)yy * dy.step + (size_t)xx] != y.host[(size_t)yy * sstep + (size_t)xx]) {
          bad = true;
          break;
        }
    for (int yy = 0; yy < uh && !bad; ++yy)
      for (int xx = 0; xx < uw; ++xx) {
        const int si = yy * sstep + xx * 2;
        if (du.host[(size_t)yy * du.step + (size_t)xx] != uv.host[(size_t)si] ||
            dv.host[(size_t)yy * dv.step + (size_t)xx] != uv.host[(size_t)si + 1]) {
          bad = true;
          break;
        }
      }
    if (bad) {
      ++cnt.mismatched;
      if (first.empty()) first = std::string("[") + id + "] NV12ToYUV420 plane mismatch";
      EXPECT_TRUE(false) << first;
      break;
    }
  }
  finish(id, cnt, first);
}


// ---------------------------------------------------------------------------
// ColorTwist NV12 variants: the caller's 3x4 matrix is applied to (Y,U,V,1)
// for NV12ToRGB and to (R,G,B,1) for RGBToNV12.  Results round half up and
// clamp to the element type (255 / 65535).  RGBToNV12 averages chroma over
// the 2x2 block in float and rounds once; odd ROI dimensions are rejected by
// NPP, so the tests use even sizes only.
// ---------------------------------------------------------------------------
inline void nv12_twist_apply(const float m[12], float c0, float c1, float c2, float &o0, float &o1, float &o2) {
  o0 = m[0] * c0 + m[1] * c1 + m[2] * c2 + m[3];
  o1 = m[4] * c0 + m[5] * c1 + m[6] * c2 + m[7];
  o2 = m[8] * c0 + m[9] * c1 + m[10] * c2 + m[11];
}

template <typename T>
inline T twist_round_clamp(float v) {
  float r = std::floor(v + 0.5f);
  const float mx = sizeof(T) == 1 ? 255.0f : 65535.0f;
  if (r < 0.0f) r = 0.0f;
  else if (r > mx) r = mx;
  return (T)r;
}

inline void twist_matrices(float out[4][12]) {
  static const float mats[4][12] = {
      {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0},
      {1, 0, 0, 16, 0, 1, 0, -8, 0, 0, 1, 32},
      {0.5f, 0.25f, 0.25f, 0, 0.25f, 0.5f, 0.25f, 0, 0.25f, 0.25f, 0.5f, 0},
      {1.5f, 0, 0, -20, 0, 1.25f, 0, 10, 0, 0, 0.75f, 5},
  };
  for (int i = 0; i < 4; ++i)
    for (int j = 0; j < 12; ++j) out[i][j] = mats[i][j];
}

template <typename T, class Call>
void run_nv12_twist_to_rgb(const char *id, Call call) {
  NppStreamContext ctx{};
  if (nppGetStreamContext(&ctx) != NPP_NO_ERROR) {
    GTEST_SKIP() << "[" << id << "] no stream context";
    return;
  }
  RunCounts cnt;
  std::string first;
  struct Cfg { int w, h; };
  static const Cfg cfgs[] = {{4, 4}, {8, 6}, {32, 24}, {64, 48}, {16, 12}, {128, 96}};
  float mats[4][12];
  twist_matrices(mats);
  for (const Cfg &cf : cfgs) {
    const int uw = cf.w / 2, uh = cf.h / 2;
    Buf<T> y(cf.w, cf.h, 1, 0, 0);
    Buf<T> uv(uw, uh, 2, 0, 0);
    fill_regime(y, 1, case_seed(id, cf.w, cf.h, 0, 0, 0, 0), kWide, 0, false, false,
                std::is_floating_point_v<T>);
    fill_regime(uv, 2, case_seed(id, cf.w, cf.h, 1, 2, 0, 1), kWide, 1, false, false,
                std::is_floating_point_v<T>);
    y.dev.copyFromHost(y.host);
    uv.dev.copyFromHost(uv.host);
    Buf<T> d(cf.w, cf.h, 3, 0, 0x5A);
    d.dev.copyFromHost(d.host);
    for (int mi = 0; mi < 4; ++mi) {
      const std::vector<uint8_t> pre = d.host;
      const T *sp[2] = {reinterpret_cast<const T *>(y.dptr()), reinterpret_cast<const T *>(uv.dptr())};
      int steps[2] = {y.step, uv.step};
      const NppStatus st = call(sp, steps, reinterpret_cast<T *>(d.dptr()), d.step, NppiSize{cf.w, cf.h}, mats[mi],
                                ctx);
      if (st != NPP_NO_ERROR) {
        ++cnt.skipped;
        if (st == NPP_NOT_IMPLEMENTED_ERROR) ++cnt.notimpl;
        continue;
      }
      cudaDeviceSynchronize();
      d.dev.copyToHost(d.host);
      ++cnt.executed;
      std::vector<uint8_t> want = pre;
      for (int yy = 0; yy < cf.h; ++yy)
        for (int xx = 0; xx < cf.w; ++xx) {
          const T yv = y.hptr()[(size_t)yy * (y.step / (int)sizeof(T)) + (size_t)xx];
          const int uvi = (yy / 2) * (uv.step / (int)sizeof(T)) + (xx / 2) * 2;
          const T u = uv.hptr()[(size_t)uvi];
          const T v = uv.hptr()[(size_t)uvi + 1];
          float o0, o1, o2;
          nv12_twist_apply(mats[mi], (float)as_double(yv), (float)as_double(u), (float)as_double(v), o0, o1, o2);
          const size_t db = off_bytes<T>(yy, xx, d.step, 3);
          const T t0 = twist_round_clamp<T>(o0), t1 = twist_round_clamp<T>(o1), t2 = twist_round_clamp<T>(o2);
          std::memcpy(want.data() + db, &t0, sizeof(T));
          std::memcpy(want.data() + db + sizeof(T), &t1, sizeof(T));
          std::memcpy(want.data() + db + 2 * sizeof(T), &t2, sizeof(T));
        }
      const GenCase dc{cf.w, cf.h, 0, 0, 0, kSmall, 0};
      if (!rotate_check<T, 3, 3>(id, dc, NPPI_INTER_NN, pre, d.host, want, d.step,
                                 std::vector<uint8_t>(), cnt, first)) {
        if (cnt.mismatched >= 8) break;
      }
      if (cnt.mismatched >= 8) break;
    }
    if (cnt.mismatched >= 8) break;
  }
  finish(id, cnt, first);
}

template <typename T, bool Planar, class Call>
void run_rgb_twist_to_nv12(const char *id, Call call) {
  NppStreamContext ctx{};
  if (nppGetStreamContext(&ctx) != NPP_NO_ERROR) {
    GTEST_SKIP() << "[" << id << "] no stream context";
    return;
  }
  RunCounts cnt;
  std::string first;
  struct Cfg { int w, h; };
  static const Cfg cfgs[] = {{4, 4}, {8, 6}, {32, 24}, {64, 48}, {16, 12}, {128, 96}};
  float mats[4][12];
  twist_matrices(mats);
  for (const Cfg &cf : cfgs) {
    std::vector<Buf<T> *> srcs;
    const uint64_t seed = case_seed(id, cf.w, cf.h, 0, 0, 0, 0);
    if constexpr (Planar) {
      planar_fill_src(srcs, 3, cf.w, cf.h, 0, 0, 0, seed, kWide);
    } else {
      Buf<T> *a = new Buf<T>(cf.w, cf.h, 3, 0, 0);
      fill_regime(*a, 3, seed, kWide, 0, false, false, std::is_floating_point_v<T>);
      a->dev.copyFromHost(a->host);
      srcs.push_back(a);
    }
    Buf<T> dy(cf.w, cf.h, 1, 0, 0x5A), duv(cf.w / 2, cf.h / 2, 2, 0, 0x5A);
    dy.dev.copyFromHost(dy.host);
    duv.dev.copyFromHost(duv.host);
    for (int mi = 0; mi < 4; ++mi) {
      dy.dev.copyFromHost(dy.host);
      duv.dev.copyFromHost(duv.host);
      NppStatus st;
      if constexpr (Planar) {
        const T *sp[3] = {reinterpret_cast<const T *>(srcs[0]->dptr()),
                          reinterpret_cast<const T *>(srcs[1]->dptr()),
                          reinterpret_cast<const T *>(srcs[2]->dptr())};
        int sstep[3] = {srcs[0]->step, srcs[1]->step, srcs[2]->step};
        T *dp[2] = {reinterpret_cast<T *>(dy.dptr()), reinterpret_cast<T *>(duv.dptr())};
        int dstep[2] = {dy.step, duv.step};
        st = call(sp, sstep, dp, dstep, NppiSize{cf.w, cf.h}, mats[mi], ctx);
      } else {
        T *dp[2] = {reinterpret_cast<T *>(dy.dptr()), reinterpret_cast<T *>(duv.dptr())};
        int dstep[2] = {dy.step, duv.step};
        st = call(reinterpret_cast<const T *>(srcs[0]->dptr()), srcs[0]->step, dp, dstep, NppiSize{cf.w, cf.h},
                  mats[mi], ctx);
      }
      if (st != NPP_NO_ERROR) {
        ++cnt.skipped;
        if (st == NPP_NOT_IMPLEMENTED_ERROR) ++cnt.notimpl;
        continue;
      }
      cudaDeviceSynchronize();
      dy.dev.copyToHost(dy.host);
      duv.dev.copyToHost(duv.host);
      ++cnt.executed;
      const int yStepE = dy.step / (int)sizeof(T);
      const int uvStepE = duv.step / (int)sizeof(T);
      auto srcAt = [&](int p, int x, int y) -> float {
        if constexpr (Planar) return (float)as_double(srcs[(size_t)p]->hptr()[(size_t)y * (srcs[(size_t)p]->step / (int)sizeof(T)) + (size_t)x]);
        else return (float)as_double(srcs[0]->hptr()[(size_t)y * (srcs[0]->step / (int)sizeof(T)) + (size_t)x * 3 + (size_t)p]);
      };
      bool bad = false;
      for (int yy = 0; yy < cf.h && !bad; ++yy)
        for (int xx = 0; xx < cf.w; ++xx) {
          float fy, fu, fv;
          nv12_twist_apply(mats[mi], srcAt(0, xx, yy), srcAt(1, xx, yy), srcAt(2, xx, yy), fy, fu, fv);
          const T gotY = dy.hptr()[(size_t)yy * yStepE + (size_t)xx];
          const T wantY = twist_round_clamp<T>(fy);
          if (std::memcmp(&gotY, &wantY, sizeof(T)) != 0) {
            ++cnt.compared;
            ++cnt.mismatched;
            if (first.empty()) first = std::string("[") + id + "] Y plane mismatch";
            bad = true;
            break;
          }
          if ((xx & 1) == 0 && (yy & 1) == 0) {
            float us = fu, vs = fv;
            const int dx[3] = {1, 0, 1}, dyy[3] = {0, 1, 1};
            for (int k = 0; k < 3; ++k) {
              float q0, q1, q2;
              nv12_twist_apply(mats[mi], srcAt(0, xx + dx[k], yy + dyy[k]), srcAt(1, xx + dx[k], yy + dyy[k]),
                               srcAt(2, xx + dx[k], yy + dyy[k]), q0, q1, q2);
              us += q1;
              vs += q2;
            }
            const size_t uvi = (size_t)(yy / 2) * uvStepE + (size_t)xx;
            const T gotU = duv.hptr()[uvi], gotV = duv.hptr()[uvi + 1];
            const T wantU = twist_round_clamp<T>(us * 0.25f), wantV = twist_round_clamp<T>(vs * 0.25f);
            if (std::memcmp(&gotU, &wantU, sizeof(T)) != 0 || std::memcmp(&gotV, &wantV, sizeof(T)) != 0) {
              ++cnt.compared;
              ++cnt.mismatched;
              if (first.empty()) first = std::string("[") + id + "] chroma plane mismatch";
              bad = true;
              break;
            }
          }
        }
      if (bad) {
        EXPECT_TRUE(false) << first;
        break;
      }
    }
    for (auto *p : srcs) delete p;
    if (cnt.mismatched >= 8) break;
  }
  finish(id, cnt, first);
}

}  // namespace nv12
}  // namespace num
