// CPU-golden coverage for the single-image nppiColorTwist families
// (nppiColorTwist, nppiColorTwist32f, nppiColorTwist32fC,
// nppiColorTwist_32fC).
//
// Measured NPP 13.4 semantics:
//   * out_c = sum_j aTwist[c][j] * in_j + aTwist[c][3] (j < 3), for the first
//     min(channels, 3) output channels; in_j for j >= channels reads 0;
//   * packed layouts with >= 3 channels add a +0.5f rounding bias before the
//     conversion (Npp32f keeps the bias in the stored value); C1/C2, planar
//     and every Npp16f variant have no bias;
//   * integer conversion truncates toward zero and saturates to the dtype
//     range; Npp32f saturates to [-FLT_MAX, FLT_MAX]; Npp16f rounds to
//     nearest;
//   * C4 (non-AC4) copies the source alpha bit-exactly; AC4 leaves the
//     destination alpha untouched;
//   * the "C" variants apply a full 4x4 twist plus per-channel additive
//     constants to all four channels (alpha is computed, never copied).
#pragma once

#include "num_nppi_geom.h"

namespace num {
namespace twist {

template <typename S>
inline S from_float(float v) {
  if constexpr (std::is_same_v<S, Npp32f>) {
    if (v > 3.402823466e+38F) return 3.402823466e+38F;
    if (v < -3.402823466e+38F) return -3.402823466e+38F;
    return v;
  } else if constexpr (std::is_same_v<S, Npp16f>) {
    return half_round(v);
  } else {
    double lo = 0.0, hi = 0.0;
    if constexpr (std::is_same_v<S, Npp8u>) { lo = 0.0; hi = 255.0; }
    else if constexpr (std::is_same_v<S, Npp8s>) { lo = -128.0; hi = 127.0; }
    else if constexpr (std::is_same_v<S, Npp16u>) { lo = 0.0; hi = 65535.0; }
    else if constexpr (std::is_same_v<S, Npp16s>) { lo = -32768.0; hi = 32767.0; }
    const float t = std::trunc(v);
    if (t < (float)lo) return (S)lo;
    if (t > (float)hi) return (S)hi;
    return (S)t;
  }
}

template <typename S, int C>
inline bool twist_bias(bool planar) {
  if constexpr (std::is_same_v<S, Npp16f>) return false;
  else return !planar && C >= 3;
}

inline float twist_dot(const float m[12], int row, const float v[4]) {
  return m[row * 4 + 0] * v[0] + m[row * 4 + 1] * v[1] + m[row * 4 + 2] * v[2] + m[row * 4 + 3];
}

// 3x4 twist matrices (12 floats each), exactly representable.
inline void twist_matrix(int mi, float m[12]) {
  static const float mats[5][12] = {
      {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0},
      {1, 0, 0, 5, 0, 1, 0, -7, 0, 0, 1, 11},
      {0.5f, 0.25f, 0.25f, 0, 0.25f, 0.5f, 0.25f, 0, 0.25f, 0.25f, 0.5f, 0},
      {2, 0, 0, 0, 0, -1, 0, 0, 0, 0, 1.5f, -3},
      {0, 1, 0, 0, 1, 0, 0, 0, 0, 0, -1, 2},
  };
  for (int i = 0; i < 12; ++i) m[i] = mats[mi][i];
}

// Fills one packed pixel's expected output.  A is the number of channels the
// kernel writes (CH, or 3 for AC4); channels >= A must keep the previous dst.
template <typename S, int CH, int A>
inline void twist_ref_pixel(const float m[12], const S *srcPix, const uint8_t *prePix, uint8_t *dstOut,
                            bool hasAlphaCopy) {
  float v[4] = {0, 0, 0, 0};
  for (int c = 0; c < CH; ++c) v[c] = (float)as_double(srcPix[c]);
  for (int c = 0; c < A; ++c) {
    float o = twist_dot(m, c, v);
    if (twist_bias<S, CH>(false)) o += 0.5f;
    const S out = from_float<S>(o);
    std::memcpy(dstOut + (size_t)c * sizeof(S), &out, sizeof(S));
  }
  if (CH == 4 && !hasAlphaCopy) {
    // AC4: alpha untouched (already in dstOut from prePix).
  }
  (void)prePix;
}

template <typename S, int C, int A, bool InPlace, class Call>
void run_twist(const char *id, Call call) {
  NppStreamContext ctx{};
  if (nppGetStreamContext(&ctx) != NPP_NO_ERROR) {
    GTEST_SKIP() << "[" << id << "] no stream context";
    return;
  }
  RunCounts cnt;
  std::string first;
  for (const GenCase &c : make_gen_cases(0)) {
    if (c.w < 2 || c.h < 2) continue;
    // NPP's 8s kernels switch to a vectorized path when the row pitch is a
    // multiple of 128 bytes; that path mis-handles signed samples (every
    // negative byte saturates high).  A pitch of align16 + 4 is never a
    // multiple of 128 and selects the scalar path.
    const int padUse = std::is_same_v<S, Npp8s> ? 4 : c.pad;
    Buf<S> s(c.w + c.ox, c.h + c.oy, C, padUse, 0);
    Buf<S> d(c.w, c.h, C, padUse, 0x5A);
    fill_regime(s, C, case_seed(id, c.w, c.h, s.step, d.step, 0, 0), c.regime, 0, false, false,
                std::is_floating_point_v<S>);
    if constexpr (std::is_floating_point_v<S>) {
      const size_t n = (size_t)(s.step / sizeof(S)) * (size_t)s.bh;
      for (size_t i = 0; i < n; ++i) {
        const double v = (double)s.hptr()[i];
        if (v > 1e3) s.hptr()[i] = S(1e3);
        else if (v < -1e3) s.hptr()[i] = S(-1e3);
        else if (v != v) s.hptr()[i] = S(0);
      }
    }
    const std::vector<uint8_t> orig = s.host;
    s.dev.copyFromHost(s.host);
    d.dev.copyFromHost(d.host);
    const S *hsrc = s.hptr() + off_bytes<S>(c.oy, c.ox, s.step, C) / sizeof(S);
    const S *dsrc = reinterpret_cast<const S *>(s.dptr() + off_bytes<S>(c.oy, c.ox, s.step, C));
    for (int mi = 0; mi < 5; ++mi) {
      float m[12];
      twist_matrix(mi, m);
      if constexpr (InPlace) {
        s.host = orig;
        s.dev.copyFromHost(s.host);
      }
      const std::vector<uint8_t> pre = InPlace ? orig : d.host;
      npp_diff::clear_cuda_error();
      NppStatus st;
      if constexpr (InPlace) {
        st = call(reinterpret_cast<S *>(s.dptr() + off_bytes<S>(c.oy, c.ox, s.step, C)), s.step,
                  NppiSize{c.w, c.h}, m, ctx);
      } else {
        st = call(dsrc, s.step, reinterpret_cast<S *>(d.dptr()), d.step, NppiSize{c.w, c.h}, m, ctx);
      }
      if (st != NPP_NO_ERROR) {
        ++cnt.skipped;
        if (st == NPP_NOT_IMPLEMENTED_ERROR) ++cnt.notimpl;
        continue;
      }
      const cudaError_t se = cudaDeviceSynchronize();
      ASSERT_EQ(se, cudaSuccess) << "[" << id << "] " << c.id() << " sync: " << cudaGetErrorString(se);
      std::vector<uint8_t> got;
      if constexpr (InPlace) s.dev.copyToHost(got);
      else d.dev.copyToHost(got);
      ++cnt.executed;
      std::vector<uint8_t> want = pre;
      for (int y = 0; y < c.h; ++y)
        for (int x = 0; x < c.w; ++x) {
          const size_t so = (size_t)y * (s.step / (int)sizeof(S)) + (size_t)x * C;
          const size_t db = InPlace ? off_bytes<S>(c.oy + y, c.ox + x, s.step, C)
                                    : off_bytes<S>(y, x, d.step, C);
          float v[4] = {0, 0, 0, 0};
          for (int ch = 0; ch < C; ++ch) v[ch] = (float)as_double(hsrc[so + (size_t)ch]);
          for (int ch = 0; ch < A && ch < 3; ++ch) {
            float o = twist_dot(m, ch, v);
            if (twist_bias<S, C>(false)) o += 0.5f;
            const S out = from_float<S>(o);
            std::memcpy(want.data() + db + (size_t)ch * sizeof(S), &out, sizeof(S));
          }
          if constexpr (C == 4 && A == 4) {
            // Packed C4: alpha is copied from the source.
            if constexpr (InPlace) {
              // source == destination buffer: alpha already equals the source.
            } else {
              std::memcpy(want.data() + db + 3 * sizeof(S), hsrc + so + 3, sizeof(S));
            }
          }
        }
      const GenCase dc{c.w, c.h, InPlace ? c.ox : 0, InPlace ? c.oy : 0, 0, kSmall, 0};
      if (!rotate_check<S, C, A>(id, dc, NPPI_INTER_NN, pre, got, want,
                                 InPlace ? s.step : d.step, std::vector<uint8_t>(), cnt, first)) {
        if (cnt.mismatched >= 8) break;
      }
      if (cnt.mismatched >= 8) break;
    }
    if (cnt.mismatched >= 8) break;
  }
  finish(id, cnt, first);
}

template <typename S, bool InPlace, class Call>
void run_twist_c(const char *id, Call call) {
  NppStreamContext ctx{};
  if (nppGetStreamContext(&ctx) != NPP_NO_ERROR) {
    GTEST_SKIP() << "[" << id << "] no stream context";
    return;
  }
  RunCounts cnt;
  std::string first;
  static const float mats[4][4][4] = {
      {{1, 0, 0, 0}, {0, 1, 0, 0}, {0, 0, 1, 0}, {0, 0, 0, 1}},
      {{1, 0, 0, 0.25f}, {0, 1, 0, -0.5f}, {0, 0, 1, 1.5f}, {0, 0, 0, 1}},
      {{0.5f, 0.25f, 0.25f, 0}, {0.25f, 0.5f, 0.25f, 0}, {0.25f, 0.25f, 0.5f, 0}, {0, 0, 0, 1}},
      {{2, 0, 0, 0}, {0, -1, 0, 0}, {0, 0, 1.5f, -3}, {0, 0, 0, 1}},
  };
  static const float cons[4][4] = {
      {0, 0, 0, 0},
      {5, -7, 11, 3},
      {0, 0, 0, 0},
      {1, 2, 3, 4},
  };
  for (const GenCase &c : make_gen_cases(0)) {
    if (c.w < 2 || c.h < 2) continue;
    const int padUse = std::is_same_v<S, Npp8s> ? 4 : c.pad;
    Buf<S> s(c.w + c.ox, c.h + c.oy, 4, padUse, 0);
    Buf<S> d(c.w, c.h, 4, padUse, 0x5A);
    fill_regime(s, 4, case_seed(id, c.w, c.h, s.step, d.step, 0, 0), c.regime, 0, false, false,
                std::is_floating_point_v<S>);
    if constexpr (std::is_floating_point_v<S>) {
      const size_t n = (size_t)(s.step / sizeof(S)) * (size_t)s.bh;
      for (size_t i = 0; i < n; ++i) {
        const double v = (double)s.hptr()[i];
        if (v > 1e3) s.hptr()[i] = S(1e3);
        else if (v < -1e3) s.hptr()[i] = S(-1e3);
        else if (v != v) s.hptr()[i] = S(0);
      }
    }
    const std::vector<uint8_t> orig = s.host;
    s.dev.copyFromHost(s.host);
    d.dev.copyFromHost(d.host);
    const S *hsrc = s.hptr() + off_bytes<S>(c.oy, c.ox, s.step, 4) / sizeof(S);
    const S *dsrc = reinterpret_cast<const S *>(s.dptr() + off_bytes<S>(c.oy, c.ox, s.step, 4));
    for (int mi = 0; mi < 4; ++mi) {
      if constexpr (InPlace) {
        s.host = orig;
        s.dev.copyFromHost(s.host);
      }
      const std::vector<uint8_t> pre = InPlace ? orig : d.host;
      npp_diff::clear_cuda_error();
      NppStatus st;
      if constexpr (InPlace) {
        st = call(reinterpret_cast<S *>(s.dptr() + off_bytes<S>(c.oy, c.ox, s.step, 4)), s.step,
                  NppiSize{c.w, c.h}, reinterpret_cast<const float *>(mats[mi]),
                  reinterpret_cast<const float *>(cons[mi]), ctx);
      } else {
        st = call(dsrc, s.step, reinterpret_cast<S *>(d.dptr()), d.step, NppiSize{c.w, c.h},
                  reinterpret_cast<const float *>(mats[mi]), reinterpret_cast<const float *>(cons[mi]), ctx);
      }
      if (st != NPP_NO_ERROR) {
        ++cnt.skipped;
        if (st == NPP_NOT_IMPLEMENTED_ERROR) ++cnt.notimpl;
        continue;
      }
      const cudaError_t se = cudaDeviceSynchronize();
      ASSERT_EQ(se, cudaSuccess) << "[" << id << "] " << c.id() << " sync: " << cudaGetErrorString(se);
      std::vector<uint8_t> got;
      if constexpr (InPlace) s.dev.copyToHost(got);
      else d.dev.copyToHost(got);
      ++cnt.executed;
      std::vector<uint8_t> want = pre;
      const float *mm = reinterpret_cast<const float *>(mats[mi]);
      const float *cc = reinterpret_cast<const float *>(cons[mi]);
      for (int y = 0; y < c.h; ++y)
        for (int x = 0; x < c.w; ++x) {
          const size_t so = (size_t)y * (s.step / (int)sizeof(S)) + (size_t)x * 4;
          const size_t db = InPlace ? off_bytes<S>(c.oy + y, c.ox + x, s.step, 4)
                                    : off_bytes<S>(y, x, d.step, 4);
          float v[4];
          for (int ch = 0; ch < 4; ++ch) v[ch] = (float)as_double(hsrc[so + (size_t)ch]);
          for (int ch = 0; ch < 4; ++ch) {
            float o = mm[ch * 4 + 0] * v[0] + mm[ch * 4 + 1] * v[1] + mm[ch * 4 + 2] * v[2] +
                      mm[ch * 4 + 3] * v[3] + cc[ch];
            if constexpr (!std::is_same_v<S, Npp16f>) o += 0.5f;
            const S out = from_float<S>(o);
            std::memcpy(want.data() + db + (size_t)ch * sizeof(S), &out, sizeof(S));
          }
        }
      const GenCase dc{c.w, c.h, InPlace ? c.ox : 0, InPlace ? c.oy : 0, 0, kSmall, 0};
      if (!rotate_check<S, 4, 4>(id, dc, NPPI_INTER_NN, pre, got, want,
                                 InPlace ? s.step : d.step, std::vector<uint8_t>(), cnt, first)) {
        if (cnt.mismatched >= 8) break;
      }
      if (cnt.mismatched >= 8) break;
    }
    if (cnt.mismatched >= 8) break;
  }
  finish(id, cnt, first);
}

template <typename S, bool InPlace, class Call>
void run_twist_planar(const char *id, Call call) {
  NppStreamContext ctx{};
  if (nppGetStreamContext(&ctx) != NPP_NO_ERROR) {
    GTEST_SKIP() << "[" << id << "] no stream context";
    return;
  }
  RunCounts cnt;
  std::string first;
  for (const GenCase &c : make_gen_cases(0)) {
    if (c.w < 2 || c.h < 2) continue;
    std::vector<Buf<S> *> srcs;
    const uint64_t seed = case_seed(id, c.w, c.h, c.pad, 0, (int)c.regime, 0);
    planar_fill_src(srcs, 3, c.w, c.h, c.ox, c.oy, c.pad, seed, c.regime);
    std::vector<Buf<S> *> dsts;
    std::vector<std::vector<uint8_t>> pre(3), orig(3);
    for (int p = 0; p < 3; ++p) {
      Buf<S> *d = new Buf<S>(c.w, c.h, 1, 0, 0x5A);
      d->dev.copyFromHost(d->host);
      dsts.push_back(d);
      pre[(size_t)p] = d->host;
      orig[(size_t)p] = srcs[(size_t)p]->host;
    }
    const S *hsrc[4] = {nullptr, nullptr, nullptr, nullptr};
    const S *dsrc[4] = {nullptr, nullptr, nullptr, nullptr};
    S *ddst[4] = {nullptr, nullptr, nullptr, nullptr};
    S *mutSrc[4] = {nullptr, nullptr, nullptr, nullptr};
    int sstep[4] = {0, 0, 0, 0};
    planar_ptrs(srcs, dsts, 3, c.ox, c.oy, hsrc, dsrc, ddst, sstep);
    for (int p = 0; p < 3; ++p)
      mutSrc[p] = reinterpret_cast<S *>(srcs[(size_t)p]->dptr() +
                                        off_bytes<S>(c.oy, c.ox, srcs[(size_t)p]->step, 1));
    for (int mi = 0; mi < 5; ++mi) {
      float m[12];
      twist_matrix(mi, m);
      if constexpr (InPlace) {
        for (int p = 0; p < 3; ++p) {
          srcs[(size_t)p]->host = orig[(size_t)p];
          srcs[(size_t)p]->dev.copyFromHost(srcs[(size_t)p]->host);
        }
      }
      npp_diff::clear_cuda_error();
      NppStatus st;
      if constexpr (InPlace) {
        st = call(mutSrc, srcs[0]->step, NppiSize{c.w, c.h}, m, ctx);
      } else {
        st = call(dsrc, srcs[0]->step, ddst, dsts[0]->step, NppiSize{c.w, c.h}, m, ctx);
      }
      if (st != NPP_NO_ERROR) {
        ++cnt.skipped;
        if (st == NPP_NOT_IMPLEMENTED_ERROR) ++cnt.notimpl;
        continue;
      }
      const cudaError_t se = cudaDeviceSynchronize();
      ASSERT_EQ(se, cudaSuccess) << "[" << id << "] " << c.id() << " sync: " << cudaGetErrorString(se);
      ++cnt.executed;
      bool failed = false;
      for (int p = 0; p < 3 && !failed; ++p) {
        std::vector<uint8_t> got;
        if constexpr (InPlace) srcs[(size_t)p]->dev.copyToHost(got);
        else dsts[(size_t)p]->dev.copyToHost(got);
        std::vector<uint8_t> want = InPlace ? orig[(size_t)p] : pre[(size_t)p];
        for (int y = 0; y < c.h; ++y)
          for (int x = 0; x < c.w; ++x) {
            const size_t so = (size_t)y * (srcs[(size_t)p]->step / (int)sizeof(S)) + (size_t)x;
            const size_t db = InPlace ? off_bytes<S>(c.oy + y, c.ox + x, srcs[(size_t)p]->step, 1)
                                      : off_bytes<S>(y, x, dsts[(size_t)p]->step, 1);
            float v[3];
            for (int q = 0; q < 3; ++q) {
              const size_t o2 = (size_t)y * (srcs[(size_t)q]->step / (int)sizeof(S)) + (size_t)x;
              v[q] = (float)as_double(hsrc[q][o2]);
            }
            (void)so;
            float o = twist_dot(m, p, v);
            const S out = from_float<S>(o);
            std::memcpy(want.data() + db, &out, sizeof(S));
          }
        const GenCase dc{c.w, c.h, InPlace ? c.ox : 0, InPlace ? c.oy : 0, 0, kSmall, 0};
        if (!rotate_check<S, 1, 1>(id, dc, NPPI_INTER_NN, InPlace ? orig[(size_t)p] : pre[(size_t)p], got,
                                   want, InPlace ? srcs[(size_t)p]->step : dsts[(size_t)p]->step,
                                   std::vector<uint8_t>(), cnt, first))
          failed = true;
        if (cnt.mismatched >= 8) break;
      }
      if (failed) break;
    }
    for (auto *p : srcs) delete p;
    for (auto *p : dsts) delete p;
    if (cnt.mismatched >= 8) break;
  }
  finish(id, cnt, first);
}


// ---------------------------------------------------------------------------
// Batch ColorTwist.  Per-entry matrices live in device memory; the batch
// kernels add the +0.5f bias before the [nMin, nMax] clamp, convert 8u with
// the LOW BYTE of the u32 conversion (no saturation), exempt AC4 channel 2
// from the clamp (saturating conversion instead), leave the C4 alpha of
// 8u/32f untouched (16f copies the source alpha) and compute alpha from the
// fourth row for the "C" (20-float) forms.
// ---------------------------------------------------------------------------
template <typename S>
inline S batch_convert(float v) {
  if constexpr (std::is_same_v<S, Npp8u>) {
    const double t = std::trunc((double)v);
    unsigned long long u;
    if (!(t > 0.0)) u = 0;
    else if (t >= 4294967295.0) u = 4294967295ull;
    else u = (unsigned long long)t;
    return (Npp8u)(u & 0xFFull);
  } else {
    return from_float<S>(v);
  }
}

template <typename S, int C, int A, bool InPlace, bool WithConstants, class Call>
void run_twist_batch(const char *id, Call call) {
  NppStreamContext ctx{};
  if (nppGetStreamContext(&ctx) != NPP_NO_ERROR) {
    GTEST_SKIP() << "[" << id << "] no stream context";
    return;
  }
  RunCounts cnt;
  std::string first;
  const int B = 3;
  const int W = 33, H = 17;
  static const float mats3[3][12] = {
      {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0},
      {1, 0, 0, 5, 0, 1, 0, -7, 0, 0, 1, 11},
      {0.5f, 0.25f, 0.25f, 0, 0.25f, 0.5f, 0.25f, 0, 0.25f, 0.25f, 0.5f, 0},
  };
  // "C" batch layout: four rows of five floats (the fifth is the constant).
  static const float matsC[3][20] = {
      {1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0},
      {1, 0, 0, 0, 5, 0, 1, 0, 0, -7, 0, 0, 1, 0, 11, 0, 0, 0, 1, 3},
      {0.5f, 0.25f, 0.25f, 0, 0.25f, 0.25f, 0.5f, 0.25f, 0, 0.5f,
       0.25f, 0.25f, 0.5f, 0, 1, 0, 0, 0, 1, 4},
  };
  const float ranges[3][2] = {{0.0f, 255.0f}, {-50.0f, 300.0f}, {300.0f, -50.0f}};
  std::vector<Buf<S> *> srcs, dsts;
  std::vector<Buf<float> *> coeffs;
  std::vector<std::vector<uint8_t>> pre(B), orig(B);
  const uint64_t seed = case_seed(id, W, H, B, 0, 0, 0);
  for (int i = 0; i < B; ++i) {
    Buf<S> *a = new Buf<S>(W, H, C, 0, 0);
    Buf<S> *d = new Buf<S>(W, H, C, 0, 0x5A);
    Buf<float> *cf = new Buf<float>(WithConstants ? 20 : 12, 1, 1, 0, 0);
    fill_regime(*a, C, seed ^ mix64((uint64_t)i + 3), (i & 1) ? kWide : kSmall, i, false, false,
                std::is_floating_point_v<S>);
    if constexpr (std::is_floating_point_v<S>) {
      const size_t n = (size_t)(a->step / sizeof(S)) * (size_t)a->bh;
      for (size_t k = 0; k < n; ++k) {
        const double v = (double)a->hptr()[k];
        if (v > 1e3) a->hptr()[k] = S(1e3);
        else if (v < -1e3) a->hptr()[k] = S(-1e3);
        else if (v != v) a->hptr()[k] = S(0);
      }
    }
    a->dev.copyFromHost(a->host);
    d->dev.copyFromHost(d->host);
    if constexpr (WithConstants) std::memcpy(cf->host.data(), matsC[i], sizeof(matsC[i]));
    else std::memcpy(cf->host.data(), mats3[i], sizeof(mats3[i]));
    cf->dev.copyFromHost(cf->host);
    srcs.push_back(a);
    dsts.push_back(d);
    coeffs.push_back(cf);
    pre[(size_t)i] = d->host;
    orig[(size_t)i] = a->host;
  }
  std::vector<NppiColorTwistBatchCXR> h(B);
  for (int i = 0; i < B; ++i) {
    h[i] = {(const void *)srcs[(size_t)i]->dptr(), srcs[(size_t)i]->step,
            (void *)(InPlace ? srcs[(size_t)i]->dptr() : dsts[(size_t)i]->dptr()), dsts[(size_t)i]->step,
            (Npp32f *)coeffs[(size_t)i]->dptr()};
  }
  Buf<NppiColorTwistBatchCXR> list(B, 1, 1, 0, 0);
  std::memcpy(list.host.data(), h.data(), sizeof(NppiColorTwistBatchCXR) * B);
  list.dev.copyFromHost(list.host);
  const bool srcAlphaCopy = !WithConstants && C == 4 && std::is_same_v<S, Npp16f>;
  for (int ri = 0; ri < 3; ++ri) {
    for (int i = 0; i < B; ++i) {
      dsts[(size_t)i]->host = pre[(size_t)i];
      dsts[(size_t)i]->dev.copyFromHost(dsts[(size_t)i]->host);
      if constexpr (InPlace) {
        srcs[(size_t)i]->host = orig[(size_t)i];
        srcs[(size_t)i]->dev.copyFromHost(srcs[(size_t)i]->host);
      }
    }
    npp_diff::clear_cuda_error();
    const NppStatus st = call(ranges[ri][0], ranges[ri][1], NppiSize{W, H},
                              reinterpret_cast<NppiColorTwistBatchCXR *>(list.dptr()), B, ctx);
    if (st != NPP_NO_ERROR) {
      ++cnt.skipped;
      if (st == NPP_NOT_IMPLEMENTED_ERROR) ++cnt.notimpl;
      continue;
    }
    const cudaError_t se = cudaDeviceSynchronize();
    ASSERT_EQ(se, cudaSuccess) << "[" << id << "] batch sync: " << cudaGetErrorString(se);
    ++cnt.executed;
    bool failed = false;
    for (int i = 0; i < B && !failed; ++i) {
      std::vector<uint8_t> got;
      if constexpr (InPlace) srcs[(size_t)i]->dev.copyToHost(got);
      else dsts[(size_t)i]->dev.copyToHost(got);
      std::vector<uint8_t> want = InPlace ? orig[(size_t)i] : pre[(size_t)i];
      const int stp = InPlace ? srcs[(size_t)i]->step : dsts[(size_t)i]->step;
      for (int y = 0; y < H; ++y)
        for (int x = 0; x < W; ++x) {
          const size_t so = (size_t)y * (srcs[(size_t)i]->step / (int)sizeof(S)) + (size_t)x * C;
          const size_t db = off_bytes<S>(y, x, stp, C);
          float v[4] = {0, 0, 0, 0};
          for (int ch = 0; ch < C; ++ch) v[ch] = (float)as_double(srcs[(size_t)i]->hptr()[so + (size_t)ch]);
          const float *m = coeffs[(size_t)i]->hptr();
          const int nOut = WithConstants ? 4 : (C < 3 ? C : 3);
          for (int ch = 0; ch < nOut; ++ch) {
            float o;
            if constexpr (WithConstants) {
              o = m[ch * 5 + 0] * v[0] + m[ch * 5 + 1] * v[1] + m[ch * 5 + 2] * v[2] + m[ch * 5 + 3] * v[3] +
                  m[ch * 5 + 4];
            } else {
              o = m[ch * 4 + 0] * v[0] + m[ch * 4 + 1] * v[1] + m[ch * 4 + 2] * v[2] + m[ch * 4 + 3];
            }
            constexpr bool kBias = std::is_same_v<S, Npp32f> || (std::is_integral_v<S> && C >= 3);
            if (kBias) o += 0.5f;
            const bool exempt = !WithConstants && C == 4 && A == 3 && ch == 2;
            if (!exempt) {
              // NPP clamps min first and max second (independent comparisons),
              // so an inverted [nMin, nMax] range yields nMax.
              if (o < ranges[ri][0]) o = ranges[ri][0];
              if (o > ranges[ri][1]) o = ranges[ri][1];
            }
            // AC4 batch kernels convert every written channel with the
            // saturating conversion (8u: 400 -> 255); C3/C4 (and the "C"
            // forms) use the low-byte batch conversion.
            const bool satConv = !WithConstants && C == 4 && A == 3;
            const S out = satConv ? from_float<S>(o) : batch_convert<S>(o);
            std::memcpy(want.data() + db + (size_t)ch * sizeof(S), &out, sizeof(S));
          }
          if (!WithConstants) {
            if constexpr (C == 4) {
              if (srcAlphaCopy) {
                // 16f batch copies the source alpha.
                std::memcpy(want.data() + db + 3 * sizeof(S), srcs[(size_t)i]->hptr() + so + 3, sizeof(S));
              }
            }
          }
        }
      const GenCase dc{W, H, 0, 0, 0, kSmall, 0};
      const int Acheck = WithConstants ? 4 : (C == 4 ? (srcAlphaCopy ? 4 : 3) : C);
      if (!rotate_check<S, C, Acheck>(id, dc, NPPI_INTER_NN, InPlace ? orig[(size_t)i] : pre[(size_t)i], got,
                                      want, stp, std::vector<uint8_t>(), cnt, first))
        failed = true;
      if (cnt.mismatched >= 8) break;
    }
    if (failed) break;
  }
  for (auto *p : srcs) delete p;
  for (auto *p : dsts) delete p;
  for (auto *p : coeffs) delete p;
  finish(id, cnt, first);
}

}  // namespace twist
}  // namespace num
