// CPU-golden coverage for the nppi point / linear / cubic LUT families
// (nppiLUT, nppiLUT_Linear, nppiLUT_Cubic).
//
// The formulas reproduce the measured NVIDIA NPP 13.4 semantics:
//   * a forward level scan returns -1 (copy the source value through) when
//     the sample is below levels[0] or at/above levels[nLevels-1];
//   * point => values[idx]; linear => y0 + (s - x0) * (y1 - y0) / (x1 - x0)
//     with C truncating integer division for integer dtypes and float math
//     for 32f; cubic => Newton/divided-difference cubic through four points
//     with linear extrapolation of the missing ends, evaluated with the
//     SASS-derived fmaf ordering;
//   * the 8u families short-circuit to dst = values[src] when every used
//     channel has exactly 256 levels (the level array is not consulted);
//   * AC4 layouts carry three tables and leave the destination alpha byte
//     untouched.
#pragma once

#include "num_nppi_geom.h"

namespace num {
namespace lut {

enum class LutMode { Point, Linear, Cubic };

template <typename V>
inline int find_index(double s, const V *levels, int nLevels) {
  if (s < (double)levels[0] || s >= (double)levels[nLevels - 1]) return -1;
  int idx = 0;
  for (int i = 1; i < nLevels; ++i) {
    if (s >= (double)levels[i]) idx = i;
    else break;
  }
  return idx < nLevels - 1 ? idx : nLevels - 2;
}

template <typename S, typename V, LutMode Mode>
inline V sample(S src, const V *values, const V *levels, int nLevels) {
  const double s = (double)src;
  const int idx = find_index<V>(s, levels, nLevels);
  if (idx < 0) return (V)src;
  if constexpr (Mode == LutMode::Point) {
    return values[idx];
  } else if constexpr (Mode == LutMode::Linear) {
    if constexpr (std::is_same_v<V, float>) {
      const float sf = (float)src;
      const float y0 = (float)values[idx], y1 = (float)values[idx + 1];
      const float x0 = (float)levels[idx], x1 = (float)levels[idx + 1];
      return (V)(y0 + (sf - x0) * (y1 - y0) / (x1 - x0));
    } else {
      const long long sl = (long long)s;
      const long long x0 = (long long)levels[idx], x1 = (long long)levels[idx + 1];
      const long long y0 = (long long)values[idx], y1 = (long long)values[idx + 1];
      return (V)(y0 + (sl - x0) * (y1 - y0) / (x1 - x0));
    }
  } else {
    // Newton divided-difference cubic with the NPP fmaf ordering.
    const float x0 = (float)levels[idx];
    const float x1 = (float)levels[idx + 1];
    const float v0 = (float)values[idx];
    const float v1 = (float)values[idx + 1];
    const float xm = idx >= 1 ? (float)levels[idx - 1] : 2.0f * x0 - x1;
    const float vm = idx >= 1 ? (float)values[idx - 1] : 2.0f * v0 - v1;
    const float x2 = idx + 2 < nLevels ? (float)levels[idx + 2] : 2.0f * x1 - x0;
    const float v2 = idx + 2 < nLevels ? (float)values[idx + 2] : 2.0f * v1 - v0;
    const float s2 = (v2 - v1) / (x2 - x1);
    const float s1 = (v1 - v0) / (x1 - x0);
    const float s0 = (v0 - vm) / (x0 - xm);
    const float dx = x1 - x0;
    const float d0 = 2.0f * (x1 - xm);
    const float q = dx / d0;
    const float a0 = 6.0f * (s1 - s0);
    const float b0 = 6.0f * (s2 - s1);
    const float e1 = std::fma(-q, dx, 2.0f * (x2 - x0));
    const float b = std::fma(-q, a0, b0) / e1;
    const float a = (a0 - dx * b) / d0;
    const float c3 = (b - a) / (6.0f * dx);
    const float c1 = std::fma(std::fma(2.0f * dx, a, dx * b), -1.0f / 6.0f, s1);
    const float h = (float)src - x0;
    const float h2 = h * h;
    const float result = std::fma(c1, h, std::fma(a * 0.5f, h2, h2 * h * c3));
    if constexpr (std::is_same_v<V, float>) {
      return v0 + result;
    } else {
      return (V)(Npp32s)(v0 + result);
    }
  }
}

template <typename S, typename V>
inline S cast_lut(V value) {
  if constexpr (std::is_same_v<S, float>) {
    return (S)value;
  } else {
    return (S)(Npp32s)value;
  }
}

// Table-set builders.  All arrays are deterministic and cover the dtype range.
template <typename S, typename V>
struct LutTables {
  std::vector<V> low, high;
  std::vector<V> lev[4];
  std::vector<V> val[4];
  int n[4] = {0, 0, 0, 0};
  bool fastPath = false;
};

template <typename S> struct LutRange;
template <> struct LutRange<Npp8u> {
  static constexpr double lo = 0.0, hi = 255.0;
  static constexpr bool isInt = true;
  static constexpr int maxLevels = 256;
};
template <> struct LutRange<Npp16u> {
  static constexpr double lo = 0.0, hi = 65535.0;
  static constexpr bool isInt = true;
  static constexpr int maxLevels = 1024;
};
template <> struct LutRange<Npp16s> {
  static constexpr double lo = -32768.0, hi = 32767.0;
  static constexpr bool isInt = true;
  static constexpr int maxLevels = 1024;
};
template <> struct LutRange<Npp32f> {
  static constexpr double lo = -1000.0, hi = 1000.0;
  static constexpr bool isInt = false;
  static constexpr int maxLevels = 1024;
};

template <typename S, typename V>
inline V make_val(double v) {
  if constexpr (std::is_same_v<V, float>) return (V)v;
  else return (V)std::llround(v);
}

template <typename S, typename V, int C>
inline void build_tables(LutTables<S, V> &t, int variant) {
  using R = LutRange<S>;
  const double lo = R::lo, hi = R::hi, span = hi - lo;
  auto set_levels = [&](int c, int n, double f0, double f1, bool unsorted) {
    t.lev[c].resize((size_t)n);
    t.val[c].resize((size_t)n);
    for (int i = 0; i < n; ++i) {
      double f = (double)i / (double)(n - 1);
      if (unsorted) {
        // Deterministic permutation of the interior indices.
        if (i > 0 && i < n - 1) {
          const int j = 1 + ((i * 7 + c * 3) % (n - 2));
          f = (double)j / (double)(n - 1);
        }
      }
      t.lev[c][(size_t)i] = make_val<S, V>(lo + (f0 + (f1 - f0) * f) * span);
      const double vf = 0.08 + 0.84 * (double)(((i * 37 + c * 11) % 97)) / 96.0;
      t.val[c][(size_t)i] = make_val<S, V>(lo + vf * span);
    }
    t.n[c] = n;
  };
  switch (variant) {
    case 0:  // five levels spanning most of the range
      for (int c = 0; c < C; ++c) set_levels(c, 5, 0.0, 0.9375, false);
      break;
    case 1: {  // exactly 256 levels (sorted; 8u takes the values[src] shortcut)
      const int n = std::min(256, R::maxLevels);
      for (int c = 0; c < C; ++c) {
        set_levels(c, n, 0.0, 1.0, false);
      }
      t.fastPath = true;
      break;
    }
    case 2:  // narrow window: both tails copy the source through
      for (int c = 0; c < C; ++c) set_levels(c, 5, 0.25, 0.75, false);
      break;
    default:  // many levels but not 256 (no shortcut)
      for (int c = 0; c < C; ++c) set_levels(c, 137, 0.1, 0.9, false);
      break;
  }
}

// Source ramp that walks the whole dtype range.
template <typename S, int C>
inline void fill_ramp(Buf<S> &b, int lo_off) {
  using R = LutRange<S>;
  const double span = R::hi - R::lo;
  const int w = b.bw, h = b.bh;
  for (int y = 0; y < h; ++y)
    for (int x = 0; x < w; ++x)
      for (int c = 0; c < C; ++c) {
        const long long k = (long long)x * 7 + (long long)y * 13 + (long long)c * 29 + lo_off;
        const double f = (double)(k % 1009) / 1008.0;
        const double v = R::lo + f * span;
        reinterpret_cast<S *>(b.host.data())[(size_t)y * (b.step / (int)sizeof(S)) + (size_t)x * C + c] = (S)v;
      }
}

template <typename S, typename V, int C, LutMode Mode>
inline void ref_apply(const S *hsrc, int sstep, int w, int h, int A, const LutTables<S, V> &t, bool fast,
                      const std::vector<V *> &hvals, const std::vector<V *> &hlev, const int *nlev,
                      std::vector<uint8_t> &want, int dstep, int dstOx, int dstOy) {
  for (int y = 0; y < h; ++y)
    for (int x = 0; x < w; ++x) {
      const size_t srcOff = (size_t)y * sstep + (size_t)x * C;
      const size_t db = off_bytes<S>(dstOy + y, dstOx + x, dstep, C);
      for (int c = 0; c < A; ++c) {
        const S src = hsrc[srcOff + (size_t)c];
        S out;
        if (fast) {
          out = cast_lut<S>(((const V *)hvals[(size_t)c])[(int)src]);
        } else {
          out = cast_lut<S>(sample<S, V, Mode>(src, hvals[(size_t)c], hlev[(size_t)c], nlev[c]));
        }
        std::memcpy(want.data() + db + (size_t)c * sizeof(S), &out, sizeof(S));
      }
    }
  (void)t;
}

// ---------------------------------------------------------------------------
// C1 drivers.
// ---------------------------------------------------------------------------
template <typename S, LutMode Mode, class Call>
void run_lut_c1(const char *id, Call call) {
  using V = typename std::conditional<std::is_same_v<S, Npp32f>, float, Npp32s>::type;
  NppStreamContext ctx{};
  if (nppGetStreamContext(&ctx) != NPP_NO_ERROR) {
    GTEST_SKIP() << "[" << id << "] no stream context";
    return;
  }
  RunCounts cnt;
  std::string first;
  for (const GenCase &c : make_gen_cases(0)) {
    if (c.w < 2 || c.h < 2) continue;
    Buf<S> s(c.w + c.ox, c.h + c.oy, 1, c.pad, 0);
    Buf<S> d(c.w, c.h, 1, 0, 0x5A);
    fill_ramp<S, 1>(s, (int)(case_seed(id, c.w, c.h, s.step, d.step, 0, 0) & 0xFF));
    s.dev.copyFromHost(s.host);
    d.dev.copyFromHost(d.host);
    const S *hsrc = s.hptr() + off_bytes<S>(c.oy, c.ox, s.step, 1) / sizeof(S);
    const S *dsrc = reinterpret_cast<const S *>(s.dptr() + off_bytes<S>(c.oy, c.ox, s.step, 1));
    for (int variant = 0; variant < 4; ++variant) {
      LutTables<S, V> t;
      build_tables<S, V, 1>(t, variant);
      const bool fast = std::is_same_v<S, Npp8u> && t.fastPath;
      Buf<V> lev((int)t.lev[0].size(), 1, 1, 0, 0);
      Buf<V> val((int)t.val[0].size(), 1, 1, 0, 0);
      std::memcpy(lev.host.data(), t.lev[0].data(), t.lev[0].size() * sizeof(V));
      std::memcpy(val.host.data(), t.val[0].data(), t.val[0].size() * sizeof(V));
      lev.dev.copyFromHost(lev.host);
      val.dev.copyFromHost(val.host);
      const std::vector<uint8_t> pre = d.host;
      npp_diff::clear_cuda_error();
      const NppStatus st = call(dsrc, s.step, reinterpret_cast<S *>(d.dptr()), d.step,
                                NppiSize{c.w, c.h}, reinterpret_cast<const V *>(val.dptr()),
                                reinterpret_cast<const V *>(lev.dptr()), t.n[0], ctx);
      if (st != NPP_NO_ERROR) {
        ++cnt.skipped;
        if (st == NPP_NOT_IMPLEMENTED_ERROR) ++cnt.notimpl;
        continue;
      }
      const cudaError_t se = cudaDeviceSynchronize();
      ASSERT_EQ(se, cudaSuccess) << "[" << id << "] " << c.id() << " sync: " << cudaGetErrorString(se);
      d.dev.copyToHost(d.host);
      ++cnt.executed;
      std::vector<uint8_t> want = pre;
      std::vector<V *> hv{(V *)t.val[0].data()}, hl{(V *)t.lev[0].data()};
      int nl[1] = {t.n[0]};
      ref_apply<S, V, 1, Mode>(hsrc, s.step / (int)sizeof(S), c.w, c.h, 1, t, fast, hv, hl, nl, want, d.step, 0, 0);
      const GenCase dc{c.w, c.h, 0, 0, 0, kSmall, 0};
      if (!rotate_check<S, 1, 1>(id, dc, NPPI_INTER_NN, pre, d.host, want, d.step,
                                 std::vector<uint8_t>(), cnt, first)) {
        if (cnt.mismatched >= 8) break;
      }
      if (cnt.mismatched >= 8) break;
    }
    if (cnt.mismatched >= 8) break;
  }
  finish(id, cnt, first);
}

template <typename S, LutMode Mode, class Call>
void run_lut_c1_ir(const char *id, Call call) {
  using V = typename std::conditional<std::is_same_v<S, Npp32f>, float, Npp32s>::type;
  NppStreamContext ctx{};
  if (nppGetStreamContext(&ctx) != NPP_NO_ERROR) {
    GTEST_SKIP() << "[" << id << "] no stream context";
    return;
  }
  RunCounts cnt;
  std::string first;
  for (const GenCase &c : make_gen_cases(0)) {
    if (c.w < 2 || c.h < 2) continue;
    Buf<S> s(c.w + c.ox, c.h + c.oy, 1, c.pad, 0);
    fill_ramp<S, 1>(s, (int)(case_seed(id, c.w, c.h, s.step, 0, 0, 0) & 0xFF));
    const std::vector<uint8_t> orig = s.host;
    s.dev.copyFromHost(s.host);
    for (int variant = 0; variant < 4; ++variant) {
      s.host = orig;
      s.dev.copyFromHost(s.host);
      LutTables<S, V> t;
      build_tables<S, V, 1>(t, variant);
      const bool fast = std::is_same_v<S, Npp8u> && t.fastPath;
      Buf<V> lev((int)t.lev[0].size(), 1, 1, 0, 0);
      Buf<V> val((int)t.val[0].size(), 1, 1, 0, 0);
      std::memcpy(lev.host.data(), t.lev[0].data(), t.lev[0].size() * sizeof(V));
      std::memcpy(val.host.data(), t.val[0].data(), t.val[0].size() * sizeof(V));
      lev.dev.copyFromHost(lev.host);
      val.dev.copyFromHost(val.host);
      const std::vector<uint8_t> pre = orig;
      const S *hsrc = s.hptr() + off_bytes<S>(c.oy, c.ox, s.step, 1) / sizeof(S);
      npp_diff::clear_cuda_error();
      const NppStatus st = call(reinterpret_cast<S *>(s.dptr() + off_bytes<S>(c.oy, c.ox, s.step, 1)), s.step,
                                NppiSize{c.w, c.h}, reinterpret_cast<const V *>(val.dptr()),
                                reinterpret_cast<const V *>(lev.dptr()), t.n[0], ctx);
      if (st != NPP_NO_ERROR) {
        ++cnt.skipped;
        if (st == NPP_NOT_IMPLEMENTED_ERROR) ++cnt.notimpl;
        continue;
      }
      const cudaError_t se = cudaDeviceSynchronize();
      ASSERT_EQ(se, cudaSuccess) << "[" << id << "] " << c.id() << " sync: " << cudaGetErrorString(se);
      std::vector<uint8_t> got;
      s.dev.copyToHost(got);
      ++cnt.executed;
      std::vector<uint8_t> want = pre;
      std::vector<V *> hv{(V *)t.val[0].data()}, hl{(V *)t.lev[0].data()};
      int nl[1] = {t.n[0]};
      ref_apply<S, V, 1, Mode>(hsrc, s.step / (int)sizeof(S), c.w, c.h, 1, t, fast, hv, hl, nl, want, s.step, c.ox, c.oy);
      const GenCase dc{c.w, c.h, c.ox, c.oy, 0, kSmall, 0};
      if (!rotate_check<S, 1, 1>(id, dc, NPPI_INTER_NN, pre, got, want, s.step,
                                 std::vector<uint8_t>(), cnt, first)) {
        if (cnt.mismatched >= 8) break;
      }
      if (cnt.mismatched >= 8) break;
    }
    if (cnt.mismatched >= 8) break;
  }
  finish(id, cnt, first);
}

// ---------------------------------------------------------------------------
// Multi-channel (C3 / C4 / AC4) drivers.  A is the number of channels the
// tables address (3 for AC4, otherwise C); the remaining channels must keep
// their destination bytes unchanged.
// ---------------------------------------------------------------------------
template <typename S, int C, int A, LutMode Mode, class Call>
void run_lut_multi(const char *id, Call call) {
  using V = typename std::conditional<std::is_same_v<S, Npp32f>, float, Npp32s>::type;
  NppStreamContext ctx{};
  if (nppGetStreamContext(&ctx) != NPP_NO_ERROR) {
    GTEST_SKIP() << "[" << id << "] no stream context";
    return;
  }
  RunCounts cnt;
  std::string first;
  for (const GenCase &c : make_gen_cases(0)) {
    if (c.w < 2 || c.h < 2) continue;
    Buf<S> s(c.w + c.ox, c.h + c.oy, C, c.pad, 0);
    Buf<S> d(c.w, c.h, C, 0, 0x5A);
    fill_ramp<S, C>(s, (int)(case_seed(id, c.w, c.h, s.step, d.step, 0, 0) & 0xFF));
    s.dev.copyFromHost(s.host);
    d.dev.copyFromHost(d.host);
    const S *hsrc = s.hptr() + off_bytes<S>(c.oy, c.ox, s.step, C) / sizeof(S);
    const S *dsrc = reinterpret_cast<const S *>(s.dptr() + off_bytes<S>(c.oy, c.ox, s.step, C));
    for (int variant = 0; variant < 4; ++variant) {
      LutTables<S, V> t;
      build_tables<S, V, A>(t, variant);
      bool fast = std::is_same_v<S, Npp8u>;
      for (int p = 0; p < A; ++p)
        if (t.n[p] != 256) fast = false;
      std::vector<Buf<V> *> lev(A), val(A);
      const V *pv[4] = {nullptr, nullptr, nullptr, nullptr};
      const V *pl[4] = {nullptr, nullptr, nullptr, nullptr};
      int nl[4] = {0, 0, 0, 0};
      for (int p = 0; p < A; ++p) {
        lev[(size_t)p] = new Buf<V>((int)t.lev[p].size(), 1, 1, 0, 0);
        val[(size_t)p] = new Buf<V>((int)t.val[p].size(), 1, 1, 0, 0);
        std::memcpy(lev[(size_t)p]->host.data(), t.lev[p].data(), t.lev[p].size() * sizeof(V));
        std::memcpy(val[(size_t)p]->host.data(), t.val[p].data(), t.val[p].size() * sizeof(V));
        lev[(size_t)p]->dev.copyFromHost(lev[(size_t)p]->host);
        val[(size_t)p]->dev.copyFromHost(val[(size_t)p]->host);
        pv[p] = reinterpret_cast<const V *>(val[(size_t)p]->dptr());
        pl[p] = reinterpret_cast<const V *>(lev[(size_t)p]->dptr());
        nl[p] = t.n[p];
      }
      const std::vector<uint8_t> pre = d.host;
      npp_diff::clear_cuda_error();
      const NppStatus st = call(dsrc, s.step, reinterpret_cast<S *>(d.dptr()), d.step,
                                NppiSize{c.w, c.h}, pv, pl, nl, ctx);
      if (st != NPP_NO_ERROR) {
        ++cnt.skipped;
        if (st == NPP_NOT_IMPLEMENTED_ERROR) ++cnt.notimpl;
        for (auto *p : lev) delete p;
        for (auto *p : val) delete p;
        continue;
      }
      const cudaError_t se = cudaDeviceSynchronize();
      ASSERT_EQ(se, cudaSuccess) << "[" << id << "] " << c.id() << " sync: " << cudaGetErrorString(se);
      d.dev.copyToHost(d.host);
      ++cnt.executed;
      std::vector<uint8_t> want = pre;
      std::vector<V *> hv, hl;
      for (int p = 0; p < A; ++p) {
        hv.push_back((V *)t.val[p].data());
        hl.push_back((V *)t.lev[p].data());
      }
      ref_apply<S, V, C, Mode>(hsrc, s.step / (int)sizeof(S), c.w, c.h, A, t, fast, hv, hl, nl, want, d.step, 0, 0);
      const GenCase dc{c.w, c.h, 0, 0, 0, kSmall, 0};
      if (!rotate_check<S, C, A>(id, dc, NPPI_INTER_NN, pre, d.host, want, d.step,
                                 std::vector<uint8_t>(), cnt, first)) {
        if (cnt.mismatched >= 8) break;
      }
      for (auto *p : lev) delete p;
      for (auto *p : val) delete p;
      if (cnt.mismatched >= 8) break;
    }
    if (cnt.mismatched >= 8) break;
  }
  finish(id, cnt, first);
}

template <typename S, int C, int A, LutMode Mode, class Call>
void run_lut_multi_ir(const char *id, Call call) {
  using V = typename std::conditional<std::is_same_v<S, Npp32f>, float, Npp32s>::type;
  NppStreamContext ctx{};
  if (nppGetStreamContext(&ctx) != NPP_NO_ERROR) {
    GTEST_SKIP() << "[" << id << "] no stream context";
    return;
  }
  RunCounts cnt;
  std::string first;
  for (const GenCase &c : make_gen_cases(0)) {
    if (c.w < 2 || c.h < 2) continue;
    Buf<S> s(c.w + c.ox, c.h + c.oy, C, c.pad, 0);
    fill_ramp<S, C>(s, (int)(case_seed(id, c.w, c.h, s.step, 0, 0, 0) & 0xFF));
    const std::vector<uint8_t> orig = s.host;
    s.dev.copyFromHost(s.host);
    for (int variant = 0; variant < 4; ++variant) {
      s.host = orig;
      s.dev.copyFromHost(s.host);
      LutTables<S, V> t;
      build_tables<S, V, A>(t, variant);
      bool fast = std::is_same_v<S, Npp8u>;
      for (int p = 0; p < A; ++p)
        if (t.n[p] != 256) fast = false;
      std::vector<Buf<V> *> lev(A), val(A);
      const V *pv[4] = {nullptr, nullptr, nullptr, nullptr};
      const V *pl[4] = {nullptr, nullptr, nullptr, nullptr};
      int nl[4] = {0, 0, 0, 0};
      for (int p = 0; p < A; ++p) {
        lev[(size_t)p] = new Buf<V>((int)t.lev[p].size(), 1, 1, 0, 0);
        val[(size_t)p] = new Buf<V>((int)t.val[p].size(), 1, 1, 0, 0);
        std::memcpy(lev[(size_t)p]->host.data(), t.lev[p].data(), t.lev[p].size() * sizeof(V));
        std::memcpy(val[(size_t)p]->host.data(), t.val[p].data(), t.val[p].size() * sizeof(V));
        lev[(size_t)p]->dev.copyFromHost(lev[(size_t)p]->host);
        val[(size_t)p]->dev.copyFromHost(val[(size_t)p]->host);
        pv[p] = reinterpret_cast<const V *>(val[(size_t)p]->dptr());
        pl[p] = reinterpret_cast<const V *>(lev[(size_t)p]->dptr());
        nl[p] = t.n[p];
      }
      const std::vector<uint8_t> pre = orig;
      const S *hsrc = s.hptr() + off_bytes<S>(c.oy, c.ox, s.step, C) / sizeof(S);
      npp_diff::clear_cuda_error();
      const NppStatus st = call(reinterpret_cast<S *>(s.dptr() + off_bytes<S>(c.oy, c.ox, s.step, C)), s.step,
                                NppiSize{c.w, c.h}, pv, pl, nl, ctx);
      if (st != NPP_NO_ERROR) {
        ++cnt.skipped;
        if (st == NPP_NOT_IMPLEMENTED_ERROR) ++cnt.notimpl;
        for (auto *p : lev) delete p;
        for (auto *p : val) delete p;
        continue;
      }
      const cudaError_t se = cudaDeviceSynchronize();
      ASSERT_EQ(se, cudaSuccess) << "[" << id << "] " << c.id() << " sync: " << cudaGetErrorString(se);
      std::vector<uint8_t> got;
      s.dev.copyToHost(got);
      ++cnt.executed;
      std::vector<uint8_t> want = pre;
      std::vector<V *> hv, hl;
      for (int p = 0; p < A; ++p) {
        hv.push_back((V *)t.val[p].data());
        hl.push_back((V *)t.lev[p].data());
      }
      ref_apply<S, V, C, Mode>(hsrc, s.step / (int)sizeof(S), c.w, c.h, A, t, fast, hv, hl, nl, want, s.step, c.ox, c.oy);
      const GenCase dc{c.w, c.h, c.ox, c.oy, 0, kSmall, 0};
      if (!rotate_check<S, C, A>(id, dc, NPPI_INTER_NN, pre, got, want, s.step,
                                 std::vector<uint8_t>(), cnt, first)) {
        if (cnt.mismatched >= 8) break;
      }
      for (auto *p : lev) delete p;
      for (auto *p : val) delete p;
      if (cnt.mismatched >= 8) break;
    }
    if (cnt.mismatched >= 8) break;
  }
  finish(id, cnt, first);
}

}  // namespace lut
}  // namespace num
