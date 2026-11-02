#pragma once

// One-dimensional (npps) numerical conformance drivers.  Signal APIs take a
// pointer and an element count instead of a 2D ROI/step, so they use their own
// length matrix: tiny, odd, vector-width boundaries and a large multi-block
// length, each under the three value regimes.

#include "num_suite.h"

namespace num {

struct NppsCase {
  int n;
  Regime regime;
  int param;
  std::string id() const {
    char b[64];
    std::snprintf(b, sizeof(b), "n%d/r%d/p%d", n, (int)regime, param);
    return b;
  }
};

inline std::vector<NppsCase> make_npps_cases(int param) {
  static const int lengths[] = {1, 2, 3, 7, 16, 33, 64, 257, 1024, 65537};
  static const Regime regimes[] = {kSmall, kWide, kExtremes};
  std::vector<NppsCase> v;
  for (int n : lengths)
    for (Regime r : regimes) v.push_back({n, r, param});
  return v;
}

template <typename T> inline void fill_npps(Buf<T> &b, uint64_t seed, Regime regime, int salt = 0,
                                            bool finite_only = false) {
  Rng rng(seed);
  T *p = b.hptr();
  for (int i = 0; i < b.bw; ++i) {
    T v = regime_value<T>(rng, regime, (size_t)i + (size_t)salt);
    if constexpr (std::is_same_v<T, Npp64s>) {
      // NPP's 64s Sfs kernels overflow (wrap) at the extremes; keep the
      // reference inputs inside a safe range where the arithmetic is exact.
      const long long lim = 1LL << 60;
      if (v > (T)lim) v = (T)lim;
      if (v < (T)(-lim)) v = (T)(-lim);
    }
    if constexpr (std::is_same_v<T, Npp16f>) {
      if (finite_only && !std::isfinite(half_to_float_host(v))) v = half_round(0.0f);
    } else if constexpr (std::is_floating_point_v<T>) {
      if (finite_only && !std::isfinite((double)v)) v = T(0);
    }
    p[i] = v;
  }
}

// ---------------------------------------------------------------------------
// Copy / Set / Zero.
// ---------------------------------------------------------------------------
template <typename T, class Call>
void run_npps_copy(const char *id, Call call) {
  NppStreamContext ctx{};
  if (nppGetStreamContext(&ctx) != NPP_NO_ERROR) {
    GTEST_SKIP() << "[" << id << "] no stream context";
    return;
  }
  RunCounts cnt;
  std::string first;
  for (const NppsCase &c : make_npps_cases(0)) {
    Buf<T> s(c.n, 1, 1, 0, 0xCD);
    Buf<T> d(c.n, 1, 1, 0, 0x5A);
    const uint64_t seed = case_seed(id, c.n, 1, 0, 0, (int)c.regime, 0);
    fill_npps(s, seed, c.regime, 0);
    s.dev.copyFromHost(s.host);
    d.dev.copyFromHost(d.host);
    npp_diff::clear_cuda_error();
    NppStatus st = call(reinterpret_cast<const T *>(s.dptr()), reinterpret_cast<T *>(d.dptr()), c.n, ctx);
    if (st != NPP_NO_ERROR) {
      ++cnt.skipped;
      if (st == NPP_NOT_IMPLEMENTED_ERROR) ++cnt.notimpl;
      continue;
    }
    const cudaError_t se = cudaDeviceSynchronize();
    ASSERT_EQ(se, cudaSuccess) << "[" << id << "] " << c.id() << " sync: " << cudaGetErrorString(se);
    d.dev.copyToHost(d.host);
    ++cnt.executed;
    bool case_failed = false;
    for (int i = 0; i < c.n && !case_failed; ++i) {
      const size_t off = (size_t)i * sizeof(T);
      ++cnt.compared;
      if (!equal_elem<T>(d.host.data() + off, s.host.data() + off, 0, 0)) {
        char buf[280];
        std::snprintf(buf, sizeof(buf), "[%s] %s mismatch at i=%d: got=%s want=%s", id,
                      c.id().c_str(), i, dbg_elem<T>(d.host.data() + off).c_str(),
                      dbg_elem<T>(s.host.data() + off).c_str());
        if (first.empty()) first = buf;
        EXPECT_TRUE(false) << buf;
        ++cnt.mismatched;
        case_failed = true;
      }
    }
    if (cnt.mismatched >= 8) break;
  }
  finish(id, cnt, first);
}

template <typename T, class Call>
void run_npps_set(const char *id, Call call) {
  NppStreamContext ctx{};
  if (nppGetStreamContext(&ctx) != NPP_NO_ERROR) {
    GTEST_SKIP() << "[" << id << "] no stream context";
    return;
  }
  RunCounts cnt;
  std::string first;
  for (const NppsCase &c : make_npps_cases(0)) {
    Buf<T> d(c.n, 1, 1, 0, 0x5A);
    const uint64_t seed = case_seed(id, c.n, 1, 0, 0, (int)c.regime, 0);
    Rng rng(seed);
    const T value = regime_value<T>(rng, c.regime, 3);
    d.dev.copyFromHost(d.host);
    npp_diff::clear_cuda_error();
    NppStatus st = call(value, reinterpret_cast<T *>(d.dptr()), c.n, ctx);
    if (st != NPP_NO_ERROR) {
      ++cnt.skipped;
      if (st == NPP_NOT_IMPLEMENTED_ERROR) ++cnt.notimpl;
      continue;
    }
    const cudaError_t se = cudaDeviceSynchronize();
    ASSERT_EQ(se, cudaSuccess) << "[" << id << "] " << c.id() << " sync: " << cudaGetErrorString(se);
    d.dev.copyToHost(d.host);
    ++cnt.executed;
    bool case_failed = false;
    for (int i = 0; i < c.n && !case_failed; ++i) {
      const size_t off = (size_t)i * sizeof(T);
      ++cnt.compared;
      if (!equal_elem<T>(d.host.data() + off, reinterpret_cast<const uint8_t *>(&value), 0, 0)) {
        char buf[280];
        std::snprintf(buf, sizeof(buf), "[%s] %s mismatch at i=%d: got=%s want=%s", id,
                      c.id().c_str(), i, dbg_elem<T>(d.host.data() + off).c_str(),
                      dbg_elem<T>(reinterpret_cast<const uint8_t *>(&value)).c_str());
        if (first.empty()) first = buf;
        EXPECT_TRUE(false) << buf;
        ++cnt.mismatched;
        case_failed = true;
      }
    }
    if (cnt.mismatched >= 8) break;
  }
  finish(id, cnt, first);
}

template <typename T, class Call>
void run_npps_zero(const char *id, Call call) {
  NppStreamContext ctx{};
  if (nppGetStreamContext(&ctx) != NPP_NO_ERROR) {
    GTEST_SKIP() << "[" << id << "] no stream context";
    return;
  }
  RunCounts cnt;
  std::string first;
  const T zero{};
  for (const NppsCase &c : make_npps_cases(0)) {
    Buf<T> d(c.n, 1, 1, 0, 0x5A);
    d.dev.copyFromHost(d.host);
    npp_diff::clear_cuda_error();
    NppStatus st = call(reinterpret_cast<T *>(d.dptr()), c.n, ctx);
    if (st != NPP_NO_ERROR) {
      ++cnt.skipped;
      if (st == NPP_NOT_IMPLEMENTED_ERROR) ++cnt.notimpl;
      continue;
    }
    const cudaError_t se = cudaDeviceSynchronize();
    ASSERT_EQ(se, cudaSuccess) << "[" << id << "] " << c.id() << " sync: " << cudaGetErrorString(se);
    d.dev.copyToHost(d.host);
    ++cnt.executed;
    bool case_failed = false;
    for (int i = 0; i < c.n && !case_failed; ++i) {
      const size_t off = (size_t)i * sizeof(T);
      ++cnt.compared;
      if (!equal_elem<T>(d.host.data() + off, reinterpret_cast<const uint8_t *>(&zero), 0, 0)) {
        char buf[280];
        std::snprintf(buf, sizeof(buf), "[%s] %s mismatch at i=%d: got=%s want=0", id,
                      c.id().c_str(), i, dbg_elem<T>(d.host.data() + off).c_str());
        if (first.empty()) first = buf;
        EXPECT_TRUE(false) << buf;
        ++cnt.mismatched;
        case_failed = true;
      }
    }
    if (cnt.mismatched >= 8) break;
  }
  finish(id, cnt, first);
}

// ---------------------------------------------------------------------------
// Convert (plain / Sfs).  Same rounding rules as the nppi Convert family:
// integer destinations truncate toward zero unless a round mode is supplied;
// Sfs divides by 2^sf first; float destinations keep the fractional value.
// ---------------------------------------------------------------------------
template <typename S, typename D, class Call>
void run_npps_convert(const char *id, Call call, bool sfs, bool round_mode,
                      bool fin_half_up = false) {
  NppStreamContext ctx{};
  if (nppGetStreamContext(&ctx) != NPP_NO_ERROR) {
    GTEST_SKIP() << "[" << id << "] no stream context";
    return;
  }
  const int modes[3] = {NPP_RND_NEAR, NPP_RND_FINANCIAL, NPP_RND_ZERO};
  RunCounts cnt;
  std::string first;
  for (int sf : {0, 1, 2}) {
    if (!sfs && sf != 0) break;
    for (const NppsCase &c : make_npps_cases(sf)) {
      Buf<S> s(c.n, 1, 1, 0, 0);
      Buf<D> d(c.n, 1, 1, 0, 0x5A);
      const uint64_t seed = case_seed(id, c.n, 1, 0, 0, (int)c.regime, sf);
      const bool finite_only = (sfs || round_mode) &&
                               (std::is_floating_point_v<S> || std::is_same_v<S, Npp16f>);
      fill_npps(s, seed, c.regime, 0, finite_only);
      s.dev.copyFromHost(s.host);
      d.dev.copyFromHost(d.host);
      for (int mi = 0; mi < (round_mode ? 3 : 1); ++mi) {
        // Sfs variants without an explicit round-mode parameter use the
        // default nearest/ties-to-even rule.
        const int mode = round_mode ? modes[mi] : NPP_RND_NEAR;
        npp_diff::clear_cuda_error();
        NppStatus st = call(reinterpret_cast<const S *>(s.dptr()), reinterpret_cast<D *>(d.dptr()),
                            c.n, mode, sf, ctx);
        if (st != NPP_NO_ERROR) {
          ++cnt.skipped;
          if (st == NPP_NOT_IMPLEMENTED_ERROR) ++cnt.notimpl;
          continue;
        }
        const cudaError_t se = cudaDeviceSynchronize();
        ASSERT_EQ(se, cudaSuccess) << "[" << id << "] " << c.id()
                                   << " sync: " << cudaGetErrorString(se);
        d.dev.copyToHost(d.host);
        std::vector<uint8_t> want = d.host;
        for (int i = 0; i < c.n; ++i) {
          double v = as_double(s.hptr()[i]);
          if (sfs) v /= (double)(1ll << sf);
          if constexpr (std::is_integral_v<D>) {
            if (round_mode) {
              // Most npps kernels round FINANCIAL ties away from zero;
              // 64s->32s uses the integer half-up form instead.
              v = round_mode_host(v, mode, fin_half_up);
            } else if (sfs) {
              // Sfs without eRoundMode defaults to round-half-away-from-zero.
              v = v >= 0.0 ? std::floor(v + 0.5) : std::ceil(v - 0.5);
            }
          }
          D r;
          if constexpr (std::is_integral_v<D>) {
            if (std::isnan(v)) v = -std::numeric_limits<double>::infinity();
            double t = v <= (double)std::numeric_limits<long long>::min()
                           ? (double)std::numeric_limits<long long>::min()
                           : (v >= (double)std::numeric_limits<long long>::max()
                                  ? (double)std::numeric_limits<long long>::max()
                                  : v);
            r = saturate_convert<D>((double)(long long)t);
          } else {
            r = saturate_convert<D>(v);
          }
          const size_t off = (size_t)i * sizeof(D);
          std::memcpy(want.data() + off, &r, sizeof(D));
        }
        ++cnt.executed;
        bool case_failed = false;
        const double tol = std::is_same_v<D, Npp32f> ? 1e-6 : (std::is_same_v<D, Npp64f> ? 1e-12 : 0.0);
        for (int i = 0; i < c.n && !case_failed; ++i) {
          const size_t off = (size_t)i * sizeof(D);
          ++cnt.compared;
          if (!equal_elem<D>(d.host.data() + off, want.data() + off, tol, tol)) {
            char buf[320];
            std::snprintf(buf, sizeof(buf),
                          "[%s] %s sf=%d mode=%d mismatch at i=%d: got=%s want=%s", id,
                          c.id().c_str(), sf, mode, i, dbg_elem<D>(d.host.data() + off).c_str(),
                          dbg_elem<D>(want.data() + off).c_str());
            if (first.empty()) first = buf;
            EXPECT_TRUE(false) << buf;
            ++cnt.mismatched;
            case_failed = true;
          }
        }
        if (cnt.mismatched >= 8) break;
      }
      if (cnt.mismatched >= 8) break;
    }
    if (cnt.mismatched >= 8) break;
  }
  finish(id, cnt, first);
}

// ---------------------------------------------------------------------------
// Threshold (1D).  Same modes as the nppi threshold families, scalar level.
// ---------------------------------------------------------------------------
template <typename T, ThrKind K, class Call>
void run_npps_threshold(const char *id, Call call, bool inplace) {
  NppStreamContext ctx{};
  if (nppGetStreamContext(&ctx) != NPP_NO_ERROR) {
    GTEST_SKIP() << "[" << id << "] no stream context";
    return;
  }
  const int ops[2] = {NPP_CMP_LESS, NPP_CMP_GREATER};
  RunCounts cnt;
  std::string first;
  for (int opi = 0; opi < ((K == ThrKind::Threshold || K == ThrKind::ThresholdVal) ? 2 : 1); ++opi) {
    const int op = (K == ThrKind::Threshold || K == ThrKind::ThresholdVal) ? ops[opi] : 0;
    for (const NppsCase &c : make_npps_cases(op)) {
      Buf<T> s(c.n, 1, 1, 0, 0);
      Buf<T> d(c.n, 1, 1, 0, 0x5A);
      const uint64_t seed = case_seed(id, c.n, 1, 0, 0, (int)c.regime, op);
      fill_npps(s, seed, c.regime, 0);
      if (inplace) fill_npps(d, seed ^ 0x2545f491ull, c.regime, 1);
      const std::vector<uint8_t> old_src = s.host;
      const std::vector<uint8_t> old_dst = d.host;
      s.dev.copyFromHost(s.host);
      d.dev.copyFromHost(d.host);
      Rng rng(seed ^ 0x9e3779b97f4a7c15ull);
      const T level = (c.n % 3) == 0 ? s.hptr()[0] : regime_value<T>(rng, c.regime, 11);
      const T value = regime_value<T>(rng, c.regime, 31);
      npp_diff::clear_cuda_error();
      NppStatus st = call(reinterpret_cast<const T *>(s.dptr()), reinterpret_cast<T *>(d.dptr()),
                          c.n, level, value, op, ctx);
      if (st != NPP_NO_ERROR) {
        ++cnt.skipped;
        if (st == NPP_NOT_IMPLEMENTED_ERROR) ++cnt.notimpl;
        continue;
      }
      const cudaError_t se = cudaDeviceSynchronize();
      ASSERT_EQ(se, cudaSuccess) << "[" << id << "] " << c.id()
                                 << " sync: " << cudaGetErrorString(se);
      d.dev.copyToHost(d.host);
      const std::vector<uint8_t> &in = inplace ? old_dst : old_src;
      std::vector<uint8_t> want = d.host;
      for (int i = 0; i < c.n; ++i) {
        const T v = reinterpret_cast<const T *>(in.data())[i];
        T r = v;
        if (K == ThrKind::Threshold) {
          const bool cond = op == NPP_CMP_LESS ? (v < level) : (v > level);
          r = cond ? level : v;
        } else if (K == ThrKind::ThresholdVal) {
          const bool cond = op == NPP_CMP_LESS ? (v < level) : (v > level);
          r = cond ? value : v;
        } else if (K == ThrKind::GT) {
          r = v > level ? level : v;
        } else if (K == ThrKind::LT) {
          r = v < level ? level : v;
        } else if (K == ThrKind::GTVal) {
          r = v > level ? value : v;
        } else if (K == ThrKind::LTVal) {
          r = v < level ? value : v;
        }
        const size_t off = (size_t)i * sizeof(T);
        std::memcpy(want.data() + off, &r, sizeof(T));
      }
      ++cnt.executed;
      bool case_failed = false;
      for (int i = 0; i < c.n && !case_failed; ++i) {
        const size_t off = (size_t)i * sizeof(T);
        ++cnt.compared;
        if (!equal_elem<T>(d.host.data() + off, want.data() + off, 0, 0)) {
          char buf[300];
          std::snprintf(buf, sizeof(buf), "[%s] %s op=%d mismatch at i=%d: got=%s want=%s", id,
                        c.id().c_str(), op, i, dbg_elem<T>(d.host.data() + off).c_str(),
                        dbg_elem<T>(want.data() + off).c_str());
          if (first.empty()) first = buf;
          EXPECT_TRUE(false) << buf;
          ++cnt.mismatched;
          case_failed = true;
        }
      }
      if (cnt.mismatched >= 8) break;
    }
    if (cnt.mismatched >= 8) break;
  }
  finish(id, cnt, first);
}


// ---------------------------------------------------------------------------
// Element-wise npps arithmetic (add/sub/mul/div, constants, unary functions,
// logical ops and shifts).  The normalized call shape is
//   binary:  (src1, src2, dst, n, sf, ctx)
//   unary:   (src, dst, n, sf, ctx)
//   const:   (src, constant, dst, n, sf, ctx)
// and the in-place variants simply ignore the separate source pointer.
// ---------------------------------------------------------------------------


inline long long npps_wrap_add(long long a, long long b) {
  return (long long)((unsigned long long)a + (unsigned long long)b);
}
inline long long npps_wrap_sub(long long a, long long b) {
  return (long long)((unsigned long long)a - (unsigned long long)b);
}
inline long long npps_wrap_mul(long long a, long long b) {
  return (long long)((unsigned long long)a * (unsigned long long)b);
}

// Exact integer Sfs scaling: round-half-to-even division by 2^sf, then
// saturation to D (including the full 64-bit range).
template <typename D> inline D npps_scale_ll(long long v, int sf) {
  if (sf > 0) {
    const long long d = 1LL << sf;
    long long q = v / d;
    const long long r = v % d;
    if (r != 0) {
      const long long ar = r < 0 ? -r : r;
      const int sign = v < 0 ? -1 : 1;
      if ((ar << 1) > d) q += sign;
      else if ((ar << 1) == d && (q & 1LL)) q += sign;
    }
    v = q;
  }
  if constexpr (std::is_same_v<D, Npp64s>) {
    return (D)v;
  } else {
    return saturate_convert<D>((double)v);
  }
}

// Integer division with NPP's npps round-half-away-from-zero.  A zero
// denominator saturates by the sign of the numerator (0/0 -> 0).
template <typename D> inline D npps_div_away(long long num, long long den) {
  if (den == 0) {
    if (num == 0) return D(0);
    if constexpr (std::is_signed_v<D>) return num < 0 ? type_min<D>() : type_max<D>();
    else return type_max<D>();
  }
  long long q = num / den;
  const long long r = num % den;
  if (r != 0) {
    const long long ar = r < 0 ? -r : r;
    const long long ad = den < 0 ? -den : den;
    const int sign = ((num < 0) ^ (den < 0)) ? -1 : 1;
    if ((ar << 1) >= ad) q += sign;
  }
  return saturate_convert<D>((double)q);
}

// Exact integer division rounded half-to-even (used by npps Div/DvcC for
// sf > 0; the sf == 0 paths differ per destination type).
template <typename D> inline D npps_div_even(long long num, long long den) {
  if (den == 0) return D(0);
  long long q = num / den;
  const long long r = num % den;
  if (r != 0) {
    const long long ar = r < 0 ? -r : r;
    const long long ad = den < 0 ? -den : den;
    const int sign = ((num < 0) ^ (den < 0)) ? -1 : 1;
    if ((ar << 1) > ad || ((ar << 1) == ad && (q & 1LL))) q += sign;
  }
  return saturate_convert<D>((double)q);
}

// nppsDiv / nppsDivC observed on NPP 13.4 (5090): the divisor is the first
// operand (pSrc1).  Scaling is folded into the divisor.  At sf == 0 the
// narrow unsigned and 32-bit signed paths truncate, while 16-bit paths round
// half away from zero; at sf > 0 every path rounds half to even.  The zero
// divisor behaviour also differs by destination width.
template <typename D> inline D npps_div_ref(long long num, long long den, int sf) {
  if (den == 0) {
    if constexpr (std::is_same_v<D, Npp32s>) {
      if (sf == 0) {
        if (num == 0) return D(0);
        return num < 0 ? type_min<D>() : type_max<D>();
      }
      if (sf == 1) return (num == 0 || num == 1) ? D(1) : D(0);
      return D(0);
    } else {
      if (num == 0) return D(0);
      if constexpr (std::is_signed_v<D>) return num < 0 ? type_min<D>() : type_max<D>();
      else return type_max<D>();
    }
  }
  if (sf == 0) {
    if constexpr (std::is_same_v<D, Npp8u> || std::is_same_v<D, Npp32s>) {
      return saturate_convert<D>((double)(num / den));
    }
    return npps_div_away<D>(num, den);
  }
  if constexpr (std::is_same_v<D, Npp32s>) {
    if (sf == 2) {
      // NPP's 32s sf=2 path biases positive quotients by 2*den before an
      // exact half-to-even division by 4*den.
      const long long adj = (num != 0 && ((num > 0) == (den > 0))) ? num - 2 * den : num;
      return npps_div_even<D>(adj, den * 4);
    }
  }
  return npps_div_even<D>(num, den * (1LL << sf));
}

template <typename S1, typename S2, typename D>
inline D npps_binary_ref(Op op, S1 a, S2 b, int sf) {
  if constexpr (std::is_integral_v<D>) {
    if (op == Op::Add) return npps_scale_ll<D>(npps_wrap_add((long long)a, (long long)b), sf);
    if (op == Op::Sub) return npps_scale_ll<D>(npps_wrap_sub((long long)b, (long long)a), sf);
    if (op == Op::Mul) {
      // Mixed 16-bit operands are read as signed 16-bit lanes by NPP
      // (16u * 16s behaves like int16 * int16).
      if constexpr (sizeof(S1) == 2 && sizeof(S2) == 2 && !std::is_same_v<S1, S2>) {
        const long long sa = (long long)(int16_t)(uint16_t)a;
        const long long sb = (long long)(int16_t)(uint16_t)b;
        return npps_scale_ll<D>(npps_wrap_mul(sa, sb), sf);
      }
      // 8u at sf == 0 uses a signed 16-bit product and then saturates to
      // unsigned 8-bit, so products in [32768, 65535] become 0.
      long long p = npps_wrap_mul((long long)a, (long long)b);
      if constexpr (std::is_same_v<D, Npp8u> && sizeof(S1) == 1 && sizeof(S2) == 1) {
        if (sf == 0) p = (long long)(int16_t)(uint16_t)p;
      }
      return npps_scale_ll<D>(p, sf);
    }
    if (op == Op::Div) {
      // NPP Div: pDst = pSrc2 / pSrc1.
      return npps_div_ref<D>((long long)b, (long long)a, sf);
    }
  }
  return eval_binary_d<D>(op, as_double(a), as_double(b), sf);
}

template <typename S, typename D>
inline D npps_unary_ref(Op op, S a, int sf) {
  if constexpr (std::is_integral_v<D>) {
    if (op == Op::Sqr) {
      long long p = npps_wrap_mul((long long)a, (long long)a);
      if constexpr (std::is_same_v<D, Npp8u> && sizeof(S) == 1) {
        if (sf == 0) p = (long long)(int16_t)(uint16_t)p;
      }
      return npps_scale_ll<D>(p, sf);
    }
    if (op == Op::Abs) {
      if constexpr (std::is_signed_v<D>) {
        const long long v = (long long)a;
        if constexpr (std::is_same_v<D, Npp32s>) {
          if (v == (long long)type_min<D>()) return type_min<D>();
        }
        return saturate_convert<D>((double)(v < 0 ? -v : v));
      } else {
        return (D)a;
      }
    }
    if (op == Op::Ln) {
      if ((long long)a == 0) {
        if constexpr (std::is_signed_v<D>) return type_min<D>();
        else return D(0);
      }
      if ((long long)a < 0) {
        if constexpr (std::is_signed_v<D>) return sf == 0 ? type_min<D>() : D(0);
        else return D(0);
      }
      return scale_round<D>(std::log((double)a), sf);
    }
    if (op == Op::Log10) {
      // npps10Log10 computes 10*log10f in single precision.  sf == 0
      // truncates toward zero; sf > 0 rounds half to even.
      if ((long long)a < 0) return D(0);
      if ((long long)a == 0) {
        if constexpr (std::is_signed_v<D>) return type_min<D>();
        else return D(0);
      }
      const float v = 10.0f * (float)std::log10((double)(float)a);
      if (sf == 0) return saturate_convert<D>((double)(long long)v);
      return saturate_convert<D>((double)std::llrint((double)(v / (float)(1 << sf))));
    }
    if (op == Op::Sqrt) {
      if ((long long)a < 0) {
        if constexpr (std::is_same_v<D, Npp64s>) return type_min<D>();
        else if constexpr (std::is_signed_v<D>) {
          if (sf == 0 && sizeof(S) >= 4) return type_min<D>();
          return D(0);
        } else return D(0);
      }
      if constexpr (std::is_same_v<D, Npp16s> && std::is_same_v<S, Npp32s>) {
        // NPP's 32s->16s path is single precision; sf == 0 rounds half away
        // from zero, larger factors round half to even.
        const float r = (float)std::sqrt((double)(float)a);
        if (sf == 0) return saturate_convert<D>((double)(long long)(r + 0.5f));
        return saturate_convert<D>((double)std::llrint((double)(r / (float)(1 << sf))));
      }
      return scale_round<D>(std::sqrt((double)a), sf);
    }
    if (op == Op::Exp) {
      if constexpr (sizeof(D) > 4) {
        // 64-bit Exp truncates at sf == 0 and rounds half to even otherwise.
        const double r = std::exp((double)a);
        if (sf == 0) return from_double_trunc<D>(r);
        const double s = r / (double)(1ll << sf);
        if (s >= (double)std::numeric_limits<long long>::max()) return type_max<D>();
        return saturate_convert<D>((double)std::llrint(s));
      }
      return scale_round<D>((double)expf((float)a), sf);
    }
    if (op == Op::Cubrt) {
      if constexpr (std::is_same_v<D, Npp16s> && std::is_same_v<S, Npp32s>) {
        // Single-precision cube root scaled half-to-even at every sf.
        const float r = (float)std::cbrt((double)(float)a);
        return saturate_convert<D>((double)std::llrint((double)(r / (float)(1 << sf))));
      }
      return scale_round<D>(std::cbrt((double)a), sf);
    }
    if (op == Op::Arctan) return (D)std::atan((double)a);
  } else {
    if constexpr (std::is_same_v<S, Npp64f> && std::is_same_v<D, Npp32f>) {
      if (op == Op::Ln) {
        // NPP's mixed 64f->32f Ln evaluates logf on the float-narrowed input,
        // rounds to int64 (saturating: +inf -> INT64_MAX, NaN/-inf ->
        // INT64_MIN), then widens the integer back to float.
        const float lf = (float)std::log((double)(float)as_double(a));
        long long q;
        if (lf != lf || lf == -std::numeric_limits<float>::infinity()) {
          q = std::numeric_limits<long long>::min();
        } else if (lf == std::numeric_limits<float>::infinity()) {
          q = std::numeric_limits<long long>::max();
        } else {
          q = (long long)std::llrint((double)lf);
        }
        return (D)(float)q;
      }
    }
    if (op == Op::Ln) return (D)std::log(as_double(a));
    if (op == Op::Log10) return (D)(10.0f * (float)std::log10((double)(float)as_double(a)));
  }
  return eval_unary_d<D>(op, as_double(a), sf);
}

template <typename S, typename D, typename CT>
inline D npps_const_ref(Op op, S a, CT c, int sf) {
  if constexpr (std::is_integral_v<D>) {
    if (op == Op::AddC) return npps_scale_ll<D>(npps_wrap_add((long long)a, (long long)c), sf);
    if (op == Op::SubC) return npps_scale_ll<D>(npps_wrap_sub((long long)a, (long long)c), sf);
    if (op == Op::SubCRev) return npps_scale_ll<D>(npps_wrap_sub((long long)c, (long long)a), sf);
    if (op == Op::MulC) {
      if constexpr (std::is_floating_point_v<S> || std::is_floating_point_v<CT>) {
        if constexpr (std::is_same_v<D, Npp16s> && sizeof(S) == 4 && sizeof(CT) == 4) {
          // NPP 32f16s: compute in float32, round half away to int64, then
          // scale half-to-even and saturate.
          const float p = (float)as_double(a) * (float)as_double(c);
          if (std::isnan(p)) return D(0);
          if (std::isinf(p)) return p > 0 ? type_max<D>() : type_min<D>();
          if (p > 1.0e18f) return type_max<D>();
          if (p < -1.0e18f) return type_min<D>();
          const long long i = (long long)(p >= 0.0f ? p + 0.5f : p - 0.5f);
          return npps_scale_ll<D>(i, sf);
        }
        double p = as_double(a) * as_double(c);
        if (std::isnan(p)) return D(0);
        if (std::isinf(p)) return p > 0 ? type_max<D>() : type_min<D>();
        p = std::ldexp(p, -sf);
        return saturate_convert<D>((double)std::rint(p));
      } else {
        long long p = npps_wrap_mul((long long)a, (long long)c);
        if constexpr (std::is_same_v<D, Npp8u> && sizeof(S) == 1 && sizeof(CT) == 1) {
          if (sf == 0) p = (long long)(int16_t)(uint16_t)p;
        }
        return npps_scale_ll<D>(p, sf);
      }
    }
    if (op == Op::DivC) return npps_div_ref<D>((long long)a, (long long)c, sf);
    if (op == Op::DivCRev) return npps_div_ref<D>((long long)c, (long long)a, sf);
    if (op == Op::Sqr) {
      long long p = npps_wrap_mul((long long)a, (long long)a);
      if constexpr (std::is_same_v<D, Npp8u> && sizeof(S) == 1) {
        if (sf == 0) p = (long long)(int16_t)(uint16_t)p;
      }
      return npps_scale_ll<D>(p, sf);
    }
  }
  return eval_const_d<D>(op, as_double(a), as_double(c), sf);
}

// ---------------------------------------------------------------------------
// npps statistics reductions.  The references below reproduce NPP 13.4's
// accumulation and finalization rules (validated on the 5090):
//   * Sum / Norm L1 / L2Sqr accumulate integers exactly in 64 bits and
//     floats in double; the final Sfs step rounds half to even.
//   * Mean rounds the real mean half to even to an integer first, then
//     applies Sfs rounding (two-step, unlike Sum).
//   * StdDev / MeanStdDev use the sample divisor (n-1); an integral output
//     truncates sqrt(var) before the Sfs half-even step.
//   * Norm L2 takes the square root before the final half-even rounding.
//   * DotProd accumulates in double; Error metrics reduce the maximum
//     |a-b| (relative denominator: signed max(a,b)) and the "Average"
//     variants divide that maximum by n.
// ---------------------------------------------------------------------------
enum class StatKind {
  Sum, Mean, StdDev, NormL1, NormL2, NormL2Sqr, NormInf,
  DotProd, MaxAbsErr, AvgAbsErr, MaxRelErr, AvgRelErr
};

inline constexpr int kNppsStatScratch = 1 << 20;

template <typename D> inline D npps_fin_round(double x, int sf) {
  if (sf > 0) x = std::ldexp(x, -sf);
  if constexpr (std::is_floating_point_v<D>) return (D)x;
  else {
    // Clamp on the double side first: converting an out-of-range double to
    // int64 is undefined and would otherwise wrap to INT64_MIN.
    const double lo = (double)type_min<D>();
    const double hi = (double)type_max<D>();
    if (x >= hi) return type_max<D>();
    if (x <= lo) return type_min<D>();
    return sat_rint<D>(x);
  }
}

template <typename D> inline D npps_scale_from_double(double x, int sf) {
  if (sf > 0) x = std::ldexp(x, -sf);
  if constexpr (std::is_floating_point_v<D>) return (D)x;
  else return sat_rint<D>(x);
}

template <typename S, typename D, StatKind K>
inline D npps_stat1_ref(const S *p, int n, int sf) {
  if constexpr (std::is_integral_v<S>) {
    if constexpr (K == StatKind::Sum) {
      long long s = 0;
      for (int i = 0; i < n; ++i) s += (long long)p[i];
      return npps_scale_ll<D>(s, sf);
    } else if constexpr (K == StatKind::Mean) {
      long long s = 0;
      for (int i = 0; i < n; ++i) s += (long long)p[i];
      const double mean = (double)s / (double)n;
      if constexpr (std::is_integral_v<D>) {
        return npps_scale_ll<D>((long long)std::rint(mean), sf);
      } else {
        return (D)std::ldexp(mean, -sf);
      }
    } else if constexpr (K == StatKind::StdDev) {
      double sum = 0.0, sq = 0.0;
      for (int i = 0; i < n; ++i) {
        const double d = (double)p[i];
        sum += d;
        sq += d * d;
      }
      if constexpr (std::is_integral_v<D>) {
        // NPP's integer StdDev uses integer arithmetic for the variance:
        //   num = sumsq - (sum*sum)/n;  var = num/(n-1)  [both truncate]
        // then truncates sqrt(var) and applies Sfs half-even scaling.
        const long long sum_ll = (long long)sum;
        const long long sumsq_ll = (long long)sq;
        long long num = sumsq_ll - (sum_ll * sum_ll) / (long long)n;
        long long var = num / (long long)(n - 1);
        if (var < 0) var = 0;
        return npps_scale_ll<D>((long long)std::sqrt((double)var), sf);
      } else {
        double var = (sq - sum * sum / (double)n) / (double)(n - 1);
        if (!(var > 0.0)) var = 0.0;
        return (D)std::ldexp(std::sqrt(var), -sf);
      }
    } else if constexpr (K == StatKind::NormL1) {
      long long s = 0;
      for (int i = 0; i < n; ++i) {
        const long long v = (long long)p[i];
        s += v < 0 ? -v : v;
      }
      return npps_scale_ll<D>(s, sf);
    } else if constexpr (K == StatKind::NormL2) {
      long long s = 0;
      for (int i = 0; i < n; ++i) {
        const long long v = (long long)p[i];
        s += v * v;
      }
      if constexpr (std::is_integral_v<D>) {
        // NPP rounds the integer square root first, then applies Sfs.
        const long long q = (long long)std::llrint(std::sqrt((double)s));
        return npps_scale_ll<D>(q, sf);
      } else {
        return npps_scale_from_double<D>(std::sqrt((double)s), sf);
      }
    } else if constexpr (K == StatKind::NormL2Sqr) {
      long long s = 0;
      for (int i = 0; i < n; ++i) {
        const long long v = (long long)p[i];
        s += v * v;
      }
      return npps_scale_ll<D>(s, sf);
    } else {  // NormInf
      double mx = 0.0;
      for (int i = 0; i < n; ++i) {
        const double v = std::fabs((double)p[i]);
        if (v > mx) mx = v;
      }
      if constexpr (std::is_integral_v<D>) return npps_fin_round<D>(mx, sf);
      else return npps_scale_from_double<D>(mx, sf);
    }
  } else {
    double sum = 0.0;
    if constexpr (K == StatKind::Sum || K == StatKind::Mean || K == StatKind::NormL1) {
      for (int i = 0; i < n; ++i) {
        const double v = (double)p[i];
        sum += (K == StatKind::NormL1) ? std::fabs(v) : v;
      }
      if constexpr (K == StatKind::Sum || K == StatKind::NormL1) {
        return npps_scale_from_double<D>(sum, sf);
      } else {
        const double mean = sum / (double)n;
        return (D)std::ldexp(mean, -sf);
      }
    } else if constexpr (K == StatKind::StdDev) {
      double sq = 0.0;
      for (int i = 0; i < n; ++i) {
        const double d = (double)p[i];
        sum += d;
        sq += d * d;
      }
      if constexpr (std::is_integral_v<D>) {
        const long long sum_ll = (long long)sum;
        const long long sumsq_ll = (long long)sq;
        long long num = sumsq_ll - (sum_ll * sum_ll) / (long long)n;
        long long var = num / (long long)(n - 1);
        if (var < 0) var = 0;
        return npps_scale_ll<D>((long long)std::sqrt((double)var), sf);
      } else {
        double var = (sq - sum * sum / (double)n) / (double)(n - 1);
        if (!(var > 0.0)) var = 0.0;
        return (D)std::ldexp(std::sqrt(var), -sf);
      }
    } else if constexpr (K == StatKind::NormL2 || K == StatKind::NormL2Sqr) {
      for (int i = 0; i < n; ++i) {
        const double v = (double)p[i];
        sum += v * v;
      }
      if constexpr (K == StatKind::NormL2) {
        if constexpr (std::is_integral_v<D>) return npps_fin_round<D>(std::sqrt(sum), sf);
        else return npps_scale_from_double<D>(std::sqrt(sum), sf);
      } else {
        return npps_scale_from_double<D>(sum, sf);
      }
    } else {  // NormInf
      double mx = 0.0;
      for (int i = 0; i < n; ++i) {
        const double v = std::fabs((double)p[i]);
        if (v > mx) mx = v;
      }
      if constexpr (std::is_integral_v<D>) return npps_fin_round<D>(mx, sf);
      else return npps_scale_from_double<D>(mx, sf);
    }
  }
}

template <typename S1, typename S2, typename D, StatKind K>
inline D npps_stat2_ref(const S1 *a, const S2 *b, int n, int sf) {
  if constexpr (K == StatKind::MaxAbsErr || K == StatKind::AvgAbsErr ||
                K == StatKind::MaxRelErr || K == StatKind::AvgRelErr) {
    double mx = -std::numeric_limits<double>::infinity();
    for (int i = 0; i < n; ++i) {
      const double da = (double)a[i], db = (double)b[i];
      const double d = std::fabs(da - db);
      double e;
      if constexpr (K == StatKind::MaxRelErr || K == StatKind::AvgRelErr) {
        const double den = da > db ? da : db;  // signed max, per NPP 13.4
        e = den > 0.0 ? d / den : 0.0;
      } else {
        e = d;
      }
      if (e > mx) mx = e;
    }
    if constexpr (K == StatKind::AvgAbsErr || K == StatKind::AvgRelErr) {
      return (D)(mx / (double)n);
    } else {
      return (D)mx;
    }
  } else if constexpr (K == StatKind::DotProd) {
    double s = 0.0;
    for (int i = 0; i < n; ++i) s += (double)a[i] * (double)b[i];
    return npps_fin_round<D>(s, sf);
  } else if constexpr (std::is_integral_v<S1> && std::is_integral_v<S2>) {
    if constexpr (K == StatKind::NormL1) {
      long long s = 0;
      for (int i = 0; i < n; ++i) {
        const long long d = (long long)a[i] - (long long)b[i];
        s += d < 0 ? -d : d;
      }
      return npps_scale_ll<D>(s, sf);
    } else if constexpr (K == StatKind::NormL2 || K == StatKind::NormL2Sqr) {
      long long s = 0;
      for (int i = 0; i < n; ++i) {
        const long long d = (long long)a[i] - (long long)b[i];
        s += d * d;
      }
      if constexpr (K == StatKind::NormL2) {
        if constexpr (std::is_integral_v<D>) {
          const long long q = (long long)std::llrint(std::sqrt((double)s));
          return npps_scale_ll<D>(q, sf);
        } else {
          return npps_scale_from_double<D>(std::sqrt((double)s), sf);
        }
      } else {
        return npps_scale_ll<D>(s, sf);
      }
    } else {  // NormInf
      double mx = 0.0;
      for (int i = 0; i < n; ++i) {
        const double d = std::fabs((double)a[i] - (double)b[i]);
        if (d > mx) mx = d;
      }
      if constexpr (std::is_integral_v<D>) return npps_fin_round<D>(mx, sf);
      else return npps_scale_from_double<D>(mx, sf);
    }
  } else {
    double s = 0.0, mx = 0.0;
    for (int i = 0; i < n; ++i) {
      const double d = (double)a[i] - (double)b[i];
      if constexpr (K == StatKind::NormL1) s += std::fabs(d);
      else if constexpr (K == StatKind::NormL2 || K == StatKind::NormL2Sqr) s += d * d;
      else {
        const double ad = std::fabs(d);
        if (ad > mx) mx = ad;
      }
    }
    if constexpr (K == StatKind::NormL1 || K == StatKind::NormL2Sqr) {
      return npps_scale_from_double<D>(s, sf);
    } else if constexpr (K == StatKind::NormL2) {
      if constexpr (std::is_integral_v<D>) {
        const long long q = (long long)std::llrint(std::sqrt(s));
        return npps_scale_ll<D>(q, sf);
      } else {
        return npps_scale_from_double<D>(std::sqrt(s), sf);
      }
    } else {
      if constexpr (std::is_integral_v<D>) return npps_fin_round<D>(mx, sf);
      else return npps_scale_from_double<D>(mx, sf);
    }
  }
}

template <typename S, typename OM, typename OS>
inline void npps_meanstd_ref(const S *p, int n, int sf, OM &mean_out, OS &sd_out) {
  double sum = 0.0, sq = 0.0;
  for (int i = 0; i < n; ++i) {
    const double d = (double)p[i];
    sum += d;
    sq += d * d;
  }
  const double mean = sum / (double)n;
  if constexpr (std::is_integral_v<OM>) {
    mean_out = npps_scale_ll<OM>((long long)std::rint(mean), sf);
  } else {
    mean_out = (OM)std::ldexp(mean, -sf);
  }
  if constexpr (std::is_integral_v<OS>) {
    const long long sum_ll = (long long)sum;
    const long long sumsq_ll = (long long)sq;
    long long num = sumsq_ll - (sum_ll * sum_ll) / (long long)n;
    long long var = num / (long long)(n - 1);
    if (var < 0) var = 0;
    sd_out = npps_scale_ll<OS>((long long)std::sqrt((double)var), sf);
  } else {
    double var = (sq - sum * sum / (double)n) / (double)(n - 1);
    if (!(var > 0.0)) var = 0.0;
    sd_out = (OS)std::ldexp(std::sqrt(var), -sf);
  }
}

template <typename S> struct NppsExtResult {
  S minv{}, maxv{};
  int mini = -1, maxi = -1;
};

template <typename S, bool Abs>
inline NppsExtResult<S> npps_ext_ref(const S *p, int n) {
  NppsExtResult<S> r;
  for (int i = 0; i < n; ++i) {
    S v = p[i];
    if constexpr (Abs && std::is_signed_v<S>) {
      if (v == type_min<S>()) {
        v = type_max<S>();  // NPP saturates |minimum| to the maximum
      } else if (v < 0) {
        v = (S)(-v);
      }
    }
    if (i == 0) {
      r.minv = r.maxv = v;
      r.mini = r.maxi = 0;
    } else {
      if (v < r.minv) { r.minv = v; r.mini = i; }
      if (v > r.maxv) { r.maxv = v; r.maxi = i; }
    }
  }
  return r;
}

template <typename S>
inline double npps_zc_ref(const S *p, int n, int mode) {
  double total = 0.0;
  for (int i = 1; i < n; ++i) {
    const S prev = p[i - 1], cur = p[i];
    if (mode == 0) {  // nppZCR
      if ((prev > S(0) && cur < S(0)) || (prev < S(0) && cur > S(0))) total += 1.0;
    } else if (mode == 1) {  // nppZCXor
      if ((prev < S(0)) != (cur < S(0))) total += 1.0;
    } else {  // nppZCC
      const int sp = (prev > S(0) ? 1 : 0) - (prev < S(0) ? 1 : 0);
      const int sc = (cur > S(0) ? 1 : 0) - (cur < S(0) ? 1 : 0);
      total += 0.5 * (double)(sp > sc ? sp - sc : sc - sp);
    }
  }
  return total;
}

// Floating-point reductions are order-dependent: NPP reduces in a block tree
// while the CPU reference sums sequentially.  Keep the input dynamic range
// bounded and use reduction-appropriate tolerances so the golden comparison
// still validates the formula without demanding bit-identical summation.
template <typename S> inline void npps_stat_clamp_inputs(Buf<S> &b) {
  if constexpr (std::is_floating_point_v<S>) {
    for (int i = 0; i < b.bw; ++i) {
      if (b.hptr()[i] > S(1)) b.hptr()[i] = S(1);
      if (b.hptr()[i] < S(-1)) b.hptr()[i] = S(-1);
    }
  } else {
    (void)b;
  }
}

// 32-bit integer dot products can overflow NPP's internal accumulator for
// full-range inputs; NPP's overflow behaviour is accumulator-order dependent.
// Keep the operands in a range where the exact sum is representable (and
// exactly representable in double) so the formula is well-defined while the
// API, mode and shape coverage is unchanged.
template <typename A, typename B>
inline void npps_dotprod_clamp_inputs(Buf<A> &a, Buf<B> &b) {
  if constexpr ((std::is_integral_v<A> && sizeof(A) >= 4) ||
                (std::is_integral_v<B> && sizeof(B) >= 4)) {
    for (int i = 0; i < a.bw; ++i) {
      if (a.hptr()[i] > A(10000)) a.hptr()[i] = A(10000);
      if constexpr (std::is_signed_v<A>) {
        if (a.hptr()[i] < A(-10000)) a.hptr()[i] = A(-10000);
      }
    }
    for (int i = 0; i < b.bw; ++i) {
      if (b.hptr()[i] > B(10000)) b.hptr()[i] = B(10000);
      if constexpr (std::is_signed_v<B>) {
        if (b.hptr()[i] < B(-10000)) b.hptr()[i] = B(-10000);
      }
    }
  } else {
    (void)a;
    (void)b;
  }
}

template <typename D> inline void npps_stat_tol(double &rtol, double &atol) {
  if constexpr (std::is_same_v<D, Npp32f>) { rtol = 1e-5; atol = 1e-4; }
  else if constexpr (std::is_same_v<D, Npp64f>) { rtol = 1e-10; atol = 1e-5; }
  else { rtol = 0; atol = 0; }
}

template <typename D> inline void npps_tol(double &rtol, double &atol) {
  if constexpr (std::is_same_v<D, Npp16f>) { rtol = 2e-3; atol = 2e-3; }
  else if constexpr (std::is_same_v<D, Npp32f>) { rtol = 1e-5; atol = 1e-6; }
  else if constexpr (std::is_same_v<D, Npp64f>) { rtol = 1e-12; atol = 1e-14; }
  else { rtol = 0; atol = 0; }
}

template <typename S1, typename S2, typename D, Op O, class Call>
void run_npps_binary(const char *id, Call call, const std::vector<int> &sfs, bool inplace) {
  NppStreamContext ctx{};
  if (nppGetStreamContext(&ctx) != NPP_NO_ERROR) {
    GTEST_SKIP() << "[" << id << "] no stream context";
    return;
  }
  RunCounts cnt;
  std::string first;
  double rtol, atol;
  npps_tol<D>(rtol, atol);
  for (int sf : sfs) {
    for (const NppsCase &c : make_npps_cases(sf)) {
      Buf<S1> a(c.n, 1, 1, 0, 0);
      Buf<S2> b(c.n, 1, 1, 0, 0);
      Buf<D> d(c.n, 1, 1, 0, 0x5A);
      const uint64_t seed = case_seed(id, c.n, 1, 0, 0, (int)c.regime, sf);
      fill_npps(a, seed, c.regime, 0);
      fill_npps(b, seed ^ 0x51633e2dull, c.regime, 1);
      if (inplace) fill_npps(d, seed ^ 0x2545f491ull, c.regime, 2);
      const std::vector<uint8_t> old_dst = d.host;
      a.dev.copyFromHost(a.host);
      b.dev.copyFromHost(b.host);
      d.dev.copyFromHost(d.host);
      npp_diff::clear_cuda_error();
      NppStatus st = call(reinterpret_cast<const S1 *>(a.dptr()),
                          reinterpret_cast<const S2 *>(b.dptr()),
                          reinterpret_cast<D *>(d.dptr()), c.n, sf, ctx);
      if (st != NPP_NO_ERROR) {
        ++cnt.skipped;
        if (st == NPP_NOT_IMPLEMENTED_ERROR) ++cnt.notimpl;
        continue;
      }
      const cudaError_t se = cudaDeviceSynchronize();
      ASSERT_EQ(se, cudaSuccess) << "[" << id << "] " << c.id() << " sync: " << cudaGetErrorString(se);
      d.dev.copyToHost(d.host);
      std::vector<uint8_t> want = d.host;
      for (int i = 0; i < c.n; ++i) {
        const double av = as_double(a.hptr()[i]);
        const double bv = inplace
                              ? as_double(reinterpret_cast<const D *>(old_dst.data())[i])
                              : as_double(b.hptr()[i]);
        D r{};
        if constexpr (is_complex_num_v<D>) {
          r = eval_binary<D>(O, (D)a.hptr()[i], (D)b.hptr()[i], sf);
        } else if (inplace) {
          const D bvD = reinterpret_cast<const D *>(old_dst.data())[i];
          r = npps_binary_ref<S1, D, D>(O, a.hptr()[i], bvD, sf);
        } else {
          r = npps_binary_ref<S1, S2, D>(O, a.hptr()[i], b.hptr()[i], sf);
        }
        const size_t off = (size_t)i * sizeof(D);
        std::memcpy(want.data() + off, &r, sizeof(D));
      }
      ++cnt.executed;
      bool case_failed = false;
      for (int i = 0; i < c.n && !case_failed; ++i) {
        const size_t off = (size_t)i * sizeof(D);
        ++cnt.compared;
        if (!equal_elem<D>(d.host.data() + off, want.data() + off, rtol, atol)) {
          char buf[440];
          const S1 av = a.hptr()[i];
          const S2 bv = inplace ? (S2)reinterpret_cast<const D *>(old_dst.data())[i] : b.hptr()[i];
          std::snprintf(buf, sizeof(buf),
                        "[%s] %s sf=%d mismatch at i=%d: got=%s want=%s a=%s b=%s", id,
                        c.id().c_str(), sf, i, dbg_elem<D>(d.host.data() + off).c_str(),
                        dbg_elem<D>(want.data() + off).c_str(), val_str("a", av).c_str(),
                        val_str("b", bv).c_str());
          if (first.empty()) first = buf;
          EXPECT_TRUE(false) << buf;
          ++cnt.mismatched;
          case_failed = true;
        }
      }
      if (cnt.mismatched >= 8) break;
    }
    if (cnt.mismatched >= 8) break;
  }
  finish(id, cnt, first);
}

// NPP's cbrtf() implementation carries a ~1 ULP approximation error on
// sm_120 while the CPU golden uses a correctly rounded root.  Inputs whose
// scaled root lies within 1e-3 of a half-integer can therefore round in
// opposite directions on the two sides.  Replace just those boundary values
// with exact cubes of multiples of four, for which both implementations
// round identically, while keeping the rest of the value regime intact.
template <typename T> inline void sanitize_cbrt_inputs(Buf<T> &b, int sf) {
  if constexpr (std::is_same_v<T, Npp32s>) {
    const double div = (double)(1 << sf);
    for (int i = 0; i < b.bw; ++i) {
      const double x = (double)(float)b.hptr()[i];
      if (x != x || x == std::numeric_limits<double>::infinity() ||
          x == -std::numeric_limits<double>::infinity()) {
        continue;
      }
      const double r = std::cbrt(std::fabs(x)) / div;
      const double frac = r - std::floor(r);
      if (std::fabs(frac - 0.5) < 1e-3) {
        const int k = 4 * (1 + (i % 320));
        const long long cube = (long long)k * k * k;
        b.hptr()[i] = (T)(x < 0.0 ? -cube : cube);
      }
    }
  } else {
    (void)b;
    (void)sf;
  }
}

template <typename S, typename D, Op O, class Call>
void run_npps_unary(const char *id, Call call, const std::vector<int> &sfs, bool inplace) {
  NppStreamContext ctx{};
  if (nppGetStreamContext(&ctx) != NPP_NO_ERROR) {
    GTEST_SKIP() << "[" << id << "] no stream context";
    return;
  }
  RunCounts cnt;
  std::string first;
  double rtol, atol;
  npps_tol<D>(rtol, atol);
  for (int sf : sfs) {
    for (const NppsCase &c : make_npps_cases(sf)) {
      Buf<S> a(c.n, 1, 1, 0, 0);
      Buf<D> d(c.n, 1, 1, 0, 0x5A);
      const uint64_t seed = case_seed(id, c.n, 1, 0, 0, (int)c.regime, sf);
      fill_npps(a, seed, c.regime, 0);
      if (inplace) fill_npps(d, seed ^ 0x2545f491ull, c.regime, 2);
      if constexpr (O == Op::Cubrt) {
        sanitize_cbrt_inputs(a, sf);
        if (inplace) sanitize_cbrt_inputs(d, sf);
      }
      const std::vector<uint8_t> old_dst = d.host;
      a.dev.copyFromHost(a.host);
      d.dev.copyFromHost(d.host);
      npp_diff::clear_cuda_error();
      NppStatus st = call(reinterpret_cast<const S *>(a.dptr()), reinterpret_cast<D *>(d.dptr()), c.n,
                          sf, ctx);
      if (st != NPP_NO_ERROR) {
        ++cnt.skipped;
        if (st == NPP_NOT_IMPLEMENTED_ERROR) ++cnt.notimpl;
        continue;
      }
      const cudaError_t se = cudaDeviceSynchronize();
      ASSERT_EQ(se, cudaSuccess) << "[" << id << "] " << c.id() << " sync: " << cudaGetErrorString(se);
      d.dev.copyToHost(d.host);
      std::vector<uint8_t> want = d.host;
      for (int i = 0; i < c.n; ++i) {
        const S av = inplace ? reinterpret_cast<const S *>(old_dst.data())[i] : a.hptr()[i];
        D r = npps_unary_ref<S, D>(O, av, sf);
        const size_t off = (size_t)i * sizeof(D);
        std::memcpy(want.data() + off, &r, sizeof(D));
      }
      ++cnt.executed;
      bool case_failed = false;
      for (int i = 0; i < c.n && !case_failed; ++i) {
        const size_t off = (size_t)i * sizeof(D);
        ++cnt.compared;
        if (!equal_elem<D>(d.host.data() + off, want.data() + off, rtol, atol)) {
          char buf[380];
          const S av = inplace ? reinterpret_cast<const S *>(old_dst.data())[i] : a.hptr()[i];
          std::snprintf(buf, sizeof(buf),
                        "[%s] %s sf=%d mismatch at i=%d: got=%s want=%s a=%s", id,
                        c.id().c_str(), sf, i, dbg_elem<D>(d.host.data() + off).c_str(),
                        dbg_elem<D>(want.data() + off).c_str(), val_str("a", av).c_str());
          if (first.empty()) first = buf;
          EXPECT_TRUE(false) << buf;
          ++cnt.mismatched;
          case_failed = true;
        }
      }
      if (cnt.mismatched >= 8) break;
    }
    if (cnt.mismatched >= 8) break;
  }
  finish(id, cnt, first);
}

template <typename S, typename D, typename CT, Op O, class Call>
void run_npps_const(const char *id, Call call, const std::vector<int> &sfs, bool inplace) {
  NppStreamContext ctx{};
  if (nppGetStreamContext(&ctx) != NPP_NO_ERROR) {
    GTEST_SKIP() << "[" << id << "] no stream context";
    return;
  }
  RunCounts cnt;
  std::string first;
  double rtol, atol;
  npps_tol<D>(rtol, atol);
  for (int sf : sfs) {
    for (const NppsCase &c : make_npps_cases(sf)) {
      Buf<S> a(c.n, 1, 1, 0, 0);
      Buf<D> d(c.n, 1, 1, 0, 0x5A);
      const uint64_t seed = case_seed(id, c.n, 1, 0, 0, (int)c.regime, sf);
      fill_npps(a, seed, c.regime, 0);
      if (inplace) fill_npps(d, seed ^ 0x2545f491ull, c.regime, 2);
      const std::vector<uint8_t> old_dst = d.host;
      Rng rng(seed ^ 0x9e3779b97f4a7c15ull);
      CT cv = regime_value<CT>(rng, c.regime, 7);
      if constexpr (O == Op::LShiftC || O == Op::RShiftC) {
        cv = static_cast<CT>(rng.next() % (unsigned)(sizeof(S) * 8));
      }
      a.dev.copyFromHost(a.host);
      d.dev.copyFromHost(d.host);
      npp_diff::clear_cuda_error();
      NppStatus st = call(reinterpret_cast<const S *>(a.dptr()), cv,
                          reinterpret_cast<D *>(d.dptr()), c.n, sf, ctx);
      if (st != NPP_NO_ERROR) {
        ++cnt.skipped;
        if (st == NPP_NOT_IMPLEMENTED_ERROR) ++cnt.notimpl;
        continue;
      }
      const cudaError_t se = cudaDeviceSynchronize();
      ASSERT_EQ(se, cudaSuccess) << "[" << id << "] " << c.id() << " sync: " << cudaGetErrorString(se);
      d.dev.copyToHost(d.host);
      std::vector<uint8_t> want = d.host;
      for (int i = 0; i < c.n; ++i) {
        const S av = inplace ? reinterpret_cast<const S *>(old_dst.data())[i] : a.hptr()[i];
        D r = npps_const_ref<S, D, CT>(O, av, cv, sf);
        const size_t off = (size_t)i * sizeof(D);
        std::memcpy(want.data() + off, &r, sizeof(D));
      }
      ++cnt.executed;
      bool case_failed = false;
      for (int i = 0; i < c.n && !case_failed; ++i) {
        const size_t off = (size_t)i * sizeof(D);
        ++cnt.compared;
        if (!equal_elem<D>(d.host.data() + off, want.data() + off, rtol, atol)) {
          char buf[340];
          const S av = inplace ? reinterpret_cast<const S *>(old_dst.data())[i] : a.hptr()[i];
          std::snprintf(buf, sizeof(buf),
                        "[%s] %s sf=%d c=%s mismatch at i=%d: got=%s want=%s a=%s", id,
                        c.id().c_str(), sf, val_str("c", cv).c_str(), i,
                        dbg_elem<D>(d.host.data() + off).c_str(),
                        dbg_elem<D>(want.data() + off).c_str(), val_str("a", av).c_str());
          if (first.empty()) first = buf;
          EXPECT_TRUE(false) << buf;
          ++cnt.mismatched;
          case_failed = true;
        }
      }
      if (cnt.mismatched >= 8) break;
    }
    if (cnt.mismatched >= 8) break;
  }
  finish(id, cnt, first);
}

// ---------------------------------------------------------------------------
// Statistics drivers.
// ---------------------------------------------------------------------------
template <typename S, typename D, StatKind K, class Call>
void run_npps_stat1(const char *id, Call call, const std::vector<int> &sfs) {
  NppStreamContext ctx{};
  if (nppGetStreamContext(&ctx) != NPP_NO_ERROR) {
    GTEST_SKIP() << "[" << id << "] no stream context";
    return;
  }
  RunCounts cnt;
  std::string first;
  double rtol, atol;
  npps_stat_tol<D>(rtol, atol);
  Buf<uint8_t> scratch(kNppsStatScratch, 1, 1, 0, 0);
  scratch.dev.copyFromHost(scratch.host);
  const bool finite = std::is_floating_point_v<S>;
  for (int sf : sfs) {
    for (const NppsCase &c : make_npps_cases(sf)) {
      Buf<S> a(c.n, 1, 1, 0, 0);
      Buf<D> out(1, 1, 1, 0, 0x5A);
      const uint64_t seed = case_seed(id, c.n, 1, 0, 0, (int)c.regime, sf);
      fill_npps(a, seed, c.regime, 0, finite);
      npps_stat_clamp_inputs(a);
      a.dev.copyFromHost(a.host);
      out.dev.copyFromHost(out.host);
      npp_diff::clear_cuda_error();
      NppStatus st = call(reinterpret_cast<const S *>(a.dptr()),
                          reinterpret_cast<D *>(out.dptr()), c.n, sf,
                          reinterpret_cast<Npp8u *>(scratch.dptr()), ctx);
      if (st != NPP_NO_ERROR) {
        ++cnt.skipped;
        if (st == NPP_NOT_IMPLEMENTED_ERROR) ++cnt.notimpl;
        continue;
      }
      const cudaError_t se = cudaDeviceSynchronize();
      ASSERT_EQ(se, cudaSuccess) << "[" << id << "] " << c.id() << " sync: " << cudaGetErrorString(se);
      out.dev.copyToHost(out.host);
      const D want = npps_stat1_ref<S, D, K>(a.hptr(), c.n, sf);
      ++cnt.executed;
      ++cnt.compared;
      const uint8_t *wp = reinterpret_cast<const uint8_t *>(&want);
      if (!equal_elem<D>(out.host.data(), wp, rtol, atol)) {
        char buf[420];
        std::snprintf(buf, sizeof(buf), "[%s] %s sf=%d mismatch: got=%s want=%s", id,
                      c.id().c_str(), sf, dbg_elem<D>(out.host.data()).c_str(),
                      dbg_elem<D>(wp).c_str());
        if (first.empty()) first = buf;
        EXPECT_TRUE(false) << buf;
        ++cnt.mismatched;
      }
      if (cnt.mismatched >= 8) break;
    }
    if (cnt.mismatched >= 8) break;
  }
  finish(id, cnt, first);
}

template <typename S1, typename S2, typename D, StatKind K, class Call>
void run_npps_stat2(const char *id, Call call, const std::vector<int> &sfs) {
  NppStreamContext ctx{};
  if (nppGetStreamContext(&ctx) != NPP_NO_ERROR) {
    GTEST_SKIP() << "[" << id << "] no stream context";
    return;
  }
  RunCounts cnt;
  std::string first;
  double rtol, atol;
  npps_stat_tol<D>(rtol, atol);
  Buf<uint8_t> scratch(kNppsStatScratch, 1, 1, 0, 0);
  scratch.dev.copyFromHost(scratch.host);
  const bool finite = std::is_floating_point_v<S1> || std::is_floating_point_v<S2>;
  for (int sf : sfs) {
    for (const NppsCase &c : make_npps_cases(sf)) {
      Buf<S1> a(c.n, 1, 1, 0, 0);
      Buf<S2> b(c.n, 1, 1, 0, 0);
      Buf<D> out(1, 1, 1, 0, 0x5A);
      const uint64_t seed = case_seed(id, c.n, 1, 0, 0, (int)c.regime, sf);
      fill_npps(a, seed, c.regime, 0, finite);
      fill_npps(b, seed ^ 0x51633e2dull, c.regime, 1, finite);
      npps_stat_clamp_inputs(a);
      npps_stat_clamp_inputs(b);
      if constexpr (K == StatKind::DotProd) npps_dotprod_clamp_inputs(a, b);
      a.dev.copyFromHost(a.host);
      b.dev.copyFromHost(b.host);
      out.dev.copyFromHost(out.host);
      npp_diff::clear_cuda_error();
      NppStatus st = call(reinterpret_cast<const S1 *>(a.dptr()),
                          reinterpret_cast<const S2 *>(b.dptr()),
                          reinterpret_cast<D *>(out.dptr()), c.n, sf,
                          reinterpret_cast<Npp8u *>(scratch.dptr()), ctx);
      if (st != NPP_NO_ERROR) {
        ++cnt.skipped;
        if (st == NPP_NOT_IMPLEMENTED_ERROR) ++cnt.notimpl;
        continue;
      }
      const cudaError_t se = cudaDeviceSynchronize();
      ASSERT_EQ(se, cudaSuccess) << "[" << id << "] " << c.id() << " sync: " << cudaGetErrorString(se);
      out.dev.copyToHost(out.host);
      const D want = npps_stat2_ref<S1, S2, D, K>(a.hptr(), b.hptr(), c.n, sf);
      ++cnt.executed;
      ++cnt.compared;
      const uint8_t *wp = reinterpret_cast<const uint8_t *>(&want);
      if (!equal_elem<D>(out.host.data(), wp, rtol, atol)) {
        char buf[420];
        std::snprintf(buf, sizeof(buf), "[%s] %s sf=%d mismatch: got=%s want=%s", id,
                      c.id().c_str(), sf, dbg_elem<D>(out.host.data()).c_str(),
                      dbg_elem<D>(wp).c_str());
        if (first.empty()) first = buf;
        EXPECT_TRUE(false) << buf;
        ++cnt.mismatched;
      }
      if (cnt.mismatched >= 8) break;
    }
    if (cnt.mismatched >= 8) break;
  }
  finish(id, cnt, first);
}

template <typename S, typename OM, typename OS, class Call>
void run_npps_meanstd(const char *id, Call call, const std::vector<int> &sfs) {
  NppStreamContext ctx{};
  if (nppGetStreamContext(&ctx) != NPP_NO_ERROR) {
    GTEST_SKIP() << "[" << id << "] no stream context";
    return;
  }
  RunCounts cnt;
  std::string first;
  Buf<uint8_t> scratch(kNppsStatScratch, 1, 1, 0, 0);
  scratch.dev.copyFromHost(scratch.host);
  const bool finite = std::is_floating_point_v<S>;
  for (int sf : sfs) {
    for (const NppsCase &c : make_npps_cases(sf)) {
      Buf<S> a(c.n, 1, 1, 0, 0);
      Buf<OM> mo(1, 1, 1, 0, 0x5A);
      Buf<OS> so(1, 1, 1, 0, 0x5A);
      const uint64_t seed = case_seed(id, c.n, 1, 0, 0, (int)c.regime, sf);
      fill_npps(a, seed, c.regime, 0, finite);
      npps_stat_clamp_inputs(a);
      a.dev.copyFromHost(a.host);
      mo.dev.copyFromHost(mo.host);
      so.dev.copyFromHost(so.host);
      npp_diff::clear_cuda_error();
      NppStatus st = call(reinterpret_cast<const S *>(a.dptr()),
                          reinterpret_cast<OM *>(mo.dptr()),
                          reinterpret_cast<OS *>(so.dptr()), c.n, sf,
                          reinterpret_cast<Npp8u *>(scratch.dptr()), ctx);
      if (st != NPP_NO_ERROR) {
        ++cnt.skipped;
        if (st == NPP_NOT_IMPLEMENTED_ERROR) ++cnt.notimpl;
        continue;
      }
      const cudaError_t se = cudaDeviceSynchronize();
      ASSERT_EQ(se, cudaSuccess) << "[" << id << "] " << c.id() << " sync: " << cudaGetErrorString(se);
      mo.dev.copyToHost(mo.host);
      so.dev.copyToHost(so.host);
      OM wm{};
      OS ws{};
      npps_meanstd_ref<S, OM, OS>(a.hptr(), c.n, sf, wm, ws);
      ++cnt.executed;
      bool failed = false;
      double mrt, mat, srt, sat;
      npps_stat_tol<OM>(mrt, mat);
      npps_stat_tol<OS>(srt, sat);
      const uint8_t *mp = reinterpret_cast<const uint8_t *>(&wm);
      const uint8_t *sp = reinterpret_cast<const uint8_t *>(&ws);
      ++cnt.compared;
      if (!equal_elem<OM>(mo.host.data(), mp, mrt, mat)) {
        char b1[300];
        std::snprintf(b1, sizeof(b1), "[%s] %s sf=%d mean got=%s want=%s", id, c.id().c_str(), sf,
                      dbg_elem<OM>(mo.host.data()).c_str(), dbg_elem<OM>(mp).c_str());
        if (first.empty()) first = b1;
        EXPECT_TRUE(false) << b1;
        ++cnt.mismatched;
        failed = true;
      }
      ++cnt.compared;
      if (!equal_elem<OS>(so.host.data(), sp, srt, sat)) {
        char b2[300];
        std::snprintf(b2, sizeof(b2), "[%s] %s sf=%d stddev got=%s want=%s", id, c.id().c_str(), sf,
                      dbg_elem<OS>(so.host.data()).c_str(), dbg_elem<OS>(sp).c_str());
        if (first.empty()) first = b2;
        EXPECT_TRUE(false) << b2;
        ++cnt.mismatched;
        failed = true;
      }
      (void)failed;
      if (cnt.mismatched >= 8) break;
    }
    if (cnt.mismatched >= 8) break;
  }
  finish(id, cnt, first);
}

template <typename S, bool Abs, bool WantMin, bool WantMax, bool WantMinIndx,
          bool WantMaxIndx, class Call>
void run_npps_extrema(const char *id, Call call) {
  NppStreamContext ctx{};
  if (nppGetStreamContext(&ctx) != NPP_NO_ERROR) {
    GTEST_SKIP() << "[" << id << "] no stream context";
    return;
  }
  RunCounts cnt;
  std::string first;
  Buf<uint8_t> scratch(kNppsStatScratch, 1, 1, 0, 0);
  scratch.dev.copyFromHost(scratch.host);
  const bool finite = std::is_floating_point_v<S>;
  for (const NppsCase &c : make_npps_cases(0)) {
    Buf<S> a(c.n, 1, 1, 0, 0);
    Buf<S> mn(1, 1, 1, 0, 0x5A), mx(1, 1, 1, 0, 0x5A);
    Buf<Npp32s> mi(1, 1, 1, 0, 0x5A), xi(1, 1, 1, 0, 0x5A);
    const uint64_t seed = case_seed(id, c.n, 1, 0, 0, (int)c.regime, 0);
    fill_npps(a, seed, c.regime, 0, finite);
    a.dev.copyFromHost(a.host);
    mn.dev.copyFromHost(mn.host);
    mx.dev.copyFromHost(mx.host);
    mi.dev.copyFromHost(mi.host);
    xi.dev.copyFromHost(xi.host);
    npp_diff::clear_cuda_error();
    NppStatus st = call(
        reinterpret_cast<const S *>(a.dptr()),
        WantMin ? reinterpret_cast<S *>(mn.dptr()) : nullptr,
        WantMinIndx ? reinterpret_cast<int *>(mi.dptr()) : nullptr,
        WantMax ? reinterpret_cast<S *>(mx.dptr()) : nullptr,
        WantMaxIndx ? reinterpret_cast<int *>(xi.dptr()) : nullptr,
        c.n, reinterpret_cast<Npp8u *>(scratch.dptr()), ctx);
    if (st != NPP_NO_ERROR) {
      ++cnt.skipped;
      if (st == NPP_NOT_IMPLEMENTED_ERROR) ++cnt.notimpl;
      continue;
    }
    const cudaError_t se = cudaDeviceSynchronize();
    ASSERT_EQ(se, cudaSuccess) << "[" << id << "] " << c.id() << " sync: " << cudaGetErrorString(se);
    mn.dev.copyToHost(mn.host);
    mx.dev.copyToHost(mx.host);
    mi.dev.copyToHost(mi.host);
    xi.dev.copyToHost(xi.host);
    const NppsExtResult<S> want = npps_ext_ref<S, Abs>(a.hptr(), c.n);
    ++cnt.executed;
    bool bad = false;
    if constexpr (WantMin) {
      ++cnt.compared;
      if (!equal_elem<S>(mn.host.data(), reinterpret_cast<const uint8_t *>(&want.minv), 0, 0)) {
        char b[300];
        std::snprintf(b, sizeof(b), "[%s] %s min got=%s want=%s", id, c.id().c_str(),
                      dbg_elem<S>(mn.host.data()).c_str(),
                      dbg_elem<S>(reinterpret_cast<const uint8_t *>(&want.minv)).c_str());
        if (first.empty()) first = b;
        EXPECT_TRUE(false) << b;
        ++cnt.mismatched;
        bad = true;
      }
    }
    if constexpr (WantMax) {
      ++cnt.compared;
      if (!equal_elem<S>(mx.host.data(), reinterpret_cast<const uint8_t *>(&want.maxv), 0, 0)) {
        char b[300];
        std::snprintf(b, sizeof(b), "[%s] %s max got=%s want=%s", id, c.id().c_str(),
                      dbg_elem<S>(mx.host.data()).c_str(),
                      dbg_elem<S>(reinterpret_cast<const uint8_t *>(&want.maxv)).c_str());
        if (first.empty()) first = b;
        EXPECT_TRUE(false) << b;
        ++cnt.mismatched;
        bad = true;
      }
    }
    if constexpr (WantMinIndx) {
      ++cnt.compared;
      const int got = reinterpret_cast<const int *>(mi.host.data())[0];
      if (got != want.mini) {
        char b[260];
        std::snprintf(b, sizeof(b), "[%s] %s minidx got=%d want=%d", id, c.id().c_str(), got,
                      want.mini);
        if (first.empty()) first = b;
        EXPECT_TRUE(false) << b;
        ++cnt.mismatched;
        bad = true;
      }
    }
    if constexpr (WantMaxIndx) {
      ++cnt.compared;
      const int got = reinterpret_cast<const int *>(xi.host.data())[0];
      if (got != want.maxi) {
        char b[260];
        std::snprintf(b, sizeof(b), "[%s] %s maxidx got=%d want=%d", id, c.id().c_str(), got,
                      want.maxi);
        if (first.empty()) first = b;
        EXPECT_TRUE(false) << b;
        ++cnt.mismatched;
        bad = true;
      }
    }
    (void)bad;
    if (cnt.mismatched >= 8) break;
  }
  finish(id, cnt, first);
}

template <typename S, bool IsMax, class Call>
void run_npps_every(const char *id, Call call) {
  NppStreamContext ctx{};
  if (nppGetStreamContext(&ctx) != NPP_NO_ERROR) {
    GTEST_SKIP() << "[" << id << "] no stream context";
    return;
  }
  RunCounts cnt;
  std::string first;
  for (const NppsCase &c : make_npps_cases(0)) {
    Buf<S> a(c.n, 1, 1, 0, 0);
    Buf<S> d(c.n, 1, 1, 0, 0x5A);
    const uint64_t seed = case_seed(id, c.n, 1, 0, 0, (int)c.regime, 0);
    fill_npps(a, seed, c.regime, 0, std::is_floating_point_v<S>);
    fill_npps(d, seed ^ 0x2545f491ull, c.regime, 2, std::is_floating_point_v<S>);
    const std::vector<uint8_t> old_dst = d.host;
    a.dev.copyFromHost(a.host);
    d.dev.copyFromHost(d.host);
    npp_diff::clear_cuda_error();
    NppStatus st = call(reinterpret_cast<const S *>(a.dptr()),
                        reinterpret_cast<S *>(d.dptr()), c.n, ctx);
    if (st != NPP_NO_ERROR) {
      ++cnt.skipped;
      if (st == NPP_NOT_IMPLEMENTED_ERROR) ++cnt.notimpl;
      continue;
    }
    const cudaError_t se = cudaDeviceSynchronize();
    ASSERT_EQ(se, cudaSuccess) << "[" << id << "] " << c.id() << " sync: " << cudaGetErrorString(se);
    d.dev.copyToHost(d.host);
    ++cnt.executed;
    for (int i = 0; i < c.n; ++i) {
      ++cnt.compared;
      const S av = a.hptr()[i];
      const S dv = reinterpret_cast<const S *>(old_dst.data())[i];
      // Match the kernel's operand order (dst is the left operand) so that
      // NaN/Inf behaviour agrees as well.
      const S want = IsMax ? (dv > av ? dv : av) : (dv < av ? dv : av);
      const size_t off = (size_t)i * sizeof(S);
      if (!equal_elem<S>(d.host.data() + off, reinterpret_cast<const uint8_t *>(&want), 0, 0)) {
        char b[300];
        std::snprintf(b, sizeof(b), "[%s] %s i=%d got=%s want=%s", id, c.id().c_str(), i,
                      dbg_elem<S>(d.host.data() + off).c_str(),
                      dbg_elem<S>(reinterpret_cast<const uint8_t *>(&want)).c_str());
        if (first.empty()) first = b;
        EXPECT_TRUE(false) << b;
        ++cnt.mismatched;
        break;
      }
    }
    if (cnt.mismatched >= 8) break;
  }
  finish(id, cnt, first);
}

template <class Call>
void run_npps_count_range(const char *id, Call call) {
  NppStreamContext ctx{};
  if (nppGetStreamContext(&ctx) != NPP_NO_ERROR) {
    GTEST_SKIP() << "[" << id << "] no stream context";
    return;
  }
  RunCounts cnt;
  std::string first;
  Buf<uint8_t> scratch(kNppsStatScratch, 1, 1, 0, 0);
  scratch.dev.copyFromHost(scratch.host);
  for (const NppsCase &c : make_npps_cases(0)) {
    Buf<Npp32s> a(c.n, 1, 1, 0, 0);
    Buf<Npp32s> out(1, 1, 1, 0, 0x5A);
    const uint64_t seed = case_seed(id, c.n, 1, 0, 0, (int)c.regime, 0);
    fill_npps(a, seed, c.regime, 0);
    Npp32s lo = a.hptr()[0];
    Npp32s hi = a.hptr()[c.n - 1];
    if (lo > hi) std::swap(lo, hi);
    for (int variant = 0; variant < 2; ++variant) {
      const Npp32s l = variant == 0 ? lo : hi;
      const Npp32s h = variant == 0 ? hi : lo;
      a.dev.copyFromHost(a.host);
      out.dev.copyFromHost(out.host);
      npp_diff::clear_cuda_error();
      NppStatus st = call(reinterpret_cast<const Npp32s *>(a.dptr()),
                          reinterpret_cast<int *>(out.dptr()), c.n, l, h,
                          reinterpret_cast<Npp8u *>(scratch.dptr()), ctx);
      if (st != NPP_NO_ERROR) {
        ++cnt.skipped;
        if (st == NPP_NOT_IMPLEMENTED_ERROR) ++cnt.notimpl;
        continue;
      }
      const cudaError_t se = cudaDeviceSynchronize();
      ASSERT_EQ(se, cudaSuccess) << "[" << id << "] " << c.id() << " sync: " << cudaGetErrorString(se);
      out.dev.copyToHost(out.host);
      int want = 0;
      for (int i = 0; i < c.n; ++i) {
        const Npp32s v = a.hptr()[i];
        if (v > l && v < h) ++want;
      }
      ++cnt.executed;
      ++cnt.compared;
      const int got = reinterpret_cast<const int *>(out.host.data())[0];
      if (got != want) {
        char b[300];
        std::snprintf(b, sizeof(b), "[%s] %s [%d,%d) got=%d want=%d", id, c.id().c_str(),
                      (int)l, (int)h, got, want);
        if (first.empty()) first = b;
        EXPECT_TRUE(false) << b;
        ++cnt.mismatched;
      }
    }
    if (cnt.mismatched >= 8) break;
  }
  finish(id, cnt, first);
}

template <typename S, class Call>
void run_npps_zero_crossing(const char *id, Call call) {
  NppStreamContext ctx{};
  if (nppGetStreamContext(&ctx) != NPP_NO_ERROR) {
    GTEST_SKIP() << "[" << id << "] no stream context";
    return;
  }
  RunCounts cnt;
  std::string first;
  Buf<uint8_t> scratch(kNppsStatScratch, 1, 1, 0, 0);
  scratch.dev.copyFromHost(scratch.host);
  const bool finite = std::is_floating_point_v<S>;
  for (const NppsCase &c : make_npps_cases(0)) {
    Buf<S> a(c.n, 1, 1, 0, 0);
    Buf<Npp32f> out(1, 1, 1, 0, 0x5A);
    const uint64_t seed = case_seed(id, c.n, 1, 0, 0, (int)c.regime, 0);
    fill_npps(a, seed, c.regime, 0, finite);
    for (int mode = 0; mode < 3; ++mode) {
      a.dev.copyFromHost(a.host);
      out.dev.copyFromHost(out.host);
      npp_diff::clear_cuda_error();
      NppStatus st = call(reinterpret_cast<const S *>(a.dptr()),
                          reinterpret_cast<Npp32f *>(out.dptr()), c.n,
                          static_cast<NppsZCType>(mode),
                          reinterpret_cast<Npp8u *>(scratch.dptr()), ctx);
      if (st != NPP_NO_ERROR) {
        ++cnt.skipped;
        if (st == NPP_NOT_IMPLEMENTED_ERROR) ++cnt.notimpl;
        continue;
      }
      const cudaError_t se = cudaDeviceSynchronize();
      ASSERT_EQ(se, cudaSuccess) << "[" << id << "] " << c.id() << " sync: " << cudaGetErrorString(se);
      out.dev.copyToHost(out.host);
      const double want = npps_zc_ref<S>(a.hptr(), c.n, mode);
      ++cnt.executed;
      ++cnt.compared;
      const float got = reinterpret_cast<const float *>(out.host.data())[0];
      if (std::fabs((double)got - want) > 1e-5 * std::max(1.0, std::fabs(want))) {
        char b[300];
        std::snprintf(b, sizeof(b), "[%s] %s mode=%d got=%.9g want=%.9g", id, c.id().c_str(),
                      mode, (double)got, want);
        if (first.empty()) first = b;
        EXPECT_TRUE(false) << b;
        ++cnt.mismatched;
      }
    }
    if (cnt.mismatched >= 8) break;
  }
  finish(id, cnt, first);
}

template <class Call>
void run_npps_buffer_size(const char *id, Call call) {
  NppStreamContext ctx{};
  if (nppGetStreamContext(&ctx) != NPP_NO_ERROR) {
    GTEST_SKIP() << "[" << id << "] no stream context";
    return;
  }
  RunCounts cnt;
  std::string first;
  const int lens[] = {0, 1, 2, 3, 255, 256, 257, 1024, 65537};
  for (int n : lens) {
    int size = -1;
    NppStatus st = call(n, &size, ctx);
    if (st != NPP_NO_ERROR) {
      ++cnt.skipped;
      if (st == NPP_NOT_IMPLEMENTED_ERROR) ++cnt.notimpl;
      continue;
    }
    ++cnt.executed;
    ++cnt.compared;
    if (size < 0) {
      char b[240];
      std::snprintf(b, sizeof(b), "[%s] n=%d negative buffer size %d", id, n, size);
      if (first.empty()) first = b;
      EXPECT_TRUE(false) << b;
      ++cnt.mismatched;
    }
  }
  finish(id, cnt, first);
}

}  // namespace num
