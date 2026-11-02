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

}  // namespace num
