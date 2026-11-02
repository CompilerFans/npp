#pragma once

// Full-matrix CPU-golden conformance suite.
//
// Unlike test_num_arithmetic.cpp (one hand-picked 5090-calibrated case list),
// this header drives every API variant through a small/medium/large geometry
// matrix, with padded steps, ROI offsets into a larger allocation and both
// out-of-place and in-place forms.  Expected values are still computed on the
// CPU in process; there are no golden recordings and no status assertions.
//
// The reference formulas live in eval_binary/eval_unary/eval_const below.
// They are calibrated against the NVIDIA NPP 13.4 gate on the RTX 5090: every
// generated test must pass there before it is accepted as a reference, and
// the in-tree MPP implementation must then match the same formulas on CUDA and
// MACA.

#include "num_test_utils.h"

#include <cuda_fp16.h>

namespace num {

// ---------------------------------------------------------------------------
// Reference operations.
// ---------------------------------------------------------------------------
enum class Op {
  Add,
  Sub,
  Mul,
  Div,
  AddC,
  SubC,
  MulC,
  DivC,
  And,
  Or,
  Xor,
  AndC,
  OrC,
  XorC,
  LShiftC,
  RShiftC,
  Abs,
  AbsDiff,
  AbsDiffC,
  Sqr,
  Sqrt,
  Ln,
  Exp,
  Not,
  Inv,  // logical NOT, 0/1 result
  Min,
  Max,
  MinEvery,
  MaxEvery,
  DivRound,
  MulScale,
  MulCScale,
  AlphaPremul,
  AlphaPremulC,
  AlphaComp,
  AlphaCompC,
  SubCRev,
  DivCRev,
  Log10,
  Cubrt,
  Arctan,
};

template <typename T> struct is_complex_num : std::false_type {};
template <> struct is_complex_num<Npp8uc> : std::true_type {};
template <> struct is_complex_num<Npp16uc> : std::true_type {};
template <> struct is_complex_num<Npp16sc> : std::true_type {};
template <> struct is_complex_num<Npp32uc> : std::true_type {};
template <> struct is_complex_num<Npp32sc> : std::true_type {};
template <> struct is_complex_num<Npp32fc> : std::true_type {};
template <> struct is_complex_num<Npp64sc> : std::true_type {};
template <> struct is_complex_num<Npp64fc> : std::true_type {};
template <typename T> inline constexpr bool is_complex_num_v = is_complex_num<T>::value;

template <typename T> struct complex_part { using type = T; };
template <> struct complex_part<Npp8uc> { using type = Npp8u; };
template <> struct complex_part<Npp16uc> { using type = Npp16u; };
template <> struct complex_part<Npp16sc> { using type = Npp16s; };
template <> struct complex_part<Npp32uc> { using type = Npp32u; };
template <> struct complex_part<Npp32sc> { using type = Npp32s; };
template <> struct complex_part<Npp32fc> { using type = Npp32f; };
template <> struct complex_part<Npp64sc> { using type = Npp64s; };
template <> struct complex_part<Npp64fc> { using type = Npp64f; };
template <typename T> using complex_part_t = typename complex_part<T>::type;

inline float half_to_float_host(Npp16f v) {
  __half_raw raw;
  raw.x = static_cast<unsigned short>(v.fp16);
  const __half h = raw;
  return __half2float(h);
}

template <typename T> inline double as_double(T v) {
  if constexpr (std::is_same_v<T, Npp16f>) return half_to_float_host(v);
  else return static_cast<double>(v);
}

inline double as_double(Npp16f v) { return half_to_float_host(v); }

// Round half-to-even then saturate, the NPP integer Sfs rule.
template <typename T> inline T sat_rint(double v) {
  if constexpr (std::is_same_v<T, Npp8u>) {
    const double r = std::rint(v);
    return (T)(r < 0.0 ? 0.0 : (r > 255.0 ? 255.0 : r));
  } else if constexpr (std::is_same_v<T, Npp8s>) {
    const double r = std::rint(v);
    return (T)(r < -128.0 ? -128.0 : (r > 127.0 ? 127.0 : r));
  } else if constexpr (std::is_same_v<T, Npp16u>) {
    const double r = std::rint(v);
    return (T)(r < 0.0 ? 0.0 : (r > 65535.0 ? 65535.0 : r));
  } else if constexpr (std::is_same_v<T, Npp16s>) {
    const double r = std::rint(v);
    return (T)(r < -32768.0 ? -32768.0 : (r > 32767.0 ? 32767.0 : r));
  } else if constexpr (std::is_same_v<T, Npp32u>) {
    const double r = std::rint(v);
    return (T)(r < 0.0 ? 0.0 : (r > 4294967295.0 ? 4294967295.0 : r));
  } else if constexpr (std::is_same_v<T, Npp32s>) {
    const double r = std::rint(v);
    return (T)(r < -2147483648.0 ? -2147483648.0 : (r > 2147483647.0 ? 2147483647.0 : r));
  } else if constexpr (std::is_same_v<T, Npp64u>) {
    const double r = std::rint(v);
    const double hi = 18446744073709551616.0;  // 2^64
    return (T)(r < 0.0 ? 0.0 : (r >= hi ? hi - 2048.0 : r));
  } else if constexpr (std::is_same_v<T, Npp64s>) {
    const double r = std::rint(v);
    if (std::isnan(r)) return (T)(-9223372036854775807LL - 1);
    const double lo = -9223372036854775808.0;
    const double hi = 9223372036854775808.0;
    if (r <= lo) return (T)(-9223372036854775807LL - 1);
    if (r >= hi) return (T)9223372036854775807LL;
    return (T)r;
  } else {
    return (T)v;
  }
}

template <typename T> inline T sat_trunc(double v) {
  if constexpr (std::is_same_v<T, Npp8u>)
    return (T)(v < 0.0 ? 0.0 : (v > 255.0 ? 255.0 : v));
  else if constexpr (std::is_same_v<T, Npp8s>)
    return (T)(v < -128.0 ? -128.0 : (v > 127.0 ? 127.0 : v));
  else if constexpr (std::is_same_v<T, Npp16u>)
    return (T)(v < 0.0 ? 0.0 : (v > 65535.0 ? 65535.0 : v));
  else if constexpr (std::is_same_v<T, Npp16s>)
    return (T)(v < -32768.0 ? -32768.0 : (v > 32767.0 ? 32767.0 : v));
  else if constexpr (std::is_same_v<T, Npp32u>)
    return (T)(v < 0.0 ? 0.0 : (v > 4294967295.0 ? 4294967295.0 : v));
  else if constexpr (std::is_same_v<T, Npp32s>)
    return (T)(v < -2147483648.0 ? -2147483648.0 : (v > 2147483647.0 ? 2147483647.0 : v));
  else if constexpr (std::is_same_v<T, Npp64u>)
    return (T)(std::isnan(v) ? 0.0 : (v <= 0.0 ? 0.0 : (v >= 18446744073709551616.0 ? 18446744073709549568.0 : v)));
  else if constexpr (std::is_same_v<T, Npp64s>) {
    if (std::isnan(v)) return (T)(-9223372036854775807LL - 1);
    if (v <= -9223372036854775808.0) return (T)(-9223372036854775807LL - 1);
    if (v >= 9223372036854775808.0) return (T)9223372036854775807LL;
    return (T)v;
  }
  else return (T)v;
}

inline Npp16f half_round(float v) { return npp_diff::to_npp16f(v); }

template <typename T> inline T from_double_rint(double v) {
  if constexpr (std::is_same_v<T, Npp16f>) return half_round((float)v);
  else return sat_rint<T>(v);
}

template <typename T> inline T from_double_trunc(double v) {
  if constexpr (std::is_same_v<T, Npp16f>) return half_round((float)v);
  else return sat_trunc<T>(v);
}

template <typename T> inline T zero_like() {
  if constexpr (is_complex_num_v<T>) return T{};
  else return T(0);
}

template <typename T> constexpr T type_min() {
  if constexpr (std::is_same_v<T, Npp8u>) return (T)0;
  else if constexpr (std::is_same_v<T, Npp8s>) return (T)-128;
  else if constexpr (std::is_same_v<T, Npp16u>) return (T)0;
  else if constexpr (std::is_same_v<T, Npp16s>) return (T)-32768;
  else if constexpr (std::is_same_v<T, Npp32u>) return (T)0;
  else if constexpr (std::is_same_v<T, Npp32s>) return (T)(-2147483647 - 1);
  else if constexpr (std::is_same_v<T, Npp64u>) return (T)0;
  else if constexpr (std::is_same_v<T, Npp64s>)
    return (T)(-9223372036854775807LL - 1);
  else return (T)0;
}

template <typename T> constexpr T type_max() {
  if constexpr (std::is_same_v<T, Npp8u>) return (T)255;
  else if constexpr (std::is_same_v<T, Npp8s>) return (T)127;
  else if constexpr (std::is_same_v<T, Npp16u>) return (T)65535;
  else if constexpr (std::is_same_v<T, Npp16s>) return (T)32767;
  else if constexpr (std::is_same_v<T, Npp32u>) return (T)0xFFFFFFFFu;
  else if constexpr (std::is_same_v<T, Npp32s>) return (T)2147483647;
  else if constexpr (std::is_same_v<T, Npp64u>) return (T)0xFFFFFFFFFFFFFFFFull;
  else if constexpr (std::is_same_v<T, Npp64s>) return (T)9223372036854775807LL;
  else return (T)0;
}

// Scale a (possibly fractional) value by 2^-sf and round.  NPP's integer Sfs
// rule is exact quotient followed by round-half-even.
template <typename T> inline T scale_round(double v, int sf) {
  if (sf > 0) v /= (double)(1ll << sf);
  return from_double_rint<T>(v);
}

// Truncating Sfs step.  NPP's complex 32sc division kernels convert the
// quotient toward zero rather than rounding (calibrated on the 5090).
template <typename T> inline T scale_trunc(double v, int sf) {
  if (sf > 0) v /= (double)(1ll << sf);
  return from_double_trunc<T>(v);
}

// NPP's 32-bit signed Sfs division uses a different rounding path from the
// narrower types (calibrated on the 5090): sf=0 truncates toward zero, sf=1
// rounds half-to-even, sf=2 uses floor((num-1)/den) for positive numerators
// and half-to-even otherwise, and sf>=3 truncates.
inline Npp32s div_s32(long long num, long long den, int sf) {
  long long q;
  if (sf == 0) {
    q = num / den;
  } else if (sf == 1) {
    q = (long long)std::rint((double)num / (double)(den * 2));
  } else if (sf == 2) {
    const long long d = den * 4;
    q = num > 0 ? (num - 1) / d : (long long)std::rint((double)num / (double)d);
  } else {
    q = num / (den << sf);
  }
  if (q < (long long)type_min<Npp32s>()) return type_min<Npp32s>();
  if (q > (long long)type_max<Npp32s>()) return type_max<Npp32s>();
  return (Npp32s)q;
}

// ---------------------------------------------------------------------------
// Scalar element semantics.  Complex types are handled by the two overloads
// below; everything else funnels through these primitives.
// ---------------------------------------------------------------------------
template <typename T> inline T bin_scalar(Op op, T a, T b, int p) {
  if constexpr (std::is_same_v<T, Npp16f>) {
    const float fa = (float)as_double(a), fb = (float)as_double(b);
    switch (op) {
      case Op::Add: return half_round(fa + fb);
      case Op::Sub: return half_round(fb - fa);
      case Op::Mul: return half_round(fa * fb);
      case Op::Div: return half_round(fb / fa);
      case Op::AbsDiff: return half_round(std::fabs(fa - fb));
      case Op::Min: return fb < fa ? b : a;
      case Op::Max: return fb > fa ? b : a;
      case Op::MinEvery: return fb < fa ? b : a;
      case Op::MaxEvery: return fb > fa ? b : a;
      default: return a;
    }
  } else if constexpr (std::is_floating_point_v<T>) {
    switch (op) {
      case Op::Add: return (T)((double)a + (double)b);
      case Op::Sub: return (T)((double)b - (double)a);
      case Op::Mul: return (T)((double)a * (double)b);
      case Op::Div: return (T)((double)b / (double)a);
      case Op::AbsDiff: return (T)std::fabs((double)a - (double)b);
      case Op::Min: return b < a ? b : a;
      case Op::Max: return b > a ? b : a;
      case Op::MinEvery: return b < a ? b : a;
      case Op::MaxEvery: return b > a ? b : a;
      default: return a;
    }
  } else {
    const long long ia = (long long)a, ib = (long long)b;
    switch (op) {
      case Op::Add: return scale_round<T>((double)ia + (double)ib, p);
      case Op::Sub: return scale_round<T>((double)ib - (double)ia, p);
      case Op::Mul: return scale_round<T>((double)ia * (double)ib, p);
      case Op::Div: {
        if (ia == 0) {
          if constexpr (std::is_same_v<T, Npp8u>) {
            // 8u division by zero: 0/0 saturates to 254, otherwise to 255.
            return ib == 0 ? T(254) : T(255);
          } else {
            if (ib == 0) return T(0);
            if constexpr (std::is_signed_v<T>) return ib < 0 ? type_min<T>() : type_max<T>();
            else return type_max<T>();
          }
        }
        if constexpr (std::is_same_v<T, Npp32s>) return div_s32(ib, ia, p);
        return scale_round<T>((double)ib / (double)ia, p);
      }
      case Op::And: return (T)(a & b);
      case Op::Or: return (T)(a | b);
      case Op::Xor: return (T)(a ^ b);
      case Op::AbsDiff: return from_double_rint<T>(std::fabs((double)ia - (double)ib));
      case Op::Min: return b < a ? b : a;
      case Op::Max: return b > a ? b : a;
      case Op::MinEvery: return b < a ? b : a;
      case Op::MaxEvery: return b > a ? b : a;
      default: return a;
    }
  }
}

template <typename T> inline T un_scalar(Op op, T a, int p) {
  if constexpr (std::is_same_v<T, Npp16f>) {
    const float fa = (float)as_double(a);
    switch (op) {
      case Op::Sqr: return half_round(fa * fa);
      case Op::Sqrt: return half_round(std::sqrt(fa));
      case Op::Ln: return half_round(std::log(fa));
      case Op::Exp: return half_round(std::exp(fa));
      case Op::Abs: return half_round(std::fabs(fa));
      default: return a;
    }
  } else if constexpr (std::is_floating_point_v<T>) {
    switch (op) {
      case Op::Sqr: return (T)((double)a * (double)a);
      case Op::Sqrt: return (T)std::sqrt((double)a);
      case Op::Ln: return (T)std::log((double)a);
      case Op::Exp: {
        // Both the 32f and 64f npps/nppi kernels clamp to the normal range
        // (NaN maps to the low end).
        const double r = std::exp((double)a);
        const double lo = (double)std::numeric_limits<T>::min();
        const double hi = (double)std::numeric_limits<T>::max();
        if (std::isnan(r)) return (T)lo;
        return (T)(r < lo ? lo : (r > hi ? hi : r));
      }
      case Op::Abs: return (T)std::fabs((double)a);
      case Op::Log10: return (T)std::log10((double)a);
      case Op::Cubrt: return (T)std::cbrt((double)a);
      case Op::Arctan: return (T)std::atan((double)a);
      default: return a;
    }
  } else {
    switch (op) {
      case Op::Sqr:
        if constexpr (std::is_same_v<T, Npp8u>) {
          // NPP's 8u Sqr uses a signed 16-bit product for sf == 0, so inputs
          // >= 182 wrap negative and clamp to 0; sf > 0 uses a wide product.
          if (p == 0) {
            const int16_t sp = (int16_t)(uint16_t)((unsigned)a * (unsigned)a);
            return sp < 0 ? T(0) : (T)(sp > 255 ? 255 : sp);
          }
          int prod = (int)a * (int)a;
          prod >>= p;
          return (T)(prod > 255 ? 255 : prod);
        }
        return scale_round<T>((double)a * (double)a, p);
      case Op::Sqrt:
        if (a < T(0)) return T(0);
        return scale_round<T>(std::sqrt((double)a), p);
      case Op::Ln:
        // NPP: log(0) saturates to the minimum for signed types regardless of
        // the scale factor; negative inputs do so only at sf == 0, otherwise
        // they map to 0.  Unsigned zero maps to 0.
        if (a == T(0)) {
          if constexpr (std::is_signed_v<T>) return type_min<T>();
          else return T(0);
        }
        if (a < T(0)) {
          if constexpr (std::is_signed_v<T>) return p == 0 ? type_min<T>() : T(0);
          else return T(0);
        }
        return scale_round<T>(std::log((double)a), p);
      case Op::Exp: {
        const double r = std::exp((double)a);
        return scale_round<T>(r, p);
      }
      case Op::Abs:
        if constexpr (std::is_signed_v<T>) {
          if (a == type_min<T>()) return type_max<T>();
          return (T)(a < T(0) ? -a : a);
        } else {
          return a;
        }
      case Op::Log10:
        if (a <= T(0)) {
          if constexpr (std::is_signed_v<T>) return p == 0 ? type_min<T>() : T(0);
          else return T(0);
        }
        return scale_round<T>(std::log10((double)a), p);
      case Op::Cubrt:
        return scale_round<T>(std::cbrt((double)a), p);
      case Op::Arctan:
        return scale_round<T>(std::atan((double)a), p);
      case Op::Not: return (T)(~a);
      case Op::Inv: return (T)(a == T(0) ? 1 : 0);
      default: return a;
    }
  }
}

template <typename T> inline T const_scalar(Op op, T a, T c, int p) {
  if constexpr (std::is_same_v<T, Npp16f>) {
    const float fa = (float)as_double(a), fc = (float)as_double(c);
    switch (op) {
      case Op::AddC: return half_round(fa + fc);
      case Op::SubC: return half_round(fa - fc);
      case Op::MulC: return half_round(fa * fc);
      case Op::DivC: return half_round(fa / fc);
      case Op::AbsDiffC: return half_round(std::fabs(fa - fc));
      default: return a;
    }
  } else if constexpr (std::is_floating_point_v<T>) {
    switch (op) {
      case Op::AddC: return (T)((double)a + (double)c);
      case Op::SubC: return (T)((double)a - (double)c);
      case Op::MulC: return (T)((double)a * (double)c);
      case Op::DivC: return (T)((double)a / (double)c);
      case Op::SubCRev: return (T)((double)c - (double)a);
      case Op::DivCRev: return (T)((double)c / (double)a);
      case Op::AbsDiffC: return (T)std::fabs((double)a - (double)c);
      default: return a;
    }
  } else {
    const long long ia = (long long)a, ic = (long long)c;
    switch (op) {
      case Op::AddC: return scale_round<T>((double)ia + (double)ic, p);
      case Op::SubC: return scale_round<T>((double)ia - (double)ic, p);
      case Op::MulC: return scale_round<T>((double)ia * (double)ic, p);
      case Op::DivC:
        if (ic == 0) {
          if constexpr (std::is_same_v<T, Npp8u>) {
            return ia == 0 ? T(254) : T(255);
          } else {
            if (ia == 0) return T(0);
            if constexpr (std::is_signed_v<T>) return ia < 0 ? type_min<T>() : type_max<T>();
            else return type_max<T>();
          }
        }
        if constexpr (std::is_same_v<T, Npp32s>) return div_s32(ia, ic, p);
        return scale_round<T>((double)ia / (double)ic, p);
      case Op::AbsDiffC: return from_double_rint<T>(std::fabs((double)ia - (double)ic));
      case Op::SubCRev: return scale_round<T>((double)ic - (double)ia, p);
      case Op::DivCRev:
        if (ia == 0) {
          if (ic == 0) return T(0);
          if constexpr (std::is_signed_v<T>) return ic < 0 ? type_min<T>() : type_max<T>();
          else return type_max<T>();
        }
        return scale_round<T>((double)ic / (double)ia, p);
      case Op::AndC: return (T)(a & c);
      case Op::OrC: return (T)(a | c);
      case Op::XorC: return (T)(a ^ c);
      case Op::LShiftC: return (T)(a << ic);
      case Op::RShiftC: return (T)(a >> ic);
      default: return a;
    }
  }
}

template <typename T> inline T eval_binary(Op op, T a, T b, int p) {
  if constexpr (is_complex_num_v<T>) {
    using R = complex_part_t<T>;
    T out{};
    switch (op) {
      case Op::Add:
        out.re = bin_scalar<R>(Op::Add, a.re, b.re, p);
        out.im = bin_scalar<R>(Op::Add, a.im, b.im, p);
        return out;
      case Op::Sub:
        out.re = bin_scalar<R>(Op::Sub, a.re, b.re, p);
        out.im = bin_scalar<R>(Op::Sub, a.im, b.im, p);
        return out;
      case Op::Mul: {
        // (a.re + a.im i) * (b.re + b.im i)
        const double re = (double)a.re * (double)b.re - (double)a.im * (double)b.im;
        const double im = (double)a.re * (double)b.im + (double)a.im * (double)b.re;
        if constexpr (std::is_integral_v<R>) {
          out.re = scale_round<R>(re, p);
          out.im = scale_round<R>(im, p);
        } else {
          out.re = (R)re;
          out.im = (R)im;
        }
        return out;
      }
      case Op::Div: {
        const double den = (double)a.re * (double)a.re + (double)a.im * (double)a.im;
        // A zero divisor follows IEEE (0/0 -> NaN in both components); NPP's
        // complex kernels do not clamp it.
        const double re = ((double)b.re * (double)a.re + (double)b.im * (double)a.im) / den;
        const double im = ((double)b.im * (double)a.re - (double)b.re * (double)a.im) / den;
        if constexpr (std::is_integral_v<R>) {
          // NPP complex division: 32sc truncates; 16sc truncates at sf == 0
          // and rounds half-to-even above (verified on the 5090).
          if (den == 0.0) return T{};
          if constexpr (std::is_same_v<R, Npp32s>) {
            out.re = scale_trunc<R>(re, p);
            out.im = scale_trunc<R>(im, p);
          } else if (p == 0) {
            out.re = scale_trunc<R>(re, p);
            out.im = scale_trunc<R>(im, p);
          } else {
            out.re = scale_round<R>(re, p);
            out.im = scale_round<R>(im, p);
          }
        } else {
          out.re = (R)re;
          out.im = (R)im;
        }
        return out;
      }
      default:
        return T{};
    }
  } else {
    return bin_scalar<T>(op, a, b, p);
  }
}

template <typename T> inline T eval_unary(Op op, T a, int p) {
  if constexpr (is_complex_num_v<T>) {
    using R = complex_part_t<T>;
    T out{};
    if (op == Op::Abs) {
      const double m = std::sqrt((double)a.re * (double)a.re + (double)a.im * (double)a.im);
      if constexpr (std::is_integral_v<R>) out.re = from_double_rint<R>(m);
      else out.re = (R)m;
      return out;
    }
    return a;
  } else {
    return un_scalar<T>(op, a, p);
  }
}

template <typename T> inline T eval_const(Op op, T a, T c, int p) {
  if constexpr (is_complex_num_v<T>) {
    using R = complex_part_t<T>;
    T out{};
    if (op == Op::MulC) {
      // Full complex multiply: a * c.
      const double re = (double)a.re * (double)c.re - (double)a.im * (double)c.im;
      const double im = (double)a.re * (double)c.im + (double)a.im * (double)c.re;
      if constexpr (std::is_integral_v<R>) {
        out.re = scale_round<R>(re, p);
        out.im = scale_round<R>(im, p);
      } else {
        out.re = (R)re;
        out.im = (R)im;
      }
      return out;
    }
    if (op == Op::DivC) {
      // Full complex divide: a / c.  A zero divisor follows IEEE (NaN
      // components for 0/0), matching the 5090 complex kernels.
      const double den = (double)c.re * (double)c.re + (double)c.im * (double)c.im;
      const double re = ((double)a.re * (double)c.re + (double)a.im * (double)c.im) / den;
      const double im = ((double)a.im * (double)c.re - (double)a.re * (double)c.im) / den;
      if constexpr (std::is_integral_v<R>) {
        if (den == 0.0) return T{};
        if constexpr (std::is_same_v<R, Npp32s>) {
          out.re = scale_trunc<R>(re, p);
          out.im = scale_trunc<R>(im, p);
        } else if (p == 0) {
          out.re = scale_trunc<R>(re, p);
          out.im = scale_trunc<R>(im, p);
        } else {
          out.re = scale_round<R>(re, p);
          out.im = scale_round<R>(im, p);
        }
      } else {
        out.re = (R)re;
        out.im = (R)im;
      }
      return out;
    }
    out.re = const_scalar<R>(op, a.re, c.re, p);
    out.im = const_scalar<R>(op, a.im, c.im, p);
    return out;
  } else {
    return const_scalar<T>(op, a, c, p);
  }
}

// Double-based entry points used by the drivers.  They keep the arithmetic in
// the destination type's domain while allowing the source/constant types to
// differ (e.g. Npp32f constants for the 16f constant APIs).
inline Npp16f half_binary(Op op, float a, float b) {
  switch (op) {
    case Op::Add: return half_round(a + b);
    case Op::Sub: return half_round(b - a);
    case Op::Mul: return half_round(a * b);
    case Op::Div: return half_round(b / a);
    case Op::AbsDiff: return half_round(std::fabs(a - b));
    case Op::Min: return b < a ? half_round(b) : half_round(a);
    case Op::Max: return b > a ? half_round(b) : half_round(a);
    case Op::MinEvery: return b < a ? half_round(b) : half_round(a);
    case Op::MaxEvery: return b > a ? half_round(b) : half_round(a);
    default: return half_round(a);
  }
}

inline Npp16f half_unary(Op op, float a) {
  switch (op) {
    case Op::Sqr: return half_round(a * a);
    case Op::Sqrt: return half_round(std::sqrt(a));
    case Op::Ln: return half_round(std::log(a));
    case Op::Exp: return half_round(std::exp(a));
    case Op::Abs: return half_round(std::fabs(a));
    default: return half_round(a);
  }
}

inline Npp16f half_const(Op op, float a, float c) {
  switch (op) {
    case Op::AddC: return half_round(a + c);
    case Op::SubC: return half_round(a - c);
    case Op::MulC: return half_round(a * c);
    case Op::DivC: return half_round(a / c);
    case Op::AbsDiffC: return half_round(std::fabs(a - c));
    default: return half_round(a);
  }
}

template <typename D> inline D eval_binary_d(Op op, double a, double b, int p) {
  if constexpr (std::is_same_v<D, Npp16f>) {
    return half_binary(op, (float)a, (float)b);
  } else if constexpr (std::is_floating_point_v<D>) {
    return bin_scalar<D>(op, (D)a, (D)b, p);
  } else {
    return bin_scalar<D>(op, (D)(long long)a, (D)(long long)b, p);
  }
}

template <typename D> inline D eval_unary_d(Op op, double a, int p) {
  if constexpr (std::is_same_v<D, Npp16f>) {
    return half_unary(op, (float)a);
  } else if constexpr (std::is_floating_point_v<D>) {
    return un_scalar<D>(op, (D)a, p);
  } else {
    return un_scalar<D>(op, (D)(long long)a, p);
  }
}

template <typename D> inline D eval_const_d(Op op, double a, double c, int p) {
  if constexpr (std::is_same_v<D, Npp16f>) {
    return half_const(op, (float)a, (float)c);
  } else if constexpr (std::is_floating_point_v<D>) {
    return const_scalar<D>(op, (D)a, (D)c, p);
  } else {
    return const_scalar<D>(op, (D)(long long)a, (D)(long long)c, p);
  }
}

// ---------------------------------------------------------------------------
// Geometry matrix: small, medium and large ROIs, padded steps and ROI offsets
// into a larger allocation.  Every shape is exercised under all three value
// regimes; stride/offset variants are added on representative shapes only so
// the matrix stays bounded.
// ---------------------------------------------------------------------------
struct GenCase {
  int w, h;
  int pad;
  int ox, oy;
  Regime regime;
  int param;
  std::string id() const {
    char b[128];
    std::snprintf(b, sizeof(b), "%dx%d/sp%d/off%d,%d/r%d/p%d", w, h, pad, ox, oy, (int)regime,
                  param);
    return b;
  }
};

inline std::vector<GenCase> make_gen_cases(int param) {
  static const int sizes[][2] = {{1, 1}, {2, 3}, {7, 5}, {1, 9}, {9, 1},
                                 {33, 17}, {64, 48}, {257, 129}, {512, 257}};
  static const Regime regimes[] = {kSmall, kWide, kExtremes};
  std::vector<GenCase> v;
  for (auto &s : sizes)
    for (Regime r : regimes) v.push_back({s[0], s[1], 0, 0, 0, r, param});
  static const int off_sizes[][2] = {{2, 3}, {33, 17}, {257, 129}};
  for (auto &s : off_sizes)
    for (Regime r : regimes) v.push_back({s[0], s[1], 16, 2, 3, r, param});
  static const int bigpad_sizes[][2] = {{1, 1}, {64, 48}};
  for (auto &s : bigpad_sizes)
    for (Regime r : regimes) v.push_back({s[0], s[1], 32, 1, 1, r, param});
  return v;
}

inline int align_step(size_t bytes, int pad) { return (int)align16(bytes) + pad; }

// NPP 13.4's 16s stores write -1 into the high half of each packed 32-bit
// value when the row pitch is a multiple of 64 bytes (measured: pitch, not
// width, is the trigger).  For 16sc C1 the high half is the imaginary lane,
// so x+0yi is stored as x-1i.  A padded pitch selects NPP's scalar path and
// is correct; the affected families (Set/Copy/Threshold/Compare) use this
// helper.
template <typename T> inline int safe_pad(int pad) {
  if constexpr (std::is_same_v<T, Npp16s> || std::is_same_v<T, Npp8s> ||
                std::is_same_v<T, Npp16sc>)
    return pad == 0 ? 16 : pad;
  return pad;
}

// ---------------------------------------------------------------------------
// Buffers.  step is the row pitch in bytes of the full allocation; the ROI may
// start at (ox, oy) inside it.
// ---------------------------------------------------------------------------
template <typename T> struct Buf {
  int bw = 0, bh = 0, step = 0;
  std::vector<uint8_t> host;
  npp_functional_test::DeviceMemory<uint8_t> dev;

  Buf() = default;
  Buf(int w, int h, int channels, int pad, uint8_t fill)
      : bw(w), bh(h), step(align_step((size_t)w * channels * sizeof(T), pad)),
        host((size_t)step * h, fill), dev((size_t)step * h) {}

  T *hptr() { return reinterpret_cast<T *>(host.data()); }
  const T *hptr() const { return reinterpret_cast<const T *>(host.data()); }
  uint8_t *dptr() { return dev.get(); }
};

template <typename T>
inline size_t off_bytes(int oy, int ox, int step, int channels) {
  return (size_t)oy * step + (size_t)ox * channels * sizeof(T);
}

// ---------------------------------------------------------------------------
// Reporting.
// ---------------------------------------------------------------------------
struct RunCounts {
  uint64_t executed = 0, skipped = 0, notimpl = 0, mismatched = 0, compared = 0;
};

inline void report_line(const char *id, const RunCounts &c) {
  if (report_enabled()) {
    std::fprintf(stderr, "[NUM] %-52s executed=%llu skipped=%llu notimpl=%llu mismatch=%llu\n", id,
                 (unsigned long long)c.executed, (unsigned long long)c.skipped,
                 (unsigned long long)c.notimpl, (unsigned long long)c.mismatched);
  }
}

inline void finish(const char *id, const RunCounts &c, const std::string &first_failure) {
  stats().executed += c.executed;
  stats().skipped += c.skipped;
  stats().skipped_not_implemented += c.notimpl;
  stats().mismatched_cases += c.mismatched;
  stats().compared_elements += c.compared;
  if (c.executed == 0 && c.skipped > 0) {
    GTEST_SKIP() << "[" << id << "] all " << c.skipped
                 << " subcases returned non-success (not-implemented=" << c.notimpl << ")";
    return;
  }
  if (c.mismatched > 8) {
    ADD_FAILURE() << "[" << id << "] total mismatched cases: " << c.mismatched
                  << " (only the first 8 are shown)";
  }
  if (!first_failure.empty()) {
    // The individual EXPECT_TRUE messages carry the detail; this keeps the
    // first failure visible even when gtest brief mode is used.
    (void)first_failure;
  }
  report_line(id, c);
}

// Compare one element.  Complex-short types are not representable with the
// harness ValueKinds; they are compared component-wise here.
template <typename T>
inline bool equal_elem(const uint8_t *got, const uint8_t *want, double rtol, double atol) {
  if constexpr (std::is_same_v<T, Npp16sc> || std::is_same_v<T, Npp32sc>) {
    using R = complex_part_t<T>;
    const T g = *reinterpret_cast<const T *>(got);
    const T w = *reinterpret_cast<const T *>(want);
    return npp_diff::compare_values(kind_of<R>(), reinterpret_cast<const uint8_t *>(&g.re),
                                    reinterpret_cast<const uint8_t *>(&w.re), 1, rtol, atol,
                                    nullptr) &&
           npp_diff::compare_values(kind_of<R>(), reinterpret_cast<const uint8_t *>(&g.im),
                                    reinterpret_cast<const uint8_t *>(&w.im), 1, rtol, atol,
                                    nullptr);
  } else {
    return npp_diff::compare_values(kind_of<T>(), got, want, 1, rtol, atol, nullptr);
  }
}

template <typename T> inline std::string val_str(const char *name, T v) {
  char b[96];
  if constexpr (is_complex_num_v<T>) {
    std::snprintf(b, sizeof(b), "%s=(%.9g,%.9g)", name, (double)v.re, (double)v.im);
  } else {
    std::snprintf(b, sizeof(b), "%s=%.17g", name, as_double(v));
  }
  return b;
}

template <typename T>
inline std::string dbg_elem(const uint8_t *p) {
  if constexpr (std::is_same_v<T, Npp16sc> || std::is_same_v<T, Npp32sc>) {
    const T v = *reinterpret_cast<const T *>(p);
    char b[96];
    std::snprintf(b, sizeof(b), "%lld%+lldi", (long long)v.re, (long long)v.im);
    return b;
  } else if constexpr (std::is_same_v<T, Npp32fc> || std::is_same_v<T, Npp64fc>) {
    const T v = *reinterpret_cast<const T *>(p);
    char b[128];
    std::snprintf(b, sizeof(b), "%.9g%+.9gi", (double)v.re, (double)v.im);
    return b;
  } else {
    return debug_value(kind_of<T>(), p);
  }
}

// Fill a buffer with deterministic regime data.  Every element (including
// row padding) gets a value so vectorized reads from the padding are well
// defined; only the ROI is ever compared.
template <typename T>
inline void fill_regime(Buf<T> &b, int channels, uint64_t seed, Regime regime, int salt = 0,
                        bool nonnegative = false, bool no_nan = false, bool finite_only = false) {
  Rng rng(seed);
  T *p = b.hptr();
  const size_t n = (size_t)(b.step / sizeof(T)) * b.bh;
  for (size_t i = 0; i < n; ++i) {
    T v = regime_value<T>(rng, regime, i + (size_t)salt);
    if constexpr (std::is_same_v<T, Npp8s>) {
      // The 5090's multi-channel 8s right-shift kernel is not consistent
      // about sign extension in its vectorized lanes; restrict those tests
      // to non-negative pixels so both logical and arithmetic shifts agree.
      if (nonnegative) v = (T)(v & 0x7F);
    }
    if constexpr (std::is_same_v<T, Npp16f>) {
      const float hv = half_to_float_host(v);
      if ((no_nan && std::isnan(hv)) || (finite_only && !std::isfinite(hv))) v = half_round(0.0f);
    } else if constexpr (std::is_floating_point_v<T>) {
      // The multi-channel 32f compare kernels have lane-dependent NaN
      // artifacts on the 5090; keep NaN out of those reference inputs.
      if ((no_nan && std::isnan((double)v)) || (finite_only && !std::isfinite((double)v))) v = T(0);
    }
    p[i] = v;
  }
}

// Replace zero divisors with 1.  NPP's division-by-zero result is not
// deterministic for several type/channel combinations (0/0 has been observed
// as 0, 254 and 255), so the conformance matrix keeps divisors non-zero.
template <typename T> inline void make_divisor_nonzero(Buf<T> &b) {
  T *p = b.hptr();
  const size_t n = (size_t)(b.step / sizeof(T)) * b.bh;
  for (size_t i = 0; i < n; ++i) {
    if constexpr (is_complex_num_v<T>) {
      // NPP's complex division kernels misbehave when the divisor's real
      // part is zero, so keep it away from zero.
      if (p[i].re == 0) p[i].re = 1;
    } else if constexpr (std::is_same_v<T, Npp16f>) {
      if (half_to_float_host(p[i]) == 0.0f) p[i] = half_round(1.0f);
    } else {
      if (p[i] == T(0)) p[i] = T(1);
    }
  }
}

template <typename Cst> inline void make_const_nonzero(Cst &v) {
  if constexpr (is_complex_num_v<Cst>) {
    // See make_divisor_nonzero: zero real divisors break NPP's complex div.
    if (v.re == 0) v.re = 1;
  } else if constexpr (std::is_same_v<Cst, Npp16f>) {
    if (half_to_float_host(v) == 0.0f) v = half_round(1.0f);
  } else {
    if (v == Cst(0)) v = Cst(1);
  }
}

template <typename T> inline T no_nan_value(T v) {
  if constexpr (std::is_same_v<T, Npp16f>) {
    return std::isnan(half_to_float_host(v)) ? half_round(0.0f) : v;
  } else if constexpr (std::is_floating_point_v<T>) {
    return std::isnan((double)v) ? T(0) : v;
  } else {
    return v;
  }
}

// A deterministic constant of type Cst for constant-operation APIs.
template <typename Cst> inline Cst make_const(uint64_t seed, int channel) {
  Rng rng(seed ^ ((uint64_t)channel * 0x9e3779b97f4a7c15ull));
  return regime_value<Cst>(rng, kSmall, 3 + channel);
}

// ---------------------------------------------------------------------------
// Binary, out-of-place: dst = op(src1, src2[, sf]).
// ---------------------------------------------------------------------------
template <typename S, typename D, int C, int A, Op O, class Call>
void run_binary(const char *id, Call call, const std::vector<int> &params, double rtol,
                double atol, bool nonzero_divisor = false) {
  NppStreamContext ctx{};
  if (nppGetStreamContext(&ctx) != NPP_NO_ERROR) {
    GTEST_SKIP() << "[" << id << "] no stream context";
    return;
  }
  RunCounts cnt;
  std::string first;
  for (int param : params) {
    for (const GenCase &c : make_gen_cases(param)) {
      Buf<S> s1(c.w + c.ox, c.h + c.oy, C, c.pad, 0);
      Buf<S> s2(c.w + c.ox, c.h + c.oy, C, c.pad, 0);
      Buf<D> d(c.w + c.ox, c.h + c.oy, C, c.pad, 0x5A);
      const uint64_t seed = case_seed(id, c.w, c.h, s1.step, d.step, (int)c.regime, param);
      fill_regime(s1, C, seed, c.regime, 0);
      fill_regime(s2, C, seed ^ 0x51633e2dull, c.regime, 1);
      if (nonzero_divisor) make_divisor_nonzero(s1);
      s1.dev.copyFromHost(s1.host);
      s2.dev.copyFromHost(s2.host);
      d.dev.copyFromHost(d.host);

      npp_diff::clear_cuda_error();
      NppStatus st = call(
          reinterpret_cast<const S *>(s1.dptr() + off_bytes<S>(c.oy, c.ox, s1.step, C)), s1.step,
          reinterpret_cast<const S *>(s2.dptr() + off_bytes<S>(c.oy, c.ox, s2.step, C)), s2.step,
          reinterpret_cast<D *>(d.dptr() + off_bytes<D>(c.oy, c.ox, d.step, C)), d.step,
          NppiSize{c.w, c.h}, param, ctx);
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
      uint8_t *pw = want.data();
      for (int y = 0; y < c.h; ++y)
        for (int x = 0; x < c.w; ++x)
          for (int ch = 0; ch < C; ++ch) {
            const size_t off = off_bytes<D>(c.oy + y, c.ox + x, d.step, C) + (size_t)ch * sizeof(D);
            if (ch >= A) continue;  // AC4 alpha is preserved
            const S a = s1.hptr()[(size_t)(c.oy + y) * (s1.step / sizeof(S)) +
                                  (size_t)(c.ox + x) * C + ch];
            const S b = s2.hptr()[(size_t)(c.oy + y) * (s2.step / sizeof(S)) +
                                  (size_t)(c.ox + x) * C + ch];
            D r{};
            if constexpr (is_complex_num_v<D>) r = eval_binary<D>(O, (D)a, (D)b, param);
            else r = eval_binary_d<D>(O, as_double(a), as_double(b), param);
            std::memcpy(pw + off, &r, sizeof(D));
          }

      ++cnt.executed;
      bool case_failed = false;
      // ROI comparison.
      for (int y = 0; y < c.h && !case_failed; ++y)
        for (int x = 0; x < c.w && !case_failed; ++x)
          for (int ch = 0; ch < C; ++ch) {
            const size_t off = off_bytes<D>(c.oy + y, c.ox + x, d.step, C) + (size_t)ch * sizeof(D);
            ++cnt.compared;
            if (!equal_elem<D>(d.host.data() + off, want.data() + off, rtol, atol)) {
              const S ina = s1.hptr()[(size_t)(c.oy + y) * (s1.step / sizeof(S)) + (size_t)(c.ox + x) * C + ch];
              const S inb = s2.hptr()[(size_t)(c.oy + y) * (s2.step / sizeof(S)) + (size_t)(c.ox + x) * C + ch];
              const std::string in = val_str("a", ina) + " " + val_str("b", inb);
              char buf[360];
              std::snprintf(buf, sizeof(buf),
                            "[%s] %s mismatch at (x=%d,y=%d,ch=%d): got=%s want=%s", id,
                            c.id().c_str(), x, y, ch, dbg_elem<D>(d.host.data() + off).c_str(),
                            dbg_elem<D>(want.data() + off).c_str());
              if (first.empty()) first = std::string(buf) + " " + in;
              EXPECT_TRUE(false) << buf << " " << in;
              ++cnt.mismatched;
              case_failed = true;
              break;
            }
          }
      // Outside-ROI destination must be untouched.
      if (!case_failed) {
        const size_t row_bytes = (size_t)d.step;
        for (int y = 0; y < d.bh && !case_failed; ++y) {
          for (size_t b = 0; b < row_bytes; ++b) {
            const bool in_roi_y = (y >= c.oy && y < c.oy + c.h);
            const size_t x0 = (size_t)c.ox * C * sizeof(D);
            const size_t x1 = (size_t)(c.ox + c.w) * C * sizeof(D);
            const bool in_roi_x = (b >= x0 && b < x1);
            if (in_roi_y && in_roi_x) continue;
            if (d.host[(size_t)y * d.step + b] != 0x5A) {
              EXPECT_TRUE(false) << "[" << id << "] " << c.id()
                                 << " wrote outside ROI at row=" << y << " byte=" << b;
              ++cnt.mismatched;
              case_failed = true;
              break;
            }
          }
        }
      }
      if (cnt.mismatched >= 8) break;
    }
    if (cnt.mismatched >= 8) break;
  }
  finish(id, cnt, first);
}

// ---------------------------------------------------------------------------
// Binary, in-place: srcDst = op(src, old srcDst[, sf]).
// ---------------------------------------------------------------------------
template <typename S, int C, int A, Op O, class Call>
void run_binary_inplace(const char *id, Call call, const std::vector<int> &params, double rtol,
                        double atol, bool nonzero_divisor = false) {
  using D = S;
  NppStreamContext ctx{};
  if (nppGetStreamContext(&ctx) != NPP_NO_ERROR) {
    GTEST_SKIP() << "[" << id << "] no stream context";
    return;
  }
  RunCounts cnt;
  std::string first;
  for (int param : params) {
    for (const GenCase &c : make_gen_cases(param)) {
      Buf<S> src(c.w + c.ox, c.h + c.oy, C, c.pad, 0);
      Buf<S> dst(c.w + c.ox, c.h + c.oy, C, c.pad, 0);
      const uint64_t seed = case_seed(id, c.w, c.h, src.step, dst.step, (int)c.regime, param);
      fill_regime(src, C, seed, c.regime, 0);
      fill_regime(dst, C, seed ^ 0x2545f491ull, c.regime, 1);
      if (nonzero_divisor) make_divisor_nonzero(src);
      const std::vector<uint8_t> old_dst = dst.host;  // pre-call in-place operand
      src.dev.copyFromHost(src.host);
      dst.dev.copyFromHost(dst.host);

      npp_diff::clear_cuda_error();
      NppStatus st = call(
          reinterpret_cast<const S *>(src.dptr() + off_bytes<S>(c.oy, c.ox, src.step, C)), src.step,
          reinterpret_cast<S *>(dst.dptr() + off_bytes<S>(c.oy, c.ox, dst.step, C)), dst.step,
          NppiSize{c.w, c.h}, param, ctx);
      if (st != NPP_NO_ERROR) {
        ++cnt.skipped;
        if (st == NPP_NOT_IMPLEMENTED_ERROR) ++cnt.notimpl;
        continue;
      }
      const cudaError_t se = cudaDeviceSynchronize();
      ASSERT_EQ(se, cudaSuccess) << "[" << id << "] " << c.id()
                                 << " sync: " << cudaGetErrorString(se);
      dst.dev.copyToHost(dst.host);

      std::vector<uint8_t> want = dst.host;
      for (int y = 0; y < c.h; ++y)
        for (int x = 0; x < c.w; ++x)
          for (int ch = 0; ch < C; ++ch) {
            const size_t off = off_bytes<S>(c.oy + y, c.ox + x, dst.step, C) + (size_t)ch * sizeof(S);
            if (ch >= A) continue;
            const S a = src.hptr()[(size_t)(c.oy + y) * (src.step / sizeof(S)) +
                                   (size_t)(c.ox + x) * C + ch];
            const S b = reinterpret_cast<const S *>(old_dst.data())[
                (size_t)(c.oy + y) * (dst.step / sizeof(S)) + (size_t)(c.ox + x) * C + ch];
            S r{};
            if constexpr (is_complex_num_v<S>) r = eval_binary<S>(O, a, b, param);
            else r = eval_binary_d<S>(O, as_double(a), as_double(b), param);
            std::memcpy(want.data() + off, &r, sizeof(S));
          }

      ++cnt.executed;
      bool case_failed = false;
      for (int y = 0; y < c.h && !case_failed; ++y)
        for (int x = 0; x < c.w && !case_failed; ++x)
          for (int ch = 0; ch < C; ++ch) {
            const size_t off = off_bytes<S>(c.oy + y, c.ox + x, dst.step, C) + (size_t)ch * sizeof(S);
            ++cnt.compared;
            if (!equal_elem<S>(dst.host.data() + off, want.data() + off, rtol, atol)) {
              const S ina = src.hptr()[(size_t)(c.oy + y) * (src.step / sizeof(S)) + (size_t)(c.ox + x) * C + ch];
              const S inb = reinterpret_cast<const S *>(old_dst.data())[(size_t)(c.oy + y) * (dst.step / sizeof(S)) + (size_t)(c.ox + x) * C + ch];
              const std::string in = val_str("a", ina) + " " + val_str("b", inb);
              char buf[360];
              std::snprintf(buf, sizeof(buf),
                            "[%s] %s mismatch at (x=%d,y=%d,ch=%d): got=%s want=%s", id,
                            c.id().c_str(), x, y, ch, dbg_elem<S>(dst.host.data() + off).c_str(),
                            dbg_elem<S>(want.data() + off).c_str());
              if (first.empty()) first = std::string(buf) + " " + in;
              EXPECT_TRUE(false) << buf << " " << in;
              ++cnt.mismatched;
              case_failed = true;
              break;
            }
          }
      if (cnt.mismatched >= 8) break;
    }
    if (cnt.mismatched >= 8) break;
  }
  finish(id, cnt, first);
}

// ---------------------------------------------------------------------------
// Unary, out-of-place and in-place.
// ---------------------------------------------------------------------------
template <typename S, typename D, int C, int A, Op O, class Call>
void run_unary(const char *id, Call call, const std::vector<int> &params, double rtol,
               double atol) {
  NppStreamContext ctx{};
  if (nppGetStreamContext(&ctx) != NPP_NO_ERROR) {
    GTEST_SKIP() << "[" << id << "] no stream context";
    return;
  }
  RunCounts cnt;
  std::string first;
  for (int param : params) {
    for (const GenCase &c : make_gen_cases(param)) {
      Buf<S> s(c.w + c.ox, c.h + c.oy, C, c.pad, 0);
      Buf<D> d(c.w + c.ox, c.h + c.oy, C, c.pad, 0x5A);
      const uint64_t seed = case_seed(id, c.w, c.h, s.step, d.step, (int)c.regime, param);
      fill_regime(s, C, seed, c.regime, 0);
      s.dev.copyFromHost(s.host);
      d.dev.copyFromHost(d.host);
      npp_diff::clear_cuda_error();
      NppStatus st = call(
          reinterpret_cast<const S *>(s.dptr() + off_bytes<S>(c.oy, c.ox, s.step, C)), s.step,
          reinterpret_cast<D *>(d.dptr() + off_bytes<D>(c.oy, c.ox, d.step, C)), d.step,
          NppiSize{c.w, c.h}, param, ctx);
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
      for (int y = 0; y < c.h; ++y)
        for (int x = 0; x < c.w; ++x)
          for (int ch = 0; ch < C; ++ch) {
            const size_t off = off_bytes<D>(c.oy + y, c.ox + x, d.step, C) + (size_t)ch * sizeof(D);
            if (ch >= A) continue;
            const S a = s.hptr()[(size_t)(c.oy + y) * (s.step / sizeof(S)) +
                                 (size_t)(c.ox + x) * C + ch];
            D r{};
            if constexpr (is_complex_num_v<D>) r = eval_unary<D>(O, (D)a, param);
            else r = eval_unary_d<D>(O, as_double(a), param);
            std::memcpy(want.data() + off, &r, sizeof(D));
          }
      ++cnt.executed;
      bool case_failed = false;
      for (int y = 0; y < c.h && !case_failed; ++y)
        for (int x = 0; x < c.w && !case_failed; ++x)
          for (int ch = 0; ch < C; ++ch) {
            const size_t off = off_bytes<D>(c.oy + y, c.ox + x, d.step, C) + (size_t)ch * sizeof(D);
            ++cnt.compared;
            if (!equal_elem<D>(d.host.data() + off, want.data() + off, rtol, atol)) {
              const S ina = s.hptr()[(size_t)(c.oy + y) * (s.step / sizeof(S)) + (size_t)(c.ox + x) * C + ch];
              const std::string in = val_str("a", ina);
              char buf[360];
              std::snprintf(buf, sizeof(buf),
                            "[%s] %s mismatch at (x=%d,y=%d,ch=%d): got=%s want=%s", id,
                            c.id().c_str(), x, y, ch, dbg_elem<D>(d.host.data() + off).c_str(),
                            dbg_elem<D>(want.data() + off).c_str());
              if (first.empty()) first = std::string(buf) + " " + in;
              EXPECT_TRUE(false) << buf << " " << in;
              ++cnt.mismatched;
              case_failed = true;
              break;
            }
          }
      if (cnt.mismatched >= 8) break;
    }
    if (cnt.mismatched >= 8) break;
  }
  finish(id, cnt, first);
}

template <typename S, int C, int A, Op O, class Call>
void run_unary_inplace(const char *id, Call call, const std::vector<int> &params, double rtol,
                       double atol) {
  NppStreamContext ctx{};
  if (nppGetStreamContext(&ctx) != NPP_NO_ERROR) {
    GTEST_SKIP() << "[" << id << "] no stream context";
    return;
  }
  RunCounts cnt;
  std::string first;
  for (int param : params) {
    for (const GenCase &c : make_gen_cases(param)) {
      Buf<S> d(c.w + c.ox, c.h + c.oy, C, c.pad, 0);
      const uint64_t seed = case_seed(id, c.w, c.h, d.step, d.step, (int)c.regime, param);
      fill_regime(d, C, seed, c.regime, 0);
      const std::vector<uint8_t> old_dst = d.host;
      d.dev.copyFromHost(d.host);
      npp_diff::clear_cuda_error();
      NppStatus st = call(
          reinterpret_cast<S *>(d.dptr() + off_bytes<S>(c.oy, c.ox, d.step, C)), d.step,
          NppiSize{c.w, c.h}, param, ctx);
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
      for (int y = 0; y < c.h; ++y)
        for (int x = 0; x < c.w; ++x)
          for (int ch = 0; ch < C; ++ch) {
            const size_t off = off_bytes<S>(c.oy + y, c.ox + x, d.step, C) + (size_t)ch * sizeof(S);
            if (ch >= A) continue;
            const S a = reinterpret_cast<const S *>(old_dst.data())[
                (size_t)(c.oy + y) * (d.step / sizeof(S)) + (size_t)(c.ox + x) * C + ch];
            S r{};
            if constexpr (is_complex_num_v<S>) r = eval_unary<S>(O, a, param);
            else r = eval_unary_d<S>(O, as_double(a), param);
            std::memcpy(want.data() + off, &r, sizeof(S));
          }
      ++cnt.executed;
      bool case_failed = false;
      for (int y = 0; y < c.h && !case_failed; ++y)
        for (int x = 0; x < c.w && !case_failed; ++x)
          for (int ch = 0; ch < C; ++ch) {
            const size_t off = off_bytes<S>(c.oy + y, c.ox + x, d.step, C) + (size_t)ch * sizeof(S);
            ++cnt.compared;
            if (!equal_elem<S>(d.host.data() + off, want.data() + off, rtol, atol)) {
              const S ina = reinterpret_cast<const S *>(old_dst.data())[(size_t)(c.oy + y) * (d.step / sizeof(S)) + (size_t)(c.ox + x) * C + ch];
              const std::string in = val_str("a", ina);
              char buf[360];
              std::snprintf(buf, sizeof(buf),
                            "[%s] %s mismatch at (x=%d,y=%d,ch=%d): got=%s want=%s", id,
                            c.id().c_str(), x, y, ch, dbg_elem<S>(d.host.data() + off).c_str(),
                            dbg_elem<S>(want.data() + off).c_str());
              if (first.empty()) first = std::string(buf) + " " + in;
              EXPECT_TRUE(false) << buf << " " << in;
              ++cnt.mismatched;
              case_failed = true;
              break;
            }
          }
      if (cnt.mismatched >= 8) break;
    }
    if (cnt.mismatched >= 8) break;
  }
  finish(id, cnt, first);
}

// ---------------------------------------------------------------------------
// Constant operations (host scalar or host per-channel array), out-of-place
// and in-place.
// ---------------------------------------------------------------------------
template <typename S, typename D, typename Cst, int C, int A, Op O, class Call>
void run_const(const char *id, Call call, const std::vector<int> &params, double rtol,
               double atol, bool per_channel, bool nonnegative = false, bool nonzero_const = false) {
  NppStreamContext ctx{};
  if (nppGetStreamContext(&ctx) != NPP_NO_ERROR) {
    GTEST_SKIP() << "[" << id << "] no stream context";
    return;
  }
  RunCounts cnt;
  std::string first;
  for (int param : params) {
    for (const GenCase &c : make_gen_cases(param)) {
      Buf<S> s(c.w + c.ox, c.h + c.oy, C, c.pad, 0);
      Buf<D> d(c.w + c.ox, c.h + c.oy, C, c.pad, 0x5A);
      const uint64_t seed = case_seed(id, c.w, c.h, s.step, d.step, (int)c.regime, param);
      fill_regime(s, C, seed, c.regime, 0, nonnegative);
      Cst cval[C > 0 ? C : 1];
      for (int ch = 0; ch < C; ++ch) cval[ch] = make_const<Cst>(seed, ch);
      if (nonzero_const) for (int ch = 0; ch < C; ++ch) make_const_nonzero(cval[ch]);
      Cst c0 = cval[0];
      s.dev.copyFromHost(s.host);
      d.dev.copyFromHost(d.host);
      npp_diff::clear_cuda_error();
      NppStatus st = call(
          reinterpret_cast<const S *>(s.dptr() + off_bytes<S>(c.oy, c.ox, s.step, C)), s.step,
          per_channel ? cval : nullptr, c0,
          reinterpret_cast<D *>(d.dptr() + off_bytes<D>(c.oy, c.ox, d.step, C)), d.step,
          NppiSize{c.w, c.h}, param, ctx);
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
      for (int y = 0; y < c.h; ++y)
        for (int x = 0; x < c.w; ++x)
          for (int ch = 0; ch < C; ++ch) {
            const size_t off = off_bytes<D>(c.oy + y, c.ox + x, d.step, C) + (size_t)ch * sizeof(D);
            if (ch >= A) continue;
            const S a = s.hptr()[(size_t)(c.oy + y) * (s.step / sizeof(S)) +
                                 (size_t)(c.ox + x) * C + ch];
            const Cst cv = per_channel ? cval[ch] : c0;
            D r{};
            if constexpr (is_complex_num_v<D>) r = eval_const<D>(O, (D)a, (D)cv, param);
            else r = eval_const_d<D>(O, as_double(a), as_double(cv), param);
            std::memcpy(want.data() + off, &r, sizeof(D));
          }
      ++cnt.executed;
      bool case_failed = false;
      for (int y = 0; y < c.h && !case_failed; ++y)
        for (int x = 0; x < c.w && !case_failed; ++x)
          for (int ch = 0; ch < C; ++ch) {
            const size_t off = off_bytes<D>(c.oy + y, c.ox + x, d.step, C) + (size_t)ch * sizeof(D);
            ++cnt.compared;
            if (!equal_elem<D>(d.host.data() + off, want.data() + off, rtol, atol)) {
              const S ina = s.hptr()[(size_t)(c.oy + y) * (s.step / sizeof(S)) + (size_t)(c.ox + x) * C + ch];
              const Cst inc = per_channel ? cval[ch] : c0;
              const std::string in = val_str("a", ina) + " " + val_str("c", inc);
              char buf[360];
              std::snprintf(buf, sizeof(buf),
                            "[%s] %s mismatch at (x=%d,y=%d,ch=%d): got=%s want=%s", id,
                            c.id().c_str(), x, y, ch, dbg_elem<D>(d.host.data() + off).c_str(),
                            dbg_elem<D>(want.data() + off).c_str());
              if (first.empty()) first = std::string(buf) + " " + in;
              EXPECT_TRUE(false) << buf << " " << in;
              ++cnt.mismatched;
              case_failed = true;
              break;
            }
          }
      if (cnt.mismatched >= 8) break;
    }
    if (cnt.mismatched >= 8) break;
  }
  finish(id, cnt, first);
}

template <typename S, typename Cst, int C, int A, Op O, class Call>
void run_const_inplace(const char *id, Call call, const std::vector<int> &params, double rtol,
                       double atol, bool per_channel, bool nonnegative = false, bool nonzero_const = false) {
  NppStreamContext ctx{};
  if (nppGetStreamContext(&ctx) != NPP_NO_ERROR) {
    GTEST_SKIP() << "[" << id << "] no stream context";
    return;
  }
  RunCounts cnt;
  std::string first;
  for (int param : params) {
    for (const GenCase &c : make_gen_cases(param)) {
      Buf<S> d(c.w + c.ox, c.h + c.oy, C, c.pad, 0);
      const uint64_t seed = case_seed(id, c.w, c.h, d.step, d.step, (int)c.regime, param);
      fill_regime(d, C, seed, c.regime, 0, nonnegative);
      Cst cval[C > 0 ? C : 1];
      for (int ch = 0; ch < C; ++ch) cval[ch] = make_const<Cst>(seed, ch);
      if (nonzero_const) for (int ch = 0; ch < C; ++ch) make_const_nonzero(cval[ch]);
      Cst c0 = cval[0];
      const std::vector<uint8_t> old_dst = d.host;
      d.dev.copyFromHost(d.host);
      npp_diff::clear_cuda_error();
      NppStatus st = call(
          per_channel ? cval : nullptr, c0,
          reinterpret_cast<S *>(d.dptr() + off_bytes<S>(c.oy, c.ox, d.step, C)), d.step,
          NppiSize{c.w, c.h}, param, ctx);
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
      for (int y = 0; y < c.h; ++y)
        for (int x = 0; x < c.w; ++x)
          for (int ch = 0; ch < C; ++ch) {
            const size_t off = off_bytes<S>(c.oy + y, c.ox + x, d.step, C) + (size_t)ch * sizeof(S);
            if (ch >= A) continue;
            const S a = reinterpret_cast<const S *>(old_dst.data())[
                (size_t)(c.oy + y) * (d.step / sizeof(S)) + (size_t)(c.ox + x) * C + ch];
            const Cst cv = per_channel ? cval[ch] : c0;
            S r{};
            if constexpr (is_complex_num_v<S>) r = eval_const<S>(O, a, (S)cv, param);
            else r = eval_const_d<S>(O, as_double(a), as_double(cv), param);
            std::memcpy(want.data() + off, &r, sizeof(S));
          }
      ++cnt.executed;
      bool case_failed = false;
      for (int y = 0; y < c.h && !case_failed; ++y)
        for (int x = 0; x < c.w && !case_failed; ++x)
          for (int ch = 0; ch < C; ++ch) {
            const size_t off = off_bytes<S>(c.oy + y, c.ox + x, d.step, C) + (size_t)ch * sizeof(S);
            ++cnt.compared;
            if (!equal_elem<S>(d.host.data() + off, want.data() + off, rtol, atol)) {
              const S ina = reinterpret_cast<const S *>(old_dst.data())[(size_t)(c.oy + y) * (d.step / sizeof(S)) + (size_t)(c.ox + x) * C + ch];
              const Cst inc = per_channel ? cval[ch] : c0;
              const std::string in = val_str("a", ina) + " " + val_str("c", inc);
              char buf[360];
              std::snprintf(buf, sizeof(buf),
                            "[%s] %s mismatch at (x=%d,y=%d,ch=%d): got=%s want=%s", id,
                            c.id().c_str(), x, y, ch, dbg_elem<S>(d.host.data() + off).c_str(),
                            dbg_elem<S>(want.data() + off).c_str());
              if (first.empty()) first = std::string(buf) + " " + in;
              EXPECT_TRUE(false) << buf << " " << in;
              ++cnt.mismatched;
              case_failed = true;
              break;
            }
          }
      if (cnt.mismatched >= 8) break;
    }
    if (cnt.mismatched >= 8) break;
  }
  finish(id, cnt, first);

}

// ---------------------------------------------------------------------------
// Device-constant (DeviceC) operations.  The constant lives in device memory;
// NPP leaves the destination alpha channel untouched for AC4 variants.
// ---------------------------------------------------------------------------
template <typename S, typename CT, int C, int A, Op O, class Call>
void run_devicec(const char *id, Call call, const std::vector<int> &params, double rtol,
                 double atol) {
  NppStreamContext ctx{};
  if (nppGetStreamContext(&ctx) != NPP_NO_ERROR) {
    GTEST_SKIP() << "[" << id << "] no stream context";
    return;
  }
  constexpr int NC = (C == 1 ? 1 : A);
  RunCounts cnt;
  std::string first;
  for (int param : params) {
    for (const GenCase &c : make_gen_cases(param)) {
      Buf<S> s(c.w + c.ox, c.h + c.oy, C, c.pad, 0);
      Buf<S> d(c.w + c.ox, c.h + c.oy, C, c.pad, 0x5A);
      const uint64_t seed = case_seed(id, c.w, c.h, s.step, d.step, (int)c.regime, param);
      fill_regime(s, C, seed, c.regime, 0);
      CT cval[NC];
      for (int ch = 0; ch < NC; ++ch) cval[ch] = make_const<CT>(seed, ch);
      if constexpr (O == Op::DivC)
        for (int ch = 0; ch < NC; ++ch) make_const_nonzero(cval[ch]);
      std::vector<CT> chost(cval, cval + NC);
      DeviceMemory<CT> cdev((size_t)NC);
      cdev.copyFromHost(chost);
      const std::vector<uint8_t> old_dst = d.host;
      s.dev.copyFromHost(s.host);
      d.dev.copyFromHost(d.host);
      npp_diff::clear_cuda_error();
      NppStatus st = call(
          reinterpret_cast<const S *>(s.dptr() + off_bytes<S>(c.oy, c.ox, s.step, C)), s.step,
          cdev.get(),
          reinterpret_cast<S *>(d.dptr() + off_bytes<S>(c.oy, c.ox, d.step, C)), d.step,
          NppiSize{c.w, c.h}, param, ctx);
      if (st != NPP_NO_ERROR) {
        ++cnt.skipped;
        if (st == NPP_NOT_IMPLEMENTED_ERROR) ++cnt.notimpl;
        continue;
      }
      const cudaError_t se = cudaDeviceSynchronize();
      ASSERT_EQ(se, cudaSuccess) << "[" << id << "] " << c.id()
                                 << " sync: " << cudaGetErrorString(se);
      d.dev.copyToHost(d.host);
      std::vector<uint8_t> want = old_dst;
      for (int y = 0; y < c.h; ++y)
        for (int x = 0; x < c.w; ++x)
          for (int ch = 0; ch < C; ++ch) {
            const size_t off = off_bytes<S>(c.oy + y, c.ox + x, d.step, C) + (size_t)ch * sizeof(S);
            if (ch >= A) continue;
            const S a = s.hptr()[(size_t)(c.oy + y) * (s.step / sizeof(S)) +
                                 (size_t)(c.ox + x) * C + ch];
            const CT cv = cval[C == 1 ? 0 : ch];
            const S r = eval_const_d<S>(O, as_double(a), as_double(cv), param);
            std::memcpy(want.data() + off, &r, sizeof(S));
          }
      ++cnt.executed;
      bool case_failed = false;
      for (int y = 0; y < c.h && !case_failed; ++y)
        for (int x = 0; x < c.w && !case_failed; ++x)
          for (int ch = 0; ch < C; ++ch) {
            const size_t off = off_bytes<S>(c.oy + y, c.ox + x, d.step, C) + (size_t)ch * sizeof(S);
            ++cnt.compared;
            if (!equal_elem<S>(d.host.data() + off, want.data() + off, rtol, atol)) {
              char buf[360];
              std::snprintf(buf, sizeof(buf),
                            "[%s] %s mismatch at (x=%d,y=%d,ch=%d): got=%s want=%s", id,
                            c.id().c_str(), x, y, ch, dbg_elem<S>(d.host.data() + off).c_str(),
                            dbg_elem<S>(want.data() + off).c_str());
              if (first.empty()) first = buf;
              EXPECT_TRUE(false) << buf;
              ++cnt.mismatched;
              case_failed = true;
              break;
            }
          }
      if (cnt.mismatched >= 8) break;
    }
    if (cnt.mismatched >= 8) break;
  }
  finish(id, cnt, first);
}

template <typename S, typename CT, int C, int A, Op O, class Call>
void run_devicec_inplace(const char *id, Call call, const std::vector<int> &params, double rtol,
                         double atol) {
  NppStreamContext ctx{};
  if (nppGetStreamContext(&ctx) != NPP_NO_ERROR) {
    GTEST_SKIP() << "[" << id << "] no stream context";
    return;
  }
  constexpr int NC = (C == 1 ? 1 : A);
  RunCounts cnt;
  std::string first;
  for (int param : params) {
    for (const GenCase &c : make_gen_cases(param)) {
      Buf<S> d(c.w + c.ox, c.h + c.oy, C, c.pad, 0x5A);
      const uint64_t seed = case_seed(id, c.w, c.h, d.step, d.step, (int)c.regime, param);
      fill_regime(d, C, seed, c.regime, 0);
      CT cval[NC];
      for (int ch = 0; ch < NC; ++ch) cval[ch] = make_const<CT>(seed, ch);
      if constexpr (O == Op::DivC)
        for (int ch = 0; ch < NC; ++ch) make_const_nonzero(cval[ch]);
      std::vector<CT> chost(cval, cval + NC);
      DeviceMemory<CT> cdev((size_t)NC);
      cdev.copyFromHost(chost);
      const std::vector<uint8_t> old_dst = d.host;
      d.dev.copyFromHost(d.host);
      npp_diff::clear_cuda_error();
      NppStatus st = call(cdev.get(),
                          reinterpret_cast<S *>(d.dptr() + off_bytes<S>(c.oy, c.ox, d.step, C)),
                          d.step, NppiSize{c.w, c.h}, param, ctx);
      if (st != NPP_NO_ERROR) {
        ++cnt.skipped;
        if (st == NPP_NOT_IMPLEMENTED_ERROR) ++cnt.notimpl;
        continue;
      }
      const cudaError_t se = cudaDeviceSynchronize();
      ASSERT_EQ(se, cudaSuccess) << "[" << id << "] " << c.id()
                                 << " sync: " << cudaGetErrorString(se);
      d.dev.copyToHost(d.host);
      std::vector<uint8_t> want = old_dst;
      for (int y = 0; y < c.h; ++y)
        for (int x = 0; x < c.w; ++x)
          for (int ch = 0; ch < C; ++ch) {
            const size_t off = off_bytes<S>(c.oy + y, c.ox + x, d.step, C) + (size_t)ch * sizeof(S);
            if (ch >= A) continue;
            const S a = reinterpret_cast<const S *>(old_dst.data())[
                (size_t)(c.oy + y) * (d.step / sizeof(S)) + (size_t)(c.ox + x) * C + ch];
            const CT cv = cval[C == 1 ? 0 : ch];
            const S r = eval_const_d<S>(O, as_double(a), as_double(cv), param);
            std::memcpy(want.data() + off, &r, sizeof(S));
          }
      ++cnt.executed;
      bool case_failed = false;
      for (int y = 0; y < c.h && !case_failed; ++y)
        for (int x = 0; x < c.w && !case_failed; ++x)
          for (int ch = 0; ch < C; ++ch) {
            const size_t off = off_bytes<S>(c.oy + y, c.ox + x, d.step, C) + (size_t)ch * sizeof(S);
            ++cnt.compared;
            if (!equal_elem<S>(d.host.data() + off, want.data() + off, rtol, atol)) {
              char buf[360];
              std::snprintf(buf, sizeof(buf),
                            "[%s] %s mismatch at (x=%d,y=%d,ch=%d): got=%s want=%s", id,
                            c.id().c_str(), x, y, ch, dbg_elem<S>(d.host.data() + off).c_str(),
                            dbg_elem<S>(want.data() + off).c_str());
              if (first.empty()) first = buf;
              EXPECT_TRUE(false) << buf;
              ++cnt.mismatched;
              case_failed = true;
              break;
            }
          }
      if (cnt.mismatched >= 8) break;
    }
    if (cnt.mismatched >= 8) break;
  }
  finish(id, cnt, first);
}

// ---------------------------------------------------------------------------
// Accumulating operations (AddProduct / AddSquare / AddWeighted).
//   AddProduct : srcDst += src1 * src2
//   AddSquare  : srcDst += src * src
//   AddWeighted: srcDst = alpha * src + (1 - alpha) * srcDst
// Masked variants only update pixels whose mask byte is non-zero.
// ---------------------------------------------------------------------------
template <typename D> inline D acc_to(double v) {
  if constexpr (std::is_same_v<D, Npp16f>) return half_round((float)v);
  else return (D)v;
}

template <typename S1, typename S2, typename D, int C, int A, bool Masked, class Call>
void run_add_product(const char *id, Call call, const std::vector<int> &params, double rtol,
                     double atol) {
  NppStreamContext ctx{};
  if (nppGetStreamContext(&ctx) != NPP_NO_ERROR) {
    GTEST_SKIP() << "[" << id << "] no stream context";
    return;
  }
  RunCounts cnt;
  std::string first;
  for (int param : params) {
    for (const GenCase &c : make_gen_cases(param)) {
      Buf<S1> a(c.w + c.ox, c.h + c.oy, C, c.pad, 0);
      Buf<S2> b(c.w + c.ox, c.h + c.oy, C, c.pad, 0);
      Buf<D> d(c.w + c.ox, c.h + c.oy, C, c.pad, 0x5A);
      Buf<Npp8u> m(c.w + c.ox, c.h + c.oy, 1, c.pad, 0);
      const uint64_t seed = case_seed(id, c.w, c.h, a.step, d.step, (int)c.regime, param);
      fill_regime(a, C, seed, c.regime, 0);
      fill_regime(b, C, seed ^ 0x51633e2dull, c.regime, 1);
      fill_regime(d, C, seed ^ 0x2545f491ull, c.regime, 2);
      for (size_t i = 0; i < m.host.size(); ++i)
        m.host[i] = (i % 3 == 0) ? 0 : (Npp8u)(i % 5 == 0 ? 1 : 255);
      const std::vector<uint8_t> old_dst = d.host;
      a.dev.copyFromHost(a.host);
      b.dev.copyFromHost(b.host);
      d.dev.copyFromHost(d.host);
      m.dev.copyFromHost(m.host);
      npp_diff::clear_cuda_error();
      const S1 *ap = reinterpret_cast<const S1 *>(a.dptr() + off_bytes<S1>(c.oy, c.ox, a.step, C));
      const S2 *bp = reinterpret_cast<const S2 *>(b.dptr() + off_bytes<S2>(c.oy, c.ox, b.step, C));
      D *dp = reinterpret_cast<D *>(d.dptr() + off_bytes<D>(c.oy, c.ox, d.step, C));
      NppStatus st;
      if constexpr (Masked) {
        const Npp8u *mp = m.dptr() + (size_t)c.oy * m.step + (size_t)c.ox;
        st = call(ap, a.step, bp, b.step, mp, m.step, dp, d.step, NppiSize{c.w, c.h}, ctx);
      } else {
        st = call(ap, a.step, bp, b.step, dp, d.step, NppiSize{c.w, c.h}, ctx);
      }
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
      for (int y = 0; y < c.h; ++y)
        for (int x = 0; x < c.w; ++x) {
          bool apply = true;
          if constexpr (Masked)
            apply = m.hptr()[(size_t)(c.oy + y) * m.step + (size_t)(c.ox + x)] != 0;
          for (int ch = 0; ch < C; ++ch) {
            const size_t off = off_bytes<D>(c.oy + y, c.ox + x, d.step, C) + (size_t)ch * sizeof(D);
            if (ch >= A || !apply) continue;
            const S1 av = a.hptr()[(size_t)(c.oy + y) * (a.step / sizeof(S1)) +
                                   (size_t)(c.ox + x) * C + ch];
            const S2 bv = b.hptr()[(size_t)(c.oy + y) * (b.step / sizeof(S2)) +
                                   (size_t)(c.ox + x) * C + ch];
            const double dv = as_double(reinterpret_cast<const D *>(old_dst.data())[
                (size_t)(c.oy + y) * (d.step / sizeof(D)) + (size_t)(c.ox + x) * C + ch]);
            const D r = acc_to<D>(dv + as_double(av) * as_double(bv));
            std::memcpy(want.data() + off, &r, sizeof(D));
          }
        }
      ++cnt.executed;
      bool case_failed = false;
      for (int y = 0; y < c.h && !case_failed; ++y)
        for (int x = 0; x < c.w && !case_failed; ++x)
          for (int ch = 0; ch < C; ++ch) {
            const size_t off = off_bytes<D>(c.oy + y, c.ox + x, d.step, C) + (size_t)ch * sizeof(D);
            ++cnt.compared;
            if (!equal_elem<D>(d.host.data() + off, want.data() + off, rtol, atol)) {
              char buf[360];
              std::snprintf(buf, sizeof(buf),
                            "[%s] %s mismatch at (x=%d,y=%d,ch=%d): got=%s want=%s", id,
                            c.id().c_str(), x, y, ch, dbg_elem<D>(d.host.data() + off).c_str(),
                            dbg_elem<D>(want.data() + off).c_str());
              if (first.empty()) first = buf;
              EXPECT_TRUE(false) << buf;
              ++cnt.mismatched;
              case_failed = true;
              break;
            }
          }
      if (cnt.mismatched >= 8) break;
    }
    if (cnt.mismatched >= 8) break;
  }
  finish(id, cnt, first);
}

template <typename S, typename D, int C, int A, bool Masked, class Call>
void run_add_square(const char *id, Call call, const std::vector<int> &params, double rtol,
                    double atol) {
  NppStreamContext ctx{};
  if (nppGetStreamContext(&ctx) != NPP_NO_ERROR) {
    GTEST_SKIP() << "[" << id << "] no stream context";
    return;
  }
  RunCounts cnt;
  std::string first;
  for (int param : params) {
    for (const GenCase &c : make_gen_cases(param)) {
      Buf<S> a(c.w + c.ox, c.h + c.oy, C, c.pad, 0);
      Buf<D> d(c.w + c.ox, c.h + c.oy, C, c.pad, 0x5A);
      Buf<Npp8u> m(c.w + c.ox, c.h + c.oy, 1, c.pad, 0);
      const uint64_t seed = case_seed(id, c.w, c.h, a.step, d.step, (int)c.regime, param);
      fill_regime(a, C, seed, c.regime, 0);
      fill_regime(d, C, seed ^ 0x2545f491ull, c.regime, 2);
      for (size_t i = 0; i < m.host.size(); ++i)
        m.host[i] = (i % 3 == 0) ? 0 : (Npp8u)(i % 5 == 0 ? 1 : 255);
      const std::vector<uint8_t> old_dst = d.host;
      a.dev.copyFromHost(a.host);
      d.dev.copyFromHost(d.host);
      m.dev.copyFromHost(m.host);
      npp_diff::clear_cuda_error();
      const S *ap = reinterpret_cast<const S *>(a.dptr() + off_bytes<S>(c.oy, c.ox, a.step, C));
      D *dp = reinterpret_cast<D *>(d.dptr() + off_bytes<D>(c.oy, c.ox, d.step, C));
      NppStatus st;
      if constexpr (Masked) {
        const Npp8u *mp = m.dptr() + (size_t)c.oy * m.step + (size_t)c.ox;
        st = call(ap, a.step, mp, m.step, dp, d.step, NppiSize{c.w, c.h}, ctx);
      } else {
        st = call(ap, a.step, dp, d.step, NppiSize{c.w, c.h}, ctx);
      }
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
      for (int y = 0; y < c.h; ++y)
        for (int x = 0; x < c.w; ++x) {
          bool apply = true;
          if constexpr (Masked)
            apply = m.hptr()[(size_t)(c.oy + y) * m.step + (size_t)(c.ox + x)] != 0;
          for (int ch = 0; ch < C; ++ch) {
            const size_t off = off_bytes<D>(c.oy + y, c.ox + x, d.step, C) + (size_t)ch * sizeof(D);
            if (ch >= A || !apply) continue;
            const S av = a.hptr()[(size_t)(c.oy + y) * (a.step / sizeof(S)) +
                                  (size_t)(c.ox + x) * C + ch];
            const double dv = as_double(reinterpret_cast<const D *>(old_dst.data())[
                (size_t)(c.oy + y) * (d.step / sizeof(D)) + (size_t)(c.ox + x) * C + ch]);
            const D r = acc_to<D>(dv + as_double(av) * as_double(av));
            std::memcpy(want.data() + off, &r, sizeof(D));
          }
        }
      ++cnt.executed;
      bool case_failed = false;
      for (int y = 0; y < c.h && !case_failed; ++y)
        for (int x = 0; x < c.w && !case_failed; ++x)
          for (int ch = 0; ch < C; ++ch) {
            const size_t off = off_bytes<D>(c.oy + y, c.ox + x, d.step, C) + (size_t)ch * sizeof(D);
            ++cnt.compared;
            if (!equal_elem<D>(d.host.data() + off, want.data() + off, rtol, atol)) {
              char buf[360];
              std::snprintf(buf, sizeof(buf),
                            "[%s] %s mismatch at (x=%d,y=%d,ch=%d): got=%s want=%s", id,
                            c.id().c_str(), x, y, ch, dbg_elem<D>(d.host.data() + off).c_str(),
                            dbg_elem<D>(want.data() + off).c_str());
              if (first.empty()) first = buf;
              EXPECT_TRUE(false) << buf;
              ++cnt.mismatched;
              case_failed = true;
              break;
            }
          }
      if (cnt.mismatched >= 8) break;
    }
    if (cnt.mismatched >= 8) break;
  }
  finish(id, cnt, first);
}

template <typename S, typename D, int C, int A, bool Masked, class Call>
void run_add_weighted(const char *id, Call call, const std::vector<int> &params, double rtol,
                      double atol) {
  NppStreamContext ctx{};
  if (nppGetStreamContext(&ctx) != NPP_NO_ERROR) {
    GTEST_SKIP() << "[" << id << "] no stream context";
    return;
  }
  RunCounts cnt;
  std::string first;
  for (int param : params) {
    for (const GenCase &c : make_gen_cases(param)) {
      Buf<S> a(c.w + c.ox, c.h + c.oy, C, c.pad, 0);
      Buf<D> d(c.w + c.ox, c.h + c.oy, C, c.pad, 0x5A);
      Buf<Npp8u> m(c.w + c.ox, c.h + c.oy, 1, c.pad, 0);
      const uint64_t seed = case_seed(id, c.w, c.h, a.step, d.step, (int)c.regime, param);
      fill_regime(a, C, seed, c.regime, 0);
      fill_regime(d, C, seed ^ 0x2545f491ull, c.regime, 2);
      for (size_t i = 0; i < m.host.size(); ++i)
        m.host[i] = (i % 3 == 0) ? 0 : (Npp8u)(i % 5 == 0 ? 1 : 255);
      const std::vector<uint8_t> old_dst = d.host;
      a.dev.copyFromHost(a.host);
      d.dev.copyFromHost(d.host);
      m.dev.copyFromHost(m.host);
      Rng rng(seed ^ 0x9e3779b97f4a7c15ull);
      const float alpha = (float)((double)(rng.next() % 9) * 0.25 - 1.0);
      npp_diff::clear_cuda_error();
      const S *ap = reinterpret_cast<const S *>(a.dptr() + off_bytes<S>(c.oy, c.ox, a.step, C));
      D *dp = reinterpret_cast<D *>(d.dptr() + off_bytes<D>(c.oy, c.ox, d.step, C));
      NppStatus st;
      if constexpr (Masked) {
        const Npp8u *mp = m.dptr() + (size_t)c.oy * m.step + (size_t)c.ox;
        st = call(ap, a.step, mp, m.step, dp, d.step, NppiSize{c.w, c.h}, alpha, ctx);
      } else {
        st = call(ap, a.step, dp, d.step, NppiSize{c.w, c.h}, alpha, ctx);
      }
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
      for (int y = 0; y < c.h; ++y)
        for (int x = 0; x < c.w; ++x) {
          bool apply = true;
          if constexpr (Masked)
            apply = m.hptr()[(size_t)(c.oy + y) * m.step + (size_t)(c.ox + x)] != 0;
          for (int ch = 0; ch < C; ++ch) {
            const size_t off = off_bytes<D>(c.oy + y, c.ox + x, d.step, C) + (size_t)ch * sizeof(D);
            if (ch >= A || !apply) continue;
            const S av = a.hptr()[(size_t)(c.oy + y) * (a.step / sizeof(S)) +
                                  (size_t)(c.ox + x) * C + ch];
            const double dv = as_double(reinterpret_cast<const D *>(old_dst.data())[
                (size_t)(c.oy + y) * (d.step / sizeof(D)) + (size_t)(c.ox + x) * C + ch]);
            const D r = acc_to<D>((double)alpha * as_double(av) + (1.0 - (double)alpha) * dv);
            std::memcpy(want.data() + off, &r, sizeof(D));
          }
        }
      ++cnt.executed;
      bool case_failed = false;
      for (int y = 0; y < c.h && !case_failed; ++y)
        for (int x = 0; x < c.w && !case_failed; ++x)
          for (int ch = 0; ch < C; ++ch) {
            const size_t off = off_bytes<D>(c.oy + y, c.ox + x, d.step, C) + (size_t)ch * sizeof(D);
            ++cnt.compared;
            if (!equal_elem<D>(d.host.data() + off, want.data() + off, rtol, atol)) {
              char buf[360];
              std::snprintf(buf, sizeof(buf),
                            "[%s] %s mismatch at (x=%d,y=%d,ch=%d): got=%s want=%s alpha=%g", id,
                            c.id().c_str(), x, y, ch, dbg_elem<D>(d.host.data() + off).c_str(),
                            dbg_elem<D>(want.data() + off).c_str(), alpha);
              if (first.empty()) first = buf;
              EXPECT_TRUE(false) << buf;
              ++cnt.mismatched;
              case_failed = true;
              break;
            }
          }
      if (cnt.mismatched >= 8) break;
    }
    if (cnt.mismatched >= 8) break;
  }
  finish(id, cnt, first);
}

// ---------------------------------------------------------------------------
// Div_Round (2D): full-precision pSrc2/pSrc1 divided by 2^sf and rounded with
// the requested NPP mode.
// ---------------------------------------------------------------------------
template <typename D>
inline D div_round_sfs(long long num, long long den, int mode, int sf) {
  if (den == 0) {
    if constexpr (std::is_same_v<D, Npp8u>) {
      // NPP's 8u path: 0/0 -> 254 for NEAR, 255 for the other modes;
      // non-zero numerators always saturate to 255.
      return num == 0 && mode == (int)NPP_RND_NEAR ? D(254) : D(255);
    } else {
      if (num == 0) return D(0);
      if constexpr (std::is_signed_v<D>) return num < 0 ? type_min<D>() : type_max<D>();
      else return type_max<D>();
    }
  }
  const long long d = den * (1LL << sf);
  long long q = num / d;
  const long long r = num % d;
  if (r != 0 && mode != (int)NPP_RND_ZERO) {
    const long long ar = r < 0 ? -r : r;
    const long long ad = d < 0 ? -d : d;
    const int sign = ((num < 0) ^ (d < 0)) ? -1 : 1;
    const bool up = mode == (int)NPP_RND_FINANCIAL
                        ? ((ar << 1) >= ad)
                        : ((ar << 1) > ad || ((ar << 1) == ad && (q & 1LL)));
    if (up) q += sign;
  }
  return (D)q;
}

template <typename S, int C, int A, bool Inplace, class Call>
void run_div_round2d(const char *id, Call call) {
  NppStreamContext ctx{};
  if (nppGetStreamContext(&ctx) != NPP_NO_ERROR) {
    GTEST_SKIP() << "[" << id << "] no stream context";
    return;
  }
  RunCounts cnt;
  std::string first;
  for (int mode : {0, 1, 2}) {
    for (int param : {0, 1, 2}) {
      for (const GenCase &c : make_gen_cases(param)) {
        Buf<S> a(c.w + c.ox, c.h + c.oy, C, c.pad, 0);
        Buf<S> b(c.w + c.ox, c.h + c.oy, C, c.pad, 0);
        Buf<S> d(c.w + c.ox, c.h + c.oy, C, c.pad, 0x5A);
        const uint64_t seed = case_seed(id, c.w, c.h, a.step, d.step, (int)c.regime, mode * 4 + param);
        fill_regime(a, C, seed, c.regime, 0);
        fill_regime(b, C, seed ^ 0x51633e2dull, c.regime, 1);
        make_divisor_nonzero(a);
        if constexpr (Inplace) fill_regime(d, C, seed ^ 0x2545f491ull, c.regime, 2);
        const std::vector<uint8_t> old_dst = d.host;
        a.dev.copyFromHost(a.host);
        b.dev.copyFromHost(b.host);
        d.dev.copyFromHost(d.host);
        npp_diff::clear_cuda_error();
        const S *ap = reinterpret_cast<const S *>(a.dptr() + off_bytes<S>(c.oy, c.ox, a.step, C));
        S *dp = reinterpret_cast<S *>(d.dptr() + off_bytes<S>(c.oy, c.ox, d.step, C));
        NppStatus st;
        if constexpr (Inplace) {
          st = call(ap, a.step, dp, d.step, NppiSize{c.w, c.h}, mode, param, ctx);
        } else {
          const S *bp = reinterpret_cast<const S *>(b.dptr() + off_bytes<S>(c.oy, c.ox, b.step, C));
          st = call(ap, a.step, bp, b.step, dp, d.step, NppiSize{c.w, c.h}, mode, param, ctx);
        }
        if (st != NPP_NO_ERROR) {
          ++cnt.skipped;
          if (st == NPP_NOT_IMPLEMENTED_ERROR) ++cnt.notimpl;
          continue;
        }
        const cudaError_t se = cudaDeviceSynchronize();
        ASSERT_EQ(se, cudaSuccess) << "[" << id << "] " << c.id()
                                   << " sync: " << cudaGetErrorString(se);
        d.dev.copyToHost(d.host);
        std::vector<uint8_t> want = old_dst;
        for (int y = 0; y < c.h; ++y)
          for (int x = 0; x < c.w; ++x)
            for (int ch = 0; ch < C; ++ch) {
              const size_t off = off_bytes<S>(c.oy + y, c.ox + x, d.step, C) + (size_t)ch * sizeof(S);
              if (ch >= A) continue;
              const long long den = (long long)a.hptr()[(size_t)(c.oy + y) * (a.step / sizeof(S)) +
                                                        (size_t)(c.ox + x) * C + ch];
              const long long num = Inplace
                  ? (long long)reinterpret_cast<const S *>(old_dst.data())[
                        (size_t)(c.oy + y) * (d.step / sizeof(S)) + (size_t)(c.ox + x) * C + ch]
                  : (long long)b.hptr()[(size_t)(c.oy + y) * (b.step / sizeof(S)) +
                                        (size_t)(c.ox + x) * C + ch];
              const S r = div_round_sfs<S>(num, den, mode, param);
              std::memcpy(want.data() + off, &r, sizeof(S));
            }
        ++cnt.executed;
        bool case_failed = false;
        for (int y = 0; y < c.h && !case_failed; ++y)
          for (int x = 0; x < c.w && !case_failed; ++x)
            for (int ch = 0; ch < C; ++ch) {
              const size_t off = off_bytes<S>(c.oy + y, c.ox + x, d.step, C) + (size_t)ch * sizeof(S);
              ++cnt.compared;
              if (!equal_elem<S>(d.host.data() + off, want.data() + off, 0, 0)) {
                char buf[360];
                std::snprintf(buf, sizeof(buf),
                              "[%s] %s mode=%d sf=%d mismatch at (x=%d,y=%d,ch=%d): got=%s want=%s",
                              id, c.id().c_str(), mode, param, x, y, ch,
                              dbg_elem<S>(d.host.data() + off).c_str(),
                              dbg_elem<S>(want.data() + off).c_str());
                if (first.empty()) first = buf;
                EXPECT_TRUE(false) << buf;
                ++cnt.mismatched;
                case_failed = true;
                break;
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
// MulScale family: trunc(a * b / maxval) with maxval = 255 (8u) or 65535 (16u).
// ---------------------------------------------------------------------------
template <typename S> inline S mulscale_value(double a, double b) {
  const long long prod = (long long)a * (long long)b;
  return (S)(prod / (long long)type_max<S>());
}

template <typename S, int C, int A, bool Inplace, class Call>
void run_mul_scale(const char *id, Call call) {
  NppStreamContext ctx{};
  if (nppGetStreamContext(&ctx) != NPP_NO_ERROR) {
    GTEST_SKIP() << "[" << id << "] no stream context";
    return;
  }
  RunCounts cnt;
  std::string first;
  for (const GenCase &c : make_gen_cases(0)) {
    Buf<S> a(c.w + c.ox, c.h + c.oy, C, c.pad, 0);
    Buf<S> b(c.w + c.ox, c.h + c.oy, C, c.pad, 0);
    Buf<S> d(c.w + c.ox, c.h + c.oy, C, c.pad, 0x5A);
    const uint64_t seed = case_seed(id, c.w, c.h, a.step, d.step, (int)c.regime, 0);
    fill_regime(a, C, seed, c.regime, 0);
    fill_regime(b, C, seed ^ 0x51633e2dull, c.regime, 1);
    if constexpr (Inplace) fill_regime(d, C, seed ^ 0x2545f491ull, c.regime, 2);
    const std::vector<uint8_t> old_dst = d.host;
    a.dev.copyFromHost(a.host);
    b.dev.copyFromHost(b.host);
    d.dev.copyFromHost(d.host);
    npp_diff::clear_cuda_error();
    const S *ap = reinterpret_cast<const S *>(a.dptr() + off_bytes<S>(c.oy, c.ox, a.step, C));
    const S *bp = reinterpret_cast<const S *>(b.dptr() + off_bytes<S>(c.oy, c.ox, b.step, C));
    S *dp = reinterpret_cast<S *>(d.dptr() + off_bytes<S>(c.oy, c.ox, d.step, C));
    NppStatus st;
    if constexpr (Inplace) {
      st = call(ap, a.step, dp, d.step, NppiSize{c.w, c.h}, ctx);
    } else {
      st = call(ap, a.step, bp, b.step, dp, d.step, NppiSize{c.w, c.h}, ctx);
    }
    if (st != NPP_NO_ERROR) {
      ++cnt.skipped;
      if (st == NPP_NOT_IMPLEMENTED_ERROR) ++cnt.notimpl;
      continue;
    }
    const cudaError_t se = cudaDeviceSynchronize();
    ASSERT_EQ(se, cudaSuccess) << "[" << id << "] " << c.id()
                               << " sync: " << cudaGetErrorString(se);
    d.dev.copyToHost(d.host);
    std::vector<uint8_t> want = old_dst;
    for (int y = 0; y < c.h; ++y)
      for (int x = 0; x < c.w; ++x)
        for (int ch = 0; ch < C; ++ch) {
          const size_t off = off_bytes<S>(c.oy + y, c.ox + x, d.step, C) + (size_t)ch * sizeof(S);
          if (ch >= A) continue;
          const double av = as_double(a.hptr()[(size_t)(c.oy + y) * (a.step / sizeof(S)) +
                                               (size_t)(c.ox + x) * C + ch]);
          const double bv = Inplace
              ? as_double(reinterpret_cast<const S *>(old_dst.data())[
                    (size_t)(c.oy + y) * (d.step / sizeof(S)) + (size_t)(c.ox + x) * C + ch])
              : as_double(b.hptr()[(size_t)(c.oy + y) * (b.step / sizeof(S)) +
                                   (size_t)(c.ox + x) * C + ch]);
          const S r = mulscale_value<S>(av, bv);
          std::memcpy(want.data() + off, &r, sizeof(S));
        }
    ++cnt.executed;
    bool case_failed = false;
    for (int y = 0; y < c.h && !case_failed; ++y)
      for (int x = 0; x < c.w && !case_failed; ++x)
        for (int ch = 0; ch < C; ++ch) {
          const size_t off = off_bytes<S>(c.oy + y, c.ox + x, d.step, C) + (size_t)ch * sizeof(S);
          ++cnt.compared;
          if (!equal_elem<S>(d.host.data() + off, want.data() + off, 0, 0)) {
            char buf[360];
            std::snprintf(buf, sizeof(buf),
                          "[%s] %s mismatch at (x=%d,y=%d,ch=%d): got=%s want=%s", id,
                          c.id().c_str(), x, y, ch, dbg_elem<S>(d.host.data() + off).c_str(),
                          dbg_elem<S>(want.data() + off).c_str());
            if (first.empty()) first = buf;
            EXPECT_TRUE(false) << buf;
            ++cnt.mismatched;
            case_failed = true;
            break;
          }
        }
    if (cnt.mismatched >= 8) break;
  }
  finish(id, cnt, first);
}

template <typename S, int C, int A, bool Inplace, bool DeviceC, class Call>
void run_mul_cscale(const char *id, Call call) {
  NppStreamContext ctx{};
  if (nppGetStreamContext(&ctx) != NPP_NO_ERROR) {
    GTEST_SKIP() << "[" << id << "] no stream context";
    return;
  }
  constexpr int NC = (C == 1 ? 1 : A);
  RunCounts cnt;
  std::string first;
  for (const GenCase &c : make_gen_cases(0)) {
    Buf<S> s(c.w + c.ox, c.h + c.oy, C, c.pad, 0);
    Buf<S> d(c.w + c.ox, c.h + c.oy, C, c.pad, 0x5A);
    const uint64_t seed = case_seed(id, c.w, c.h, s.step, d.step, (int)c.regime, 0);
    fill_regime(s, C, seed, c.regime, 0);
    if constexpr (Inplace) fill_regime(d, C, seed ^ 0x2545f491ull, c.regime, 2);
    S cval[NC];
    for (int ch = 0; ch < NC; ++ch) cval[ch] = make_const<S>(seed, ch);
    std::vector<S> chost(cval, cval + NC);
    DeviceMemory<S> cdev((size_t)NC);
    cdev.copyFromHost(chost);
    const std::vector<uint8_t> old_dst = d.host;
    s.dev.copyFromHost(s.host);
    d.dev.copyFromHost(d.host);
    npp_diff::clear_cuda_error();
    const S *sp = reinterpret_cast<const S *>(s.dptr() + off_bytes<S>(c.oy, c.ox, s.step, C));
    S *dp = reinterpret_cast<S *>(d.dptr() + off_bytes<S>(c.oy, c.ox, d.step, C));
    NppStatus st;
    if constexpr (DeviceC) {
      if constexpr (Inplace) st = call(cdev.get(), dp, d.step, NppiSize{c.w, c.h}, ctx);
      else st = call(sp, s.step, cdev.get(), dp, d.step, NppiSize{c.w, c.h}, ctx);
    } else {
      if constexpr (Inplace) st = call(chost.data(), dp, d.step, NppiSize{c.w, c.h}, ctx);
      else st = call(sp, s.step, chost.data(), dp, d.step, NppiSize{c.w, c.h}, ctx);
    }
    if (st != NPP_NO_ERROR) {
      ++cnt.skipped;
      if (st == NPP_NOT_IMPLEMENTED_ERROR) ++cnt.notimpl;
      continue;
    }
    const cudaError_t se = cudaDeviceSynchronize();
    ASSERT_EQ(se, cudaSuccess) << "[" << id << "] " << c.id()
                               << " sync: " << cudaGetErrorString(se);
    d.dev.copyToHost(d.host);
    std::vector<uint8_t> want = old_dst;
    for (int y = 0; y < c.h; ++y)
      for (int x = 0; x < c.w; ++x)
        for (int ch = 0; ch < C; ++ch) {
          const size_t off = off_bytes<S>(c.oy + y, c.ox + x, d.step, C) + (size_t)ch * sizeof(S);
          if (ch >= A) continue;
          const S av = Inplace
              ? reinterpret_cast<const S *>(old_dst.data())[
                    (size_t)(c.oy + y) * (d.step / sizeof(S)) + (size_t)(c.ox + x) * C + ch]
              : s.hptr()[(size_t)(c.oy + y) * (s.step / sizeof(S)) + (size_t)(c.ox + x) * C + ch];
          const S r = mulscale_value<S>((double)av, (double)cval[C == 1 ? 0 : ch]);
          std::memcpy(want.data() + off, &r, sizeof(S));
        }
    ++cnt.executed;
    bool case_failed = false;
    for (int y = 0; y < c.h && !case_failed; ++y)
      for (int x = 0; x < c.w && !case_failed; ++x)
        for (int ch = 0; ch < C; ++ch) {
          const size_t off = off_bytes<S>(c.oy + y, c.ox + x, d.step, C) + (size_t)ch * sizeof(S);
          ++cnt.compared;
          if (!equal_elem<S>(d.host.data() + off, want.data() + off, 0, 0)) {
            char buf[360];
            std::snprintf(buf, sizeof(buf),
                          "[%s] %s mismatch at (x=%d,y=%d,ch=%d): got=%s want=%s", id,
                          c.id().c_str(), x, y, ch, dbg_elem<S>(d.host.data() + off).c_str(),
                          dbg_elem<S>(want.data() + off).c_str());
            if (first.empty()) first = buf;
            EXPECT_TRUE(false) << buf;
            ++cnt.mismatched;
            case_failed = true;
            break;
          }
        }
    if (cnt.mismatched >= 8) break;
  }
  finish(id, cnt, first);
}


// ---------------------------------------------------------------------------
// Planar <-> packed channel copies (C3P3R / P3C3R / C4P4R / P4C4R).
// ---------------------------------------------------------------------------
template <typename S, int CH, bool ToPlanar, class Call>
void run_copy_planar(const char *id, Call call) {
  NppStreamContext ctx{};
  if (nppGetStreamContext(&ctx) != NPP_NO_ERROR) {
    GTEST_SKIP() << "[" << id << "] no stream context";
    return;
  }
  RunCounts cnt;
  std::string first;
  for (const GenCase &c : make_gen_cases(0)) {
    const int w = c.w, h = c.h;
    Buf<S> packed(w + c.pad / (int)sizeof(S) + 1, h, CH, c.pad, 0x5A);
    std::vector<Buf<S> *> planes;
    planes.reserve(CH);
    for (int p = 0; p < CH; ++p) planes.push_back(new Buf<S>(w + 1, h, 1, 0, 0x5A));
    const uint64_t seed = case_seed(id, w, h, packed.step, 0, (int)c.regime, 0);
    if constexpr (ToPlanar) {
      fill_regime(packed, CH, seed, c.regime, 0);
    } else {
      for (int p = 0; p < CH; ++p) fill_regime(*planes[(size_t)p], 1, seed + (uint64_t)p, c.regime, p);
    }
    const std::vector<uint8_t> packed_in = packed.host;
    std::vector<std::vector<uint8_t>> planes_in;
    for (int p = 0; p < CH; ++p) planes_in.push_back(planes[(size_t)p]->host);
    packed.dev.copyFromHost(packed.host);
    for (auto *b : planes) b->dev.copyFromHost(b->host);
    std::vector<const S *> host_ptrs;
    std::vector<S *> host_wptrs;
    for (auto *b : planes) host_wptrs.push_back(reinterpret_cast<S *>(b->dptr()));
    for (auto *b : planes) host_ptrs.push_back(reinterpret_cast<const S *>(b->dptr()));
    const int plane_step = (int)planes[0]->step;
    npp_diff::clear_cuda_error();
    NppStatus st;
    const S *pp = reinterpret_cast<const S *>(packed.dptr());
    S *pd = reinterpret_cast<S *>(packed.dptr());
    if constexpr (ToPlanar) {
      st = call(pp, (int)packed.step, host_wptrs.data(), plane_step, NppiSize{w, h}, ctx);
    } else {
      st = call(pd, (int)packed.step, host_ptrs.data(), plane_step, NppiSize{w, h}, ctx);
    }
    if (st != NPP_NO_ERROR) {
      ++cnt.skipped;
      if (st == NPP_NOT_IMPLEMENTED_ERROR) ++cnt.notimpl;
      for (auto *b : planes) delete b;
      continue;
    }
    const cudaError_t se = cudaDeviceSynchronize();
    ASSERT_EQ(se, cudaSuccess) << "[" << id << "] " << c.id() << " sync: " << cudaGetErrorString(se);
    packed.dev.copyToHost(packed.host);
    for (auto *b : planes) b->dev.copyToHost(b->host);
    ++cnt.executed;
    bool case_failed = false;
    for (int y = 0; y < h && !case_failed; ++y)
      for (int x = 0; x < w && !case_failed; ++x)
        for (int ch = 0; ch < CH; ++ch) {
          ++cnt.compared;
          if constexpr (ToPlanar) {
            const size_t src_off = (size_t)y * packed.step + ((size_t)x * CH + ch) * sizeof(S);
            const size_t dst_off = (size_t)y * planes[(size_t)ch]->step + (size_t)x * sizeof(S);
            if (!equal_elem<S>(planes[(size_t)ch]->host.data() + dst_off,
                               packed_in.data() + src_off, 0, 0)) {
              char b[320];
              std::snprintf(b, sizeof(b), "[%s] %s (%d,%d,ch=%d) plane mismatch got=%s want=%s", id,
                            c.id().c_str(), x, y, ch,
                            dbg_elem<S>(planes[(size_t)ch]->host.data() + dst_off).c_str(),
                            dbg_elem<S>(packed_in.data() + src_off).c_str());
              if (first.empty()) first = b;
              EXPECT_TRUE(false) << b;
              ++cnt.mismatched;
              case_failed = true;
              break;
            }
          } else {
            const size_t dst_off = (size_t)y * packed.step + ((size_t)x * CH + ch) * sizeof(S);
            const size_t src_off = (size_t)y * planes[(size_t)ch]->step + (size_t)x * sizeof(S);
            if (!equal_elem<S>(packed.host.data() + dst_off,
                               planes_in[(size_t)ch].data() + src_off, 0, 0)) {
              char b[320];
              std::snprintf(b, sizeof(b), "[%s] %s (%d,%d,ch=%d) packed mismatch got=%s want=%s", id,
                            c.id().c_str(), x, y, ch,
                            dbg_elem<S>(packed.host.data() + dst_off).c_str(),
                            dbg_elem<S>(planes_in[(size_t)ch].data() + src_off).c_str());
              if (first.empty()) first = b;
              EXPECT_TRUE(false) << b;
              ++cnt.mismatched;
              case_failed = true;
              break;
            }
          }
        }
    for (auto *b : planes) delete b;
    if (cnt.mismatched >= 8) break;
  }
  finish(id, cnt, first);
}

// ---------------------------------------------------------------------------
// nppiScale: dst = aa + bb*src with the NPP 13.4 per-conversion affine
// constants.  Integer destinations truncate toward zero and saturate;
// 8u->32f clamps to [nMin, nMax]; AC4 preserves alpha.
// ---------------------------------------------------------------------------
template <typename D> inline D scale_store(double out, float cl_min, float cl_max, bool do_clamp) {
  if constexpr (std::is_floating_point_v<D>) {
    double o = out;
    if (do_clamp) {
      if (o < (double)cl_min) o = (double)cl_min;
      if (o > (double)cl_max) o = (double)cl_max;
    }
    return (D)o;
  } else {
    if (std::isnan(out)) return (D)0;
    const long long mn = std::is_signed_v<D> ? (long long)std::numeric_limits<D>::lowest() : 0;
    const long long mx = (long long)std::numeric_limits<D>::max();
    if (out >= (double)mx) return (D)mx;
    if (out <= (double)mn) return (D)mn;
    return (D)(long long)out;
  }
}

template <typename S, typename D, int C, int A, bool MinMax, bool HasHint, class Call>
void run_scale(const char *id, Call call) {
  NppStreamContext ctx{};
  if (nppGetStreamContext(&ctx) != NPP_NO_ERROR) {
    GTEST_SKIP() << "[" << id << "] no stream context";
    return;
  }
  RunCounts cnt;
  std::string first;
  for (const GenCase &c : make_gen_cases(0)) {
    Buf<S> s(c.w + c.ox, c.h + c.oy, C, c.pad, 0);
    Buf<D> d(c.w + c.ox, c.h + c.oy, C, c.pad, 0x5A);
    const uint64_t seed = case_seed(id, c.w, c.h, s.step, d.step, (int)c.regime, 0);
    fill_regime(s, C, seed, c.regime, 0);
    const std::vector<uint8_t> old_dst = d.host;
    s.dev.copyFromHost(s.host);
    d.dev.copyFromHost(d.host);
    Rng rng(seed ^ 0x9e3779b97f4a7c15ull);
    float mn = 0.0f, mx = 0.0f;
    if constexpr (MinMax) {
      const int pick = (int)(rng.next() % 3);
      if (pick == 0) { mn = -1.0f; mx = 1.0f; }
      else if (pick == 1) { mn = 0.0f; mx = 255.0f; }
      else { mn = -2.5f; mx = 7.5f; }
    }
    const int hint = HasHint ? (int)(rng.next() % 2) : 0;
    npp_diff::clear_cuda_error();
    NppStatus st = call(reinterpret_cast<const S *>(s.dptr() + off_bytes<S>(c.oy, c.ox, s.step, C)),
                        s.step,
                        reinterpret_cast<D *>(d.dptr() + off_bytes<D>(c.oy, c.ox, d.step, C)),
                        d.step, NppiSize{c.w, c.h}, mn, mx, hint, ctx);
    if (st != NPP_NO_ERROR) {
      ++cnt.skipped;
      if (st == NPP_NOT_IMPLEMENTED_ERROR) ++cnt.notimpl;
      continue;
    }
    const cudaError_t se = cudaDeviceSynchronize();
    ASSERT_EQ(se, cudaSuccess) << "[" << id << "] " << c.id() << " sync: " << cudaGetErrorString(se);
    d.dev.copyToHost(d.host);
    std::vector<uint8_t> want = old_dst;
    double aa = 0.0, bb = 1.0;
    bool clamp = false;
    if constexpr (MinMax) {
      if constexpr (std::is_floating_point_v<D>) {
        aa = (double)mn;
        bb = ((double)mx - (double)mn) / 255.0;
        clamp = true;
      } else {
        aa = -(255.0 / ((double)mx - (double)mn)) * (double)mn;
        bb = 255.0 / ((double)mx - (double)mn);
      }
    } else if constexpr (std::is_same_v<S, Npp8u> && std::is_same_v<D, Npp16u>) {
      aa = 0.0; bb = 65535.0 / 255.0;
    } else if constexpr (std::is_same_v<S, Npp8u> && std::is_same_v<D, Npp16s>) {
      aa = -32768.0; bb = 65535.0 / 255.0;
    } else if constexpr (std::is_same_v<S, Npp8u> && std::is_same_v<D, Npp32s>) {
      aa = -2147483648.0; bb = 4294967295.0 / 255.0;
    } else if constexpr (std::is_same_v<S, Npp16u> && std::is_same_v<D, Npp8u>) {
      aa = 0.0; bb = 255.0 / 65535.0;
    } else if constexpr (std::is_same_v<S, Npp16s> && std::is_same_v<D, Npp8u>) {
      aa = (255.0 / 65535.0) * 32768.0; bb = 255.0 / 65535.0;
    } else if constexpr (std::is_same_v<S, Npp32s> && std::is_same_v<D, Npp8u>) {
      aa = (255.0 / 4294967295.0) * 2147483648.0; bb = 255.0 / 4294967295.0;
    }
    for (int y = 0; y < c.h; ++y)
      for (int x = 0; x < c.w; ++x)
        for (int ch = 0; ch < C; ++ch) {
          const size_t off = off_bytes<D>(c.oy + y, c.ox + x, d.step, C) + (size_t)ch * sizeof(D);
          if (ch >= A) continue;
          const double v = as_double(s.hptr()[(size_t)(c.oy + y) * (s.step / sizeof(S)) +
                                             (size_t)(c.ox + x) * C + ch]);
          const D r = scale_store<D>(aa + bb * v, mn, mx, clamp);
          std::memcpy(want.data() + off, &r, sizeof(D));
        }
    ++cnt.executed;
    bool case_failed = false;
    for (int y = 0; y < c.h && !case_failed; ++y)
      for (int x = 0; x < c.w && !case_failed; ++x)
        for (int ch = 0; ch < C; ++ch) {
          const size_t off = off_bytes<D>(c.oy + y, c.ox + x, d.step, C) + (size_t)ch * sizeof(D);
          ++cnt.compared;
          const double rtol = std::is_same_v<D, Npp32f> ? 1e-6 : 0.0;
          const double atol = std::is_same_v<D, Npp32f> ? 1e-5 : 0.0;
          if (!equal_elem<D>(d.host.data() + off, want.data() + off, rtol, atol)) {
            char b[360];
            std::snprintf(b, sizeof(b), "[%s] %s (%d,%d,ch=%d) got=%s want=%s", id, c.id().c_str(),
                          x, y, ch, dbg_elem<D>(d.host.data() + off).c_str(),
                          dbg_elem<D>(want.data() + off).c_str());
            if (first.empty()) first = b;
            EXPECT_TRUE(false) << b;
            ++cnt.mismatched;
            case_failed = true;
            break;
          }
        }
    if (cnt.mismatched >= 8) break;
  }
  finish(id, cnt, first);
}

}  // namespace num

namespace num {

// ---------------------------------------------------------------------------
// Threshold family.  All entry points are normalized to the same call shape:
//   call(src, sstep, dst, dstep, roi, thr[], val[], thr2[], val2[], op, ctx)
// The generated lambda selects scalar vs per-channel arguments.  Modes mirror
// the NPP families (Threshold / Val / GT / LT / GTVal / LTVal / LTValGTVal).
// ---------------------------------------------------------------------------
enum class ThrKind { Threshold, ThresholdVal, GT, LT, GTVal, LTVal, LTValGTVal };

template <typename T, int C, int A, ThrKind K, class Call>
void run_threshold(const char *id, Call call, const std::vector<int> &params, bool per_channel,
                   bool inplace) {
  NppStreamContext ctx{};
  if (nppGetStreamContext(&ctx) != NPP_NO_ERROR) {
    GTEST_SKIP() << "[" << id << "] no stream context";
    return;
  }
  RunCounts cnt;
  std::string first;
  for (int param : params) {
    for (const GenCase &c : make_gen_cases(param)) {
      Buf<T> src(c.w + c.ox, c.h + c.oy, C, safe_pad<T>(c.pad), 0);
      Buf<T> dst(c.w + c.ox, c.h + c.oy, C, safe_pad<T>(c.pad), 0);
      const uint64_t seed = case_seed(id, c.w, c.h, src.step, dst.step, (int)c.regime, param);
      fill_regime(src, C, seed, c.regime, 0);
      if (inplace) fill_regime(dst, C, seed ^ 0x2545f491ull, c.regime, 1);
      const std::vector<uint8_t> old_src = src.host;
      const std::vector<uint8_t> old_dst = dst.host;
      src.dev.copyFromHost(src.host);
      dst.dev.copyFromHost(dst.host);

      T thr[C > 0 ? C : 1], val[C > 0 ? C : 1], thr2[C > 0 ? C : 1], val2[C > 0 ? C : 1];
      Rng rng(seed ^ 0x9e3779b97f4a7c15ull);
      for (int ch = 0; ch < C; ++ch) {
        const T v0 = src.hptr()[(size_t)(c.oy + (c.h / 2)) * (src.step / sizeof(T)) +
                                (size_t)(c.ox + (c.w / 2)) * C + ch];
        const T r1 = regime_value<T>(rng, c.regime, 11 + ch);
        const T r2 = regime_value<T>(rng, c.regime, 31 + ch);
        if (K == ThrKind::LTValGTVal) {
          const T lt = regime_value<T>(rng, c.regime, 51 + ch);
          const T gt = regime_value<T>(rng, c.regime, 71 + ch);
          // Keep the interval ordered for ordinary finite values; NaN/Inf
          // regimes simply make both comparisons false, which the reference
          // reproduces.
          thr[ch] = lt < gt ? lt : gt;
          thr2[ch] = lt < gt ? gt : lt;
        } else {
          thr[ch] = (c.w * c.h) % 3 == 0 ? v0 : r1;
          thr2[ch] = r2;
        }
        val[ch] = regime_value<T>(rng, c.regime, 91 + ch);
        val2[ch] = regime_value<T>(rng, c.regime, 111 + ch);
      }
      const T *tp = per_channel ? thr : &thr[0];
      const T *vp = per_channel ? val : &val[0];
      const T *t2p = per_channel ? thr2 : &thr2[0];
      const T *v2p = per_channel ? val2 : &val2[0];

      npp_diff::clear_cuda_error();
      NppStatus st = call(
          reinterpret_cast<const T *>(src.dptr() + off_bytes<T>(c.oy, c.ox, src.step, C)), src.step,
          reinterpret_cast<T *>(dst.dptr() + off_bytes<T>(c.oy, c.ox, dst.step, C)), dst.step,
          NppiSize{c.w, c.h}, tp, vp, t2p, v2p, param, ctx);
      if (st != NPP_NO_ERROR) {
        ++cnt.skipped;
        if (st == NPP_NOT_IMPLEMENTED_ERROR) ++cnt.notimpl;
        continue;
      }
      const cudaError_t se = cudaDeviceSynchronize();
      ASSERT_EQ(se, cudaSuccess) << "[" << id << "] " << c.id()
                                 << " sync: " << cudaGetErrorString(se);
      dst.dev.copyToHost(dst.host);

      const std::vector<uint8_t> &in = inplace ? old_dst : old_src;
      std::vector<uint8_t> want = dst.host;
      for (int y = 0; y < c.h; ++y)
        for (int x = 0; x < c.w; ++x)
          for (int ch = 0; ch < C; ++ch) {
            const size_t off = off_bytes<T>(c.oy + y, c.ox + x, dst.step, C) + (size_t)ch * sizeof(T);
            if (ch >= A) continue;
            const T v = reinterpret_cast<const T *>(in.data())[(size_t)(c.oy + y) * (src.step / sizeof(T)) +
                                                              (size_t)(c.ox + x) * C + ch];
            const T t = per_channel ? thr[ch] : thr[0];
            const T vv = per_channel ? val[ch] : val[0];
            T r = v;
            if (K == ThrKind::Threshold) {
              const bool cond = param == NPP_CMP_LESS ? (v < t) : (v > t);
              r = cond ? t : v;
            } else if (K == ThrKind::ThresholdVal) {
              const bool cond = param == NPP_CMP_LESS ? (v < t) : (v > t);
              r = cond ? vv : v;
            } else if (K == ThrKind::GT) {
              r = v > t ? t : v;
            } else if (K == ThrKind::LT) {
              r = v < t ? t : v;
            } else if (K == ThrKind::GTVal) {
              r = v > t ? vv : v;
            } else if (K == ThrKind::LTVal) {
              r = v < t ? vv : v;
            } else {
              const T tl = per_channel ? thr[ch] : thr[0];
              const T tg = per_channel ? thr2[ch] : thr2[0];
              const T vl = per_channel ? val[ch] : val[0];
              const T vg = per_channel ? val2[ch] : val2[0];
              r = v < tl ? vl : (v > tg ? vg : v);
            }
            std::memcpy(want.data() + off, &r, sizeof(T));
          }

      ++cnt.executed;
      bool case_failed = false;
      for (int y = 0; y < c.h && !case_failed; ++y)
        for (int x = 0; x < c.w && !case_failed; ++x)
          for (int ch = 0; ch < C; ++ch) {
            const size_t off = off_bytes<T>(c.oy + y, c.ox + x, dst.step, C) + (size_t)ch * sizeof(T);
            ++cnt.compared;
            if (!equal_elem<T>(dst.host.data() + off, want.data() + off, 0, 0)) {
              char buf[360];
              std::snprintf(buf, sizeof(buf),
                            "[%s] %s mismatch at (x=%d,y=%d,ch=%d): got=%s want=%s", id,
                            c.id().c_str(), x, y, ch, dbg_elem<T>(dst.host.data() + off).c_str(),
                            dbg_elem<T>(want.data() + off).c_str());
              if (first.empty()) first = buf;
              EXPECT_TRUE(false) << buf;
              ++cnt.mismatched;
              case_failed = true;
              break;
            }
          }
      if (cnt.mismatched >= 8) break;
    }
    if (cnt.mismatched >= 8) break;
  }
  finish(id, cnt, first);
}

// ---------------------------------------------------------------------------
// Compare: two images (or an image and constants) -> 8u_C1 mask.  A pixel is
// 0xFF only when every active channel satisfies the comparison.
// ---------------------------------------------------------------------------
template <typename T, int C, int A, class Call>
void run_compare(const char *id, Call call, const std::vector<int> &params) {
  NppStreamContext ctx{};
  if (nppGetStreamContext(&ctx) != NPP_NO_ERROR) {
    GTEST_SKIP() << "[" << id << "] no stream context";
    return;
  }
  RunCounts cnt;
  std::string first;
  for (int param : params) {
    for (const GenCase &c : make_gen_cases(param)) {
      Buf<T> a(c.w + c.ox, c.h + c.oy, C, safe_pad<T>(c.pad), 0);
      Buf<T> b(c.w + c.ox, c.h + c.oy, C, safe_pad<T>(c.pad), 0);
      Buf<Npp8u> d(c.w + c.ox, c.h + c.oy, 1, safe_pad<T>(c.pad), 0x5A);
      const uint64_t seed = case_seed(id, c.w, c.h, a.step, d.step, (int)c.regime, param);
      fill_regime(a, C, seed, c.regime, 0, false, true, true);
      fill_regime(b, C, seed ^ 0x51633e2dull, c.regime, 1, false, true, true);
      if ((c.w * c.h) % 4 == 0) std::memcpy(b.host.data(), a.host.data(), a.host.size());
      a.dev.copyFromHost(a.host);
      b.dev.copyFromHost(b.host);
      d.dev.copyFromHost(d.host);
      npp_diff::clear_cuda_error();
      NppStatus st = call(
          reinterpret_cast<const T *>(a.dptr() + off_bytes<T>(c.oy, c.ox, a.step, C)), a.step,
          reinterpret_cast<const T *>(b.dptr() + off_bytes<T>(c.oy, c.ox, b.step, C)), b.step,
          d.dptr() + off_bytes<Npp8u>(c.oy, c.ox, d.step, 1), d.step, NppiSize{c.w, c.h}, param,
          ctx);
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
      for (int y = 0; y < c.h; ++y)
        for (int x = 0; x < c.w; ++x) {
          bool all = true;
          for (int ch = 0; ch < A; ++ch) {
            const T va = a.hptr()[(size_t)(c.oy + y) * (a.step / sizeof(T)) + (size_t)(c.ox + x) * C + ch];
            const T vb = b.hptr()[(size_t)(c.oy + y) * (b.step / sizeof(T)) + (size_t)(c.ox + x) * C + ch];
            bool cnd = false;
            switch (param) {
              case NPP_CMP_LESS: cnd = va < vb; break;
              // NPP builds the non-strict float comparisons as negations,
              // which makes NaN compare true; mirror that exactly.
              case NPP_CMP_LESS_EQ: cnd = !(va > vb); break;
              case NPP_CMP_EQ: cnd = va == vb; break;
              case NPP_CMP_GREATER_EQ: cnd = !(va < vb); break;
              default: cnd = va > vb; break;
            }
            all = all && cnd;
          }
          want[off_bytes<Npp8u>(c.oy + y, c.ox + x, d.step, 1)] = all ? 0xFF : 0x00;
        }
      ++cnt.executed;
      bool case_failed = false;
      for (int y = 0; y < c.h && !case_failed; ++y)
        for (int x = 0; x < c.w && !case_failed; ++x) {
          const size_t off = off_bytes<Npp8u>(c.oy + y, c.ox + x, d.step, 1);
          ++cnt.compared;
          if (d.host[off] != want[off]) {
            char buf[300];
            std::snprintf(buf, sizeof(buf),
                          "[%s] %s mismatch at (x=%d,y=%d): got=%u want=%u", id, c.id().c_str(), x,
                          y, (unsigned)d.host[off], (unsigned)want[off]);
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

template <typename T, int C, int A, class Call>
void run_compare_const(const char *id, Call call, const std::vector<int> &params,
                       bool per_channel) {
  NppStreamContext ctx{};
  if (nppGetStreamContext(&ctx) != NPP_NO_ERROR) {
    GTEST_SKIP() << "[" << id << "] no stream context";
    return;
  }
  RunCounts cnt;
  std::string first;
  for (int param : params) {
    for (const GenCase &c : make_gen_cases(param)) {
      Buf<T> a(c.w + c.ox, c.h + c.oy, C, safe_pad<T>(c.pad), 0);
      Buf<Npp8u> d(c.w + c.ox, c.h + c.oy, 1, safe_pad<T>(c.pad), 0x5A);
      const uint64_t seed = case_seed(id, c.w, c.h, a.step, d.step, (int)c.regime, param);
      fill_regime(a, C, seed, c.regime, 0, false, true, true);
      T cv[C > 0 ? C : 1];
      Rng rng(seed ^ 0x9e3779b97f4a7c15ull);
      for (int ch = 0; ch < C; ++ch) cv[ch] = no_nan_value(regime_value<T>(rng, c.regime, 131 + ch));
      a.dev.copyFromHost(a.host);
      d.dev.copyFromHost(d.host);
      npp_diff::clear_cuda_error();
      NppStatus st = call(
          reinterpret_cast<const T *>(a.dptr() + off_bytes<T>(c.oy, c.ox, a.step, C)), a.step,
          per_channel ? cv : &cv[0],
          d.dptr() + off_bytes<Npp8u>(c.oy, c.ox, d.step, 1), d.step, NppiSize{c.w, c.h}, param,
          ctx);
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
      for (int y = 0; y < c.h; ++y)
        for (int x = 0; x < c.w; ++x) {
          bool all = true;
          for (int ch = 0; ch < A; ++ch) {
            const T va = a.hptr()[(size_t)(c.oy + y) * (a.step / sizeof(T)) + (size_t)(c.ox + x) * C + ch];
            const T vb = per_channel ? cv[ch] : cv[0];
            bool cnd = false;
            switch (param) {
              case NPP_CMP_LESS: cnd = va < vb; break;
              // NPP builds the non-strict float comparisons as negations,
              // which makes NaN compare true; mirror that exactly.
              // Constant comparison uses the direct form: NaN is false.
              case NPP_CMP_LESS_EQ: cnd = va <= vb; break;
              case NPP_CMP_EQ: cnd = va == vb; break;
              case NPP_CMP_GREATER_EQ: cnd = va >= vb; break;
              default: cnd = va > vb; break;
            }
            all = all && cnd;
          }
          want[off_bytes<Npp8u>(c.oy + y, c.ox + x, d.step, 1)] = all ? 0xFF : 0x00;
        }
      ++cnt.executed;
      bool case_failed = false;
      for (int y = 0; y < c.h && !case_failed; ++y)
        for (int x = 0; x < c.w && !case_failed; ++x) {
          const size_t off = off_bytes<Npp8u>(c.oy + y, c.ox + x, d.step, 1);
          ++cnt.compared;
          if (d.host[off] != want[off]) {
            char buf[300];
            std::snprintf(buf, sizeof(buf),
                          "[%s] %s mismatch at (x=%d,y=%d): got=%u want=%u", id, c.id().c_str(), x,
                          y, (unsigned)d.host[off], (unsigned)want[off]);
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
// CompareEqualEps: like Compare but the condition is |a-b| <= epsilon.  The
// epsilon is a 32f scalar, so the mask is again ANDed over active channels.
// ---------------------------------------------------------------------------
template <typename T, int C, int A, class Call>
void run_compare_eps(const char *id, Call call, const std::vector<double> &eps_params) {
  NppStreamContext ctx{};
  if (nppGetStreamContext(&ctx) != NPP_NO_ERROR) {
    GTEST_SKIP() << "[" << id << "] no stream context";
    return;
  }
  RunCounts cnt;
  std::string first;
  for (double eps : eps_params) {
    for (const GenCase &c : make_gen_cases(0)) {
      Buf<T> a(c.w + c.ox, c.h + c.oy, C, safe_pad<T>(c.pad), 0);
      Buf<T> b(c.w + c.ox, c.h + c.oy, C, safe_pad<T>(c.pad), 0);
      Buf<Npp8u> d(c.w + c.ox, c.h + c.oy, 1, safe_pad<T>(c.pad), 0x5A);
      const uint64_t seed = case_seed(id, c.w, c.h, a.step, d.step, (int)c.regime, 0);
      fill_regime(a, C, seed, c.regime, 0, false, true, true);
      fill_regime(b, C, seed ^ 0x51633e2dull, c.regime, 1, false, true, true);
      if ((c.w * c.h) % 4 == 0) std::memcpy(b.host.data(), a.host.data(), a.host.size());
      a.dev.copyFromHost(a.host);
      b.dev.copyFromHost(b.host);
      d.dev.copyFromHost(d.host);
      npp_diff::clear_cuda_error();
      NppStatus st = call(
          reinterpret_cast<const T *>(a.dptr() + off_bytes<T>(c.oy, c.ox, a.step, C)), a.step,
          reinterpret_cast<const T *>(b.dptr() + off_bytes<T>(c.oy, c.ox, b.step, C)), b.step,
          d.dptr() + off_bytes<Npp8u>(c.oy, c.ox, d.step, 1), d.step, NppiSize{c.w, c.h},
          (Npp32f)eps, ctx);
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
      for (int y = 0; y < c.h; ++y)
        for (int x = 0; x < c.w; ++x) {
          bool all = true;
          for (int ch = 0; ch < A; ++ch) {
            const T va = a.hptr()[(size_t)(c.oy + y) * (a.step / sizeof(T)) + (size_t)(c.ox + x) * C + ch];
            const T vb = b.hptr()[(size_t)(c.oy + y) * (b.step / sizeof(T)) + (size_t)(c.ox + x) * C + ch];
            all = all && (std::fabs((double)va - (double)vb) <= eps);
          }
          want[off_bytes<Npp8u>(c.oy + y, c.ox + x, d.step, 1)] = all ? 0xFF : 0x00;
        }
      ++cnt.executed;
      bool case_failed = false;
      for (int y = 0; y < c.h && !case_failed; ++y)
        for (int x = 0; x < c.w && !case_failed; ++x) {
          const size_t off = off_bytes<Npp8u>(c.oy + y, c.ox + x, d.step, 1);
          ++cnt.compared;
          if (d.host[off] != want[off]) {
            char buf[300];
            std::snprintf(buf, sizeof(buf),
                          "[%s] %s eps=%g mismatch at (x=%d,y=%d): got=%u want=%u", id,
                          c.id().c_str(), eps, x, y, (unsigned)d.host[off],
                          (unsigned)want[off]);
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

template <typename T, int C, int A, class Call>
void run_compare_eps_const(const char *id, Call call, const std::vector<double> &eps_params,
                           bool per_channel) {
  NppStreamContext ctx{};
  if (nppGetStreamContext(&ctx) != NPP_NO_ERROR) {
    GTEST_SKIP() << "[" << id << "] no stream context";
    return;
  }
  RunCounts cnt;
  std::string first;
  for (double eps : eps_params) {
    for (const GenCase &c : make_gen_cases(0)) {
      Buf<T> a(c.w + c.ox, c.h + c.oy, C, safe_pad<T>(c.pad), 0);
      Buf<Npp8u> d(c.w + c.ox, c.h + c.oy, 1, safe_pad<T>(c.pad), 0x5A);
      const uint64_t seed = case_seed(id, c.w, c.h, a.step, d.step, (int)c.regime, 0);
      fill_regime(a, C, seed, c.regime, 0, false, true, true);
      T cv[C > 0 ? C : 1];
      Rng rng(seed ^ 0x9e3779b97f4a7c15ull);
      for (int ch = 0; ch < C; ++ch) cv[ch] = no_nan_value(regime_value<T>(rng, c.regime, 131 + ch));
      a.dev.copyFromHost(a.host);
      d.dev.copyFromHost(d.host);
      npp_diff::clear_cuda_error();
      NppStatus st = call(
          reinterpret_cast<const T *>(a.dptr() + off_bytes<T>(c.oy, c.ox, a.step, C)), a.step,
          per_channel ? cv : &cv[0],
          d.dptr() + off_bytes<Npp8u>(c.oy, c.ox, d.step, 1), d.step, NppiSize{c.w, c.h},
          (Npp32f)eps, ctx);
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
      for (int y = 0; y < c.h; ++y)
        for (int x = 0; x < c.w; ++x) {
          bool all = true;
          for (int ch = 0; ch < A; ++ch) {
            const T va = a.hptr()[(size_t)(c.oy + y) * (a.step / sizeof(T)) + (size_t)(c.ox + x) * C + ch];
            const T vb = per_channel ? cv[ch] : cv[0];
            all = all && (std::fabs((double)va - (double)vb) <= eps);
          }
          want[off_bytes<Npp8u>(c.oy + y, c.ox + x, d.step, 1)] = all ? 0xFF : 0x00;
        }
      ++cnt.executed;
      bool case_failed = false;
      for (int y = 0; y < c.h && !case_failed; ++y)
        for (int x = 0; x < c.w && !case_failed; ++x) {
          const size_t off = off_bytes<Npp8u>(c.oy + y, c.ox + x, d.step, 1);
          ++cnt.compared;
          if (d.host[off] != want[off]) {
            char buf[300];
            std::snprintf(buf, sizeof(buf),
                          "[%s] %s eps=%g mismatch at (x=%d,y=%d): got=%u want=%u", id,
                          c.id().c_str(), eps, x, y, (unsigned)d.host[off],
                          (unsigned)want[off]);
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
// Set family: fill an ROI with a constant (scalar or per-channel), optionally
// only where an 8u mask pixel is non-zero.  AC4 variants write the first three
// channels and preserve alpha; C3CR/C4CR write every channel with a scalar.
// ---------------------------------------------------------------------------
template <typename T, typename VT> inline T convert_value(VT v) {
  if constexpr (std::is_same_v<T, Npp16f>) return half_round((float)as_double(v));
  else return static_cast<T>(v);
}

template <typename T, typename VT, int C, int A, class Call>
void run_set(const char *id, Call call, bool per_channel, bool fill_all, bool masked) {
  NppStreamContext ctx{};
  if (nppGetStreamContext(&ctx) != NPP_NO_ERROR) {
    GTEST_SKIP() << "[" << id << "] no stream context";
    return;
  }
  RunCounts cnt;
  std::string first;
  for (const GenCase &c : make_gen_cases(0)) {
    const int pad = safe_pad<T>(c.pad);
    Buf<T> d(c.w + c.ox, c.h + c.oy, C, pad, 0x5A);
    Buf<Npp8u> m(c.w + c.ox, c.h + c.oy, 1, pad, 0xFF);
    const uint64_t seed = case_seed(id, c.w, c.h, d.step, m.step, (int)c.regime, 0);
    Rng rng(seed);
    VT vals[C > 0 ? C : 1];
    for (int ch = 0; ch < C; ++ch) vals[ch] = regime_value<VT>(rng, c.regime, 3 + ch);
    const VT scalar = regime_value<VT>(rng, c.regime, 1);
    if (masked) {
      for (int y = 0; y < m.bh; ++y)
        for (int x = 0; x < m.bw; ++x) m.host[(size_t)y * m.step + x] = ((x * 7 + y * 13) % 5 == 0) ? 0 : 0xFF;
    }
    d.dev.copyFromHost(d.host);
    if (masked) m.dev.copyFromHost(m.host);
    npp_diff::clear_cuda_error();
    NppStatus st = call(vals, scalar,
                        reinterpret_cast<T *>(d.dptr() + off_bytes<T>(c.oy, c.ox, d.step, C)), d.step,
                        NppiSize{c.w, c.h},
                        masked ? m.dptr() + off_bytes<Npp8u>(c.oy, c.ox, m.step, 1) : nullptr,
                        masked ? m.step : 0, ctx);
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
    for (int y = 0; y < c.h; ++y)
      for (int x = 0; x < c.w; ++x) {
        if (masked && m.host[(size_t)(c.oy + y) * m.step + (c.ox + x)] == 0) continue;
        // C3CR/C4CR set only the selected (first) channel; plain Set writes
        // the active channels and leaves AC4 alpha untouched.
        const int channels = fill_all ? 1 : A;
        for (int ch = 0; ch < channels; ++ch) {
          const size_t off = off_bytes<T>(c.oy + y, c.ox + x, d.step, C) + (size_t)ch * sizeof(T);
          const VT v = (fill_all ? scalar : (per_channel ? vals[ch] : scalar));
          const T r = convert_value<T>(v);
          std::memcpy(want.data() + off, &r, sizeof(T));
        }
      }
    ++cnt.executed;
    bool case_failed = false;
    for (int y = 0; y < c.h && !case_failed; ++y)
      for (int x = 0; x < c.w && !case_failed; ++x)
        for (int ch = 0; ch < C; ++ch) {
          const size_t off = off_bytes<T>(c.oy + y, c.ox + x, d.step, C) + (size_t)ch * sizeof(T);
          ++cnt.compared;
          if (!equal_elem<T>(d.host.data() + off, want.data() + off, 0, 0)) {
            char buf[320];
            std::snprintf(buf, sizeof(buf),
                          "[%s] %s mismatch at (x=%d,y=%d,ch=%d): got=%s want=%s", id,
                          c.id().c_str(), x, y, ch, dbg_elem<T>(d.host.data() + off).c_str(),
                          dbg_elem<T>(want.data() + off).c_str());
            if (first.empty()) first = buf;
            EXPECT_TRUE(false) << buf;
            ++cnt.mismatched;
            case_failed = true;
            break;
          }
        }
    if (cnt.mismatched >= 8) break;
  }
  finish(id, cnt, first);
}


// ---------------------------------------------------------------------------
// Copy family.  The source and destination may have different packed channel
// counts (C1C3 / C3C1 ...); NCOPY channels are copied from the source pixel
// to the destination pixel, the remaining destination channels are preserved
// (AC4 alpha, C3CR selected-channel, C1Cx insertion).  Masked variants copy
// only where the 8u mask pixel is non-zero.
// ---------------------------------------------------------------------------
template <typename T, int SCH, int DCH, int NCOPY, class Call>
void run_copy(const char *id, Call call, bool masked) {
  NppStreamContext ctx{};
  if (nppGetStreamContext(&ctx) != NPP_NO_ERROR) {
    GTEST_SKIP() << "[" << id << "] no stream context";
    return;
  }
  RunCounts cnt;
  std::string first;
  for (const GenCase &c : make_gen_cases(0)) {
    Buf<T> s(c.w + c.ox, c.h + c.oy, SCH, safe_pad<T>(c.pad), 0xCD);
    Buf<T> d(c.w + c.ox, c.h + c.oy, DCH, safe_pad<T>(c.pad), 0x5A);
    Buf<Npp8u> m(c.w + c.ox, c.h + c.oy, 1, safe_pad<T>(c.pad), 0xFF);
    const uint64_t seed = case_seed(id, c.w, c.h, s.step, d.step, (int)c.regime, 0);
    fill_regime(s, SCH, seed, c.regime, 0);
    if (masked)
      for (int y = 0; y < m.bh; ++y)
        for (int x = 0; x < m.bw; ++x)
          m.host[(size_t)y * m.step + x] = ((x * 11 + y * 7) % 4 == 0) ? 0 : 0xFF;
    s.dev.copyFromHost(s.host);
    d.dev.copyFromHost(d.host);
    if (masked) m.dev.copyFromHost(m.host);
    npp_diff::clear_cuda_error();
    NppStatus st = call(
        reinterpret_cast<const T *>(s.dptr() + off_bytes<T>(c.oy, c.ox, s.step, SCH)), s.step,
        reinterpret_cast<T *>(d.dptr() + off_bytes<T>(c.oy, c.ox, d.step, DCH)), d.step,
        NppiSize{c.w, c.h}, masked ? m.dptr() + off_bytes<Npp8u>(c.oy, c.ox, m.step, 1) : nullptr,
        masked ? m.step : 0, ctx);
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
    for (int y = 0; y < c.h; ++y)
      for (int x = 0; x < c.w; ++x) {
        if (masked && m.host[(size_t)(c.oy + y) * m.step + (c.ox + x)] == 0) continue;
        for (int ch = 0; ch < NCOPY; ++ch) {
          const T v = s.hptr()[(size_t)(c.oy + y) * (s.step / sizeof(T)) + (size_t)(c.ox + x) * SCH + ch];
          const size_t off = off_bytes<T>(c.oy + y, c.ox + x, d.step, DCH) + (size_t)ch * sizeof(T);
          std::memcpy(want.data() + off, &v, sizeof(T));
        }
      }
    ++cnt.executed;
    bool case_failed = false;
    for (int y = 0; y < c.h && !case_failed; ++y)
      for (int x = 0; x < c.w && !case_failed; ++x)
        for (int ch = 0; ch < DCH; ++ch) {
          const size_t off = off_bytes<T>(c.oy + y, c.ox + x, d.step, DCH) + (size_t)ch * sizeof(T);
          ++cnt.compared;
          if (!equal_elem<T>(d.host.data() + off, want.data() + off, 0, 0)) {
            char buf[320];
            std::snprintf(buf, sizeof(buf),
                          "[%s] %s mismatch at (x=%d,y=%d,ch=%d): got=%s want=%s", id,
                          c.id().c_str(), x, y, ch, dbg_elem<T>(d.host.data() + off).c_str(),
                          dbg_elem<T>(want.data() + off).c_str());
            if (first.empty()) first = buf;
            EXPECT_TRUE(false) << buf;
            ++cnt.mismatched;
            case_failed = true;
            break;
          }
        }
    if (cnt.mismatched >= 8) break;
  }
  finish(id, cnt, first);
}


// ---------------------------------------------------------------------------
// Transpose: dst(x',y') = src(y',x'), ROI is the source size; the destination
// ROI is (height,width) of the source.  ROI offsets are applied to both.
// ---------------------------------------------------------------------------
template <typename T, int C, class Call>
void run_transpose(const char *id, Call call) {
  NppStreamContext ctx{};
  if (nppGetStreamContext(&ctx) != NPP_NO_ERROR) {
    GTEST_SKIP() << "[" << id << "] no stream context";
    return;
  }
  RunCounts cnt;
  std::string first;
  for (const GenCase &c : make_gen_cases(0)) {
    const int pad = safe_pad<T>(c.pad);
    Buf<T> s(c.w + c.ox, c.h + c.oy, C, pad, 0xCD);
    Buf<T> d(c.h + c.ox, c.w + c.oy, C, pad, 0x5A);
    const uint64_t seed = case_seed(id, c.w, c.h, s.step, d.step, (int)c.regime, 0);
    fill_regime(s, C, seed, c.regime, 0);
    s.dev.copyFromHost(s.host);
    d.dev.copyFromHost(d.host);
    npp_diff::clear_cuda_error();
    NppStatus st = call(
        reinterpret_cast<const T *>(s.dptr() + off_bytes<T>(c.oy, c.ox, s.step, C)), s.step,
        reinterpret_cast<T *>(d.dptr() + off_bytes<T>(c.oy, c.ox, d.step, C)), d.step,
        NppiSize{c.w, c.h}, ctx);
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
    for (int y = 0; y < c.h; ++y)
      for (int x = 0; x < c.w; ++x)
        for (int ch = 0; ch < C; ++ch) {
          const T v = s.hptr()[(size_t)(c.oy + y) * (s.step / sizeof(T)) + (size_t)(c.ox + x) * C + ch];
          const size_t off = off_bytes<T>(c.oy + x, c.ox + y, d.step, C) + (size_t)ch * sizeof(T);
          std::memcpy(want.data() + off, &v, sizeof(T));
        }
    ++cnt.executed;
    bool case_failed = false;
    for (int y = 0; y < c.w && !case_failed; ++y)
      for (int x = 0; x < c.h && !case_failed; ++x)
        for (int ch = 0; ch < C; ++ch) {
          const size_t off = off_bytes<T>(c.oy + y, c.ox + x, d.step, C) + (size_t)ch * sizeof(T);
          ++cnt.compared;
          if (!equal_elem<T>(d.host.data() + off, want.data() + off, 0, 0)) {
            char buf[320];
            std::snprintf(buf, sizeof(buf),
                          "[%s] %s mismatch at (x=%d,y=%d,ch=%d): got=%s want=%s", id,
                          c.id().c_str(), x, y, ch, dbg_elem<T>(d.host.data() + off).c_str(),
                          dbg_elem<T>(want.data() + off).c_str());
            if (first.empty()) first = buf;
            EXPECT_TRUE(false) << buf;
            ++cnt.mismatched;
            case_failed = true;
            break;
          }
        }
    if (cnt.mismatched >= 8) break;
  }
  finish(id, cnt, first);
}

// ---------------------------------------------------------------------------
// Dup: duplicate the single source channel into FILL channels of a STRIDE
// destination; C1AC4R keeps alpha untouched.
// ---------------------------------------------------------------------------
template <typename T, int STRIDE, int FILL, class Call>
void run_dup(const char *id, Call call) {
  NppStreamContext ctx{};
  if (nppGetStreamContext(&ctx) != NPP_NO_ERROR) {
    GTEST_SKIP() << "[" << id << "] no stream context";
    return;
  }
  RunCounts cnt;
  std::string first;
  for (const GenCase &c : make_gen_cases(0)) {
    const int pad = safe_pad<T>(c.pad);
    Buf<T> s(c.w + c.ox, c.h + c.oy, 1, pad, 0xCD);
    Buf<T> d(c.w + c.ox, c.h + c.oy, STRIDE, pad, 0x5A);
    const uint64_t seed = case_seed(id, c.w, c.h, s.step, d.step, (int)c.regime, 0);
    fill_regime(s, 1, seed, c.regime, 0);
    s.dev.copyFromHost(s.host);
    d.dev.copyFromHost(d.host);
    npp_diff::clear_cuda_error();
    NppStatus st = call(
        reinterpret_cast<const T *>(s.dptr() + off_bytes<T>(c.oy, c.ox, s.step, 1)), s.step,
        reinterpret_cast<T *>(d.dptr() + off_bytes<T>(c.oy, c.ox, d.step, STRIDE)), d.step,
        NppiSize{c.w, c.h}, ctx);
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
    for (int y = 0; y < c.h; ++y)
      for (int x = 0; x < c.w; ++x) {
        const T v = s.hptr()[(size_t)(c.oy + y) * (s.step / sizeof(T)) + (size_t)(c.ox + x)];
        for (int ch = 0; ch < FILL; ++ch) {
          const size_t off = off_bytes<T>(c.oy + y, c.ox + x, d.step, STRIDE) + (size_t)ch * sizeof(T);
          std::memcpy(want.data() + off, &v, sizeof(T));
        }
      }
    ++cnt.executed;
    bool case_failed = false;
    for (int y = 0; y < c.h && !case_failed; ++y)
      for (int x = 0; x < c.w && !case_failed; ++x)
        for (int ch = 0; ch < STRIDE; ++ch) {
          const size_t off = off_bytes<T>(c.oy + y, c.ox + x, d.step, STRIDE) + (size_t)ch * sizeof(T);
          ++cnt.compared;
          if (!equal_elem<T>(d.host.data() + off, want.data() + off, 0, 0)) {
            char buf[320];
            std::snprintf(buf, sizeof(buf),
                          "[%s] %s mismatch at (x=%d,y=%d,ch=%d): got=%s want=%s", id,
                          c.id().c_str(), x, y, ch, dbg_elem<T>(d.host.data() + off).c_str(),
                          dbg_elem<T>(want.data() + off).c_str());
            if (first.empty()) first = buf;
            EXPECT_TRUE(false) << buf;
            ++cnt.mismatched;
            case_failed = true;
            break;
          }
        }
    if (cnt.mismatched >= 8) break;
  }
  finish(id, cnt, first);
}

// ---------------------------------------------------------------------------
// SwapChannels: dst[c] = src[aDstOrder[c]] for packed C3/C4; AC4 remaps the
// first three channels and preserves alpha; C3C4 writes nValue for order 3
// and leaves channels with larger orders untouched; C4C3 drops one channel.
// ---------------------------------------------------------------------------
template <typename T, int SC, int DC, int NCOPY, bool C3C4, class Call>
void run_swap(const char *id, Call call, bool inplace) {
  NppStreamContext ctx{};
  if (nppGetStreamContext(&ctx) != NPP_NO_ERROR) {
    GTEST_SKIP() << "[" << id << "] no stream context";
    return;
  }
  RunCounts cnt;
  std::string first;
  for (const GenCase &c : make_gen_cases(0)) {
    const int pad = safe_pad<T>(c.pad);
    Buf<T> s(c.w + c.ox, c.h + c.oy, SC, pad, 0xCD);
    Buf<T> d(c.w + c.ox, c.h + c.oy, DC, pad, 0x5A);
    const uint64_t seed = case_seed(id, c.w, c.h, s.step, d.step, (int)c.regime, 0);
    fill_regime(s, SC, seed, c.regime, 0);
    // In-place APIs operate on the source buffer; out-of-place keeps a
    // separate destination initialized with the sentinel.
    const std::vector<uint8_t> old = inplace ? s.host : d.host;
    s.dev.copyFromHost(s.host);
    if (!inplace) d.dev.copyFromHost(d.host);
    Rng rng(seed ^ 0x2545f491ull);
    const T value = regime_value<T>(rng, c.regime, 7);
    int ord[4];
    static const int perm3[3][3] = {{0, 1, 2}, {2, 0, 1}, {1, 2, 0}};
    static const int perm4[3][4] = {{0, 1, 2, 3}, {3, 2, 1, 0}, {1, 0, 3, 2}};
    if (C3C4) {
      static const int c3c4[3][4] = {{2, 0, 3, 1}, {0, 1, 2, 3}, {3, 1, 0, 2}};
      for (int i = 0; i < 4; ++i) ord[i] = c3c4[(c.w + (int)c.regime) % 3][i];
    } else if (SC == 4 && DC == 3) {
      // C4C3R: NPP's handling of order 3 is not reproducible across entry
      // paths; restrict the permutation to the three real source channels.
      const int *p3 = perm3[(c.h + (int)c.regime) % 3];
      ord[0] = p3[0]; ord[1] = p3[1]; ord[2] = p3[2]; ord[3] = 3;
    } else if (SC <= 3) {
      for (int i = 0; i < 3; ++i) ord[i] = perm3[(c.h + (int)c.regime) % 3][i];
      ord[3] = 3;
    } else {
      for (int i = 0; i < 4; ++i) ord[i] = perm4[(c.h + (int)c.regime) % 3][i];
    }
    T *dptr = reinterpret_cast<T *>(d.dptr() + off_bytes<T>(c.oy, c.ox, d.step, DC));
    if (inplace) dptr = reinterpret_cast<T *>(s.dptr() + off_bytes<T>(c.oy, c.ox, s.step, SC));
    const T *sptr = reinterpret_cast<const T *>(s.dptr() + off_bytes<T>(c.oy, c.ox, s.step, SC));
    npp_diff::clear_cuda_error();
    NppStatus st = call(sptr, s.step, dptr, inplace ? s.step : d.step, NppiSize{c.w, c.h}, ord,
                        value, ctx);
    if (st != NPP_NO_ERROR) {
      ++cnt.skipped;
      if (st == NPP_NOT_IMPLEMENTED_ERROR) ++cnt.notimpl;
      continue;
    }
    const cudaError_t se = cudaDeviceSynchronize();
    ASSERT_EQ(se, cudaSuccess) << "[" << id << "] " << c.id()
                               << " sync: " << cudaGetErrorString(se);
    if (inplace) s.dev.copyToHost(s.host); else d.dev.copyToHost(d.host);
    std::vector<uint8_t> want = inplace ? s.host : d.host;
    const int dstep = inplace ? s.step : d.step;
    const std::vector<uint8_t> &src_host = inplace ? old : s.host;
    for (int y = 0; y < c.h; ++y)
      for (int x = 0; x < c.w; ++x) {
        for (int ch = 0; ch < DC; ++ch) {
          if (!C3C4 && ch >= NCOPY) continue;  // AC4 alpha is preserved
          const int oc = ord[ch];
          if (C3C4 && oc > 3) continue;  // untouched
          T r{};
          if (C3C4 && oc == 3) {
            r = value;
          } else if (!C3C4 && SC == 4 && DC == 3 && oc == 3) {
            // NPP's C4C3R writes 0 when the order selects source channel 3.
            r = T(0);
          } else {
            if (oc < 0 || oc >= SC) continue;
            r = reinterpret_cast<const T *>(src_host.data())[
                (size_t)(c.oy + y) * (s.step / sizeof(T)) + (size_t)(c.ox + x) * SC + oc];
          }
          const size_t off = off_bytes<T>(c.oy + y, c.ox + x, dstep, DC) + (size_t)ch * sizeof(T);
          std::memcpy(want.data() + off, &r, sizeof(T));
        }
      }
    ++cnt.executed;
    bool case_failed = false;
    for (int y = 0; y < c.h && !case_failed; ++y)
      for (int x = 0; x < c.w && !case_failed; ++x)
        for (int ch = 0; ch < DC; ++ch) {
          const size_t off = off_bytes<T>(c.oy + y, c.ox + x, dstep, DC) + (size_t)ch * sizeof(T);
          ++cnt.compared;
          if (!equal_elem<T>((inplace ? s.host : d.host).data() + off, want.data() + off, 0, 0)) {
            char buf[460];
            char sdesc[200];
            const std::vector<uint8_t> &dbgsrc = inplace ? old : s.host;
            int used = std::snprintf(sdesc, sizeof(sdesc), "src0..%d=", SC - 1);
            for (int q = 0; q < SC; ++q) {
              used += std::snprintf(sdesc + used, sizeof(sdesc) - (size_t)used, " %s",
                                    dbg_elem<T>(dbgsrc.data() +
                                                off_bytes<T>(c.oy + y, c.ox + x, s.step, SC) +
                                                (size_t)q * sizeof(T)).c_str());
              if (used >= (int)sizeof(sdesc)) break;
            }
            std::snprintf(buf, sizeof(buf),
                          "[%s] %s mismatch at (x=%d,y=%d,ch=%d): got=%s want=%s "
                          "ord=%d,%d,%d,%d %s",
                          id, c.id().c_str(), x, y, ch,
                          dbg_elem<T>((inplace ? s.host : d.host).data() + off).c_str(),
                          dbg_elem<T>(want.data() + off).c_str(), ord[0], ord[1], ord[2], ord[3],
                          sdesc);
            if (first.empty()) first = buf;
            EXPECT_TRUE(false) << buf;
            ++cnt.mismatched;
            case_failed = true;
            break;
          }
        }
    if (cnt.mismatched >= 8) break;
  }
  finish(id, cnt, first);
}


// ---------------------------------------------------------------------------
// Border copies: the (sw,sh) source is placed at (left,top) in a larger
// destination.  Const borders write the supplied value(s), replicate borders
// clamp to the source edge and wrap borders use modulo coordinates.  AC4
// leaves the destination alpha untouched.
// ---------------------------------------------------------------------------
enum class BorderKind { Const, Replicate, Wrap };

template <typename T, typename VT, int C, int A, BorderKind K, class Call>
void run_border(const char *id, Call call, bool per_channel) {
  NppStreamContext ctx{};
  if (nppGetStreamContext(&ctx) != NPP_NO_ERROR) {
    GTEST_SKIP() << "[" << id << "] no stream context";
    return;
  }
  RunCounts cnt;
  std::string first;
  for (const GenCase &c : make_gen_cases(0)) {
    const int sw = c.w, sh = c.h, top = 1, left = 2;
    const int dw = sw + 2 * left, dh = sh + 2 * top;
    const int pad = safe_pad<T>(c.pad);
    Buf<T> s(sw + 0, sh + 0, C, pad, 0xCD);
    Buf<T> d(dw + 0, dh + 0, C, pad, 0x5A);
    const uint64_t seed = case_seed(id, c.w, c.h, s.step, d.step, (int)c.regime, 0);
    fill_regime(s, C, seed, c.regime, 0);
    s.dev.copyFromHost(s.host);
    d.dev.copyFromHost(d.host);
    Rng rng(seed ^ 0x9e3779b97f4a7c15ull);
    VT vals[C > 0 ? C : 1];
    for (int ch = 0; ch < C; ++ch) vals[ch] = regime_value<VT>(rng, c.regime, 5 + ch);
    const VT scalar = vals[0];
    npp_diff::clear_cuda_error();
    NppStatus st = call(
        reinterpret_cast<const T *>(s.dptr() + off_bytes<T>(0, 0, s.step, C)), s.step,
        NppiSize{sw, sh},
        reinterpret_cast<T *>(d.dptr() + off_bytes<T>(0, 0, d.step, C)), d.step,
        NppiSize{dw, dh}, top, left, per_channel ? vals : nullptr, scalar, ctx);
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
    for (int y = 0; y < dh; ++y)
      for (int x = 0; x < dw; ++x) {
        const int sx = x - left, sy = y - top;
        const bool inner = (sx >= 0 && sx < sw && sy >= 0 && sy < sh);
        for (int ch = 0; ch < A; ++ch) {
          T r{};
          if (inner) {
            r = s.hptr()[(size_t)(0 + sy) * (s.step / sizeof(T)) + (size_t)(0 + sx) * C + ch];
          } else if (K == BorderKind::Const) {
            r = convert_value<T>(per_channel ? vals[ch] : scalar);
          } else {
            int rx = sx, ry = sy;
            if (K == BorderKind::Replicate) {
              if (rx < 0) rx = 0; else if (rx >= sw) rx = sw - 1;
              if (ry < 0) ry = 0; else if (ry >= sh) ry = sh - 1;
            } else {
              rx = ((sx % sw) + sw) % sw;
              ry = ((sy % sh) + sh) % sh;
            }
            r = s.hptr()[(size_t)(0 + ry) * (s.step / sizeof(T)) + (size_t)(0 + rx) * C + ch];
          }
          const size_t off = off_bytes<T>(0 + y, 0 + x, d.step, C) + (size_t)ch * sizeof(T);
          std::memcpy(want.data() + off, &r, sizeof(T));
        }
      }
    ++cnt.executed;
    bool case_failed = false;
    for (int y = 0; y < dh && !case_failed; ++y)
      for (int x = 0; x < dw && !case_failed; ++x)
        for (int ch = 0; ch < C; ++ch) {
          const size_t off = off_bytes<T>(0 + y, 0 + x, d.step, C) + (size_t)ch * sizeof(T);
          ++cnt.compared;
          if (!equal_elem<T>(d.host.data() + off, want.data() + off, 0, 0)) {
            char buf[320];
            std::snprintf(buf, sizeof(buf),
                          "[%s] %s mismatch at (x=%d,y=%d,ch=%d): got=%s want=%s", id,
                          c.id().c_str(), x, y, ch, dbg_elem<T>(d.host.data() + off).c_str(),
                          dbg_elem<T>(want.data() + off).c_str());
            if (first.empty()) first = buf;
            EXPECT_TRUE(false) << buf;
            ++cnt.mismatched;
            case_failed = true;
            break;
          }
        }
    if (cnt.mismatched >= 8) break;
  }
  finish(id, cnt, first);
}

// ---------------------------------------------------------------------------
// Sub-pixel copy: bilinear sample of the source at (x+dx, y+dy), clamped to
// the source edges.  Integer destinations round half away from zero; 32f
// passes the interpolated value through.  AC4 preserves alpha.
// ---------------------------------------------------------------------------
template <typename T, int C, int A, class Call>
void run_subpix(const char *id, Call call) {
  NppStreamContext ctx{};
  if (nppGetStreamContext(&ctx) != NPP_NO_ERROR) {
    GTEST_SKIP() << "[" << id << "] no stream context";
    return;
  }
  // NPP's negative sub-pixel offsets take a different, vectorization-dependent
  // edge path; the matrix uses non-negative offsets where bilinear sampling is
  // well defined.
  static const float offsets[][2] = {{0.0f, 0.0f},  {0.5f, 0.0f},  {0.0f, 0.5f},
                                     {0.25f, 0.75f}, {0.5f, 0.25f}, {1.5f, 0.5f}};
  RunCounts cnt;
  std::string first;
  for (int oi = 0; oi < (int)(sizeof(offsets) / sizeof(offsets[0])); ++oi) {
    const float dx = offsets[oi][0], dy = offsets[oi][1];
    // NPP uses only the fractional part of the sub-pixel offset; the integer
    // part is discarded (probe: dx=1.5 behaves exactly like dx=0.5).
    const float fdx = dx - std::trunc(dx);
    const float fdy = dy - std::trunc(dy);
    for (const GenCase &c : make_gen_cases(0)) {
      const int pad = safe_pad<T>(c.pad);
      Buf<T> s(c.w + c.ox + 2, c.h + c.oy + 2, C, pad, 0xCD);
      Buf<T> d(c.w + c.ox, c.h + c.oy, C, pad, 0x5A);
      const uint64_t seed = case_seed(id, c.w, c.h, s.step, d.step, (int)c.regime, oi);
      fill_regime(s, C, seed, c.regime, 0);
      s.dev.copyFromHost(s.host);
      d.dev.copyFromHost(d.host);
      npp_diff::clear_cuda_error();
      NppStatus st = call(
          reinterpret_cast<const T *>(s.dptr() + off_bytes<T>(c.oy, c.ox, s.step, C)), s.step,
          reinterpret_cast<T *>(d.dptr() + off_bytes<T>(c.oy, c.ox, d.step, C)), d.step,
          NppiSize{c.w, c.h}, dx, dy, ctx);
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
      for (int y = 0; y < c.h; ++y)
        for (int x = 0; x < c.w; ++x) {
          const float sx = (float)x + fdx, sy = (float)y + fdy;
          int xa0, xa1, ya0, ya1;
          float fx, fy;
          if (sx < 0.f) { xa0 = 0; fx = 0.f; }
          else if (sx >= (float)c.w) { xa0 = c.w - 1; fx = 0.f; }
          else { xa0 = (int)sx; fx = sx - (float)xa0; }
          xa1 = (xa0 + 1 < c.w) ? xa0 + 1 : xa0;
          if (sy < 0.f) { ya0 = 0; fy = 0.f; }
          else if (sy >= (float)c.h) { ya0 = c.h - 1; fy = 0.f; }
          else { ya0 = (int)sy; fy = sy - (float)ya0; }
          ya1 = (ya0 + 1 < c.h) ? ya0 + 1 : ya0;
          for (int ch = 0; ch < A; ++ch) {
            auto at = [&](int yy, int xx) {
              return s.hptr()[(size_t)(c.oy + yy) * (s.step / sizeof(T)) + (size_t)(c.ox + xx) * C + ch];
            };
            // The 5090 kernel interpolates in float even for 32s inputs;
            // mirroring the float expression keeps large integers bit-equal.
            const float v00 = (float)at(ya0, xa0), v10 = (float)at(ya0, xa1);
            const float v01 = (float)at(ya1, xa0), v11 = (float)at(ya1, xa1);
            const float ro = v00 * (1.f - fx) * (1.f - fy) + v10 * fx * (1.f - fy) +
                             v01 * (1.f - fx) * fy + v11 * fx * fy;
            T r{};
            if constexpr (std::is_same_v<T, Npp32f>) {
              r = (T)ro;
            } else if constexpr (std::is_same_v<T, Npp16f>) {
              r = half_round(ro);
            } else {
              // Saturate like the device conversion: float(INT_MAX) rounds up
              // to 2^31, and casting that back on the host would be UB.
              if (ro <= (float)type_min<T>()) r = type_min<T>();
              else if (ro >= (float)type_max<T>()) r = type_max<T>();
              else r = (T)std::round(ro);
            }
            const size_t off = off_bytes<T>(c.oy + y, c.ox + x, d.step, C) + (size_t)ch * sizeof(T);
            std::memcpy(want.data() + off, &r, sizeof(T));
          }
        }
      ++cnt.executed;
      bool case_failed = false;
      const double rtol = std::is_same_v<T, Npp32f> ? 1e-5 : 0.0;
      const double atol = std::is_same_v<T, Npp32f> ? 1e-4 : 0.0;
      // Integer interpolation is computed through float in the kernel; the
      // achievable accuracy is a few ulps of the largest source magnitude,
      // not of the (possibly cancelled) result.
      double src_scale = 0.0;
      for (int y = 0; y < c.h; ++y)
        for (int x = 0; x < c.w; ++x)
          for (int ch = 0; ch < C; ++ch)
            {
              const double av = std::fabs(as_double(
                  s.hptr()[(size_t)(c.oy + y) * (s.step / sizeof(T)) + (size_t)(c.ox + x) * C + ch]));
              if (std::isfinite(av)) src_scale = std::max(src_scale, av);
            }
      const double itol = 1.0 + 2e-6 * src_scale;
      for (int y = 0; y < c.h && !case_failed; ++y)
        for (int x = 0; x < c.w && !case_failed; ++x)
          for (int ch = 0; ch < C; ++ch) {
            const size_t off = off_bytes<T>(c.oy + y, c.ox + x, d.step, C) + (size_t)ch * sizeof(T);
            ++cnt.compared;
            bool ok;
            if constexpr (std::is_floating_point_v<T> || std::is_same_v<T, Npp16f>) {
              double fat = atol;
              if constexpr (std::is_same_v<T, Npp32f>) {
                if (std::isfinite(src_scale)) fat += 1e-6 * src_scale;
              }
              ok = equal_elem<T>(d.host.data() + off, want.data() + off, rtol, fat);
            } else {
              // Integer interpolation goes through float in the kernel; allow
              // a few ulps of float slop plus one rounding unit.
              const double gv = as_double(*reinterpret_cast<const T *>(d.host.data() + off));
              const double wv = as_double(*reinterpret_cast<const T *>(want.data() + off));
              ok = std::fabs(gv - wv) <= itol;
            }
            if (!ok) {
              char buf[320];
              std::snprintf(buf, sizeof(buf),
                            "[%s] %s dx=%g dy=%g mismatch at (x=%d,y=%d,ch=%d): got=%s want=%s", id,
                            c.id().c_str(), dx, dy, x, y, ch,
                            dbg_elem<T>(d.host.data() + off).c_str(),
                            dbg_elem<T>(want.data() + off).c_str());
              if (first.empty()) first = buf;
              EXPECT_TRUE(false) << buf;
              ++cnt.mismatched;
              case_failed = true;
              break;
            }
          }
      if (cnt.mismatched >= 8) break;
    }
    if (cnt.mismatched >= 8) break;
  }
  finish(id, cnt, first);
}


// ---------------------------------------------------------------------------
// Convert family.  Plain conversions truncate toward zero and saturate for
// integer destinations; Sfs variants divide by 2^sf first and round per
// NppRoundMode (NEAR ties-to-even, FINANCIAL ties away from zero, ZERO toward
// zero).  Floating destinations keep the fractional value (16f rounds the
// mantissa to nearest); AC4 leaves alpha untouched.
// ---------------------------------------------------------------------------
inline double round_mode_host(double v, int mode, bool integer_source = false) {
  if (std::isnan(v)) return v;
  // Values beyond the 64-bit range are already saturated on emission; skip the
  // integer rounding to avoid an undefined host cast.
  if (std::fabs(v) >= 9.2233720368547758e18) return v;
  // NPP's integer-source Sfs kernels implement FINANCIAL as (v+0.5) floor,
  // i.e. ties toward +infinity; the floating-source kernels round away.
  if (mode == NPP_RND_FINANCIAL)
    return integer_source ? std::floor(v + 0.5)
                          : (v >= 0.0 ? std::floor(v + 0.5) : std::ceil(v - 0.5));
  if (mode == NPP_RND_ZERO) return v >= 0.0 ? std::floor(v) : std::ceil(v);
  const double av = std::fabs(v);
  const double lo = std::floor(av);
  const double fr = av - lo;
  long long m = (long long)lo;
  const bool up = (fr > 0.5) || (fr == 0.5 && (m & 1LL));
  const double mag = (double)m + (up ? 1.0 : 0.0);
  return v < 0.0 ? -mag : mag;
}

template <typename D> inline D saturate_convert(double v) {
  if constexpr (std::is_same_v<D, Npp16f>) {
    return half_round((float)v);
  } else if constexpr (!std::is_integral_v<D>) {
    return (D)v;  // floating destinations keep NaN/Inf
  } else {
    if (std::isnan(v)) v = -std::numeric_limits<double>::infinity();
    if constexpr (std::is_unsigned_v<D>) {
      const double hi = (double)type_max<D>();
      return (D)(v <= 0.0 ? 0.0 : (v >= hi ? hi : v));
    } else {
      const double lo = (double)type_min<D>();
      const double hi = (double)type_max<D>();
      return (D)(v <= lo ? lo : (v >= hi ? hi : v));
    }
  }
}

template <typename S, typename D, int C, int A, class Call>
void run_convert(const char *id, Call call, bool sfs, bool round_mode, bool fin_away = false) {
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
    for (const GenCase &c : make_gen_cases(sf)) {
      const int pad = safe_pad<S>(c.pad);
      Buf<S> s(c.w + c.ox, c.h + c.oy, C, pad, 0xCD);
      Buf<D> d(c.w + c.ox, c.h + c.oy, C, pad, 0x5A);
      const uint64_t seed = case_seed(id, c.w, c.h, s.step, d.step, (int)c.regime, sf);
      const bool finite_only = (sfs || round_mode) &&
                               (std::is_floating_point_v<S> || std::is_same_v<S, Npp16f>);
      fill_regime(s, C, seed, c.regime, 0, false, false, finite_only);
      s.dev.copyFromHost(s.host);
      d.dev.copyFromHost(d.host);
      for (int mi = 0; mi < (round_mode ? 3 : 1); ++mi) {
        // Sfs variants without an explicit round-mode parameter use the
        // default nearest/ties-to-even rule.
        const int mode = round_mode ? modes[mi] : NPP_RND_NEAR;
        npp_diff::clear_cuda_error();
        NppStatus st = call(
            reinterpret_cast<const S *>(s.dptr() + off_bytes<S>(c.oy, c.ox, s.step, C)), s.step,
            reinterpret_cast<D *>(d.dptr() + off_bytes<D>(c.oy, c.ox, d.step, C)), d.step,
            NppiSize{c.w, c.h}, mode, sf, ctx);
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
        for (int y = 0; y < c.h; ++y)
          for (int x = 0; x < c.w; ++x)
            for (int ch = 0; ch < A; ++ch) {
              double v = as_double(s.hptr()[(size_t)(c.oy + y) * (s.step / sizeof(S)) +
                                            (size_t)(c.ox + x) * C + ch]);
              if (sfs) v /= (double)(1ll << sf);
              // NPP applies eRoundMode to the value before the destination
              // format conversion, including 32f -> 16f.
              if (sfs || round_mode)
                v = round_mode_host(v, mode, std::is_integral_v<S> && !fin_away);
              D r;
              if constexpr (std::is_integral_v<D>) {
                // Match the device cast: trunc toward zero, with NaN/Inf
                // clamped before the 64-bit conversion.
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
              const size_t off = off_bytes<D>(c.oy + y, c.ox + x, d.step, C) + (size_t)ch * sizeof(D);
              std::memcpy(want.data() + off, &r, sizeof(D));
            }
        ++cnt.executed;
        bool case_failed = false;
        for (int y = 0; y < c.h && !case_failed; ++y)
          for (int x = 0; x < c.w && !case_failed; ++x)
            for (int ch = 0; ch < C; ++ch) {
              const size_t off = off_bytes<D>(c.oy + y, c.ox + x, d.step, C) + (size_t)ch * sizeof(D);
              ++cnt.compared;
              const double rt = std::is_same_v<D, Npp32f> ? 1e-6 : 0.0;
              if (!equal_elem<D>(d.host.data() + off, want.data() + off, rt, rt)) {
                char buf[360];
                std::snprintf(buf, sizeof(buf),
                              "[%s] %s sf=%d mode=%d mismatch at (x=%d,y=%d,ch=%d): got=%s want=%s",
                              id, c.id().c_str(), sf, mode, x, y, ch,
                              dbg_elem<D>(d.host.data() + off).c_str(),
                              dbg_elem<D>(want.data() + off).c_str());
                if (first.empty()) first = buf;
                EXPECT_TRUE(false) << buf;
                ++cnt.mismatched;
                case_failed = true;
                break;
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

}  // namespace num
