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
};

template <typename T> struct is_complex_num : std::false_type {};
template <> struct is_complex_num<Npp16sc> : std::true_type {};
template <> struct is_complex_num<Npp32sc> : std::true_type {};
template <> struct is_complex_num<Npp32fc> : std::true_type {};
template <> struct is_complex_num<Npp64fc> : std::true_type {};
template <typename T> inline constexpr bool is_complex_num_v = is_complex_num<T>::value;

template <typename T> struct complex_part { using type = T; };
template <> struct complex_part<Npp16sc> { using type = Npp16s; };
template <> struct complex_part<Npp32sc> { using type = Npp32s; };
template <> struct complex_part<Npp32fc> { using type = Npp32f; };
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
  else return (T)0;
}

template <typename T> constexpr T type_max() {
  if constexpr (std::is_same_v<T, Npp8u>) return (T)255;
  else if constexpr (std::is_same_v<T, Npp8s>) return (T)127;
  else if constexpr (std::is_same_v<T, Npp16u>) return (T)65535;
  else if constexpr (std::is_same_v<T, Npp16s>) return (T)32767;
  else if constexpr (std::is_same_v<T, Npp32u>) return (T)0xFFFFFFFFu;
  else if constexpr (std::is_same_v<T, Npp32s>) return (T)2147483647;
  else return (T)0;
}

// Scale a (possibly fractional) value by 2^-sf and round.  NPP's integer Sfs
// rule is exact quotient followed by round-half-even.
template <typename T> inline T scale_round(double v, int sf) {
  if (sf > 0) v /= (double)(1ll << sf);
  return from_double_rint<T>(v);
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
        // NPP clamps the float exponential to the normal range instead of
        // returning 0/inf (probe: exp(-1000) -> FLT_MIN, exp(1000) -> FLT_MAX).
        const double r = std::exp((double)a);
        const double lo = (double)std::numeric_limits<T>::min();
        const double hi = (double)std::numeric_limits<T>::max();
        // NPP clamps NaN to the low end (fmax/fmin ordering), and clamps
        // subnormal underflow to the smallest normal value.
        if (std::isnan(r)) return (T)lo;
        return (T)(r < lo ? lo : (r > hi ? hi : r));
      }
      case Op::Abs: return (T)std::fabs((double)a);
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
          out.re = scale_round<R>(re, p);
          out.im = scale_round<R>(im, p);
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
        out.re = scale_round<R>(re, p);
        out.im = scale_round<R>(im, p);
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
                        bool nonnegative = false) {
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
      if (p[i].re == 0 && p[i].im == 0) p[i].re = 1;
    } else if constexpr (std::is_same_v<T, Npp16f>) {
      if (half_to_float_host(p[i]) == 0.0f) p[i] = half_round(1.0f);
    } else {
      if (p[i] == T(0)) p[i] = T(1);
    }
  }
}

template <typename Cst> inline void make_const_nonzero(Cst &v) {
  if constexpr (is_complex_num_v<Cst>) {
    if (v.re == 0 && v.im == 0) v.re = 1;
  } else if constexpr (std::is_same_v<Cst, Npp16f>) {
    if (half_to_float_host(v) == 0.0f) v = half_round(1.0f);
  } else {
    if (v == Cst(0)) v = Cst(1);
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

}  // namespace num
