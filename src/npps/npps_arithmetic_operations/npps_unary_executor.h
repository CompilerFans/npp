#pragma once

// Shared 1D executor for the npps scalar / bitwise / shift / reduction
// arithmetic families implemented.1c:
//   nppsSqrt / nppsSqr / nppsExp / nppsLn / npps10Log10 / nppsAbs / nppsArctan
//   / nppsCubrt  (scalar unary, scaled and plain)
//   nppsAnd / nppsOr / nppsXor / nppsNot (bitwise)
//   nppsLShiftC / nppsRShiftC (constant shift)
//   nppsMul_Low
//   nppsSumLn / nppsSumLnGetBufferSize (reduction)
//   nppsCauchy / nppsCauchyD / nppsCauchyDD2 (in-place distribution transform)
//
// Host-visible surface (safe for plain.cpp translation units): executor
// entry-point templates are declared here and defined + explicitly
// instantiated in npps_unary_ops.cu.
//
// Semantics (validated against NVIDIA NPP 13.4 by reference comparisons):
//   * scaled integer unary: d[i] = F::apply(src[i], sf)
//   * plain float unary: d[i] = F::apply(src[i]).
//   * in-place (_I/_ISfs): pSrcDst = F(pSrcDst).
//   * bitwise And/Or/Xor: pDst = pSrc1 op pSrc2 (integer only)
//   * nppsNot: pDst = ~pSrc
//   * nppsLShiftC/RShiftC: pDst = pSrc << nValue | pSrc >> nValue
//   * nppsMul_Low_32s_Sfs: low 32 bits of the 64-bit product, then scaled
//   * nppsSumLn: scalar sum of natural logarithms

#include "npp.h"
#include "npp_version_compat.h"
#include "npps_elementwise_executor.h"

#include <cuda_runtime.h>
#include <cmath>
#include <cstdint>

namespace npps {
namespace unary {

namespace ed = npps::elementwise;
namespace ek = npps::elementwise::detail;

// ---------------------------------------------------------------------------
// Scalar unary functors (shared with generated wrappers).
// ---------------------------------------------------------------------------
// NOTE: the host toolchain (g++ via cu-bridge) does not expose the C99
// long-double <cmath> family (floorl/ceill/sqrtl/logl/expl/atanl/cbrtl/hypotl),
// so all transcendental intermediate math is done in `double`; only the
// fixed-point scaling/rounding step is integer-based.  Reference comparisons use
// rtol/atol=1e-5 so double precision is ample.
__device__ __host__ inline long long rnd(double v) {
  // round to nearest, ties to even.
  if (v >= 0.0) {
    const double lo = std::floor(v);
    const double f = v - lo;
    long long r = (long long)lo;
    if (f > 0.5 || (f == 0.5 && (r & 1LL))) ++r;
    return r;
  } else {
    const double hi = std::ceil(v);
    const double f = hi - v;
    long long r = (long long)hi;
    if (f > 0.5 || (f == 0.5 && (r & 1LL))) --r;
    return r;
  }
}

template <typename T> __device__ __host__ inline double rp(const T& a) { return (double)ek::re_l(a); }
template <typename T> __device__ __host__ inline double ip(const T& a) { return (double)ek::im_l(a); }

// Sfs scaling for transcendental results: round-half-even(value / 2^sf) in a
// single step, then saturate.  Unlike round-then-scale this avoids double
// rounding, which NPP does NOT apply (validated against NPP 13.4 for Sqrt).
template <typename D> __device__ __host__ inline D ssc(double val, int sf) {
  if (sf > 0) val = val / (double)(1LL << sf);
  else if (sf < 0) val = val * (double)(1LL << (-sf));
  return ek::sat_ll<D>(rnd(val));
}

struct SqrtOp {
  template <typename A, typename D> __device__ __host__ D applyF(const A& a) const {
    using C = ed::component_t_ty<D>;
    if constexpr (ed::is_complex_v<D>) {
      if constexpr (std::is_same_v<C, Npp32f>) {
        // NPP evaluates the 32fc square root end-to-end in single precision.
        const float re = (float)rp(a), im = (float)ip(a);
        const float r = std::hypot(re, im);
        const float gamma = std::sqrt((r + re) / 2.0f);
        const float delta = std::sqrt((r - re) / 2.0f);
        D d; d.re = gamma; d.im = (im < 0.0f ? -delta: delta); return d;
      } else {
        double re = rp(a), im = ip(a);
        double r = std::hypot(re, im);
        double gamma = std::sqrt((r + re) / 2.0);
        double delta = std::sqrt((r - re) / 2.0);
        D d; d.re = (C)gamma; d.im = (C)(im < 0 ? -delta: delta); return d;
      }
    } else return (D)std::sqrt(rp(a));
  }
  template <typename A, typename D> __device__ __host__ D applyS(const A& a, int sf) const {
    using C = ed::component_t_ty<D>;
    if constexpr (ed::is_complex_v<D>) {
      // NPP 13.4 truncates (floor) the scaled components of the complex sqrt,
      // rather than round-half-even (validated against reference outputs).
      double re = rp(a), im = ip(a);
      double r = std::hypot(re, im);
      double gamma = std::sqrt((r + re) / 2.0), delta = std::sqrt((r - re) / 2.0);
      // Principal square root: real part is gamma, imaginary part carries
      // the sign of the input's imaginary component.
      double gr = gamma, gi = (im < 0 ? -delta: delta);
      double s = (1ll << sf);
      D d; d.re = (C)(gr / s); d.im = (C)(gi / s); return d;
    } else {
      double x = rp(a);
      if (x < 0) {
        // NPP: 16-bit destinations map negative inputs to 0; wider signed
        // destinations saturate only when sf == 0, except 64s which always
        // maps negative inputs to its minimum.
        if constexpr (std::is_same_v<D, Npp64s>) {
          using L = std::numeric_limits<D>;
          return (D)L::lowest();
        } else if constexpr (std::is_signed_v<D>) {
          if (sf == 0 && sizeof(A) >= 4) {
            using L = std::numeric_limits<D>;
            return (D)L::lowest();
          }
          return (D)0;
        } else {
          return (D)0;
        }
      }
      if constexpr (std::is_same_v<D, Npp16s> && std::is_same_v<A, Npp32s>) {
        // Single-precision path: sf == 0 rounds half away from zero,
        // sf > 0 rounds half to even.
        const float r = (float)std::sqrt((double)(float)x);
        if (sf == 0) return ek::sat_ll<D>((long long)(r + 0.5f));
        return ek::sat_ll<D>(rnd((double)(r / (float)(1 << sf))));
      }
      return ssc<D>(std::sqrt(x), sf);
    }
  }
};

struct SqrOp {
  template <typename A, typename D> __device__ __host__ D applyF(const A& a) const {
    using C = ed::component_t_ty<D>;
    if constexpr (ed::is_complex_v<D>) {
      double re = rp(a), im = ip(a);
      D d; d.re = (C)(re*re - im*im); d.im = (C)(2.0*re*im); return d;
    } else { double x = rp(a); return (D)(x * x); }
  }
  template <typename A, typename D> __device__ __host__ D applyS(const A& a, int sf) const {
    using C = ed::component_t_ty<D>;
    if constexpr (ed::is_complex_v<D>) {
      double re = rp(a), im = ip(a);
      D d; d.re = (C)ek::scale_sat<C>(rnd(re*re - im*im), sf);
          d.im = (C)ek::scale_sat<C>(rnd(2.0*re*im), sf); return d;
    } else {
      long long v = rnd(rp(a) * rp(a));
      // 8u at sf == 0 mirrors the narrow 16-bit multiply path of nppsMul.
      if constexpr (std::is_same_v<D, Npp8u> && sizeof(A) == 1) {
        if (sf == 0) v = (long long)(int16_t)(uint16_t)v;
      }
      return ek::scale_sat<D>(v, sf);
    }
  }
};

struct ExpOp {
  template <typename A, typename D> __device__ __host__ D applyF(const A& a) const {
    if constexpr (std::is_floating_point_v<D>) {
      const double r = std::exp(rp(a));
      using L = std::numeric_limits<D>;
      if (r != r) return (D)L::min();
      return (D)(r < (double)L::min() ? (double)L::min(): (r > (double)L::max() ? (double)L::max(): r));
    }
    return (D)std::exp(rp(a));
  }
  template <typename A, typename D> __device__ __host__ D applyS(const A& a, int sf) const {
    // NPP evaluates 32-bit integer Exp in single precision and rounds to
    // nearest; 64-bit destinations truncate toward zero.
    if constexpr (sizeof(D) <= 4) {
      return ssc<D>((double)expf((float)rp(a)), sf);
    } else {
      // 64-bit Exp truncates at sf == 0 and rounds half to even otherwise.
      const double v = std::exp(rp(a));
      const double lim = (double)std::numeric_limits<long long>::max();
      if (sf == 0) {
        if (v >= lim) return (D)std::numeric_limits<D>::max();
        return ek::sat_ll<D>((long long)v);
      }
      const double s = v / (double)(1LL << sf);
      if (s >= lim) return (D)std::numeric_limits<D>::max();
      return ek::sat_ll<D>(rnd(s));
    }
  }
};

struct LnOp {
  template <typename A, typename D> __device__ __host__ D applyF(const A& a) const {
    if constexpr (std::is_same_v<A, Npp64f> && std::is_same_v<D, Npp32f>) {
      // NPP's mixed 64f->32f entry point narrows the input to float, rounds
      // logf(x) to int64 (saturating: +inf -> INT64_MAX, NaN/-inf ->
      // INT64_MIN), then widens the integer back to float.
      const float lf = (float)std::log((double)(float)rp(a));
      long long q;
      if (lf != lf || lf == -std::numeric_limits<float>::infinity()) {
        q = std::numeric_limits<long long>::min();
      } else if (lf == std::numeric_limits<float>::infinity()) {
        q = std::numeric_limits<long long>::max();
      } else {
        q = rnd((double)lf);
      }
      return (D)(float)q;
    }
    return (D)std::log(rp(a));
  }
  template <typename A, typename D> __device__ __host__ D applyS(const A& a, int sf) const {
    if constexpr (std::is_integral_v<A>) {
      // NPP: log(0) -> signed minimum; negative inputs -> minimum only at
      // sf == 0, otherwise 0.  Unsigned destinations map both to 0.
      const double x = rp(a);
      if (x == 0.0 || (x < 0.0 && sf == 0)) {
        if constexpr (std::is_signed_v<D>) {
          using L = std::numeric_limits<D>;
          return (D)L::lowest();
        } else {
          return (D)0;
        }
      }
      if (x < 0.0) return (D)0;
    }
    return ssc<D>(std::log(rp(a)), sf);
  }
};

struct Log10Op {
  template <typename A, typename D> __device__ __host__ D applyF(const A& a) const {
    return (D)(10.0 * std::log(rp(a)) / std::log(10.0));
  }
  template <typename A, typename D> __device__ __host__ D applyS(const A& a, int sf) const {
    const double x = rp(a);
    // NPP 10Log10 evaluates 10*log10f in single precision.  sf == 0
    // truncates toward zero; sf > 0 rounds half to even.  Negative -> 0,
    // zero -> signed minimum.
    if (x < 0.0) return (D)0;
    if (x == 0.0) {
      if constexpr (std::is_signed_v<D>) {
        using L = std::numeric_limits<D>;
        return (D)L::lowest();
      } else {
        return (D)0;
      }
    }
    const float v = 10.0f * (float)std::log10((double)(float)x);
    if (sf == 0) return ek::sat_ll<D>((long long)v);
    return ek::sat_ll<D>(rnd((double)(v / (float)(1 << sf))));
  }
};

struct AbsOp {
  template <typename A, typename D> __device__ __host__ D applyF(const A& a) const { return (D)std::fabs(rp(a)); }
  template <typename A, typename D> __device__ __host__ D applyS(const A& a, int sf) const {
    (void)sf;
    const long long v = ek::re_ll(a);
    // 32s Abs wraps INT_MIN back to itself; 16-bit destinations saturate.
    if constexpr (std::is_same_v<D, Npp32s>) {
      if (v == (long long)std::numeric_limits<Npp32s>::min()) return (D)v;
    }
    return ek::sat_ll<D>(v < 0 ? -v: v);
  }
};

struct ArctanOp {
  template <typename A, typename D> __device__ __host__ D applyF(const A& a) const { return (D)std::atan(rp(a)); }
  template <typename A, typename D> __device__ __host__ D applyS(const A& a, int sf) const { (void)sf; return applyF<A, D>(a); }
};

struct CubrtOp {
  template <typename A, typename D> __device__ __host__ D applyF(const A& a) const {
    double x = rp(a);
    return (D)(x < 0 ? -std::cbrt(-x): std::cbrt(x));
  }
  template <typename A, typename D> __device__ __host__ D applyS(const A& a, int sf) const {
    if constexpr (std::is_same_v<D, Npp16s> && std::is_same_v<A, Npp32s>) {
      // NPP's 32s->16s cube root is single precision; sf == 0 rounds half
      // away from zero and sf > 0 rounds half to even.
      const float r = (float)std::cbrt((double)(float)rp(a));
      return ek::sat_ll<D>(rnd((double)(r / (float)(1 << sf))));
    }
    double x = rp(a);
    double cb = x < 0 ? -std::cbrt(-x): std::cbrt(x);
    return ssc<D>(cb, sf);
  }
};

// ---------------------------------------------------------------------------
// Scalar unary entry points
// ---------------------------------------------------------------------------
template <typename F, typename A, typename D>
NppStatus runSfs(const A* src, D* dst, NppSignalLength nLength, int nScaleFactor, cudaStream_t stream);

template <typename F, typename A>
NppStatus runSfsI(A* pSrcDst, NppSignalLength nLength, int nScaleFactor, cudaStream_t stream);

template <typename F, typename A, typename D>
NppStatus runF(const A* src, D* dst, NppSignalLength nLength, cudaStream_t stream);

template <typename F, typename A>
NppStatus runFI(A* pSrcDst, NppSignalLength nLength, cudaStream_t stream);

// ---------------------------------------------------------------------------
// Constant-shift entry points (integer only).
// ---------------------------------------------------------------------------
template <typename A>
NppStatus lShiftC(const A* src, int nValue, A* dst, NppSignalLength nLength, cudaStream_t stream);
template <typename A>
NppStatus rShiftC(const A* src, int nValue, A* dst, NppSignalLength nLength, cudaStream_t stream);
template <typename A>
NppStatus lShiftCI(int nValue, A* pSrcDst, NppSignalLength nLength, cudaStream_t stream);
template <typename A>
NppStatus rShiftCI(int nValue, A* pSrcDst, NppSignalLength nLength, cudaStream_t stream);

// ---------------------------------------------------------------------------
// Bitwise binary entry points (And/Or/Xor), out-of-place and in-place.
// ---------------------------------------------------------------------------
template <typename A>
NppStatus bitwiseAnd(const A* a, const A* b, A* d, NppSignalLength nLength, cudaStream_t stream);
template <typename A>
NppStatus bitwiseOr(const A* a, const A* b, A* d, NppSignalLength nLength, cudaStream_t stream);
template <typename A>
NppStatus bitwiseXor(const A* a, const A* b, A* d, NppSignalLength nLength, cudaStream_t stream);
template <typename A>
NppStatus bitwiseAndI(const A* a, A* d, NppSignalLength nLength, cudaStream_t stream);
template <typename A>
NppStatus bitwiseOrI(const A* a, A* d, NppSignalLength nLength, cudaStream_t stream);
template <typename A>
NppStatus bitwiseXorI(const A* a, A* d, NppSignalLength nLength, cudaStream_t stream);

// ---------------------------------------------------------------------------
// Bitwise Not (unary).
// ---------------------------------------------------------------------------
template <typename A>
NppStatus notOp(const A* src, A* dst, NppSignalLength nLength, cudaStream_t stream);
template <typename A>
NppStatus notOpI(A* pSrcDst, NppSignalLength nLength, cudaStream_t stream);

// ---------------------------------------------------------------------------
// nppsMul_Low (32s scaled).
// ---------------------------------------------------------------------------
NppStatus mulLow(const Npp32s* a, const Npp32s* b, Npp32s* d, NppSignalLength nLength, int nScaleFactor, cudaStream_t stream);

// ---------------------------------------------------------------------------
// nppsSumLn reduction.
// ---------------------------------------------------------------------------
template <typename A, typename D>
NppStatus sumLn(const A* src, NppSignalLength nLength, D* dst, Npp8u* pDeviceBuffer, cudaStream_t stream);
template <typename A>
NppStatus sumLnGetBufferSize(NppSignalLength nLength, NPP_BUFFER_SIZE_TYPE* hpBufferSize);

// ---------------------------------------------------------------------------
// nppsCauchy robustness transforms (Npp32f in-place).
// ---------------------------------------------------------------------------
NppStatus cauchy(Npp32f* pSrcDst, NppSignalLength nLength, Npp32f nParam, cudaStream_t stream);
NppStatus cauchyD(Npp32f* pSrcDst, NppSignalLength nLength, Npp32f nParam, cudaStream_t stream);
NppStatus cauchyDD2(Npp32f* pSrcDst, Npp32f* pD2FVal, NppSignalLength nLength, Npp32f nParam, cudaStream_t stream);

}  // namespace unary
}  // namespace npps
