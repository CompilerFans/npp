#pragma once

// Shared 1D executor for the npps scalar / bitwise / shift / reduction
// arithmetic families implemented by package p1.1c:
//   nppsSqrt / nppsSqr / nppsExp / nppsLn / npps10Log10 / nppsAbs / nppsArctan
//   / nppsCubrt  (scalar unary, scaled and plain)
//   nppsAnd / nppsOr / nppsXor / nppsNot (bitwise)
//   nppsLShiftC / nppsRShiftC (constant shift)
//   nppsMul_Low
//   nppsSumLn / nppsSumLnGetBufferSize (reduction)
//   nppsCauchy / nppsCauchyD / nppsCauchyDD2 (in-place distribution transform)
//
// Host-visible surface (safe for plain .cpp translation units): executor
// entry-point templates are declared here and defined + explicitly
// instantiated in npps_unary_ops.cu.
//
// Semantics (validated against NVIDIA NPP 13.4 by differential tests):
//   * scaled integer unary : d[i] = F::apply(src[i], sf)
//   * plain float unary    : d[i] = F::apply(src[i]).
//   * in-place (_I/_ISfs)  : pSrcDst = F(pSrcDst).
//   * bitwise And/Or/Xor   : pDst = pSrc1 op pSrc2 (integer only)
//   * nppsNot              : pDst = ~pSrc
//   * nppsLShiftC/RShiftC  : pDst = pSrc << nValue | pSrc >> nValue
//   * nppsMul_Low_32s_Sfs  : low 32 bits of the 64-bit product, then scaled
//   * nppsSumLn            : scalar sum of natural logarithms

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
// fixed-point scaling/rounding step is integer-based.  Differential tests use
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
      double re = rp(a), im = ip(a);
      double r = std::hypot(re, im);
      double gamma = std::sqrt((r + re) / 2.0);
      double delta = std::sqrt((r - re) / 2.0);
      D d; d.re = (C)gamma; d.im = (C)(im < 0 ? -delta : delta); return d;
    } else return (D)std::sqrt(rp(a));
  }
  template <typename A, typename D> __device__ __host__ D applyS(const A& a, int sf) const {
    using C = ed::component_t_ty<D>;
    if constexpr (ed::is_complex_v<D>) {
      // NPP 13.4 truncates (floor) the scaled components of the complex sqrt,
      // rather than round-half-even (validated against goldens).
      double re = rp(a), im = ip(a);
      double r = std::hypot(re, im);
      double gamma = std::sqrt((r + re) / 2.0), delta = std::sqrt((r - re) / 2.0);
      double gr = (im < 0 ? -delta : gamma), gi = (im < 0 ? gamma : delta);
      double s = (1ll << sf);
      D d; d.re = (C)(gr / s); d.im = (C)(gi / s); return d;
    } else {
      double x = rp(a);
      if (x < 0) {
        // NPP 13.4: 16s clamps to 0, 64s saturates to INT64_MIN.
        if constexpr (std::is_same_v<D, Npp64s>) return (D)(-9223372036854775807LL - 1);
        else return (D)0;
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
      return ek::scale_sat<D>(v, sf);
    }
  }
};

struct ExpOp {
  template <typename A, typename D> __device__ __host__ D applyF(const A& a) const { return (D)std::exp(rp(a)); }
  template <typename A, typename D> __device__ __host__ D applyS(const A& a, int sf) const {
    // NPP evaluates integer Exp in single precision (validated on 13.4:
    // expf(28)/2^16 == 22068132, while double gives ...131).
    if constexpr (sizeof(D) <= 4) return ssc<D>((double)expf((float)rp(a)), sf);
    else return ssc<D>(std::exp(rp(a)), sf);
  }
};

struct LnOp {
  template <typename A, typename D> __device__ __host__ D applyF(const A& a) const { return (D)std::log(rp(a)); }
  template <typename A, typename D> __device__ __host__ D applyS(const A& a, int sf) const {
    return ssc<D>(std::log(rp(a)), sf);
  }
};

struct Log10Op {
  template <typename A, typename D> __device__ __host__ D applyF(const A& a) const {
    return (D)(std::log(rp(a)) / std::log(10.0));
  }
  template <typename A, typename D> __device__ __host__ D applyS(const A& a, int sf) const {
    return ssc<D>(std::log(rp(a)) / std::log(10.0), sf);
  }
};

struct AbsOp {
  template <typename A, typename D> __device__ __host__ D applyF(const A& a) const { return (D)std::fabs(rp(a)); }
  template <typename A, typename D> __device__ __host__ D applyS(const A& a, int sf) const {
    long long v = ek::re_ll(a); (void)sf; return (D)(v < 0 ? -v : v);
  }
};

struct ArctanOp {
  template <typename A, typename D> __device__ __host__ D applyF(const A& a) const { return (D)std::atan(rp(a)); }
  template <typename A, typename D> __device__ __host__ D applyS(const A& a, int sf) const { (void)sf; return applyF<A, D>(a); }
};

struct CubrtOp {
  template <typename A, typename D> __device__ __host__ D applyF(const A& a) const {
    double x = rp(a);
    return (D)(x < 0 ? -std::cbrt(-x) : std::cbrt(x));
  }
  template <typename A, typename D> __device__ __host__ D applyS(const A& a, int sf) const {
    double x = rp(a);
    double cb = x < 0 ? -std::cbrt(-x) : std::cbrt(x);
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
NppStatus sumLnGetBufferSize(NppSignalLength nLength, int* hpBufferSize);

// ---------------------------------------------------------------------------
// nppsCauchy robustness transforms (Npp32f in-place).
// ---------------------------------------------------------------------------
NppStatus cauchy(Npp32f* pSrcDst, NppSignalLength nLength, Npp32f nParam, cudaStream_t stream);
NppStatus cauchyD(Npp32f* pSrcDst, NppSignalLength nLength, Npp32f nParam, cudaStream_t stream);
NppStatus cauchyDD2(Npp32f* pSrcDst, Npp32f* pD2FVal, NppSignalLength nLength, Npp32f nParam, cudaStream_t stream);

}  // namespace unary
}  // namespace npps
