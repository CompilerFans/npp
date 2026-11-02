#pragma once

// Shared 1D reduction executor for the npps statistics families
// (package p2.4: npps_statistics_functions.h).
//
// Host-visible surface (safe for plain .cpp translation units):
//   * accumulator type traits + GetBufferSize helpers (inline, host-only)
//   * executor entry-point templates (declared here, defined in
//     npps_stats_reduce.cu / npps_stats_extrema.cu and explicitly
//     instantiated there).
//
// GPU kernels / functors live under #if defined(__CUDACC__).
//
// Semantics (validated against NVIDIA NPP 13.4 on an RTX 5090 by the
// differential tests in test_npps_statistics_generated.cpp):
//   * Sum / Mean / DotProd / Norm(+Diff) Sfs : divide by 2^nScaleFactor with
//     round-half-to-even (banker's rounding), then saturate.
//   * StdDev / MeanStdDev : SAMPLE normalization (n-1 divisor);
//     n == 1 -> NPP_SIZE_ERROR.  The Sfs std-dev output truncates toward
//     zero while the mean output uses banker's rounding.
//   * Min / Max (+Indx/Abs) : ties resolve to the lowest index; a NaN on
//     the right-hand side of the comparison propagates, a NaN on the
//     left-hand side loses.
//   * MinEvery / MaxEvery : pSrcDst = op(pSrcDst, pSrc) (dst is the left
//     operand); no device buffer is used.
//   * Norm L1 = sum|x| (complex: sum of moduli), L2 = sqrt(sum x^2),
//     L2Sqr = sum x^2, Inf = max|x|.
//   * DotProd complex variants use the PLAIN complex product (no
//     conjugation); mixed real x complex variants multiply the complex
//     value by the real scalar.
//   * AverageError / AverageRelativeError divide the MAXIMUM per-element
//     error by n (measured NVIDIA 13.4 behaviour, not the arithmetic mean);
//     relative error is |a-b| / max(|a|,|b|) with 0 when both are 0.
//   * CountInRange counts elements STRICTLY between the bounds
//     ((x-lo)*(x-hi) < 0, evaluated in 64 bits, order-insensitive).
//   * ZeroCrossing: nppZCR = strict sign changes of non-zero neighbours,
//     nppZCXor = sign-bit flips, nppZCC = 0.5 * sum |sgn3(x[i-1])-sgn3(x[i])|.
//
// Reductions are two-phase: one partial per 256-element block written to
// pDeviceBuffer, then a single-block finalize kernel.  Integer inputs are
// accumulated in long long (real) or a long long pair (complex integer);
// floating inputs accumulate in double.

#include "npp.h"
#include "npp_version_compat.h"
#include "npp_kernel_utils.h"

#include <cuda_runtime.h>
#include <climits>
#include <cmath>
#include <cstdint>
#include <limits>
#include <type_traits>

// Reuse the validated npps elementwise helpers (saturation + Sfs rounding)
// and the complex type traits from the arithmetic executor.
#include "../npps_arithmetic_operations/npps_elementwise_executor.h"

namespace npps {
namespace statistics {

using npps::elementwise::is_complex_v;
using npps::elementwise::component_t_ty;
using npps::elementwise::detail::sat_ll;
using npps::elementwise::detail::scale_sat;

// ---------------------------------------------------------------------------
// Host-safe accumulator traits
// ---------------------------------------------------------------------------

// complex integer pair / complex float pair accumulators
struct LL2 {
  long long re, im;
  __host__ __device__ LL2() : re(0), im(0) {}
  __host__ __device__ LL2(long long r, long long i) : re(r), im(i) {}
};
struct D2 {
  double re, im;
  __host__ __device__ D2() : re(0.0), im(0.0) {}
  __host__ __device__ D2(double r, double i) : re(r), im(i) {}
};
struct MSD {  // mean/std-dev partial: sum and sum of squares
  double sum, sumsq;
  __host__ __device__ MSD() : sum(0.0), sumsq(0.0) {}
  __host__ __device__ MSD(double s, double q) : sum(s), sumsq(q) {}
};

__host__ __device__ inline LL2 operator+(LL2 a, LL2 b) { return LL2(a.re + b.re, a.im + b.im); }
__host__ __device__ inline D2 operator+(D2 a, D2 b) { return D2(a.re + b.re, a.im + b.im); }
__host__ __device__ inline MSD operator+(MSD a, MSD b) { return MSD(a.sum + b.sum, a.sumsq + b.sumsq); }

// Additive accumulator for Sum / Mean style reductions:
//   integral real -> long long, float real -> double,
//   integral complex -> LL2, float complex -> D2.
template <typename T>
struct AddAccT {
  using type = std::conditional_t<
      is_complex_v<T>,
      std::conditional_t<std::is_integral_v<component_t_ty<T>>, LL2, D2>,
      std::conditional_t<std::is_integral_v<T>, long long, double>>;
};
template <typename T> using AddAcc = typename AddAccT<T>::type;

// Additive accumulator for Norm L1/L2/L2Sqr reductions:
//   integral real -> long long, everything else -> double.
template <typename T>
struct NormAccT {
  using type = std::conditional_t<std::is_integral_v<T> && !is_complex_v<T>, long long, double>;
};
template <typename T> using NormAcc = typename NormAccT<T>::type;

// extrema (min/max) partial slot
template <typename T> struct ExtV {
  T v;
  int idx;
  bool valid;
};
template <typename T> struct ExtSlot {
  ExtV<T> vmin, vmax;
};

enum class NormKind { L1, L2, L2Sqr, Inf };
enum class ErrKind { MaxAbs, AvgAbs, MaxRel, AvgRel };

namespace detail {
constexpr int kBlock = 256;

// ---------------------------------------------------------------------------
// Device-side final rounding helpers
//   fin_round : divide by 2^sf, round half-to-even, saturate
//               (Sum/Mean/Norm, and StdDev after its integer sqrt truncation)
//   fin_trunc : divide by 2^sf, truncate toward zero, saturate (kept for the
//               callers that really truncate; StdDev does not use it)
// ---------------------------------------------------------------------------
template <typename D>
__device__ __host__ __forceinline__ D fin_round(double x, int sf) {
  if (sf > 0) x = ldexp(x, -sf);
  if constexpr (std::is_floating_point_v<D>) {
    return static_cast<D>(x);
  } else {
    // Clamp on the double side: out-of-range double -> int64 conversion is
    // undefined and can wrap to INT64_MIN (e.g. overflowing 32s dot products).
    const double lo = static_cast<double>(std::numeric_limits<D>::lowest());
    const double hi = static_cast<double>(std::numeric_limits<D>::max());
    if (x >= hi) return static_cast<D>(std::numeric_limits<D>::max());
    if (x <= lo) return static_cast<D>(std::numeric_limits<D>::lowest());
    return sat_ll<D>(static_cast<long long>(rint(x)));
  }
}

template <typename D>
__device__ __host__ __forceinline__ D fin_trunc(double x, int sf) {
  if (sf > 0) x = ldexp(x, -sf);
  if constexpr (std::is_floating_point_v<D>) {
    return static_cast<D>(x);
  } else {
    return sat_ll<D>(static_cast<long long>(trunc(x)));
  }
}

template <typename Acc> struct is_ll2 : std::false_type {};
template <> struct is_ll2<LL2> : std::true_type {};

}  // namespace detail

// ---------------------------------------------------------------------------
// Executor entry points (defined in the .cu translation units)
// ---------------------------------------------------------------------------

// Sum / Mean / StdDev / MeanStdDev.  nScaleFactor is 0 for the non-Sfs
// variants.  StdDev/MeanStdDev reject nLength == 1 (sample normalization).
template <typename T, typename O>
NppStatus sum(const T* pSrc, NppSignalLength nLength, O* pSum, int nScaleFactor,
              Npp8u* pDeviceBuffer, cudaStream_t stream);

template <typename T, typename O>
NppStatus mean(const T* pSrc, NppSignalLength nLength, O* pMean, int nScaleFactor,
               Npp8u* pDeviceBuffer, cudaStream_t stream);

template <typename T, typename O>
NppStatus stddev(const T* pSrc, NppSignalLength nLength, O* pStdDev, int nScaleFactor,
                 Npp8u* pDeviceBuffer, cudaStream_t stream);

template <typename T, typename OM, typename OS>
NppStatus mean_stddev(const T* pSrc, NppSignalLength nLength, OM* pMean, OS* pStdDev,
                      int nScaleFactor, Npp8u* pDeviceBuffer, cudaStream_t stream);

// Min/Max family.  One kernel computes both extrema; the want* flags say
// which outputs this particular API function has (they drive validation:
// NVIDIA rejects a null pointer for every output the function declares).
// AbsOnly implements MinAbs/MaxAbs (comparator on |x|, output |x|).
template <typename T, bool AbsOnly>
NppStatus extrema(const T* pSrc, NppSignalLength nLength, T* pMin, int* pMinIndx, T* pMax,
                  int* pMaxIndx, bool wantMin, bool wantMax, bool wantIndx,
                  Npp8u* pDeviceBuffer, cudaStream_t stream);

// MinEvery / MaxEvery : pSrcDst = op(pSrcDst, pSrc) (no device buffer).
template <typename T, bool IsMax>
NppStatus every(const T* pSrc, T* pSrcDst, NppSignalLength nLength, cudaStream_t stream);

// Norm / NormDiff of a 1D signal.
template <typename T, NormKind K, typename O>
NppStatus norm(const T* pSrc, NppSignalLength nLength, O* pNorm, int nScaleFactor,
               Npp8u* pDeviceBuffer, cudaStream_t stream);

template <typename T, NormKind K, typename O>
NppStatus normdiff(const T* pSrc1, const T* pSrc2, NppSignalLength nLength, O* pNorm,
                   int nScaleFactor, Npp8u* pDeviceBuffer, cudaStream_t stream);

// Dot product.  Complex variants use the plain (non-conjugated) product;
// mixed real x complex variants scale the complex value by the real input.
template <typename A, typename B, typename O>
NppStatus dotprod(const A* pSrc1, const B* pSrc2, NppSignalLength nLength, O* pDp,
                  int nScaleFactor, Npp8u* pDeviceBuffer, cudaStream_t stream);

// Error metrics (all output Npp64f).  Per measured NPP 13.4 behaviour the
// "average" variants divide the maximum per-element error by n.
template <typename T, ErrKind K>
NppStatus error_metric(const T* pSrc1, const T* pSrc2, NppSignalLength nLength, Npp64f* pDst,
                       Npp8u* pDeviceBuffer, cudaStream_t stream);

// Count elements strictly between nLowerBound and nUpperBound.
NppStatus count_in_range(const Npp32s* pSrc, NppSignalLength nLength, int* pCounts,
                         Npp32s nLowerBound, Npp32s nUpperBound, Npp8u* pDeviceBuffer,
                         cudaStream_t stream);

// Zero crossing measures (output Npp32f for both input types).
template <typename T>
NppStatus zero_crossing(const T* pSrc, NppSignalLength nLength, Npp32f* pValZC,
                        NppsZCType tZCType, Npp8u* pDeviceBuffer, cudaStream_t stream);

// ---------------------------------------------------------------------------
// GetBufferSize helpers (host-only, inline).  NVIDIA never validates
// nLength here (any value returns NPP_NO_ERROR); only a null hpBufferSize
// pointer is rejected.  The byte count itself is an implementation detail
// (it only has to be large enough for the executor's partials).
// ---------------------------------------------------------------------------
namespace detail {

inline NppStatus buffer_size_bytes(size_t slot_bytes, int nLength, int* hpBufferSize) {
  if (!hpBufferSize) return NPP_NULL_POINTER_ERROR;
  // NVIDIA never validates the length here: negative values simply become a
  // huge element count and the size saturates.
  const size_t n = static_cast<size_t>(nLength);
  size_t blocks;
  if (n == 0) {
    blocks = 0;
  } else if (n > (SIZE_MAX - static_cast<size_t>(kBlock - 1)) / static_cast<size_t>(kBlock)) {
    blocks = SIZE_MAX / static_cast<size_t>(kBlock);  // overflow guard
  } else {
    blocks = (n + static_cast<size_t>(kBlock - 1)) / static_cast<size_t>(kBlock);
  }
  size_t bytes = (blocks > SIZE_MAX / slot_bytes) ? SIZE_MAX : blocks * slot_bytes;
  if (bytes > static_cast<size_t>(INT_MAX)) bytes = static_cast<size_t>(INT_MAX);
  *hpBufferSize = static_cast<int>(bytes);
  return NPP_NO_ERROR;
}

}  // namespace detail

template <typename T>
inline NppStatus sum_buffer_size(int nLength, int* hpBufferSize) {
  return detail::buffer_size_bytes(sizeof(AddAcc<T>), nLength, hpBufferSize);
}

template <typename T>
inline NppStatus mean_buffer_size(int nLength, int* hpBufferSize) {
  return detail::buffer_size_bytes(sizeof(AddAcc<T>), nLength, hpBufferSize);
}

template <typename T>
inline NppStatus stddev_buffer_size(int nLength, int* hpBufferSize) {
  return detail::buffer_size_bytes(sizeof(MSD), nLength, hpBufferSize);
}

template <typename T>
inline NppStatus extrema_buffer_size(int nLength, int* hpBufferSize) {
  return detail::buffer_size_bytes(sizeof(ExtSlot<T>), nLength, hpBufferSize);
}

template <typename T, NormKind K>
inline NppStatus norm_buffer_size(int nLength, int* hpBufferSize) {
  const size_t slot = (K == NormKind::Inf) ? sizeof(double) : sizeof(NormAcc<T>);
  return detail::buffer_size_bytes(slot, nLength, hpBufferSize);
}

// DotProd partials are always a double2 pair so one helper serves every
// mixed-type variant.
inline NppStatus dotprod_buffer_size(int nLength, int* hpBufferSize) {
  return detail::buffer_size_bytes(sizeof(D2), nLength, hpBufferSize);
}

inline NppStatus err_buffer_size(int nLength, int* hpBufferSize) {
  return detail::buffer_size_bytes(sizeof(double), nLength, hpBufferSize);
}

inline NppStatus count_buffer_size(int nLength, int* hpBufferSize) {
  return detail::buffer_size_bytes(sizeof(int), nLength, hpBufferSize);
}

inline NppStatus zc_buffer_size(int nLength, int* hpBufferSize) {
  return detail::buffer_size_bytes(sizeof(double), nLength, hpBufferSize);
}

#if defined(__CUDACC__)

// ---------------------------------------------------------------------------
// Device functors
// ---------------------------------------------------------------------------
namespace detail {

using npps::elementwise::detail::re_ll;
using npps::elementwise::detail::im_ll;

// ---- finalizers (run in the single-block finalize kernel) ----------------

template <typename O>
struct FinSum {
  O* out;
  int sf;
  template <typename Acc>
  __device__ void operator()(const Acc& v, size_t) const {
    if constexpr (is_complex_v<O>) {
      using C = component_t_ty<O>;
      if constexpr (is_ll2<Acc>::value) {
        out->re = scale_sat<C>(v.re, sf);
        out->im = scale_sat<C>(v.im, sf);
      } else {
        out->re = fin_round<C>(v.re, sf);
        out->im = fin_round<C>(v.im, sf);
      }
    } else {
      if constexpr (std::is_same_v<Acc, long long>) {
        *out = scale_sat<O>(v, sf);
      } else {
        *out = fin_round<O>(static_cast<double>(v), sf);
      }
    }
  }
};

template <typename O>
struct FinMean {
  O* out;
  int sf;
  template <typename Acc>
  __device__ void operator()(const Acc& v, size_t n) const {
    const double dn = static_cast<double>(n);
    if constexpr (is_complex_v<O>) {
      using C = component_t_ty<O>;
      out->re = fin_round<C>(static_cast<double>(v.re) / dn, sf);
      out->im = fin_round<C>(static_cast<double>(v.im) / dn, sf);
    } else if constexpr (std::is_floating_point_v<O>) {
      *out = fin_round<O>(static_cast<double>(v) / dn, sf);
    } else {
      // NPP rounds the real mean to an integer first (half-to-even), then
      // applies the Sfs half-to-even rounding as a second step.
      const long long m = static_cast<long long>(rint(static_cast<double>(v) / dn));
      *out = scale_sat<O>(m, sf);
    }
  }
};

template <typename O>
struct FinStd {
  O* out;
  int sf;
  __device__ void operator()(const MSD& v, size_t n) const {
    double var = (v.sumsq - v.sum * v.sum / static_cast<double>(n)) / static_cast<double>(n - 1);
    if (!(var > 0.0)) var = 0.0;  // clamp cancellation noise (and NaN)
    if constexpr (std::is_floating_point_v<O>) {
      *out = fin_round<O>(sqrt(var), sf);
    } else {
      // NPP's integer path uses integer variance arithmetic:
      //   num = sumsq - (sum*sum)/n; var = num/(n-1), then trunc(sqrt(var)).
      const long long S = static_cast<long long>(v.sum);
      const long long Q = static_cast<long long>(v.sumsq);
      long long num = Q - (S * S) / static_cast<long long>(n);
      long long ivar = num / static_cast<long long>(n - 1);
      if (ivar < 0) ivar = 0;
      *out = scale_sat<O>(static_cast<long long>(sqrt(static_cast<double>(ivar))), sf);
    }
  }
};

template <typename OM, typename OS>
struct FinMeanStd {
  OM* pMean;
  OS* pStd;
  int sf;
  __device__ void operator()(const MSD& v, size_t n) const {
    const double dn = static_cast<double>(n);
    if constexpr (std::is_floating_point_v<OM>) {
      *pMean = fin_round<OM>(v.sum / dn, sf);
    } else {
      // Integer means round the real mean first, then apply Sfs scaling.
      const long long m = static_cast<long long>(rint(v.sum / dn));
      *pMean = scale_sat<OM>(m, sf);
    }
    double var = (v.sumsq - v.sum * v.sum / dn) / static_cast<double>(n - 1);
    if (!(var > 0.0)) var = 0.0;
    if constexpr (std::is_floating_point_v<OS>) {
      *pStd = fin_round<OS>(sqrt(var), sf);
    } else {
      const long long S = static_cast<long long>(v.sum);
      const long long Q = static_cast<long long>(v.sumsq);
      long long num = Q - (S * S) / static_cast<long long>(n);
      long long ivar = num / static_cast<long long>(n - 1);
      if (ivar < 0) ivar = 0;
      *pStd = scale_sat<OS>(static_cast<long long>(sqrt(static_cast<double>(ivar))), sf);
    }
  }
};

template <typename O, NormKind K>
struct FinNorm {
  O* out;
  int sf;
  template <typename Acc>
  __device__ void operator()(const Acc& v, size_t) const {
    if constexpr (K == NormKind::L2) {
      if constexpr (std::is_integral_v<O>) {
        // NPP rounds the integer square root to the nearest even integer
        // first, then applies the Sfs half-even scale.
        const long long q = static_cast<long long>(rint(sqrt(static_cast<double>(v))));
        *out = scale_sat<O>(q, sf);
      } else {
        *out = fin_round<O>(sqrt(static_cast<double>(v)), sf);
      }
    } else {
      if constexpr (std::is_same_v<Acc, long long>) {
        *out = scale_sat<O>(v, sf);
      } else {
        *out = fin_round<O>(static_cast<double>(v), sf);
      }
    }
  }
};

template <typename O>
struct FinInf {  // finalize of the max-|x| (Norm Inf) reduction
  O* out;
  int sf;
  __device__ void operator()(double v, size_t) const { *out = fin_round<O>(v, sf); }
};

template <typename O>
struct FinDot {
  O* out;
  int sf;
  __device__ void operator()(const D2& v, size_t) const {
    if constexpr (is_complex_v<O>) {
      using C = component_t_ty<O>;
      out->re = fin_round<C>(v.re, sf);
      out->im = fin_round<C>(v.im, sf);
    } else {
      *out = fin_round<O>(v.re, sf);
    }
  }
};

struct FinErr {
  Npp64f* out;
  bool avg;
  __device__ void operator()(double v, size_t n) const {
    *out = avg ? v / static_cast<double>(n) : v;
  }
};

struct FinCount {
  int* out;
  __device__ void operator()(int v, size_t) const { *out = v; }
};

struct FinZC {
  Npp32f* out;
  __device__ void operator()(double v, size_t) const { *out = static_cast<Npp32f>(v); }
};

template <typename T>
struct FinExt {
  T* pMin;
  int* pMinIndx;
  T* pMax;
  int* pMaxIndx;
  __device__ void operator()(const ExtSlot<T>& r, size_t) const {
    if (pMin) *pMin = r.vmin.v;
    if (pMinIndx) *pMinIndx = r.vmin.idx;
    if (pMax) *pMax = r.vmax.v;
    if (pMaxIndx) *pMaxIndx = r.vmax.idx;
  }
};

// ---- loaders (per-element contribution) -----------------------------------

template <typename T>
struct LoadSum {
  using Acc = AddAcc<T>;
  __device__ Acc operator()(const T& x) const {
    if constexpr (is_complex_v<T>) {
      if constexpr (std::is_integral_v<component_t_ty<T>>) {
        return Acc(re_ll(x), im_ll(x));
      } else {
        return Acc(static_cast<double>(x.re), static_cast<double>(x.im));
      }
    } else {
      if constexpr (std::is_integral_v<T>) {
        return static_cast<Acc>(re_ll(x));
      } else {
        return static_cast<Acc>(static_cast<double>(x));
      }
    }
  }
};

template <typename T>  // T real (StdDev / MeanStdDev inputs are real)
struct LoadMSD {
  __device__ MSD operator()(const T& x) const {
    const double d = static_cast<double>(x);
    return MSD(d, d * d);
  }
};

template <typename T, NormKind K>  // K != Inf
struct LoadNorm {
  using Acc = NormAcc<T>;
  __device__ Acc operator()(const T& x) const {
    if constexpr (is_complex_v<T>) {
      const double re = static_cast<double>(x.re), im = static_cast<double>(x.im);
      const double m2 = re * re + im * im;
      if constexpr (K == NormKind::L1) {
        return static_cast<Acc>(sqrt(m2));
      } else {
        return static_cast<Acc>(m2);
      }
    } else if constexpr (std::is_integral_v<T>) {
      const long long v = re_ll(x);
      if constexpr (K == NormKind::L1) {
        return static_cast<Acc>(v < 0 ? -v : v);
      } else {
        return static_cast<Acc>(v * v);
      }
    } else {
      const double v = static_cast<double>(x);
      if constexpr (K == NormKind::L1) {
        return static_cast<Acc>(fabs(v));
      } else {
        return static_cast<Acc>(v * v);
      }
    }
  }
};

template <typename T, NormKind K>  // K != Inf
struct LoadNormDiff {
  using Acc = NormAcc<T>;
  __device__ Acc operator()(const T& a, const T& b) const {
    if constexpr (is_complex_v<T>) {
      const double dre = static_cast<double>(a.re) - static_cast<double>(b.re);
      const double dim = static_cast<double>(a.im) - static_cast<double>(b.im);
      const double m2 = dre * dre + dim * dim;
      if constexpr (K == NormKind::L1) {
        return static_cast<Acc>(sqrt(m2));
      } else {
        return static_cast<Acc>(m2);
      }
    } else if constexpr (std::is_integral_v<T>) {
      const long long d = re_ll(a) - re_ll(b);
      if constexpr (K == NormKind::L1) {
        return static_cast<Acc>(d < 0 ? -d : d);
      } else {
        return static_cast<Acc>(d * d);
      }
    } else {
      const double d = static_cast<double>(a) - static_cast<double>(b);
      if constexpr (K == NormKind::L1) {
        return static_cast<Acc>(fabs(d));
      } else {
        return static_cast<Acc>(d * d);
      }
    }
  }
};

template <typename T>
struct LoadNormInf {
  __device__ double operator()(const T& x) const {
    if constexpr (is_complex_v<T>) {
      const double re = static_cast<double>(x.re), im = static_cast<double>(x.im);
      return sqrt(re * re + im * im);
    } else {
      return fabs(static_cast<double>(x));
    }
  }
};

template <typename T>
struct LoadNormDiffInf {
  __device__ double operator()(const T& a, const T& b) const {
    if constexpr (is_complex_v<T>) {
      const double dre = static_cast<double>(a.re) - static_cast<double>(b.re);
      const double dim = static_cast<double>(a.im) - static_cast<double>(b.im);
      return sqrt(dre * dre + dim * dim);
    } else {
      return fabs(static_cast<double>(a) - static_cast<double>(b));
    }
  }
};

// Dot product contribution, always accumulated as a double pair so that a
// single buffer-size helper serves every mixed-type variant.
template <typename A, typename B>
struct LoadDot {
  __device__ D2 operator()(const A& a, const B& b) const {
    if constexpr (is_complex_v<A> && is_complex_v<B>) {
      const double ar = static_cast<double>(a.re), ai = static_cast<double>(a.im);
      const double br = static_cast<double>(b.re), bi = static_cast<double>(b.im);
      return D2(ar * br - ai * bi, ar * bi + ai * br);
    } else if constexpr (is_complex_v<B>) {  // real x complex
      const double av = static_cast<double>(a);
      return D2(av * static_cast<double>(b.re), av * static_cast<double>(b.im));
    } else if constexpr (is_complex_v<A>) {  // complex x real (not in NPP, kept for symmetry)
      const double bv = static_cast<double>(b);
      return D2(bv * static_cast<double>(a.re), bv * static_cast<double>(a.im));
    } else {
      return D2(static_cast<double>(a) * static_cast<double>(b), 0.0);
    }
  }
};

// Error metric contribution (max-reduced).  MaxAbs/AvgAbs use |a-b|;
// MaxRel/AvgRel use |a-b| / max(a,b), i.e. the SIGNED maximum of the two
// operands, which is what NPP 13.4 actually computes (the documented
// max(|a|,|b|) is wrong for real types; measured on the RTX 5090).  A
// non-positive denominator contributes 0 (probe: (-46,0), (-5,-3) and
// all-negative inputs all return 0).  Complex variants work on moduli,
// where |z| >= 0 makes the signed/absolute distinction moot.
template <typename T, ErrKind K>
struct LoadErr {
  static constexpr bool kRel = (K == ErrKind::MaxRel || K == ErrKind::AvgRel);
  __device__ double operator()(const T& a, const T& b) const {
    if constexpr (is_complex_v<T>) {
      const double dre = static_cast<double>(a.re) - static_cast<double>(b.re);
      const double dim = static_cast<double>(a.im) - static_cast<double>(b.im);
      const double d = sqrt(dre * dre + dim * dim);
      if constexpr (kRel) {
        const double ar = static_cast<double>(a.re), ai = static_cast<double>(a.im);
        const double br = static_cast<double>(b.re), bi = static_cast<double>(b.im);
        const double ma = sqrt(ar * ar + ai * ai);
        const double mb = sqrt(br * br + bi * bi);
        const double den = ma > mb ? ma : mb;
        return den > 0.0 ? d / den : 0.0;
      } else {
        return d;
      }
    } else {
      const double da = static_cast<double>(a);
      const double db = static_cast<double>(b);
      const double d = fabs(da - db);
      if constexpr (kRel) {
        const double den = da > db ? da : db;  // signed max, per NPP 13.4
        return den > 0.0 ? d / den : 0.0;
      } else {
        return d;
      }
    }
  }
};

// ---- extrema combiners -----------------------------------------------------

// Measured NPP 13.4 comparator: strict comparison, ties to the lower index,
// and a NaN comparison failure always favours the RIGHT operand.
template <typename T>
__device__ __forceinline__ ExtV<T> ext_min(ExtV<T> a, ExtV<T> b) {
  if (!b.valid) return a;
  if (!a.valid) return b;
  if (a.v < b.v) return a;
  if (a.v == b.v) return a.idx < b.idx ? a : b;
  return b;
}

template <typename T>
__device__ __forceinline__ ExtV<T> ext_max(ExtV<T> a, ExtV<T> b) {
  if (!b.valid) return a;
  if (!a.valid) return b;
  if (a.v > b.v) return a;
  if (a.v == b.v) return a.idx < b.idx ? a : b;
  return b;
}

template <typename T>
__device__ __forceinline__ ExtSlot<T> ext_combine(ExtSlot<T> a, ExtSlot<T> b) {
  ExtSlot<T> r;
  r.vmin = ext_min(a.vmin, b.vmin);
  r.vmax = ext_max(a.vmax, b.vmax);
  return r;
}

// ---------------------------------------------------------------------------
// Kernels
// ---------------------------------------------------------------------------

// partial: one additive accumulator per block (single input)
template <typename A, typename Acc, typename Load>
static __global__ void partial_add1_kernel(const A* src, size_t n, Acc* partials, Load load) {
  __shared__ Acc sdata[kBlock];
  const unsigned tid = threadIdx.x;
  const size_t i = static_cast<size_t>(blockIdx.x) * kBlock + tid;
  Acc v{};
  if (i < n) v = load(src[i]);
  sdata[tid] = v;
  __syncthreads();
  for (unsigned s = kBlock / 2; s > 0; s >>= 1) {
    if (tid < s) sdata[tid] = sdata[tid] + sdata[tid + s];
    __syncthreads();
  }
  if (tid == 0) partials[blockIdx.x] = sdata[0];
}

// partial: one additive accumulator per block (two inputs)
template <typename A, typename B, typename Acc, typename Load>
static __global__ void partial_add2_kernel(const A* a, const B* b, size_t n, Acc* partials, Load load) {
  __shared__ Acc sdata[kBlock];
  const unsigned tid = threadIdx.x;
  const size_t i = static_cast<size_t>(blockIdx.x) * kBlock + tid;
  Acc v{};
  if (i < n) v = load(a[i], b[i]);
  sdata[tid] = v;
  __syncthreads();
  for (unsigned s = kBlock / 2; s > 0; s >>= 1) {
    if (tid < s) sdata[tid] = sdata[tid] + sdata[tid + s];
    __syncthreads();
  }
  if (tid == 0) partials[blockIdx.x] = sdata[0];
}

// partial: block maximum of a per-element double (single input)
template <typename A, typename Load>
static __global__ void partial_max1_kernel(const A* a, size_t n, double* partials, Load load) {
  __shared__ double sdata[kBlock];
  const unsigned tid = threadIdx.x;
  const size_t i = static_cast<size_t>(blockIdx.x) * kBlock + tid;
  double v = -INFINITY;
  if (i < n) v = load(a[i]);
  sdata[tid] = v;
  __syncthreads();
  for (unsigned s = kBlock / 2; s > 0; s >>= 1) {
    if (tid < s) sdata[tid] = sdata[tid] > sdata[tid + s] ? sdata[tid] : sdata[tid + s];
    __syncthreads();
  }
  if (tid == 0) partials[blockIdx.x] = sdata[0];
}

// partial: block maximum of a per-element double (two inputs)
template <typename A, typename B, typename Load>
static __global__ void partial_max2_kernel(const A* a, const B* b, size_t n, double* partials, Load load) {
  __shared__ double sdata[kBlock];
  const unsigned tid = threadIdx.x;
  const size_t i = static_cast<size_t>(blockIdx.x) * kBlock + tid;
  double v = -INFINITY;
  if (i < n) v = load(a[i], b[i]);
  sdata[tid] = v;
  __syncthreads();
  for (unsigned s = kBlock / 2; s > 0; s >>= 1) {
    if (tid < s) sdata[tid] = sdata[tid] > sdata[tid + s] ? sdata[tid] : sdata[tid + s];
    __syncthreads();
  }
  if (tid == 0) partials[blockIdx.x] = sdata[0];
}

// partial: per-block count of elements strictly inside (lo, hi)
static __global__ void count_partial_kernel(const Npp32s* src, size_t n, Npp32s lo, Npp32s hi,
                                     int* partials) {
  __shared__ int sdata[kBlock];
  const unsigned tid = threadIdx.x;
  const size_t i = static_cast<size_t>(blockIdx.x) * kBlock + tid;
  int c = 0;
  if (i < n) {
    // NPP counts the strict open interval lo < x < hi (probed on the 5090:
    // {10,11,19,20,21} with lo=10/hi=20 -> 2), so an inverted range counts 0.
    c = (src[i] > lo && src[i] < hi) ? 1 : 0;
  }
  sdata[tid] = c;
  __syncthreads();
  for (unsigned s = kBlock / 2; s > 0; s >>= 1) {
    if (tid < s) sdata[tid] += sdata[tid + s];
    __syncthreads();
  }
  if (tid == 0) partials[blockIdx.x] = sdata[0];
}

// partial: per-block min/max (optionally on |x|)
template <typename T, bool AbsOnly>
static __global__ void partial_ext_kernel(const T* src, size_t n, ExtSlot<T>* partials) {
  __shared__ ExtSlot<T> sdata[kBlock];
  const unsigned tid = threadIdx.x;
  const size_t i = static_cast<size_t>(blockIdx.x) * kBlock + tid;
  ExtSlot<T> v{};
  if (i < n) {
    T x = src[i];
    if constexpr (AbsOnly && std::is_integral_v<T>) {
      if constexpr (std::is_signed_v<T>) {
        using L = std::numeric_limits<T>;
        if (x == L::lowest()) x = L::max();       // NPP saturates |min|
        else if (x < 0) x = static_cast<T>(-x);
      }
    }
    if constexpr (AbsOnly && !std::is_integral_v<T>) x = fabs(x);
    v.vmin = ExtV<T>{x, static_cast<int>(i), true};
    v.vmax = ExtV<T>{x, static_cast<int>(i), true};
  }
  sdata[tid] = v;
  __syncthreads();
  for (unsigned s = kBlock / 2; s > 0; s >>= 1) {
    if (tid < s) sdata[tid] = ext_combine(sdata[tid], sdata[tid + s]);
    __syncthreads();
  }
  if (tid == 0) partials[blockIdx.x] = sdata[0];
}

// partial: per-block zero-crossing contribution sum.  Each block owns the
// pairs (i-1, i) for i in [base, base+kBlock); the pair crossing the block
// boundary is handled by loading src[base-1] into the extra shared slot.
template <typename T>
__device__ __forceinline__ double zc_pair(const T& prev, const T& cur, NppsZCType mode) {
  switch (mode) {
    case nppZCR: {
      const bool up = prev > 0 && cur < 0;
      const bool dn = prev < 0 && cur > 0;
      return (up || dn) ? 1.0 : 0.0;
    }
    case nppZCXor: {
      // NPP compares against zero (it does not use signbit), so -0.0 and
      // +0.0 are treated identically.
      return ((prev < T(0)) != (cur < T(0))) ? 1.0 : 0.0;
    }
    default: {  // nppZCC : 0.5 * |sgn3(prev) - sgn3(cur)|
      const int sp = (prev > 0 ? 1 : 0) - (prev < 0 ? 1 : 0);
      const int sc = (cur > 0 ? 1 : 0) - (cur < 0 ? 1 : 0);
      return 0.5 * static_cast<double>(sp > sc ? sp - sc : sc - sp);
    }
  }
}

template <typename T>
static __global__ void zc_partial_kernel(const T* src, size_t n, NppsZCType mode, double* partials) {
  __shared__ T sx[kBlock + 1];
  const unsigned tid = threadIdx.x;
  const size_t base = static_cast<size_t>(blockIdx.x) * kBlock;
  const size_t i = base + tid;
  if (i < n) sx[tid] = src[i];
  if (tid == 0 && base > 0) sx[kBlock] = src[base - 1];
  __syncthreads();
  double c = 0.0;
  if (i >= 1 && i < n) {
    const T cur = sx[tid];
    const T prev = (tid == 0) ? sx[kBlock] : sx[tid - 1];
    c = zc_pair(prev, cur, mode);
  }
  __shared__ double sdata[kBlock];
  sdata[tid] = c;
  __syncthreads();
  for (unsigned s = kBlock / 2; s > 0; s >>= 1) {
    if (tid < s) sdata[tid] += sdata[tid + s];
    __syncthreads();
  }
  if (tid == 0) partials[blockIdx.x] = sdata[0];
}

// elementwise MinEvery / MaxEvery : pSrcDst = op(pSrcDst, pSrc)
template <typename T, bool IsMax>
static __global__ void every_kernel(const T* src, T* d, size_t n) {
  const size_t i = static_cast<size_t>(blockIdx.x) * kBlock + threadIdx.x;
  if (i >= n) return;
  const T a = d[i];
  const T b = src[i];
  if constexpr (IsMax) {
    d[i] = a > b ? a : b;
  } else {
    d[i] = a < b ? a : b;
  }
}

// finalize: sum the partials in one block, then run the finalizer functor
template <typename Acc, typename Fin>
static __global__ void finalize_add_kernel(const Acc* partials, size_t nblocks, size_t n, Fin fin) {
  __shared__ Acc sdata[kBlock];
  const unsigned tid = threadIdx.x;
  Acc v{};
  for (size_t i = tid; i < nblocks; i += kBlock) v = v + partials[i];
  sdata[tid] = v;
  __syncthreads();
  for (unsigned s = kBlock / 2; s > 0; s >>= 1) {
    if (tid < s) sdata[tid] = sdata[tid] + sdata[tid + s];
    __syncthreads();
  }
  if (tid == 0) fin(sdata[0], n);
}

// finalize: max of the double partials, then run the finalizer functor
template <typename Fin>
static __global__ void finalize_max_kernel(const double* partials, size_t nblocks, size_t n, Fin fin) {
  __shared__ double sdata[kBlock];
  const unsigned tid = threadIdx.x;
  double v = -INFINITY;
  for (size_t i = tid; i < nblocks; i += kBlock) v = v > partials[i] ? v : partials[i];
  sdata[tid] = v;
  __syncthreads();
  for (unsigned s = kBlock / 2; s > 0; s >>= 1) {
    if (tid < s) sdata[tid] = sdata[tid] > sdata[tid + s] ? sdata[tid] : sdata[tid + s];
    __syncthreads();
  }
  if (tid == 0) fin(sdata[0], n);
}

// finalize: combine the extrema partials, then write the requested outputs
template <typename T, typename Fin>
static __global__ void finalize_ext_kernel(const ExtSlot<T>* partials, size_t nblocks, Fin fin) {
  __shared__ ExtSlot<T> sdata[kBlock];
  const unsigned tid = threadIdx.x;
  ExtSlot<T> v{};
  for (size_t i = tid; i < nblocks; i += kBlock) v = ext_combine(v, partials[i]);
  sdata[tid] = v;
  __syncthreads();
  for (unsigned s = kBlock / 2; s > 0; s >>= 1) {
    if (tid < s) sdata[tid] = ext_combine(sdata[tid], sdata[tid + s]);
    __syncthreads();
  }
  if (tid == 0) fin(sdata[0], 0);
}

}  // namespace detail
#endif  // __CUDACC__

}  // namespace statistics
}  // namespace npps
