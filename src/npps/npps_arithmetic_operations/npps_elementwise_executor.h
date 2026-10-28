#pragma once

// Shared 1D element-wise executor for the npps arithmetic families
// (nppsAdd / nppsSub / nppsMul / nppsDiv / nppsAddProduct / nppsDiv_Round).
//
// Host-visible surface (safe for plain .cpp translation units):
//   * type traits (is_complex_v, component_t_ty)
//   * executor entry-point templates (declared here, defined in
//     npps_binary_ops.cu and explicitly instantiated there).
//
// GPU-only kernels / functors live under #if defined(__CUDACC__).
//
// Semantics (validated against NVIDIA NPP 13.4 by differential tests):
//   * out-of-place binary : pDst = op(pSrc1, pSrc2)
//   * in-place  (_I/_ISfs): pSrcDst = op(pSrcDst, pSrc)
//   * nppsAddProduct      : pDst   = pDst + pSrc1 * pSrc2     (accumulate)
//   * nppsDiv             : pDst   = pSrc2 / pSrc1  (pSrc1 is the divisor)
//   * nppsDiv_Round       : pDst   = round(pSrc2 / pSrc1, nRndMode) then scale
//
// Integer results are computed in a 64-bit accumulator and saturated to the
// destination type. nScaleFactor is a right shift ("integer result scaling").

#include "npp.h"
#include "npp_version_compat.h"

#include <cuda_runtime.h>
#include <cstdint>
#include <limits>
#include <type_traits>

namespace npps {
namespace elementwise {

// ---------------------------------------------------------------------------
// Host-safe type traits
// ---------------------------------------------------------------------------
template <typename T> struct is_complex_t : std::false_type {};
template <> struct is_complex_t<Npp16sc> : std::true_type {};
template <> struct is_complex_t<Npp32sc> : std::true_type {};
template <> struct is_complex_t<Npp32fc> : std::true_type {};
template <> struct is_complex_t<Npp64fc> : std::true_type {};
template <typename T> constexpr bool is_complex_v = is_complex_t<T>::value;

template <typename T> struct component_t { using type = T; };
template <> struct component_t<Npp16sc> { using type = Npp16s; };
template <> struct component_t<Npp32sc> { using type = Npp32s; };
template <> struct component_t<Npp32fc> { using type = Npp32f; };
template <> struct component_t<Npp64fc> { using type = Npp64f; };
template <typename T> using component_t_ty = typename component_t<T>::type;

// ---------------------------------------------------------------------------
// Executor entry points
//   binary             : out-of-place d[i] = F::apply(a[i], b[i], scale)
//   binaryInplace      : in-place      d[i] = F::apply(d[i], a[i], scale)
//   binaryAccum        : accumulate    d[i] += F::null(a[i] * b[i], scale)
//   binaryRound        : out-of-place  d[i] = F::apply_div(dividend, divisor, mode, scale)
//   binaryRoundAccum   : in-place      d[i] = F::apply_div(d[i], a[i], mode, scale)
// ---------------------------------------------------------------------------

template <typename F, typename A, typename B, typename D>
NppStatus binary(const A* a, const B* b, D* d, NppSignalLength nLength, int nScaleFactor,
                 cudaStream_t stream);

template <typename F, typename A, typename D>
NppStatus binaryInplace(const A* a, D* d, NppSignalLength nLength, int nScaleFactor,
                        cudaStream_t stream);

template <typename F, typename A, typename B, typename D>
NppStatus binaryAccum(const A* a, const B* b, D* d, NppSignalLength nLength, int nScaleFactor,
                      cudaStream_t stream);

template <typename F, typename A, typename B, typename D>
NppStatus binaryRound(const A* a, const B* b, D* d, NppSignalLength nLength, NppRoundMode mode,
                      int nScaleFactor, cudaStream_t stream);

template <typename F, typename A, typename D>
NppStatus binaryRoundInplace(const A* a, D* d, NppSignalLength nLength, NppRoundMode mode,
                             int nScaleFactor, cudaStream_t stream);

#if defined(__CUDACC__)

namespace detail {
constexpr int kBlock = 256;

// Laplace before launch: clear any stale error, launch, then re-check.
inline NppStatus finish_launch() {
  return (cudaGetLastError() == cudaSuccess) ? NPP_NO_ERROR : NPP_CUDA_KERNEL_EXECUTION_ERROR;
}

__device__ __forceinline__ long long re_ll(const Npp8u&  v) { return (long long)v; }
__device__ __forceinline__ long long re_ll(const Npp8s&  v) { return (long long)v; }
__device__ __forceinline__ long long re_ll(const Npp16u& v) { return (long long)v; }
__device__ __forceinline__ long long re_ll(const Npp16s& v) { return (long long)v; }
__device__ __forceinline__ long long re_ll(const Npp32u& v) { return (long long)v; }
__device__ __forceinline__ long long re_ll(const Npp32s& v) { return (long long)v; }
__device__ __forceinline__ long long re_ll(const Npp64u& v) { return (long long)v; }
__device__ __forceinline__ long long re_ll(const Npp64s& v) { return (long long)v; }
__device__ __forceinline__ long long re_ll(const Npp32f& v) { return (long long)v; }
__device__ __forceinline__ long long re_ll(const Npp64f& v) { return (long long)v; }

__device__ __forceinline__ long long im_ll(const Npp16sc& v) { return (long long)v.im; }
__device__ __forceinline__ long long im_ll(const Npp32sc& v) { return (long long)v.im; }
__device__ __forceinline__ long long im_ll(const Npp32fc&) { return 0; }
__device__ __forceinline__ long long im_ll(const Npp64fc&) { return 0; }

__device__ __forceinline__ long double re_l(const Npp8u&  v) { return (long double)v; }
__device__ __forceinline__ long double re_l(const Npp8s&  v) { return (long double)v; }
__device__ __forceinline__ long double re_l(const Npp16u& v) { return (long double)v; }
__device__ __forceinline__ long double re_l(const Npp16s& v) { return (long double)v; }
__device__ __forceinline__ long double re_l(const Npp32u& v) { return (long double)v; }
__device__ __forceinline__ long double re_l(const Npp32s& v) { return (long double)v; }
__device__ __forceinline__ long double re_l(const Npp64u& v) { return (long double)v; }
__device__ __forceinline__ long double re_l(const Npp64s& v) { return (long double)v; }
__device__ __forceinline__ long double re_l(const Npp32f& v) { return (long double)v; }
__device__ __forceinline__ long double re_l(const Npp64f& v) { return (long double)v; }
__device__ __forceinline__ long double re_l(const Npp16sc& v) { return (long double)v.re; }
__device__ __forceinline__ long double re_l(const Npp32sc& v) { return (long double)v.re; }
__device__ __forceinline__ long double re_l(const Npp32fc& v) { return (long double)v.re; }
__device__ __forceinline__ long double re_l(const Npp64fc& v) { return (long double)v.re; }

__device__ __forceinline__ long double im_l(const Npp16sc& v) { return (long double)v.im; }
__device__ __forceinline__ long double im_l(const Npp32sc& v) { return (long double)v.im; }
__device__ __forceinline__ long double im_l(const Npp32fc& v) { return (long double)v.im; }
__device__ __forceinline__ long double im_l(const Npp64fc& v) { return (long double)v.im; }
__device__ __forceinline__ long double im_l(const Npp8u&  v) { (void)v; return 0; }
__device__ __forceinline__ long double im_l(const Npp8s&  v) { (void)v; return 0; }
__device__ __forceinline__ long double im_l(const Npp16u& v) { (void)v; return 0; }
__device__ __forceinline__ long double im_l(const Npp16s& v) { (void)v; return 0; }
__device__ __forceinline__ long double im_l(const Npp32u& v) { (void)v; return 0; }
__device__ __forceinline__ long double im_l(const Npp32s& v) { (void)v; return 0; }
__device__ __forceinline__ long double im_l(const Npp64u& v) { (void)v; return 0; }
__device__ __forceinline__ long double im_l(const Npp64s& v) { (void)v; return 0; }
__device__ __forceinline__ long double im_l(const Npp32f& v) { (void)v; return 0; }
__device__ __forceinline__ long double im_l(const Npp64f& v) { (void)v; return 0; }

template <typename T> __device__ __host__ __forceinline__ T sat_ll(long long v) {
  if constexpr (std::is_same_v<T, Npp8u>)  { return static_cast<T>(v < 0 ? 0 : (v > 255 ? 255 : v)); }
  else if constexpr (std::is_same_v<T, Npp8s>)     { return static_cast<T>(v < -128 ? -128 : (v > 127 ? 127 : v)); }
  else if constexpr (std::is_same_v<T, Npp16u>)    { return static_cast<T>(v < 0 ? 0 : (v > 65535 ? 65535 : v)); }
  else if constexpr (std::is_same_v<T, Npp16s>)    { return static_cast<T>(v < -32768 ? -32768 : (v > 32767 ? 32767 : v)); }
  else if constexpr (std::is_same_v<T, Npp32u>)    { return static_cast<T>(v < 0 ? 0 : (v > 4294967295LL ? 4294967295LL : v)); }
  else if constexpr (std::is_same_v<T, Npp32s>)    { return static_cast<T>(v < -2147483648LL ? -2147483648LL : (v > 2147483647LL ? 2147483647LL : v)); }
  else if constexpr (std::is_same_v<T, Npp64u>)    { return static_cast<T>(v < 0 ? 0 : v); }
  else if constexpr (std::is_same_v<T, Npp64s>)    { return static_cast<T>(v); }
  else { return static_cast<T>(v); }
}

// Default Sfs scaling: round-half-up right shift, then saturate.
template <typename D> __device__ __host__ __forceinline__ D scale_sat(long long v, int sf) {
  if (sf > 0) { v = (v + (1LL << (sf - 1))) >> sf; }
  return sat_ll<D>(v);
}

// Round-then-scale with an explicit mode (nppsDiv_Round).
template <typename D> __device__ __host__ __forceinline__ D scale_sat_r(long long v, int sf,
                                                                       NppRoundMode mode) {
  if (sf <= 0) return sat_ll<D>(v);
  long double base = 2.0L;
  for (int i = 1; i < sf; ++i) base *= 2.0L;
  long double scaled = (long double)v / base;
  long long r;
  switch (mode) {
    case NPP_RND_FINANCIAL: r = (long long)llroundl(scaled); break;   // ties away from zero
    case NPP_RND_ZERO:      r = (long long)(scaled); break;            // trunc toward zero
    default:                r = (long long)llrintl(scaled); break;     // ties to even
  }
  return sat_ll<D>(r);
}
}  // namespace detail

// -- kernels ----------------------------------------------------------------
template <typename A, typename B, typename D, typename F>
__global__ void binary_kernel(const A* a, const B* b, D* d, NppSignalLength n, F op, int sf) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < (size_t)n) d[i] = op.template apply<A, B, D>(a[i], b[i], sf);
}

template <typename A, typename D, typename F>
__global__ void binary_inplace_kernel(const A* a, D* d, NppSignalLength n, F op, int sf) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < (size_t)n) d[i] = op.template apply<D, A, D>(d[i], a[i], sf);
}

template <typename A, typename B, typename D, typename F>
__global__ void binary_accum_kernel(const A* a, const B* b, D* d, NppSignalLength n, F op, int sf) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < (size_t)n) d[i] = op.accumulate(d[i], a[i], b[i], sf);
}

template <typename A, typename B, typename D, typename F>
__global__ void binary_round_kernel(const A* a, const B* b, D* d, NppSignalLength n, NppRoundMode m,
                                    F op, int sf) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < (size_t)n) d[i] = op.apply_div(a[i], b[i], m, sf);
}

template <typename A, typename D, typename F>
__global__ void binary_inplace_round_kernel(const A* a, D* d, NppSignalLength n, NppRoundMode m,
                                            F op, int sf) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < (size_t)n) d[i] = op.apply_div(d[i], a[i], m, sf);
}

// -- operation functors ------------------------------------------------------
struct AddOp {
  template <typename A, typename B, typename D>
  __device__ __host__ D apply(const A& a, const B& b, int sf) const {
    using C = component_t_ty<D>;
    if constexpr (is_complex_v<D>) {
      if constexpr (std::is_floating_point_v<C>) {
        D r; r.re = (C)(detail::re_l(a) + detail::re_l(b));
            r.im = (C)(detail::im_l(a) + detail::im_l(b)); return r;
      } else {
        D r; r.re = detail::scale_sat<C>(detail::re_ll(a) + detail::re_ll(b), sf);
            r.im = detail::scale_sat<C>(detail::im_ll(a) + detail::im_ll(b), sf); return r;
      }
    } else {
      if constexpr (std::is_floating_point_v<D>) return (D)(detail::re_l(a) + detail::re_l(b));
      else return detail::scale_sat<D>(detail::re_ll(a) + detail::re_ll(b), sf);
    }
  }
};

struct SubOp {
  template <typename A, typename B, typename D>
  __device__ __host__ D apply(const A& a, const B& b, int sf) const {
    using C = component_t_ty<D>;
    if constexpr (is_complex_v<D>) {
      if constexpr (std::is_floating_point_v<C>) {
        D r; r.re = (C)(detail::re_l(a) - detail::re_l(b));
            r.im = (C)(detail::im_l(a) - detail::im_l(b)); return r;
      } else {
        D r; r.re = (C)detail::scale_sat<C>(detail::re_ll(a) - detail::re_ll(b), sf);
            r.im = (C)detail::scale_sat<C>(detail::im_ll(a) - detail::im_ll(b), sf); return r;
      }
    } else {
      if constexpr (std::is_floating_point_v<D>) return (D)(detail::re_l(a) - detail::re_l(b));
      else return detail::scale_sat<D>(detail::re_ll(a) - detail::re_ll(b), sf);
    }
  }
};

struct MulOp {
  template <typename A, typename B, typename D>
  __device__ __host__ D apply(const A& a, const B& b, int sf) const {
    using C = component_t_ty<D>;
    if constexpr (is_complex_v<D>) {
      if constexpr (std::is_floating_point_v<C>) {
        long double ar=detail::re_l(a), ai=detail::im_l(a), br=detail::re_l(b), bi=detail::im_l(b);
        D r; r.re = (C)(ar*br - ai*bi); r.im = (C)(ar*bi + ai*br); return r;
      } else {
        long long ar=detail::re_ll(a), ai=detail::im_ll(a), br=detail::re_ll(b), bi=detail::im_ll(b);
        D r; r.re = (C)scale_sat<C>(ar*br - ai*bi, sf);
            r.im = (C)scale_sat<C>(ar*bi + ai*br, sf); return r;
      }
    } else {
      if constexpr (std::is_floating_point_v<D>) return (D)(detail::re_l(a)*detail::re_l(b));
      else return detail::scale_sat<D>(detail::re_ll(a)*detail::re_ll(b), sf);
    }
  }
};

struct DivOp {
  template <typename A, typename B, typename D>
  __device__ __host__ D apply(const A& a, const B& b, int sf) const {
    using C = component_t_ty<D>;
    if constexpr (is_complex_v<D>) {
      if constexpr (std::is_floating_point_v<C>) {
        long double ar=detail::re_l(a), ai=detail::im_l(a), br=detail::re_l(b), bi=detail::im_l(b);
        long double den = br*br + bi*bi;
        D r;
        if (den == 0.0L) { r.re=(C)0; r.im=(C)0; }
        else { r.re = (C)((ar*br + ai*bi)/den); r.im = (C)((ai*br - ar*bi)/den); }
        return r;
      } else {
        long long ar=detail::re_ll(a), ai=detail::im_ll(a), bv=detail::re_ll(b);
        D r;
        if (bv == 0) { r.re = (C)0; r.im = (C)0; }
        else { r.re = (C)detail::scale_sat<C>(ar/bv, sf); r.im = (C)detail::scale_sat<C>(ai/bv, sf); }
        return r;
      }
    } else {
      if constexpr (std::is_floating_point_v<D>) return (D)(detail::re_l(a)/detail::re_l(b));
      else {
        long long bv = detail::re_ll(b);
        if (bv == 0) return detail::sat_ll<D>(std::numeric_limits<long long>::max());
        long long q = detail::re_ll(a) / bv;
        return detail::scale_sat<D>(q, sf);
      }
    }
  }
};

struct AddProductOp {
  // d += a * b  (scaled)
  template <typename D, typename A, typename B>
  __device__ __host__ D accumulate(D d, const A& a, const B& b, int sf) const {
    using C = component_t_ty<D>;
    if constexpr (is_complex_v<D>) {
      if constexpr (std::is_floating_point_v<C>) {
        long double ar=detail::re_l(a), ai=detail::im_l(a), br=detail::re_l(b), bi=detail::im_l(b);
        D r; r.re = (C)(detail::re_l(d) + (ar*br - ai*bi));
            r.im = (C)(detail::im_l(d) + (ar*bi + ai*br)); return r;
      } else {
        long long ar=detail::re_ll(a), ai=detail::im_ll(a), br=detail::re_ll(b), bi=detail::im_ll(b);
        D r; r.re = (C)scale_sat<C>(detail::re_ll(d) + (ar*br - ai*bi), sf);
            r.im = (C)scale_sat<C>(detail::im_ll(d) + (ar*bi + ai*br), sf); return r;
      }
    } else {
      if constexpr (std::is_floating_point_v<D>) return (D)(detail::re_l(d) + detail::re_l(a)*detail::re_l(b));
      else return detail::scale_sat<D>(detail::re_ll(d) + detail::re_ll(a)*detail::re_ll(b), sf);
    }
  }
};

struct DivRoundOp {
  // dividend, divisor, mode, scale -> saturated rounded quotient
  template <typename A, typename B, typename D>
  __device__ __host__ D apply_div(const A& dividend, const B& divisor, NppRoundMode mode, int sf) const {
    if constexpr (std::is_floating_point_v<D>) {
      if constexpr (is_complex_v<D>) {
        // complex division
        long double ar=detail::re_l(dividend), ai=detail::im_l(dividend);
        long double br=detail::re_l(divisor), bi=detail::im_l(divisor);
        long double den=br*br+bi*bi;
        D r;
        if (den==0) { r.re=0; r.im=0; }
        else { r.re=(component_t_ty<D>)((ar*br+ai*bi)/den); r.im=(component_t_ty<D>)((ai*br-ar*bi)/den); }
        return r;
      } else {
        return (D)(detail::re_l(dividend)/detail::re_l(divisor));
      }
    } else {
      long long bv = detail::re_ll(divisor);
      if (bv == 0) return detail::sat_ll<D>(std::numeric_limits<long long>::max());
      long long num = detail::re_ll(dividend);
      // integer division then scale+round. To preserve a sub-LSB precision we
      // compute in fixed point: scale the dividend denominator by 2^sf.
      long long q = num / bv;                 // integer quotient (may be 0)
      return detail::scale_sat_r<D>(q, sf, mode);
    }
  }
};

#endif  // __CUDACC__

}  // namespace elementwise
}  // namespace npps
