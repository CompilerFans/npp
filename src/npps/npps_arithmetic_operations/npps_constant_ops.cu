// Constant (scalar) 1D element-wise kernels + executors for the npps constant
// arithmetic families (AddC/SubC/SubCRev/MulC/DivC/DivCRev/AndC/OrC/XorC/
// AddProductC/MulC_Low/Normalize).  Mirrors npps_binary_ops.cu.
#include "npp.h"
#include "npp_version_compat.h"
#include "npps_constant_executor.h"

#include <cuda_runtime.h>
#include <cmath>
#include <cstdint>
#include <limits>

namespace npps {
namespace constant {

namespace ed = npps::elementwise;
namespace ek = npps::elementwise::detail;

namespace {
inline NppStatus finish_launch() {
  return (cudaGetLastError() == cudaSuccess) ? NPP_NO_ERROR : NPP_CUDA_KERNEL_EXECUTION_ERROR;
}

// integer division with round-half-to-even, matching the element-wise DivOp.
template <typename D>
__device__ __host__ D div_round_even(long long num, long long den) {
  if (den == 0) return ek::sat_ll<D>(std::numeric_limits<long long>::max());
  long long q = num / den;
  long long r = num % den;
  if (r != 0) {
    long long ar = r < 0 ? -r : r;
    long long ad = den < 0 ? -den : den;
    int sign = ((num < 0) ^ (den < 0)) ? -1 : 1;
    if ((ar << 1) > ad) q += sign;
    else if ((ar << 1) == ad && (q & 1LL)) q += sign;
  }
  return ek::sat_ll<D>(q);
}


// integer division with round-half-away-from-zero (nppsNormalize real).
template <typename D>
__device__ __host__ D div_round_away(long long num, long long den) {
  if (den == 0) return ek::sat_ll<D>(std::numeric_limits<long long>::max());
  long double q = (long double)num / (long double)den;
  long long r;
  if (q >= 0.0L) r = (long long)(q + 0.5L);
  else r = - (long long)(-q + 0.5L);
  return ek::sat_ll<D>(r);
}

}  // namespace

namespace kernel {

template <typename F, typename A, typename V, typename D>
__global__ void constant_kernel(const A* src, V value, D* dst, NppSignalLength n, F op, int sf) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < (size_t)n) dst[i] = op.template apply<A, V, D>(src[i], value, sf);
}

template <typename F, typename V, typename AOUT>
__global__ void inplace_mixed_kernel(V value, AOUT * dst, NppSignalLength n, F op, int sf) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < (size_t)n) dst[i] = op.template apply<AOUT, V, AOUT>(dst[i], value, sf);
}

template <typename F, typename A, typename V>
__global__ void normalize_kernel(const A * src, A vSub, V vDiv, A * dst, NppSignalLength n, F op, int sf) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < (size_t)n) dst[i] = op.template apply<A, V, A>(src[i], vSub, vDiv, sf);
}

template <typename A, typename V>
__global__ void accumulate_kernel(const A * src, V value, A * dst, NppSignalLength n) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < (size_t)n) dst[i] = dst[i] + static_cast<A>(src[i] * value);
}
}  // namespace kernel

namespace {

struct AddCOp {
  template <typename A, typename V, typename D>
  __device__ __host__ D apply(const A& a, const V& b, int sf) const {
    using C = ed::component_t_ty<D>;
    if constexpr (ed::is_complex_v<D>) {
      if constexpr (std::is_floating_point_v<C>) {
        D c; c.re = (C)(ek::re_l(a) + ek::re_l(b)); c.im = (C)(ek::im_l(a) + ek::im_l(b)); return c;
      } else {
        D c; c.re = (C)ek::scale_sat<C>(ek::re_ll(a) + ek::re_ll(b), sf);
            c.im = (C)ek::scale_sat<C>(ek::im_ll(a) + ek::im_ll(b), sf); return c;
      }
    } else {
      if constexpr (std::is_floating_point_v<D>) return (A)(ek::re_l(a) + ek::re_l(b));
      else return ek::scale_sat<D>(ek::re_ll(a) + ek::re_ll(b), sf);
    }
  }
};

struct SubCOp {
  template <typename A, typename V, typename D>
  __device__ __host__ D apply(const A& a, const V& b, int sf) const {
    using C = ed::component_t_ty<D>;
    if constexpr (ed::is_complex_v<D>) {
      if constexpr (std::is_floating_point_v<C>) {
        D c; c.re = (C)(ek::re_l(a) - ek::re_l(b)); c.im = (C)(ek::im_l(a) - ek::im_l(b)); return c;
      } else {
        D c; c.re = (C)ek::scale_sat<C>(ek::re_ll(a) - ek::re_ll(b), sf);
            c.im = (C)ek::scale_sat<C>(ek::im_ll(a) - ek::im_ll(b), sf); return c;
      }
    } else {
      if constexpr (std::is_floating_point_v<D>) return (A)(ek::re_l(a) - ek::re_l(b));
      else return ek::scale_sat<D>(ek::re_ll(a) - ek::re_ll(b), sf);
    }
  }
};

struct SubCRevOp {
  template <typename A, typename V, typename D>
  __device__ __host__ D apply(const A& a, const V& b, int sf) const {
    using C = ed::component_t_ty<D>;
    if constexpr (ed::is_complex_v<D>) {
      if constexpr (std::is_floating_point_v<C>) {
        D c; c.re = (C)(ek::re_l(b) - ek::re_l(a)); c.im = (C)(ek::im_l(b) - ek::im_l(a)); return c;
      } else {
        D c; c.re = (C)ek::scale_sat<C>(ek::re_ll(b) - ek::re_ll(a), sf);
            c.im = (C)ek::scale_sat<C>(ek::im_ll(b) - ek::im_ll(a), sf); return c;
      }
    } else {
      if constexpr (std::is_floating_point_v<D>) return (A)(ek::re_l(b) - ek::re_l(a));
      else return ek::scale_sat<D>(ek::re_ll(b) - ek::re_ll(a), sf);
    }
  }
};

struct MulCOp {
  template <typename A, typename V, typename D>
  __device__ __host__ D apply(const A& a, const V& b, int sf) const {
    using C = ed::component_t_ty<D>;
    if constexpr (ed::is_complex_v<D>) {
      if constexpr (std::is_floating_point_v<C>) {
        long double ar=ek::re_l(a), ai=ek::im_l(a), br=ek::re_l(b), bi=ek::im_l(b);
        D c; c.re=(C)(ar*br - ai*bi); c.im=(C)(ar*bi + ai*br); return c;
      } else {
        long long ar=ek::re_ll(a), ai=ek::im_ll(a), br=ek::re_ll(b), bi=ek::im_ll(b);
        D c; c.re=(C)ek::scale_sat<C>(ar*br - ai*bi, sf);
            c.im=(C)(ek::scale_sat<C>(ar*bi + ai*br, sf)); return c;
      }
    } else {
      if constexpr (std::is_floating_point_v<D>) return (A)(ek::re_l(a) * ek::re_l(b));
      else return ek::scale_sat<D>(ek::re_ll(a) * ek::re_ll(b), sf);
    }
  }
};

struct DivCOp {
  template <typename A, typename V, typename D>
  __device__ __host__ D apply(const A& a, const V& b, int sf) const {
    using C = ed::component_t_ty<D>;
    if constexpr (ed::is_complex_v<D>) {
      if constexpr (std::is_floating_point_v<C>) {
        long double ar=ek::re_l(a), ai=ek::im_l(a), br=ek::re_l(b), bi=ek::im_l(b);
        long double den = br*br + bi*bi; D c;
        if (den == 0.0L) { c.re=(C)0; c.im=(C)0; }
        else { c.re=(C)((ar*br + ai*bi)/den); c.im=(C)((ai*br - ar*bi)/den); }
        return c;
      } else {
        long long ar=ek::re_ll(a), ai=ek::im_ll(a), br=ek::re_ll(b), bi=ek::im_ll(b);
        long long den = (br*br + bi*bi) << sf;
        D c;
        if (den == 0) { c.re=(C)0; c.im=(C)0; }
        else { c.re=(C)(div_round_even<C>(ar*br + ai*bi, den)); c.im=(C)(div_round_even<C>(ai*br - ar*bi, den)); }
        return c;
      }
    } else {
      if constexpr (std::is_floating_point_v<D>) return (A)(ek::re_l(a) / ek::re_l(b));
      else return div_round_even<D>(ek::re_ll(a), ek::re_ll(b) << sf);
    }
  }
};

struct DivCRevOp {
  // pDst = nValue / pSrc  (real signal forms only: _16u / _32f in this package)
  template <typename A, typename V, typename D>
  __device__ __host__ D apply(const A& a, const V& b, int sf) const {
    if constexpr (std::is_floating_point_v<D>) return (A)(ek::re_l(b) / ek::re_l(a));
    else return div_round_even<D>(ek::re_ll(b), ek::re_ll(a) << sf);
  }
};

struct AndCOp {
  template <typename A, typename V, typename D>
  __device__ __host__ D apply(const A& a, const V& b, int sf) const {
    (void)sf; return (D)((unsigned long long)ek::re_ll(a) & (unsigned long long)ek::re_ll(b));
  }
};
struct OrCOp {
  template <typename A, typename V, typename D>
  __device__ __host__ D apply(const A& a, const V& b, int sf) const {
    (void)sf; return (D)((unsigned long long)ek::re_ll(a) | (unsigned long long)ek::re_ll(b));
  }
};
struct XorCOp {
  template <typename A, typename V, typename D>
  __device__ __host__ D apply(const A& a, const V& b, int sf) const {
    (void)sf; return (D)((unsigned long long)ek::re_ll(a) ^ (unsigned long long)ek::re_ll(b));
  }
};

struct MulCF2I16Op {
  template <typename A, typename V, typename D>
  __device__ __host__ D apply(const A& a, const V& b, int sf) const {
    double v = (double)ek::re_l(a) * (double)ek::re_l(b);
    v = ldexp(v, -sf);
    return ek::sat_ll<D>(static_cast<long long>(llrint(v)));
  }
};

struct MulCF2I64Op {
  // NPP 13.4 rounds toward +infinity (validated on the 5090).
  template <typename A, typename V, typename D>
  __device__ __host__ D apply(const A& a, const V& b, int sf) const {
    double v = (double)a * (double)b;
    v = ldexp(v, -sf);
    return ek::sat_ll<D>(static_cast<long long>(ceil(v)));
  }
};

struct MulCLowOp {
  template <typename A, typename V, typename D>
  __device__ __host__ D apply(const A& a, const V& b, int sf) const {
    (void)sf;
    return ek::sat_ll<D>(static_cast<long long>(llrint(static_cast<double>(ek::re_l(a) * ek::re_l(b)))));
  }
};

struct NormalizeOp {
  template <typename A, typename DV, typename D>
  __device__ __host__ D apply(const A& x, const A& vSub, const DV& vDiv, int sf) const {
    using C = ed::component_t_ty<D>;
    if constexpr (std::is_floating_point_v<C>) {
      if constexpr (ed::is_complex_v<D>) {
        long double dv = (long double)ek::re_l(vDiv);
        D c; c.re=(C)((ek::re_l(x)-ek::re_l(vSub))/dv); c.im=(C)((ek::im_l(x)-ek::im_l(vSub))/dv); return c;
      } else {
        return (A)((ek::re_l(x)-ek::re_l(vSub))/ek::re_l(vDiv));
      }
    } else {
      long long num = ek::re_ll(x) - ek::re_ll(vSub);
      long long den = (long long)ek::re_ll(vDiv) << sf;
      auto cx_rt = [&](long long nn)->C {
        long double vv = (long double)nn / (long double)den;
        if (vv >= 0.0L) return (C)static_cast<long long>(vv + 0.5L);
        return (C)static_cast<long long>(vv);  // trunc toward zero for negatives (NPP observed on 16sc)
      };
      if constexpr (ed::is_complex_v<D>) {
        long long numi = ek::im_ll(x) - ek::im_ll(vSub);
        D c; c.re=(C)cx_rt(num); c.im=(C)cx_rt(numi); return c;
      } else {
        // round-half-away-from-zero (NPP observed: -1506.5 -> -1507)
        return div_round_away<D>(num, den);
      }
    }
  }
};
}  // namespace

namespace {
template <typename F, typename A>
NppStatus run_constant(const A * src, A value, A * dst, NppSignalLength nLength, int sf, cudaStream_t stream) {
  if (!src || !dst) return NPP_NULL_POINTER_ERROR;
  if (nLength == 0) return NPP_SIZE_ERROR;
  const size_t n = (size_t)nLength;
  const size_t grid = (n + ek::kBlock - 1) / ek::kBlock;
  F op; (void)cudaGetLastError();
  kernel::constant_kernel<F, A, A, A><<<(unsigned)grid, ek::kBlock, 0, stream>>>(src, value, dst, nLength, op, sf);
  return finish_launch();
}

template <typename F, typename AIN, typename V, typename AOUT>
NppStatus run_mixed(const AIN* src, V value, AOUT* dst, NppSignalLength nLength, int sf, cudaStream_t stream) {
  if (!src || !dst) return NPP_NULL_POINTER_ERROR;
  if (nLength == 0) return NPP_SIZE_ERROR;
  const size_t n = (size_t)nLength;
  const size_t grid = (n + ek::kBlock - 1) / ek::kBlock;
  F op; (void)cudaGetLastError();
  kernel::constant_kernel<F, AIN, V, AOUT><<<(unsigned)grid, ek::kBlock, 0, stream>>>(src, value, dst, n, op, sf);
  return finish_launch();
}

template <typename F, typename V, typename AOUT>
NppStatus run_inplace_mixed(V value, AOUT* dst, NppSignalLength n_len, int sf, cudaStream_t stream) {
  if (!dst) return NPP_NULL_POINTER_ERROR;
  if (n_len == 0) return NPP_SIZE_ERROR;
  const size_t n = (size_t)n_len;
  const size_t grid = (n + ek::kBlock - 1) / ek::kBlock;
  F op; (void)cudaGetLastError();
  kernel::inplace_mixed_kernel<F, V, AOUT><<<(unsigned)grid, ek::kBlock, 0, stream>>>(value, dst, n, op, sf);
  return finish_launch();
}

template <typename A, typename V>
NppStatus run_acc(const A* src, V value, A* dst, NppSignalLength n_len, cudaStream_t stream) {
  if (!src || !dst) return NPP_NULL_POINTER_ERROR;
  if (n_len == 0) return NPP_SIZE_ERROR;
  const size_t n = (size_t)n_len;
  const size_t grid = (n + ek::kBlock - 1) / ek::kBlock;
  (void)cudaGetLastError();
  kernel::accumulate_kernel<A, V><<<(unsigned)grid, ek::kBlock, 0, stream>>>(src, value, dst, n);
  return finish_launch();
}
template <typename F, typename A, typename DV>
NppStatus run_normalize(const A* src, A vSub, DV vDiv, A* dst, NppSignalLength nLength, int sf, cudaStream_t stream) {
  if (!src || !dst) return NPP_NULL_POINTER_ERROR;
  if (nLength == 0) return NPP_SIZE_ERROR;
  const size_t n = (size_t)nLength;
  const size_t grid = (n + ek::kBlock - 1) / ek::kBlock;
  F op; (void)cudaGetLastError();
  kernel::normalize_kernel<F, A, DV><<<(unsigned)grid, ek::kBlock, 0, stream>>>(src, vSub, vDiv, dst, n, op, sf);
  return finish_launch();
}
}  // namespace

// --- public same-type entry points ---
template <typename A>
NppStatus addC(const A* src, A nValue, A* dst, NppSignalLength nLength, int nScaleFactor, cudaStream_t stream) {
  return run_constant<AddCOp, A>(src, nValue, dst, nLength, nScaleFactor, stream);
}
template <typename A>
NppStatus subC(const A* src, A nValue, A* dst, NppSignalLength nLength, int nScaleFactor, cudaStream_t stream) {
  return run_constant<SubCOp, A>(src, nValue, dst, nLength, nScaleFactor, stream);
}
template <typename A>
NppStatus subCRev(const A* src, A nValue, A* dst, NppSignalLength nLength, int nScaleFactor, cudaStream_t stream) {
  return run_constant<SubCRevOp, A>(src, nValue, dst, nLength, nScaleFactor, stream);
}
template <typename A>
NppStatus mulC(const A* src, A nValue, A* dst, NppSignalLength nLength, int nScaleFactor, cudaStream_t stream) {
  return run_constant<MulCOp, A>(src, nValue, dst, nLength, nScaleFactor, stream);
}
template <typename A>
NppStatus divC(const A* src, A nValue, A* dst, NppSignalLength nLength, int nScaleFactor, cudaStream_t stream) {
  return run_constant<DivCOp, A>(src, nValue, dst, nLength, nScaleFactor, stream);
}
template <typename A>
NppStatus divCRev(const A* src, A nValue, A* dst, NppSignalLength nLength, int nScaleFactor, cudaStream_t stream) {
  return run_constant<DivCRevOp, A>(src, nValue, dst, nLength, nScaleFactor, stream);
}
template <typename A>
NppStatus andC(const A* src, A nValue, A* dst, NppSignalLength nLength, cudaStream_t stream) {
  return run_constant<AndCOp, A>(src, nValue, dst, nLength, 0, stream);
}
template <typename A>
NppStatus orC(const A* src, A nValue, A* dst, NppSignalLength nLength, cudaStream_t stream) {
  return run_constant<OrCOp, A>(src, nValue, dst, nLength, 0, stream);
}
template <typename A>
NppStatus xorC(const A* src, A nValue, A* dst, NppSignalLength nLength, cudaStream_t stream) {
  return run_constant<XorCOp, A>(src, nValue, dst, nLength, 0, stream);
}

NppStatus mulCFloatToI16(const Npp32f* src, Npp32f nValue, Npp16s* dst, NppSignalLength nLength, int nScaleFactor, cudaStream_t stream) {
  return run_mixed<MulCF2I16Op, Npp32f, Npp32f, Npp16s>(src, nValue, dst, nLength, nScaleFactor, stream);
}
NppStatus mulCInplaceFloatToI64(Npp64f nValue, Npp64s* pSrcDst, NppSignalLength nLength, int nScaleFactor, cudaStream_t stream) {
  return run_inplace_mixed<MulCF2I64Op, Npp64f, Npp64s>(nValue, pSrcDst, nLength, nScaleFactor, stream);
}
NppStatus mulCLow(const Npp32f* src, Npp32f nValue, Npp16s* dst, NppSignalLength nLength, cudaStream_t stream) {
  return run_mixed<MulCLowOp, Npp32f, Npp32f, Npp16s>(src, nValue, dst, nLength, 0, stream);
}
NppStatus addProductC(const Npp32f* src, Npp32f nValue, Npp32f* dst, NppSignalLength nLength, cudaStream_t stream) {
  return run_acc<Npp32f, Npp32f>(src, nValue, dst, nLength, stream);
}

template <typename A, typename DV>
NppStatus normalize(const A* src, A vSub, DV vDiv, A* dst, NppSignalLength nLength, int nScaleFactor, cudaStream_t stream) {
  return run_normalize<NormalizeOp, A, DV>(src, vSub, vDiv, dst, nLength, nScaleFactor, stream);
}


template NppStatus addC<Npp8u>(const Npp8u*, Npp8u, Npp8u*, NppSignalLength, int, cudaStream_t);
template NppStatus addC<Npp16u>(const Npp16u*, Npp16u, Npp16u*, NppSignalLength, int, cudaStream_t);
template NppStatus addC<Npp16s>(const Npp16s*, Npp16s, Npp16s*, NppSignalLength, int, cudaStream_t);
template NppStatus addC<Npp16sc>(const Npp16sc*, Npp16sc, Npp16sc*, NppSignalLength, int, cudaStream_t);
template NppStatus addC<Npp32f>(const Npp32f*, Npp32f, Npp32f*, NppSignalLength, int, cudaStream_t);
template NppStatus addC<Npp32fc>(const Npp32fc*, Npp32fc, Npp32fc*, NppSignalLength, int, cudaStream_t);
template NppStatus addC<Npp32s>(const Npp32s*, Npp32s, Npp32s*, NppSignalLength, int, cudaStream_t);
template NppStatus addC<Npp32sc>(const Npp32sc*, Npp32sc, Npp32sc*, NppSignalLength, int, cudaStream_t);
template NppStatus addC<Npp64f>(const Npp64f*, Npp64f, Npp64f*, NppSignalLength, int, cudaStream_t);
template NppStatus addC<Npp64fc>(const Npp64fc*, Npp64fc, Npp64fc*, NppSignalLength, int, cudaStream_t);
template NppStatus subC<Npp8u>(const Npp8u*, Npp8u, Npp8u*, NppSignalLength, int, cudaStream_t);
template NppStatus subC<Npp16u>(const Npp16u*, Npp16u, Npp16u*, NppSignalLength, int, cudaStream_t);
template NppStatus subC<Npp16s>(const Npp16s*, Npp16s, Npp16s*, NppSignalLength, int, cudaStream_t);
template NppStatus subC<Npp16sc>(const Npp16sc*, Npp16sc, Npp16sc*, NppSignalLength, int, cudaStream_t);
template NppStatus subC<Npp32f>(const Npp32f*, Npp32f, Npp32f*, NppSignalLength, int, cudaStream_t);
template NppStatus subC<Npp32fc>(const Npp32fc*, Npp32fc, Npp32fc*, NppSignalLength, int, cudaStream_t);
template NppStatus subC<Npp32s>(const Npp32s*, Npp32s, Npp32s*, NppSignalLength, int, cudaStream_t);
template NppStatus subC<Npp32sc>(const Npp32sc*, Npp32sc, Npp32sc*, NppSignalLength, int, cudaStream_t);
template NppStatus subC<Npp64f>(const Npp64f*, Npp64f, Npp64f*, NppSignalLength, int, cudaStream_t);
template NppStatus subC<Npp64fc>(const Npp64fc*, Npp64fc, Npp64fc*, NppSignalLength, int, cudaStream_t);
template NppStatus subCRev<Npp8u>(const Npp8u*, Npp8u, Npp8u*, NppSignalLength, int, cudaStream_t);
template NppStatus subCRev<Npp16u>(const Npp16u*, Npp16u, Npp16u*, NppSignalLength, int, cudaStream_t);
template NppStatus subCRev<Npp16s>(const Npp16s*, Npp16s, Npp16s*, NppSignalLength, int, cudaStream_t);
template NppStatus subCRev<Npp16sc>(const Npp16sc*, Npp16sc, Npp16sc*, NppSignalLength, int, cudaStream_t);
template NppStatus subCRev<Npp32f>(const Npp32f*, Npp32f, Npp32f*, NppSignalLength, int, cudaStream_t);
template NppStatus subCRev<Npp32fc>(const Npp32fc*, Npp32fc, Npp32fc*, NppSignalLength, int, cudaStream_t);
template NppStatus subCRev<Npp32s>(const Npp32s*, Npp32s, Npp32s*, NppSignalLength, int, cudaStream_t);
template NppStatus subCRev<Npp32sc>(const Npp32sc*, Npp32sc, Npp32sc*, NppSignalLength, int, cudaStream_t);
template NppStatus subCRev<Npp64f>(const Npp64f*, Npp64f, Npp64f*, NppSignalLength, int, cudaStream_t);
template NppStatus subCRev<Npp64fc>(const Npp64fc*, Npp64fc, Npp64fc*, NppSignalLength, int, cudaStream_t);
template NppStatus mulC<Npp8u>(const Npp8u*, Npp8u, Npp8u*, NppSignalLength, int, cudaStream_t);
template NppStatus mulC<Npp16u>(const Npp16u*, Npp16u, Npp16u*, NppSignalLength, int, cudaStream_t);
template NppStatus mulC<Npp16s>(const Npp16s*, Npp16s, Npp16s*, NppSignalLength, int, cudaStream_t);
template NppStatus mulC<Npp16sc>(const Npp16sc*, Npp16sc, Npp16sc*, NppSignalLength, int, cudaStream_t);
template NppStatus mulC<Npp32f>(const Npp32f*, Npp32f, Npp32f*, NppSignalLength, int, cudaStream_t);
template NppStatus mulC<Npp32fc>(const Npp32fc*, Npp32fc, Npp32fc*, NppSignalLength, int, cudaStream_t);
template NppStatus mulC<Npp32s>(const Npp32s*, Npp32s, Npp32s*, NppSignalLength, int, cudaStream_t);
template NppStatus mulC<Npp32sc>(const Npp32sc*, Npp32sc, Npp32sc*, NppSignalLength, int, cudaStream_t);
template NppStatus mulC<Npp64f>(const Npp64f*, Npp64f, Npp64f*, NppSignalLength, int, cudaStream_t);
template NppStatus mulC<Npp64fc>(const Npp64fc*, Npp64fc, Npp64fc*, NppSignalLength, int, cudaStream_t);
template NppStatus divC<Npp8u>(const Npp8u*, Npp8u, Npp8u*, NppSignalLength, int, cudaStream_t);
template NppStatus divC<Npp16s>(const Npp16s*, Npp16s, Npp16s*, NppSignalLength, int, cudaStream_t);
template NppStatus divC<Npp16sc>(const Npp16sc*, Npp16sc, Npp16sc*, NppSignalLength, int, cudaStream_t);
template NppStatus divC<Npp16u>(const Npp16u*, Npp16u, Npp16u*, NppSignalLength, int, cudaStream_t);
template NppStatus divC<Npp32f>(const Npp32f*, Npp32f, Npp32f*, NppSignalLength, int, cudaStream_t);
template NppStatus divC<Npp32fc>(const Npp32fc*, Npp32fc, Npp32fc*, NppSignalLength, int, cudaStream_t);
template NppStatus divC<Npp64f>(const Npp64f*, Npp64f, Npp64f*, NppSignalLength, int, cudaStream_t);
template NppStatus divC<Npp64fc>(const Npp64fc*, Npp64fc, Npp64fc*, NppSignalLength, int, cudaStream_t);
template NppStatus divCRev<Npp16u>(const Npp16u*, Npp16u, Npp16u*, NppSignalLength, int, cudaStream_t);
template NppStatus divCRev<Npp32f>(const Npp32f*, Npp32f, Npp32f*, NppSignalLength, int, cudaStream_t);
template NppStatus andC<Npp8u>(const Npp8u*, Npp8u, Npp8u*, NppSignalLength, cudaStream_t);
template NppStatus orC<Npp8u>(const Npp8u*, Npp8u, Npp8u*, NppSignalLength, cudaStream_t);
template NppStatus xorC<Npp8u>(const Npp8u*, Npp8u, Npp8u*, NppSignalLength, cudaStream_t);
template NppStatus andC<Npp16u>(const Npp16u*, Npp16u, Npp16u*, NppSignalLength, cudaStream_t);
template NppStatus orC<Npp16u>(const Npp16u*, Npp16u, Npp16u*, NppSignalLength, cudaStream_t);
template NppStatus xorC<Npp16u>(const Npp16u*, Npp16u, Npp16u*, NppSignalLength, cudaStream_t);
template NppStatus andC<Npp32u>(const Npp32u*, Npp32u, Npp32u*, NppSignalLength, cudaStream_t);
template NppStatus orC<Npp32u>(const Npp32u*, Npp32u, Npp32u*, NppSignalLength, cudaStream_t);
template NppStatus xorC<Npp32u>(const Npp32u*, Npp32u, Npp32u*, NppSignalLength, cudaStream_t);
template NppStatus normalize<Npp32f, Npp32f>(const Npp32f*, Npp32f, Npp32f, Npp32f*, NppSignalLength, int, cudaStream_t);
template NppStatus normalize<Npp32fc, Npp32f>(const Npp32fc*, Npp32fc, Npp32f, Npp32fc*, NppSignalLength, int, cudaStream_t);
template NppStatus normalize<Npp64f, Npp64f>(const Npp64f*, Npp64f, Npp64f, Npp64f*, NppSignalLength, int, cudaStream_t);
template NppStatus normalize<Npp64fc, Npp64f>(const Npp64fc*, Npp64fc, Npp64f, Npp64fc*, NppSignalLength, int, cudaStream_t);
template NppStatus normalize<Npp16s, int>(const Npp16s*, Npp16s, int, Npp16s*, NppSignalLength, int, cudaStream_t);
template NppStatus normalize<Npp16sc, int>(const Npp16sc*, Npp16sc, int, Npp16sc*, NppSignalLength, int, cudaStream_t);

}  // namespace constant
}  // namespace npps
