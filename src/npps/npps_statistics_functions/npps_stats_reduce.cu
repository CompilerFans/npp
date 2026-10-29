// Executor definitions for the additive / max-based npps statistics
// families: Sum, Mean, StdDev, MeanStdDev, Norm(+Diff), DotProd, error
// metrics, CountInRange and ZeroCrossing.  Entry-point templates are
// declared in npps_statistics_executor.h; kernels live there under
// __CUDACC__.  Semantics comments and the measured NVIDIA NPP 13.4
// contracts are documented in that header.
#include "npp.h"
#include "npp_version_compat.h"
#include "npps_statistics_executor.h"

namespace npps {
namespace statistics {

// ---------------------------------------------------------------------------
// Sum / Mean / StdDev / MeanStdDev
// ---------------------------------------------------------------------------

template <typename T, typename O>
NppStatus sum(const T* pSrc, NppSignalLength nLength, O* pSum, int nScaleFactor,
              Npp8u* pDeviceBuffer, cudaStream_t stream) {
  if (!pSrc || !pSum || !pDeviceBuffer) return NPP_NULL_POINTER_ERROR;
  if (nLength <= 0) return NPP_SIZE_ERROR;
  const size_t n = static_cast<size_t>(nLength);
  const size_t grid = (n + detail::kBlock - 1) / detail::kBlock;
  using Acc = AddAcc<T>;
  (void)cudaGetLastError();
  detail::partial_add1_kernel<T, Acc, detail::LoadSum<T>>
      <<<(unsigned)grid, detail::kBlock, 0, stream>>>(
          pSrc, n, reinterpret_cast<Acc*>(pDeviceBuffer), detail::LoadSum<T>{});
  NppStatus st = npp_kernel::checkCudaKernelError();
  if (st != NPP_NO_ERROR) return st;
  detail::finalize_add_kernel<Acc, detail::FinSum<O>>
      <<<1, detail::kBlock, 0, stream>>>(
          reinterpret_cast<const Acc*>(pDeviceBuffer), grid, n,
          detail::FinSum<O>{pSum, nScaleFactor});
  return npp_kernel::checkCudaKernelError();
}

template <typename T, typename O>
NppStatus mean(const T* pSrc, NppSignalLength nLength, O* pMean, int nScaleFactor,
               Npp8u* pDeviceBuffer, cudaStream_t stream) {
  if (!pSrc || !pMean || !pDeviceBuffer) return NPP_NULL_POINTER_ERROR;
  if (nLength <= 0) return NPP_SIZE_ERROR;
  const size_t n = static_cast<size_t>(nLength);
  const size_t grid = (n + detail::kBlock - 1) / detail::kBlock;
  using Acc = AddAcc<T>;
  (void)cudaGetLastError();
  detail::partial_add1_kernel<T, Acc, detail::LoadSum<T>>
      <<<(unsigned)grid, detail::kBlock, 0, stream>>>(
          pSrc, n, reinterpret_cast<Acc*>(pDeviceBuffer), detail::LoadSum<T>{});
  NppStatus st = npp_kernel::checkCudaKernelError();
  if (st != NPP_NO_ERROR) return st;
  detail::finalize_add_kernel<Acc, detail::FinMean<O>>
      <<<1, detail::kBlock, 0, stream>>>(
          reinterpret_cast<const Acc*>(pDeviceBuffer), grid, n,
          detail::FinMean<O>{pMean, nScaleFactor});
  return npp_kernel::checkCudaKernelError();
}

template <typename T, typename O>
NppStatus stddev(const T* pSrc, NppSignalLength nLength, O* pStdDev, int nScaleFactor,
                 Npp8u* pDeviceBuffer, cudaStream_t stream) {
  if (!pSrc || !pStdDev || !pDeviceBuffer) return NPP_NULL_POINTER_ERROR;
  if (nLength < 2) return NPP_SIZE_ERROR;  // sample normalization needs n >= 2
  const size_t n = static_cast<size_t>(nLength);
  const size_t grid = (n + detail::kBlock - 1) / detail::kBlock;
  (void)cudaGetLastError();
  detail::partial_add1_kernel<T, MSD, detail::LoadMSD<T>>
      <<<(unsigned)grid, detail::kBlock, 0, stream>>>(
          pSrc, n, reinterpret_cast<MSD*>(pDeviceBuffer), detail::LoadMSD<T>{});
  NppStatus st = npp_kernel::checkCudaKernelError();
  if (st != NPP_NO_ERROR) return st;
  detail::finalize_add_kernel<MSD, detail::FinStd<O>>
      <<<1, detail::kBlock, 0, stream>>>(
          reinterpret_cast<const MSD*>(pDeviceBuffer), grid, n,
          detail::FinStd<O>{pStdDev, nScaleFactor});
  return npp_kernel::checkCudaKernelError();
}

template <typename T, typename OM, typename OS>
NppStatus mean_stddev(const T* pSrc, NppSignalLength nLength, OM* pMean, OS* pStdDev,
                      int nScaleFactor, Npp8u* pDeviceBuffer, cudaStream_t stream) {
  if (!pSrc || !pMean || !pStdDev || !pDeviceBuffer) return NPP_NULL_POINTER_ERROR;
  if (nLength < 2) return NPP_SIZE_ERROR;  // sample normalization needs n >= 2
  const size_t n = static_cast<size_t>(nLength);
  const size_t grid = (n + detail::kBlock - 1) / detail::kBlock;
  (void)cudaGetLastError();
  detail::partial_add1_kernel<T, MSD, detail::LoadMSD<T>>
      <<<(unsigned)grid, detail::kBlock, 0, stream>>>(
          pSrc, n, reinterpret_cast<MSD*>(pDeviceBuffer), detail::LoadMSD<T>{});
  NppStatus st = npp_kernel::checkCudaKernelError();
  if (st != NPP_NO_ERROR) return st;
  detail::finalize_add_kernel<MSD, detail::FinMeanStd<OM, OS>>
      <<<1, detail::kBlock, 0, stream>>>(
          reinterpret_cast<const MSD*>(pDeviceBuffer), grid, n,
          detail::FinMeanStd<OM, OS>{pMean, pStdDev, nScaleFactor});
  return npp_kernel::checkCudaKernelError();
}

// ---------------------------------------------------------------------------
// Norm / NormDiff
// ---------------------------------------------------------------------------

template <typename T, NormKind K, typename O>
NppStatus norm(const T* pSrc, NppSignalLength nLength, O* pNorm, int nScaleFactor,
               Npp8u* pDeviceBuffer, cudaStream_t stream) {
  if (!pSrc || !pNorm || !pDeviceBuffer) return NPP_NULL_POINTER_ERROR;
  if (nLength <= 0) return NPP_SIZE_ERROR;
  const size_t n = static_cast<size_t>(nLength);
  const size_t grid = (n + detail::kBlock - 1) / detail::kBlock;
  (void)cudaGetLastError();
  if constexpr (K == NormKind::Inf) {
    detail::partial_max2_kernel<T, T, detail::LoadNormInf<T>>
        <<<(unsigned)grid, detail::kBlock, 0, stream>>>(
            pSrc, pSrc, n, reinterpret_cast<double*>(pDeviceBuffer),
            detail::LoadNormInf<T>{});
    NppStatus st = npp_kernel::checkCudaKernelError();
    if (st != NPP_NO_ERROR) return st;
    detail::finalize_max_kernel<detail::FinInf<O>>
        <<<1, detail::kBlock, 0, stream>>>(
            reinterpret_cast<const double*>(pDeviceBuffer), grid, n,
            detail::FinInf<O>{pNorm, nScaleFactor});
  } else {
    using Acc = NormAcc<T>;
    detail::partial_add1_kernel<T, Acc, detail::LoadNorm<T, K>>
        <<<(unsigned)grid, detail::kBlock, 0, stream>>>(
            pSrc, n, reinterpret_cast<Acc*>(pDeviceBuffer), detail::LoadNorm<T, K>{});
    NppStatus st = npp_kernel::checkCudaKernelError();
    if (st != NPP_NO_ERROR) return st;
    detail::finalize_add_kernel<Acc, detail::FinNorm<O, K>>
        <<<1, detail::kBlock, 0, stream>>>(
            reinterpret_cast<const Acc*>(pDeviceBuffer), grid, n,
            detail::FinNorm<O, K>{pNorm, nScaleFactor});
  }
  return npp_kernel::checkCudaKernelError();
}

template <typename T, NormKind K, typename O>
NppStatus normdiff(const T* pSrc1, const T* pSrc2, NppSignalLength nLength, O* pNorm,
                   int nScaleFactor, Npp8u* pDeviceBuffer, cudaStream_t stream) {
  if (!pSrc1 || !pSrc2 || !pNorm || !pDeviceBuffer) return NPP_NULL_POINTER_ERROR;
  if (nLength <= 0) return NPP_SIZE_ERROR;
  const size_t n = static_cast<size_t>(nLength);
  const size_t grid = (n + detail::kBlock - 1) / detail::kBlock;
  (void)cudaGetLastError();
  if constexpr (K == NormKind::Inf) {
    detail::partial_max2_kernel<T, T, detail::LoadNormDiffInf<T>>
        <<<(unsigned)grid, detail::kBlock, 0, stream>>>(
            pSrc1, pSrc2, n, reinterpret_cast<double*>(pDeviceBuffer),
            detail::LoadNormDiffInf<T>{});
    NppStatus st = npp_kernel::checkCudaKernelError();
    if (st != NPP_NO_ERROR) return st;
    detail::finalize_max_kernel<detail::FinInf<O>>
        <<<1, detail::kBlock, 0, stream>>>(
            reinterpret_cast<const double*>(pDeviceBuffer), grid, n,
            detail::FinInf<O>{pNorm, nScaleFactor});
  } else {
    using Acc = NormAcc<T>;
    detail::partial_add2_kernel<T, T, Acc, detail::LoadNormDiff<T, K>>
        <<<(unsigned)grid, detail::kBlock, 0, stream>>>(
            pSrc1, pSrc2, n, reinterpret_cast<Acc*>(pDeviceBuffer),
            detail::LoadNormDiff<T, K>{});
    NppStatus st = npp_kernel::checkCudaKernelError();
    if (st != NPP_NO_ERROR) return st;
    detail::finalize_add_kernel<Acc, detail::FinNorm<O, K>>
        <<<1, detail::kBlock, 0, stream>>>(
            reinterpret_cast<const Acc*>(pDeviceBuffer), grid, n,
            detail::FinNorm<O, K>{pNorm, nScaleFactor});
  }
  return npp_kernel::checkCudaKernelError();
}

// ---------------------------------------------------------------------------
// DotProd
// ---------------------------------------------------------------------------

template <typename A, typename B, typename O>
NppStatus dotprod(const A* pSrc1, const B* pSrc2, NppSignalLength nLength, O* pDp,
                  int nScaleFactor, Npp8u* pDeviceBuffer, cudaStream_t stream) {
  if (!pSrc1 || !pSrc2 || !pDp || !pDeviceBuffer) return NPP_NULL_POINTER_ERROR;
  if (nLength <= 0) return NPP_SIZE_ERROR;
  const size_t n = static_cast<size_t>(nLength);
  const size_t grid = (n + detail::kBlock - 1) / detail::kBlock;
  (void)cudaGetLastError();
  detail::partial_add2_kernel<A, B, D2, detail::LoadDot<A, B>>
      <<<(unsigned)grid, detail::kBlock, 0, stream>>>(
          pSrc1, pSrc2, n, reinterpret_cast<D2*>(pDeviceBuffer),
          detail::LoadDot<A, B>{});
  NppStatus st = npp_kernel::checkCudaKernelError();
  if (st != NPP_NO_ERROR) return st;
  detail::finalize_add_kernel<D2, detail::FinDot<O>>
      <<<1, detail::kBlock, 0, stream>>>(
          reinterpret_cast<const D2*>(pDeviceBuffer), grid, n,
          detail::FinDot<O>{pDp, nScaleFactor});
  return npp_kernel::checkCudaKernelError();
}

// ---------------------------------------------------------------------------
// Error metrics
// ---------------------------------------------------------------------------

template <typename T, ErrKind K>
NppStatus error_metric(const T* pSrc1, const T* pSrc2, NppSignalLength nLength, Npp64f* pDst,
                       Npp8u* pDeviceBuffer, cudaStream_t stream) {
  if (!pSrc1 || !pSrc2 || !pDst || !pDeviceBuffer) return NPP_NULL_POINTER_ERROR;
  if (nLength <= 0) return NPP_SIZE_ERROR;
  const size_t n = static_cast<size_t>(nLength);
  const size_t grid = (n + detail::kBlock - 1) / detail::kBlock;
  constexpr bool kAvg = (K == ErrKind::AvgAbs || K == ErrKind::AvgRel);
  (void)cudaGetLastError();
  detail::partial_max2_kernel<T, T, detail::LoadErr<T, K>>
      <<<(unsigned)grid, detail::kBlock, 0, stream>>>(
          pSrc1, pSrc2, n, reinterpret_cast<double*>(pDeviceBuffer),
          detail::LoadErr<T, K>{});
  NppStatus st = npp_kernel::checkCudaKernelError();
  if (st != NPP_NO_ERROR) return st;
  detail::finalize_max_kernel<detail::FinErr>
      <<<1, detail::kBlock, 0, stream>>>(
          reinterpret_cast<const double*>(pDeviceBuffer), grid, n,
          detail::FinErr{pDst, kAvg});
  return npp_kernel::checkCudaKernelError();
}

// ---------------------------------------------------------------------------
// CountInRange
// ---------------------------------------------------------------------------

NppStatus count_in_range(const Npp32s* pSrc, NppSignalLength nLength, int* pCounts,
                         Npp32s nLowerBound, Npp32s nUpperBound, Npp8u* pDeviceBuffer,
                         cudaStream_t stream) {
  if (!pSrc || !pCounts || !pDeviceBuffer) return NPP_NULL_POINTER_ERROR;
  if (nLength <= 0) return NPP_SIZE_ERROR;
  const size_t n = static_cast<size_t>(nLength);
  const size_t grid = (n + detail::kBlock - 1) / detail::kBlock;
  (void)cudaGetLastError();
  detail::count_partial_kernel<<<(unsigned)grid, detail::kBlock, 0, stream>>>(
      pSrc, n, nLowerBound, nUpperBound, reinterpret_cast<int*>(pDeviceBuffer));
  NppStatus st = npp_kernel::checkCudaKernelError();
  if (st != NPP_NO_ERROR) return st;
  detail::finalize_add_kernel<int, detail::FinCount>
      <<<1, detail::kBlock, 0, stream>>>(
          reinterpret_cast<const int*>(pDeviceBuffer), grid, n, detail::FinCount{pCounts});
  return npp_kernel::checkCudaKernelError();
}

// ---------------------------------------------------------------------------
// ZeroCrossing
// ---------------------------------------------------------------------------

template <typename T>
NppStatus zero_crossing(const T* pSrc, NppSignalLength nLength, Npp32f* pValZC,
                        NppsZCType tZCType, Npp8u* pDeviceBuffer, cudaStream_t stream) {
  if (!pSrc || !pValZC || !pDeviceBuffer) return NPP_NULL_POINTER_ERROR;
  if (nLength <= 0) return NPP_SIZE_ERROR;
  const size_t n = static_cast<size_t>(nLength);
  const size_t grid = (n + detail::kBlock - 1) / detail::kBlock;
  (void)cudaGetLastError();
  detail::zc_partial_kernel<T>
      <<<(unsigned)grid, detail::kBlock, 0, stream>>>(
          pSrc, n, tZCType, reinterpret_cast<double*>(pDeviceBuffer));
  NppStatus st = npp_kernel::checkCudaKernelError();
  if (st != NPP_NO_ERROR) return st;
  detail::finalize_add_kernel<double, detail::FinZC>
      <<<1, detail::kBlock, 0, stream>>>(
          reinterpret_cast<const double*>(pDeviceBuffer), grid, n,
          detail::FinZC{pValZC});
  return npp_kernel::checkCudaKernelError();
}

// ---------------------------------------------------------------------------
// Explicit instantiations (one per API function variant)
// ---------------------------------------------------------------------------

// Sum
template NppStatus sum<Npp16s, Npp32s>(const Npp16s*, NppSignalLength, Npp32s*, int, Npp8u*, cudaStream_t);
template NppStatus sum<Npp16sc, Npp32sc>(const Npp16sc*, NppSignalLength, Npp32sc*, int, Npp8u*, cudaStream_t);
template NppStatus sum<Npp16sc, Npp16sc>(const Npp16sc*, NppSignalLength, Npp16sc*, int, Npp8u*, cudaStream_t);
template NppStatus sum<Npp16s, Npp16s>(const Npp16s*, NppSignalLength, Npp16s*, int, Npp8u*, cudaStream_t);
template NppStatus sum<Npp32fc, Npp32fc>(const Npp32fc*, NppSignalLength, Npp32fc*, int, Npp8u*, cudaStream_t);
template NppStatus sum<Npp32f, Npp32f>(const Npp32f*, NppSignalLength, Npp32f*, int, Npp8u*, cudaStream_t);
template NppStatus sum<Npp32s, Npp32s>(const Npp32s*, NppSignalLength, Npp32s*, int, Npp8u*, cudaStream_t);
template NppStatus sum<Npp64fc, Npp64fc>(const Npp64fc*, NppSignalLength, Npp64fc*, int, Npp8u*, cudaStream_t);
template NppStatus sum<Npp64f, Npp64f>(const Npp64f*, NppSignalLength, Npp64f*, int, Npp8u*, cudaStream_t);

// Mean
template NppStatus mean<Npp16sc, Npp16sc>(const Npp16sc*, NppSignalLength, Npp16sc*, int, Npp8u*, cudaStream_t);
template NppStatus mean<Npp16s, Npp16s>(const Npp16s*, NppSignalLength, Npp16s*, int, Npp8u*, cudaStream_t);
template NppStatus mean<Npp32fc, Npp32fc>(const Npp32fc*, NppSignalLength, Npp32fc*, int, Npp8u*, cudaStream_t);
template NppStatus mean<Npp32f, Npp32f>(const Npp32f*, NppSignalLength, Npp32f*, int, Npp8u*, cudaStream_t);
template NppStatus mean<Npp32s, Npp32s>(const Npp32s*, NppSignalLength, Npp32s*, int, Npp8u*, cudaStream_t);
template NppStatus mean<Npp64fc, Npp64fc>(const Npp64fc*, NppSignalLength, Npp64fc*, int, Npp8u*, cudaStream_t);
template NppStatus mean<Npp64f, Npp64f>(const Npp64f*, NppSignalLength, Npp64f*, int, Npp8u*, cudaStream_t);

// StdDev
template NppStatus stddev<Npp16s, Npp32s>(const Npp16s*, NppSignalLength, Npp32s*, int, Npp8u*, cudaStream_t);
template NppStatus stddev<Npp16s, Npp16s>(const Npp16s*, NppSignalLength, Npp16s*, int, Npp8u*, cudaStream_t);
template NppStatus stddev<Npp32f, Npp32f>(const Npp32f*, NppSignalLength, Npp32f*, int, Npp8u*, cudaStream_t);
template NppStatus stddev<Npp64f, Npp64f>(const Npp64f*, NppSignalLength, Npp64f*, int, Npp8u*, cudaStream_t);

// MeanStdDev
template NppStatus mean_stddev<Npp16s, Npp32s, Npp32s>(const Npp16s*, NppSignalLength, Npp32s*, Npp32s*, int, Npp8u*, cudaStream_t);
template NppStatus mean_stddev<Npp16s, Npp16s, Npp16s>(const Npp16s*, NppSignalLength, Npp16s*, Npp16s*, int, Npp8u*, cudaStream_t);
template NppStatus mean_stddev<Npp32f, Npp32f, Npp32f>(const Npp32f*, NppSignalLength, Npp32f*, Npp32f*, int, Npp8u*, cudaStream_t);
template NppStatus mean_stddev<Npp64f, Npp64f, Npp64f>(const Npp64f*, NppSignalLength, Npp64f*, Npp64f*, int, Npp8u*, cudaStream_t);

// Norm L1
template NppStatus norm<Npp16s, NormKind::L1, Npp32f>(const Npp16s*, NppSignalLength, Npp32f*, int, Npp8u*, cudaStream_t);
template NppStatus norm<Npp16s, NormKind::L1, Npp32s>(const Npp16s*, NppSignalLength, Npp32s*, int, Npp8u*, cudaStream_t);
template NppStatus norm<Npp16s, NormKind::L1, Npp64s>(const Npp16s*, NppSignalLength, Npp64s*, int, Npp8u*, cudaStream_t);
template NppStatus norm<Npp32fc, NormKind::L1, Npp64f>(const Npp32fc*, NppSignalLength, Npp64f*, int, Npp8u*, cudaStream_t);
template NppStatus norm<Npp32f, NormKind::L1, Npp32f>(const Npp32f*, NppSignalLength, Npp32f*, int, Npp8u*, cudaStream_t);
template NppStatus norm<Npp64fc, NormKind::L1, Npp64f>(const Npp64fc*, NppSignalLength, Npp64f*, int, Npp8u*, cudaStream_t);
template NppStatus norm<Npp64f, NormKind::L1, Npp64f>(const Npp64f*, NppSignalLength, Npp64f*, int, Npp8u*, cudaStream_t);
// Norm L2
template NppStatus norm<Npp16s, NormKind::L2, Npp32f>(const Npp16s*, NppSignalLength, Npp32f*, int, Npp8u*, cudaStream_t);
template NppStatus norm<Npp16s, NormKind::L2, Npp32s>(const Npp16s*, NppSignalLength, Npp32s*, int, Npp8u*, cudaStream_t);
template NppStatus norm<Npp32fc, NormKind::L2, Npp64f>(const Npp32fc*, NppSignalLength, Npp64f*, int, Npp8u*, cudaStream_t);
template NppStatus norm<Npp32f, NormKind::L2, Npp32f>(const Npp32f*, NppSignalLength, Npp32f*, int, Npp8u*, cudaStream_t);
template NppStatus norm<Npp64fc, NormKind::L2, Npp64f>(const Npp64fc*, NppSignalLength, Npp64f*, int, Npp8u*, cudaStream_t);
template NppStatus norm<Npp64f, NormKind::L2, Npp64f>(const Npp64f*, NppSignalLength, Npp64f*, int, Npp8u*, cudaStream_t);
// Norm L2Sqr
template NppStatus norm<Npp16s, NormKind::L2Sqr, Npp64s>(const Npp16s*, NppSignalLength, Npp64s*, int, Npp8u*, cudaStream_t);
// Norm Inf
template NppStatus norm<Npp16s, NormKind::Inf, Npp32f>(const Npp16s*, NppSignalLength, Npp32f*, int, Npp8u*, cudaStream_t);
template NppStatus norm<Npp16s, NormKind::Inf, Npp32s>(const Npp16s*, NppSignalLength, Npp32s*, int, Npp8u*, cudaStream_t);
template NppStatus norm<Npp32fc, NormKind::Inf, Npp32f>(const Npp32fc*, NppSignalLength, Npp32f*, int, Npp8u*, cudaStream_t);
template NppStatus norm<Npp32f, NormKind::Inf, Npp32f>(const Npp32f*, NppSignalLength, Npp32f*, int, Npp8u*, cudaStream_t);
template NppStatus norm<Npp64fc, NormKind::Inf, Npp64f>(const Npp64fc*, NppSignalLength, Npp64f*, int, Npp8u*, cudaStream_t);
template NppStatus norm<Npp64f, NormKind::Inf, Npp64f>(const Npp64f*, NppSignalLength, Npp64f*, int, Npp8u*, cudaStream_t);

// NormDiff L1
template NppStatus normdiff<Npp16s, NormKind::L1, Npp32f>(const Npp16s*, const Npp16s*, NppSignalLength, Npp32f*, int, Npp8u*, cudaStream_t);
template NppStatus normdiff<Npp16s, NormKind::L1, Npp32s>(const Npp16s*, const Npp16s*, NppSignalLength, Npp32s*, int, Npp8u*, cudaStream_t);
template NppStatus normdiff<Npp16s, NormKind::L1, Npp64s>(const Npp16s*, const Npp16s*, NppSignalLength, Npp64s*, int, Npp8u*, cudaStream_t);
template NppStatus normdiff<Npp32fc, NormKind::L1, Npp64f>(const Npp32fc*, const Npp32fc*, NppSignalLength, Npp64f*, int, Npp8u*, cudaStream_t);
template NppStatus normdiff<Npp32f, NormKind::L1, Npp32f>(const Npp32f*, const Npp32f*, NppSignalLength, Npp32f*, int, Npp8u*, cudaStream_t);
template NppStatus normdiff<Npp64fc, NormKind::L1, Npp64f>(const Npp64fc*, const Npp64fc*, NppSignalLength, Npp64f*, int, Npp8u*, cudaStream_t);
template NppStatus normdiff<Npp64f, NormKind::L1, Npp64f>(const Npp64f*, const Npp64f*, NppSignalLength, Npp64f*, int, Npp8u*, cudaStream_t);
// NormDiff L2
template NppStatus normdiff<Npp16s, NormKind::L2, Npp32f>(const Npp16s*, const Npp16s*, NppSignalLength, Npp32f*, int, Npp8u*, cudaStream_t);
template NppStatus normdiff<Npp16s, NormKind::L2, Npp32s>(const Npp16s*, const Npp16s*, NppSignalLength, Npp32s*, int, Npp8u*, cudaStream_t);
template NppStatus normdiff<Npp32fc, NormKind::L2, Npp64f>(const Npp32fc*, const Npp32fc*, NppSignalLength, Npp64f*, int, Npp8u*, cudaStream_t);
template NppStatus normdiff<Npp32f, NormKind::L2, Npp32f>(const Npp32f*, const Npp32f*, NppSignalLength, Npp32f*, int, Npp8u*, cudaStream_t);
template NppStatus normdiff<Npp64fc, NormKind::L2, Npp64f>(const Npp64fc*, const Npp64fc*, NppSignalLength, Npp64f*, int, Npp8u*, cudaStream_t);
template NppStatus normdiff<Npp64f, NormKind::L2, Npp64f>(const Npp64f*, const Npp64f*, NppSignalLength, Npp64f*, int, Npp8u*, cudaStream_t);
// NormDiff L2Sqr
template NppStatus normdiff<Npp16s, NormKind::L2Sqr, Npp64s>(const Npp16s*, const Npp16s*, NppSignalLength, Npp64s*, int, Npp8u*, cudaStream_t);
// NormDiff Inf
template NppStatus normdiff<Npp16s, NormKind::Inf, Npp32f>(const Npp16s*, const Npp16s*, NppSignalLength, Npp32f*, int, Npp8u*, cudaStream_t);
template NppStatus normdiff<Npp16s, NormKind::Inf, Npp32s>(const Npp16s*, const Npp16s*, NppSignalLength, Npp32s*, int, Npp8u*, cudaStream_t);
template NppStatus normdiff<Npp32fc, NormKind::Inf, Npp32f>(const Npp32fc*, const Npp32fc*, NppSignalLength, Npp32f*, int, Npp8u*, cudaStream_t);
template NppStatus normdiff<Npp32f, NormKind::Inf, Npp32f>(const Npp32f*, const Npp32f*, NppSignalLength, Npp32f*, int, Npp8u*, cudaStream_t);
template NppStatus normdiff<Npp64fc, NormKind::Inf, Npp64f>(const Npp64fc*, const Npp64fc*, NppSignalLength, Npp64f*, int, Npp8u*, cudaStream_t);
template NppStatus normdiff<Npp64f, NormKind::Inf, Npp64f>(const Npp64f*, const Npp64f*, NppSignalLength, Npp64f*, int, Npp8u*, cudaStream_t);

// DotProd
template NppStatus dotprod<Npp16s, Npp16sc, Npp32fc>(const Npp16s*, const Npp16sc*, NppSignalLength, Npp32fc*, int, Npp8u*, cudaStream_t);
template NppStatus dotprod<Npp16s, Npp16sc, Npp32sc>(const Npp16s*, const Npp16sc*, NppSignalLength, Npp32sc*, int, Npp8u*, cudaStream_t);
template NppStatus dotprod<Npp16s, Npp16sc, Npp64sc>(const Npp16s*, const Npp16sc*, NppSignalLength, Npp64sc*, int, Npp8u*, cudaStream_t);
template NppStatus dotprod<Npp16s, Npp16sc, Npp16sc>(const Npp16s*, const Npp16sc*, NppSignalLength, Npp16sc*, int, Npp8u*, cudaStream_t);
template NppStatus dotprod<Npp16s, Npp16s, Npp32f>(const Npp16s*, const Npp16s*, NppSignalLength, Npp32f*, int, Npp8u*, cudaStream_t);
template NppStatus dotprod<Npp16s, Npp32s, Npp32s>(const Npp16s*, const Npp32s*, NppSignalLength, Npp32s*, int, Npp8u*, cudaStream_t);
template NppStatus dotprod<Npp16s, Npp16s, Npp32s>(const Npp16s*, const Npp16s*, NppSignalLength, Npp32s*, int, Npp8u*, cudaStream_t);
template NppStatus dotprod<Npp16s, Npp16s, Npp64s>(const Npp16s*, const Npp16s*, NppSignalLength, Npp64s*, int, Npp8u*, cudaStream_t);
template NppStatus dotprod<Npp16sc, Npp16sc, Npp32fc>(const Npp16sc*, const Npp16sc*, NppSignalLength, Npp32fc*, int, Npp8u*, cudaStream_t);
template NppStatus dotprod<Npp16sc, Npp16sc, Npp32sc>(const Npp16sc*, const Npp16sc*, NppSignalLength, Npp32sc*, int, Npp8u*, cudaStream_t);
template NppStatus dotprod<Npp16sc, Npp16sc, Npp64sc>(const Npp16sc*, const Npp16sc*, NppSignalLength, Npp64sc*, int, Npp8u*, cudaStream_t);
template NppStatus dotprod<Npp16sc, Npp16sc, Npp16sc>(const Npp16sc*, const Npp16sc*, NppSignalLength, Npp16sc*, int, Npp8u*, cudaStream_t);
template NppStatus dotprod<Npp16s, Npp16s, Npp16s>(const Npp16s*, const Npp16s*, NppSignalLength, Npp16s*, int, Npp8u*, cudaStream_t);
template NppStatus dotprod<Npp32f, Npp32fc, Npp64fc>(const Npp32f*, const Npp32fc*, NppSignalLength, Npp64fc*, int, Npp8u*, cudaStream_t);
template NppStatus dotprod<Npp32f, Npp32fc, Npp32fc>(const Npp32f*, const Npp32fc*, NppSignalLength, Npp32fc*, int, Npp8u*, cudaStream_t);
template NppStatus dotprod<Npp32f, Npp32f, Npp64f>(const Npp32f*, const Npp32f*, NppSignalLength, Npp64f*, int, Npp8u*, cudaStream_t);
template NppStatus dotprod<Npp32fc, Npp32fc, Npp64fc>(const Npp32fc*, const Npp32fc*, NppSignalLength, Npp64fc*, int, Npp8u*, cudaStream_t);
template NppStatus dotprod<Npp32fc, Npp32fc, Npp32fc>(const Npp32fc*, const Npp32fc*, NppSignalLength, Npp32fc*, int, Npp8u*, cudaStream_t);
template NppStatus dotprod<Npp32f, Npp32f, Npp32f>(const Npp32f*, const Npp32f*, NppSignalLength, Npp32f*, int, Npp8u*, cudaStream_t);
template NppStatus dotprod<Npp32s, Npp32sc, Npp32sc>(const Npp32s*, const Npp32sc*, NppSignalLength, Npp32sc*, int, Npp8u*, cudaStream_t);
template NppStatus dotprod<Npp32sc, Npp32sc, Npp32sc>(const Npp32sc*, const Npp32sc*, NppSignalLength, Npp32sc*, int, Npp8u*, cudaStream_t);
template NppStatus dotprod<Npp32s, Npp32s, Npp32s>(const Npp32s*, const Npp32s*, NppSignalLength, Npp32s*, int, Npp8u*, cudaStream_t);
template NppStatus dotprod<Npp64f, Npp64fc, Npp64fc>(const Npp64f*, const Npp64fc*, NppSignalLength, Npp64fc*, int, Npp8u*, cudaStream_t);
template NppStatus dotprod<Npp64fc, Npp64fc, Npp64fc>(const Npp64fc*, const Npp64fc*, NppSignalLength, Npp64fc*, int, Npp8u*, cudaStream_t);
template NppStatus dotprod<Npp64f, Npp64f, Npp64f>(const Npp64f*, const Npp64f*, NppSignalLength, Npp64f*, int, Npp8u*, cudaStream_t);

// Error metrics (4 kinds x 14 dtypes)
#define NPPS_ERR_INST(T)                                                        \
  template NppStatus error_metric<T, ErrKind::MaxAbs>(const T*, const T*, NppSignalLength, Npp64f*, Npp8u*, cudaStream_t); \
  template NppStatus error_metric<T, ErrKind::AvgAbs>(const T*, const T*, NppSignalLength, Npp64f*, Npp8u*, cudaStream_t); \
  template NppStatus error_metric<T, ErrKind::MaxRel>(const T*, const T*, NppSignalLength, Npp64f*, Npp8u*, cudaStream_t); \
  template NppStatus error_metric<T, ErrKind::AvgRel>(const T*, const T*, NppSignalLength, Npp64f*, Npp8u*, cudaStream_t);
NPPS_ERR_INST(Npp8u)
NPPS_ERR_INST(Npp8s)
NPPS_ERR_INST(Npp16u)
NPPS_ERR_INST(Npp16s)
NPPS_ERR_INST(Npp32u)
NPPS_ERR_INST(Npp32s)
NPPS_ERR_INST(Npp64s)
NPPS_ERR_INST(Npp32f)
NPPS_ERR_INST(Npp64f)
NPPS_ERR_INST(Npp16sc)
NPPS_ERR_INST(Npp32sc)
NPPS_ERR_INST(Npp64sc)
NPPS_ERR_INST(Npp32fc)
NPPS_ERR_INST(Npp64fc)
#undef NPPS_ERR_INST

// ZeroCrossing
template NppStatus zero_crossing<Npp32f>(const Npp32f*, NppSignalLength, Npp32f*, NppsZCType, Npp8u*, cudaStream_t);
template NppStatus zero_crossing<Npp16s>(const Npp16s*, NppSignalLength, Npp32f*, NppsZCType, Npp8u*, cudaStream_t);

}  // namespace statistics
}  // namespace npps
