// Executor definitions for the min/max npps statistics families:
// Min/Max (+Indx/Abs variants), MinMax(+Indx) and MinEvery/MaxEvery.
// Entry-point templates are declared in npps_statistics_executor.h; the
// kernels and the measured NVIDIA NPP 13.4 comparator semantics are
// documented there.
#include "npp.h"
#include "npp_version_compat.h"
#include "npps_statistics_executor.h"

namespace npps {
namespace statistics {

template <typename T, bool AbsOnly>
NppStatus extrema(const T* pSrc, NppSignalLength nLength, T* pMin, int* pMinIndx, T* pMax,
                  int* pMaxIndx, bool wantMin, bool wantMax, bool wantIndx,
                  Npp8u* pDeviceBuffer, cudaStream_t stream) {
  if (!pSrc || !pDeviceBuffer) return NPP_NULL_POINTER_ERROR;
  if (nLength <= 0) return NPP_SIZE_ERROR;
  // NVIDIA rejects a null pointer for every output the API function
  // declares, so the want* flags drive the validation here.
  if (!wantMin && !wantMax) return NPP_NULL_POINTER_ERROR;
  if (wantMin) {
    if (!pMin) return NPP_NULL_POINTER_ERROR;
    if (wantIndx && !pMinIndx) return NPP_NULL_POINTER_ERROR;
  }
  if (wantMax) {
    if (!pMax) return NPP_NULL_POINTER_ERROR;
    if (wantIndx && !pMaxIndx) return NPP_NULL_POINTER_ERROR;
  }
  const size_t n = static_cast<size_t>(nLength);
  const size_t grid = (n + detail::kBlock - 1) / detail::kBlock;
  (void)cudaGetLastError();
  detail::partial_ext_kernel<T, AbsOnly>
      <<<(unsigned)grid, detail::kBlock, 0, stream>>>(
          pSrc, n, reinterpret_cast<detail::ExtSlot<T>*>(pDeviceBuffer));
  NppStatus st = npp_kernel::checkCudaKernelError();
  if (st != NPP_NO_ERROR) return st;
  detail::finalize_ext_kernel<T, detail::FinExt<T>>
      <<<1, detail::kBlock, 0, stream>>>(
          reinterpret_cast<const detail::ExtSlot<T>*>(pDeviceBuffer), grid,
          detail::FinExt<T>{pMin, pMinIndx, pMax, pMaxIndx});
  return npp_kernel::checkCudaKernelError();
}

template <typename T, bool IsMax>
NppStatus every(const T* pSrc, T* pSrcDst, NppSignalLength nLength, cudaStream_t stream) {
  // NVIDIA returns NPP_NO_ERROR for a null pSrc but then faults the kernel
  // (measured 13.4 behaviour); we reject the null pointer instead.
  if (!pSrc || !pSrcDst) return NPP_NULL_POINTER_ERROR;
  if (nLength <= 0) return NPP_SIZE_ERROR;
  const size_t n = static_cast<size_t>(nLength);
  const size_t grid = (n + detail::kBlock - 1) / detail::kBlock;
  (void)cudaGetLastError();
  detail::every_kernel<T, IsMax><<<(unsigned)grid, detail::kBlock, 0, stream>>>(pSrc, pSrcDst, n);
  return npp_kernel::checkCudaKernelError();
}

// ---------------------------------------------------------------------------
// Explicit instantiations
// ---------------------------------------------------------------------------

// extrema (Min/Max/Indx/MinMax/MinMaxIndx/MinAbs/MaxAbs/MinAbsIndx/MaxAbsIndx)
template NppStatus extrema<Npp8u, false>(const Npp8u*, NppSignalLength, Npp8u*, int*, Npp8u*, int*, bool, bool, bool, Npp8u*, cudaStream_t);
template NppStatus extrema<Npp16u, false>(const Npp16u*, NppSignalLength, Npp16u*, int*, Npp16u*, int*, bool, bool, bool, Npp8u*, cudaStream_t);
template NppStatus extrema<Npp16s, false>(const Npp16s*, NppSignalLength, Npp16s*, int*, Npp16s*, int*, bool, bool, bool, Npp8u*, cudaStream_t);
template NppStatus extrema<Npp32u, false>(const Npp32u*, NppSignalLength, Npp32u*, int*, Npp32u*, int*, bool, bool, bool, Npp8u*, cudaStream_t);
template NppStatus extrema<Npp32s, false>(const Npp32s*, NppSignalLength, Npp32s*, int*, Npp32s*, int*, bool, bool, bool, Npp8u*, cudaStream_t);
template NppStatus extrema<Npp32f, false>(const Npp32f*, NppSignalLength, Npp32f*, int*, Npp32f*, int*, bool, bool, bool, Npp8u*, cudaStream_t);
template NppStatus extrema<Npp64f, false>(const Npp64f*, NppSignalLength, Npp64f*, int*, Npp64f*, int*, bool, bool, bool, Npp8u*, cudaStream_t);
template NppStatus extrema<Npp16s, true>(const Npp16s*, NppSignalLength, Npp16s*, int*, Npp16s*, int*, bool, bool, bool, Npp8u*, cudaStream_t);
template NppStatus extrema<Npp32s, true>(const Npp32s*, NppSignalLength, Npp32s*, int*, Npp32s*, int*, bool, bool, bool, Npp8u*, cudaStream_t);

// MinEvery / MaxEvery
template NppStatus every<Npp8u, false>(const Npp8u*, Npp8u*, NppSignalLength, cudaStream_t);
template NppStatus every<Npp16s, false>(const Npp16s*, Npp16s*, NppSignalLength, cudaStream_t);
template NppStatus every<Npp16u, false>(const Npp16u*, Npp16u*, NppSignalLength, cudaStream_t);
template NppStatus every<Npp32f, false>(const Npp32f*, Npp32f*, NppSignalLength, cudaStream_t);
template NppStatus every<Npp32s, false>(const Npp32s*, Npp32s*, NppSignalLength, cudaStream_t);
template NppStatus every<Npp64f, false>(const Npp64f*, Npp64f*, NppSignalLength, cudaStream_t);
template NppStatus every<Npp8u, true>(const Npp8u*, Npp8u*, NppSignalLength, cudaStream_t);
template NppStatus every<Npp16s, true>(const Npp16s*, Npp16s*, NppSignalLength, cudaStream_t);
template NppStatus every<Npp16u, true>(const Npp16u*, Npp16u*, NppSignalLength, cudaStream_t);
template NppStatus every<Npp32f, true>(const Npp32f*, Npp32f*, NppSignalLength, cudaStream_t);
template NppStatus every<Npp32s, true>(const Npp32s*, Npp32s*, NppSignalLength, cudaStream_t);

}  // namespace statistics
}  // namespace npps
