#include "npp.h"
#include "npp_version_compat.h"
#include <algorithm>
#include <cstring>
#include <cuda_runtime.h>

// Kernel declarations (implemented in .cu file)

extern "C" {
cudaError_t nppsSum_32f_kernel(const Npp32f *pSrc, size_t nLength, Npp32f *pSum, Npp8u *pDeviceBuffer,
                               cudaStream_t stream);
cudaError_t nppsSum_32fc_kernel(const Npp32fc *pSrc, size_t nLength, Npp32fc *pSum, Npp8u *pDeviceBuffer,
                                cudaStream_t stream);
}

// ==============================================================================
// Internal implementation functions
// ==============================================================================

namespace {

template <typename T> NppStatus nppsSumGetBufferSizeImpl(size_t nLength, NppSignalLength *hpBufferSize) {
  if (!hpBufferSize) {
    return NPP_NULL_POINTER_ERROR;
  }
  if (nLength == 0) {
    return NPP_SIZE_ERROR;
  }

  const int blockSize = 256;
  const size_t gridSize = (nLength + blockSize - 1) / blockSize;
  *hpBufferSize = static_cast<NppSignalLength>(gridSize * sizeof(T));

  return NPP_NO_ERROR;
}

template <typename T>
NppStatus nppsSumImpl(const T *pSrc, size_t nLength, T *pSum, Npp8u *pDeviceBuffer, cudaStream_t stream);

template <>
NppStatus nppsSumImpl<Npp32f>(const Npp32f *pSrc, size_t nLength, Npp32f *pSum, Npp8u *pDeviceBuffer,
                              cudaStream_t stream) {
  if (!pSrc || !pSum || !pDeviceBuffer) {
    return NPP_NULL_POINTER_ERROR;
  }
  if (nLength == 0) {
    return NPP_SIZE_ERROR;
  }

  cudaError_t cudaStatus = nppsSum_32f_kernel(pSrc, nLength, pSum, pDeviceBuffer, stream);
  return (cudaStatus == cudaSuccess) ? NPP_NO_ERROR : NPP_CUDA_KERNEL_EXECUTION_ERROR;
}

template <>
NppStatus nppsSumImpl<Npp32fc>(const Npp32fc *pSrc, size_t nLength, Npp32fc *pSum, Npp8u *pDeviceBuffer,
                               cudaStream_t stream) {
  if (!pSrc || !pSum || !pDeviceBuffer) {
    return NPP_NULL_POINTER_ERROR;
  }
  if (nLength == 0) {
    return NPP_SIZE_ERROR;
  }

  cudaError_t cudaStatus = nppsSum_32fc_kernel(pSrc, nLength, pSum, pDeviceBuffer, stream);
  return (cudaStatus == cudaSuccess) ? NPP_NO_ERROR : NPP_CUDA_KERNEL_EXECUTION_ERROR;
}

} // namespace

// ==============================================================================
// Buffer Size Functions - Required for reduction operations
// ==============================================================================

NppStatus nppsSumGetBufferSize_32f_Ctx(NppSignalLength nLength, NppSignalLength *hpBufferSize,
                                       NppStreamContext nppStreamCtx) {
  (void)nppStreamCtx;
  return nppsSumGetBufferSizeImpl<Npp32f>(static_cast<size_t>(nLength), hpBufferSize);
}

NppStatus nppsSumGetBufferSize_32f(NppSignalLength nLength, NppSignalLength *hpBufferSize) {
  return nppsSumGetBufferSize_32f_Ctx(nLength, hpBufferSize, {});
}

NppStatus nppsSumGetBufferSize_32fc_Ctx(NppSignalLength nLength, NppSignalLength *hpBufferSize,
                                        NppStreamContext nppStreamCtx) {
  (void)nppStreamCtx;
  return nppsSumGetBufferSizeImpl<Npp32fc>(static_cast<size_t>(nLength), hpBufferSize);
}

NppStatus nppsSumGetBufferSize_32fc(NppSignalLength nLength, NppSignalLength *hpBufferSize) {
  return nppsSumGetBufferSize_32fc_Ctx(nLength, hpBufferSize, {});
}

// ==============================================================================
// Sum Operations - Element-wise summation of signal values
// ==============================================================================

NppStatus nppsSum_32f_Ctx(const Npp32f *pSrc, NppSignalLength nLength, Npp32f *pSum, Npp8u *pDeviceBuffer,
                          NppStreamContext nppStreamCtx) {
  return nppsSumImpl<Npp32f>(pSrc, static_cast<size_t>(nLength), pSum, pDeviceBuffer, nppStreamCtx.hStream);
}

NppStatus nppsSum_32f(const Npp32f *pSrc, NppSignalLength nLength, Npp32f *pSum, Npp8u *pDeviceBuffer) {
  NppStreamContext defaultContext;
  NppStatus status = nppGetStreamContext(&defaultContext);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppsSum_32f_Ctx(pSrc, nLength, pSum, pDeviceBuffer, defaultContext);
}

NppStatus nppsSum_32fc_Ctx(const Npp32fc *pSrc, NppSignalLength nLength, Npp32fc *pSum, Npp8u *pDeviceBuffer,
                           NppStreamContext nppStreamCtx) {
  return nppsSumImpl<Npp32fc>(pSrc, static_cast<size_t>(nLength), pSum, pDeviceBuffer, nppStreamCtx.hStream);
}

NppStatus nppsSum_32fc(const Npp32fc *pSrc, NppSignalLength nLength, Npp32fc *pSum, Npp8u *pDeviceBuffer) {
  NppStreamContext defaultContext;
  NppStatus status = nppGetStreamContext(&defaultContext);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppsSum_32fc_Ctx(pSrc, nLength, pSum, pDeviceBuffer, defaultContext);
}
