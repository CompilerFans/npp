#include "npp.h"
#include "npp_version_compat.h"
#include <cstring>
#include <cuda_runtime.h>

// Kernel declarations (implemented in .cu file)

extern "C" {
cudaError_t nppsAdd_32f_kernel(const Npp32f *pSrc1, const Npp32f *pSrc2, Npp32f *pDst, size_t nLength,
                               cudaStream_t stream);
cudaError_t nppsAdd_16s_kernel(const Npp16s *pSrc1, const Npp16s *pSrc2, Npp16s *pDst, size_t nLength,
                               cudaStream_t stream);
cudaError_t nppsAdd_32fc_kernel(const Npp32fc *pSrc1, const Npp32fc *pSrc2, Npp32fc *pDst, size_t nLength,
                                cudaStream_t stream);
}

// ==============================================================================
// Internal implementation functions
// ==============================================================================

namespace {

template <typename T>
NppStatus nppsAddImpl(const T *pSrc1, const T *pSrc2, T *pDst, size_t nLength, cudaStream_t stream);

template <>
NppStatus nppsAddImpl<Npp32f>(const Npp32f *pSrc1, const Npp32f *pSrc2, Npp32f *pDst, size_t nLength,
                              cudaStream_t stream) {
  if (!pSrc1 || !pSrc2 || !pDst) {
    return NPP_NULL_POINTER_ERROR;
  }
  if (nLength == 0) {
    return NPP_SIZE_ERROR;
  }

  cudaError_t cudaStatus = nppsAdd_32f_kernel(pSrc1, pSrc2, pDst, nLength, stream);
  return (cudaStatus == cudaSuccess) ? NPP_NO_ERROR : NPP_CUDA_KERNEL_EXECUTION_ERROR;
}

template <>
NppStatus nppsAddImpl<Npp16s>(const Npp16s *pSrc1, const Npp16s *pSrc2, Npp16s *pDst, size_t nLength,
                              cudaStream_t stream) {
  if (!pSrc1 || !pSrc2 || !pDst) {
    return NPP_NULL_POINTER_ERROR;
  }
  if (nLength == 0) {
    return NPP_SIZE_ERROR;
  }

  cudaError_t cudaStatus = nppsAdd_16s_kernel(pSrc1, pSrc2, pDst, nLength, stream);
  return (cudaStatus == cudaSuccess) ? NPP_NO_ERROR : NPP_CUDA_KERNEL_EXECUTION_ERROR;
}

template <>
NppStatus nppsAddImpl<Npp32fc>(const Npp32fc *pSrc1, const Npp32fc *pSrc2, Npp32fc *pDst, size_t nLength,
                               cudaStream_t stream) {
  if (!pSrc1 || !pSrc2 || !pDst) {
    return NPP_NULL_POINTER_ERROR;
  }
  if (nLength == 0) {
    return NPP_SIZE_ERROR;
  }

  cudaError_t cudaStatus = nppsAdd_32fc_kernel(pSrc1, pSrc2, pDst, nLength, stream);
  return (cudaStatus == cudaSuccess) ? NPP_NO_ERROR : NPP_CUDA_KERNEL_EXECUTION_ERROR;
}

} // namespace

// ==============================================================================
// Add Operations - Element-wise addition of two signals
// ==============================================================================

NppStatus nppsAdd_32f_Ctx(const Npp32f *pSrc1, const Npp32f *pSrc2, Npp32f *pDst, NppSignalLength nLength,
                          NppStreamContext nppStreamCtx) {
  return nppsAddImpl<Npp32f>(pSrc1, pSrc2, pDst, static_cast<size_t>(nLength), nppStreamCtx.hStream);
}

NppStatus nppsAdd_32f(const Npp32f *pSrc1, const Npp32f *pSrc2, Npp32f *pDst, NppSignalLength nLength) {
  NppStreamContext defaultContext;
  NppStatus status = nppGetStreamContext(&defaultContext);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppsAdd_32f_Ctx(pSrc1, pSrc2, pDst, nLength, defaultContext);
}

NppStatus nppsAdd_16s_Ctx(const Npp16s *pSrc1, const Npp16s *pSrc2, Npp16s *pDst, NppSignalLength nLength,
                          NppStreamContext nppStreamCtx) {
  return nppsAddImpl<Npp16s>(pSrc1, pSrc2, pDst, static_cast<size_t>(nLength), nppStreamCtx.hStream);
}

NppStatus nppsAdd_16s(const Npp16s *pSrc1, const Npp16s *pSrc2, Npp16s *pDst, NppSignalLength nLength) {
  NppStreamContext defaultContext;
  NppStatus status = nppGetStreamContext(&defaultContext);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppsAdd_16s_Ctx(pSrc1, pSrc2, pDst, nLength, defaultContext);
}

NppStatus nppsAdd_32fc_Ctx(const Npp32fc *pSrc1, const Npp32fc *pSrc2, Npp32fc *pDst, NppSignalLength nLength,
                           NppStreamContext nppStreamCtx) {
  return nppsAddImpl<Npp32fc>(pSrc1, pSrc2, pDst, static_cast<size_t>(nLength), nppStreamCtx.hStream);
}

NppStatus nppsAdd_32fc(const Npp32fc *pSrc1, const Npp32fc *pSrc2, Npp32fc *pDst, NppSignalLength nLength) {
  NppStreamContext defaultContext;
  NppStatus status = nppGetStreamContext(&defaultContext);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppsAdd_32fc_Ctx(pSrc1, pSrc2, pDst, nLength, defaultContext);
}
