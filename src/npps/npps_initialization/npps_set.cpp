#include "npp.h"
#include "npp_version_compat.h"
#include <cstring>
#include <cuda_runtime.h>

// Kernel declarations (implemented in .cu file)

extern "C" {
cudaError_t nppsSet_8u_kernel(Npp8u nValue, Npp8u *pDst, size_t nLength, cudaStream_t stream);
cudaError_t nppsSet_32f_kernel(Npp32f nValue, Npp32f *pDst, size_t nLength, cudaStream_t stream);
cudaError_t nppsSet_32fc_kernel(Npp32fc nValue, Npp32fc *pDst, size_t nLength, cudaStream_t stream);
}

// ==============================================================================
// Internal implementation functions
// ==============================================================================

namespace {

template <typename T> NppStatus nppsSetImpl(T nValue, T *pDst, size_t nLength, cudaStream_t stream);

template <> NppStatus nppsSetImpl<Npp8u>(Npp8u nValue, Npp8u *pDst, size_t nLength, cudaStream_t stream) {
  if (!pDst) {
    return NPP_NULL_POINTER_ERROR;
  }
  if (nLength == 0) {
    return NPP_SIZE_ERROR;
  }

  cudaError_t cudaStatus = nppsSet_8u_kernel(nValue, pDst, nLength, stream);
  return (cudaStatus == cudaSuccess) ? NPP_NO_ERROR : NPP_CUDA_KERNEL_EXECUTION_ERROR;
}

template <> NppStatus nppsSetImpl<Npp32f>(Npp32f nValue, Npp32f *pDst, size_t nLength, cudaStream_t stream) {
  if (!pDst) {
    return NPP_NULL_POINTER_ERROR;
  }
  if (nLength == 0) {
    return NPP_SIZE_ERROR;
  }

  cudaError_t cudaStatus = nppsSet_32f_kernel(nValue, pDst, nLength, stream);
  return (cudaStatus == cudaSuccess) ? NPP_NO_ERROR : NPP_CUDA_KERNEL_EXECUTION_ERROR;
}

template <> NppStatus nppsSetImpl<Npp32fc>(Npp32fc nValue, Npp32fc *pDst, size_t nLength, cudaStream_t stream) {
  if (!pDst) {
    return NPP_NULL_POINTER_ERROR;
  }
  if (nLength == 0) {
    return NPP_SIZE_ERROR;
  }

  cudaError_t cudaStatus = nppsSet_32fc_kernel(nValue, pDst, nLength, stream);
  return (cudaStatus == cudaSuccess) ? NPP_NO_ERROR : NPP_CUDA_KERNEL_EXECUTION_ERROR;
}

} // namespace

// ==============================================================================
// Set Operations - Initialize signal elements with constant values
// ==============================================================================

NppStatus nppsSet_8u_Ctx(Npp8u nValue, Npp8u *pDst, NppSignalLength nLength, NppStreamContext nppStreamCtx) {
  return nppsSetImpl<Npp8u>(nValue, pDst, static_cast<size_t>(nLength), nppStreamCtx.hStream);
}

NppStatus nppsSet_8u(Npp8u nValue, Npp8u *pDst, NppSignalLength nLength) {
  NppStreamContext defaultContext;
  NppStatus status = nppGetStreamContext(&defaultContext);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppsSet_8u_Ctx(nValue, pDst, nLength, defaultContext);
}

NppStatus nppsSet_32f_Ctx(Npp32f nValue, Npp32f *pDst, NppSignalLength nLength, NppStreamContext nppStreamCtx) {
  return nppsSetImpl<Npp32f>(nValue, pDst, static_cast<size_t>(nLength), nppStreamCtx.hStream);
}

NppStatus nppsSet_32f(Npp32f nValue, Npp32f *pDst, NppSignalLength nLength) {
  NppStreamContext defaultContext;
  NppStatus status = nppGetStreamContext(&defaultContext);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppsSet_32f_Ctx(nValue, pDst, nLength, defaultContext);
}

NppStatus nppsSet_32fc_Ctx(Npp32fc nValue, Npp32fc *pDst, NppSignalLength nLength, NppStreamContext nppStreamCtx) {
  return nppsSetImpl<Npp32fc>(nValue, pDst, static_cast<size_t>(nLength), nppStreamCtx.hStream);
}

NppStatus nppsSet_32fc(Npp32fc nValue, Npp32fc *pDst, NppSignalLength nLength) {
  NppStreamContext defaultContext;
  NppStatus status = nppGetStreamContext(&defaultContext);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppsSet_32fc_Ctx(nValue, pDst, nLength, defaultContext);
}

NppStatus nppsZero_32f_Ctx(Npp32f *pDst, NppSignalLength nLength, NppStreamContext nppStreamCtx) {
  return nppsSet_32f_Ctx(0.0f, pDst, nLength, nppStreamCtx);
}

NppStatus nppsZero_32f(Npp32f *pDst, NppSignalLength nLength) { return nppsSet_32f(0.0f, pDst, nLength); }
