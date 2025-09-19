#include "npp.h"
#include <cstring>
#include <cuda_runtime.h>

// Forward declarations for CUDA kernels (implemented in .cu file)
extern "C" {
cudaError_t nppsAdd_32f_kernel(const Npp32f *pSrc1, const Npp32f *pSrc2, Npp32f *pDst, size_t nLength,
                               cudaStream_t stream);
cudaError_t nppsAdd_16s_kernel(const Npp16s *pSrc1, const Npp16s *pSrc2, Npp16s *pDst, size_t nLength,
                               cudaStream_t stream);
cudaError_t nppsAdd_32fc_kernel(const Npp32fc *pSrc1, const Npp32fc *pSrc2, Npp32fc *pDst, size_t nLength,
                                cudaStream_t stream);
}

/**
 * NPPS Arithmetic Operations Implementation - Add Functions
 * Implements nppsAdd functions for 1D signal addition operations.
 */

// ==============================================================================
// Add Operations - Element-wise addition of two signals
// ==============================================================================

/**
 * 32-bit float signal addition
 */
NppStatus nppsAdd_32f_Ctx(const Npp32f *pSrc1, const Npp32f *pSrc2, Npp32f *pDst, size_t nLength,
                          NppStreamContext nppStreamCtx) {
  // Parameter validation
  if (!pSrc1 || !pSrc2 || !pDst) {
    return NPP_NULL_POINTER_ERROR;
  }
  if (nLength == 0) {
    return NPP_SIZE_ERROR;
  }

  // Launch CUDA kernel for element-wise addition
  cudaError_t cudaStatus = nppsAdd_32f_kernel(pSrc1, pSrc2, pDst, nLength, nppStreamCtx.hStream);

  if (cudaStatus != cudaSuccess) {
    return NPP_CUDA_KERNEL_EXECUTION_ERROR;
  }

  return NPP_NO_ERROR;
}

NppStatus nppsAdd_32f(const Npp32f *pSrc1, const Npp32f *pSrc2, Npp32f *pDst, size_t nLength) {
  NppStreamContext defaultContext;
  NppStatus status = nppGetStreamContext(&defaultContext);
  if (status != NPP_NO_ERROR) {
    return status;
  }

  return nppsAdd_32f_Ctx(pSrc1, pSrc2, pDst, nLength, defaultContext);
}

/**
 * 16-bit signed integer signal addition
 */
NppStatus nppsAdd_16s_Ctx(const Npp16s *pSrc1, const Npp16s *pSrc2, Npp16s *pDst, size_t nLength,
                          NppStreamContext nppStreamCtx) {
  // Parameter validation
  if (!pSrc1 || !pSrc2 || !pDst) {
    return NPP_NULL_POINTER_ERROR;
  }
  if (nLength == 0) {
    return NPP_SIZE_ERROR;
  }

  // Launch CUDA kernel for element-wise addition
  cudaError_t cudaStatus = nppsAdd_16s_kernel(pSrc1, pSrc2, pDst, nLength, nppStreamCtx.hStream);

  if (cudaStatus != cudaSuccess) {
    return NPP_CUDA_KERNEL_EXECUTION_ERROR;
  }

  return NPP_NO_ERROR;
}

NppStatus nppsAdd_16s(const Npp16s *pSrc1, const Npp16s *pSrc2, Npp16s *pDst, size_t nLength) {
  NppStreamContext defaultContext;
  NppStatus status = nppGetStreamContext(&defaultContext);
  if (status != NPP_NO_ERROR) {
    return status;
  }

  return nppsAdd_16s_Ctx(pSrc1, pSrc2, pDst, nLength, defaultContext);
}

/**
 * 32-bit float complex signal addition
 */
NppStatus nppsAdd_32fc_Ctx(const Npp32fc *pSrc1, const Npp32fc *pSrc2, Npp32fc *pDst, size_t nLength,
                           NppStreamContext nppStreamCtx) {
  // Parameter validation
  if (!pSrc1 || !pSrc2 || !pDst) {
    return NPP_NULL_POINTER_ERROR;
  }
  if (nLength == 0) {
    return NPP_SIZE_ERROR;
  }

  // Launch CUDA kernel for element-wise complex addition
  cudaError_t cudaStatus = nppsAdd_32fc_kernel(pSrc1, pSrc2, pDst, nLength, nppStreamCtx.hStream);

  if (cudaStatus != cudaSuccess) {
    return NPP_CUDA_KERNEL_EXECUTION_ERROR;
  }

  return NPP_NO_ERROR;
}

NppStatus nppsAdd_32fc(const Npp32fc *pSrc1, const Npp32fc *pSrc2, Npp32fc *pDst, size_t nLength) {
  NppStreamContext defaultContext;
  NppStatus status = nppGetStreamContext(&defaultContext);
  if (status != NPP_NO_ERROR) {
    return status;
  }

  return nppsAdd_32fc_Ctx(pSrc1, pSrc2, pDst, nLength, defaultContext);
}
