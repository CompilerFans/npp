#include "npp.h"
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
// Buffer Size Functions - Required for reduction operations
// ==============================================================================

NppStatus nppsSumGetBufferSize_32f_Ctx(size_t nLength, size_t *hpBufferSize, NppStreamContext nppStreamCtx) {
  if (!hpBufferSize) {
    return NPP_NULL_POINTER_ERROR;
  }
  if (nLength == 0) {
    return NPP_SIZE_ERROR;
  }

  // Use stream context parameter to avoid unused warning
  if (nppStreamCtx.nCudaDeviceId < -1) {
    return NPP_BAD_ARGUMENT_ERROR;
  }

  // Calculate required buffer size for reduction
  // We need space for partial sums from each block
  const int blockSize = 256;
  const int gridSize = (nLength + blockSize - 1) / blockSize;
  *hpBufferSize = gridSize * sizeof(Npp32f);

  return NPP_NO_ERROR;
}

NppStatus nppsSumGetBufferSize_32f(size_t nLength, size_t *hpBufferSize) {
  return nppsSumGetBufferSize_32f_Ctx(nLength, hpBufferSize, {});
}

NppStatus nppsSumGetBufferSize_32fc_Ctx(size_t nLength, size_t *hpBufferSize, NppStreamContext nppStreamCtx) {
  if (!hpBufferSize) {
    return NPP_NULL_POINTER_ERROR;
  }

  // Use stream context parameter to avoid unused warning
  if (nppStreamCtx.nCudaDeviceId < -1) {
    return NPP_BAD_ARGUMENT_ERROR;
  }
  if (nLength == 0) {
    return NPP_SIZE_ERROR;
  }

  // Buffer size for complex reduction (2x floats per complex number)
  const int blockSize = 256;
  const int gridSize = (nLength + blockSize - 1) / blockSize;
  *hpBufferSize = gridSize * sizeof(Npp32fc);

  return NPP_NO_ERROR;
}

NppStatus nppsSumGetBufferSize_32fc(size_t nLength, size_t *hpBufferSize) {
  return nppsSumGetBufferSize_32fc_Ctx(nLength, hpBufferSize, {});
}

// ==============================================================================
// Sum Operations - Element-wise summation of signal values
// ==============================================================================
NppStatus nppsSum_32f_Ctx(const Npp32f *pSrc, size_t nLength, Npp32f *pSum, Npp8u *pDeviceBuffer,
                          NppStreamContext nppStreamCtx) {
  // Parameter validation
  if (!pSrc || !pSum || !pDeviceBuffer) {
    return NPP_NULL_POINTER_ERROR;
  }
  if (nLength == 0) {
    return NPP_SIZE_ERROR;
  }

  // Launch kernel for sum reduction
  cudaError_t cudaStatus = nppsSum_32f_kernel(pSrc, nLength, pSum, pDeviceBuffer, nppStreamCtx.hStream);

  if (cudaStatus != cudaSuccess) {
    return NPP_CUDA_KERNEL_EXECUTION_ERROR;
  }

  return NPP_NO_ERROR;
}

NppStatus nppsSum_32f(const Npp32f *pSrc, size_t nLength, Npp32f *pSum, Npp8u *pDeviceBuffer) {
  NppStreamContext defaultContext;
  NppStatus status = nppGetStreamContext(&defaultContext);
  if (status != NPP_NO_ERROR) {
    return status;
  }

  return nppsSum_32f_Ctx(pSrc, nLength, pSum, pDeviceBuffer, defaultContext);
}

NppStatus nppsSum_32fc_Ctx(const Npp32fc *pSrc, size_t nLength, Npp32fc *pSum, Npp8u *pDeviceBuffer,
                           NppStreamContext nppStreamCtx) {
  // Parameter validation
  if (!pSrc || !pSum || !pDeviceBuffer) {
    return NPP_NULL_POINTER_ERROR;
  }
  if (nLength == 0) {
    return NPP_SIZE_ERROR;
  }

  // Launch kernel for complex sum reduction
  cudaError_t cudaStatus = nppsSum_32fc_kernel(pSrc, nLength, pSum, pDeviceBuffer, nppStreamCtx.hStream);

  if (cudaStatus != cudaSuccess) {
    return NPP_CUDA_KERNEL_EXECUTION_ERROR;
  }

  return NPP_NO_ERROR;
}

NppStatus nppsSum_32fc(const Npp32fc *pSrc, size_t nLength, Npp32fc *pSum, Npp8u *pDeviceBuffer) {
  NppStreamContext defaultContext;
  NppStatus status = nppGetStreamContext(&defaultContext);
  if (status != NPP_NO_ERROR) {
    return status;
  }

  return nppsSum_32fc_Ctx(pSrc, nLength, pSum, pDeviceBuffer, defaultContext);
}
