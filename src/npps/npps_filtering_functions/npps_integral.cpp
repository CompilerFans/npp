#include "npp.h"
#include "npp_version_compat.h"
#include <cuda_runtime.h>

// Kernel declarations (implemented in .cu file)
extern "C" {
cudaError_t nppsIntegral_32s_kernel(const Npp32s *pSrc, Npp32s *pDst, size_t nLength, Npp8u *pDeviceBuffer,
                                    cudaStream_t stream);
}

// ==============================================================================
// Internal implementation functions
// ==============================================================================

namespace {

NppStatus nppsIntegralGetBufferSizeImpl(size_t nLength, NppSignalLength *hpBufferSize) {
  if (!hpBufferSize) {
    return NPP_NULL_POINTER_ERROR;
  }
  if (nLength == 0) {
    return NPP_SIZE_ERROR;
  }

  // Buffer for parallel prefix sum (scan) algorithm
  // Need space for block-level partial sums
  const size_t blockSize = 256;
  const size_t numBlocks = (nLength + blockSize - 1) / blockSize;
  // Buffer for block sums + additional space for multi-level scan
  *hpBufferSize = static_cast<NppSignalLength>(numBlocks * sizeof(Npp32s) * 2);

  return NPP_NO_ERROR;
}

NppStatus nppsIntegralImpl(const Npp32s *pSrc, Npp32s *pDst, size_t nLength, Npp8u *pDeviceBuffer,
                           cudaStream_t stream) {
  if (!pSrc || !pDst || !pDeviceBuffer) {
    return NPP_NULL_POINTER_ERROR;
  }
  if (nLength == 0) {
    return NPP_SIZE_ERROR;
  }

  cudaError_t cudaStatus = nppsIntegral_32s_kernel(pSrc, pDst, nLength, pDeviceBuffer, stream);

  return (cudaStatus == cudaSuccess) ? NPP_NO_ERROR : NPP_CUDA_KERNEL_EXECUTION_ERROR;
}

} // namespace

// ==============================================================================
// Buffer Size Functions
// ==============================================================================

NppStatus nppsIntegralGetBufferSize_32s(NppSignalLength nLength, NppSignalLength *hpBufferSize) {
  return nppsIntegralGetBufferSizeImpl(static_cast<size_t>(nLength), hpBufferSize);
}

// ==============================================================================
// Integral (Prefix Sum) Operations
// ==============================================================================

NppStatus nppsIntegral_32s_Ctx(const Npp32s *pSrc, Npp32s *pDst, NppSignalLength nLength, Npp8u *pDeviceBuffer,
                               NppStreamContext nppStreamCtx) {
  return nppsIntegralImpl(pSrc, pDst, static_cast<size_t>(nLength), pDeviceBuffer, nppStreamCtx.hStream);
}

NppStatus nppsIntegral_32s(const Npp32s *pSrc, Npp32s *pDst, NppSignalLength nLength, Npp8u *pDeviceBuffer) {
  NppStreamContext defaultContext;
  NppStatus status = nppGetStreamContext(&defaultContext);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppsIntegral_32s_Ctx(pSrc, pDst, nLength, pDeviceBuffer, defaultContext);
}
