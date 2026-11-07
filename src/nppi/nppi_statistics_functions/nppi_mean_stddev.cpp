#include "npp.h"
#include <cstdio>
#include <cstring>
#include <cuda_runtime.h>

// Kernel declarations

extern "C" {
cudaError_t nppiMean_StdDev_8u_C1R_kernel(const Npp8u *pSrc, int nSrcStep, NppiSize oSizeROI, Npp8u *pDeviceBuffer,
                                          Npp64f *pMean, Npp64f *pStdDev, cudaStream_t stream);
cudaError_t nppiMean_StdDev_32f_C1R_kernel(const Npp32f *pSrc, int nSrcStep, NppiSize oSizeROI, Npp8u *pDeviceBuffer,
                                           Npp64f *pMean, Npp64f *pStdDev, cudaStream_t stream);

// Masked versions
cudaError_t nppiMean_StdDev_8u_C1MR_kernel(const Npp8u *pSrc, int nSrcStep, const Npp8u *pMask, int nMaskStep,
                                           NppiSize oSizeROI, Npp8u *pDeviceBuffer, Npp64f *pMean, Npp64f *pStdDev,
                                           cudaStream_t stream);
cudaError_t nppiMean_StdDev_32f_C1MR_kernel(const Npp32f *pSrc, int nSrcStep, const Npp8u *pMask, int nMaskStep,
                                            NppiSize oSizeROI, Npp8u *pDeviceBuffer, Npp64f *pMean, Npp64f *pStdDev,
                                            cudaStream_t stream);

cudaError_t nppiMean_8u_C1R_kernel(const Npp8u *pSrc, int nSrcStep, NppiSize oSizeROI, Npp8u *pDeviceBuffer,
                                   Npp64f *pMean, cudaStream_t stream);
}

// ============ cuda 12.8 ===================
// Buffer size calculation for 8u single channel
NPP_BUFFER_ABI_EXTERN_C
NppStatus nppiMeanStdDevGetBufferHostSize_8u_C1R_Ctx(NppiSize oSizeROI, size_t *hpBufferSize,
                                                     NppStreamContext /*nppStreamCtx*/) {
  // Parameter validation
  if (!hpBufferSize) {
    return NPP_NULL_POINTER_ERROR;
  }
  if (oSizeROI.width <= 0 || oSizeROI.height <= 0) {
    return NPP_SIZE_ERROR;
  }

  // Calculate required buffer size for reduction operations
  // We need space for sum, sum of squares, and temporary reduction buffers
  size_t numPixels = static_cast<size_t>(oSizeROI.width) * static_cast<size_t>(oSizeROI.height);
  size_t numBlocks = (numPixels + 255) / 256; // Assuming 256 threads per block

  // Buffer for reduction: 2 values per block (sum and sum of squares)
  *hpBufferSize = numBlocks * 2 * sizeof(double);

  return NPP_SUCCESS;
}

NPP_BUFFER_ABI_EXTERN_C
NppStatus nppiMeanStdDevGetBufferHostSize_8u_C1R(NppiSize oSizeROI, size_t *hpBufferSize) {
  NppStreamContext ctx{};
  ctx.hStream = nppGetStream(); // NPP managed stream (nppSetStream)
  return nppiMeanStdDevGetBufferHostSize_8u_C1R_Ctx(oSizeROI, hpBufferSize, ctx);
}

// Buffer size calculation for 32-bit float single channel
NPP_BUFFER_ABI_EXTERN_C
NppStatus nppiMeanStdDevGetBufferHostSize_32f_C1R_Ctx(NppiSize oSizeROI, size_t *hpBufferSize,
                                                      NppStreamContext /*nppStreamCtx*/) {
  // Parameter validation
  if (!hpBufferSize) {
    return NPP_NULL_POINTER_ERROR;
  }
  if (oSizeROI.width <= 0 || oSizeROI.height <= 0) {
    return NPP_SIZE_ERROR;
  }

  // Calculate required buffer size for reduction operations
  size_t numPixels = static_cast<size_t>(oSizeROI.width) * static_cast<size_t>(oSizeROI.height);
  size_t numBlocks = (numPixels + 255) / 256; // Assuming 256 threads per block

  // Buffer for reduction: 2 values per block (sum and sum of squares)
  *hpBufferSize = numBlocks * 2 * sizeof(double);

  return NPP_SUCCESS;
}

NPP_BUFFER_ABI_EXTERN_C
NppStatus nppiMeanStdDevGetBufferHostSize_32f_C1R(NppiSize oSizeROI, size_t *hpBufferSize) {
  NppStreamContext ctx{};
  ctx.hStream = nppGetStream(); // NPP managed stream (nppSetStream)
  return nppiMeanStdDevGetBufferHostSize_32f_C1R_Ctx(oSizeROI, hpBufferSize, ctx);
}

// ================== below 12.8 ============================
// Buffer size calculation for 8u single channel
#if !NPP_BUFFER_SIZE_IS_SIZE_T
NPP_BUFFER_ABI_EXTERN_C
NppStatus nppiMeanStdDevGetBufferHostSize_8u_C1R_Ctx(NppiSize oSizeROI, NPP_BUFFER_SIZE_TYPE *hpBufferSize,
                                                     NppStreamContext nppStreamCtx) {
  size_t size = 0;
  auto ret = nppiMeanStdDevGetBufferHostSize_8u_C1R_Ctx(oSizeROI, &size, nppStreamCtx);
  *hpBufferSize = static_cast<int>(size);
  return ret;
}
#endif

#if !NPP_BUFFER_SIZE_IS_SIZE_T
NPP_BUFFER_ABI_EXTERN_C
NppStatus nppiMeanStdDevGetBufferHostSize_8u_C1R(NppiSize oSizeROI, NPP_BUFFER_SIZE_TYPE *hpBufferSize) {
  size_t size = 0;
  auto ret = nppiMeanStdDevGetBufferHostSize_8u_C1R(oSizeROI, &size);
  *hpBufferSize = static_cast<int>(size);
  return ret;
}
#endif

// Buffer size calculation for 32-bit float single channel
#if !NPP_BUFFER_SIZE_IS_SIZE_T
NPP_BUFFER_ABI_EXTERN_C
NppStatus nppiMeanStdDevGetBufferHostSize_32f_C1R_Ctx(NppiSize oSizeROI, NPP_BUFFER_SIZE_TYPE *hpBufferSize,
                                                      NppStreamContext nppStreamCtx) {
  size_t size = 0;
  auto ret = nppiMeanStdDevGetBufferHostSize_32f_C1R_Ctx(oSizeROI, &size, nppStreamCtx);
  *hpBufferSize = static_cast<int>(size);
  return ret;
}
#endif

#if !NPP_BUFFER_SIZE_IS_SIZE_T
NPP_BUFFER_ABI_EXTERN_C
NppStatus nppiMeanStdDevGetBufferHostSize_32f_C1R(NppiSize oSizeROI, NPP_BUFFER_SIZE_TYPE *hpBufferSize) {
  size_t size = 0;
  auto ret = nppiMeanStdDevGetBufferHostSize_32f_C1R(oSizeROI, &size);
  *hpBufferSize = static_cast<int>(size);
  return ret;
}
#endif
// ==========================

// Mean and standard deviation calculation for 8-bit unsigned single channel
NppStatus nppiMean_StdDev_8u_C1R_Ctx(const Npp8u *pSrc, int nSrcStep, NppiSize oSizeROI, Npp8u *pDeviceBuffer,
                                     Npp64f *pMean, Npp64f *pStdDev, NppStreamContext nppStreamCtx) {
  // Parameter validation
  if (!pSrc || !pDeviceBuffer || !pMean || !pStdDev) {
    return NPP_NULL_POINTER_ERROR;
  }
  if (oSizeROI.width <= 0 || oSizeROI.height <= 0) {
    return NPP_SIZE_ERROR;
  }
  if (nSrcStep < oSizeROI.width) {
    return NPP_STEP_ERROR;
  }

  // Call GPU kernel
  cudaError_t cudaStatus =
      nppiMean_StdDev_8u_C1R_kernel(pSrc, nSrcStep, oSizeROI, pDeviceBuffer, pMean, pStdDev, nppStreamCtx.hStream);

  return (cudaStatus == cudaSuccess) ? NPP_SUCCESS : NPP_CUDA_KERNEL_EXECUTION_ERROR;
}

NppStatus nppiMean_StdDev_8u_C1R(const Npp8u *pSrc, int nSrcStep, NppiSize oSizeROI, Npp8u *pDeviceBuffer,
                                 Npp64f *pMean, Npp64f *pStdDev) {
  NppStreamContext ctx{};
  ctx.hStream = nppGetStream(); // NPP managed stream (nppSetStream)
  return nppiMean_StdDev_8u_C1R_Ctx(pSrc, nSrcStep, oSizeROI, pDeviceBuffer, pMean, pStdDev, ctx);
}

// Mean and standard deviation calculation for 32-bit float single channel
NppStatus nppiMean_StdDev_32f_C1R_Ctx(const Npp32f *pSrc, int nSrcStep, NppiSize oSizeROI, Npp8u *pDeviceBuffer,
                                      Npp64f *pMean, Npp64f *pStdDev, NppStreamContext nppStreamCtx) {
  // Parameter validation
  if (!pSrc || !pDeviceBuffer || !pMean || !pStdDev) {
    return NPP_NULL_POINTER_ERROR;
  }
  if (oSizeROI.width <= 0 || oSizeROI.height <= 0) {
    return NPP_SIZE_ERROR;
  }
  if (nSrcStep < static_cast<int>(oSizeROI.width * sizeof(Npp32f))) {
    return NPP_STEP_ERROR;
  }

  // Call GPU kernel
  cudaError_t cudaStatus =
      nppiMean_StdDev_32f_C1R_kernel(pSrc, nSrcStep, oSizeROI, pDeviceBuffer, pMean, pStdDev, nppStreamCtx.hStream);

  return (cudaStatus == cudaSuccess) ? NPP_SUCCESS : NPP_CUDA_KERNEL_EXECUTION_ERROR;
}

NppStatus nppiMean_StdDev_32f_C1R(const Npp32f *pSrc, int nSrcStep, NppiSize oSizeROI, Npp8u *pDeviceBuffer,
                                  Npp64f *pMean, Npp64f *pStdDev) {
  NppStreamContext ctx{};
  ctx.hStream = nppGetStream(); // NPP managed stream (nppSetStream)
  return nppiMean_StdDev_32f_C1R_Ctx(pSrc, nSrcStep, oSizeROI, pDeviceBuffer, pMean, pStdDev, ctx);
}

// ============ MASKED VERSIONS C1MR ===================

// ============ cuda 12.8 ===================
// Buffer size calculation for 8u single channel masked
NPP_BUFFER_ABI_EXTERN_C
NppStatus nppiMeanStdDevGetBufferHostSize_8u_C1MR_Ctx(NppiSize oSizeROI, size_t *hpBufferSize,
                                                      NppStreamContext /*nppStreamCtx*/) {
  // Parameter validation
  if (!hpBufferSize) {
    return NPP_NULL_POINTER_ERROR;
  }
  if (oSizeROI.width <= 0 || oSizeROI.height <= 0) {
    return NPP_SIZE_ERROR;
  }

  // Masked versions need extra space for valid counts
  size_t numPixels = static_cast<size_t>(oSizeROI.width) * static_cast<size_t>(oSizeROI.height);
  size_t numBlocks = (numPixels + 255) / 256;

  // Buffer for reduction: 3 values per block (sum, sum of squares, valid counts)
  *hpBufferSize = numBlocks * 3 * sizeof(double);

  return NPP_SUCCESS;
}

NPP_BUFFER_ABI_EXTERN_C
NppStatus nppiMeanStdDevGetBufferHostSize_8u_C1MR(NppiSize oSizeROI, size_t *hpBufferSize) {
  NppStreamContext ctx{};
  ctx.hStream = nppGetStream();
  return nppiMeanStdDevGetBufferHostSize_8u_C1MR_Ctx(oSizeROI, hpBufferSize, ctx);
}

// Buffer size calculation for 32f single channel masked
NPP_BUFFER_ABI_EXTERN_C
NppStatus nppiMeanStdDevGetBufferHostSize_32f_C1MR_Ctx(NppiSize oSizeROI, size_t *hpBufferSize,
                                                       NppStreamContext /*nppStreamCtx*/) {
  // Parameter validation
  if (!hpBufferSize) {
    return NPP_NULL_POINTER_ERROR;
  }
  if (oSizeROI.width <= 0 || oSizeROI.height <= 0) {
    return NPP_SIZE_ERROR;
  }

  // Masked versions need extra space for valid counts
  size_t numPixels = static_cast<size_t>(oSizeROI.width) * static_cast<size_t>(oSizeROI.height);
  size_t numBlocks = (numPixels + 255) / 256;

  // Buffer for reduction: 3 values per block (sum, sum of squares, valid counts)
  *hpBufferSize = numBlocks * 3 * sizeof(double);

  return NPP_SUCCESS;
}

NPP_BUFFER_ABI_EXTERN_C
NppStatus nppiMeanStdDevGetBufferHostSize_32f_C1MR(NppiSize oSizeROI, size_t *hpBufferSize) {
  NppStreamContext ctx{};
  ctx.hStream = nppGetStream();
  return nppiMeanStdDevGetBufferHostSize_32f_C1MR_Ctx(oSizeROI, hpBufferSize, ctx);
}

// ================== below 12.8 ============================
// Buffer size calculation for 8u single channel masked (int version)
#if !NPP_BUFFER_SIZE_IS_SIZE_T
NPP_BUFFER_ABI_EXTERN_C
NppStatus nppiMeanStdDevGetBufferHostSize_8u_C1MR_Ctx(NppiSize oSizeROI, NPP_BUFFER_SIZE_TYPE *hpBufferSize,
                                                      NppStreamContext nppStreamCtx) {
  size_t size = 0;
  auto ret = nppiMeanStdDevGetBufferHostSize_8u_C1MR_Ctx(oSizeROI, &size, nppStreamCtx);
  *hpBufferSize = static_cast<int>(size);
  return ret;
}
#endif

#if !NPP_BUFFER_SIZE_IS_SIZE_T
NPP_BUFFER_ABI_EXTERN_C
NppStatus nppiMeanStdDevGetBufferHostSize_8u_C1MR(NppiSize oSizeROI, NPP_BUFFER_SIZE_TYPE *hpBufferSize) {
  size_t size = 0;
  auto ret = nppiMeanStdDevGetBufferHostSize_8u_C1MR(oSizeROI, &size);
  *hpBufferSize = static_cast<int>(size);
  return ret;
}
#endif

// Buffer size calculation for 32f single channel masked (int version)
#if !NPP_BUFFER_SIZE_IS_SIZE_T
NPP_BUFFER_ABI_EXTERN_C
NppStatus nppiMeanStdDevGetBufferHostSize_32f_C1MR_Ctx(NppiSize oSizeROI, NPP_BUFFER_SIZE_TYPE *hpBufferSize,
                                                       NppStreamContext nppStreamCtx) {
  size_t size = 0;
  auto ret = nppiMeanStdDevGetBufferHostSize_32f_C1MR_Ctx(oSizeROI, &size, nppStreamCtx);
  *hpBufferSize = static_cast<int>(size);
  return ret;
}
#endif

#if !NPP_BUFFER_SIZE_IS_SIZE_T
NPP_BUFFER_ABI_EXTERN_C
NppStatus nppiMeanStdDevGetBufferHostSize_32f_C1MR(NppiSize oSizeROI, NPP_BUFFER_SIZE_TYPE *hpBufferSize) {
  size_t size = 0;
  auto ret = nppiMeanStdDevGetBufferHostSize_32f_C1MR(oSizeROI, &size);
  *hpBufferSize = static_cast<int>(size);
  return ret;
}
#endif

// Mean and standard deviation calculation for 8u single channel with mask
NppStatus nppiMean_StdDev_8u_C1MR_Ctx(const Npp8u *pSrc, int nSrcStep, const Npp8u *pMask, int nMaskStep,
                                      NppiSize oSizeROI, Npp8u *pDeviceBuffer, Npp64f *pMean, Npp64f *pStdDev,
                                      NppStreamContext nppStreamCtx) {
  // Parameter validation
  if (!pSrc || !pMask || !pDeviceBuffer || !pMean || !pStdDev) {
    return NPP_NULL_POINTER_ERROR;
  }
  if (oSizeROI.width <= 0 || oSizeROI.height <= 0) {
    return NPP_SIZE_ERROR;
  }
  if (nSrcStep < oSizeROI.width || nMaskStep < oSizeROI.width) {
    return NPP_STEP_ERROR;
  }

  // Call GPU kernel
  cudaError_t cudaStatus = nppiMean_StdDev_8u_C1MR_kernel(pSrc, nSrcStep, pMask, nMaskStep, oSizeROI, pDeviceBuffer,
                                                          pMean, pStdDev, nppStreamCtx.hStream);

  return (cudaStatus == cudaSuccess) ? NPP_SUCCESS : NPP_CUDA_KERNEL_EXECUTION_ERROR;
}

NppStatus nppiMean_StdDev_8u_C1MR(const Npp8u *pSrc, int nSrcStep, const Npp8u *pMask, int nMaskStep, NppiSize oSizeROI,
                                  Npp8u *pDeviceBuffer, Npp64f *pMean, Npp64f *pStdDev) {
  NppStreamContext ctx{};
  ctx.hStream = nppGetStream();
  return nppiMean_StdDev_8u_C1MR_Ctx(pSrc, nSrcStep, pMask, nMaskStep, oSizeROI, pDeviceBuffer, pMean, pStdDev, ctx);
}

// Mean and standard deviation calculation for 32f single channel with mask
NppStatus nppiMean_StdDev_32f_C1MR_Ctx(const Npp32f *pSrc, int nSrcStep, const Npp8u *pMask, int nMaskStep,
                                       NppiSize oSizeROI, Npp8u *pDeviceBuffer, Npp64f *pMean, Npp64f *pStdDev,
                                       NppStreamContext nppStreamCtx) {
  // Parameter validation
  if (!pSrc || !pMask || !pDeviceBuffer || !pMean || !pStdDev) {
    return NPP_NULL_POINTER_ERROR;
  }
  if (oSizeROI.width <= 0 || oSizeROI.height <= 0) {
    return NPP_SIZE_ERROR;
  }
  if (nSrcStep < static_cast<int>(oSizeROI.width * sizeof(Npp32f)) || nMaskStep < oSizeROI.width) {
    return NPP_STEP_ERROR;
  }

  // Call GPU kernel
  cudaError_t cudaStatus = nppiMean_StdDev_32f_C1MR_kernel(pSrc, nSrcStep, pMask, nMaskStep, oSizeROI, pDeviceBuffer,
                                                           pMean, pStdDev, nppStreamCtx.hStream);

  return (cudaStatus == cudaSuccess) ? NPP_SUCCESS : NPP_CUDA_KERNEL_EXECUTION_ERROR;
}

NppStatus nppiMean_StdDev_32f_C1MR(const Npp32f *pSrc, int nSrcStep, const Npp8u *pMask, int nMaskStep,
                                   NppiSize oSizeROI, Npp8u *pDeviceBuffer, Npp64f *pMean, Npp64f *pStdDev) {
  NppStreamContext ctx{};
  ctx.hStream = nppGetStream();
  return nppiMean_StdDev_32f_C1MR_Ctx(pSrc, nSrcStep, pMask, nMaskStep, oSizeROI, pDeviceBuffer, pMean, pStdDev, ctx);
}

// Buffer size calculation for Mean 8u single channel
NPP_BUFFER_ABI_EXTERN_C
NppStatus nppiMeanGetBufferHostSize_8u_C1R_Ctx(NppiSize oSizeROI, size_t *hpBufferSize,
                                                NppStreamContext /*nppStreamCtx*/) {
  // Parameter validation
  if (!hpBufferSize) {
    return NPP_NULL_POINTER_ERROR;
  }
  if (oSizeROI.width <= 0 || oSizeROI.height <= 0) {
    return NPP_SIZE_ERROR;
  }

  // Calculate required buffer size for reduction operations
  size_t numPixels = static_cast<size_t>(oSizeROI.width) * static_cast<size_t>(oSizeROI.height);
  size_t numBlocks = (numPixels + 255) / 256; // Assuming 256 threads per block

  // Buffer for reduction: 1 value per block (sum)
  *hpBufferSize = numBlocks * sizeof(double);

  return NPP_SUCCESS;
}

NPP_BUFFER_ABI_EXTERN_C
NppStatus nppiMeanGetBufferHostSize_8u_C1R(NppiSize oSizeROI, size_t *hpBufferSize) {
  NppStreamContext ctx{};
  ctx.hStream = nppGetStream();
  return nppiMeanGetBufferHostSize_8u_C1R_Ctx(oSizeROI, hpBufferSize, ctx);
}

// Buffer size calculation for Mean 8u single channel (int version)
#if !NPP_BUFFER_SIZE_IS_SIZE_T
NPP_BUFFER_ABI_EXTERN_C
NppStatus nppiMeanGetBufferHostSize_8u_C1R_Ctx(NppiSize oSizeROI, NPP_BUFFER_SIZE_TYPE *hpBufferSize,
                                                NppStreamContext nppStreamCtx) {
  size_t size = 0;
  auto ret = nppiMeanGetBufferHostSize_8u_C1R_Ctx(oSizeROI, &size, nppStreamCtx);
  *hpBufferSize = static_cast<int>(size);
  return ret;
}
#endif

#if !NPP_BUFFER_SIZE_IS_SIZE_T
NPP_BUFFER_ABI_EXTERN_C
NppStatus nppiMeanGetBufferHostSize_8u_C1R(NppiSize oSizeROI, NPP_BUFFER_SIZE_TYPE *hpBufferSize) {
  size_t size = 0;
  auto ret = nppiMeanGetBufferHostSize_8u_C1R(oSizeROI, &size);
  *hpBufferSize = static_cast<int>(size);
  return ret;
}
#endif

// Mean calculation for 8-bit unsigned single channel
NppStatus nppiMean_8u_C1R_Ctx(const Npp8u *pSrc, int nSrcStep, NppiSize oSizeROI, Npp8u *pDeviceBuffer,
                               Npp64f *pMean, NppStreamContext nppStreamCtx) {
  // Parameter validation
  if (!pSrc || !pDeviceBuffer || !pMean) {
    return NPP_NULL_POINTER_ERROR;
  }
  if (oSizeROI.width <= 0 || oSizeROI.height <= 0) {
    return NPP_SIZE_ERROR;
  }
  if (nSrcStep < oSizeROI.width) {
    return NPP_STEP_ERROR;
  }

  cudaError_t cudaStatus =
      nppiMean_8u_C1R_kernel(pSrc, nSrcStep, oSizeROI, pDeviceBuffer, pMean, nppStreamCtx.hStream);

  return (cudaStatus == cudaSuccess) ? NPP_SUCCESS : NPP_CUDA_KERNEL_EXECUTION_ERROR;
}

NppStatus nppiMean_8u_C1R(const Npp8u *pSrc, int nSrcStep, NppiSize oSizeROI, Npp8u *pDeviceBuffer,
                           Npp64f *pMean) {
  NppStreamContext ctx{};
  ctx.hStream = nppGetStream();
  return nppiMean_8u_C1R_Ctx(pSrc, nSrcStep, oSizeROI, pDeviceBuffer, pMean, ctx);
}
