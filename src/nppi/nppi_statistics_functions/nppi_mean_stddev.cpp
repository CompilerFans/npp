#include "npp.h"
#include <cstring>
#include <cstdio>
#include <cuda_runtime.h>

// Kernel declarations
extern "C" {
cudaError_t nppiMean_StdDev_8u_C1R_kernel(const Npp8u *pSrc, int nSrcStep, NppiSize oSizeROI, 
                                          Npp8u *pDeviceBuffer, Npp64f *pMean, Npp64f *pStdDev, 
                                          cudaStream_t stream);
cudaError_t nppiMean_StdDev_32f_C1R_kernel(const Npp32f *pSrc, int nSrcStep, NppiSize oSizeROI, 
                                           Npp8u *pDeviceBuffer, Npp64f *pMean, Npp64f *pStdDev, 
                                           cudaStream_t stream);
}

// Buffer size calculation for 8u single channel
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
  size_t numBlocks = (numPixels + 255) / 256;  // Assuming 256 threads per block
  
  // Buffer for reduction: 2 values per block (sum and sum of squares)
  *hpBufferSize = numBlocks * 2 * sizeof(double);
  
  return NPP_SUCCESS;
}

NppStatus nppiMeanStdDevGetBufferHostSize_8u_C1R(NppiSize oSizeROI, size_t *hpBufferSize) {
  NppStreamContext ctx;
  ctx.hStream = 0; // Default stream
  return nppiMeanStdDevGetBufferHostSize_8u_C1R_Ctx(oSizeROI, hpBufferSize, ctx);
}

// Buffer size calculation for 32-bit float single channel
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
  size_t numBlocks = (numPixels + 255) / 256;  // Assuming 256 threads per block
  
  // Buffer for reduction: 2 values per block (sum and sum of squares)
  *hpBufferSize = numBlocks * 2 * sizeof(double);
  
  return NPP_SUCCESS;
}

NppStatus nppiMeanStdDevGetBufferHostSize_32f_C1R(NppiSize oSizeROI, size_t *hpBufferSize) {
  NppStreamContext ctx;
  ctx.hStream = 0; // Default stream
  return nppiMeanStdDevGetBufferHostSize_32f_C1R_Ctx(oSizeROI, hpBufferSize, ctx);
}

// Mean and standard deviation calculation for 8-bit unsigned single channel
NppStatus nppiMean_StdDev_8u_C1R_Ctx(const Npp8u *pSrc, int nSrcStep, NppiSize oSizeROI, 
                                     Npp8u *pDeviceBuffer, Npp64f *pMean, Npp64f *pStdDev, 
                                     NppStreamContext nppStreamCtx) {
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

  // Call CUDA kernel
  cudaError_t cudaStatus = nppiMean_StdDev_8u_C1R_kernel(pSrc, nSrcStep, oSizeROI, 
                                                         pDeviceBuffer, pMean, pStdDev, 
                                                         nppStreamCtx.hStream);

  return (cudaStatus == cudaSuccess) ? NPP_SUCCESS : NPP_CUDA_KERNEL_EXECUTION_ERROR;
}

NppStatus nppiMean_StdDev_8u_C1R(const Npp8u *pSrc, int nSrcStep, NppiSize oSizeROI, 
                                Npp8u *pDeviceBuffer, Npp64f *pMean, Npp64f *pStdDev) {
  NppStreamContext ctx;
  ctx.hStream = 0; // Default stream
  return nppiMean_StdDev_8u_C1R_Ctx(pSrc, nSrcStep, oSizeROI, pDeviceBuffer, pMean, pStdDev, ctx);
}

// Mean and standard deviation calculation for 32-bit float single channel
NppStatus nppiMean_StdDev_32f_C1R_Ctx(const Npp32f *pSrc, int nSrcStep, NppiSize oSizeROI, 
                                      Npp8u *pDeviceBuffer, Npp64f *pMean, Npp64f *pStdDev, 
                                      NppStreamContext nppStreamCtx) {
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

  // Call CUDA kernel
  cudaError_t cudaStatus = nppiMean_StdDev_32f_C1R_kernel(pSrc, nSrcStep, oSizeROI, 
                                                          pDeviceBuffer, pMean, pStdDev, 
                                                          nppStreamCtx.hStream);

  return (cudaStatus == cudaSuccess) ? NPP_SUCCESS : NPP_CUDA_KERNEL_EXECUTION_ERROR;
}

NppStatus nppiMean_StdDev_32f_C1R(const Npp32f *pSrc, int nSrcStep, NppiSize oSizeROI, 
                                 Npp8u *pDeviceBuffer, Npp64f *pMean, Npp64f *pStdDev) {
  NppStreamContext ctx;
  ctx.hStream = 0; // Default stream
  return nppiMean_StdDev_32f_C1R_Ctx(pSrc, nSrcStep, oSizeROI, pDeviceBuffer, pMean, pStdDev, ctx);
}