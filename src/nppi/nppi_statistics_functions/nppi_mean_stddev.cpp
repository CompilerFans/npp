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
cudaError_t nppiMean_StdDev_8u_C3CR_kernel(const Npp8u *pSrc, int nSrcStep, NppiSize oSizeROI, int nCOI,
                                           Npp8u *pDeviceBuffer, Npp64f *pMean, Npp64f *pStdDev, cudaStream_t stream);

// Masked versions
cudaError_t nppiMean_StdDev_8u_C1MR_kernel(const Npp8u *pSrc, int nSrcStep, const Npp8u *pMask, int nMaskStep,
                                           NppiSize oSizeROI, Npp8u *pDeviceBuffer, Npp64f *pMean, Npp64f *pStdDev,
                                           cudaStream_t stream);
cudaError_t nppiMean_StdDev_32f_C1MR_kernel(const Npp32f *pSrc, int nSrcStep, const Npp8u *pMask, int nMaskStep,
                                            NppiSize oSizeROI, Npp8u *pDeviceBuffer, Npp64f *pMean, Npp64f *pStdDev,
                                            cudaStream_t stream);
cudaError_t nppiMean_StdDev_8u_C3CMR_kernel(const Npp8u *pSrc, int nSrcStep, const Npp8u *pMask, int nMaskStep,
                                            NppiSize oSizeROI, int nCOI, Npp8u *pDeviceBuffer, Npp64f *pMean,
                                            Npp64f *pStdDev, cudaStream_t stream);
cudaError_t nppiMean_8u_C1R_kernel(const Npp8u *pSrc, int nSrcStep, NppiSize oSizeROI, Npp8u *pDeviceBuffer,
                                   Npp64f *pMean, cudaStream_t stream);
cudaError_t nppiMean_8u_CxR_kernel(const Npp8u *pSrc, int nSrcStep, NppiSize oSizeROI, int nSourceChannels,
                                   int nOutputChannels, Npp8u *pDeviceBuffer, Npp64f *pMean, cudaStream_t stream);
cudaError_t nppiMean_16u_CxR_kernel(const Npp16u *pSrc, int nSrcStep, NppiSize oSizeROI, int nSourceChannels,
                                    int nOutputChannels, Npp8u *pDeviceBuffer, Npp64f *pMean, cudaStream_t stream);
cudaError_t nppiAverageError_8u_C1R_kernel(const Npp8u *pSrc1, int nSrc1Step, const Npp8u *pSrc2, int nSrc2Step,
                                           NppiSize oSizeROI, Npp64f *pError, Npp8u *pDeviceBuffer,
                                           cudaStream_t stream);
}

namespace {

NppStatus meanStdDevBufferSize(NppiSize roi, size_t arraysPerBlock, size_t *bufferSize) {
  if (!bufferSize) {
    return NPP_NULL_POINTER_ERROR;
  }
  if (roi.width <= 0 || roi.height <= 0) {
    return NPP_SIZE_ERROR;
  }
  const size_t pixels = static_cast<size_t>(roi.width) * static_cast<size_t>(roi.height);
  const size_t blocks = (pixels + 255) / 256;
  *bufferSize = blocks * arraysPerBlock * sizeof(double);
  return NPP_SUCCESS;
}

} // namespace

NppStatus nppiMeanGetBufferHostSize_8u_C1R_Ctx(NppiSize oSizeROI, int *hpBufferSize,
                                                NppStreamContext /*nppStreamCtx*/) {
  if (!hpBufferSize) {
    return NPP_NULL_POINTER_ERROR;
  }
  size_t size = 0;
  const NppStatus status = meanStdDevBufferSize(oSizeROI, 1, &size);
  if (status == NPP_SUCCESS) {
    *hpBufferSize = static_cast<int>(size);
  }
  return status;
}

NppStatus nppiMeanGetBufferHostSize_8u_C1R(NppiSize oSizeROI, int *hpBufferSize) {
  NppStreamContext context{};
  return nppiMeanGetBufferHostSize_8u_C1R_Ctx(oSizeROI, hpBufferSize, context);
}

namespace {

NppStatus meanMultiChannelBufferSize(NppiSize roi, int channels, int *bufferSize) {
  if (!bufferSize) {
    return NPP_NULL_POINTER_ERROR;
  }
  size_t size = 0;
  const NppStatus status = meanStdDevBufferSize(roi, static_cast<size_t>(channels), &size);
  if (status == NPP_SUCCESS) {
    *bufferSize = static_cast<int>(size);
  }
  return status;
}

NppStatus mean8uMultiChannel(const Npp8u *pSrc, int nSrcStep, NppiSize oSizeROI, int sourceChannels,
                             int outputChannels, Npp8u *pDeviceBuffer, Npp64f *pMean,
                             NppStreamContext nppStreamCtx) {
  if (!pSrc || !pDeviceBuffer || !pMean) {
    return NPP_NULL_POINTER_ERROR;
  }
  if (oSizeROI.width <= 0 || oSizeROI.height <= 0) {
    return NPP_SIZE_ERROR;
  }
  if (nSrcStep < oSizeROI.width * sourceChannels) {
    return NPP_STEP_ERROR;
  }
  const cudaError_t status = nppiMean_8u_CxR_kernel(pSrc, nSrcStep, oSizeROI, sourceChannels, outputChannels,
                                                     pDeviceBuffer, pMean, nppStreamCtx.hStream);
  return status == cudaSuccess ? NPP_SUCCESS : NPP_CUDA_KERNEL_EXECUTION_ERROR;
}

NppStatus mean16uMultiChannel(const Npp16u *pSrc, int nSrcStep, NppiSize oSizeROI, int sourceChannels,
                              int outputChannels, Npp8u *pDeviceBuffer, Npp64f *pMean,
                              NppStreamContext nppStreamCtx) {
  if (!pSrc || !pDeviceBuffer || !pMean) {
    return NPP_NULL_POINTER_ERROR;
  }
  if (oSizeROI.width <= 0 || oSizeROI.height <= 0) {
    return NPP_SIZE_ERROR;
  }
  if (nSrcStep < oSizeROI.width * sourceChannels * static_cast<int>(sizeof(Npp16u))) {
    return NPP_STEP_ERROR;
  }
  const cudaError_t status = nppiMean_16u_CxR_kernel(pSrc, nSrcStep, oSizeROI, sourceChannels, outputChannels,
                                                      pDeviceBuffer, pMean, nppStreamCtx.hStream);
  return status == cudaSuccess ? NPP_SUCCESS : NPP_CUDA_KERNEL_EXECUTION_ERROR;
}

} // namespace

NppStatus nppiMeanGetBufferHostSize_8u_C3R_Ctx(NppiSize roi, int *size, NppStreamContext) {
  return meanMultiChannelBufferSize(roi, 3, size);
}
NppStatus nppiMeanGetBufferHostSize_8u_C3R(NppiSize roi, int *size) {
  NppStreamContext context{};
  return nppiMeanGetBufferHostSize_8u_C3R_Ctx(roi, size, context);
}
NppStatus nppiMeanGetBufferHostSize_8u_C4R_Ctx(NppiSize roi, int *size, NppStreamContext) {
  return meanMultiChannelBufferSize(roi, 4, size);
}
NppStatus nppiMeanGetBufferHostSize_8u_C4R(NppiSize roi, int *size) {
  NppStreamContext context{};
  return nppiMeanGetBufferHostSize_8u_C4R_Ctx(roi, size, context);
}
NppStatus nppiMeanGetBufferHostSize_8u_AC4R_Ctx(NppiSize roi, int *size, NppStreamContext) {
  return meanMultiChannelBufferSize(roi, 3, size);
}
NppStatus nppiMeanGetBufferHostSize_8u_AC4R(NppiSize roi, int *size) {
  NppStreamContext context{};
  return nppiMeanGetBufferHostSize_8u_AC4R_Ctx(roi, size, context);
}

NppStatus nppiMean_8u_C3R_Ctx(const Npp8u *src, int step, NppiSize roi, Npp8u *buffer, Npp64f *mean,
                              NppStreamContext context) {
  return mean8uMultiChannel(src, step, roi, 3, 3, buffer, mean, context);
}
NppStatus nppiMean_8u_C3R(const Npp8u *src, int step, NppiSize roi, Npp8u *buffer, Npp64f *mean) {
  NppStreamContext context{};
  return nppiMean_8u_C3R_Ctx(src, step, roi, buffer, mean, context);
}
NppStatus nppiMean_8u_C4R_Ctx(const Npp8u *src, int step, NppiSize roi, Npp8u *buffer, Npp64f *mean,
                              NppStreamContext context) {
  return mean8uMultiChannel(src, step, roi, 4, 4, buffer, mean, context);
}
NppStatus nppiMean_8u_C4R(const Npp8u *src, int step, NppiSize roi, Npp8u *buffer, Npp64f *mean) {
  NppStreamContext context{};
  return nppiMean_8u_C4R_Ctx(src, step, roi, buffer, mean, context);
}
NppStatus nppiMean_8u_AC4R_Ctx(const Npp8u *src, int step, NppiSize roi, Npp8u *buffer, Npp64f *mean,
                               NppStreamContext context) {
  return mean8uMultiChannel(src, step, roi, 4, 3, buffer, mean, context);
}
NppStatus nppiMean_8u_AC4R(const Npp8u *src, int step, NppiSize roi, Npp8u *buffer, Npp64f *mean) {
  NppStreamContext context{};
  return nppiMean_8u_AC4R_Ctx(src, step, roi, buffer, mean, context);
}

NppStatus nppiMeanGetBufferHostSize_16u_C1R_Ctx(NppiSize roi, int *size, NppStreamContext) {
  return meanMultiChannelBufferSize(roi, 1, size);
}
NppStatus nppiMeanGetBufferHostSize_16u_C1R(NppiSize roi, int *size) {
  NppStreamContext context{};
  return nppiMeanGetBufferHostSize_16u_C1R_Ctx(roi, size, context);
}
NppStatus nppiMeanGetBufferHostSize_16u_C3R_Ctx(NppiSize roi, int *size, NppStreamContext) {
  return meanMultiChannelBufferSize(roi, 3, size);
}
NppStatus nppiMeanGetBufferHostSize_16u_C3R(NppiSize roi, int *size) {
  NppStreamContext context{};
  return nppiMeanGetBufferHostSize_16u_C3R_Ctx(roi, size, context);
}
NppStatus nppiMeanGetBufferHostSize_16u_C4R_Ctx(NppiSize roi, int *size, NppStreamContext) {
  return meanMultiChannelBufferSize(roi, 4, size);
}
NppStatus nppiMeanGetBufferHostSize_16u_C4R(NppiSize roi, int *size) {
  NppStreamContext context{};
  return nppiMeanGetBufferHostSize_16u_C4R_Ctx(roi, size, context);
}
NppStatus nppiMeanGetBufferHostSize_16u_AC4R_Ctx(NppiSize roi, int *size, NppStreamContext) {
  return meanMultiChannelBufferSize(roi, 3, size);
}
NppStatus nppiMeanGetBufferHostSize_16u_AC4R(NppiSize roi, int *size) {
  NppStreamContext context{};
  return nppiMeanGetBufferHostSize_16u_AC4R_Ctx(roi, size, context);
}

NppStatus nppiMean_16u_C1R_Ctx(const Npp16u *src, int step, NppiSize roi, Npp8u *buffer, Npp64f *mean,
                               NppStreamContext context) {
  return mean16uMultiChannel(src, step, roi, 1, 1, buffer, mean, context);
}
NppStatus nppiMean_16u_C1R(const Npp16u *src, int step, NppiSize roi, Npp8u *buffer, Npp64f *mean) {
  NppStreamContext context{};
  return nppiMean_16u_C1R_Ctx(src, step, roi, buffer, mean, context);
}
NppStatus nppiMean_16u_C3R_Ctx(const Npp16u *src, int step, NppiSize roi, Npp8u *buffer, Npp64f *mean,
                               NppStreamContext context) {
  return mean16uMultiChannel(src, step, roi, 3, 3, buffer, mean, context);
}
NppStatus nppiMean_16u_C3R(const Npp16u *src, int step, NppiSize roi, Npp8u *buffer, Npp64f *mean) {
  NppStreamContext context{};
  return nppiMean_16u_C3R_Ctx(src, step, roi, buffer, mean, context);
}
NppStatus nppiMean_16u_C4R_Ctx(const Npp16u *src, int step, NppiSize roi, Npp8u *buffer, Npp64f *mean,
                               NppStreamContext context) {
  return mean16uMultiChannel(src, step, roi, 4, 4, buffer, mean, context);
}
NppStatus nppiMean_16u_C4R(const Npp16u *src, int step, NppiSize roi, Npp8u *buffer, Npp64f *mean) {
  NppStreamContext context{};
  return nppiMean_16u_C4R_Ctx(src, step, roi, buffer, mean, context);
}
NppStatus nppiMean_16u_AC4R_Ctx(const Npp16u *src, int step, NppiSize roi, Npp8u *buffer, Npp64f *mean,
                                NppStreamContext context) {
  return mean16uMultiChannel(src, step, roi, 4, 3, buffer, mean, context);
}
NppStatus nppiMean_16u_AC4R(const Npp16u *src, int step, NppiSize roi, Npp8u *buffer, Npp64f *mean) {
  NppStreamContext context{};
  return nppiMean_16u_AC4R_Ctx(src, step, roi, buffer, mean, context);
}

NppStatus nppiAverageErrorGetBufferHostSize_8u_C1R_Ctx(NppiSize oSizeROI, int *hpBufferSize,
                                                        NppStreamContext /*nppStreamCtx*/) {
  if (!hpBufferSize) {
    return NPP_NULL_POINTER_ERROR;
  }
  size_t size = 0;
  const NppStatus status = meanStdDevBufferSize(oSizeROI, 1, &size);
  if (status == NPP_SUCCESS) {
    *hpBufferSize = static_cast<int>(size);
  }
  return status;
}

NppStatus nppiAverageErrorGetBufferHostSize_8u_C1R(NppiSize oSizeROI, int *hpBufferSize) {
  NppStreamContext context{};
  return nppiAverageErrorGetBufferHostSize_8u_C1R_Ctx(oSizeROI, hpBufferSize, context);
}

NppStatus nppiMean_8u_C1R_Ctx(const Npp8u *pSrc, int nSrcStep, NppiSize oSizeROI, Npp8u *pDeviceBuffer,
                              Npp64f *pMean, NppStreamContext nppStreamCtx) {
  if (!pSrc || !pDeviceBuffer || !pMean) {
    return NPP_NULL_POINTER_ERROR;
  }
  if (oSizeROI.width <= 0 || oSizeROI.height <= 0) {
    return NPP_SIZE_ERROR;
  }
  if (nSrcStep < oSizeROI.width) {
    return NPP_STEP_ERROR;
  }
  const cudaError_t status =
      nppiMean_8u_C1R_kernel(pSrc, nSrcStep, oSizeROI, pDeviceBuffer, pMean, nppStreamCtx.hStream);
  return status == cudaSuccess ? NPP_SUCCESS : NPP_CUDA_KERNEL_EXECUTION_ERROR;
}

NppStatus nppiMean_8u_C1R(const Npp8u *pSrc, int nSrcStep, NppiSize oSizeROI, Npp8u *pDeviceBuffer,
                          Npp64f *pMean) {
  NppStreamContext context{};
  return nppiMean_8u_C1R_Ctx(pSrc, nSrcStep, oSizeROI, pDeviceBuffer, pMean, context);
}

NppStatus nppiAverageError_8u_C1R_Ctx(const Npp8u *pSrc1, int nSrc1Step, const Npp8u *pSrc2, int nSrc2Step,
                                      NppiSize oSizeROI, Npp64f *pError, Npp8u *pDeviceBuffer,
                                      NppStreamContext nppStreamCtx) {
  if (!pSrc1 || !pSrc2 || !pError || !pDeviceBuffer) {
    return NPP_NULL_POINTER_ERROR;
  }
  if (oSizeROI.width <= 0 || oSizeROI.height <= 0) {
    return NPP_SIZE_ERROR;
  }
  if (nSrc1Step < oSizeROI.width || nSrc2Step < oSizeROI.width) {
    return NPP_STEP_ERROR;
  }
  const cudaError_t status = nppiAverageError_8u_C1R_kernel(
      pSrc1, nSrc1Step, pSrc2, nSrc2Step, oSizeROI, pError, pDeviceBuffer, nppStreamCtx.hStream);
  return status == cudaSuccess ? NPP_SUCCESS : NPP_CUDA_KERNEL_EXECUTION_ERROR;
}

NppStatus nppiAverageError_8u_C1R(const Npp8u *pSrc1, int nSrc1Step, const Npp8u *pSrc2, int nSrc2Step,
                                  NppiSize oSizeROI, Npp64f *pError, Npp8u *pDeviceBuffer) {
  NppStreamContext context{};
  return nppiAverageError_8u_C1R_Ctx(pSrc1, nSrc1Step, pSrc2, nSrc2Step, oSizeROI, pError, pDeviceBuffer, context);
}

// ============ cuda 12.8 ===================
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
  size_t numBlocks = (numPixels + 255) / 256; // Assuming 256 threads per block

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
  size_t numBlocks = (numPixels + 255) / 256; // Assuming 256 threads per block

  // Buffer for reduction: 2 values per block (sum and sum of squares)
  *hpBufferSize = numBlocks * 2 * sizeof(double);

  return NPP_SUCCESS;
}

NppStatus nppiMeanStdDevGetBufferHostSize_32f_C1R(NppiSize oSizeROI, size_t *hpBufferSize) {
  NppStreamContext ctx;
  ctx.hStream = 0; // Default stream
  return nppiMeanStdDevGetBufferHostSize_32f_C1R_Ctx(oSizeROI, hpBufferSize, ctx);
}

NppStatus nppiMeanStdDevGetBufferHostSize_8u_C3CR_Ctx(NppiSize oSizeROI, size_t *hpBufferSize,
                                                      NppStreamContext /*nppStreamCtx*/) {
  return meanStdDevBufferSize(oSizeROI, 2, hpBufferSize);
}

NppStatus nppiMeanStdDevGetBufferHostSize_8u_C3CR(NppiSize oSizeROI, size_t *hpBufferSize) {
  NppStreamContext ctx{};
  return nppiMeanStdDevGetBufferHostSize_8u_C3CR_Ctx(oSizeROI, hpBufferSize, ctx);
}

NppStatus nppiMeanStdDevGetBufferHostSize_8u_C3CMR_Ctx(NppiSize oSizeROI, size_t *hpBufferSize,
                                                       NppStreamContext /*nppStreamCtx*/) {
  return meanStdDevBufferSize(oSizeROI, 3, hpBufferSize);
}

NppStatus nppiMeanStdDevGetBufferHostSize_8u_C3CMR(NppiSize oSizeROI, size_t *hpBufferSize) {
  NppStreamContext ctx{};
  return nppiMeanStdDevGetBufferHostSize_8u_C3CMR_Ctx(oSizeROI, hpBufferSize, ctx);
}

// ================== below 12.8 ============================
// Buffer size calculation for 8u single channel
NppStatus nppiMeanStdDevGetBufferHostSize_8u_C1R_Ctx(NppiSize oSizeROI, int *hpBufferSize,
                                                     NppStreamContext nppStreamCtx) {
  size_t size = 0;
  auto ret = nppiMeanStdDevGetBufferHostSize_8u_C1R_Ctx(oSizeROI, &size, nppStreamCtx);
  *hpBufferSize = static_cast<int>(size);
  return ret;
}

NppStatus nppiMeanStdDevGetBufferHostSize_8u_C1R(NppiSize oSizeROI, int *hpBufferSize) {
  size_t size = 0;
  auto ret = nppiMeanStdDevGetBufferHostSize_8u_C1R(oSizeROI, &size);
  *hpBufferSize = static_cast<int>(size);
  return ret;
}

// Buffer size calculation for 32-bit float single channel
NppStatus nppiMeanStdDevGetBufferHostSize_32f_C1R_Ctx(NppiSize oSizeROI, int *hpBufferSize,
                                                      NppStreamContext nppStreamCtx) {
  size_t size = 0;
  auto ret = nppiMeanStdDevGetBufferHostSize_32f_C1R_Ctx(oSizeROI, &size, nppStreamCtx);
  *hpBufferSize = static_cast<int>(size);
  return ret;
}

NppStatus nppiMeanStdDevGetBufferHostSize_32f_C1R(NppiSize oSizeROI, int *hpBufferSize) {
  size_t size = 0;
  auto ret = nppiMeanStdDevGetBufferHostSize_32f_C1R(oSizeROI, &size);
  *hpBufferSize = static_cast<int>(size);
  return ret;
}

NppStatus nppiMeanStdDevGetBufferHostSize_8u_C3CR_Ctx(NppiSize oSizeROI, int *hpBufferSize,
                                                      NppStreamContext nppStreamCtx) {
  if (!hpBufferSize) {
    return NPP_NULL_POINTER_ERROR;
  }
  size_t size = 0;
  const NppStatus status = nppiMeanStdDevGetBufferHostSize_8u_C3CR_Ctx(oSizeROI, &size, nppStreamCtx);
  if (status == NPP_SUCCESS) {
    *hpBufferSize = static_cast<int>(size);
  }
  return status;
}

NppStatus nppiMeanStdDevGetBufferHostSize_8u_C3CR(NppiSize oSizeROI, int *hpBufferSize) {
  if (!hpBufferSize) {
    return NPP_NULL_POINTER_ERROR;
  }
  size_t size = 0;
  const NppStatus status = nppiMeanStdDevGetBufferHostSize_8u_C3CR(oSizeROI, &size);
  if (status == NPP_SUCCESS) {
    *hpBufferSize = static_cast<int>(size);
  }
  return status;
}

NppStatus nppiMeanStdDevGetBufferHostSize_8u_C3CMR_Ctx(NppiSize oSizeROI, int *hpBufferSize,
                                                       NppStreamContext nppStreamCtx) {
  if (!hpBufferSize) {
    return NPP_NULL_POINTER_ERROR;
  }
  size_t size = 0;
  const NppStatus status = nppiMeanStdDevGetBufferHostSize_8u_C3CMR_Ctx(oSizeROI, &size, nppStreamCtx);
  if (status == NPP_SUCCESS) {
    *hpBufferSize = static_cast<int>(size);
  }
  return status;
}

NppStatus nppiMeanStdDevGetBufferHostSize_8u_C3CMR(NppiSize oSizeROI, int *hpBufferSize) {
  if (!hpBufferSize) {
    return NPP_NULL_POINTER_ERROR;
  }
  size_t size = 0;
  const NppStatus status = nppiMeanStdDevGetBufferHostSize_8u_C3CMR(oSizeROI, &size);
  if (status == NPP_SUCCESS) {
    *hpBufferSize = static_cast<int>(size);
  }
  return status;
}
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
  NppStreamContext ctx;
  ctx.hStream = 0; // Default stream
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
  NppStreamContext ctx;
  ctx.hStream = 0; // Default stream
  return nppiMean_StdDev_32f_C1R_Ctx(pSrc, nSrcStep, oSizeROI, pDeviceBuffer, pMean, pStdDev, ctx);
}

NppStatus nppiMean_StdDev_8u_C3CR_Ctx(const Npp8u *pSrc, int nSrcStep, NppiSize oSizeROI, int nCOI,
                                      Npp8u *pDeviceBuffer, Npp64f *pMean, Npp64f *pStdDev,
                                      NppStreamContext nppStreamCtx) {
  if (!pSrc || !pDeviceBuffer || !pMean || !pStdDev) {
    return NPP_NULL_POINTER_ERROR;
  }
  if (oSizeROI.width <= 0 || oSizeROI.height <= 0) {
    return NPP_SIZE_ERROR;
  }
  if (nSrcStep < oSizeROI.width * 3) {
    return NPP_STEP_ERROR;
  }
  if (nCOI < 1 || nCOI > 3) {
    return NPP_COI_ERROR;
  }
  const cudaError_t status = nppiMean_StdDev_8u_C3CR_kernel(pSrc, nSrcStep, oSizeROI, nCOI, pDeviceBuffer, pMean,
                                                            pStdDev, nppStreamCtx.hStream);
  return status == cudaSuccess ? NPP_SUCCESS : NPP_CUDA_KERNEL_EXECUTION_ERROR;
}

NppStatus nppiMean_StdDev_8u_C3CR(const Npp8u *pSrc, int nSrcStep, NppiSize oSizeROI, int nCOI,
                                  Npp8u *pDeviceBuffer, Npp64f *pMean, Npp64f *pStdDev) {
  NppStreamContext ctx{};
  return nppiMean_StdDev_8u_C3CR_Ctx(pSrc, nSrcStep, oSizeROI, nCOI, pDeviceBuffer, pMean, pStdDev, ctx);
}

// ============ MASKED VERSIONS C1MR ===================

// ============ cuda 12.8 ===================
// Buffer size calculation for 8u single channel masked
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

NppStatus nppiMeanStdDevGetBufferHostSize_8u_C1MR(NppiSize oSizeROI, size_t *hpBufferSize) {
  NppStreamContext ctx;
  ctx.hStream = 0;
  return nppiMeanStdDevGetBufferHostSize_8u_C1MR_Ctx(oSizeROI, hpBufferSize, ctx);
}

// Buffer size calculation for 32f single channel masked
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

NppStatus nppiMeanStdDevGetBufferHostSize_32f_C1MR(NppiSize oSizeROI, size_t *hpBufferSize) {
  NppStreamContext ctx;
  ctx.hStream = 0;
  return nppiMeanStdDevGetBufferHostSize_32f_C1MR_Ctx(oSizeROI, hpBufferSize, ctx);
}

// ================== below 12.8 ============================
// Buffer size calculation for 8u single channel masked (int version)
NppStatus nppiMeanStdDevGetBufferHostSize_8u_C1MR_Ctx(NppiSize oSizeROI, int *hpBufferSize,
                                                      NppStreamContext nppStreamCtx) {
  size_t size = 0;
  auto ret = nppiMeanStdDevGetBufferHostSize_8u_C1MR_Ctx(oSizeROI, &size, nppStreamCtx);
  *hpBufferSize = static_cast<int>(size);
  return ret;
}

NppStatus nppiMeanStdDevGetBufferHostSize_8u_C1MR(NppiSize oSizeROI, int *hpBufferSize) {
  size_t size = 0;
  auto ret = nppiMeanStdDevGetBufferHostSize_8u_C1MR(oSizeROI, &size);
  *hpBufferSize = static_cast<int>(size);
  return ret;
}

// Buffer size calculation for 32f single channel masked (int version)
NppStatus nppiMeanStdDevGetBufferHostSize_32f_C1MR_Ctx(NppiSize oSizeROI, int *hpBufferSize,
                                                       NppStreamContext nppStreamCtx) {
  size_t size = 0;
  auto ret = nppiMeanStdDevGetBufferHostSize_32f_C1MR_Ctx(oSizeROI, &size, nppStreamCtx);
  *hpBufferSize = static_cast<int>(size);
  return ret;
}

NppStatus nppiMeanStdDevGetBufferHostSize_32f_C1MR(NppiSize oSizeROI, int *hpBufferSize) {
  size_t size = 0;
  auto ret = nppiMeanStdDevGetBufferHostSize_32f_C1MR(oSizeROI, &size);
  *hpBufferSize = static_cast<int>(size);
  return ret;
}

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
  NppStreamContext ctx;
  ctx.hStream = 0;
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
  NppStreamContext ctx;
  ctx.hStream = 0;
  return nppiMean_StdDev_32f_C1MR_Ctx(pSrc, nSrcStep, pMask, nMaskStep, oSizeROI, pDeviceBuffer, pMean, pStdDev, ctx);
}

NppStatus nppiMean_StdDev_8u_C3CMR_Ctx(const Npp8u *pSrc, int nSrcStep, const Npp8u *pMask, int nMaskStep,
                                       NppiSize oSizeROI, int nCOI, Npp8u *pDeviceBuffer, Npp64f *pMean,
                                       Npp64f *pStdDev, NppStreamContext nppStreamCtx) {
  if (!pSrc || !pMask || !pDeviceBuffer || !pMean || !pStdDev) {
    return NPP_NULL_POINTER_ERROR;
  }
  if (oSizeROI.width <= 0 || oSizeROI.height <= 0) {
    return NPP_SIZE_ERROR;
  }
  if (nSrcStep < oSizeROI.width * 3 || nMaskStep < oSizeROI.width) {
    return NPP_STEP_ERROR;
  }
  if (nCOI < 1 || nCOI > 3) {
    return NPP_COI_ERROR;
  }
  const cudaError_t status = nppiMean_StdDev_8u_C3CMR_kernel(
      pSrc, nSrcStep, pMask, nMaskStep, oSizeROI, nCOI, pDeviceBuffer, pMean, pStdDev, nppStreamCtx.hStream);
  return status == cudaSuccess ? NPP_SUCCESS : NPP_CUDA_KERNEL_EXECUTION_ERROR;
}

NppStatus nppiMean_StdDev_8u_C3CMR(const Npp8u *pSrc, int nSrcStep, const Npp8u *pMask, int nMaskStep,
                                   NppiSize oSizeROI, int nCOI, Npp8u *pDeviceBuffer, Npp64f *pMean,
                                   Npp64f *pStdDev) {
  NppStreamContext ctx{};
  return nppiMean_StdDev_8u_C3CMR_Ctx(pSrc, nSrcStep, pMask, nMaskStep, oSizeROI, nCOI, pDeviceBuffer, pMean, pStdDev,
                                      ctx);
}
