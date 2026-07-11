#include "../../npp_internal.h"
#include "npp.h"
#include <cuda_runtime.h>

// Forward declarations for GPU functions (only _impl functions need extern "C")
extern "C" {
NppStatus nppiLabelMarkersUFGetBufferSize_32u_C1R_Ctx_impl(NppiSize oSizeROI, int *hpBufferSize);
NppStatus nppiLabelMarkersUF_8u32u_C1R_Ctx_impl(const Npp8u *pSrc, int nSrcStep,
                                                 Npp32u *pDst, int nDstStep,
                                                 NppiSize oSizeROI, NppiNorm eNorm,
                                                 Npp8u *pBuffer,
                                                 NppStreamContext nppStreamCtx);
NppStatus nppiLabelMarkersUFBatch_8u32u_C1R_Advanced_Ctx_impl(
    const NppiImageDescriptor *pSrcBatchList,
    NppiImageDescriptor *pDstBatchList,
    int nBatchSize, NppiSize oMaxSizeROI,
    NppiNorm eNorm, NppStreamContext nppStreamCtx);
}

// Get required buffer size for label markers
NppStatus nppiLabelMarkersUFGetBufferSize_32u_C1R(NppiSize oSizeROI, int *hpBufferSize) {
  if (hpBufferSize == nullptr) {
    return NPP_NULL_POINTER_ERROR;
  }

  if (oSizeROI.width <= 0 || oSizeROI.height <= 0) {
    return NPP_SIZE_ERROR;
  }

  return nppiLabelMarkersUFGetBufferSize_32u_C1R_Ctx_impl(oSizeROI, hpBufferSize);
}

// Single image label markers using Union-Find (non-Ctx version)
NppStatus nppiLabelMarkersUF_8u32u_C1R(Npp8u *pSrc, int nSrcStep,
                                        Npp32u *pDst, int nDstStep,
                                        NppiSize oSizeROI, NppiNorm eNorm,
                                        Npp8u *pBuffer) {
  // Parameter validation
  if (pSrc == nullptr || pDst == nullptr || pBuffer == nullptr) {
    return NPP_NULL_POINTER_ERROR;
  }

  if (oSizeROI.width <= 0 || oSizeROI.height <= 0) {
    return NPP_SIZE_ERROR;
  }

  if (nSrcStep <= 0 || nDstStep <= 0) {
    return NPP_STEP_ERROR;
  }

  // Validate norm type (only L1 and Inf are supported)
  if (eNorm != nppiNormL1 && eNorm != nppiNormInf) {
    return NPP_BAD_ARGUMENT_ERROR;
  }

  NppStreamContext nppStreamCtx = nppCreateDefaultStreamContext();

  return nppiLabelMarkersUF_8u32u_C1R_Ctx_impl(pSrc, nSrcStep, pDst, nDstStep,
                                                oSizeROI, eNorm, pBuffer, nppStreamCtx);
}

// Single image label markers using Union-Find (Ctx version)
NppStatus nppiLabelMarkersUF_8u32u_C1R_Ctx(Npp8u *pSrc, int nSrcStep,
                                            Npp32u *pDst, int nDstStep,
                                            NppiSize oSizeROI, NppiNorm eNorm,
                                            Npp8u *pBuffer,
                                            NppStreamContext nppStreamCtx) {
  // Parameter validation
  if (pSrc == nullptr || pDst == nullptr || pBuffer == nullptr) {
    return NPP_NULL_POINTER_ERROR;
  }

  if (oSizeROI.width <= 0 || oSizeROI.height <= 0) {
    return NPP_SIZE_ERROR;
  }

  if (nSrcStep <= 0 || nDstStep <= 0) {
    return NPP_STEP_ERROR;
  }

  // Validate norm type (only L1 and Inf are supported)
  if (eNorm != nppiNormL1 && eNorm != nppiNormInf) {
    return NPP_BAD_ARGUMENT_ERROR;
  }

  // Validate stream context
  if (nppStreamCtx.nCudaDeviceId < -1) {
    return NPP_BAD_ARGUMENT_ERROR;
  }

  return nppiLabelMarkersUF_8u32u_C1R_Ctx_impl(pSrc, nSrcStep, pDst, nDstStep,
                                                oSizeROI, eNorm, pBuffer, nppStreamCtx);
}

// Batch image label markers using Union-Find (Advanced Ctx version)
NppStatus nppiLabelMarkersUFBatch_8u32u_C1R_Advanced_Ctx(
    const NppiImageDescriptor *pSrcBatchList,
    NppiImageDescriptor *pDstBatchList,
    int nBatchSize, NppiSize oMaxSizeROI,
    NppiNorm eNorm, NppStreamContext nppStreamCtx) {
  // Parameter validation
  if (pSrcBatchList == nullptr || pDstBatchList == nullptr) {
    return NPP_NULL_POINTER_ERROR;
  }

  if (nBatchSize <= 0) {
    return NPP_BAD_ARGUMENT_ERROR;
  }

  if (oMaxSizeROI.width <= 0 || oMaxSizeROI.height <= 0) {
    return NPP_SIZE_ERROR;
  }

  // Validate norm type (only L1 and Inf are supported)
  if (eNorm != nppiNormL1 && eNorm != nppiNormInf) {
    return NPP_BAD_ARGUMENT_ERROR;
  }

  // Validate stream context
  if (nppStreamCtx.nCudaDeviceId < -1) {
    return NPP_BAD_ARGUMENT_ERROR;
  }

  return nppiLabelMarkersUFBatch_8u32u_C1R_Advanced_Ctx_impl(
      pSrcBatchList, pDstBatchList, nBatchSize, oMaxSizeROI, eNorm, nppStreamCtx);
}
