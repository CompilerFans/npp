#include "../../npp_internal.h"
#include "npp.h"
#include <cuda_runtime.h>

// Declare GPU functions
extern "C" {
NppStatus nppiFilterCannyBorderGetBufferSize_8u_C1R_Ctx_impl(NppiSize oSizeROI, int *hpBufferSize);
NppStatus nppiFilterCannyBorder_8u_C1R_Ctx_impl(const Npp8u *pSrc, int nSrcStep, NppiSize oSrcSizeROI,
                                                NppiPoint oSrcOffset, Npp8u *pDst, int nDstStep, NppiSize oDstSizeROI,
                                                NppiDifferentialKernel eFilterType, NppiMaskSize eMaskSize,
                                                Npp16s nLowThreshold, Npp16s nHighThreshold, NppiNorm eNorm,
                                                NppiBorderType eBorderType, Npp8u *pDeviceBuffer,
                                                NppStreamContext nppStreamCtx);
}

// Get required buffer size - simplified version (used by GPU Samples)
NppStatus nppiFilterCannyBorderGetBufferSize(NppiSize oSizeROI, int *hpBufferSize) {
  if (hpBufferSize == nullptr) {
    return NPP_NULL_POINTER_ERROR;
  }

  if (oSizeROI.width < 0 || oSizeROI.height < 0) {
    return NPP_SIZE_ERROR;
  }

  return nppiFilterCannyBorderGetBufferSize_8u_C1R_Ctx_impl(oSizeROI, hpBufferSize);
}

// Get required buffer size
NppStatus nppiFilterCannyBorderGetBufferSize_8u_C1R(NppiSize oSizeROI, int *hpBufferSize) {
  if (hpBufferSize == nullptr) {
    return NPP_NULL_POINTER_ERROR;
  }

  if (oSizeROI.width < 0 || oSizeROI.height < 0) {
    return NPP_SIZE_ERROR;
  }

  return nppiFilterCannyBorderGetBufferSize_8u_C1R_Ctx_impl(oSizeROI, hpBufferSize);
}

NppStatus nppiFilterCannyBorderGetBufferSize_8u_C1R_Ctx(NppiSize oSizeROI, int *hpBufferSize,
                                                        NppStreamContext nppStreamCtx) {
  // Use nppStreamCtx parameter for simple validation to avoid unused warning
  if (nppStreamCtx.nCudaDeviceId < -1) {
    return NPP_BAD_ARGUMENT_ERROR;
  }
  return nppiFilterCannyBorderGetBufferSize_8u_C1R(oSizeROI, hpBufferSize);
}

// CannyEdge detectionimplementation
NppStatus nppiFilterCannyBorder_8u_C1R(const Npp8u *pSrc, int nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset,
                                       Npp8u *pDst, int nDstStep, NppiSize oSizeROI, NppiDifferentialKernel eFilterType,
                                       NppiMaskSize eMaskSize, Npp16s nLowThreshold, Npp16s nHighThreshold,
                                       NppiNorm eNorm, NppiBorderType eBorderType, Npp8u *pDeviceBuffer) {
  // Parameter validation
  if (pSrc == nullptr || pDst == nullptr || pDeviceBuffer == nullptr) {
    return NPP_NULL_POINTER_ERROR;
  }

  if (oSrcSize.width <= 0 || oSrcSize.height <= 0 || oSizeROI.width <= 0 || oSizeROI.height <= 0) {
    return NPP_SIZE_ERROR;
  }

  if (nSrcStep <= 0 || nDstStep <= 0) {
    return NPP_STEP_ERROR;
  }

  if (eMaskSize != NPP_MASK_SIZE_3_X_3 && eMaskSize != NPP_MASK_SIZE_5_X_5) {
    return NPP_MASK_SIZE_ERROR;
  }

  if (nLowThreshold < 0 || nHighThreshold < 0 || nLowThreshold >= nHighThreshold) {
    return NPP_BAD_ARGUMENT_ERROR;
  }

  if (eBorderType != NPP_BORDER_REPLICATE && eBorderType != NPP_BORDER_CONSTANT) {
    return NPP_BAD_ARGUMENT_ERROR;
  }

  NppStreamContext nppStreamCtx = nppCreateDefaultStreamContext();

  return nppiFilterCannyBorder_8u_C1R_Ctx_impl(pSrc, nSrcStep, oSrcSize, oSrcOffset, pDst, nDstStep, oSizeROI,
                                               eFilterType, eMaskSize, nLowThreshold, nHighThreshold, eNorm,
                                               eBorderType, pDeviceBuffer, nppStreamCtx);
}

NppStatus nppiFilterCannyBorder_8u_C1R_Ctx(const Npp8u *pSrc, int nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset,
                                           Npp8u *pDst, int nDstStep, NppiSize oSizeROI,
                                           NppiDifferentialKernel eFilterType, NppiMaskSize eMaskSize,
                                           Npp16s nLowThreshold, Npp16s nHighThreshold, NppiNorm eNorm,
                                           NppiBorderType eBorderType, Npp8u *pDeviceBuffer,
                                           NppStreamContext nppStreamCtx) {
  // Parameter validation
  if (pSrc == nullptr || pDst == nullptr || pDeviceBuffer == nullptr) {
    return NPP_NULL_POINTER_ERROR;
  }

  if (oSrcSize.width <= 0 || oSrcSize.height <= 0 || oSizeROI.width <= 0 || oSizeROI.height <= 0) {
    return NPP_SIZE_ERROR;
  }

  if (nSrcStep <= 0 || nDstStep <= 0) {
    return NPP_STEP_ERROR;
  }

  if (eMaskSize != NPP_MASK_SIZE_3_X_3 && eMaskSize != NPP_MASK_SIZE_5_X_5) {
    return NPP_MASK_SIZE_ERROR;
  }

  if (nLowThreshold < 0 || nHighThreshold < 0 || nLowThreshold >= nHighThreshold) {
    return NPP_BAD_ARGUMENT_ERROR;
  }

  if (eBorderType != NPP_BORDER_REPLICATE && eBorderType != NPP_BORDER_CONSTANT) {
    return NPP_BAD_ARGUMENT_ERROR;
  }

  return nppiFilterCannyBorder_8u_C1R_Ctx_impl(pSrc, nSrcStep, oSrcSize, oSrcOffset, pDst, nDstStep, oSizeROI,
                                               eFilterType, eMaskSize, nLowThreshold, nHighThreshold, eNorm,
                                               eBorderType, pDeviceBuffer, nppStreamCtx);
}
