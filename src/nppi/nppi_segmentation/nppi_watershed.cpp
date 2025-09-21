#include "../../npp_internal.h"
#include "npp.h"
#include <cuda_runtime.h>

// Forward declarations for GPU functions
extern "C" {
NppStatus nppiSegmentWatershedGetBufferSize_8u_C1R_Ctx_impl(NppiSize oSizeROI, size_t *hpBufferSize);
NppStatus nppiSegmentWatershed_8u_C1IR_Ctx_impl(Npp8u *pSrcDst, Npp32s nSrcDstStep, Npp32u *pMarkerLabels,
                                                Npp32s nMarkerLabelsStep, NppiNorm eNorm, NppiSize oSizeROI,
                                                Npp8u *pDeviceBuffer, NppStreamContext nppStreamCtx);
}

// Get buffer size required for Watershed segmentation
NppStatus nppiSegmentWatershedGetBufferSize_8u_C1R(NppiSize oSizeROI, size_t *hpDeviceMemoryBufferSize) {
  if (hpDeviceMemoryBufferSize == nullptr) {
    return NPP_NULL_POINTER_ERROR;
  }

  if (oSizeROI.width < 0 || oSizeROI.height < 0) {
    return NPP_SIZE_ERROR;
  }

  return nppiSegmentWatershedGetBufferSize_8u_C1R_Ctx_impl(oSizeROI, hpDeviceMemoryBufferSize);
}

// Watershed image segmentation
NppStatus nppiSegmentWatershed_8u_C1IR(Npp8u *pSrcDst, Npp32s nSrcDstStep, Npp32u *pMarkerLabels,
                                       Npp32s nMarkerLabelsStep, NppiNorm eNorm,
                                       NppiWatershedSegmentBoundaryType eSegmentBoundaryType, NppiSize oSizeROI,
                                       Npp8u *pDeviceMemoryBuffer) {
  // Parameter validation
  if (pSrcDst == nullptr || pMarkerLabels == nullptr || pDeviceMemoryBuffer == nullptr) {
    return NPP_NULL_POINTER_ERROR;
  }

  if (oSizeROI.width < 0 || oSizeROI.height < 0) {
    return NPP_SIZE_ERROR;
  }

  if (nSrcDstStep <= 0 || nMarkerLabelsStep <= 0) {
    return NPP_STEP_ERROR;
  }

  // Basic validation of boundary type parameter to avoid unused warning
  if (eSegmentBoundaryType < 0) {
    return NPP_BAD_ARGUMENT_ERROR;
  }

  NppStreamContext nppStreamCtx = nppCreateDefaultStreamContext();

  return nppiSegmentWatershed_8u_C1IR_Ctx_impl(pSrcDst, nSrcDstStep, pMarkerLabels, nMarkerLabelsStep, eNorm, oSizeROI,
                                               pDeviceMemoryBuffer, nppStreamCtx);
}

NppStatus nppiSegmentWatershed_8u_C1IR_Ctx(Npp8u *pSrcDst, Npp32s nSrcDstStep, Npp32u *pMarkerLabels,
                                           Npp32s nMarkerLabelsStep, NppiNorm eNorm,
                                           NppiWatershedSegmentBoundaryType eSegmentBoundaryType, NppiSize oSizeROI,
                                           Npp8u *pDeviceMemoryBuffer, NppStreamContext nppStreamCtx) {
  // Parameter validation
  if (pSrcDst == nullptr || pMarkerLabels == nullptr || pDeviceMemoryBuffer == nullptr) {
    return NPP_NULL_POINTER_ERROR;
  }

  if (oSizeROI.width < 0 || oSizeROI.height < 0) {
    return NPP_SIZE_ERROR;
  }

  if (nSrcDstStep <= 0 || nMarkerLabelsStep <= 0) {
    return NPP_STEP_ERROR;
  }

  // Basic validation of boundary type parameter to avoid unused warning
  if (eSegmentBoundaryType < 0) {
    return NPP_BAD_ARGUMENT_ERROR;
  }

  // Validate stream context parameter
  if (nppStreamCtx.nCudaDeviceId < -1) {
    return NPP_BAD_ARGUMENT_ERROR;
  }

  return nppiSegmentWatershed_8u_C1IR_Ctx_impl(pSrcDst, nSrcDstStep, pMarkerLabels, nMarkerLabelsStep, eNorm, oSizeROI,
                                               pDeviceMemoryBuffer, nppStreamCtx);
}
