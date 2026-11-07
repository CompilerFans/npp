#include "../../npp_internal.h"
#include "npp.h"
#include <cuda_runtime.h>

// Forward declarations for GPU functions

extern "C" {
NppStatus nppiSegmentWatershedGetBufferSize_8u_C1R_Ctx_impl(NppiSize oSizeROI, NPP_BUFFER_SIZE_TYPE *hpBufferSize);
NppStatus nppiSegmentWatershed_8u_C1IR_Ctx_impl(Npp8u *pSrcDst, Npp32s nSrcDstStep, Npp32u *pMarkerLabels,
                                                Npp32s nMarkerLabelsStep, NppiNorm eNorm,
                                                NppiWatershedSegmentBoundaryType eSegmentBoundaryType,
                                                NppiSize oSizeROI, Npp8u *pDeviceBuffer,
                                                NppStreamContext nppStreamCtx);
}

// Get buffer size required for Watershed segmentation

#if !NPP_BUFFER_SIZE_IS_SIZE_T
NppStatus nppiSegmentWatershedGetBufferSize_8u_C1R(NppiSize oSizeROI, size_t *hpDeviceMemoryBufferSize) {
  if (hpDeviceMemoryBufferSize == nullptr) {
    return NPP_NULL_POINTER_ERROR;
  }

  if (oSizeROI.width < 0 || oSizeROI.height < 0) {
    return NPP_SIZE_ERROR;
  }

  int size = 0;
  NppStatus status = nppiSegmentWatershedGetBufferSize_8u_C1R_Ctx_impl(oSizeROI, &size);
  if (status == NPP_NO_ERROR && hpDeviceMemoryBufferSize) *hpDeviceMemoryBufferSize = (size_t)size;
  return status;
}
#endif

NppStatus nppiSegmentWatershedGetBufferSize_8u_C1R(NppiSize oSizeROI, NPP_BUFFER_SIZE_TYPE *hpDeviceMemoryBufferSize) {
#if NPP_BUFFER_SIZE_IS_SIZE_T
  return nppiSegmentWatershedGetBufferSize_8u_C1R_Ctx_impl(oSizeROI, hpDeviceMemoryBufferSize);
#else
  size_t size = 0;
  auto ret = nppiSegmentWatershedGetBufferSize_8u_C1R(oSizeROI, &size);
  *hpDeviceMemoryBufferSize = size;
  return ret;
#endif
}

// Watershed image segmentation
NppStatus nppiSegmentWatershed_8u_C1IR(Npp8u *pSrcDst, Npp32s nSrcDstStep, Npp32u *pMarkerLabels,
                                       Npp32s nMarkerLabelsStep, NppiNorm eNorm,
                                       NppiWatershedSegmentBoundaryType eSegmentBoundaryType, NppiSize oSizeROI,
                                       Npp8u *pDeviceMemoryBuffer) {
  // Parameter validation
  if (pSrcDst == nullptr || pDeviceMemoryBuffer == nullptr) {
    return NPP_NULL_POINTER_ERROR;
  }

  if (oSizeROI.width < 0 || oSizeROI.height < 0) {
    return NPP_SIZE_ERROR;
  }

  if (nSrcDstStep <= 0) {
    return NPP_STEP_ERROR;
  }

  // pMarkerLabels can be nullptr (skip label output)
  if (pMarkerLabels != nullptr && nMarkerLabelsStep <= 0) {
    return NPP_STEP_ERROR;
  }

  // Basic validation of boundary type parameter to avoid unused warning
  if (eSegmentBoundaryType < 0) {
    return NPP_BAD_ARGUMENT_ERROR;
  }

  NppStreamContext nppStreamCtx = nppCreateDefaultStreamContext();

  return nppiSegmentWatershed_8u_C1IR_Ctx_impl(pSrcDst, nSrcDstStep, pMarkerLabels, nMarkerLabelsStep, eNorm,
                                               eSegmentBoundaryType, oSizeROI, pDeviceMemoryBuffer, nppStreamCtx);
}

NppStatus nppiSegmentWatershed_8u_C1IR_Ctx(Npp8u *pSrcDst, Npp32s nSrcDstStep, Npp32u *pMarkerLabels,
                                           Npp32s nMarkerLabelsStep, NppiNorm eNorm,
                                           NppiWatershedSegmentBoundaryType eSegmentBoundaryType, NppiSize oSizeROI,
                                           Npp8u *pDeviceMemoryBuffer, NppStreamContext nppStreamCtx) {
  // Parameter validation
  if (pSrcDst == nullptr || pDeviceMemoryBuffer == nullptr) {
    return NPP_NULL_POINTER_ERROR;
  }

  if (oSizeROI.width < 0 || oSizeROI.height < 0) {
    return NPP_SIZE_ERROR;
  }

  if (nSrcDstStep <= 0) {
    return NPP_STEP_ERROR;
  }

  // pMarkerLabels can be nullptr (skip label output)
  if (pMarkerLabels != nullptr && nMarkerLabelsStep <= 0) {
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

  return nppiSegmentWatershed_8u_C1IR_Ctx_impl(pSrcDst, nSrcDstStep, pMarkerLabels, nMarkerLabelsStep, eNorm,
                                               eSegmentBoundaryType, oSizeROI, pDeviceMemoryBuffer, nppStreamCtx);
}