#include "../../npp_version_compat.h"
#include "npp.h"
#include <cmath>
#include <cstdio>
#include <cstring>
#include <cuda_runtime.h>

// Forward declarations for mpp host func implementations
extern "C" {
NppStatus nppiHistogramEvenGetBufferSize_8u_C1R_Ctx_impl(NppiSize oSizeROI, int nLevels, size_t *hpBufferSize);
NppStatus nppiHistogramEven_8u_C1R_Ctx_impl(const Npp8u *pSrc, int nSrcStep, NppiSize oSizeROI, Npp32s *pHist,
                                            int nLevels, Npp32s nLowerLevel, Npp32s nUpperLevel, Npp8u *pDeviceBuffer,
                                            NppStreamContext nppStreamCtx);
}

// Internal implementation (always uses size_t internally)
static NppStatus nppiHistogramEvenGetBufferSize_8u_C1R_Ctx_internal(NppiSize oSizeROI, int nLevels,
                                                                    size_t *hpBufferSize,
                                                                    NppStreamContext nppStreamCtx) {
  // Validate input parameters
  if (!hpBufferSize) {
    return NPP_NULL_POINTER_ERROR;
  }

  // Stream context validation - accept any valid device ID
  (void)nppStreamCtx; // Avoid unused warning

  if (oSizeROI.width < 0 || oSizeROI.height < 0) {
    return NPP_SIZE_ERROR;
  }

  if (nLevels < 2) {
    return NPP_HISTOGRAM_NUMBER_OF_LEVELS_ERROR;
  }

  return nppiHistogramEvenGetBufferSize_8u_C1R_Ctx_impl(oSizeROI, nLevels, hpBufferSize);
}

NppStatus nppiHistogramEvenGetBufferSize_8u_C1R_Ctx(NppiSize oSizeROI, int nLevels, size_t *hpBufferSize,
                                                    NppStreamContext nppStreamCtx) {
  return nppiHistogramEvenGetBufferSize_8u_C1R_Ctx_internal(oSizeROI, nLevels, hpBufferSize, nppStreamCtx);
}

NppStatus nppiHistogramEvenGetBufferSize_8u_C1R(NppiSize oSizeROI, int nLevels, size_t *hpBufferSize) {
  NppStreamContext nppStreamCtx;
  nppGetStreamContext(&nppStreamCtx);
  return nppiHistogramEvenGetBufferSize_8u_C1R_Ctx(oSizeROI, nLevels, hpBufferSize, nppStreamCtx);
}

// CUDA SDK 12.2 and earlier API (uses int) - always provide these for backward compatibility
NppStatus nppiHistogramEvenGetBufferSize_8u_C1R_Ctx_int(NppiSize oSizeROI, int nLevels, int *hpBufferSize,
                                                        NppStreamContext nppStreamCtx) {
  size_t temp_size;
  NppStatus status = nppiHistogramEvenGetBufferSize_8u_C1R_Ctx_internal(oSizeROI, nLevels, &temp_size, nppStreamCtx);

  if (status == NPP_SUCCESS) {
    // Convert size_t to int (with overflow check)
    if (temp_size > static_cast<size_t>(INT_MAX)) {
      return NPP_MEMORY_ALLOCATION_ERR;
    }
    *hpBufferSize = static_cast<int>(temp_size);
  }

  return status;
}

NppStatus nppiHistogramEvenGetBufferSize_8u_C1R_int(NppiSize oSizeROI, int nLevels, int *hpBufferSize) {
  NppStreamContext nppStreamCtx;
  nppGetStreamContext(&nppStreamCtx);
  return nppiHistogramEvenGetBufferSize_8u_C1R_Ctx_int(oSizeROI, nLevels, hpBufferSize, nppStreamCtx);
}

NppStatus nppiHistogramEvenGetBufferSize_8u_C1R(NppiSize oSizeROI, int nLevels, int *hpBufferSize) {
  return nppiHistogramEvenGetBufferSize_8u_C1R_int(oSizeROI, nLevels, hpBufferSize);
}

NppStatus nppiHistogramEvenGetBufferSize_8u_C1R_Ctx(NppiSize oSizeROI, int nLevels, int *hpBufferSize,
                                                    NppStreamContext nppStreamCtx) {
  return nppiHistogramEvenGetBufferSize_8u_C1R_Ctx_int(oSizeROI, nLevels, hpBufferSize, nppStreamCtx);
}

NppStatus nppiHistogramEven_8u_C1R_Ctx(const Npp8u *pSrc, int nSrcStep, NppiSize oSizeROI, Npp32s *pHist, int nLevels,
                                       Npp32s nLowerLevel, Npp32s nUpperLevel, Npp8u *pDeviceBuffer,
                                       NppStreamContext nppStreamCtx) {
  // Validate input parameters
  if (!pSrc || !pHist || !pDeviceBuffer) {
    return NPP_NULL_POINTER_ERROR;
  }

  if (oSizeROI.width < 0 || oSizeROI.height < 0) {
    return NPP_SIZE_ERROR;
  }

  if (nLevels < 2) {
    return NPP_HISTOGRAM_NUMBER_OF_LEVELS_ERROR;
  }

  if (nSrcStep < oSizeROI.width) {
    return NPP_STEP_ERROR;
  }

  if (nLowerLevel >= nUpperLevel) {
    return NPP_RANGE_ERROR;
  }

  // Call GPU kernel implementation
  return nppiHistogramEven_8u_C1R_Ctx_impl(pSrc, nSrcStep, oSizeROI, pHist, nLevels, nLowerLevel, nUpperLevel,
                                           pDeviceBuffer, nppStreamCtx);
}

NppStatus nppiHistogramEven_8u_C1R(const Npp8u *pSrc, int nSrcStep, NppiSize oSizeROI, Npp32s *pHist, int nLevels,
                                   Npp32s nLowerLevel, Npp32s nUpperLevel, Npp8u *pDeviceBuffer) {
  NppStreamContext nppStreamCtx;
  nppGetStreamContext(&nppStreamCtx);
  return nppiHistogramEven_8u_C1R_Ctx(pSrc, nSrcStep, oSizeROI, pHist, nLevels, nLowerLevel, nUpperLevel, pDeviceBuffer,
                                      nppStreamCtx);
}

// Host function to generate evenly spaced histogram levels
NppStatus nppiEvenLevelsHost_32s(Npp32s *pLevels, int nLevels, Npp32s nLowerBound, Npp32s nUpperBound) {
  // Validate input parameters
  if (pLevels == nullptr) {
    return NPP_NULL_POINTER_ERROR;
  }

  if (nLevels < 2) {
    return NPP_HISTOGRAM_NUMBER_OF_LEVELS_ERROR;
  }

  if (nLowerBound >= nUpperBound) {
    return NPP_RANGE_ERROR;
  }

  // Generate evenly spaced levels
  for (int i = 0; i < nLevels; i++) {
    double ratio = static_cast<double>(i) / static_cast<double>(nLevels - 1);
    pLevels[i] = static_cast<Npp32s>(nLowerBound + ratio * (nUpperBound - nLowerBound));
  }

  return NPP_SUCCESS;
}