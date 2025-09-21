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


NppStatus nppiEvenLevelsHost_32s(Npp32s *pLevels, int nLevels, Npp32s nLowerBound, Npp32s nUpperBound) {
  // Validate input parameters
  if (!pLevels) {
    return NPP_NULL_POINTER_ERROR;
  }

  if (nLevels < 2) {
    return NPP_HISTOGRAM_NUMBER_OF_LEVELS_ERROR;
  }

  if (nLowerBound >= nUpperBound) {
    return NPP_BAD_ARGUMENT_ERROR;
  }

  // Calculate the step size for even distribution
  double range = (double)(nUpperBound - nLowerBound);
  double step = range / (double)(nLevels - 1);

  // Generate even levels
  for (int i = 0; i < nLevels; i++) {
    pLevels[i] = nLowerBound + (Npp32s)(i * step);
  }

  // Ensure the last level is exactly the upper bound
  pLevels[nLevels - 1] = nUpperBound;

  return NPP_SUCCESS;
}


NppStatus nppiEvenLevelsHost_32f(Npp32f *pLevels, int nLevels, Npp32f nLowerBound, Npp32f nUpperBound) {
  // Validate input parameters
  if (!pLevels) {
    return NPP_NULL_POINTER_ERROR;
  }

  if (nLevels < 2) {
    return NPP_HISTOGRAM_NUMBER_OF_LEVELS_ERROR;
  }

  if (nLowerBound >= nUpperBound) {
    return NPP_BAD_ARGUMENT_ERROR;
  }

  // Calculate the step size for even distribution
  float range = nUpperBound - nLowerBound;
  float step = range / (float)(nLevels - 1);

  // Generate even levels
  for (int i = 0; i < nLevels; i++) {
    pLevels[i] = nLowerBound + i * step;
  }

  // Ensure the last level is exactly the upper bound
  pLevels[nLevels - 1] = nUpperBound;

  return NPP_SUCCESS;
}

NppStatus nppiHistogramEvenGetBufferSize_8u_C1R(NppiSize oSizeROI, int nLevels, size_t *hpBufferSize) {
  NppStreamContext nppStreamCtx;
  nppGetStreamContext(&nppStreamCtx);
  return nppiHistogramEvenGetBufferSize_8u_C1R_Ctx(oSizeROI, nLevels, hpBufferSize, nppStreamCtx);
}


NppStatus nppiHistogramEvenGetBufferSize_8u_C1R_Ctx(NppiSize oSizeROI, int nLevels, size_t *hpBufferSize,
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

  if (nSrcStep <= 0) {
    return NPP_STEP_ERROR;
  }

  if (nLevels < 2) {
    return NPP_HISTOGRAM_NUMBER_OF_LEVELS_ERROR;
  }

  if (nLowerLevel >= nUpperLevel) {
    return NPP_BAD_ARGUMENT_ERROR;
  }

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
