#ifndef NPP_COLOR_CONVERSION_VALIDATION_HPP
#define NPP_COLOR_CONVERSION_VALIDATION_HPP

#include "npp.h"

static inline NppStatus validatePackedInput(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep,
                                            NppiSize oSizeROI, int channels) {
  if (!pSrc || !pDst) {
    return NPP_NULL_POINTER_ERROR;
  }
  if (oSizeROI.width <= 0 || oSizeROI.height <= 0) {
    return NPP_SIZE_ERROR;
  }
  if (nSrcStep < oSizeROI.width * channels || nDstStep < oSizeROI.width * channels) {
    return NPP_STEP_ERROR;
  }
  return NPP_SUCCESS;
}

static inline NppStatus validatePackedToPlanarInput(const Npp8u *pSrc, int nSrcStep, Npp8u *const pDst[],
                                                    int nDstStep, int planes, NppiSize oSizeROI, int srcChannels) {
  if (!pSrc || !pDst) {
    return NPP_NULL_POINTER_ERROR;
  }
  for (int i = 0; i < planes; ++i) {
    if (!pDst[i]) {
      return NPP_NULL_POINTER_ERROR;
    }
  }
  if (oSizeROI.width <= 0 || oSizeROI.height <= 0) {
    return NPP_SIZE_ERROR;
  }
  if (nSrcStep < oSizeROI.width * srcChannels || nDstStep < oSizeROI.width) {
    return NPP_STEP_ERROR;
  }
  return NPP_SUCCESS;
}

static inline NppStatus validatePlanarToPackedInput(const Npp8u *const pSrc[], int nSrcStep, Npp8u *pDst,
                                                    int nDstStep, NppiSize oSizeROI, int dstChannels, int planes) {
  if (!pSrc || !pDst) {
    return NPP_NULL_POINTER_ERROR;
  }
  for (int i = 0; i < planes; ++i) {
    if (!pSrc[i]) {
      return NPP_NULL_POINTER_ERROR;
    }
  }
  if (oSizeROI.width <= 0 || oSizeROI.height <= 0) {
    return NPP_SIZE_ERROR;
  }
  if (nSrcStep < oSizeROI.width || nDstStep < oSizeROI.width * dstChannels) {
    return NPP_STEP_ERROR;
  }
  return NPP_SUCCESS;
}

static inline NppStatus validatePlanarToPackedInput(const Npp8u *const pSrc[], int nSrcStep, Npp8u *pDst,
                                                    int nDstStep, NppiSize oSizeROI, int dstChannels) {
  return validatePlanarToPackedInput(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, dstChannels, 3);
}

static inline NppStatus validatePlanarInput(const Npp8u *const pSrc[], int nSrcStep, Npp8u *pDst[], int nDstStep,
                                            NppiSize oSizeROI, int planes) {
  if (!pSrc || !pDst) {
    return NPP_NULL_POINTER_ERROR;
  }
  for (int i = 0; i < planes; ++i) {
    if (!pSrc[i] || !pDst[i]) {
      return NPP_NULL_POINTER_ERROR;
    }
  }
  if (oSizeROI.width <= 0 || oSizeROI.height <= 0) {
    return NPP_SIZE_ERROR;
  }
  if (nSrcStep < oSizeROI.width || nDstStep < oSizeROI.width) {
    return NPP_STEP_ERROR;
  }
  return NPP_SUCCESS;
}

static inline NppStatus validatePlanarInput(const Npp8u *const pSrc[], int nSrcStep, Npp8u *pDst[], int nDstStep,
                                            NppiSize oSizeROI) {
  return validatePlanarInput(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, 3);
}

#endif
