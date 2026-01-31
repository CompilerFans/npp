#include "npp.h"
#include <cuda_runtime.h>

// Forward declarations for CUDA implementations
extern "C" {
NppStatus nppiLUT_8u_C1R_Ctx_impl(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep,
                                   NppiSize oSizeROI, const Npp32s *pValues, const Npp32s *pLevels,
                                   int nLevels, NppStreamContext nppStreamCtx);

NppStatus nppiLUT_8u_C3R_Ctx_impl(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep,
                                   NppiSize oSizeROI, const Npp32s *pValues[3], const Npp32s *pLevels[3],
                                   int nLevels[3], NppStreamContext nppStreamCtx);

NppStatus nppiLUT_8u_C4R_Ctx_impl(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep,
                                   NppiSize oSizeROI, const Npp32s *pValues[4], const Npp32s *pLevels[4],
                                   int nLevels[4], NppStreamContext nppStreamCtx);

NppStatus nppiLUT_16u_C1R_Ctx_impl(const Npp16u *pSrc, int nSrcStep, Npp16u *pDst, int nDstStep,
                                    NppiSize oSizeROI, const Npp32s *pValues, const Npp32s *pLevels,
                                    int nLevels, NppStreamContext nppStreamCtx);
}

// Input validation helper
static inline NppStatus validateLUTInputs(const void *pSrc, int nSrcStep, void *pDst, int nDstStep,
                                          NppiSize oSizeROI, const void *pValues, const void *pLevels,
                                          int nLevels) {
  if (!pSrc || !pDst) {
    return NPP_NULL_POINTER_ERROR;
  }

  if (!pValues || !pLevels) {
    return NPP_NULL_POINTER_ERROR;
  }

  if (oSizeROI.width <= 0 || oSizeROI.height <= 0) {
    return NPP_SIZE_ERROR;
  }

  if (nSrcStep <= 0 || nDstStep <= 0) {
    return NPP_STEP_ERROR;
  }

  if (nLevels < 2) {
    return NPP_LUT_NUMBER_OF_LEVELS_ERROR;
  }

  return NPP_SUCCESS;
}

// 8u C1R
NppStatus nppiLUT_8u_C1R_Ctx(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep,
                              NppiSize oSizeROI, const Npp32s *pValues, const Npp32s *pLevels,
                              int nLevels, NppStreamContext nppStreamCtx) {
  NppStatus status = validateLUTInputs(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, pValues, pLevels, nLevels);
  if (status != NPP_SUCCESS) {
    return status;
  }

  return nppiLUT_8u_C1R_Ctx_impl(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, pValues, pLevels, nLevels, nppStreamCtx);
}

NppStatus nppiLUT_8u_C1R(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep,
                          NppiSize oSizeROI, const Npp32s *pValues, const Npp32s *pLevels, int nLevels) {
  NppStreamContext nppStreamCtx;
  nppStreamCtx.hStream = 0;
  return nppiLUT_8u_C1R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, pValues, pLevels, nLevels, nppStreamCtx);
}

// 8u C3R
NppStatus nppiLUT_8u_C3R_Ctx(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep,
                              NppiSize oSizeROI, const Npp32s *pValues[3], const Npp32s *pLevels[3],
                              int nLevels[3], NppStreamContext nppStreamCtx) {
  if (!pSrc || !pDst || !pValues || !pLevels) {
    return NPP_NULL_POINTER_ERROR;
  }

  if (oSizeROI.width <= 0 || oSizeROI.height <= 0) {
    return NPP_SIZE_ERROR;
  }

  if (nSrcStep <= 0 || nDstStep <= 0) {
    return NPP_STEP_ERROR;
  }

  for (int i = 0; i < 3; ++i) {
    if (!pValues[i] || !pLevels[i] || nLevels[i] < 2) {
      return NPP_LUT_NUMBER_OF_LEVELS_ERROR;
    }
  }

  return nppiLUT_8u_C3R_Ctx_impl(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, pValues, pLevels, nLevels, nppStreamCtx);
}

NppStatus nppiLUT_8u_C3R(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep,
                          NppiSize oSizeROI, const Npp32s *pValues[3], const Npp32s *pLevels[3], int nLevels[3]) {
  NppStreamContext nppStreamCtx;
  nppStreamCtx.hStream = 0;
  return nppiLUT_8u_C3R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, pValues, pLevels, nLevels, nppStreamCtx);
}

// 8u C4R
NppStatus nppiLUT_8u_C4R_Ctx(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep,
                              NppiSize oSizeROI, const Npp32s *pValues[4], const Npp32s *pLevels[4],
                              int nLevels[4], NppStreamContext nppStreamCtx) {
  if (!pSrc || !pDst || !pValues || !pLevels) {
    return NPP_NULL_POINTER_ERROR;
  }

  if (oSizeROI.width <= 0 || oSizeROI.height <= 0) {
    return NPP_SIZE_ERROR;
  }

  if (nSrcStep <= 0 || nDstStep <= 0) {
    return NPP_STEP_ERROR;
  }

  for (int i = 0; i < 4; ++i) {
    if (!pValues[i] || !pLevels[i] || nLevels[i] < 2) {
      return NPP_LUT_NUMBER_OF_LEVELS_ERROR;
    }
  }

  return nppiLUT_8u_C4R_Ctx_impl(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, pValues, pLevels, nLevels, nppStreamCtx);
}

NppStatus nppiLUT_8u_C4R(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep,
                          NppiSize oSizeROI, const Npp32s *pValues[4], const Npp32s *pLevels[4], int nLevels[4]) {
  NppStreamContext nppStreamCtx;
  nppStreamCtx.hStream = 0;
  return nppiLUT_8u_C4R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, pValues, pLevels, nLevels, nppStreamCtx);
}

// 16u C1R
NppStatus nppiLUT_16u_C1R_Ctx(const Npp16u *pSrc, int nSrcStep, Npp16u *pDst, int nDstStep,
                               NppiSize oSizeROI, const Npp32s *pValues, const Npp32s *pLevels,
                               int nLevels, NppStreamContext nppStreamCtx) {
  NppStatus status = validateLUTInputs(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, pValues, pLevels, nLevels);
  if (status != NPP_SUCCESS) {
    return status;
  }

  return nppiLUT_16u_C1R_Ctx_impl(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, pValues, pLevels, nLevels, nppStreamCtx);
}

NppStatus nppiLUT_16u_C1R(const Npp16u *pSrc, int nSrcStep, Npp16u *pDst, int nDstStep,
                           NppiSize oSizeROI, const Npp32s *pValues, const Npp32s *pLevels, int nLevels) {
  NppStreamContext nppStreamCtx;
  nppStreamCtx.hStream = 0;
  return nppiLUT_16u_C1R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, pValues, pLevels, nLevels, nppStreamCtx);
}
