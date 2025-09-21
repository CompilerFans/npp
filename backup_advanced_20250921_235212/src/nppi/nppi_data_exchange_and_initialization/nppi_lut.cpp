#include "npp.h"
#include <cstdio>
#include <cstring>
#include <cuda_runtime.h>

// Implementation file

// Forward declarations for mpp host func implementations
extern "C" {
NppStatus nppiLUT_Linear_8u_C1R_Ctx_impl(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep, NppiSize oSizeROI,
                                         const Npp32s *pValues, const Npp32s *pLevels, int nLevels,
                                         NppStreamContext nppStreamCtx);
NppStatus nppiLUT_Linear_8u_C3R_Ctx_impl(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep, NppiSize oSizeROI,
                                         const Npp32s *pValues[3], const Npp32s *pLevels[3], int nLevels[3],
                                         NppStreamContext nppStreamCtx);
NppStatus nppiLUT_Linear_16u_C1R_Ctx_impl(const Npp16u *pSrc, int nSrcStep, Npp16u *pDst, int nDstStep,
                                          NppiSize oSizeROI, const Npp32s *pValues, const Npp32s *pLevels, int nLevels,
                                          NppStreamContext nppStreamCtx);
}

// Input validation helper
static inline NppStatus validateLUTLinearInputs(const void *pSrc, int nSrcStep, void *pDst, int nDstStep,
                                                NppiSize oSizeROI, const void *pValues, const void *pLevels,
                                                int nLevels) {
  if (!pSrc || !pDst) {
    return NPP_NULL_POINTER_ERROR;
  }

  if (!pValues || !pLevels) {
    return NPP_NULL_POINTER_ERROR;
  }

  if (oSizeROI.width < 0 || oSizeROI.height < 0) {
    return NPP_SIZE_ERROR;
  }

  if (nSrcStep <= 0 || nDstStep <= 0) {
    return NPP_STEP_ERROR;
  }

  if (nLevels < 2) { // 至少需要2个等级点进行线性插值
    return NPP_LUT_NUMBER_OF_LEVELS_ERROR;
  }

  return NPP_SUCCESS;
}

// 8-bit unsigned single channel linear LUT
NppStatus nppiLUT_Linear_8u_C1R_Ctx(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep, NppiSize oSizeROI,
                                    const Npp32s *pValues, const Npp32s *pLevels, int nLevels,
                                    NppStreamContext nppStreamCtx) {
  NppStatus status = validateLUTLinearInputs(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, pValues, pLevels, nLevels);
  if (status != NPP_SUCCESS) {
    return status;
  }

  return nppiLUT_Linear_8u_C1R_Ctx_impl(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, pValues, pLevels, nLevels,
                                        nppStreamCtx);
}

NppStatus nppiLUT_Linear_8u_C1R(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep, NppiSize oSizeROI,
                                const Npp32s *pValues, const Npp32s *pLevels, int nLevels) {
  NppStreamContext nppStreamCtx;
  nppStreamCtx.hStream = 0;
  return nppiLUT_Linear_8u_C1R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, pValues, pLevels, nLevels, nppStreamCtx);
}

// 8-bit unsigned three channel linear LUT
NppStatus nppiLUT_Linear_8u_C3R_Ctx(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep, NppiSize oSizeROI,
                                    const Npp32s *pValues[3], const Npp32s *pLevels[3], int nLevels[3],
                                    NppStreamContext nppStreamCtx) {
  // Validate基本参数
  if (!pSrc || !pDst) {
    return NPP_NULL_POINTER_ERROR;
  }

  if (!pValues || !pLevels || !nLevels) {
    return NPP_NULL_POINTER_ERROR;
  }

  if (oSizeROI.width < 0 || oSizeROI.height < 0) {
    return NPP_SIZE_ERROR;
  }

  if (nSrcStep <= 0 || nDstStep <= 0) {
    return NPP_STEP_ERROR;
  }

  // Validate每个通道的参数
  for (int c = 0; c < 3; c++) {
    if (!pValues[c] || !pLevels[c]) {
      return NPP_NULL_POINTER_ERROR;
    }
    if (nLevels[c] < 2) {
      return NPP_LUT_NUMBER_OF_LEVELS_ERROR;
    }
  }

  return nppiLUT_Linear_8u_C3R_Ctx_impl(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, pValues, pLevels, nLevels,
                                        nppStreamCtx);
}

NppStatus nppiLUT_Linear_8u_C3R(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep, NppiSize oSizeROI,
                                const Npp32s *pValues[3], const Npp32s *pLevels[3], int nLevels[3]) {
  NppStreamContext nppStreamCtx;
  nppStreamCtx.hStream = 0;
  return nppiLUT_Linear_8u_C3R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, pValues, pLevels, nLevels, nppStreamCtx);
}

// 16-bit unsigned single channel linear LUT
NppStatus nppiLUT_Linear_16u_C1R_Ctx(const Npp16u *pSrc, int nSrcStep, Npp16u *pDst, int nDstStep, NppiSize oSizeROI,
                                     const Npp32s *pValues, const Npp32s *pLevels, int nLevels,
                                     NppStreamContext nppStreamCtx) {
  NppStatus status = validateLUTLinearInputs(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, pValues, pLevels, nLevels);
  if (status != NPP_SUCCESS) {
    return status;
  }

  return nppiLUT_Linear_16u_C1R_Ctx_impl(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, pValues, pLevels, nLevels,
                                         nppStreamCtx);
}

NppStatus nppiLUT_Linear_16u_C1R(const Npp16u *pSrc, int nSrcStep, Npp16u *pDst, int nDstStep, NppiSize oSizeROI,
                                 const Npp32s *pValues, const Npp32s *pLevels, int nLevels) {
  NppStreamContext nppStreamCtx;
  nppStreamCtx.hStream = 0;
  return nppiLUT_Linear_16u_C1R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, pValues, pLevels, nLevels, nppStreamCtx);
}